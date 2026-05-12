///==========================================================================///
/// File: DistributedHostLoopOutlining.cpp
///
/// Outlines eligible host-side loops into arts.edt<task> regions so they can
/// flow through the normal distributed arts.for lowering path. This is used
/// for multinode execution in general and is also relied on by
/// --distributed-db mode.
///==========================================================================///

#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/heuristics/DistributionHeuristics.h"
#include "arts/utils/ValueAnalysis.h"
#define GEN_PASS_DEF_DISTRIBUTEDHOSTLOOPOUTLINING
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/DbUtils.h"
#include "arts/utils/LoopUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/Utils.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include <cassert>

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(distributed_host_loop_outlining);

using namespace mlir;
using namespace mlir::arts;

namespace {

static bool hasNestedIterArgLoops(scf::ForOp loop) {
  bool hasIterArgs = false;
  loop.walk([&](scf::ForOp nested) {
    if (!nested.getInitArgs().empty()) {
      hasIterArgs = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasIterArgs;
}

static Operation *getSingleStoreRoot(scf::ForOp loop) {
  bool hasStores = false;
  bool hasUnsupportedStore = false;
  Operation *singleStoreRoot = nullptr;

  loop.walk([&](Operation *op) {
    auto access = DbUtils::getMemoryAccessInfo(op);
    if (!access || !access->isWrite())
      return WalkResult::advance();

    Operation *root = ValueAnalysis::getUnderlyingOperation(access->memref);
    if (!root || !isa<memref::AllocOp, memref::AllocaOp>(root)) {
      hasUnsupportedStore = true;
      return WalkResult::interrupt();
    }

    if (!singleStoreRoot)
      singleStoreRoot = root;
    else if (singleStoreRoot != root) {
      hasUnsupportedStore = true;
      return WalkResult::interrupt();
    }

    hasStores = true;
    return WalkResult::advance();
  });

  if (!hasStores || hasUnsupportedStore)
    return nullptr;
  return singleStoreRoot;
}

static bool hasOnlySupportedEffects(scf::ForOp loop, Operation *storeRoot) {
  if (!storeRoot)
    return false;

  bool hasStores = false;
  bool hasUnsupportedOp = false;

  loop.walk([&](Operation *op) {
    if (op == loop.getOperation() || isa<scf::YieldOp>(op) ||
        isa<scf::ForOp, scf::IfOp>(op))
      return WalkResult::advance();

    if (auto access = DbUtils::getMemoryAccessInfo(op)) {
      Operation *root = ValueAnalysis::getUnderlyingOperation(access->memref);
      if (!root) {
        hasUnsupportedOp = true;
        return WalkResult::interrupt();
      }

      if (access->isWrite()) {
        if (root != storeRoot) {
          hasUnsupportedOp = true;
          return WalkResult::interrupt();
        }
        hasStores = true;
        return WalkResult::advance();
      }

      /// Keep this path focused on producer loops. Reading the same root would
      /// introduce self-dependence that the original host loop did not model as
      /// a task dependency.
      if (root == storeRoot) {
        hasUnsupportedOp = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (isa<memref::DimOp, memref::DeallocOp>(op))
      return WalkResult::advance();

    if (isMemoryEffectFree(op))
      return WalkResult::advance();

    hasUnsupportedOp = true;
    return WalkResult::interrupt();
  });

  return hasStores && !hasUnsupportedOp;
}

static bool edtWillRunInternode(EdtOp edt, const RuntimeConfig *machine) {
  if (!edt)
    return false;
  if (edt.getConcurrency() == EdtConcurrency::internode)
    return true;
  if (!machine || !machine->isDistributed())
    return false;
  return edt.getTypeAttr().getValue() == EdtType::parallel;
}

static bool hasAlignedInternodeForUseAfter(scf::ForOp loop,
                                           Operation *storeRoot,
                                           const RuntimeConfig *machine) {
  if (!loop || !storeRoot)
    return false;

  func::FuncOp parentFunc = loop->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return false;

  bool foundAlignedUse = false;
  parentFunc.walk([&](Operation *op) {
    if (foundAlignedUse)
      return WalkResult::interrupt();

    auto access = DbUtils::getMemoryAccessInfo(op);
    if (!access || access->indices.empty())
      return WalkResult::advance();

    if (ValueAnalysis::getUnderlyingOperation(access->memref) != storeRoot)
      return WalkResult::advance();
    if (!isOrderedAfter(loop.getOperation(), op, parentFunc))
      return WalkResult::advance();

    auto edtOp = op->getParentOfType<EdtOp>();
    if (!edtOp || !edtWillRunInternode(edtOp, machine))
      return WalkResult::advance();

    Value loopIV;
    if (auto artsFor = op->getParentOfType<arts::ForOp>())
      loopIV = getLoopInductionVar(artsFor.getOperation());
    else if (auto scfFor = op->getParentOfType<scf::ForOp>())
      loopIV = getLoopInductionVar(scfFor.getOperation());
    if (!loopIV)
      return WalkResult::advance();

    Value leadingIdx =
        ValueAnalysis::stripNumericCasts(access->indices.front());
    if (ValueAnalysis::sameValue(leadingIdx, loopIV) ||
        ValueAnalysis::dependsOn(leadingIdx, loopIV) ||
        ValueAnalysis::dependsOn(loopIV, leadingIdx)) {
      ARTS_DEBUG("Aligned later multinode use found for store root "
                 << *storeRoot << " via " << *op);
      foundAlignedUse = true;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return foundAlignedUse;
}

static bool isEligibleDistributedHostLoop(scf::ForOp loop,
                                          const RuntimeConfig *machine) {
  if (!loop)
    return false;

  /// Only outline top-level host loops. Nested loops stay inside the cloned
  /// body of the selected outer loop.
  if (loop->getParentOfType<EdtOp>() || loop->getParentOfType<arts::ForOp>() ||
      loop->getParentOfType<scf::ForOp>()) {
    ARTS_DEBUG("Skip loop (not host top-level): " << loop);
    return false;
  }

  if (!loop.getInitArgs().empty()) {
    ARTS_DEBUG("Skip loop (iter_args unsupported): " << loop);
    return false;
  }

  if (hasNestedIterArgLoops(loop)) {
    ARTS_DEBUG("Skip loop (nested iter_args loop): " << loop);
    return false;
  }

  Operation *storeRoot = getSingleStoreRoot(loop);
  if (!storeRoot) {
    ARTS_DEBUG("Skip loop (store target unsupported): " << loop);
    return false;
  }

  /// Allow producer/transform loops between EDTs, not just constant init
  /// loops, but stay conservative on side effects and self-dependences.
  if (!hasOnlySupportedEffects(loop, storeRoot)) {
    ARTS_DEBUG("Skip loop (unsupported reads/writes/effects): " << loop);
    return false;
  }

  /// Only outline loops whose written allocation is later consumed by aligned
  /// internode work. This allows A/C-style GEMM loops and rejects B-style
  /// loops whose leading dimension is not distributed by the downstream task.
  if (!hasAlignedInternodeForUseAfter(loop, storeRoot, machine)) {
    ARTS_DEBUG(
        "Skip loop (no aligned later internode use for store root): " << loop);
    return false;
  }

  return true;
}

static void cloneScfLoopBodyIntoArtsFor(scf::ForOp sourceLoop,
                                        arts::ForOp targetLoop,
                                        OpBuilder &builder) {
  Region &targetRegion = targetLoop.getRegion();
  if (targetRegion.empty())
    targetRegion.push_back(new Block());

  Block &targetBlock = targetRegion.front();
  if (targetBlock.getNumArguments() == 0)
    targetBlock.addArgument(builder.getIndexType(), sourceLoop.getLoc());

  IRMapping mapper;
  mapper.map(sourceLoop.getInductionVar(), targetBlock.getArgument(0));

  Block &sourceBlock = sourceLoop.getRegion().front();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&targetBlock);
  for (Operation &op : sourceBlock.without_terminator())
    builder.clone(op, mapper);
  arts::YieldOp::create(builder, sourceLoop.getLoc());
}

static void applyMachineWorkerTopology(EdtOp outlinedEdt,
                                       const RuntimeConfig *machine,
                                       const EdtHeuristics *heuristics) {
  if (!outlinedEdt || outlinedEdt.getConcurrency() != EdtConcurrency::internode)
    return;

  auto workerConfig =
      heuristics
          ? heuristics->resolveWorkerConfig(outlinedEdt)
          : DistributionHeuristics::resolveWorkerConfig(outlinedEdt, machine);
  if (!workerConfig)
    return;

  if (workerConfig->totalWorkers > 0)
    setWorkers(outlinedEdt.getOperation(), workerConfig->totalWorkers);
  if (workerConfig->workersPerNode > 0)
    setWorkersPerNode(outlinedEdt.getOperation(), workerConfig->workersPerNode);
}

static void outlineLoop(scf::ForOp loop, const RuntimeConfig *machine,
                        const EdtHeuristics *heuristics) {
  Location loc = loop.getLoc();
  OpBuilder builder(loop);

  /// Preserve program order: the outlined EDT must complete before later host
  /// code observes loop results.
  auto epoch = EpochOp::create(builder, loc);
  Region &epochRegion = epoch.getRegion();
  if (epochRegion.empty())
    epochRegion.push_back(new Block());
  Block &epochBlock = epochRegion.front();
  OpBuilder epochBuilder = OpBuilder::atBlockBegin(&epochBlock);

  Value routeZero = arith::ConstantIntOp::create(epochBuilder, loc, 0, 32);
  auto outlinedEdt =
      EdtOp::create(epochBuilder, loc, EdtType::task, EdtConcurrency::internode,
                    routeZero, ValueRange{});
  outlinedEdt.setNoVerifyAttr(NoVerifyAttr::get(epochBuilder.getContext()));
  applyMachineWorkerTopology(outlinedEdt, machine, heuristics);

  Block &edtBody = outlinedEdt.getBody().front();
  OpBuilder::InsertionGuard edtGuard(epochBuilder);
  epochBuilder.setInsertionPointToStart(&edtBody);

  auto schedAttr = ForScheduleKindAttr::get(epochBuilder.getContext(),
                                            ForScheduleKind::Static);
  auto outlinedFor = arts::ForOp::create(
      epochBuilder, loc, ValueRange{loop.getLowerBound()},
      ValueRange{loop.getUpperBound()}, ValueRange{loop.getStep()}, schedAttr,
      Value(), ValueRange{});

  copyArtsMetadataAttrs(loop.getOperation(), outlinedFor.getOperation());
  cloneScfLoopBodyIntoArtsFor(loop, outlinedFor, epochBuilder);

  epochBuilder.setInsertionPointToEnd(&edtBody);
  arts::YieldOp::create(epochBuilder, loc);

  epochBuilder.setInsertionPointToEnd(&epochBlock);
  arts::YieldOp::create(epochBuilder, loc);

  ARTS_DEBUG("Outlined host scf.for into distributed arts.edt<task>: " << loop);
  loop.erase();
}

struct DistributedHostLoopOutliningPass
    : public impl::DistributedHostLoopOutliningBase<
          DistributedHostLoopOutliningPass> {
  explicit DistributedHostLoopOutliningPass(mlir::arts::AnalysisManager *AM)
      : AM(AM) {
    assert(AM && "AnalysisManager must be provided externally");
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto &edtHeuristics = AM->getEdtHeuristics();
    const auto &machine = AM->getRuntimeConfig();
    SmallVector<scf::ForOp> candidates;
    module.walk([&](func::FuncOp funcOp) {
      if (funcOp.isDeclaration())
        return;

      bool hasDbControlOps = false;
      funcOp.walk([&](DbControlOp) {
        hasDbControlOps = true;
        return WalkResult::interrupt();
      });

      funcOp.walk([&](scf::ForOp loop) {
        /// Task-dependency kernels already carry explicit DB control hints.
        /// Auto-outlining host loops in those functions would perturb existing
        /// dependency planning.
        if (hasDbControlOps) {
          ARTS_DEBUG("Skip loop (function has DbControlOps): " << loop);
          return;
        }
        if (isEligibleDistributedHostLoop(loop, &machine))
          candidates.push_back(loop);
      });
    });

    unsigned numConverted = 0;
    const unsigned numCandidates = static_cast<unsigned>(candidates.size());
    for (scf::ForOp loop : candidates) {
      if (!loop || !loop->getBlock())
        continue;
      outlineLoop(loop, &machine, &edtHeuristics);
      ++numConverted;
    }

    ARTS_INFO("DistributedHostLoopOutlining outlined "
              << numConverted << " / " << numCandidates
              << " eligible host loop(s)");
  }

private:
  mlir::arts::AnalysisManager *AM = nullptr;
};

} // namespace

namespace mlir {
namespace arts {
std::unique_ptr<Pass>
createDistributedHostLoopOutliningPass(mlir::arts::AnalysisManager *AM) {
  return std::make_unique<DistributedHostLoopOutliningPass>(AM);
}
} // namespace arts
} // namespace mlir
