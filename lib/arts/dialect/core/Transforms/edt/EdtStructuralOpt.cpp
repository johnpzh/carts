///==========================================================================///
/// File: EdtStructuralOpt.cpp
///
/// EDT analysis/cleanup pass used in multiple pipeline stages.
///
/// Example:
///   Before:
///     arts.edt ... { dead ops / stale attributes / unnormalized deps }
///
///   After:
///     arts.edt ... { normalized dependency form + cleaned/annotated body }
///==========================================================================///

/// Dialects
#include "arts/utils/DbUtils.h"
#include "arts/utils/EdtUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/RemovalUtils.h"
#include "arts/utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
/// Arts
#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/db/DbAnalysis.h"
#include "arts/dialect/core/Analysis/graphs/edt/EdtGraph.h"
#include "arts/passes/Passes.h"
#include "mlir/Pass/Pass.h"
/// Other
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
/// Debug
#include "arts/utils/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arts;

#define GEN_PASS_DEF_EDTSTRUCTURALOPT
#include "arts/passes/Passes.h.inc"

ARTS_DEBUG_SETUP(edt_structural_opt);

static llvm::Statistic numExternalAllocasSunkStat{
    "edt_structural_opt", "NumExternalAllocasSunk",
    "Number of external allocas cloned into EDT-local storage"};
static llvm::Statistic numNoDepEdtsInlinedStat{
    "edt_structural_opt", "NumNoDepEdtsInlined",
    "Number of dependency-free EDTs inlined"};
static llvm::Statistic numSyncEdtsConvertedToEpochsStat{
    "edt_structural_opt", "NumSyncEdtsConvertedToEpochs",
    "Number of top-level sync EDTs converted into epochs"};
static llvm::Statistic numBarriersRemovedStat{
    "edt_structural_opt", "NumBarriersRemoved",
    "Number of redundant barriers removed"};

namespace {

unsigned sinkExternalAllocasInEdt(EdtOp edt) {
  Block &body = edt.getBody().front();
  DenseMap<Operation *, SmallVector<Operation *, 4>> usesByAlloca;
  DenseMap<Operation *, unsigned> operationOrder;

  body.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      auto allocaOp = operand.getDefiningOp<memref::AllocaOp>();
      if (!allocaOp)
        continue;
      if (edt.getBody().isAncestor(allocaOp->getParentRegion()))
        continue;
      usesByAlloca[allocaOp.getOperation()].push_back(op);
    }
  });

  if (usesByAlloca.empty())
    return 0;

  if (func::FuncOp parentFunc = edt->getParentOfType<func::FuncOp>()) {
    unsigned ordinal = 0;
    parentFunc.walk([&](Operation *op) { operationOrder[op] = ordinal++; });
  }

  OpBuilder builder(edt);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&body);

  unsigned sunkAllocas = 0;
  for (const auto &entry : usesByAlloca) {
    auto allocaOp = cast<memref::AllocaOp>(entry.first);
    // Honor the RaiseMemrefToTensor marker: allocas tagged as "arts.preserve"
    // are used to bridge raised tensor DBs back to the user-visible memref so
    // post-region readers observe the updated values. Sinking them into the
    // EDT would orphan those stores and the external alloca would still carry
    // its initial (zero) contents at the post-region readers. Skip the sink.
    if (allocaOp->hasAttr(AttrNames::Operation::Preserve))
      continue;
    bool hasStoreInEdt = false;
    for (Operation *user : entry.second) {
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemRef() == allocaOp.getResult()) {
          hasStoreInEdt = true;
          break;
        }
      }
    }

    bool hasUnsafeStore = false;
    SmallVector<memref::StoreOp, 4> initStores;
    for (Operation *user : allocaOp->getUsers()) {
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemRef() != allocaOp.getResult())
          continue;
        /// Stores inside the EDT body are private writes (e.g., OpenMP
        /// private(buffer) lowered by cgeist outside omp.parallel). These
        /// should not prevent sinking — they are the reason we want to sink.
        if (edt.getBody().isAncestor(store->getParentRegion()))
          continue;
        /// Stores in nested regions (e.g., scf.for body) reference values
        /// scoped to that region (loop induction variables, local
        /// computations). Cloning such stores into the EDT would produce
        /// references to values that do not dominate the EDT body. Treat
        /// these as unsafe to prevent sinking when the alloca is initialized
        /// by a loop — the EDT must share the original buffer.
        if (!store->getParentRegion()->isAncestor(edt->getParentRegion())) {
          hasUnsafeStore = true;
          break;
        }
        if (!EdtUtils::canCloneAllocaInitStore(store, allocaOp.getResult())) {
          hasUnsafeStore = true;
          break;
        }
        initStores.push_back(store);
        continue;
      }
    }
    std::stable_sort(initStores.begin(), initStores.end(),
                     [&](memref::StoreOp lhs, memref::StoreOp rhs) {
                       Operation *lhsOp = lhs.getOperation();
                       Operation *rhsOp = rhs.getOperation();
                       if (lhsOp == rhsOp)
                         return false;
                       if (lhsOp->getBlock() == rhsOp->getBlock())
                         return lhsOp->isBeforeInBlock(rhsOp);
                       auto lhsIt = operationOrder.find(lhsOp);
                       auto rhsIt = operationOrder.find(rhsOp);
                       if (lhsIt != operationOrder.end() &&
                           rhsIt != operationOrder.end())
                         return lhsIt->second < rhsIt->second;
                       return lhsOp < rhsOp;
                     });

    if (hasUnsafeStore)
      continue;

    /// If the alloca is only read inside the EDT, sink it only when we can
    /// safely clone its initialization stores. This preserves scalar values
    /// that would otherwise be uninitialized after outlining.
    if (!hasStoreInEdt && initStores.empty())
      continue;

    Operation *clonedOp = builder.clone(*allocaOp.getOperation());
    ++sunkAllocas;
    auto newAlloca = cast<memref::AllocaOp>(clonedOp);
    IRMapping mapping;
    mapping.map(allocaOp.getResult(), newAlloca.getResult());

    for (Operation *user : entry.second)
      user->replaceUsesOfWith(allocaOp.getResult(), newAlloca.getResult());

    if (!initStores.empty()) {
      builder.setInsertionPointAfter(newAlloca);
      for (memref::StoreOp store : initStores)
        builder.clone(*store.getOperation(), mapping);
      builder.setInsertionPointToStart(&body);
    }
  }
  return sunkAllocas;
}
} // namespace

///===----------------------------------------------------------------------===///
/// Pass Implementation
///===----------------------------------------------------------------------===///
namespace {
struct EdtStructuralOptPass
    : public ::impl::EdtStructuralOptBase<EdtStructuralOptPass> {
  EdtStructuralOptPass(mlir::arts::AnalysisManager *AM, bool runAnalysis)
      : AM(AM), numExternalAllocasSunk(
                    this, "num-external-allocas-sunk",
                    "Number of external allocas cloned into EDT-local storage"),
        numNoDepEdtsInlined(this, "num-no-dep-edts-inlined",
                            "Number of dependency-free EDTs inlined"),
        numSyncEdtsConvertedToEpochs(
            this, "num-sync-edts-converted-to-epochs",
            "Number of top-level sync EDTs converted into epochs"),
        numBarriersRemoved(this, "num-barriers-removed",
                           "Number of redundant barriers removed") {
    assert(AM && "AnalysisManager must be provided externally");
    this->runAnalysis = runAnalysis;
  }
  EdtStructuralOptPass(const EdtStructuralOptPass &other)
      : ::impl::EdtStructuralOptBase<EdtStructuralOptPass>(other), AM(other.AM),
        numExternalAllocasSunk(
            this, "num-external-allocas-sunk",
            "Number of external allocas cloned into EDT-local storage"),
        numNoDepEdtsInlined(this, "num-no-dep-edts-inlined",
                            "Number of dependency-free EDTs inlined"),
        numSyncEdtsConvertedToEpochs(
            this, "num-sync-edts-converted-to-epochs",
            "Number of top-level sync EDTs converted into epochs"),
        numBarriersRemoved(this, "num-barriers-removed",
                           "Number of redundant barriers removed") {}
  void runOnOperation() override;

  /// Inline EDTs with no dependencies by splicing their bodies into the parent
  /// block. This removes task creation overhead for independent work.
  bool inlineNoDepEdts();

  bool processSyncTaskEdts();
  bool removeBarriers();

  /// Graph-driven cleanups
  bool removeRedundantBarriersWithGraphs(func::FuncOp func,
                                         arts::EdtGraph &graph);

private:
  ModuleOp module;
  mlir::arts::AnalysisManager *AM = nullptr;
  Statistic numExternalAllocasSunk;
  Statistic numNoDepEdtsInlined;
  Statistic numSyncEdtsConvertedToEpochs;
  Statistic numBarriersRemoved;
  SetVector<Operation *> opsToRemove;
};

} // namespace

void EdtStructuralOptPass::runOnOperation() {
  module = getOperation();
  ARTS_INFO_HEADER(EdtStructuralOptPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// TODO(PERF): This pass walks all EdtOps multiple times for alloca sinking,
  /// no-dependency inlining, and sync/task cleanup. These could be combined
  /// into a single categorized walk.

  module.walk([&](EdtOp edt) {
    unsigned sunk = sinkExternalAllocasInEdt(edt);
    numExternalAllocasSunk += sunk;
    numExternalAllocasSunkStat += sunk;
  });

  if (runAnalysis) {
    ARTS_INFO("Running EDT pass with analysis");
    /// IMM-2: Re-enable graph-driven barrier removal (AM is guaranteed
    /// non-null).
    removeBarriers();
  } else {
    ARTS_INFO("Running EDT pass without analysis");
    inlineNoDepEdts();
    processSyncTaskEdts();
  }

  /// Re-run alloca sinking after EDT rewrites to keep task-local buffers inside
  /// their regions.
  module.walk([&](EdtOp edt) {
    unsigned sunk = sinkExternalAllocasInEdt(edt);
    numExternalAllocasSunk += sunk;
    numExternalAllocasSunkStat += sunk;
  });

  /// Remove ops marked for removal
  ARTS_DEBUG("Ops to remove: " << opsToRemove.size());
  RemovalUtils removalMgr;
  for (Operation *op : opsToRemove) {
    ARTS_DEBUG("  Marking for removal: " << op->getName().getStringRef());
    for (Value result : op->getResults()) {
      ARTS_DEBUG("    Result has "
                 << std::distance(result.use_begin(), result.use_end())
                 << " uses");
    }
    removalMgr.markForRemoval(op);
  }
  ARTS_DEBUG("Calling removeAllMarked...");
  removalMgr.removeAllMarked(module, /*recursive=*/true);
  ARTS_DEBUG("removeAllMarked completed");

  ARTS_INFO_FOOTER(EdtStructuralOptPass);
  ARTS_DEBUG_REGION(module.dump(););
}

bool EdtStructuralOptPass::inlineNoDepEdts() {
  SmallVector<EdtOp, 8> candidates;

  module.walk([&](EdtOp edt) {
    if (edt.getType() != arts::EdtType::task &&
        edt.getType() != arts::EdtType::sync)
      return;
    /// A dependency-free task inside an epoch is still a scheduled unit of
    /// work. Inlining it into the epoch body serializes the dispatch loop and
    /// erases the SDE-authored runtime work distribution.
    if (EdtUtils::isInsideEpoch(edt))
      return;
    if (!edt.getDependencies().empty())
      return;
    candidates.push_back(edt);
  });

  if (candidates.empty())
    return false;

  bool changed = false;
  for (EdtOp edt : candidates) {
    Block &edtBody = edt.getRegion().front();
    if (edtBody.getNumArguments() != 0) {
      ARTS_DEBUG("Skipping no-dep EDT with unexpected block args");
      continue;
    }

    Operation *insertBefore = edt.getOperation();
    SmallVector<Operation *, 8> opsToMove;
    for (Operation &childOp : edtBody.without_terminator())
      opsToMove.push_back(&childOp);

    for (Operation *childOp : opsToMove)
      childOp->moveBefore(insertBefore);

    edt.erase();
    ++numNoDepEdtsInlined;
    ++numNoDepEdtsInlinedStat;
    changed = true;
    ARTS_DEBUG("Inlined no-dep EDT");
  }
  return changed;
}

/// Remove unconditional barriers that provide no additional ordering beyond
/// computed EDT dependencies (graph-informed pruning).
bool EdtStructuralOptPass::removeBarriers() {
  bool changed = false;
  module.walk([&](func::FuncOp func) {
    auto &edtGraph = AM->getEdtAnalysis().getOrCreateEdtGraph(func);
    if (edtGraph.size() == 0)
      return;

    changed |= removeRedundantBarriersWithGraphs(func, edtGraph);
  });
  return changed;
}

bool EdtStructuralOptPass::processSyncTaskEdts() {
  /// If the given single EdtOp is not nested within another EdtOp (i.e., is
  /// top-level), and is marked as sync, embed its region's contents in an
  /// arts::EpochOp. This effectively assigns the work to the master thread,
  /// avoiding unnecessary signal/sync overhead. The EdtOp itself is erased
  /// after its body is moved.
  auto convertToEpoch = [&](EdtOp &op) -> bool {
    OpBuilder builder(op);
    /// If the op is not top-level, return false.
    if (op->getParentOfType<EdtOp>())
      return false;

    /// If the sync EDT has dependencies, don't convert - the EDT context is
    /// needed for proper acquire/release semantics.
    if (!op.getDependencies().empty())
      return false;

    /// Create an arts::EpochOp and its block
    auto loc = op.getLoc();
    auto epochOp = arts::EpochOp::create(builder, loc);
    auto &epochBlock = epochOp.getRegion().emplaceBlock();
    builder.setInsertionPointToEnd(&epochBlock);
    arts::YieldOp::create(builder, loc);

    /// Move all operations except the terminator from the EdtOp's region to the
    /// epoch block
    Block *edtBody = &op.getRegion().front();

    /// Replace block argument uses with actual dependency values before moving.
    /// This is necessary because the block arguments will be destroyed when
    /// the EDT is erased.
    ValueRange deps = op.getDependencies();
    for (auto [idx, blockArg] : llvm::enumerate(edtBody->getArguments())) {
      if (idx < deps.size()) {
        blockArg.replaceAllUsesWith(deps[idx]);
      }
    }

    /// Collect operations to move before moving them
    SmallVector<Operation *, 8> opsToMove;
    for (Operation &childOp : edtBody->without_terminator())
      opsToMove.push_back(&childOp);

    /// Move all operations to the epoch block
    for (Operation *childOp : opsToMove)
      childOp->moveBefore(epochBlock.getTerminator());

    /// Erase the now-empty EdtOp
    op.erase();
    ++numSyncEdtsConvertedToEpochs;
    ++numSyncEdtsConvertedToEpochsStat;

    builder.setInsertionPointAfter(epochOp);
    return true;
  };

  /// Collect all sync task EDTs
  SmallVector<EdtOp, 8> syncTaskOps;
  module.walk([&](EdtOp edt) {
    if (edt.getType() == arts::EdtType::sync)
      syncTaskOps.push_back(edt);
  });

  if (syncTaskOps.empty())
    return false;

  /// Try to convert each sync task-EDT to an EpochOp.
  bool changed = false;
  for (EdtOp op : syncTaskOps) {
    op.setType(arts::EdtType::task);
    changed |= convertToEpoch(op);
  }
  return changed;
}

bool EdtStructuralOptPass::removeRedundantBarriersWithGraphs(
    func::FuncOp func, arts::EdtGraph &graph) {
  bool changed = false;

  /// Collect barriers within this function and check redundancy
  SmallVector<arts::BarrierOp, 8> toErase;
  func.walk([&](arts::BarrierOp barrier) {
    Block *block = barrier->getBlock();
    /// Partition EDTs in the same block into before/after
    SmallVector<arts::EdtOp, 8> beforeTasks;
    SmallVector<arts::EdtOp, 8> afterTasks;
    bool pastBarrier = false;
    for (Operation &op : *block) {
      if (&op == barrier.getOperation()) {
        pastBarrier = true;
        continue;
      }
      if (auto edt = dyn_cast<arts::EdtOp>(&op)) {
        (pastBarrier ? afterTasks : beforeTasks).push_back(edt);
      }
    }

    if (beforeTasks.empty() || afterTasks.empty())
      return;

    bool redundant = true;
    for (arts::EdtOp a : beforeTasks) {
      for (arts::EdtOp b : afterTasks) {
        bool connected = graph.isEdtReachable(a, b);
        bool independent = graph.areEdtsIndependent(a, b);
        /// Barrier redundant if: connected (dependency already enforced) OR
        /// independent (no dependency needed)
        if (!connected && !independent) {
          redundant = false;
          break;
        }
      }
      if (!redundant)
        break;
    }

    if (redundant)
      toErase.push_back(barrier);
  });

  for (auto b : toErase) {
    ARTS_INFO("Removing redundant barrier");
    b.erase();
    ++numBarriersRemoved;
    ++numBarriersRemovedStat;
    changed = true;
  }
  return changed;
}

////===----------------------------------------------------------------------===////
/// Pass creation
////===----------------------------------------------------------------------===////
namespace mlir {
namespace arts {
std::unique_ptr<Pass>
createEdtStructuralOptPass(mlir::arts::AnalysisManager *AM, bool runAnalysis) {
  return std::make_unique<EdtStructuralOptPass>(AM, runAnalysis);
}
} // namespace arts
} // namespace mlir
