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
static llvm::Statistic numParallelEdtsFusedStat{
    "edt_structural_opt", "NumParallelEdtsFused",
    "Number of consecutive parallel EDT pairs fused"};
static llvm::Statistic numNoDepEdtsInlinedStat{
    "edt_structural_opt", "NumNoDepEdtsInlined",
    "Number of dependency-free EDTs inlined"};
static llvm::Statistic numParallelEdtsConvertedToSyncStat{
    "edt_structural_opt", "NumParallelEdtsConvertedToSync",
    "Number of parallel EDTs converted into sync EDTs"};
static llvm::Statistic numParallelEdtsConvertedWithAcquireRewiringStat{
    "edt_structural_opt", "NumParallelEdtsConvertedWithAcquireRewiring",
    "Number of parallel EDTs converted into sync EDTs by rewiring inner "
    "acquires"};
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
///===----------------------------------------------------------------------===///
/// Parallel EDT Fusion
///===----------------------------------------------------------------------===///
///
/// Fuses consecutive arts.edt<parallel> operations in the same block into a
/// single parallel EDT. This merges independent parallel regions that each
/// contain arts.for loops, reducing epoch overhead (each parallel EDT becomes
/// a separate epoch later in the pipeline).
///
/// BEFORE:
///   arts.edt <parallel> { arts.for(%0) to(%N) step(%1) { ... } }
///   arts.edt <parallel> { arts.for(%0) to(%N) step(%1) { ... } }
///
/// AFTER:
///   arts.edt <parallel> {
///     arts.for(%0) to(%N) step(%1) { ... }
///     arts.for(%0) to(%N) step(%1) { ... }
///   }
///
///===----------------------------------------------------------------------===///

/// Fuses consecutive fusable parallel EDTs in a block.
/// Returns true if any fusion was performed.
static unsigned fuseConsecutiveParallelEdts(Block &block,
                                            const EdtHeuristics &heuristics) {
  unsigned fusedPairs = 0;
  (void)EdtUtils::fuseConsecutivePairs<EdtOp>(
      block,
      [&](EdtOp a, EdtOp b) {
        ParallelEdtFusionDecision decision =
            heuristics.evaluateParallelEdtFusion(a, b);
        if (!decision.shouldFuse)
          ARTS_DEBUG("Skipping parallel EDT fusion: " << decision.rationale);
        return decision.shouldFuse;
      },
      [&](EdtOp a, EdtOp b) {
        Block &firstBody = a.getBody().front();
        Block &secondBody = b.getBody().front();
        EdtUtils::spliceBodyBeforeTerminator(secondBody, firstBody);
        ARTS_DEBUG("Fused consecutive parallel EDTs");
        ++fusedPairs;
        b.erase();
      });
  return fusedPairs;
}

/// Processes top-level blocks in a region for parallel EDT fusion.
/// Does NOT recurse into scf::ForOp bodies to avoid breaking patterns
/// that JacobiAlternatingBuffersPattern needs to match (exactly 2 parallel EDTs
/// inside a time-stepping loop).
static unsigned
processRegionForParallelEdtFusion(Region &region,
                                  const EdtHeuristics &heuristics) {
  unsigned fusedPairs = 0;
  for (Block &block : region) {
    fusedPairs += fuseConsecutiveParallelEdts(block, heuristics);
    for (Operation &op : block) {
      if (isa<scf::ForOp>(op))
        continue;
      for (Region &nested : op.getRegions())
        fusedPairs += processRegionForParallelEdtFusion(nested, heuristics);
    }
  }
  return fusedPairs;
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
        numParallelEdtsFused(this, "num-parallel-edts-fused",
                             "Number of consecutive parallel EDT pairs fused"),
        numNoDepEdtsInlined(this, "num-no-dep-edts-inlined",
                            "Number of dependency-free EDTs inlined"),
        numParallelEdtsConvertedToSync(
            this, "num-parallel-edts-converted-to-sync",
            "Number of parallel EDTs converted into sync EDTs"),
        numParallelEdtsConvertedWithAcquireRewiring(
            this, "num-parallel-edts-converted-with-acquire-rewiring",
            "Number of parallel EDTs converted into sync EDTs by rewiring "
            "inner acquires"),
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
        numParallelEdtsFused(this, "num-parallel-edts-fused",
                             "Number of consecutive parallel EDT pairs fused"),
        numNoDepEdtsInlined(this, "num-no-dep-edts-inlined",
                            "Number of dependency-free EDTs inlined"),
        numParallelEdtsConvertedToSync(
            this, "num-parallel-edts-converted-to-sync",
            "Number of parallel EDTs converted into sync EDTs"),
        numParallelEdtsConvertedWithAcquireRewiring(
            this, "num-parallel-edts-converted-with-acquire-rewiring",
            "Number of parallel EDTs converted into sync EDTs by rewiring "
            "inner acquires"),
        numSyncEdtsConvertedToEpochs(
            this, "num-sync-edts-converted-to-epochs",
            "Number of top-level sync EDTs converted into epochs"),
        numBarriersRemoved(this, "num-barriers-removed",
                           "Number of redundant barriers removed") {}
  void runOnOperation() override;
  bool convertParallelIntoSingle(EdtOp &op);

  /// Utility for creating EDT with merged dependencies and region transfer
  EdtOp createEdtWithMergedDepsAndRegion(OpBuilder &builder, Location loc,
                                         arts::EdtType newType, EdtOp sourceOp,
                                         ArrayRef<Value> additionalDeps = {});

  /// Remove barriers immediately before and after a single EDT
  unsigned removeBarriersAroundSingleEdt(EdtOp parallelOp, EdtOp singleOp);

  /// Convert parallel EDT with inner acquires to sync EDT by rewiring outer
  /// acquires
  bool convertParallelWithAcquiresToSync(EdtOp parallelOp, EdtOp singleOp,
                                         ArrayRef<DbAcquireOp> innerAcquires,
                                         ArrayRef<DbReleaseOp> innerReleases,
                                         bool needsBarrier);

  /// Inline EDTs with no dependencies by splicing their bodies into the parent
  /// block. This removes task creation overhead for independent work.
  bool inlineNoDepEdts();

  bool processSingleEdts();
  bool processParallelEdts();
  bool processSyncTaskEdts();
  bool removeBarriers();

  /// Graph-driven cleanups
  bool removeRedundantBarriersWithGraphs(func::FuncOp func,
                                         arts::EdtGraph &graph);

private:
  ModuleOp module;
  mlir::arts::AnalysisManager *AM = nullptr;
  Statistic numExternalAllocasSunk;
  Statistic numParallelEdtsFused;
  Statistic numNoDepEdtsInlined;
  Statistic numParallelEdtsConvertedToSync;
  Statistic numParallelEdtsConvertedWithAcquireRewiring;
  Statistic numSyncEdtsConvertedToEpochs;
  Statistic numBarriersRemoved;
  SetVector<Operation *> opsToRemove;
};

} // namespace

void EdtStructuralOptPass::runOnOperation() {
  module = getOperation();
  ARTS_INFO_HEADER(EdtStructuralOptPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// TODO(PERF): This pass walks all EdtOps 4 separate times
  /// (inlineNoDepEdts, processSingleEdts, processParallelEdts,
  /// processSyncTaskEdts). These could be combined into a single walk that
  /// categorizes EDTs by type into buckets, then processes each bucket.

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

    /// Fuse consecutive parallel EDTs before any other transformation.
    /// This merges independent parallel regions (e.g., 7 separate activation
    /// functions inlined into main) into one, reducing epoch overhead.
    unsigned fusedParallelEdts = processRegionForParallelEdtFusion(
        module.getRegion(), AM->getEdtHeuristics());
    numParallelEdtsFused += fusedParallelEdts;
    numParallelEdtsFusedStat += fusedParallelEdts;
    if (fusedParallelEdts != 0)
      ARTS_INFO("Parallel EDT fusion applied");

    processSingleEdts();
    processParallelEdts();
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
    /// parallel work. Inlining it into the epoch body serializes the dispatch
    /// loop and erases the runtime work distribution that ForLowering/EDT
    /// distribution just created.
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

bool EdtStructuralOptPass::processSingleEdts() {
  /// Find all single EDTs and remove barriers around them since single EDTs
  /// have implicit barrier semantics
  SmallVector<EdtOp, 8> singleOps;
  module.walk([&](EdtOp edt) {
    if (edt.getType() == arts::EdtType::single)
      singleOps.push_back(edt);
  });

  bool changed = false;
  for (EdtOp singleEdt : singleOps) {
    /// Find parent parallel EDT if it exists
    EdtOp parallelOp = singleEdt->getParentOfType<EdtOp>();
    if (parallelOp && parallelOp.getType() == arts::EdtType::parallel) {
      unsigned removed = removeBarriersAroundSingleEdt(parallelOp, singleEdt);
      numBarriersRemoved += removed;
      numBarriersRemovedStat += removed;
      changed |= removed != 0;
    }
  }

  return changed;
}

bool EdtStructuralOptPass::processParallelEdts() {
  /// Gather all parallel-EDT ops in the module.
  SmallVector<EdtOp, 8> parallelOps;
  module.walk([&](EdtOp edt) {
    if (edt.getType() == arts::EdtType::parallel)
      parallelOps.push_back(edt);
  });

  /// Try to convert parallel EDTs into single EDTs if they contain exactly one
  /// child.
  bool changed = false;
  for (EdtOp op : parallelOps)
    changed |= convertParallelIntoSingle(op);
  return changed;
}

/// Converts a parallel EDT region into a sync-task EDT when it contains a
/// single inner `arts.edt` and no other ops beyond terminators/barriers.
bool EdtStructuralOptPass::convertParallelIntoSingle(EdtOp &op) {
  uint32_t numOps = 0;
  EdtOp singleOp = nullptr;

  /// Track inner acquires and releases for rewiring
  SmallVector<DbAcquireOp, 4> innerAcquires;
  SmallVector<DbReleaseOp, 4> innerReleases;

  for (auto &block : op.getRegion()) {
    for (auto &inst : block) {
      ++numOps;
      if (auto edt = dyn_cast<arts::EdtOp>(&inst)) {
        if (edt.getType() == arts::EdtType::single) {
          if (singleOp)
            ARTS_UNREACHABLE(
                "Multiple single ops in parallel op not supported");
          singleOp = edt;
        } else
          return false;

      } else if (isa<arts::YieldOp>(&inst) || isa<arts::BarrierOp>(&inst)) {
        continue;
      } else if (auto acqOp = dyn_cast<DbAcquireOp>(&inst)) {
        innerAcquires.push_back(acqOp);
      } else if (auto relOp = dyn_cast<DbReleaseOp>(&inst)) {
        innerReleases.push_back(relOp);
      } else {
        /// Any other operation means we keep parallel structure
        return false;
      }
    }
  }

  /// Only convert to sync if it's JUST a single EDT with barriers/yield
  if (!singleOp || numOps < 3)
    return false;

  /// Preserve implicit single barrier semantics when nested EDTs are present.
  /// Only insert a barrier if the original parallel region had a trailing
  /// barrier after the single (i.e., not a nowait single).
  Operation *nextOp = singleOp->getNextNode();
  bool hadTrailingBarrier = nextOp && isa<arts::BarrierOp>(nextOp);
  auto *singleNode = AM->getEdtAnalysis().getEdtNode(singleOp);
  bool needsBarrier =
      singleNode && singleNode->hasNestedEdts() && hadTrailingBarrier;

  /// If we have inner acquires, use the enhanced conversion path
  if (!innerAcquires.empty()) {
    return convertParallelWithAcquiresToSync(op, singleOp, innerAcquires,
                                             innerReleases, needsBarrier);
  }

  /// Original path for simple case (no inner acquires)
  OpBuilder builder(op);
  SmallVector<Value> parallelDeps(op.getDependencies().begin(),
                                  op.getDependencies().end());
  auto newEdt = createEdtWithMergedDepsAndRegion(
      builder, op.getLoc(), arts::EdtType::sync, singleOp, parallelDeps);

  /// If nested EDTs were present and we had a trailing barrier, wrap top-level
  /// tasks in an epoch to preserve ordering without relying on barriers.
  if (needsBarrier)
    EdtUtils::wrapBodyInEpoch(newEdt.getRegion().front(), op.getLoc());

  /// Replace the parallel EDT with the new EDT and erase the single EDT
  op->replaceAllUsesWith(newEdt);
  singleOp->erase();

  /// Mark the parallel EDT for removal
  opsToRemove.insert(op);

  ++numParallelEdtsConvertedToSync;
  ++numParallelEdtsConvertedToSyncStat;
  ARTS_INFO("Converted parallel EDT into sync EDT");
  return true;
}

/// Convert parallel EDT with inner acquires to sync EDT by rewiring outer
/// acquires to use the correct mode and feeding them directly to the sync EDT.
/// TODO(REFACTOR): convertParallelWithAcquiresToSync (~165 lines) manually
/// builds a sync EDT with block arguments, body cloning, and yield insertion.
/// Consider reusing createEdtWithMergedDepsAndRegion with precomputed deps.
bool EdtStructuralOptPass::convertParallelWithAcquiresToSync(
    EdtOp parallelOp, EdtOp singleOp, ArrayRef<DbAcquireOp> innerAcquires,
    ArrayRef<DbReleaseOp> innerReleases, bool needsBarrier) {

  Location loc = parallelOp.getLoc();
  Block &parallelBlock = parallelOp.getRegion().front();
  ValueRange parallelDeps = parallelOp.getDependencies();

  ARTS_DEBUG("Converting parallel with " << innerAcquires.size()
                                         << " inner acquires to sync");

  /// Step 1: Build mapping from inner acquire source → outer acquire
  /// Inner acquire sources from block arg, which maps to parallel's dependency
  DenseMap<DbAcquireOp, DbAcquireOp> innerToOuterAcquire;
  DenseSet<DbAcquireOp> usedOuterAcquires;

  for (DbAcquireOp innerAcq : innerAcquires) {
    /// Inner acquire's source should be a block argument
    Value srcPtr = innerAcq.getSourcePtr();
    auto blockArg = dyn_cast<BlockArgument>(srcPtr);
    if (!blockArg || blockArg.getOwner() != &parallelBlock) {
      ARTS_DEBUG("Inner acquire source is not a parallel block arg, aborting");
      return false;
    }

    /// Map block arg index to parallel's dependency (outer acquire)
    unsigned argIdx = blockArg.getArgNumber();
    if (argIdx >= parallelDeps.size()) {
      ARTS_DEBUG("Block arg index out of range, aborting");
      return false;
    }

    Value outerDep = parallelDeps[argIdx];
    auto outerAcq = outerDep.getDefiningOp<DbAcquireOp>();
    if (!outerAcq) {
      ARTS_DEBUG("Parallel dependency is not from DbAcquireOp, aborting");
      return false;
    }

    innerToOuterAcquire[innerAcq] = outerAcq;
    usedOuterAcquires.insert(outerAcq);
    ARTS_DEBUG("  Mapped inner acquire to outer acquire");
  }

  /// Step 2: For each inner acquire used by single, rewire outer acquire
  DenseMap<Value, Value> oldPtrToNewPtr;
  SmallVector<Value> syncDeps;

  for (Value singleDep : singleOp.getDependencies()) {
    /// Find which inner acquire produced this dependency
    DbAcquireOp innerAcq = singleDep.getDefiningOp<DbAcquireOp>();
    if (!innerAcq) {
      ARTS_DEBUG("Single dependency not from acquire, aborting");
      return false;
    }

    auto it = innerToOuterAcquire.find(innerAcq);
    if (it == innerToOuterAcquire.end()) {
      ARTS_DEBUG("Inner acquire not in map, aborting");
      return false;
    }

    DbAcquireOp outerAcq = it->second;

    /// Create replacement outer acquire with inner's mode
    OpBuilder builder(outerAcq);
    auto newAcq = DbAcquireOp::create(
        builder, outerAcq.getLoc(), innerAcq.getMode(),
        outerAcq.getSourceGuid(), outerAcq.getSourcePtr(),
        outerAcq.getPtr().getType(), outerAcq.getPartitionMode(),
        SmallVector<Value>(outerAcq.getIndices()),
        SmallVector<Value>(outerAcq.getOffsets()),
        SmallVector<Value>(outerAcq.getSizes()),
        SmallVector<Value>(outerAcq.getPartitionIndices()),
        SmallVector<Value>(outerAcq.getPartitionOffsets()),
        SmallVector<Value>(outerAcq.getPartitionSizes()));
    newAcq.copyPartitionSegmentsFrom(outerAcq);
    if (outerAcq.getPreserveAccessMode())
      newAcq.setPreserveAccessMode();
    if (outerAcq.getPreserveDepEdge())
      newAcq.setPreserveDepEdge();

    /// Track the mapping and dependency
    oldPtrToNewPtr[outerAcq.getPtr()] = newAcq.getPtr();
    syncDeps.push_back(newAcq.getPtr());

    /// Replace uses of old outer acquire
    /// Note: We don't erase or mark the old outer acquire for removal.
    /// DCE will clean it up since all its uses have been replaced.
    outerAcq.getPtr().replaceAllUsesWith(newAcq.getPtr());
    outerAcq.getGuid().replaceAllUsesWith(newAcq.getGuid());

    ARTS_DEBUG("  Rewired outer acquire with mode " << (int)innerAcq.getMode());
  }

  /// Step 3: Create sync EDT with rewired dependencies
  OpBuilder builder(parallelOp);
  auto syncEdt =
      EdtOp::create(builder, loc, EdtType::sync, singleOp.getConcurrency());

  /// Setup region and block arguments
  Region &syncRegion = syncEdt.getRegion();
  if (syncRegion.empty())
    syncRegion.push_back(new Block());
  Block &syncBody = syncRegion.front();

  /// Build mapping from single's block args to sync's block args
  IRMapping argMapper;
  Block &singleBody = singleOp.getRegion().front();

  for (auto [idx, dep] : llvm::enumerate(syncDeps)) {
    BlockArgument newArg = syncBody.addArgument(dep.getType(), loc);
    if (idx < singleBody.getNumArguments()) {
      argMapper.map(singleBody.getArgument(idx), newArg);
    }
  }

  syncEdt.setDependencies(syncDeps);

  /// Step 4: Clone single EDT body into sync EDT
  builder.setInsertionPointToEnd(&syncBody);
  for (Operation &bodyOp : singleBody.without_terminator()) {
    builder.clone(bodyOp, argMapper);
  }

  /// If nested EDTs were present and we had a trailing barrier, wrap top-level
  /// tasks in an epoch to preserve ordering without relying on barriers.
  if (needsBarrier)
    EdtUtils::wrapBodyInEpoch(syncBody, loc);

  /// Add yield terminator if needed
  if (syncBody.empty() || !syncBody.back().hasTrait<OpTrait::IsTerminator>())
    YieldOp::create(builder, loc);

  /// Step 5: Clean up old operations
  /// First erase the single EDT (its operands reference inner acquire results)
  singleOp->erase();

  /// Erase inner releases (they use block args but produce no results)
  for (DbReleaseOp rel : innerReleases)
    rel->erase();

  /// Erase inner acquires (their results are now unused after single removal)
  for (DbAcquireOp acq : innerAcquires)
    acq->erase();

  /// Debug: Check what's left in the parallel and block arg uses
  ARTS_DEBUG("Remaining ops in parallel before erase:");
  for (Operation &op : parallelOp.getRegion().front()) {
    ARTS_DEBUG("  - " << op.getName().getStringRef());
  }
  ARTS_DEBUG("Block arg uses:");
  for (BlockArgument arg : parallelOp.getRegion().front().getArguments()) {
    ARTS_DEBUG("  - arg has " << std::distance(arg.use_begin(), arg.use_end())
                              << " uses");
  }

  /// Drop all references in the parallel region and erase it
  ARTS_DEBUG("About to drop references and erase parallel...");
  parallelOp.getRegion().dropAllReferences();
  ARTS_DEBUG("References dropped, now erasing...");
  parallelOp->erase();
  ARTS_DEBUG("Parallel erased successfully");

  ++numParallelEdtsConvertedToSync;
  ++numParallelEdtsConvertedToSyncStat;
  ++numParallelEdtsConvertedWithAcquireRewiring;
  ++numParallelEdtsConvertedWithAcquireRewiringStat;
  ARTS_INFO("Converted parallel EDT with inner acquires into sync EDT");
  return true;
}

/// Remove barriers immediately before and after a single EDT within a parallel
/// region. Since arts.edt<single> has implicit barrier semantics, explicit
/// barriers around it are redundant and cause issues with the CreateEpochs
/// pass.
unsigned EdtStructuralOptPass::removeBarriersAroundSingleEdt(EdtOp parallelOp,
                                                             EdtOp singleOp) {
  Block *parentBlock = singleOp->getBlock();
  if (!parentBlock)
    return 0;

  /// If the single region spawns nested EDTs (tasks/parallel/sync), we must
  /// keep explicit barriers to preserve ordering for CreateEpochs.
  auto *singleNode = AM->getEdtAnalysis().getEdtNode(singleOp);
  if (singleNode && singleNode->hasNestedEdts()) {
    ARTS_DEBUG("Keeping barriers around single EDT with nested EDTs");
    return 0;
  }

  unsigned removed = 0;
  /// Find and remove barrier immediately before the single EDT
  Operation *prevOp = singleOp->getPrevNode();
  if (prevOp && isa<arts::BarrierOp>(prevOp) && opsToRemove.insert(prevOp))
    ++removed;

  /// Find and remove barrier immediately after the single EDT
  Operation *nextOp = singleOp->getNextNode();
  if (nextOp && isa<arts::BarrierOp>(nextOp) && opsToRemove.insert(nextOp))
    ++removed;
  return removed;
}

/// Utility function to create a new EDT operation with merged dependencies
/// and transfer the region from the source operation.
EdtOp EdtStructuralOptPass::createEdtWithMergedDepsAndRegion(
    OpBuilder &builder, Location loc, arts::EdtType newType, EdtOp sourceOp,
    ArrayRef<Value> additionalDeps) {

  /// Collect all dependencies from source operation
  SetVector<Value> allDeps;
  for (Value dep : sourceOp.getDependencies())
    allDeps.insert(dep);

  /// Add any additional dependencies
  for (Value dep : additionalDeps)
    allDeps.insert(dep);

  /// Create new EDT operation with merged dependencies and intranode
  /// concurrency. Create with empty deps first, then set them after adding
  /// block args.
  auto newEdt =
      arts::EdtOp::create(builder, loc, newType, EdtConcurrency::intranode);

  /// Setup block and block arguments
  Region &newRegion = newEdt.getRegion();
  if (newRegion.empty())
    newRegion.push_back(new Block());
  Block &newBody = newRegion.front();

  /// Add block arguments for each dependency and build mapping from
  /// source block args to new block args
  IRMapping argMapper;
  Block &sourceBody = sourceOp.getRegion().front();
  SmallVector<Value> depVec;

  for (auto [idx, dep] : llvm::enumerate(allDeps.getArrayRef())) {
    BlockArgument newArg = newBody.addArgument(dep.getType(), loc);
    depVec.push_back(dep);

    /// Map source's block arg to new block arg if this dep came from source
    if (idx < sourceOp.getDependencies().size() &&
        idx < sourceBody.getNumArguments()) {
      argMapper.map(sourceBody.getArgument(idx), newArg);
    }
  }

  /// Set dependencies on the new EDT
  newEdt.setDependencies(depVec);

  /// Clone operations from source region to new region with argument remapping
  builder.setInsertionPointToEnd(&newBody);
  for (Operation &op : sourceBody.without_terminator()) {
    builder.clone(op, argMapper);
  }

  /// Add yield terminator if needed
  if (newBody.empty() || !newBody.back().hasTrait<OpTrait::IsTerminator>())
    YieldOp::create(builder, loc);

  return newEdt;
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
