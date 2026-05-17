///==========================================================================///
/// File: DbTransformsPass.cpp
///
/// Post-DB transforms pass. Hosts DB-level transformations after partitioning
/// and ownership decisions, and is also re-run after EDT cleanup so DB-root
/// lifetime cleanup stays in the DB layer.
///
/// Current transforms:
///   DT-1: Contract persistence -- persist refined AcquireContractSummary
///         back to IR via upsertLoweringContract().
///   DT-2: Stencil halo consolidation -- unify halo bounds from graph
///         analysis, raw IR attrs, and contract into unified min/max offsets.
///   DT-6: DB lifetime shortening -- remove cleanup-only acquire chains once
///         the reachable DB-use graph proves they feed no computation.
///   DT-7: Dead DB elimination -- remove root DBs whose reachable use graph is
///         entirely cleanup/forwarding plumbing.
///
///==========================================================================///

/// Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
/// Arts
#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbGraph.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/utils/ValueAnalysis.h"
#define GEN_PASS_DEF_DBTRANSFORMS
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/LoweringContractUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/RemovalUtils.h"
#include "carts/utils/StencilAttributes.h"
#include "carts/utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Statistic.h"
#include <memory>
/// Debug
#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_transforms);

static llvm::Statistic numContractsPersisted{
    "db_transforms", "NumContractsPersisted",
    "Number of refined lowering contracts persisted back to IR"};
static llvm::Statistic numStencilHalosConsolidated{
    "db_transforms", "NumStencilHalosConsolidated",
    "Number of stencil acquires whose halo bounds were consolidated"};
static llvm::Statistic numCleanupOnlyAcquireChainsRemoved{
    "db_transforms", "NumCleanupOnlyAcquireChainsRemoved",
    "Number of cleanup-only acquire chains removed from the DB graph"};
static llvm::Statistic numDeadDbRootsEliminated{
    "db_transforms", "NumDeadDbRootsEliminated",
    "Number of dead datablock root chains eliminated"};

using namespace mlir;
using namespace mlir::func;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {
struct DbTransformsPass : public impl::DbTransformsBase<DbTransformsPass> {
  DbTransformsPass(mlir::carts::arts::AnalysisManager *AM) : AM(AM) {
    assert(AM && "AnalysisManager must be provided externally");
  }

  void runOnOperation() override;

private:
  mlir::carts::arts::AnalysisManager *AM = nullptr;

  /// DT-1: Contract persistence
  unsigned persistContracts();

  /// DT-2: Stencil halo consolidation
  unsigned consolidateStencilHalos();

  /// DT-6: Cleanup-only acquire elimination
  unsigned shortenDbLifetimes();

  /// DT-7: Root DB cleanup once no semantic users remain
  unsigned eliminateDeadDbs();
};
} // namespace

void DbTransformsPass::runOnOperation() {
  ARTS_INFO_HEADER(DbTransformsPass);

  /// DbTransforms runs immediately after IR-mutating DB/EDT passes in the
  /// explicit post-partition refinement stages and is also re-run later after
  /// additional cleanup. Always refresh shared analyses up front so node
  /// lookups and access classification never observe stale graph state.
  AM->getEdtAnalysis().invalidate();
  AM->getDbAnalysis().invalidate();

  ///===------------------------------------------------------------------===///
  /// DT-1: Contract persistence
  ///
  /// Walk all DbAcquireOps, query DbAnalysis for the refined
  /// AcquireContractSummary, and when refinedByDbAnalysis is true persist
  /// the contract back to IR via upsertLoweringContract(). This ensures
  /// downstream lowering passes see the refined depPattern, ownerDims, etc.
  ///===------------------------------------------------------------------===///
  unsigned dt1Count = persistContracts();
  numContractsPersisted += dt1Count;
  if (dt1Count > 0)
    ARTS_INFO("DT-1: persisted " << dt1Count << " refined contracts");

  ///===------------------------------------------------------------------===///
  /// DT-2: Stencil halo consolidation
  ///
  /// Unify halo bounds from graph analysis, raw IR attrs, and the existing
  /// LoweringContractOp into the contract's minOffsets/maxOffsets. After
  /// consolidation, remove redundant raw stencil attrs from the acquire op.
  ///===------------------------------------------------------------------===///
  unsigned dt2Count = consolidateStencilHalos();
  numStencilHalosConsolidated += dt2Count;
  if (dt2Count > 0)
    ARTS_INFO("DT-2: consolidated stencil halos on " << dt2Count
                                                     << " acquires");

  ///===------------------------------------------------------------------===///
  /// DT-6: DB lifetime shortening
  ///
  /// Remove cleanup-only acquire subgraphs after proving through the reachable
  /// DB-use graph that they do not feed computation, task dependencies, or
  /// other semantic consumers.
  ///===------------------------------------------------------------------===///
  unsigned dt6Count = shortenDbLifetimes();
  numCleanupOnlyAcquireChainsRemoved += dt6Count;
  if (dt6Count > 0)
    ARTS_INFO("DT-6: shortened " << dt6Count
                                 << " cleanup-only acquire lifetimes");

  ///===------------------------------------------------------------------===///
  /// DT-7: Dead DB elimination
  ///
  /// Remove root DB chains when the full reachable DB-use graph is reduced to
  /// forwarding and cleanup plumbing only.
  ///===------------------------------------------------------------------===///
  unsigned dt7Count = eliminateDeadDbs();
  numDeadDbRootsEliminated += dt7Count;
  if (dt7Count > 0)
    ARTS_INFO("DT-7: eliminated " << dt7Count << " dead DB roots");

  ARTS_INFO_FOOTER(DbTransformsPass);
}

///===----------------------------------------------------------------------===///
/// DT-1: Contract persistence
///===----------------------------------------------------------------------===///
unsigned DbTransformsPass::persistContracts() {
  ModuleOp module = getOperation();
  unsigned count = 0;

  module.walk([&](func::FuncOp func) {
    DbAnalysis &dbAnalysis = AM->getDbAnalysis();
    SmallVector<DbAcquireOp, 16> acquires;
    func.walk([&](DbAcquireOp acquire) { acquires.push_back(acquire); });

    for (DbAcquireOp acquire : acquires) {
      auto contractSummary = dbAnalysis.getAcquireContractSummary(acquire);
      if (!contractSummary)
        continue;

      if (!contractSummary->derivedFactEvidence)
        continue;

      LoweringContractInfo persistedContract = contractSummary->contract;
      const bool needsRefinedMarker = !persistedContract.analysis.postDbRefined;
      if (needsRefinedMarker)
        persistedContract.analysis.postDbRefined = true;

      if (!contractSummary->refinedByDbAnalysis && !needsRefinedMarker)
        continue;

      /// Persist post-DB contract payload on the acquire pointer as the
      /// canonical source consumed by downstream passes.
      OpBuilder builder(acquire.getContext());
      builder.setInsertionPointAfter(acquire.getOperation());
      upsertLoweringContract(builder, acquire.getLoc(), acquire.getPtr(),
                             persistedContract);
      ++count;

      ARTS_DEBUG("DT-1: persisted contract for acquire " << acquire);
    }
  });

  return count;
}

///===----------------------------------------------------------------------===///
/// DT-2: Stencil halo consolidation
///===----------------------------------------------------------------------===///
unsigned DbTransformsPass::consolidateStencilHalos() {
  ModuleOp module = getOperation();
  unsigned count = 0;

  module.walk([&](func::FuncOp func) {
    DbAnalysis &dbAnalysis = AM->getDbAnalysis();
    SmallVector<DbAcquireOp, 16> acquires;
    func.walk([&](DbAcquireOp acquire) { acquires.push_back(acquire); });

    for (DbAcquireOp acquire : acquires) {
      auto contractSummary = dbAnalysis.getAcquireContractSummary(acquire);
      if (!contractSummary || !contractSummary->usesStencilSemantics())
        continue;

      ARTS_DEBUG("DT-2: processing stencil acquire " << acquire);

      unsigned rank = 0;

      auto rawMinOffsets = getStencilMinOffsets(acquire.getOperation());
      auto rawMaxOffsets = getStencilMaxOffsets(acquire.getOperation());
      if (rawMinOffsets)
        rank = std::max(rank, static_cast<unsigned>(rawMinOffsets->size()));
      if (rawMaxOffsets)
        rank = std::max(rank, static_cast<unsigned>(rawMaxOffsets->size()));

      auto contractHalo = projectHaloWindow(contractSummary->contract);
      if (contractHalo) {
        rank =
            std::max(rank, static_cast<unsigned>(contractHalo->first.size()));
        rank =
            std::max(rank, static_cast<unsigned>(contractHalo->second.size()));
      }

      if (rank == 0) {
        ARTS_DEBUG("DT-2: no halo bounds available, skipping");
        continue;
      }

      SmallVector<int64_t, 4> unifiedMin(rank, 0);
      SmallVector<int64_t, 4> unifiedMax(rank, 0);

      if (rawMinOffsets) {
        for (unsigned d = 0; d < rawMinOffsets->size() && d < rank; ++d)
          unifiedMin[d] = std::min(unifiedMin[d], (*rawMinOffsets)[d]);
        ARTS_DEBUG(
            "DT-2:   raw min offsets present, size=" << rawMinOffsets->size());
      }
      if (rawMaxOffsets) {
        for (unsigned d = 0; d < rawMaxOffsets->size() && d < rank; ++d)
          unifiedMax[d] = std::max(unifiedMax[d], (*rawMaxOffsets)[d]);
        ARTS_DEBUG(
            "DT-2:   raw max offsets present, size=" << rawMaxOffsets->size());
      }

      if (contractHalo) {
        auto &[contractMins, contractMaxs] = *contractHalo;
        for (unsigned d = 0; d < contractMins.size() && d < rank; ++d)
          unifiedMin[d] = std::min(unifiedMin[d], contractMins[d]);
        for (unsigned d = 0; d < contractMaxs.size() && d < rank; ++d)
          unifiedMax[d] = std::max(unifiedMax[d], contractMaxs[d]);
        ARTS_DEBUG("DT-2:   contract halo present, min_rank="
                   << contractMins.size()
                   << " max_rank=" << contractMaxs.size());
      }

      bool allZero = true;
      for (unsigned d = 0; d < rank; ++d) {
        if (unifiedMin[d] != 0 || unifiedMax[d] != 0) {
          allZero = false;
          break;
        }
      }
      if (allZero) {
        ARTS_DEBUG("DT-2: unified bounds are all zero, skipping");
        continue;
      }

      LoweringContractInfo newContract = contractSummary->contract;
      if (newContract.pattern.kind == ContractKind::Unknown &&
          contractSummary->usesStencilSemantics())
        newContract.pattern.kind = ContractKind::Stencil;
      if (!newContract.pattern.distributionPattern &&
          contractSummary->usesStencilSemantics())
        newContract.pattern.distributionPattern =
            EdtDistributionPattern::stencil;

      OpBuilder builder(acquire.getContext());
      builder.setInsertionPointAfter(acquire.getOperation());
      Location loc = acquire.getLoc();

      SmallVector<Value, 4> minOffsetValues;
      SmallVector<Value, 4> maxOffsetValues;
      minOffsetValues.reserve(rank);
      maxOffsetValues.reserve(rank);
      for (unsigned d = 0; d < rank; ++d) {
        minOffsetValues.push_back(
            createConstantIndex(builder, loc, unifiedMin[d]));
        maxOffsetValues.push_back(
            createConstantIndex(builder, loc, unifiedMax[d]));
      }

      newContract.spatial.minOffsets.assign(minOffsetValues.begin(),
                                            minOffsetValues.end());
      newContract.spatial.maxOffsets.assign(maxOffsetValues.begin(),
                                            maxOffsetValues.end());

      upsertLoweringContract(builder, loc, acquire.getPtr(), newContract);

      Operation *acquireOp = acquire.getOperation();
      acquireOp->removeAttr(acquire.getStencilMinOffsetsAttrName());
      acquireOp->removeAttr(acquire.getStencilMaxOffsetsAttrName());

      ++count;

      ARTS_DEBUG("DT-2: consolidated halo for acquire "
                 << acquire << " -> unified min/max with rank=" << rank);
    }
  });

  return count;
}

///===----------------------------------------------------------------------===///
/// DT-6: DB lifetime shortening
///===----------------------------------------------------------------------===///
unsigned DbTransformsPass::shortenDbLifetimes() {
  ModuleOp module = getOperation();
  unsigned removedAcquireChains = 0;

  module.walk([&](func::FuncOp func) {
    DbAnalysis &dbAnalysis = AM->getDbAnalysis();
    (void)dbAnalysis.getOrCreateGraph(func);

    SmallVector<DbAcquireOp, 16> acquires;
    func.walk([&](DbAcquireOp acquire) { acquires.push_back(acquire); });

    llvm::SetVector<Operation *> opsToRemove;
    DenseSet<Operation *> scheduledRoots;

    for (DbAcquireOp acquire : acquires) {
      if (!acquire || scheduledRoots.contains(acquire.getOperation()))
        continue;

      DbAcquireNode *acquireNode = dbAnalysis.getDbAcquireNode(acquire);
      if (!acquireNode)
        continue;

      llvm::SetVector<Operation *> cleanupChain;
      if (!DbUtils::collectCleanupOnlyUseChain(acquire.getGuid(), cleanupChain))
        continue;
      if (!DbUtils::collectCleanupOnlyUseChain(acquire.getPtr(), cleanupChain))
        continue;

      cleanupChain.insert(acquire.getOperation());
      for (Operation *op : cleanupChain) {
        opsToRemove.insert(op);
        if (isa<DbAcquireOp>(op))
          scheduledRoots.insert(op);
      }

      ++removedAcquireChains;
      ARTS_DEBUG("DT-6: removing cleanup-only acquire chain rooted at "
                 << acquireNode->getHierId());
    }

    if (opsToRemove.empty())
      return;

    RemovalUtils removalMgr;
    for (Operation *op : opsToRemove)
      removalMgr.markForRemoval(op);
    removalMgr.removeAllMarked(module, /*recursive=*/true);
    AM->invalidateFunction(func);
  });

  return removedAcquireChains;
}

///===----------------------------------------------------------------------===///
/// DT-7: Dead DB elimination
///===----------------------------------------------------------------------===///
unsigned DbTransformsPass::eliminateDeadDbs() {
  ModuleOp module = getOperation();
  unsigned removedRoots = 0;

  module.walk([&](func::FuncOp func) {
    DbAnalysis &dbAnalysis = AM->getDbAnalysis();
    (void)dbAnalysis.getOrCreateGraph(func);

    SmallVector<DbAllocOp, 16> allocs;
    func.walk([&](DbAllocOp alloc) { allocs.push_back(alloc); });

    llvm::SetVector<Operation *> opsToRemove;
    DenseSet<Operation *> scheduledRoots;

    for (DbAllocOp alloc : allocs) {
      if (!alloc || scheduledRoots.contains(alloc.getOperation()))
        continue;

      DbAllocNode *allocNode = dbAnalysis.getDbAllocNode(alloc);
      if (!allocNode)
        continue;

      llvm::SetVector<Operation *> cleanupChain;
      if (!DbUtils::collectCleanupOnlyUseChain(alloc.getGuid(), cleanupChain))
        continue;
      if (!DbUtils::collectCleanupOnlyUseChain(alloc.getPtr(), cleanupChain))
        continue;

      cleanupChain.insert(alloc.getOperation());
      for (Operation *op : cleanupChain) {
        opsToRemove.insert(op);
        if (isa<DbAllocOp>(op))
          scheduledRoots.insert(op);
      }

      ++removedRoots;
      ARTS_DEBUG("DT-7: removing dead DB root " << allocNode->getHierId());
    }

    if (opsToRemove.empty())
      return;

    RemovalUtils removalMgr;
    for (Operation *op : opsToRemove)
      removalMgr.markForRemoval(op);
    removalMgr.removeAllMarked(module, /*recursive=*/true);
    AM->invalidateFunction(func);
  });

  return removedRoots;
}

///===----------------------------------------------------------------------===///
/// DT-3: ESD annotation
///===----------------------------------------------------------------------===///
///===----------------------------------------------------------------------===///
/// Pass creation
///===----------------------------------------------------------------------===///
namespace mlir {
namespace carts::arts {
std::unique_ptr<Pass> createDbTransformsPass(mlir::carts::arts::AnalysisManager *AM) {
  return std::make_unique<DbTransformsPass>(AM);
}
} // namespace carts::arts
} // namespace mlir
