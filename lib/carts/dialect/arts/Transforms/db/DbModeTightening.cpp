///==========================================================================///
/// File: DbModeTightening.cpp
/// Pass for DB mode tightening and storage-type inference.
/// Partitioning logic has been moved to DbPartitioning.cpp.
///
/// Example:
///   Before:
///     arts.db_acquire[<inout>] ...   // conservative mode
///
///   After:
///     arts.db_acquire[<in>] ...      // mode tightened when write not observed
///==========================================================================///

/// Dialects
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
/// Arts
#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/OwnershipProof.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbGraph.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "carts/passes/Passes.h"
#include "mlir/Pass/Pass.h"
/// Debug
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/dialect/arts/Utils/BlockedAccessUtils.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/utils/Debug.h"
#include "carts/utils/LoopUtils.h"
#include "carts/dialect/arts/Utils/LoweringContractUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arts;

#define GEN_PASS_DEF_DBMODETIGHTENING
#include "carts/passes/Passes.h.inc"

ARTS_DEBUG_SETUP(db_mode_tightening);

#include "llvm/ADT/Statistic.h"
static llvm::Statistic numAcquiresTightenedToRead{
    "db_mode_tightening", "NumAcquiresTightenedToRead",
    "Number of acquires tightened to read mode"};
static llvm::Statistic numAcquiresTightenedToOut{
    "db_mode_tightening", "NumAcquiresTightenedToOut",
    "Number of acquires tightened to out mode"};
static llvm::Statistic numAllocModesAdjusted{
    "db_mode_tightening", "NumAllocModesAdjusted",
    "Number of alloc modes adjusted to match acquire patterns"};
static llvm::Statistic numDbsMarkedLocalOnly{
    "db_mode_tightening", "NumDbsMarkedLocalOnly",
    "Number of DBs annotated as local-only"};
static llvm::Statistic numDbsMarkedReadOnlyAfterInit{
    "db_mode_tightening", "NumDbsMarkedReadOnlyAfterInit",
    "Number of DBs annotated as read-only-after-init"};

namespace {

static bool isIndexFullCoverage(Value idx, Value dimSize,
                                ArrayRef<LoopNode *> loops) {
  if (!idx || !dimSize)
    return false;

  auto dimConstOpt = ValueAnalysis::tryFoldConstantIndex(
      ValueAnalysis::stripNumericCasts(dimSize));
  if (dimConstOpt && *dimConstOpt == 1)
    return true;

  idx = ValueAnalysis::stripNumericCasts(idx);

  /// Constant index only covers full range if size == 1.
  auto idxConstOpt = ValueAnalysis::tryFoldConstantIndex(idx);
  if (idxConstOpt)
    /// A constant index alone cannot cover a non-unit dimension.
    return false;

  LoopNode *best = nullptr;
  LoopNode::IVExpr bestExpr;
  int bestDepth = -1;

  for (LoopNode *loop : loops) {
    if (!loop || !loop->dependsOnInductionVarNormalized(idx))
      continue;
    auto expr = loop->analyzeIndexExpr(idx);
    if (!expr.dependsOnIV)
      continue;
    int depth = loop->getNestingDepth();
    if (!best || depth > bestDepth) {
      best = loop;
      bestExpr = expr;
      bestDepth = depth;
    }
  }

  if (!best || !bestExpr.multiplier || !bestExpr.offset)
    return false;
  if (*bestExpr.multiplier != 1)
    return false;
  if (*bestExpr.offset != 0)
    return false;

  return isLoopFullRange(best, dimSize);
}

static SmallVector<Value> getEffectiveSliceDimSizes(DbAnalysis &dbAnalysis,
                                                    DbAcquireNode *acqNode,
                                                    DbAllocOp allocOp) {
  SmallVector<Value> dimSizes(allocOp.getElementSizes().begin(),
                              allocOp.getElementSizes().end());
  if (!acqNode || !allocOp || dimSizes.empty())
    return dimSizes;

  DbAcquireOp acquire = acqNode->getDbAcquireOp();
  if (!acquire)
    return dimSizes;

  OpBuilder builder(acquire);
  DbAnalysis::AcquirePartitionSummary partitionSummary =
      dbAnalysis.analyzeAcquirePartition(acquire, builder);

  for (auto [idx, size] : llvm::enumerate(partitionSummary.partitionSizes)) {
    if (!size || idx >= partitionSummary.partitionDims.size())
      continue;
    unsigned dim = partitionSummary.partitionDims[idx];
    if (dim < dimSizes.size())
      dimSizes[dim] = size;
  }

  return dimSizes;
}

static bool writesFullAllocation(DbAnalysis &dbAnalysis, DbAcquireNode *acqNode,
                                 DbAllocOp allocOp,
                                 LoopAnalysis &loopAnalysis) {
  if (!acqNode || !allocOp)
    return true;
  if (!acqNode->hasStores())
    return true;

  EdtOp edt = acqNode->getEdtUser();
  if (!edt)
    return true;

  SmallVector<LoopNode *> loops;
  loopAnalysis.collectLoopsInOperation(edt, loops);

  DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
  acqNode->collectAccessOperations(dbRefToMemOps);
  if (dbRefToMemOps.empty())
    return true;

  SmallVector<Value> dimSizes =
      getEffectiveSliceDimSizes(dbAnalysis, acqNode, allocOp);
  if (dimSizes.empty())
    return true;

  for (auto &entry : dbRefToMemOps) {
    DbRefOp dbRef = entry.first;
    for (Operation *memOp : entry.second) {
      if (!isa<memref::StoreOp>(memOp))
        continue;
      SmallVector<Value> indexChain =
          DbUtils::collectFullIndexChain(dbRef, memOp);
      if (indexChain.empty())
        return false;

      unsigned memrefStart = dbRef.getIndices().size();
      if (indexChain.size() <= memrefStart)
        return false;

      unsigned memrefRank = indexChain.size() - memrefStart;
      unsigned sizeRank = dimSizes.size();
      unsigned checkRank = std::min(memrefRank, sizeRank);

      for (unsigned d = 0; d < checkRank; ++d) {
        Value idx = indexChain[memrefStart + d];
        Value dimSize = dimSizes[d];
        if (!isIndexFullCoverage(idx, dimSize, loops))
          return false;
      }

      if (memrefRank != sizeRank)
        return false;
    }
  }

  return true;
}

static bool canPreserveProofTrustedPartitionedWrite(DbAcquireOp acquire) {
  if (!acquire)
    return false;

  auto partitionMode =
      acquire.getPartitionMode().value_or(PartitionMode::coarse);
  if (partitionMode == PartitionMode::coarse)
    return false;

  if (auto info = resolveAcquireContract(acquire))
    if (info->hasExplicitStencilContract() && info->supportsBlockHalo())
      return false;

  LoweringContractOp contract = getLoweringContractOp(acquire.getPtr());
  if (!contract)
    contract = getLoweringContractOp(acquire.getSourcePtr());
  if (!contract)
    return false;

  OwnershipProof proof = readOwnershipProof(contract.getOperation());
  if (!(proof.partitionAccessMapping && proof.depSliceSoundness))
    proof = computeOwnershipProof(contract);
  return proof.partitionAccessMapping && proof.depSliceSoundness;
}

struct MemAccessSite {
  Value source;
  SmallVector<Value, 4> fullIndexChain;
};

static std::optional<MemAccessSite> getMemAccessSite(DbRefOp dbRef,
                                                     Operation *op) {
  if (!dbRef || !op)
    return std::nullopt;
  SmallVector<Value> fullIndexChain = DbUtils::collectFullIndexChain(dbRef, op);
  if (fullIndexChain.empty())
    return std::nullopt;
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return MemAccessSite{
        dbRef.getSource(),
        SmallVector<Value, 4>(fullIndexChain.begin(), fullIndexChain.end())};
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return MemAccessSite{
        dbRef.getSource(),
        SmallVector<Value, 4>(fullIndexChain.begin(), fullIndexChain.end())};
  if (auto load = dyn_cast<affine::AffineLoadOp>(op))
    return MemAccessSite{
        dbRef.getSource(),
        SmallVector<Value, 4>(fullIndexChain.begin(), fullIndexChain.end())};
  if (auto store = dyn_cast<affine::AffineStoreOp>(op))
    return MemAccessSite{
        dbRef.getSource(),
        SmallVector<Value, 4>(fullIndexChain.begin(), fullIndexChain.end())};
  return std::nullopt;
}

static bool sameMemAccessSite(const MemAccessSite &lhs,
                              const MemAccessSite &rhs) {
  if (!ValueAnalysis::sameValue(lhs.source, rhs.source) ||
      lhs.fullIndexChain.size() != rhs.fullIndexChain.size())
    return false;
  for (auto [lhsIdx, rhsIdx] :
       llvm::zip(lhs.fullIndexChain, rhs.fullIndexChain)) {
    if (!ValueAnalysis::sameValue(lhsIdx, rhsIdx))
      return false;
  }
  return true;
}

static bool loadsAreSatisfiedByDominatingStores(DbAcquireNode *acqNode,
                                                func::FuncOp func) {
  if (!acqNode || !func)
    return false;

  DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
  acqNode->collectAccessOperations(dbRefToMemOps);
  if (dbRefToMemOps.empty())
    return false;

  SmallVector<std::pair<Operation *, MemAccessSite>, 8> stores;
  SmallVector<std::pair<Operation *, MemAccessSite>, 8> loads;
  for (auto &entry : dbRefToMemOps) {
    DbRefOp dbRef = entry.first;
    for (Operation *memOp : entry.second) {
      auto site = getMemAccessSite(dbRef, memOp);
      if (!site)
        continue;
      if (isa<memref::StoreOp, affine::AffineStoreOp>(memOp))
        stores.push_back({memOp, std::move(*site)});
      else if (isa<memref::LoadOp, affine::AffineLoadOp>(memOp))
        loads.push_back({memOp, std::move(*site)});
    }
  }

  if (stores.empty() || loads.empty())
    return false;

  DominanceInfo domInfo(func);
  for (auto &[loadOp, loadSite] : loads) {
    bool covered = false;
    for (auto &[storeOp, storeSite] : stores) {
      if (!domInfo.dominates(storeOp, loadOp))
        continue;
      if (!sameMemAccessSite(storeSite, loadSite))
        continue;
      covered = true;
      break;
    }
    if (!covered) {
      return false;
    }
  }

  return true;
}

/// Recursively collect combined access modes from all acquire levels.
static void collectModesRecursive(DbAcquireNode *acqNode, ArtsMode &combined) {
  ArtsMode mode = acqNode->getDbAcquireOp().getMode();
  combined = combineAccessModes(combined, mode);
  acqNode->forEachChildNode([&](NodeBase *child) {
    if (auto *nestedAcq = dyn_cast<DbAcquireNode>(child))
      collectModesRecursive(nestedAcq, combined);
  });
}

/// Recursively check whether any acquire has a distribution contract.
static void checkDistributedRecursive(DbAcquireNode *acqNode,
                                      DbAnalysis &dbAnalysis,
                                      bool &isLocalOnly) {
  if (!isLocalOnly)
    return;
  DbAcquireOp acquire = acqNode ? acqNode->getDbAcquireOp() : DbAcquireOp();
  if (auto summary = dbAnalysis.getAcquireContractSummary(acquire);
      summary && summary->hasDistributionContract()) {
    isLocalOnly = false;
    return;
  }
  acqNode->forEachChildNode([&](NodeBase *child) {
    if (auto *nestedAcq = dyn_cast<DbAcquireNode>(child))
      checkDistributedRecursive(nestedAcq, dbAnalysis, isLocalOnly);
  });
}

/// Entry for collectAcquiresRecursive: acquire info with program order.
struct AcquireOrderEntry {
  unsigned order;
  DbAcquireNode *node;
};

/// Recursively collect all acquire nodes with their program order.
static void collectAcquiresRecursive(DbAcquireNode *acqNode, DbGraph &graph,
                                     SmallVectorImpl<AcquireOrderEntry> &out) {
  DbAcquireOp acqOp = acqNode->getDbAcquireOp();
  unsigned order = graph.getOpOrder(acqOp.getOperation());
  out.push_back({order, acqNode});
  acqNode->forEachChildNode([&](NodeBase *child) {
    if (auto *nestedAcq = dyn_cast<DbAcquireNode>(child))
      collectAcquiresRecursive(nestedAcq, graph, out);
  });
}

static bool isInsideRepeatableControl(Operation *op) {
  if (!op)
    return false;

  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp, scf::ParallelOp, scf::ForallOp,
            affine::AffineForOp, affine::AffineParallelOp>(parent))
      return true;
  }

  return false;
}

} // namespace

///===----------------------------------------------------------------------===///
/// Pass Implementation
/// Tighten Datablock access modes and infer storage types
///===----------------------------------------------------------------------===///
namespace {
struct DbModeTighteningPass
    : public ::impl::DbModeTighteningBase<DbModeTighteningPass> {
  DbModeTighteningPass(mlir::arts::AnalysisManager *AM, bool forceInout)
      : AM(AM) {
    assert(AM && "AnalysisManager must be provided externally");
    this->forceInout = forceInout;
  }

  void runOnOperation() override;

  /// Mode adjustment
  bool adjustDbModes();

  /// DB storage-type inference annotations
  void inferDbStorageTypes();

  /// Graph rebuild
  void invalidateAndRebuildGraph();

private:
  ModuleOp module;
  mlir::arts::AnalysisManager *AM = nullptr;
};
} // namespace

void DbModeTighteningPass::runOnOperation() {
  module = getOperation();

  ARTS_INFO_HEADER(DbModeTighteningPass);
  ARTS_DEBUG_REGION(module.dump(););

  assert(AM && "AnalysisManager must be provided externally");

  /// TODO(PERF): adjustDbModes and inferDbStorageTypes both perform separate
  /// module.walk(FuncOp) + getOrCreateGraph. These could be fused into a
  /// single walk if analysis invalidation between phases is handled carefully.

  /// Graph construction and analysis
  invalidateAndRebuildGraph();

  /// Adjust DB modes based on access patterns
  bool changed = adjustDbModes();

  if (changed) {
    ARTS_INFO(" Module has changed, invalidating analyses");
    AM->invalidate();
  }

  /// Infer DB storage-type annotations (local_only, read_only_after_init)
  inferDbStorageTypes();

  ARTS_INFO_FOOTER(DbModeTighteningPass);
  ARTS_DEBUG_REGION(module.dump(););
}

///===----------------------------------------------------------------------===///
/// Adjust DB modes based on accesses.
/// Based on the collected loads and stores, adjust the DB mode to in, out, or
/// inout.
///===----------------------------------------------------------------------===///
bool DbModeTighteningPass::adjustDbModes() {
  ARTS_DEBUG_HEADER(AdjustDBModes);
  bool changed = false;

  module.walk([&](func::FuncOp func) {
    DbAnalysis &dbAnalysis = AM->getDbAnalysis();
    DbGraph &graph = dbAnalysis.getOrCreateGraph(func);

    /// First, adjust per-acquire modes
    graph.forEachAcquireNode([&](DbAcquireNode *acqNode) {
      bool hasLoads = acqNode->hasLoads();
      bool hasStores = acqNode->hasStores();

      DbAcquireOp acqOp = acqNode->getDbAcquireOp();
      /// Some acquires already carry an authoritative dependency mode
      /// (explicit control dependencies, worker-local partial reductions).
      /// Do not re-infer or optimize those modes from local memory accesses.
      if (acqOp.getPreserveAccessMode()) {
        ARTS_DEBUG("AcquireOp: " << acqOp
                                 << " preserving explicit dependency mode "
                                 << acqOp.getMode());
        return;
      }

      /// If the rewritten IR no longer exposes concrete memory accesses
      /// through this acquire, do not degrade the previously established
      /// access contract. Later lowering stages may still rely on that mode
      /// even when the access path has been rewritten through db_ref users or
      /// opaque pointer views that this analysis cannot classify precisely.
      if (!hasLoads && !hasStores) {
        ARTS_DEBUG("AcquireOp: " << acqOp
                                 << " has no visible memory accesses; "
                                    "preserving existing mode "
                                 << acqOp.getMode());
        return;
      }

      bool loadsCoveredByLocalStores =
          hasLoads && hasStores &&
          loadsAreSatisfiedByDominatingStores(acqNode, func);

      ArtsMode newMode = ArtsMode::in;
      if (hasStores && (!hasLoads || loadsCoveredByLocalStores))
        newMode = ArtsMode::out;
      else if (hasLoads && hasStores)
        newMode = ArtsMode::inout;
      else
        newMode = ArtsMode::in;

      if (loadsCoveredByLocalStores) {
        ARTS_DEBUG("AcquireOp: " << acqOp
                                 << " reloads only task-local stored values; "
                                    "treating access as out");
      }

      if (hasStores && forceInout) {
        ARTS_DEBUG("AcquireOp: " << acqOp
                                 << " forcing inout mode (CARTS_FORCE_INOUT)");
        newMode = ArtsMode::inout;
      }

      if (newMode == ArtsMode::out) {
        DbAllocNode *allocNode = acqNode->getRootAlloc();
        DbAllocOp allocOp = allocNode ? allocNode->getDbAllocOp() : DbAllocOp();
        LoopAnalysis &loopAnalysis = AM->getLoopAnalysis();
        bool fullWrite =
            !allocOp ||
            writesFullAllocation(dbAnalysis, acqNode, allocOp, loopAnalysis);
        bool proofTrustedPartitionedWrite =
            allocOp && !fullWrite &&
            canPreserveProofTrustedPartitionedWrite(acqOp);
        if (proofTrustedPartitionedWrite) {
          ARTS_DEBUG("AcquireOp: "
                     << acqOp
                     << " writes within a proof-trusted partitioned slice; "
                        "keeping out mode");
        } else if (allocOp && !fullWrite) {
          ARTS_DEBUG("AcquireOp: " << acqOp
                                   << " writes partial region; upgrading to "
                                      "inout to preserve untouched elements");
          newMode = ArtsMode::inout;
        }
      }

      /// Each DbAcquireNode's mode is derived from its own accesses only.
      /// Nested acquires will be processed separately by forEachAcquireNode.
      if (newMode == acqOp.getMode())
        return;

      ARTS_DEBUG("AcquireOp: " << acqOp << " from " << acqOp.getMode() << " to "
                               << newMode);
      acqOp.setModeAttr(ArtsModeAttr::get(acqOp.getContext(), newMode));
      if (newMode == ArtsMode::in)
        ++numAcquiresTightenedToRead;
      else if (newMode == ArtsMode::out)
        ++numAcquiresTightenedToOut;
      changed = true;
    });

    /// Then, adjust alloc dbMode - collect modes from all acquires in hierarchy
    graph.forEachAllocNode([&](DbAllocNode *allocNode) {
      ArtsMode maxMode = ArtsMode::in;
      bool sawAcquireMode = false;

      allocNode->forEachChildNode([&](NodeBase *child) {
        if (auto *acqNode = dyn_cast<DbAcquireNode>(child)) {
          sawAcquireMode = true;
          collectModesRecursive(acqNode, maxMode);
        }
      });

      if (!sawAcquireMode) {
        ARTS_DEBUG("AllocOp: " << allocNode->getDbAllocOp()
                               << " has no child acquires with visible modes; "
                                  "preserving existing alloc mode");
        return;
      }

      /// Update the alloc mode
      DbAllocOp allocOp = allocNode->getDbAllocOp();
      ArtsMode currentDbMode = allocOp.getMode();
      if (currentDbMode == maxMode)
        return;
      ARTS_DEBUG("AllocOp: " << allocOp << " from " << currentDbMode << " to "
                             << maxMode);
      allocOp.setModeAttr(ArtsModeAttr::get(allocOp.getContext(), maxMode));
      ++numAllocModesAdjusted;

      changed = true;
    });
  });

  ARTS_DEBUG_FOOTER(AdjustDBModes);
  return changed;
}

///===----------------------------------------------------------------------===///
/// Infer DB storage-type annotations.
/// Walks all DbAllocOps and sets UnitAttr annotations based on access patterns:
///   - arts.local_only: all acquires are intranode (no distributed contract)
///   - arts.read_only_after_init: after the first write, all subsequent
///   acquires
///     are read-only
///===----------------------------------------------------------------------===///
void DbModeTighteningPass::inferDbStorageTypes() {
  ARTS_DEBUG_HEADER(InferDbStorageTypes);

  module.walk([&](func::FuncOp func) {
    DbGraph &graph = AM->getDbAnalysis().getOrCreateGraph(func);
    DbAnalysis &dbAnalysis = AM->getDbAnalysis();

    graph.forEachAllocNode([&](DbAllocNode *allocNode) {
      DbAllocOp allocOp = allocNode->getDbAllocOp();
      if (!allocOp)
        return;

      /// ---------------------------------------------------------------
      /// 1. Local-only (PIN candidate):
      ///    A DB is local-only when the alloc itself is not distributed
      ///    AND none of its acquires have a distribution contract.
      /// ---------------------------------------------------------------
      bool isLocalOnly = !hasDistributedDbAllocation(allocOp.getOperation());

      if (isLocalOnly) {
        /// Check all acquire children (recursively) for distribution contracts
        allocNode->forEachChildNode([&](NodeBase *child) {
          if (auto *acqNode = dyn_cast<DbAcquireNode>(child))
            checkDistributedRecursive(acqNode, dbAnalysis, isLocalOnly);
        });
      }

      if (isLocalOnly) {
        allocOp->setAttr(AttrNames::Operation::LocalOnly,
                         UnitAttr::get(allocOp.getContext()));
        ++numDbsMarkedLocalOnly;
        ARTS_DEBUG("AllocOp: " << allocOp << " => local_only (PIN candidate)");
      }

      /// ---------------------------------------------------------------
      /// 2. Read-only-after-write:
      ///    Collect all acquires with their program order, then check if
      ///    after the first writer, every subsequent acquire is read-only.
      /// ---------------------------------------------------------------
      SmallVector<AcquireOrderEntry> orderedAcquires;

      allocNode->forEachChildNode([&](NodeBase *child) {
        if (auto *acqNode = dyn_cast<DbAcquireNode>(child))
          collectAcquiresRecursive(acqNode, graph, orderedAcquires);
      });

      /// Need at least one acquire to reason about
      if (orderedAcquires.empty())
        return;

      /// Sort by program order
      llvm::sort(orderedAcquires,
                 [](const AcquireOrderEntry &a, const AcquireOrderEntry &b) {
                   return a.order < b.order;
                 });

      /// Find the first writer and check all subsequent acquires. The
      /// read_only_after_init runtime hint is only valid when the initializer
      /// writer is a single dynamic write. A writer nested under repeatable
      /// control can execute once per loop iteration/epoch, even if static IR
      /// order shows all readers after it.
      bool foundWriter = false;
      bool allReadAfterWrite = true;
      unsigned writerCount = 0;
      bool writerInRepeatableControl = false;

      for (const auto &entry : orderedAcquires) {
        DbAcquireOp acquireOp = entry.node->getDbAcquireOp();
        ArtsMode mode = acquireOp.getMode();
        bool isWriter = DbUtils::isWriterMode(mode);

        if (!foundWriter) {
          if (isWriter) {
            foundWriter = true;
            writerInRepeatableControl =
                isInsideRepeatableControl(acquireOp.getOperation());
          }
          if (isWriter)
            ++writerCount;
          continue;
        }

        /// After the first writer, all must be read-only
        if (isWriter) {
          allReadAfterWrite = false;
          ++writerCount;
          continue;
        }
      }

      if (foundWriter && allReadAfterWrite && writerCount == 1 &&
          !writerInRepeatableControl) {
        allocOp->setAttr(AttrNames::Operation::ReadOnlyAfterInit,
                         UnitAttr::get(allocOp.getContext()));
        ++numDbsMarkedReadOnlyAfterInit;
        ARTS_DEBUG("AllocOp: " << allocOp
                               << " => read_only_after_init (ONCE candidate)");
      }
    });
  });

  ARTS_DEBUG_FOOTER(InferDbStorageTypes);
}

///===----------------------------------------------------------------------===///
/// Invalidate and rebuild the graph
///===----------------------------------------------------------------------===///
void DbModeTighteningPass::invalidateAndRebuildGraph() {
  AM->invalidateAndRebuildGraphs(module);
}

////===----------------------------------------------------------------------===////
/// Pass creation
////===----------------------------------------------------------------------===////
namespace mlir {
namespace arts {
std::unique_ptr<Pass>
createDbModeTighteningPass(mlir::arts::AnalysisManager *AM, bool forceInout) {
  return std::make_unique<DbModeTighteningPass>(AM, forceInout);
}
} // namespace arts
} // namespace mlir
