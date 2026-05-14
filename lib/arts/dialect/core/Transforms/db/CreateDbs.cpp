///==========================================================================///
/// File: CreateDbs.cpp
///
/// This pass creates ARTS Dbs (DB) operations to handle memory
/// dependencies for EDTs (Event-Driven Tasks). Each EDT may execute in a
/// separate memory environment, so external memory references must be properly
/// managed.
///
/// Pass Overview:
/// 1. Identify external memory allocations used by EDTs
/// 2. Create arts.db_alloc operations for each allocation
/// 3. Analyze memory access patterns within each EDT
/// 4. Insert arts.db_acquire operations before EDTs with computed dependencies
/// 5. Update load/store operations to use acquired memory views
/// 6. Insert arts.db_release operations before EDT terminators
/// 7. Insert arts.db_free operations for db_alloc operations
///
/// Example:
///   Before:
///     memref.load/store inside arts.edt body on captured buffers
///
///   After:
///     arts.db_alloc + arts.db_acquire/arts.db_release around EDT body access
///     with loads/stores redirected through acquired DB views
///
/// Allocation Strategy:
/// This pass analyzes memory access patterns from task dependencies
/// (DbControlOps) to determine the optimal datablock allocation granularity:
///
/// 1. Fine-grained allocation (preferred when possible):
///    - Detects consistent single-dimension indexing patterns (e.g., A[i])
///    - Verifies ALL db_acquires have indices for fine-grained access
///    - Creates multiple datablocks: sizes=[N], elementSizes=[remaining dims]
///
/// 2. Coarse-grained allocation (fallback):
///    - Used when no task dependencies are provided (no DbControlOps)
///    - Used when access patterns are inconsistent across EDTs
///    - Used when some db_acquires lack indices (full array access needed)
///
/// Access Mode Strategy:
/// Access modes are extracted from DbControlOps (from OpenMP depend clauses).
/// If no control DB is found, conservatively defaults to inout mode.
///==========================================================================///

#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/db/DbAnalysis.h"
#include "arts/dialect/core/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "arts/dialect/core/Transforms/db/DbLayoutPlanUtils.h"
#include "arts/dialect/core/Transforms/db/DbRewriter.h"
#include "arts/dialect/core/Transforms/db/block/DbBlockIndexer.h"
#include "arts/dialect/core/Transforms/db/elementwise/DbElementWiseIndexer.h"
#include "arts/utils/ValueAnalysis.h"
#define GEN_PASS_DEF_CREATEDBS
#include "arts/dialect/core/Transforms/db/DbTransforms.h"
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/DbUtils.h"
#include "arts/utils/LoweringContractUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/RemovalUtils.h"
#include "arts/utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include <cstdint>
#include <memory>

/// Debug
#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(create_dbs);

using namespace mlir::arts;

static llvm::Statistic numDbAllocsCreated{
    "create_dbs", "NumDbAllocsCreated",
    "Number of arts.db_alloc ops created for shared allocations"};
static llvm::Statistic numPrivateEdtAllocsSkipped{
    "create_dbs", "NumPrivateEdtAllocsSkipped",
    "Number of EDT-local allocations skipped because they never escape"};
static llvm::Statistic numEdtAccessModesInferred{
    "create_dbs", "NumEdtAccessModesInferred",
    "Number of EDT/memref pairs that required inferred access modes"};
static llvm::Statistic numMemrefsDefaultedToInOut{
    "create_dbs", "NumMemrefsDefaultedToInOut",
    "Number of memrefs defaulted to inout because no access evidence existed"};
static llvm::Statistic numDbAcquireGroupsCreated{
    "create_dbs", "NumDbAcquireGroupsCreated",
    "Number of DbAcquireOp groups created for EDT dependencies"};
static llvm::Statistic numGlobalDbInitializations{
    "create_dbs", "NumGlobalDbInitializations",
    "Number of global datablocks initialized from memref.global contents"};
static llvm::Statistic numOpsRewrittenToDbViews{
    "create_dbs", "NumOpsRewrittenToDbViews",
    "Number of operations rewritten to access datablock-backed views"};

using namespace mlir;

///===----------------------------------------------------------------------===///
/// Helper Functions
///===----------------------------------------------------------------------===///
namespace {

static bool isForwardingMemrefAliasOp(Operation *op, Value source) {
  if (auto viewLike = dyn_cast<ViewLikeOpInterface>(op))
    return viewLike.getViewSource() == source && op->getNumResults() == 1;

  if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(op)) {
    return unrealized.getInputs().size() == 1 &&
           unrealized.getInputs().front() == source &&
           unrealized.getOutputs().size() == 1;
  }

  if (auto subindex = dyn_cast<polygeist::SubIndexOp>(op))
    return subindex.getSource() == source && op->getNumResults() == 1;

  return false;
}

///===----------------------------------------------------------------------===///
/// Pass Implementation
///===----------------------------------------------------------------------===///

struct CreateDbsPass : public impl::CreateDbsBase<CreateDbsPass> {
  CreateDbsPass(mlir::arts::AnalysisManager *AM) : AM(AM) {
    assert(AM && "AnalysisManager must be provided externally");
  }

  void runOnOperation() override;

private:
  ModuleOp module;
  mlir::arts::AnalysisManager *AM = nullptr;
  ArtsCodegen *AC = nullptr;
  DenseMap<Operation *, Operation *> dbPtrToOriginalAlloc;
  SetVector<Operation *> opsToRemove;

  /// Structure to track all usage information for a memref
  struct MemrefInfo {
    Operation *alloc = nullptr;
    EdtOp parentEdt = nullptr;
    /// True if this allocation is used by an EDT other than its defining EDT.
    bool usedByOtherEdts = false;
    ArtsMode accessMode = ArtsMode::uninitialized;
    bool usedFineGrained = false;
    bool usedBlock = false;
    SmallVector<Value> blockSizes;
    DbAllocOp dbAllocOp = nullptr;

    /// The computed allocation strategy plan
    std::optional<DbRewritePlan> rewritePlan;

    /// Structure to track a specific DB access (mode + indices + chunks)
    struct DbDep {
      ArtsMode mode;
      SmallVector<Value, 4> indices, offsets, sizes;
      SmallVector<Operation *, 4> operations;
      bool preserveDepEdge = true;
    };

    /// Map from EDT operation to all DB accesses for that EDT
    DenseMap<EdtOp, SmallVector<DbDep>> edtToDeps;

    /// Add an access for this memref and update combined mode
    void addAccess(EdtOp edtOp, ArtsMode mode, SmallVector<Value, 4> indices,
                   SmallVector<Value, 4> offsets = {},
                   SmallVector<Value, 4> sizes = {},
                   SmallVector<Operation *, 4> operations = {},
                   bool preserveDepEdge = true) {
      accessMode = combineAccessModes(accessMode, mode);
      edtToDeps[edtOp].push_back(
          {mode, indices, offsets, sizes, operations, preserveDepEdge});
    }

    /// Compute index pattern information from stored accesses
    struct AccessPatternInfo {
      /// Whether all accesses have the same number of indices
      bool isConsistent = true;
      /// Whether all accesses have indices
      bool allAccessesHaveIndices = true;
      /// Minimum number of pinned dimensions across all accesses
      unsigned pinnedDimCount = 0;

      /// Any deps have offsets/sizes?
      bool hasChunkDeps = false;
      /// Number of chunk dimensions
      unsigned chunkDimCount = 0;
      /// SSA values for block sizes
      SmallVector<Value> blockSizes;
      /// All deps use same block size count?
      bool blockSizesAreConsistent = true;
    };

    AccessPatternInfo getAccessPatternInfo() const {
      AccessPatternInfo info;
      std::optional<unsigned> expectedDimCount;
      bool hasPinnedAccess = false;

      for (const auto &[edtOp, accesses] : edtToDeps) {
        for (const auto &access : accesses) {
          /// If no operations matched this dependency (SSA value mismatch),
          /// fallback to coarse-grained allocation
          if (access.operations.empty()) {
            info.isConsistent = false;
            info.allAccessesHaveIndices = false;
            continue;
          }

          if (access.indices.empty()) {
            info.allAccessesHaveIndices = false;
          } else {
            unsigned dimCount = access.indices.size();

            /// Track minimum pinned dimensions
            if (!hasPinnedAccess) {
              info.pinnedDimCount = dimCount;
              hasPinnedAccess = true;
            } else {
              info.pinnedDimCount = std::min(info.pinnedDimCount, dimCount);
            }

            /// Check consistency
            if (!expectedDimCount) {
              expectedDimCount = dimCount;
              info.pinnedDimCount = dimCount;
            } else if (*expectedDimCount != dimCount) {
              info.isConsistent = false;
            }
          }

          /// Track chunk dependencies from DbControlOp.offsets/sizes
          /// Chunk deps have: indices=[], offsets=[offset], sizes=[blockSize]
          if (!access.offsets.empty() && !access.sizes.empty()) {
            info.hasChunkDeps = true;

            if (info.blockSizes.empty()) {
              /// First chunk dep - record the sizes
              info.blockSizes.assign(access.sizes.begin(), access.sizes.end());
              info.chunkDimCount = access.sizes.size();
            } else {
              /// Subsequent chunk deps - check consistency (dimension count)
              /// We can't compare SSA values statically, so we just check count
              if (access.sizes.size() != info.chunkDimCount) {
                info.blockSizesAreConsistent = false;
              }
            }
          }
        }
      }

      return info;
    }
  };

  DenseMap<Operation *, MemrefInfo> memrefInfo;
  DenseMap<EdtOp, SetVector<Value>> edtExternalValues;

  void collectMemrefs();
  void collectControlDbOps();
  void reconcileExternalDepAccessModes();
  void ensureInitializedAccessModes();
  void createDbAllocOps();
  void lowerEdtExternalDependencies();
  void cleanupAndFinalize();
  void projectSemanticContractToDbValue(Operation *sourceOp,
                                        Operation *targetOp,
                                        Value contractTarget);
  void createDbAcquireOps(EdtOp edt, SetVector<Value> &externalDeps);
  Value findLocalAcquireView(EdtOp edt, Operation *dbAllocOp,
                             bool requiresIndexedAccess = false);
  Value findParentAcquireSource(EdtOp edt, Operation *dbAllocOp,
                                ArtsMode requestedMode,
                                bool requiresIndexedAccess = false);
  void initializeGlobalDbIfNeeded(Operation *alloc, DbAllocOp dbAllocOp,
                                  const DbRewritePlan &plan,
                                  DbAllocType allocType);

  DbAllocType inferAllocType(Operation *alloc);
  void insertDbFreeForDbAlloc(DbAllocOp dbAlloc, Operation *alloc);
  void rewriteOpsToUseDbAcquire(EdtOp edt, SmallVector<Operation *> &operations,
                                Operation *dbOp,
                                SmallVector<Value> &acquireIndices,
                                SmallVector<Value> &acquireOffsets,
                                Value localAcquireView,
                                const DbRewritePlan &plan);
  void rewriteUsesInParentEdt(MemrefInfo &memrefInfo);
  Operation *findPhysicalLayoutPlanSource(Operation *alloc);
  void projectStructuredPlanToDbValue(Operation *sourceOp, DbAllocOp dbAlloc);
  void rewriteUsesEverywhereWithPlan(Operation *alloc, DbAllocOp dbAlloc,
                                     const DbRewritePlan &plan);
};
} // namespace

///===----------------------------------------------------------------------===///
/// Pass Entry Point
///===----------------------------------------------------------------------===///
void CreateDbsPass::runOnOperation() {
  module = getOperation();
  opsToRemove.clear();
  memrefInfo.clear();

  /// Initialize ArtsCodegen
  auto ownedAC = std::make_unique<ArtsCodegen>(module, false);
  AC = ownedAC.get();

  ARTS_INFO_HEADER(CreateDbsPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// Phase 1: Collect memrefs used in EDTs
  ARTS_INFO("Phase 1: Collecting memrefs used in EDTs");
  collectMemrefs();
  ARTS_DEBUG(" - Found " << memrefInfo.size() << " memrefs used in EDTs");

  /// Phase 2: collect and clean up existing Control DB operations
  ARTS_INFO("Phase 2: collect and clean up existing Control DB operations");
  collectControlDbOps();

  /// Phase 3.5: Reconcile missing DbControl hints and fallback access modes.
  reconcileExternalDepAccessModes();
  ensureInitializedAccessModes();

  /// Phase 3: Create DbAlloc operations
  ARTS_INFO("Phase 3: Creating DbAlloc operations for " << memrefInfo.size()
                                                        << " memrefs");
  createDbAllocOps();

  /// Phase 4: Process EDTs for dependencies.
  lowerEdtExternalDependencies();

  /// Phase 5-7: Cleanup and validation post-checks.
  cleanupAndFinalize();

  AC = nullptr;
  ARTS_INFO_FOOTER(CreateDbsPass);
  ARTS_DEBUG_REGION(module.dump(););
}

void CreateDbsPass::reconcileExternalDepAccessModes() {
  /// For each EDT with external dependencies, ensure memrefs missing explicit
  /// DbControl entries still get a conservative access mode inferred from
  /// concrete memory uses.
  for (auto &edtEntry : edtExternalValues) {
    EdtOp edt = edtEntry.first;
    SetVector<Value> &externalDeps = edtEntry.second;

    for (Value externalDep : externalDeps) {
      Operation *underlyingOp =
          ValueAnalysis::getUnderlyingOperation(externalDep);
      if (!underlyingOp)
        continue;
      if (isa<DbAllocOp>(underlyingOp))
        continue; // already a DB

      MemrefInfo &info = memrefInfo[underlyingOp];
      bool hasControlDb = !info.edtToDeps[edt].empty();
      if (hasControlDb)
        continue;

      ArtsMode inferredMode =
          AM->getDbAnalysis().inferEdtAccessMode(underlyingOp, edt);
      if (inferredMode == ArtsMode::uninitialized)
        inferredMode = ArtsMode::inout;

      ARTS_DEBUG(" - Memref "
                 << *underlyingOp
                 << " used in EDT without DbControlOps, inferred mode="
                 << inferredMode);
      info.accessMode = combineAccessModes(info.accessMode, inferredMode);
      ++numEdtAccessModesInferred;
    }
  }
}

void CreateDbsPass::ensureInitializedAccessModes() {
  /// Any memref that still has no mode evidence falls back to inout to preserve
  /// dependency correctness.
  for (auto &entry : memrefInfo) {
    MemrefInfo &info = entry.second;
    if (info.accessMode != ArtsMode::uninitialized)
      continue;

    ARTS_DEBUG(" - Memref "
               << *entry.first
               << " has no DbControlOps at all, defaulting to inout mode");
    info.accessMode = ArtsMode::inout;
    ++numMemrefsDefaultedToInOut;
  }
}

void CreateDbsPass::lowerEdtExternalDependencies() {
  /// Use pre-order walk to visit parent EDTs before nested EDTs so nested
  /// acquire seeding can reuse in-scope handles when legal.
  ARTS_INFO("Phase 4: Creating DbAcquire operations for "
            << edtExternalValues.size() << " EDTs");
  module.walk<WalkOrder::PreOrder>([&](EdtOp edt) {
    auto it = edtExternalValues.find(edt);
    if (it != edtExternalValues.end())
      createDbAcquireOps(edt, it->second);
  });
}

void CreateDbsPass::cleanupAndFinalize() {
  ARTS_INFO("Phase 5: Cleaning up replaced allocations");
  RemovalUtils removalMgr;
  for (Operation *op : opsToRemove)
    removalMgr.markForRemoval(op);
  removalMgr.removeAllMarked(module, /*recursive=*/true);

  /// Re-enable verifier on all EDT ops after structural rewrites complete.
  module.walk([](EdtOp edt) { edt.removeNoVerifyAttr(); });

  DominanceInfo domInfo(module);
  arts::simplifyIR(module, domInfo);
}

void CreateDbsPass::projectSemanticContractToDbValue(Operation *sourceOp,
                                                     Operation *targetOp,
                                                     Value contractTarget) {
  if (!sourceOp || !targetOp || !contractTarget)
    return;

  transferOperationContract(sourceOp, targetOp);

  OpBuilder::InsertionGuard guard(AC->getBuilder());
  AC->setInsertionPointAfter(targetOp);
  transferLoweringContract(sourceOp, contractTarget, AC->getBuilder(),
                           targetOp->getLoc());
}

Operation *CreateDbsPass::findPhysicalLayoutPlanSource(Operation *alloc) {
  if (!alloc || alloc->getNumResults() == 0)
    return nullptr;

  auto compatiblePlanTarget = [&](Operation *candidate) {
    auto memRefType = dyn_cast<MemRefType>(alloc->getResult(0).getType());
    if (!memRefType || memRefType.getRank() == 0)
      return false;

    std::optional<SmallVector<int64_t, 4>> ownerDims =
        readI64ArrayAttr(getPlanOwnerDimsAttr(candidate));
    std::optional<SmallVector<int64_t, 4>> blockShape =
        readI64ArrayAttr(getPlanPhysicalBlockShapeAttr(candidate));
    if (!ownerDims || ownerDims->empty() || !blockShape ||
        blockShape->empty())
      return false;

    const unsigned rank = memRefType.getRank();
    if (blockShape->size() != rank && blockShape->size() != ownerDims->size())
      return false;

    for (auto [ownerSlot, rawDim] : llvm::enumerate(*ownerDims)) {
      if (rawDim < 0 || static_cast<uint64_t>(rawDim) >= rank)
        return false;
      unsigned physicalDim = static_cast<unsigned>(rawDim);
      size_t blockShapeIdx =
          blockShape->size() == rank ? physicalDim : ownerSlot;
      int64_t blockSize = (*blockShape)[blockShapeIdx];
      if (blockSize <= 0)
        return false;
      int64_t staticExtent = memRefType.getDimSize(physicalDim);
      if (staticExtent != ShapedType::kDynamic && blockSize > staticExtent)
        return false;
    }
    return true;
  };

  auto equivalentPlan = [](Operation *lhs, Operation *rhs) {
    return getPlanOwnerDimsAttr(lhs) == getPlanOwnerDimsAttr(rhs) &&
           getPlanPhysicalBlockShapeAttr(lhs) ==
               getPlanPhysicalBlockShapeAttr(rhs) &&
           getPlanLogicalWorkerSliceAttr(lhs) ==
               getPlanLogicalWorkerSliceAttr(rhs) &&
           getPlanHaloShapeAttr(lhs) == getPlanHaloShapeAttr(rhs) &&
           getPlanIterationTopologyAttr(lhs) ==
               getPlanIterationTopologyAttr(rhs);
  };

  auto writesAlloc = [&](Operation *root) {
    bool found = false;
    root->walk([&](Operation *nested) {
      if (found)
        return WalkResult::interrupt();
      auto access = DbUtils::getMemoryAccessInfo(nested);
      if (!access)
        return WalkResult::advance();
      if (!access->isWrite())
        return WalkResult::advance();
      Operation *underlying =
          ValueAnalysis::getUnderlyingOperation(access->memref);
      if (underlying == alloc) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found;
  };

  SmallVector<Operation *, 4> candidates;
  module.walk([&](EdtOp edt) {
    if (!hasPhysicalDbLayoutPlan(edt.getOperation()))
      return WalkResult::advance();
    if (!compatiblePlanTarget(edt.getOperation()))
      return WalkResult::advance();
    if (!writesAlloc(edt.getOperation()))
      return WalkResult::advance();
    candidates.push_back(edt.getOperation());
    return WalkResult::advance();
  });

  if (candidates.empty())
    return nullptr;

  Operation *selected = candidates.front();
  for (Operation *candidate : ArrayRef<Operation *>(candidates).drop_front()) {
    if (equivalentPlan(selected, candidate))
      continue;
    InFlightDiagnostic diag =
        alloc->emitError("conflicting SDE-authored physical DB layout plans for "
                         "one allocation");
    diag.attachNote(selected->getLoc()) << "first layout plan source";
    diag.attachNote(candidate->getLoc()) << "conflicting layout plan source";
    signalPassFailure();
    return nullptr;
  }
  return selected;
}

void CreateDbsPass::projectStructuredPlanToDbValue(Operation *sourceOp,
                                                   DbAllocOp dbAlloc) {
  if (!sourceOp || !dbAlloc)
    return;
  copyPlanAttrs(sourceOp, dbAlloc.getOperation());
  transferOperationContract(sourceOp, dbAlloc.getOperation());

  OpBuilder::InsertionGuard guard(AC->getBuilder());
  AC->setInsertionPointAfter(dbAlloc);
  transferLoweringContract(sourceOp, dbAlloc.getPtr(), AC->getBuilder(),
                           dbAlloc.getLoc());
}

///===----------------------------------------------------------------------===///
/// Allocation Type Inference
///===----------------------------------------------------------------------===///
DbAllocType CreateDbsPass::inferAllocType(Operation *alloc) {
  assert(alloc && "Allocation operation not found");

  if (isa<memref::AllocaOp>(alloc))
    return DbAllocType::stack;
  if (isa<memref::AllocOp>(alloc))
    return DbAllocType::heap;
  if (isa<memref::GetGlobalOp>(alloc))
    return DbAllocType::global;

  /// Default to heap for unknown allocation types
  assert(false && "Unknown allocation type");
  return DbAllocType::heap;
}

///===----------------------------------------------------------------------===///
/// Collect Allocations Used in EDTs
///===----------------------------------------------------------------------===///
void CreateDbsPass::collectMemrefs() {
  memrefInfo.clear();
  module.walk([&](EdtOp edt) {
    edt.walk([&](Operation *op) {
      /// Skip self-references
      if (auto edtOp = dyn_cast<EdtOp>(op)) {
        if (edtOp == edt)
          return;
      }

      /// Check all operands of all operations for memory references
      for (Value operand : op->getOperands()) {
        if (!isa<MemRefType>(operand.getType()))
          continue;

        /// Get the underlying value of the memory reference
        Operation *underlyingOp =
            ValueAnalysis::getUnderlyingOperation(operand);
        if (!underlyingOp) {
          if (auto load = operand.getDefiningOp<memref::LoadOp>())
            underlyingOp =
                ValueAnalysis::getUnderlyingOperation(load.getMemref());
        }
        if (!underlyingOp) {
          op->emitError("cannot trace memref operand to its underlying "
                        "allocation; all memory accessed by EDTs must have a "
                        "traceable allocation");
          signalPassFailure();
          return;
        }

        /// Skip allocations that are already a DB (e.g. produced directly by
        /// the tensor-path lowering in SDE->ARTS, which materializes
        /// `arts.db_alloc` + `arts.db_acquire` with `bufferization.to_tensor`
        /// views up front). Those handles are already first-class DBs; only
        /// the memref->DB conversion below needs to run on non-DB allocs.
        if (isa<DbAllocOp>(underlyingOp))
          continue;

        /// If it is found, check the parent edt
        EdtOp parentEdt = underlyingOp->getParentOfType<EdtOp>();

        /// Create memref info for the underlying operation
        MemrefInfo &info = memrefInfo[underlyingOp];
        info.alloc = underlyingOp;
        info.parentEdt = parentEdt;

        /// If the parent edt is different from the current edt, it means
        /// it is an external dependency
        if (parentEdt != edt) {
          edtExternalValues[edt].insert(operand);
          info.usedByOtherEdts = true;
        }
      }
    });
  });
  ARTS_DEBUG("collectMemrefs: Total " << memrefInfo.size()
                                      << " unique memrefs");
}

///===----------------------------------------------------------------------===///
/// Collect Control DB Operations
///===----------------------------------------------------------------------===///
void CreateDbsPass::collectControlDbOps() {
  /// First, extract dependency mode and index information from all db_control
  /// ops before erasing them
  SmallVector<DbControlOp, 4> dbControlOps;
  module.walk([&](DbControlOp dbControl) {
    dbControlOps.push_back(dbControl);
    Value ptr = dbControl.getPtr();
    ArtsMode mode = dbControl.getMode();
    auto indices = dbControl.getIndices();
    auto offsets = dbControl.getOffsets();
    auto sizes = dbControl.getSizes();

    /// Get the underlying operation for this pointer
    Operation *underlyingOp = ValueAnalysis::getUnderlyingOperation(ptr);
    if (!underlyingOp)
      return;
    if (isa<DbAllocOp>(underlyingOp))
      return; // already a DB, no DbControl conversion required

    ARTS_DEBUG(" - Found DbControlOp for "
               << *underlyingOp << " with mode: " << mode << ", "
               << indices.size() << " indices, " << offsets.size()
               << " offsets, " << sizes.size() << " sizes");

    /// Get user EDT
    EdtOp userEdt = dyn_cast_or_null<EdtOp>(DbAnalysis::findUserEdt(dbControl));

    SmallVector<Value, 4> indexValues(indices.begin(), indices.end());
    SmallVector<Value, 4> offsetValues(offsets.begin(), offsets.end());
    SmallVector<Value, 4> sizeValues(sizes.begin(), sizes.end());

    /// Track all operations in this EDT that use this memref AND match the
    /// indices
    SmallVector<Operation *, 4> opsToRewrite;
    if (userEdt) {
      /// Walk the user EDT and collect operations whose leading indices match
      /// the pinned DbControl dimensions. The exact SSA prefix check
      /// distinguishes chunks such as A[i-1], A[i], and A[i+1].
      userEdt.walk([&](Operation *op) {
        if (auto load = dyn_cast<memref::LoadOp>(op)) {
          if (ValueAnalysis::getUnderlyingOperation(load.getMemref()) ==
              underlyingOp) {
            if (DbAnalysis::opMatchesAccessMode(op, underlyingOp, mode) &&
                ValueAnalysis::hasValueRangePrefix(load.getIndices(),
                                                   indexValues))
              opsToRewrite.push_back(op);
          }
        } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
          if (ValueAnalysis::getUnderlyingOperation(store.getMemref()) ==
              underlyingOp) {
            if (DbAnalysis::opMatchesAccessMode(op, underlyingOp, mode) &&
                ValueAnalysis::hasValueRangePrefix(store.getIndices(),
                                                   indexValues))
              opsToRewrite.push_back(op);
          }
        }
      });
      ARTS_DEBUG("    - Found "
                 << opsToRewrite.size()
                 << " operations to rewrite for this specific dependency");
    }

    memrefInfo[underlyingOp].addAccess(
        userEdt, mode, indexValues, offsetValues, sizeValues, opsToRewrite,
        /*preserveDepEdge=*/!isLocalityOnly(dbControl.getOperation()));
  });

  /// Now erase all db_control ops, replacing uses with pointer.
  /// We must also update edtExternalValues: any Value that was a result of an
  /// erased DbControlOp becomes dangling after erase().  Replace such entries
  /// with the ptr operand that replaceAllUsesWith substitutes in the IR.
  for (DbControlOp dbControl : dbControlOps) {
    Value subview = dbControl.getSubview();
    Value ptr = dbControl.getPtr();
    /// Update edtExternalValues to replace any reference to the about-to-be-
    /// erased subview result with the ptr it forwards to.
    for (auto &entry : edtExternalValues) {
      if (entry.second.contains(subview)) {
        entry.second.remove(subview);
        entry.second.insert(ptr);
      }
    }
    subview.replaceAllUsesWith(ptr);
    dbControl.erase();
  }

  /// Clear all existing EDT dependencies and block args that are not used
  /// within the block.  We must also remove any about-to-be-erased block
  /// arguments from edtExternalValues to avoid dangling Value references.
  ///
  /// EDTs whose dependencies are already fully wired to DbAllocOp-backed
  /// `arts.db_acquire` ops (produced up-front by the tensor-path SDE->ARTS
  /// lowering) are left untouched: createDbAcquireOps would not re-wire them
  /// (they never enter `edtExternalValues` since their underlying ops are
  /// already DbAlloc), so stripping their operands would leave the verifier
  /// with more block arguments than dependencies.
  module.walk([&](EdtOp edt) {
    Block &block = edt.getBody().front();

    /// If every EDT dependency comes from a DbAllocOp-backed acquire already,
    /// skip the clear step — the acquires ARE the deps.
    bool allDepsArePreWiredAcquires = edt.getDependencies().size() > 0;
    for (Value dep : edt.getDependencies()) {
      Operation *underlying = ValueAnalysis::getUnderlyingOperation(dep);
      bool dbAllocBacked = underlying && isa<DbAllocOp>(underlying);
      if (!dbAllocBacked) {
        if (auto acquire = dep.getDefiningOp<DbAcquireOp>()) {
          dbAllocBacked = true;
          Operation *sourceUnderlying =
              ValueAnalysis::getUnderlyingOperation(acquire.getSourcePtr());
          if (sourceUnderlying)
            dbAllocBacked = isa<DbAllocOp>(sourceUnderlying) || dbAllocBacked;
        }
      }
      if (!dbAllocBacked) {
        allDepsArePreWiredAcquires = false;
        break;
      }
    }
    if (allDepsArePreWiredAcquires)
      return;

    /// Collect unused block arguments so we can scrub them from
    /// edtExternalValues BEFORE the actual erasure invalidates them.
    SmallVector<BlockArgument> unusedArgs;
    for (BlockArgument arg : block.getArguments()) {
      bool isUsed = false;
      block.walk([&](Operation *op) {
        for (Value operand : op->getOperands()) {
          if (operand == arg) {
            isUsed = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (!isUsed)
        unusedArgs.push_back(arg);
    }

    /// Remove stale Values from edtExternalValues before erasure.
    auto extIt = edtExternalValues.find(edt);
    if (extIt != edtExternalValues.end()) {
      for (BlockArgument arg : unusedArgs)
        extIt->second.remove(arg);
    }

    /// Now erase unused block arguments.
    block.eraseArguments(
        [&](BlockArgument arg) { return llvm::is_contained(unusedArgs, arg); });

    /// Preserve the route operand when clearing dependencies
    SmallVector<Value> preservedOperands;
    if (Value route = edt.getRoute())
      preservedOperands.push_back(route);
    edt->setOperands(preservedOperands);
  });
}

/// Create DB Allocation Operations
void CreateDbsPass::createDbAllocOps() {
  /// All memrefs in memrefInfo are converted to DBs
  for (auto &entry : memrefInfo) {
    Operation *alloc = entry.first;
    ARTS_DEBUG("Creating DB Alloc Op for memref " << *alloc);
    MemrefInfo &info = entry.second;
    Location loc = alloc->getLoc();
    OpBuilder::InsertionGuard IG(AC->getBuilder());
    AC->setInsertionPointAfter(alloc);

    /// Skip local allocations that do not escape their defining EDT.
    /// These buffers are private to a single EDT and do not need datablocks.
    if (info.parentEdt && !info.usedByOtherEdts) {
      bool escapesEdt = false;
      for (auto &use : alloc->getUses()) {
        Operation *user = use.getOwner();
        if (!info.parentEdt->isAncestor(user)) {
          escapesEdt = true;
          break;
        }
      }
      if (!escapesEdt) {
        ARTS_DEBUG(" - Skipping local EDT allocation (no external uses)");
        ++numPrivateEdtAllocsSkipped;
        continue;
      }
    }

    /// Reject allocations with nested memref element types.  These are
    /// double-indirection patterns (e.g. float**) that MemrefNormalization
    /// should have raised to N-dimensional memrefs.  If they reach here,
    /// the pipeline has a gap that will produce broken executables.
    if (auto memRefType = dyn_cast<MemRefType>(alloc->getResult(0).getType())) {
      if (isa<MemRefType>(memRefType.getElementType())) {
        alloc->emitError("un-normalizable nested memref pattern (element type "
                         "is memref); MemrefNormalization should have raised "
                         "this to an N-dimensional memref");
        signalPassFailure();
        return;
      }
    }

    /// Infer allocation type based on the defining operation
    DbAllocType allocType = inferAllocType(alloc);

    /// Use the dependency mode from DbControlOp
    ArtsMode mode = info.accessMode;
    ARTS_DEBUG(" - Using access mode " << mode);

    /// Set DbMode based on the access mode
    DbMode dbMode = DbMode::write;
    if (mode == ArtsMode::in)
      dbMode = DbMode::read;
    else if (mode == ArtsMode::out || mode == ArtsMode::inout)
      dbMode = DbMode::write;

    /// Get the element type of the memref
    Value allocValue = alloc->getResult(0);
    MemRefType memRefType = cast<MemRefType>(allocValue.getType());
    Type elementType = memRefType.getElementType();
    while (auto nestedMemRef = dyn_cast<MemRefType>(elementType))
      elementType = nestedMemRef.getElementType();

    /// Determine allocation granularity using unified heuristics
    SmallVector<Value> sizes, logicalElementSizes, elementSizes;
    const bool isRankZero = memRefType.getRank() == 0;
    const unsigned rank = std::max<unsigned>(1, memRefType.getRank());
    const auto accessPatternInfo = info.getAccessPatternInfo();

    /// Store block sizes for later use in createDbAcquireOps
    if (accessPatternInfo.hasChunkDeps) {
      info.blockSizes.assign(accessPatternInfo.blockSizes.begin(),
                             accessPatternInfo.blockSizes.end());
    }

    /// Build the original logical allocation extents. These are the source
    /// extents for any SDE-authored physical layout plan.
    logicalElementSizes.reserve(rank);
    for (unsigned i = 0; i < rank; ++i) {
      if (!isRankZero && memRefType.isDynamicDim(i)) {
        logicalElementSizes.push_back(
            AC->create<arts::DbDimOp>(loc, allocValue, (int64_t)i));
      } else {
        int64_t dimSize =
            isRankZero ? 1 : static_cast<int64_t>(memRefType.getDimSize(i));
        logicalElementSizes.push_back(AC->createIndexConstant(dimSize, loc));
      }
    }

    /// Coarse is the conservative fallback. If SDE has already authored a
    /// physical tensor layout plan, materialize it now so CreateDbs is the
    /// point where the final DB shape first appears.
    DbRewritePlan plan(PartitionMode::coarse);
    sizes.push_back(AC->createIndexConstant(1, loc));
    elementSizes.assign(logicalElementSizes.begin(), logicalElementSizes.end());
    PartitionMode partitionMode = PartitionMode::coarse;
    Operation *planSource = nullptr;

    planSource = findPhysicalLayoutPlanSource(alloc);
    if (planSource && DbUtils::hasNonPartitionableHostViewUses(allocValue)) {
      InFlightDiagnostic diag =
          alloc->emitError("SDE-authored physical DB layout cannot be "
                           "materialized because the allocation still has "
                           "whole-view host uses");
      diag.attachNote(planSource->getLoc()) << "layout plan source";
      signalPassFailure();
      return;
    }
    if (planSource) {
      FailureOr<DbRewritePlan> resolvedPlan = resolvePhysicalDbLayoutPlan(
          planSource, ValueRange(logicalElementSizes), AC->getBuilder(), loc);
      if (failed(resolvedPlan)) {
        InFlightDiagnostic diag =
            alloc->emitError("invalid SDE-authored physical DB layout plan");
        diag.attachNote(planSource->getLoc()) << "layout plan source";
        signalPassFailure();
        return;
      }

      plan = *resolvedPlan;
      sizes.assign(plan.outerSizes.begin(), plan.outerSizes.end());
      elementSizes.assign(plan.innerSizes.begin(), plan.innerSizes.end());
      partitionMode = plan.mode;
      ARTS_DEBUG(" - Using SDE-authored physical DB layout");
    } else if (DbUtils::hasNonPartitionableHostViewUses(allocValue)) {
      ARTS_DEBUG(" - Keeping coarse DB layout due to host whole-view use");
    }

    if (plan.isCoarse())
      ARTS_DEBUG(" - Using coarse-grained allocation");

    /// Create the db_alloc operation
    /// DBs without an explicit route stay on the creating node. Lowering
    /// materializes this sentinel into the runtime hint instead of pinning the
    /// allocation to rank 0.
    auto route = createCurrentNodeRoute(AC->getBuilder(), loc);
    auto dbAllocOp =
        AC->create<DbAllocOp>(loc, mode, route, allocType, dbMode, elementType,
                              sizes, elementSizes, partitionMode);
    ++numDbAllocsCreated;

    projectSemanticContractToDbValue(alloc, dbAllocOp.getOperation(),
                                     dbAllocOp.getPtr());
    projectStructuredPlanToDbValue(planSource, dbAllocOp);

    /// Initialize global DBs
    initializeGlobalDbIfNeeded(alloc, dbAllocOp, plan, allocType);

    /// Copy ARTS ID from original allocation to DbAllocOp
    copyArtsMetadataAttrs(alloc, dbAllocOp.getOperation());

    info.rewritePlan = plan;

    /// Record allocation strategy decision for diagnostics
    AM->getDbHeuristics().recordDecision(
        "AllocationStrategy", true,
        plan.isCoarse() ? "Coarse allocation"
                        : "SDE-authored physical block allocation",
        alloc,
        {{"outerRank", static_cast<int64_t>(plan.outerRank())},
         {"innerRank", static_cast<int64_t>(plan.innerRank())}});

    /// Store mappings for later use
    info.dbAllocOp = dbAllocOp;
    dbPtrToOriginalAlloc[dbAllocOp] = alloc;

    /// Insert DbFreeOp and remove associated dealloc operations
    insertDbFreeForDbAlloc(dbAllocOp, alloc);

    /// Rewire uses of the original allocation to use the db_alloc operation.
    /// If the allocation has a defining EDT, rewrite within that EDT. If it
    /// lives outside any EDT (host/global), rewrite all uses so host code sees
    /// the datablock contents as well.
    if (info.parentEdt) {
      rewriteUsesInParentEdt(info);
    } else {
      /// Redirect non-EDT aliases/host accesses to the canonical DB-backed
      /// coarse view so initialization and verification code observe the
      /// datablock contents instead of a disconnected host allocation.
      rewriteUsesEverywhereWithPlan(alloc, dbAllocOp, plan);

      /// Step 2: Rewrite EDT uses without explicit deps
      Type elementMemRefType = dbAllocOp.getAllocatedElementType();
      SmallVector<Operation *> edtUsesToRewrite;
      for (auto &use : alloc->getUses()) {
        Operation *user = use.getOwner();
        if (user == dbAllocOp.getOperation())
          continue;
        if (auto parentEdt = user->getParentOfType<EdtOp>()) {
          if (info.edtToDeps.count(parentEdt) == 0)
            edtUsesToRewrite.push_back(user);
        }
      }
      if (!edtUsesToRewrite.empty()) {
        PartitionInfo partInfo;
        partInfo.mode = PartitionMode::coarse;
        DbElementWiseIndexer indexer(partInfo, 0, rank, {});
        indexer.transformOps(edtUsesToRewrite, dbAllocOp.getPtr(),
                             elementMemRefType, *AC, opsToRemove);
      }
    }
  }
}

/// Initialize global constants for DB-backed globals.
void CreateDbsPass::initializeGlobalDbIfNeeded(Operation *alloc,
                                               DbAllocOp dbAllocOp,
                                               const DbRewritePlan &plan,
                                               DbAllocType allocType) {
  if (allocType != DbAllocType::global)
    return;

  auto getGlobal = dyn_cast<memref::GetGlobalOp>(alloc);
  if (!getGlobal)
    return;

  auto globalOp = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      getGlobal, getGlobal.getNameAttr());
  if (!globalOp || !globalOp.getInitialValue().has_value())
    return;
  ++numGlobalDbInitializations;

  Location loc = alloc->getLoc();
  OpBuilder::InsertionGuard initGuard(AC->getBuilder());
  AC->setInsertionPointAfter(dbAllocOp);
  Value globalMemref = AC->create<memref::GetGlobalOp>(loc, globalOp.getType(),
                                                       getGlobal.getNameAttr());

  SmallVector<Value> zeroIndices;
  zeroIndices.reserve(dbAllocOp.getSizes().size());
  for (size_t i = 0; i < dbAllocOp.getSizes().size(); ++i)
    zeroIndices.push_back(AC->createIndexConstant(0, loc));

  auto memRefType = cast<MemRefType>(globalOp.getType());
  unsigned rank = memRefType.getRank();
  if (rank == 0) {
    Value dbRef = AC->create<DbRefOp>(loc, dbAllocOp.getPtr(), zeroIndices);
    Value initVal = AC->create<memref::LoadOp>(loc, globalMemref);
    AC->create<memref::StoreOp>(loc, initVal, dbRef);
    return;
  }

  auto storeToDb = [&](Value initVal, ArrayRef<Value> indices) {
    if (plan.usesBlockedLayout()) {
      PartitionInfo blockInfo;
      blockInfo.mode = PartitionMode::block;
      blockInfo.sizes.assign(plan.blockSizes.begin(), plan.blockSizes.end());
      blockInfo.partitionedDims.assign(plan.partitionedDims.begin(),
                                       plan.partitionedDims.end());
      SmallVector<Value> startBlocks;
      startBlocks.reserve(plan.blockSizes.size());
      for (size_t i = 0; i < plan.blockSizes.size(); ++i)
        startBlocks.push_back(AC->createIndexConstant(0, loc));
      DbBlockIndexer indexer(blockInfo, startBlocks, plan.outerRank(),
                             plan.innerRank());
      LocalizedIndices localized =
          indexer.localize(indices, AC->getBuilder(), loc);
      Value dbRef =
          AC->create<DbRefOp>(loc, dbAllocOp.getAllocatedElementType(),
                              dbAllocOp.getPtr(), localized.dbRefIndices);
      AC->create<memref::StoreOp>(loc, initVal, dbRef, localized.memrefIndices);
      return;
    }

    Value dbRef = AC->create<DbRefOp>(loc, dbAllocOp.getPtr(), zeroIndices);
    AC->create<memref::StoreOp>(loc, initVal, dbRef, indices);
  };

  SmallVector<Value> indices;
  indices.reserve(rank);
  auto emitLoopCopy = [&](auto &self, unsigned dim) -> void {
    if (dim == rank) {
      Value initVal = AC->create<memref::LoadOp>(loc, globalMemref, indices);
      storeToDb(initVal, indices);
      return;
    }

    Value lower = AC->createIndexConstant(0, loc);
    Value upper;
    if (memRefType.isDynamicDim(dim)) {
      upper = AC->create<memref::DimOp>(loc, globalMemref, dim);
    } else {
      upper = AC->createIndexConstant(memRefType.getDimSize(dim), loc);
    }
    Value step = AC->createIndexConstant(1, loc);

    auto loop = AC->create<scf::ForOp>(loc, lower, upper, step);
    OpBuilder::InsertionGuard loopGuard(AC->getBuilder());
    AC->setInsertionPointToStart(loop.getBody());
    indices.push_back(loop.getInductionVar());
    self(self, dim + 1);
    indices.pop_back();
  };
  emitLoopCopy(emitLoopCopy, 0);
}

/// Find parent scope's acquired handle for a datablock
/// Checks if the parent EDT already has an acquired handle for the given
/// DbAllocOp.
Value CreateDbsPass::findLocalAcquireView(EdtOp edt, Operation *dbAllocOp,
                                          bool requiresIndexedAccess) {
  auto dbAlloc = dyn_cast_or_null<DbAllocOp>(dbAllocOp);
  if (!dbAlloc || !edt)
    return nullptr;

  Type targetType = dbAlloc.getPtr().getType();
  Block &body = edt.getBody().front();
  auto deps = edt.getDependenciesAsVector();

  auto matchesDb = [&](Value v) -> bool {
    if (!v || v.getType() != targetType)
      return false;

    Operation *db = DbUtils::getUnderlyingDbAlloc(v);
    if (!db || db != dbAllocOp)
      return false;

    if (requiresIndexedAccess) {
      if (Operation *dbOp = DbUtils::getUnderlyingDb(v)) {
        if (DbAnalysis::hasSingleSize(dbOp))
          return false;
      }
    }
    return true;
  };

  for (auto [idx, dep] : llvm::enumerate(deps)) {
    if (idx >= body.getNumArguments())
      break;
    if (matchesDb(dep))
      return body.getArgument(idx);
  }

  return nullptr;
}

Value CreateDbsPass::findParentAcquireSource(EdtOp edt, Operation *dbAllocOp,
                                             ArtsMode requestedMode,
                                             bool requiresIndexedAccess) {
  auto dbAlloc = dyn_cast_or_null<DbAllocOp>(dbAllocOp);
  if (!dbAlloc)
    return nullptr;

  ARTS_DEBUG("   - Checking if parent has acquired handle for this memref");

  Block *parentBlock = edt->getBlock();
  if (!parentBlock)
    return nullptr;

  Type targetType = dbAlloc.getPtr().getType();

  auto matchesDb = [&](Value v) -> bool {
    if (!v || v.getType() != targetType)
      return false;

    Operation *db = DbUtils::getUnderlyingDbAlloc(v);
    if (!db || db != dbAllocOp)
      return false;

    /// If the caller intends to index into this handle, avoid reusing a
    /// single-element datablock view.
    if (requiresIndexedAccess) {
      if (Operation *dbOp = DbUtils::getUnderlyingDb(v)) {
        if (DbAnalysis::hasSingleSize(dbOp))
          return false;
      }
    }

    /// Avoid treating the DbAllocOp results themselves as reusable handles
    if (auto *defOp = v.getDefiningOp())
      if (isa<DbAllocOp>(defOp))
        return false;

    if (Operation *dbOp = DbUtils::getUnderlyingDb(v)) {
      if (auto acquire = dyn_cast<DbAcquireOp>(dbOp)) {
        if (!DbAnalysis::accessModeCanSeedNestedAcquire(acquire.getMode(),
                                                        requestedMode))
          return false;
      }
    }

    return true;
  };

  /// Walk outward through enclosing block arguments and look for a handle that
  /// is already available in scope at the current insertion point. This handles
  /// nested EDTs inside scf.for regions where the immediate block has only loop
  /// IV arguments.
  Block *scanBlock = parentBlock;
  Operation *anchorOp = edt.getOperation();

  while (scanBlock && anchorOp) {
    for (BlockArgument arg : scanBlock->getArguments()) {
      if (matchesDb(arg)) {
        ARTS_DEBUG("   - Found matching handle in enclosing block arguments");
        return arg;
      }
    }

    Operation *ownerOp = scanBlock->getParentOp();
    if (!ownerOp)
      break;
    scanBlock = ownerOp->getBlock();
    anchorOp = ownerOp;
  }

  ARTS_DEBUG("   - No existing handle found in parent scope");
  return nullptr;
}

/// Create DB Acquire Operations for EDT Dependencies
void CreateDbsPass::createDbAcquireOps(EdtOp edt,
                                       SetVector<Value> &externalDeps) {
  ARTS_DEBUG(" - Creating DbAcquire operations for "
             << externalDeps.size() << " external dependencies");

  /// Accumulate dependency operands to set on the EDT.  The tensor-path
  /// SDE->ARTS lowering may have already wired DbAcquire-backed dependencies
  /// before this pass; preserve those while appending any dependencies that
  /// CreateDbs materializes from ordinary memrefs.
  SmallVector<Value> dependencyOperands(edt.getDependencies().begin(),
                                        edt.getDependencies().end());

  /// Track which DbAllocOps have already been processed for coarse-grained
  /// acquires. For fine-grained acquires, we may need multiple acquires per
  /// DbAllocOp with different indices/offsets, so we don't skip those.
  DenseSet<Operation *> processedCoarseAllocs;

  /// For each external value, create acquire and release operations
  for (Value externalDep : externalDeps) {
    /// Locate the underlying operation
    Operation *underlyingOp =
        ValueAnalysis::getUnderlyingOperation(externalDep);
    if (!underlyingOp)
      if (auto load = externalDep.getDefiningOp<memref::LoadOp>())
        underlyingOp = ValueAnalysis::getUnderlyingOperation(load.getMemref());
    if (!underlyingOp) {
      if (auto *defOp = externalDep.getDefiningOp())
        defOp->emitError(
            "cannot trace external EDT dependency to its underlying "
            "allocation; all memory accessed by EDTs must have a traceable "
            "allocation");
      else
        edt->emitError("cannot trace external EDT dependency (block argument) "
                       "to its underlying allocation");
      signalPassFailure();
      return;
    }
    if (isa<DbAllocOp>(underlyingOp))
      continue; // already a DB, no DbAlloc wiring required

    /// Get the memref info for the underlying operation
    MemrefInfo &info = memrefInfo[underlyingOp];
    DbAllocOp dbAllocOp = info.dbAllocOp;
    ARTS_DEBUG("DbAllocOp " << dbAllocOp);
    if (!dbAllocOp) {
      underlyingOp->emitError("no DbAllocOp for memref operand — "
                              "un-normalizable nested memref pattern reached "
                              "CreateDbs");
      signalPassFailure();
      return;
    }

    // Skip if the db_alloc is not visible at the insertion point (before
    // `edt`).  This happens when the alloc lives inside a nested EDT whose
    // body does not dominate the outer scope.
    if (auto allocParentEdt = dbAllocOp->getParentOfType<EdtOp>()) {
      if (allocParentEdt != edt && !allocParentEdt->isAncestor(edt)) {
        ARTS_DEBUG("   - Skipping: DbAllocOp in nested EDT, not in scope");
        continue;
      }
    }

    /// Get the source guid and ptr
    auto sourceGuid = dbAllocOp.getGuid();
    auto sourcePtr = dbAllocOp.getPtr();
    assert((sourceGuid && sourcePtr) && "Source guid and ptr must be non-null");

    /// Create acquire operation (pass both source guid and ptr)
    OpBuilder::InsertionGuard IG(AC->getBuilder());
    AC->setInsertionPoint(edt);

    /// Create acquires based on DbControlOp dependencies
    const auto &deps = info.edtToDeps[edt];
    const auto &plan = info.rewritePlan.value();

    /// Deduplicate: one acquire per allocation per EDT
    if (!processedCoarseAllocs.insert(dbAllocOp.getOperation()).second) {
      ARTS_DEBUG("   - Skipping duplicate acquire for already-processed "
                 "DbAllocOp");
      continue;
    }

    /// Build full DB-space range for the already-created physical layout.
    SmallVector<Value> dbOffsets, dbSizes;
    SmallVector<Value> allocSizes(dbAllocOp.getSizes().begin(),
                                  dbAllocOp.getSizes().end());
    if (allocSizes.empty()) {
      dbOffsets.push_back(AC->createIndexConstant(0, edt.getLoc()));
      dbSizes.push_back(AC->createIndexConstant(1, edt.getLoc()));
    } else {
      for (Value s : allocSizes) {
        dbOffsets.push_back(AC->createIndexConstant(0, edt.getLoc()));
        dbSizes.push_back(s);
      }
    }

    SmallVector<SmallVector<const MemrefInfo::DbDep *, 4>, 4> depGroups;
    SmallVector<ArtsMode, 4> depGroupModes;
    SmallVector<bool, 4> depGroupPreserveDepEdges;
    SmallVector<PartitionMode, 4> depGroupPartitionModes;
    if (deps.empty()) {
      depGroups.emplace_back();
      depGroupModes.push_back(ArtsMode::uninitialized);
      depGroupPreserveDepEdges.push_back(true);
      depGroupPartitionModes.push_back(
          plan.usesBlockedLayout() ? plan.mode : PartitionMode::coarse);
    } else {
      for (const auto &dep : deps) {
        PartitionMode depPartitionMode =
            plan.usesBlockedLayout() ? plan.mode : PartitionMode::coarse;
        if (!plan.usesBlockedLayout()) {
          if (!dep.indices.empty()) {
            depPartitionMode = PartitionMode::fine_grained;
          } else if (!dep.offsets.empty() && !dep.sizes.empty()) {
            depPartitionMode = PartitionMode::block;
          }
        }
        int groupIndex = -1;
        for (auto [idx, groupMode] : llvm::enumerate(depGroupModes)) {
          if (groupMode == dep.mode &&
              depGroupPreserveDepEdges[idx] == dep.preserveDepEdge &&
              depGroupPartitionModes[idx] == depPartitionMode) {
            groupIndex = static_cast<int>(idx);
            break;
          }
        }
        if (groupIndex < 0) {
          depGroupModes.push_back(dep.mode);
          depGroupPreserveDepEdges.push_back(dep.preserveDepEdge);
          depGroupPartitionModes.push_back(depPartitionMode);
          depGroups.emplace_back();
          depGroups.back().push_back(&dep);
          continue;
        }
        depGroups[groupIndex].push_back(&dep);
      }
    }

    /// Track the first dependency-backed local view materialized for this DB
    /// so locality-only acquires inside the EDT can derive nested views from
    /// an in-scope handle instead of the outer db_alloc result.
    Value inScopeAcquireView;

    for (auto [groupIdx, depGroup] : llvm::enumerate(depGroups)) {
      /// Create the parent acquire over the full physical DB range selected at
      /// allocation time. Element-space partition hints from DbControlOp remain
      /// attached so task lowering can derive worker-local dependency windows.
      /// Keep different access modes in separate acquires so downstream
      /// rec_dep lowering preserves read vs write slots.
      const bool preserveDepEdge = depGroupPreserveDepEdges[groupIdx];
      ARTS_DEBUG(" - Creating coarse-grained acquire group "
                 << groupIdx << " for mode " << depGroupModes[groupIdx]
                 << ", preserveDepEdge=" << preserveDepEdge);

      /// Collect partition hints from this group's deps.
      SmallVector<Value> partIndices, partOffsets, partSizes;
      SmallVector<int32_t> indicesSegments, offsetsSegments, sizesSegments;
      SmallVector<int32_t> entryModes;
      ArtsMode acquireMode = depGroupModes[groupIdx];

      for (const MemrefInfo::DbDep *dep : depGroup) {
        PartitionMode entryMode = PartitionMode::coarse;
        if (plan.usesBlockedLayout()) {
          entryMode = plan.mode;
        } else if (!dep->indices.empty()) {
          entryMode = PartitionMode::fine_grained;
        } else if (!dep->offsets.empty() && !dep->sizes.empty()) {
          entryMode = PartitionMode::block;
        }
        entryModes.push_back(static_cast<int32_t>(entryMode));

        indicesSegments.push_back(static_cast<int32_t>(dep->indices.size()));
        for (Value v : dep->indices)
          partIndices.push_back(v);

        offsetsSegments.push_back(static_cast<int32_t>(dep->offsets.size()));
        for (Value v : dep->offsets)
          partOffsets.push_back(v);

        sizesSegments.push_back(static_cast<int32_t>(dep->sizes.size()));
        for (Value v : dep->sizes)
          partSizes.push_back(v);
      }

      /// If this EDT has no explicit db_control hints for the memref, infer
      /// per-EDT access from actual loads/stores before falling back.
      if (acquireMode == ArtsMode::uninitialized) {
        acquireMode = AM->getDbAnalysis().inferEdtAccessMode(underlyingOp, edt);
        if (acquireMode == ArtsMode::uninitialized) {
          acquireMode = (info.accessMode == ArtsMode::uninitialized)
                            ? ArtsMode::inout
                            : info.accessMode;
        }
      }

      PartitionMode partMode = depGroupPartitionModes[groupIdx];
      if (!plan.usesBlockedLayout() && !entryModes.empty())
        partMode = static_cast<PartitionMode>(entryModes[0]);

      Value acqGuid = sourceGuid;
      Value acqPtr = sourcePtr;
      if (!preserveDepEdge && inScopeAcquireView) {
        ARTS_DEBUG("   - Reusing existing datablock handle inside EDT");
        acqGuid = Value();
        acqPtr = inScopeAcquireView;
      }
      if (acqPtr == sourcePtr)
        if (Value availableHandle =
                findParentAcquireSource(edt, dbAllocOp, acquireMode)) {
          ARTS_DEBUG("   - Reusing existing datablock handle in parent scope");
          acqGuid = Value();
          acqPtr = availableHandle;
        }

      auto createAcquire = [&](Location acquireLoc) {
        Type ptrType = acqPtr ? acqPtr.getType() : Type();
        if (DbUtils::getUnderlyingDbAlloc(acqPtr)) {
          return AC->create<DbAcquireOp>(
              acquireLoc, acquireMode, acqGuid, acqPtr, partMode,
              /*indices=*/SmallVector<Value>{}, /*offsets=*/dbOffsets,
              /*sizes=*/dbSizes,
              /*partition_indices=*/partIndices,
              /*partition_offsets=*/partOffsets,
              /*partition_sizes=*/partSizes);
        }

        return AC->create<DbAcquireOp>(
            acquireLoc, acquireMode, acqGuid, acqPtr, ptrType, partMode,
            /*indices=*/SmallVector<Value>{}, /*offsets=*/dbOffsets,
            /*sizes=*/dbSizes,
            /*partition_indices=*/partIndices,
            /*partition_offsets=*/partOffsets,
            /*partition_sizes=*/partSizes);
      };

      if (!preserveDepEdge)
        AC->setInsertionPointToStart(&edt.getBody().front());
      auto acquireOp = createAcquire(edt.getLoc());
      ++numDbAcquireGroupsCreated;

      projectSemanticContractToDbValue(
          edt.getOperation(), acquireOp.getOperation(), acquireOp.getPtr());

      if (!depGroup.empty()) {
        /// Explicit DbControlOp-derived acquires always preserve their access
        /// mode contract. Only true predecessor controls also preserve the EDT
        /// dependency edge.
        acquireOp.setPreserveAccessMode();
        if (preserveDepEdge)
          acquireOp.setPreserveDepEdge();
      }

      acquireOp.setPartitionSegments(indicesSegments, offsetsSegments,
                                     sizesSegments, entryModes);

      ARTS_DEBUG(" - Created acquire group with "
                 << depGroup.size() << " partition entries, primary mode="
                 << static_cast<int>(partMode) << ": " << acquireOp);

      Value localAcquireView = acquireOp.getPtr();
      if (preserveDepEdge) {
        auto sourceType = dyn_cast<MemRefType>(localAcquireView.getType());
        BlockArgument dbAcquireArg =
            edt.getBody().front().addArgument(sourceType, edt.getLoc());
        dependencyOperands.push_back(localAcquireView);
        localAcquireView = dbAcquireArg;
        if (!inScopeAcquireView)
          inScopeAcquireView = localAcquireView;
      }

      SmallVector<Operation *> opsToRewrite;
      llvm::SmallPtrSet<Operation *, 16> seenOps;
      auto addOpIfNew = [&](Operation *op) {
        if (!op)
          return;
        if (seenOps.insert(op).second)
          opsToRewrite.push_back(op);
      };

      bool missingOpsForDeps = deps.empty();
      for (const MemrefInfo::DbDep *dep : depGroup) {
        for (Operation *op : dep->operations)
          addOpIfNew(op);
      }

      /// Fallback: collect all operations using this memref in EDT when there
      /// are no explicit dependency entries for the memref.
      if (opsToRewrite.empty() || missingOpsForDeps) {
        ARTS_DEBUG(
            " - Fallback: collecting all operations using this memref in EDT");
        edt.walk([&](Operation *op) {
          if (op->getParentOfType<EdtOp>() != edt)
            return;
          if (!DbAnalysis::opMatchesAccessMode(op, underlyingOp, acquireMode))
            return;
          addOpIfNew(op);
        });
        ARTS_DEBUG(" - Found " << opsToRewrite.size()
                               << " operations to rewrite");
      }

      if (!opsToRewrite.empty()) {
        SmallVector<Value> dbRefIndices;
        dbRefIndices.push_back(AC->createIndexConstant(0, edt.getLoc()));
        rewriteOpsToUseDbAcquire(edt, opsToRewrite, acquireOp, dbRefIndices,
                                 dbOffsets, localAcquireView, plan);
      }

      OpBuilder::InsertionGuard releaseGuard(AC->getBuilder());
      AC->setInsertionPoint(edt.getBody().front().getTerminator());
      AC->create<DbReleaseOp>(edt.getLoc(), localAcquireView);
    }
  }

  /// After processing all memrefs, set EDT dependencies
  edt.setDependencies(dependencyOperands);
}

/// Insert db_free for DbAllocOp at the appropriate location and remove
/// associated dealloc operations
void CreateDbsPass::insertDbFreeForDbAlloc(DbAllocOp dbAlloc,
                                           Operation *alloc) {
  Location loc = dbAlloc.getLoc();
  Value allocResult = alloc->getResult(0);

  /// Find associated dealloc operation
  std::optional<Operation *> deallocOp = memref::findDealloc(allocResult);
  if (deallocOp.has_value() && *deallocOp)
    opsToRemove.insert(*deallocOp);

  /// Determine where to insert DbFreeOp based on where dbAlloc is located
  /// Insert at the end of the block containing dbAlloc
  Block *allocBlock = dbAlloc->getBlock();
  Operation *insertionPoint = allocBlock->getTerminator();

  /// Insert DbFreeOp if we found a valid insertion point
  assert(insertionPoint && "Could not find insertion point for DbFreeOp");
  OpBuilder::InsertionGuard IG(AC->getBuilder());
  AC->setInsertionPoint(insertionPoint);
  AC->create<DbFreeOp>(loc, dbAlloc.getGuid());
  AC->create<DbFreeOp>(loc, dbAlloc.getPtr());
}

/// Rewrite uses in parent EDT based on DB allocation granularity.
void CreateDbsPass::rewriteUsesInParentEdt(MemrefInfo &memrefInfo) {
  assert(memrefInfo.dbAllocOp && "No DbAllocOp found");

  auto dbAlloc = memrefInfo.dbAllocOp;
  Type elementMemRefType = dbAlloc.getAllocatedElementType();

  /// Collect users in the parent edt
  SmallVector<Operation *, 8> users;
  unsigned totalUses = 0;
  unsigned skippedUses = 0;
  for (auto &use : memrefInfo.alloc->getUses()) {
    totalUses++;
    Operation *user = use.getOwner();
    EdtOp userParentEdt = user->getParentOfType<EdtOp>();
    if (userParentEdt == memrefInfo.parentEdt) {
      users.push_back(user);
      ARTS_DEBUG("   Collected user: " << user->getName() << " at "
                                       << user->getLoc());
    } else {
      skippedUses++;
      ARTS_DEBUG("   Skipped user (different EDT): " << user->getName());
    }
  }

  ARTS_DEBUG(" - Alloc has " << totalUses << " total uses, collected "
                             << users.size() << " in parent EDT, skipped "
                             << skippedUses);

  /// Coarse-grained: db_ref[0] + load/store[indices]
  /// Fine-grained: db_ref[indices] + load/store[0]
  const DbRewritePlan &plan = memrefInfo.rewritePlan.value();
  unsigned innerRank = cast<MemRefType>(elementMemRefType).getRank();

  std::unique_ptr<DbIndexerBase> indexer;
  if (plan.usesBlockedLayout()) {
    PartitionInfo blockInfo;
    blockInfo.mode = PartitionMode::block;
    blockInfo.sizes.assign(plan.blockSizes.begin(), plan.blockSizes.end());
    blockInfo.partitionedDims.assign(plan.partitionedDims.begin(),
                                     plan.partitionedDims.end());
    SmallVector<Value> startBlocks(plan.blockSizes.size());
    for (Value &startBlock : startBlocks)
      startBlock = AC->createIndexConstant(0, dbAlloc.getLoc());
    indexer = std::make_unique<DbBlockIndexer>(blockInfo, startBlocks,
                                               plan.outerRank(), innerRank);
  } else {
    PartitionInfo info;
    info.mode = PartitionMode::coarse;
    unsigned outerRank =
        memrefInfo.usedFineGrained ? dbAlloc.getSizes().size() : 0;
    indexer =
        std::make_unique<DbElementWiseIndexer>(info, outerRank, innerRank);
  }

  for (Operation *user : users) {
    size_t sizeBefore = opsToRemove.size();
    indexer->transformOps({user}, dbAlloc.getPtr(), elementMemRefType, *AC,
                          opsToRemove);
    if (opsToRemove.size() > sizeBefore) {
      ARTS_DEBUG("   Transformed: " << user->getName()
                                    << " -> added to remove");
    } else {
      ARTS_DEBUG("   NOT transformed: " << user->getName() << " at "
                                        << user->getLoc());
    }
  }
}

/// Rewrite uses of a DB allocation in the parent region.
/// This is used when the allocation is not inside an EDT, but is still shared
/// with EDTs and the host needs to see the updated data.
void CreateDbsPass::rewriteUsesEverywhereWithPlan(Operation *alloc,
                                                  DbAllocOp dbAlloc,
                                                  const DbRewritePlan &plan) {
  Value originalValue = alloc->getResult(0);
  if (!originalValue)
    return;

  if (plan.usesBlockedLayout()) {
    for (auto &use : llvm::make_early_inc_range(originalValue.getUses()))
      if (isa<memref::DeallocOp>(use.getOwner()))
        opsToRemove.insert(use.getOwner());

    PartitionInfo blockInfo;
    blockInfo.mode = PartitionMode::block;
    blockInfo.sizes.assign(plan.blockSizes.begin(), plan.blockSizes.end());
    blockInfo.partitionedDims.assign(plan.partitionedDims.begin(),
                                     plan.partitionedDims.end());
    SmallVector<Value> startBlocks(plan.blockSizes.size());
    for (Value &startBlock : startBlocks)
      startBlock = AC->createIndexConstant(0, dbAlloc.getLoc());
    DbBlockIndexer indexer(blockInfo, startBlocks, plan.outerRank(),
                           plan.innerRank());
    indexer.transformUsesInParentRegion(alloc, dbAlloc, *AC, opsToRemove);
    return;
  }

  auto getAnchorInDbAllocBlock = [&](Operation *user) -> Operation * {
    if (!user)
      return nullptr;
    Block *dbAllocBlock = dbAlloc->getBlock();
    Operation *anchor = user;
    while (anchor && anchor->getBlock() != dbAllocBlock)
      anchor = anchor->getParentOp();
    return anchor;
  };

  auto shouldRewrite = [&](Operation *user) {
    if (!user || user == dbAlloc.getOperation() ||
        user->getParentOfType<EdtOp>())
      return false;

    Operation *anchor = getAnchorInDbAllocBlock(user);
    if (!anchor)
      return false;
    if (anchor->getBlock() == dbAlloc->getBlock() &&
        anchor->isBeforeInBlock(dbAlloc.getOperation()))
      return false;
    return true;
  };

  bool hasHostUses = false;
  for (auto &use : originalValue.getUses()) {
    if (shouldRewrite(use.getOwner())) {
      hasHostUses = true;
      break;
    }
  }
  if (!hasHostUses)
    return;

  auto materializeMemrefAsType = [&](Value value, Type targetType,
                                     Operation *insertBefore) -> Value {
    if (!value || value.getType() == targetType)
      return value;
    if (!isa<MemRefType>(value.getType()) || !isa<MemRefType>(targetType))
      return Value();

    auto srcType = cast<MemRefType>(value.getType());
    auto dstType = cast<MemRefType>(targetType);

    // Handle rank-1 -> rank-0 for scalar DBs (memref<?xT> -> memref<T>):
    // use subview to extract a rank-reducing 0-d slice at index 0.
    if (srcType.getRank() != dstType.getRank()) {
      if (srcType.getRank() == 1 && dstType.getRank() == 0 &&
          srcType.getElementType() == dstType.getElementType()) {
        OpBuilder::InsertionGuard guard(AC->getBuilder());
        AC->setInsertionPoint(insertBefore);
        OpFoldResult zero = AC->getBuilder().getIndexAttr(0);
        OpFoldResult one = AC->getBuilder().getIndexAttr(1);
        return AC->create<memref::SubViewOp>(
            insertBefore->getLoc(), dstType, value,
            /*offsets=*/SmallVector<OpFoldResult>{zero},
            /*sizes=*/SmallVector<OpFoldResult>{one},
            /*strides=*/SmallVector<OpFoldResult>{one});
      }
      return Value();
    }

    if (!memref::CastOp::areCastCompatible(srcType, dstType))
      return Value();

    OpBuilder::InsertionGuard guard(AC->getBuilder());
    AC->setInsertionPoint(insertBefore);
    return AC->create<memref::CastOp>(insertBefore->getLoc(), dstType, value);
  };

  auto materializeCoarseView = [&](Type targetType,
                                   Operation *insertBefore) -> Value {
    OpBuilder::InsertionGuard guard(AC->getBuilder());
    AC->setInsertionPoint(insertBefore);
    Value zero = AC->createIndexConstant(0, insertBefore->getLoc());
    Value view = AC->create<DbRefOp>(insertBefore->getLoc(),
                                     dbAlloc.getAllocatedElementType(),
                                     dbAlloc.getPtr(), ValueRange{zero});
    return materializeMemrefAsType(view, targetType, insertBefore);
  };

  std::function<void(Value, Value)> rewriteForwardedUses =
      [&](Value oldValue, Value replacementValue) {
        SmallVector<OpOperand *> usesToRewrite;
        for (OpOperand &use : llvm::make_early_inc_range(oldValue.getUses())) {
          if (shouldRewrite(use.getOwner()))
            usesToRewrite.push_back(&use);
        }

        for (OpOperand *use : usesToRewrite) {
          Operation *user = use->getOwner();

          if (isa<memref::DeallocOp>(user)) {
            opsToRemove.insert(user);
            continue;
          }

          Value baseReplacement = replacementValue;
          if (!baseReplacement)
            baseReplacement = materializeCoarseView(oldValue.getType(), user);
          if (!baseReplacement)
            continue;

          if (isForwardingMemrefAliasOp(user, oldValue)) {
            Value mappedSource = materializeMemrefAsType(
                baseReplacement, oldValue.getType(), user);
            if (!mappedSource)
              continue;

            IRMapping mapper;
            mapper.map(oldValue, mappedSource);

            OpBuilder::InsertionGuard guard(AC->getBuilder());
            AC->setInsertionPoint(user);
            Operation *cloned = AC->clone(*user, mapper);

            for (auto [oldResult, newResult] :
                 llvm::zip(user->getResults(), cloned->getResults()))
              rewriteForwardedUses(oldResult, newResult);

            if (user->use_empty())
              opsToRemove.insert(user);
            continue;
          }

          Value typedReplacement = materializeMemrefAsType(
              baseReplacement, oldValue.getType(), user);
          if (!typedReplacement)
            continue;
          use->set(typedReplacement);
        }
      };

  rewriteForwardedUses(originalValue, Value());
}

/// Rewrite operations to use DbAcquire
void CreateDbsPass::rewriteOpsToUseDbAcquire(
    EdtOp edt, SmallVector<Operation *> &operations, Operation *dbOp,
    SmallVector<Value> &acquireIndices, SmallVector<Value> &acquireOffsets,
    Value localAcquireView, const DbRewritePlan &plan) {
  ARTS_DEBUG(" - Rewriting " << operations.size() << " operations in EDT");
  numOpsRewrittenToDbViews += operations.size();

  /// Get the element type from the acquired datablock
  /// For fine-grained: memref<?xmemref<?xT>> -> extract memref<?xT>
  Type elementMemRefType = localAcquireView.getType();
  if (auto outerMemrefType = dyn_cast<MemRefType>(elementMemRefType)) {
    if (auto innerMemrefType =
            dyn_cast<MemRefType>(outerMemrefType.getElementType())) {
      elementMemRefType = innerMemrefType;
    }
  }
  ARTS_DEBUG(" - Element memref type: " << elementMemRefType);

  /// Use stored ranks from plan, with fallback for innerRank
  unsigned innerRank = plan.innerRank();
  if (innerRank == 0)
    if (auto memrefType = dyn_cast<MemRefType>(elementMemRefType))
      innerRank = memrefType.getRank();

  /// Build value mapping from parent EDT block args/acquires to this EDT's args
  IRMapping scopeMapping;
  Block &edtBlock = edt.getBody().front();
  auto deps = edt.getDependenciesAsVector();

  for (auto [idx, blockArg] : llvm::enumerate(edtBlock.getArguments())) {
    if (idx < deps.size()) {
      Value externalDep = deps[idx];
      if (auto parentArg = dyn_cast<BlockArgument>(externalDep)) {
        scopeMapping.map(parentArg, blockArg);
        ARTS_DEBUG("   - Mapping parent block arg to local");
      }
      if (auto acquireOp = externalDep.getDefiningOp<DbAcquireOp>()) {
        scopeMapping.map(acquireOp.getPtr(), blockArg);
        if (Value srcPtr = acquireOp.getSourcePtr())
          scopeMapping.map(srcPtr, blockArg);
        ARTS_DEBUG("   - Mapping acquire result to local block arg");
      }
    }
  }

  /// Rewrite each tracked operation with DbRefOp pattern
  for (Operation *op : operations) {
    ARTS_DEBUG(" - Rewriting operation: " << *op);
    OpBuilder::InsertionGuard ig(AC->getBuilder());
    AC->setInsertionPoint(op);

    /// Apply scope mapping to operands before rewriting
    for (OpOperand &operand : op->getOpOperands()) {
      Value mappedVal = scopeMapping.lookupOrDefault(operand.get());
      if (mappedVal != operand.get()) {
        operand.set(mappedVal);
        ARTS_DEBUG("   - Remapped operand from outer scope to inner scope");
      }
    }

    if (plan.isCoarse()) {
      /// Coarse mode: db_ref index must be constant zero.
      PartitionInfo coarseInfo;
      coarseInfo.mode = PartitionMode::coarse;
      DbElementWiseIndexer indexer(coarseInfo, /*outerRank=*/0, innerRank, {});
      llvm::SetVector<Operation *> localOpsToRemove;
      indexer.transformOps({op}, localAcquireView, elementMemRefType, *AC,
                           localOpsToRemove);
      for (Operation *toRemove : localOpsToRemove)
        opsToRemove.insert(toRemove);
      continue;
    }

    if (plan.usesBlockedLayout()) {
      /// Block mode: Use DbBlockIndexer with stored block size
      ARTS_DEBUG(" - Using DbBlockIndexer with stored plan");
      Location loc = op->getLoc();
      unsigned chunkOuterRank = plan.outerRank();
      if (chunkOuterRank == 0)
        if (auto acquireOp = dyn_cast<DbAcquireOp>(dbOp))
          chunkOuterRank = acquireOp.getSizes().size();

      SmallVector<Value> blockSizes;
      SmallVector<Value> startBlocks;
      blockSizes.reserve(plan.blockSizes.size());
      startBlocks.reserve(plan.blockSizes.size());

      Value zero = AC->createIndexConstant(0, loc);
      for (auto [dimIdx, blockSize] : llvm::enumerate(plan.blockSizes)) {
        if (!blockSize)
          continue;
        blockSizes.push_back(blockSize);
        Value startOffset =
            dimIdx < acquireOffsets.size() ? acquireOffsets[dimIdx] : zero;
        startBlocks.push_back(
            AC->create<arith::DivUIOp>(loc, startOffset, blockSize));
      }

      if (blockSizes.empty()) {
        Value fallbackBlockSize = plan.getBlockSize(0);
        if (fallbackBlockSize) {
          blockSizes.push_back(fallbackBlockSize);
          Value startOffset = acquireOffsets.empty() ? zero : acquireOffsets[0];
          startBlocks.push_back(
              AC->create<arith::DivUIOp>(loc, startOffset, fallbackBlockSize));
        }
      }

      /// Build PartitionInfo from stored N-D block plan.
      PartitionInfo blockInfo;
      blockInfo.mode = PartitionMode::block;
      blockInfo.sizes.assign(blockSizes.begin(), blockSizes.end());
      blockInfo.partitionedDims.assign(plan.partitionedDims.begin(),
                                       plan.partitionedDims.end());
      DbBlockIndexer indexer(blockInfo, startBlocks, chunkOuterRank, innerRank);
      llvm::SetVector<Operation *> localOpsToRemove;
      indexer.transformOps({op}, localAcquireView, elementMemRefType, *AC,
                           localOpsToRemove);
      for (Operation *toRemove : localOpsToRemove)
        opsToRemove.insert(toRemove);
      continue;
    }

    /// ElementWise mode
    Value elemOffset = AC->createIndexConstant(0, op->getLoc());

    /// Build PartitionInfo from elemOffset
    PartitionInfo ewInfo;
    ewInfo.mode = PartitionMode::fine_grained;
    ewInfo.indices.push_back(elemOffset);
    DbElementWiseIndexer indexer(ewInfo, plan.outerRank(), innerRank, {});

    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      auto localized = indexer.localizeForFineGrained(
          load.getIndices(), acquireIndices, acquireOffsets, AC->getBuilder(),
          op->getLoc());
      auto dbRef =
          AC->create<DbRefOp>(op->getLoc(), elementMemRefType, localAcquireView,
                              localized.dbRefIndices);
      auto newLoad = AC->create<memref::LoadOp>(op->getLoc(), dbRef,
                                                localized.memrefIndices);
      load.replaceAllUsesWith(newLoad.getResult());
      opsToRemove.insert(op);
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      auto localized = indexer.localizeForFineGrained(
          store.getIndices(), acquireIndices, acquireOffsets, AC->getBuilder(),
          op->getLoc());
      auto dbRef =
          AC->create<DbRefOp>(op->getLoc(), elementMemRefType, localAcquireView,
                              localized.dbRefIndices);
      AC->create<memref::StoreOp>(op->getLoc(), store.getValue(), dbRef,
                                  localized.memrefIndices);
      opsToRemove.insert(op);
    } else {
      llvm::SetVector<Operation *> localOpsToRemove;
      indexer.transformOps({op}, localAcquireView, elementMemRefType, *AC,
                           localOpsToRemove);
      for (Operation *toRemove : localOpsToRemove)
        opsToRemove.insert(toRemove);
    }
  }
}

///===----------------------------------------------------------------------===///
/// Pass creation
///===----------------------------------------------------------------------===///
namespace mlir {
namespace arts {
std::unique_ptr<Pass> createCreateDbsPass(mlir::arts::AnalysisManager *AM) {
  return std::make_unique<CreateDbsPass>(AM);
}
} // namespace arts
} // namespace mlir
