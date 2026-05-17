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
/// Responsibility split:
/// - SDE chooses structured patterns, task grain, owner dimensions, physical
///   block shapes, and task element slices.
/// - The target SDE/CODIR path emits explicit MU storage and codelet
///   deps/params that materialize to DB/EDT ops before this raw bridge.
/// - CreateDbs materializes only remaining coarse raw EDT memref captures into
///   ARTS DB ops. Tiled/block-local access rewriting is an SDE MU/token
///   responsibility and must not be rediscovered here.
///
/// CreateDbs must not rediscover tensor/linalg partition policy.  Its
/// remaining raw bridge is intentionally limited to a whole-storage DB ref
/// (`db_ref[0]`) plus the original memref access. Structured/tiled layouts
/// must lower through canonical MU/token/codelet form before ARTS.
///==========================================================================///

#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Transforms/db/DbLayoutPlanUtils.h"
#include "carts/utils/ValueAnalysis.h"
#define GEN_PASS_DEF_CREATEDBS
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/LoweringContractUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/RemovalUtils.h"
#include "carts/utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
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
#include <utility>

/// Debug
#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(create_dbs);

using namespace mlir::carts;
using namespace mlir::carts::arts;

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

template <typename OpTy, typename... Args>
static OpTy createOp(OpBuilder &builder, Location loc, Args &&...args) {
  return OpTy::create(builder, loc, std::forward<Args>(args)...);
}

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

static Value materializeMemrefAsType(Value value, Type targetType,
                                     Operation *insertBefore,
                                     OpBuilder &builder) {
  if (!value || value.getType() == targetType)
    return value;
  if (!isa<MemRefType>(value.getType()) || !isa<MemRefType>(targetType))
    return Value();

  auto srcType = cast<MemRefType>(value.getType());
  auto dstType = cast<MemRefType>(targetType);

  if (srcType.getRank() != dstType.getRank()) {
    if (srcType.getRank() == 1 && dstType.getRank() == 0 &&
        srcType.getElementType() == dstType.getElementType()) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(insertBefore);
      OpFoldResult zero = builder.getIndexAttr(0);
      OpFoldResult one = builder.getIndexAttr(1);
      return createOp<memref::SubViewOp>(
          builder, insertBefore->getLoc(), dstType, value,
          /*offsets=*/SmallVector<OpFoldResult>{zero},
          /*sizes=*/SmallVector<OpFoldResult>{one},
          /*strides=*/SmallVector<OpFoldResult>{one});
    }
    return Value();
  }

  if (!memref::CastOp::areCastCompatible(srcType, dstType))
    return Value();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertBefore);
  return createOp<memref::CastOp>(builder, insertBefore->getLoc(), dstType,
                                  value);
}

static Value createCoarseDbRef(Value dbPtr, Operation *insertBefore,
                               OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertBefore);
  Value zero =
      createOp<arith::ConstantIndexOp>(builder, insertBefore->getLoc(), 0);
  SmallVector<Value, 1> indices{zero};
  return createOp<DbRefOp>(builder, insertBefore->getLoc(), dbPtr,
                           ArrayRef<Value>(indices))
      .getResult();
}

static LogicalResult rewriteCoarseRawAccess(Operation *op, Value expectedRoot,
                                            Value dbPtr, OpBuilder &builder) {
  if (isa<polygeist::SubIndexOp>(op)) {
    return op->emitError(
        "raw memref subindex reached ARTS DB materialization; SDE must "
        "rewrite tiled or sliced MU/token views before ARTS conversion");
  }

  auto access = DbUtils::getMemoryAccessInfo(op);
  if (!access) {
    if (isa<memref::CopyOp>(op)) {
      return op->emitError(
          "raw memref copy reached ARTS DB materialization; SDE/CODIR must "
          "materialize explicit MU/token copies before ARTS conversion");
    }
    return success();
  }

  if (expectedRoot && access->memref != expectedRoot) {
    return op->emitError(
        "raw memref view/alias access reached ARTS DB materialization; "
        "SDE must rewrite accesses to token-local memref views before ARTS "
        "conversion");
  }

  Value dbView = createCoarseDbRef(dbPtr, op, builder);
  Value typedView =
      materializeMemrefAsType(dbView, access->memref.getType(), op, builder);
  if (!typedView) {
    return op->emitError(
        "cannot type the coarse DB view for raw memref access; SDE/CODIR "
        "must materialize an explicit token-local view");
  }

  bool replaced = false;
  for (OpOperand &operand : op->getOpOperands()) {
    if (operand.get() != access->memref)
      continue;
    operand.set(typedView);
    replaced = true;
  }

  if (!replaced)
    return op->emitError("failed to replace raw memref operand with DB view");

  return success();
}

///===----------------------------------------------------------------------===///
/// Pass Implementation
///===----------------------------------------------------------------------===///

struct CreateDbsPass : public impl::CreateDbsBase<CreateDbsPass> {
  CreateDbsPass(mlir::carts::arts::AnalysisManager *AM) : AM(AM) {
    assert(AM && "AnalysisManager must be provided externally");
  }

  void runOnOperation() override;

private:
  ModuleOp module;
  mlir::carts::arts::AnalysisManager *AM = nullptr;
  OpBuilder *builder = nullptr;
  SetVector<Operation *> opsToRemove;

  /// Structure to track all usage information for a memref
  struct MemrefInfo {
    Operation *alloc = nullptr;
    EdtOp parentEdt = nullptr;
    /// True if this allocation is used by an EDT other than its defining EDT.
    bool usedByOtherEdts = false;
    ArtsMode accessMode = ArtsMode::uninitialized;
    DbAllocOp dbAllocOp = nullptr;
  };

  DenseMap<Operation *, MemrefInfo> memrefInfo;
  DenseMap<EdtOp, SetVector<Value>> edtExternalValues;

  void collectMemrefs();
  void reconcileExternalDepAccessModes();
  void ensureInitializedAccessModes();
  void createDbAllocOps();
  void lowerEdtExternalDependencies();
  void cleanupAndFinalize();
  void projectSemanticContractToDbValue(Operation *sourceOp,
                                        Operation *targetOp,
                                        Value contractTarget);
  void createDbAcquireOps(EdtOp edt, SetVector<Value> &externalDeps);
  Value findParentAcquireSource(EdtOp edt, Operation *dbAllocOp,
                                ArtsMode requestedMode);
  void initializeGlobalDbIfNeeded(Operation *alloc, DbAllocOp dbAllocOp,
                                  DbAllocType allocType);

  DbAllocType inferAllocType(Operation *alloc);
  void insertDbFreeForDbAlloc(DbAllocOp dbAlloc, Operation *alloc);
  void rewriteOpsToUseDbAcquire(EdtOp edt, SmallVector<Operation *> &operations,
                                Operation *rawAlloc,
                                Value localAcquireView);
  void rewriteUsesInParentEdt(MemrefInfo &memrefInfo);
  Operation *findPhysicalLayoutPlanSource(Operation *alloc);
  void rewriteUsesEverywhere(Operation *alloc, DbAllocOp dbAlloc);
};
} // namespace

///===----------------------------------------------------------------------===///
/// Pass Entry Point
///===----------------------------------------------------------------------===///
void CreateDbsPass::runOnOperation() {
  module = getOperation();
  opsToRemove.clear();
  memrefInfo.clear();

  OpBuilder ownedBuilder(module.getContext());
  builder = &ownedBuilder;

  ARTS_INFO_HEADER(CreateDbsPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// Phase 1: Collect memrefs used in EDTs
  ARTS_INFO("Phase 1: Collecting memrefs used in EDTs");
  collectMemrefs();
  ARTS_DEBUG(" - Found " << memrefInfo.size() << " memrefs used in EDTs");

  /// Phase 2: Reconcile raw external memref access modes.
  reconcileExternalDepAccessModes();
  ensureInitializedAccessModes();

  /// Phase 3: Create DbAlloc operations.
  ARTS_INFO("Phase 3: Creating DbAlloc operations for " << memrefInfo.size()
                                                        << " memrefs");
  createDbAllocOps();

  /// Phase 4: Process EDTs for dependencies.
  lowerEdtExternalDependencies();

  /// Phase 5-7: Cleanup and validation post-checks.
  cleanupAndFinalize();

  builder = nullptr;
  ARTS_INFO_FOOTER(CreateDbsPass);
  ARTS_DEBUG_REGION(module.dump(););
}

void CreateDbsPass::reconcileExternalDepAccessModes() {
  /// For each EDT with external dependencies, infer conservative access modes
  /// from concrete memory uses when no explicit SDE materialized acquire is
  /// already present.
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
      ArtsMode inferredMode =
          AM->getDbAnalysis().inferEdtAccessMode(underlyingOp, edt);
      if (inferredMode == ArtsMode::uninitialized)
        inferredMode = ArtsMode::inout;

      ARTS_DEBUG(" - Memref "
                 << *underlyingOp
                 << " used in EDT without explicit SDE acquire metadata, "
                    "inferred mode="
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
               << " has no explicit access metadata, defaulting to inout "
                  "mode");
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
}

void CreateDbsPass::projectSemanticContractToDbValue(Operation *sourceOp,
                                                     Operation *targetOp,
                                                     Value contractTarget) {
  if (!sourceOp || !targetOp || !contractTarget)
    return;

  transferOperationContract(sourceOp, targetOp);

  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointAfter(targetOp);
  transferLoweringContract(sourceOp, contractTarget, *builder,
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
    if (!ownerDims || ownerDims->empty() || !blockShape || blockShape->empty())
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
  module.walk([&](Operation *candidate) {
    if (!candidate || candidate == alloc)
      return WalkResult::advance();
    if (!hasPhysicalDbLayoutPlan(candidate))
      return WalkResult::advance();
    if (!compatiblePlanTarget(candidate))
      return WalkResult::advance();
    if (!writesAlloc(candidate))
      return WalkResult::advance();
    candidates.push_back(candidate);
    return WalkResult::advance();
  });

  if (candidates.empty())
    return nullptr;

  Operation *selected = candidates.front();
  for (Operation *candidate : ArrayRef<Operation *>(candidates).drop_front()) {
    if (equivalentPlan(selected, candidate))
      continue;
    InFlightDiagnostic diag = alloc->emitError(
        "conflicting SDE-authored physical DB layout plans for "
        "one allocation");
    diag.attachNote(selected->getLoc()) << "first layout plan source";
    diag.attachNote(candidate->getLoc()) << "conflicting layout plan source";
    signalPassFailure();
    return nullptr;
  }
  return selected;
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

/// Create DB Allocation Operations
void CreateDbsPass::createDbAllocOps() {
  /// All memrefs in memrefInfo are converted to DBs
  for (auto &entry : memrefInfo) {
    Operation *alloc = entry.first;
    ARTS_DEBUG("Creating DB Alloc Op for memref " << *alloc);
    MemrefInfo &info = entry.second;
    Location loc = alloc->getLoc();
    OpBuilder::InsertionGuard IG(*builder);
    builder->setInsertionPointAfter(alloc);

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
    /// double-indirection patterns (e.g. float**) that SdeMemrefNormalization
    /// should have raised to N-dimensional memrefs.  If they reach here,
    /// the pipeline has a gap that will produce broken executables.
    if (auto memRefType = dyn_cast<MemRefType>(alloc->getResult(0).getType())) {
      if (isa<MemRefType>(memRefType.getElementType())) {
        alloc->emitError("un-normalizable nested memref pattern (element type "
                         "is memref); SdeMemrefNormalization should have raised "
                         "this to an N-dimensional memref");
        signalPassFailure();
        return;
      }
    }

    /// Infer allocation type based on the defining operation
    DbAllocType allocType = inferAllocType(alloc);

    /// Use the combined dependency/access mode inferred for this memref.
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

    /// Determine allocation granularity using unified heuristics
    SmallVector<Value> sizes, logicalElementSizes, elementSizes;
    const bool isRankZero = memRefType.getRank() == 0;
    const unsigned rank = std::max<unsigned>(1, memRefType.getRank());

    /// Build the original logical allocation extents. These are the source
    /// extents for any SDE-authored physical layout plan.
    logicalElementSizes.reserve(rank);
    for (unsigned i = 0; i < rank; ++i) {
      if (!isRankZero && memRefType.isDynamicDim(i)) {
        logicalElementSizes.push_back(
            createOp<arts::DbDimOp>(*builder, loc, allocValue, (int64_t)i));
      } else {
        int64_t dimSize =
            isRankZero ? 1 : static_cast<int64_t>(memRefType.getDimSize(i));
        logicalElementSizes.push_back(
            createOp<arith::ConstantIndexOp>(*builder, loc, dimSize));
      }
    }

    /// Coarse is the only raw-memref bridge left in ARTS. If SDE authored a
    /// physical block layout, the corresponding MU/token/codelet materializer
    /// must have already rewritten the accesses before this pass.
    sizes.push_back(createOp<arith::ConstantIndexOp>(*builder, loc, 1));
    elementSizes.assign(logicalElementSizes.begin(), logicalElementSizes.end());

    if (Operation *planSource = findPhysicalLayoutPlanSource(alloc)) {
      InFlightDiagnostic diag = alloc->emitError(
          "SDE-authored physical DB layout reached CreateDbs as a raw "
          "memref; SDE must materialize MU/token/codelet storage and "
          "token-local access rewrites before ARTS conversion");
      diag.attachNote(planSource->getLoc()) << "layout plan source";
      signalPassFailure();
      return;
    }

    ARTS_DEBUG(" - Using coarse-grained raw DB bridge allocation");

    /// Create the db_alloc operation
    /// DBs without an explicit route stay on the creating node. Lowering
    /// materializes this sentinel into the runtime hint instead of pinning the
    /// allocation to rank 0.
    auto route = createCurrentNodeRoute(*builder, loc);
    auto dbAllocOp = createOp<DbAllocOp>(
        *builder,
        loc, mode, route, allocType, dbMode, elementType, sizes, elementSizes,
        PartitionMode::coarse);
    ++numDbAllocsCreated;

    projectSemanticContractToDbValue(alloc, dbAllocOp.getOperation(),
                                     dbAllocOp.getPtr());

    /// Initialize global DBs
    initializeGlobalDbIfNeeded(alloc, dbAllocOp, allocType);

    /// Copy ARTS ID from original allocation to DbAllocOp
    copyArtsMetadataAttrs(alloc, dbAllocOp.getOperation());

    /// Record allocation strategy decision for diagnostics
    AM->getDbHeuristics().recordDecision(
        "AllocationStrategy", true,
        "Coarse raw-memref bridge allocation",
        alloc,
        {{"outerRank", static_cast<int64_t>(sizes.size())},
         {"innerRank", static_cast<int64_t>(elementSizes.size())}});

    /// Store mappings for later use
    info.dbAllocOp = dbAllocOp;

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
      rewriteUsesEverywhere(alloc, dbAllocOp);

      /// EDT uses are rewritten only after a matching db_acquire has been
      /// materialized in createDbAcquireOps. Rewriting them here would capture
      /// the allocation DB pointer directly inside the task body.
    }
  }
}

/// Initialize global constants for DB-backed globals.
void CreateDbsPass::initializeGlobalDbIfNeeded(Operation *alloc,
                                               DbAllocOp dbAllocOp,
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
  OpBuilder::InsertionGuard initGuard(*builder);
  builder->setInsertionPointAfter(dbAllocOp);
  Value globalMemref = createOp<memref::GetGlobalOp>(
      *builder, loc, globalOp.getType(), getGlobal.getNameAttr());

  SmallVector<Value> zeroIndices;
  zeroIndices.reserve(dbAllocOp.getSizes().size());
  for (size_t i = 0; i < dbAllocOp.getSizes().size(); ++i)
    zeroIndices.push_back(createOp<arith::ConstantIndexOp>(*builder, loc, 0));

  auto memRefType = cast<MemRefType>(globalOp.getType());
  unsigned rank = memRefType.getRank();
  if (rank == 0) {
    Value dbRef =
        createOp<DbRefOp>(*builder, loc, dbAllocOp.getPtr(), zeroIndices);
    Value initVal = createOp<memref::LoadOp>(*builder, loc, globalMemref);
    createOp<memref::StoreOp>(*builder, loc, initVal, dbRef);
    return;
  }

  auto storeToDb = [&](Value initVal, ArrayRef<Value> indices) {
    Value dbRef =
        createOp<DbRefOp>(*builder, loc, dbAllocOp.getPtr(), zeroIndices);
    createOp<memref::StoreOp>(*builder, loc, initVal, dbRef, indices);
  };

  SmallVector<Value> indices;
  indices.reserve(rank);
  auto emitLoopCopy = [&](auto &self, unsigned dim) -> void {
    if (dim == rank) {
      Value initVal =
          createOp<memref::LoadOp>(*builder, loc, globalMemref, indices);
      storeToDb(initVal, indices);
      return;
    }

    Value lower = createOp<arith::ConstantIndexOp>(*builder, loc, 0);
    Value upper;
    if (memRefType.isDynamicDim(dim)) {
      upper = createOp<memref::DimOp>(*builder, loc, globalMemref, dim);
    } else {
      upper = createOp<arith::ConstantIndexOp>(*builder, loc,
                                               memRefType.getDimSize(dim));
    }
    Value step = createOp<arith::ConstantIndexOp>(*builder, loc, 1);

    auto loop = createOp<scf::ForOp>(*builder, loc, lower, upper, step);
    OpBuilder::InsertionGuard loopGuard(*builder);
    builder->setInsertionPointToStart(loop.getBody());
    indices.push_back(loop.getInductionVar());
    self(self, dim + 1);
    indices.pop_back();
  };
  emitLoopCopy(emitLoopCopy, 0);
}

Value CreateDbsPass::findParentAcquireSource(EdtOp edt, Operation *dbAllocOp,
                                             ArtsMode requestedMode) {
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

  /// Accumulate dependency operands to set on the EDT. CODIR-to-ARTS lowering
  /// may have already wired DB-backed dependencies before this pass; preserve
  /// those while appending dependencies that this raw bridge materializes from
  /// ordinary memrefs.
  SmallVector<Value> dependencyOperands(edt.getDependencies().begin(),
                                        edt.getDependencies().end());

  /// One raw bridge acquire per allocation per EDT. Canonical SDE tokens are
  /// consumed by the CODIR boundary and do not reach this path.
  DenseSet<Operation *> processedAllocs;

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

    if (!processedAllocs.insert(underlyingOp).second) {
      ARTS_DEBUG("   - Skipping duplicate external dependency for allocation");
      continue;
    }

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
    OpBuilder::InsertionGuard IG(*builder);
    builder->setInsertionPoint(edt);

    /// Raw bridge acquires cover the whole physical DB range selected at
    /// allocation time. SDE-tokenized slices lower before this pass.
    /// Build full DB-space range for the already-created coarse layout.
    SmallVector<Value> dbOffsets, dbSizes;
    SmallVector<Value> allocSizes(dbAllocOp.getSizes().begin(),
                                  dbAllocOp.getSizes().end());
    if (allocSizes.empty()) {
      dbOffsets.push_back(
          createOp<arith::ConstantIndexOp>(*builder, edt.getLoc(), 0));
      dbSizes.push_back(
          createOp<arith::ConstantIndexOp>(*builder, edt.getLoc(), 1));
    } else {
      for (Value s : allocSizes) {
        dbOffsets.push_back(
            createOp<arith::ConstantIndexOp>(*builder, edt.getLoc(), 0));
        dbSizes.push_back(s);
      }
    }

    ArtsMode acquireMode =
        AM->getDbAnalysis().inferEdtAccessMode(underlyingOp, edt);
    if (acquireMode == ArtsMode::uninitialized) {
      acquireMode = (info.accessMode == ArtsMode::uninitialized)
                        ? ArtsMode::inout
                        : info.accessMode;
    }

    Value acqGuid = sourceGuid;
    Value acqPtr = sourcePtr;
    if (Value availableHandle =
            findParentAcquireSource(edt, dbAllocOp, acquireMode)) {
      ARTS_DEBUG("   - Reusing existing datablock handle in parent scope");
      acqGuid = Value();
      acqPtr = availableHandle;
    }

    auto createAcquire = [&](Location acquireLoc) {
      Type ptrType = acqPtr ? acqPtr.getType() : Type();
      if (DbUtils::getUnderlyingDbAlloc(acqPtr)) {
        return createOp<DbAcquireOp>(
            *builder, acquireLoc, acquireMode, acqGuid, acqPtr,
            PartitionMode::coarse,
            /*indices=*/SmallVector<Value>{}, /*offsets=*/dbOffsets,
            /*sizes=*/dbSizes,
            /*partition_indices=*/SmallVector<Value>{},
            /*partition_offsets=*/SmallVector<Value>{},
            /*partition_sizes=*/SmallVector<Value>{});
      }

      return createOp<DbAcquireOp>(
          *builder, acquireLoc, acquireMode, acqGuid, acqPtr, ptrType,
          PartitionMode::coarse,
          /*indices=*/SmallVector<Value>{}, /*offsets=*/dbOffsets,
          /*sizes=*/dbSizes,
          /*partition_indices=*/SmallVector<Value>{},
          /*partition_offsets=*/SmallVector<Value>{},
          /*partition_sizes=*/SmallVector<Value>{});
    };

    auto acquireOp = createAcquire(edt.getLoc());
    ++numDbAcquireGroupsCreated;

    projectSemanticContractToDbValue(
        edt.getOperation(), acquireOp.getOperation(), acquireOp.getPtr());

    ARTS_DEBUG(" - Created raw bridge acquire, mode="
               << acquireMode << ", partition="
               << static_cast<int>(PartitionMode::coarse) << ": "
               << acquireOp);

    Value localAcquireView = acquireOp.getPtr();
    auto sourceType = dyn_cast<MemRefType>(localAcquireView.getType());
    BlockArgument dbAcquireArg =
        edt.getBody().front().addArgument(sourceType, edt.getLoc());
    dependencyOperands.push_back(localAcquireView);
    localAcquireView = dbAcquireArg;

    SmallVector<Operation *> opsToRewrite;
    llvm::SmallPtrSet<Operation *, 16> seenOps;
    auto addOpIfNew = [&](Operation *op) {
      if (!op)
        return;
      if (seenOps.insert(op).second)
        opsToRewrite.push_back(op);
    };

    edt.walk([&](Operation *op) {
      if (op->getParentOfType<EdtOp>() != edt)
        return;
      if (!DbAnalysis::opMatchesAccessMode(op, underlyingOp, acquireMode))
        return;
      addOpIfNew(op);
    });
    ARTS_DEBUG(" - Found " << opsToRewrite.size()
                           << " raw memref operation(s) to rewrite");

    if (!opsToRewrite.empty()) {
      rewriteOpsToUseDbAcquire(edt, opsToRewrite, underlyingOp,
                               localAcquireView);
    }

    OpBuilder::InsertionGuard releaseGuard(*builder);
    builder->setInsertionPoint(edt.getBody().front().getTerminator());
    createOp<DbReleaseOp>(*builder, edt.getLoc(), localAcquireView);
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
  OpBuilder::InsertionGuard IG(*builder);
  builder->setInsertionPoint(insertionPoint);
  createOp<DbFreeOp>(*builder, loc, dbAlloc.getGuid());
  createOp<DbFreeOp>(*builder, loc, dbAlloc.getPtr());
}

/// Rewrite uses in parent EDT based on DB allocation granularity.
void CreateDbsPass::rewriteUsesInParentEdt(MemrefInfo &memrefInfo) {
  assert(memrefInfo.dbAllocOp && "No DbAllocOp found");

  auto dbAlloc = memrefInfo.dbAllocOp;

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

  Value rootValue = memrefInfo.alloc->getResult(0);
  for (Operation *user : users) {
    if (failed(
            rewriteCoarseRawAccess(user, rootValue, dbAlloc.getPtr(), *builder))) {
      signalPassFailure();
      return;
    }
    ARTS_DEBUG("   Rewritten or validated coarse raw user: "
               << user->getName() << " at " << user->getLoc());
  }
}

/// Rewrite uses of a DB allocation in the parent region.
/// This is used when the allocation is not inside an EDT, but is still shared
/// with EDTs and the host needs to see the updated data.
void CreateDbsPass::rewriteUsesEverywhere(Operation *alloc,
                                          DbAllocOp dbAlloc) {
  Value originalValue = alloc->getResult(0);
  if (!originalValue)
    return;

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

  auto materializeCoarseView = [&](Type targetType,
                                   Operation *insertBefore) -> Value {
    Value view = createCoarseDbRef(dbAlloc.getPtr(), insertBefore, *builder);
    return materializeMemrefAsType(view, targetType, insertBefore, *builder);
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
                baseReplacement, oldValue.getType(), user, *builder);
            if (!mappedSource)
              continue;

            IRMapping mapper;
            mapper.map(oldValue, mappedSource);

            OpBuilder::InsertionGuard guard(*builder);
            builder->setInsertionPoint(user);
            Operation *cloned = builder->clone(*user, mapper);

            for (auto [oldResult, newResult] :
                 llvm::zip(user->getResults(), cloned->getResults()))
              rewriteForwardedUses(oldResult, newResult);

            if (user->use_empty())
              opsToRemove.insert(user);
            continue;
          }

          Value typedReplacement = materializeMemrefAsType(
              baseReplacement, oldValue.getType(), user, *builder);
          if (!typedReplacement)
            continue;
          use->set(typedReplacement);
        }
      };

  rewriteForwardedUses(originalValue, Value());
}

/// Rewrite operations to use DbAcquire
void CreateDbsPass::rewriteOpsToUseDbAcquire(
    EdtOp edt, SmallVector<Operation *> &operations, Operation *rawAlloc,
    Value localAcquireView) {
  ARTS_DEBUG(" - Rewriting " << operations.size() << " operations in EDT");
  numOpsRewrittenToDbViews += operations.size();

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
    OpBuilder::InsertionGuard ig(*builder);
    builder->setInsertionPoint(op);

    /// Apply scope mapping to operands before rewriting
    for (OpOperand &operand : op->getOpOperands()) {
      Value mappedVal = scopeMapping.lookupOrDefault(operand.get());
      if (mappedVal != operand.get()) {
        operand.set(mappedVal);
        ARTS_DEBUG("   - Remapped operand from outer scope to inner scope");
      }
    }

    Value rootValue = rawAlloc && rawAlloc->getNumResults() > 0
                          ? rawAlloc->getResult(0)
                          : Value();
    if (failed(
            rewriteCoarseRawAccess(op, rootValue, localAcquireView, *builder))) {
      signalPassFailure();
      return;
    }
  }
}

///===----------------------------------------------------------------------===///
/// Pass creation
///===----------------------------------------------------------------------===///
namespace mlir {
namespace carts::arts {
std::unique_ptr<Pass> createCreateDbsPass(mlir::carts::arts::AnalysisManager *AM) {
  return std::make_unique<CreateDbsPass>(AM);
}
} // namespace carts::arts
} // namespace mlir
