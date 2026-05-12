///==========================================================================///
/// File: DbElementWiseRewriter.cpp
///
/// Element-wise rewriter: each element is a separate datablock when the
/// allocation is fine-grained (outerRank > 0). Coarse layouts (outerRank == 0)
/// also use this rewriter, but degenerate to a single-DB layout where db_ref
/// indices are constant and accesses stay logically coarse.
///
/// Example transformation (N=100 elements, 4 workers):
///
///   BEFORE:
///     %db = arts.db_alloc memref<100xf64>
///     arts.db_acquire %db [0] [100]          /// worker acquires all
///     %ref = arts.db_ref %db[%i]
///     memref.load %ref[%j]                   /// global index j
///
///   AFTER:
///     %db = arts.db_alloc memref<100x1xf64>  /// outer=100, inner=1
///     arts.db_acquire %db [%offset] [%size]  /// worker 0: [0][25]
///     %ref = arts.db_ref %db[%i]
///     memref.load %ref[0]                    /// local index (always 0)
///
/// Index localization: local = global - elemOffset
///==========================================================================///

#include "arts/dialect/core/Transforms/db/elementwise/DbElementWiseRewriter.h"
#include "arts/dialect/core/Transforms/db/elementwise/DbElementWiseIndexer.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/Debug.h"
#include "arts/utils/EdtUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/RemovalUtils.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/Utils.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

ARTS_DEBUG_SETUP(db_transforms);

using namespace mlir;
using namespace mlir::arts;

void DbElementWiseRewriter::transformAcquire(const DbRewriteAcquire &info,
                                             DbAllocOp newAlloc,
                                             OpBuilder &builder) {
  DbAcquireOp acquire = info.acquire;

  ARTS_DEBUG("DbElementWiseRewriter::transformAcquire (dims="
             << info.partitionInfo.indices.size() << ")");

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(acquire);

  /// Match block/stencil rewrite semantics: acquires already nested in EDT
  /// regions must keep their in-scope source chain for dependency validation.
  bool inEdtRegion = static_cast<bool>(acquire->getParentOfType<EdtOp>());
  if (!inEdtRegion) {
    acquire.getSourceGuidMutable().assign(newAlloc.getGuid());
    acquire.getSourcePtrMutable().assign(newAlloc.getPtr());
  }

  /// Update acquire's ptr result type to match new source
  MemRefType newPtrType = cast<MemRefType>(acquire.getSourcePtr().getType());
  Type oldAcqPtrType = acquire.getPtr().getType();
  if (oldAcqPtrType != newPtrType) {
    acquire.getPtr().setType(newPtrType);

    /// Also update EDT block argument type if this acquire feeds an EDT
    auto [edt, blockArg] = EdtUtils::getBlockArgumentForAcquire(acquire);
    if (blockArg && blockArg.getType() != newPtrType)
      blockArg.setType(newPtrType);
  }

  /// ELEMENT-WISE: Update indices to the element offsets for fine_grained mode.
  /// Supports multi-dimensional fine-grained (e.g., A[i][j] -> indices [i, j]).
  /// Keep offsets at 0 and sizes at 1 to avoid double-indexing during lowering.
  Location loc = acquire.getLoc();
  Value zero = arts::createZeroIndex(builder, loc);
  Value one = arts::createOneIndex(builder, loc);

  bool hasRange =
      !info.partitionInfo.offsets.empty() && !info.partitionInfo.sizes.empty();

  if (hasRange) {
    ARTS_DEBUG("  Fine-grained range acquire: offsets="
               << info.partitionInfo.offsets.size()
               << ", sizes=" << info.partitionInfo.sizes.size());
    if (!info.partitionInfo.offsets.empty() &&
        !info.partitionInfo.sizes.empty())
      ARTS_DEBUG("    offset0=" << info.partitionInfo.offsets.front()
                                << ", size0="
                                << info.partitionInfo.sizes.front());
    auto refineRangeSizeFromMin = [&](Value offset, Value sizeHint) -> Value {
      if (!offset || !sizeHint)
        return Value();
      int64_t sizeConst = 0;
      bool sizeIsConst = ValueAnalysis::getConstantIndex(sizeHint, sizeConst);
      auto isSameConst = [&](Value v) -> bool {
        int64_t val = 0;
        return ValueAnalysis::getConstantIndex(v, val) && val == sizeConst;
      };
      for (Operation &op : *acquire->getBlock()) {
        Value refined;
        auto tryRefine = [&](Value lhs, Value rhs, Value result) {
          bool lhsIsHint =
              (lhs == sizeHint) || (sizeIsConst && isSameConst(lhs));
          bool rhsIsHint =
              (rhs == sizeHint) || (sizeIsConst && isSameConst(rhs));
          if (lhsIsHint && ValueAnalysis::dependsOn(rhs, offset))
            refined = result;
          else if (rhsIsHint && ValueAnalysis::dependsOn(lhs, offset))
            refined = result;
        };

        if (auto minOp = dyn_cast<arith::MinUIOp>(&op)) {
          tryRefine(minOp.getLhs(), minOp.getRhs(), minOp.getResult());
        } else if (auto minOp = dyn_cast<arith::MinSIOp>(&op)) {
          tryRefine(minOp.getLhs(), minOp.getRhs(), minOp.getResult());
        }

        if (refined)
          return refined;
      }
      return Value();
    };

    /// Range acquire: use DB-space offsets/sizes to cover multiple elements.
    acquire.getIndicesMutable().clear();
    acquire.getOffsetsMutable().assign(info.partitionInfo.offsets);
    SmallVector<Value> rangeSizes(info.partitionInfo.sizes.begin(),
                                  info.partitionInfo.sizes.end());
    if (info.partitionInfo.offsets.size() == 1 && rangeSizes.size() == 1) {
      Value offset = info.partitionInfo.offsets.front();
      Value size = rangeSizes.front();
      if (Value refined = refineRangeSizeFromMin(offset, size)) {
        rangeSizes.clear();
        rangeSizes.push_back(refined);
        ARTS_DEBUG("  Refined fine-grained range size: " << refined);
      }
    }
    acquire.getSizesMutable().assign(rangeSizes);
  } else {
    /// Single element: use element coordinates (fine-grained indices).
    SmallVector<Value> newIndices(info.partitionInfo.indices.begin(),
                                  info.partitionInfo.indices.end());
    acquire.getIndicesMutable().assign(newIndices);

    /// Set offsets to 0 and sizes to 1 for each dimension
    SmallVector<Value> newOffsets, newSizes;
    for (size_t i = 0; i < info.partitionInfo.indices.size(); ++i) {
      newOffsets.push_back(zero);
      newSizes.push_back(one);
    }
    acquire.getOffsetsMutable().assign(newOffsets);
    acquire.getSizesMutable().assign(newSizes);
  }

  /// Clear partition hints - they've been consumed by partitioning.
  /// The runtime indices now carry the element coordinates.
  acquire.clearPartitionHints();
  setPartitionMode(acquire.getOperation(), plan.mode);

  /// Store original stride for linearized access (stride = D1 * D2 * ...)
  if (auto staticStride = DbUtils::getStaticElementStride(oldAlloc)) {
    if (*staticStride > 1) {
      setElementStride(acquire.getOperation(), *staticStride);
      ARTS_DEBUG("Element-wise rewrite: stored stride=" << *staticStride);
    }
  }

  /// Rebase EDT users via the element-wise indexer
  rebaseEdtUsers(acquire, builder);
}

void DbElementWiseRewriter::transformDbRef(DbRefOp ref, DbAllocOp newAlloc,
                                           OpBuilder &builder) {
  Location loc = ref.getLoc();
  Type newElementType = newAlloc.getAllocatedElementType();
  Value newSource = (ref.getSource() == oldAlloc.getPtr()) ? newAlloc.getPtr()
                                                           : ref.getSource();

  unsigned newOuterRank = plan.outerRank();
  unsigned newInnerRank = plan.innerRank();

  /// Collect load/store users of this db_ref
  SmallVector<Operation *> refUsers(ref.getResult().getUsers().begin(),
                                    ref.getResult().getUsers().end());
  RemovalUtils removal;

  bool hasLoadStoreUsers = false;
  for (Operation *user : refUsers) {
    auto access = DbUtils::getMemoryAccessInfo(user);
    if (!access)
      continue;

    hasLoadStoreUsers = true;
    OpBuilder::InsertionGuard ig(builder);
    builder.setInsertionPoint(user);
    Location userLoc = user->getLoc();

    ValueRange loadStoreIndices = access->indices;

    SmallVector<Value> newOuter, newInner;
    auto zero = [&]() { return arts::createZeroIndex(builder, userLoc); };

    /// ElementWise: direct mapping of indices
    for (unsigned i = 0; i < newOuterRank; ++i) {
      if (i < loadStoreIndices.size())
        newOuter.push_back(loadStoreIndices[i]);
      else
        newOuter.push_back(zero());
    }
    for (unsigned i = 0; i < newInnerRank; ++i) {
      unsigned src = newOuterRank + i;
      if (src < loadStoreIndices.size())
        newInner.push_back(loadStoreIndices[src]);
      else
        newInner.push_back(zero());
    }

    if (newOuter.empty())
      newOuter.push_back(zero());
    if (newInner.empty())
      newInner.push_back(zero());

    /// Create new db_ref with transformed outer indices
    auto newRef =
        DbRefOp::create(builder, userLoc, newElementType, newSource, newOuter);

    /// Create new load/store with transformed inner indices
    if (access->isRead()) {
      auto load = cast<memref::LoadOp>(user);
      auto newLoad = memref::LoadOp::create(builder, userLoc, newRef, newInner);
      load.replaceAllUsesWith(newLoad.getResult());
    } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
      memref::StoreOp::create(builder, userLoc, store.getValue(), newRef,
                              newInner);
    }
    removal.markForRemoval(user);
  }

  /// For db_refs without load/store users, just update the db_ref type
  if (!hasLoadStoreUsers && !ref.getResult().use_empty()) {
    OpBuilder::InsertionGuard ig(builder);
    builder.setInsertionPoint(ref);
    SmallVector<Value> indices(ref.getIndices().begin(),
                               ref.getIndices().end());
    if (indices.empty())
      indices.push_back(arts::createZeroIndex(builder, loc));
    auto newRef =
        DbRefOp::create(builder, loc, newElementType, newSource, indices);
    ref.replaceAllUsesWith(newRef.getResult());
  }

  /// Remove old load/store operations
  removal.removeAllMarked();
}

bool DbElementWiseRewriter::rebaseEdtUsers(DbAcquireOp acquire,
                                           OpBuilder &builder,
                                           Value /*startBlock*/,
                                           bool /*isSingleChunk*/) {
  ARTS_DEBUG("DbElementWiseRewriter::rebaseEdtUsers");

  auto [edt, blockArg] = EdtUtils::getBlockArgumentForAcquire(acquire);
  Value localView = blockArg ? Value(blockArg) : acquire.getPtr();
  if (!localView) {
    edt = acquire->getParentOfType<EdtOp>();
    localView = acquire.getPtr();
  }
  if (!localView)
    return false;

  /// Determine element type from users or allocation
  Type targetType;
  for (Operation *user : localView.getUsers()) {
    if (auto ref = dyn_cast<DbRefOp>(user)) {
      targetType = ref.getResult().getType();
      break;
    }
  }
  if (!targetType) {
    targetType = localView.getType();
    if (auto outer = dyn_cast<MemRefType>(targetType))
      if (auto inner = dyn_cast<MemRefType>(outer.getElementType()))
        targetType = inner;
  }
  if (!targetType || !isa<MemRefType>(targetType))
    return false;

  Type derivedType = targetType;
  if (auto dbAlloc =
          dyn_cast_or_null<DbAllocOp>(DbUtils::getUnderlyingDbAlloc(localView)))
    derivedType = dbAlloc.getAllocatedElementType();

  /// Build PartitionInfo for fine-grained mode.
  /// The offsets/sizes are now 0/1 after transformAcquire, so we get
  /// the actual element indices from indices (supports multi-dimensional).
  /// For 2D fine-grained (e.g., A[i][j]), indices will be [%i, %j].
  PartitionInfo info;
  info.mode = PartitionMode::fine_grained;
  info.indices.assign(acquire.getIndices().begin(), acquire.getIndices().end());
  info.offsets.assign(acquire.getOffsets().begin(), acquire.getOffsets().end());
  info.sizes.assign(acquire.getSizes().begin(), acquire.getSizes().end());

  /// Create element-wise indexer with PartitionInfo
  auto indexer = std::make_unique<DbElementWiseIndexer>(
      info, plan.outerRank(), plan.innerRank(), oldAlloc.getElementSizes());

  SmallVector<Operation *> users(localView.getUsers().begin(),
                                 localView.getUsers().end());
  llvm::SetVector<Operation *> opsToRemove;
  bool rewritten = false;

  for (Operation *user : users) {
    if (auto ref = dyn_cast<DbRefOp>(user)) {
      indexer->transformDbRefUsers(ref, localView, derivedType, builder,
                                   opsToRemove);
      rewritten = true;
    }
  }

  RemovalUtils removal;
  for (Operation *op : opsToRemove)
    removal.markForRemoval(op);
  removal.removeAllMarked();

  return rewritten;
}
