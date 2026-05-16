///==========================================================================///
/// File: RtDbUtils.cpp
///
/// Runtime-dialect datablock provenance helpers.
///==========================================================================///
#include "carts/dialect/arts-rt/Utils/RtDbUtils.h"

#include "carts/utils/OperationAttributes.h"
#include "carts/utils/ValueAnalysis.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
using namespace mlir::carts::arts_rt;

namespace {

static Operation *resolveRootAllocFromAttr(Value value) {
  if (!value)
    return nullptr;

  std::optional<int64_t> rootId;
  llvm::SmallPtrSet<Operation *, 8> visited;
  for (unsigned depth = 0; depth < 16 && value; ++depth) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp || !visited.insert(defOp).second)
      break;

    if (auto candidate = getDbRootAllocId(defOp)) {
      rootId = candidate;
      break;
    }

    if (auto dbGep = dyn_cast<DbGepOp>(defOp)) {
      value = dbGep.getBasePtr();
      continue;
    }

    break;
  }

  if (!rootId || *rootId <= 0)
    return nullptr;

  auto module = value.getParentRegion()
                    ? value.getParentRegion()->getParentOfType<ModuleOp>()
                    : ModuleOp();
  if (!module)
    return nullptr;

  Operation *resolved = nullptr;
  module.walk([&](DbAllocOp alloc) {
    if (getArtsId(alloc) != *rootId)
      return WalkResult::advance();
    resolved = alloc.getOperation();
    return WalkResult::interrupt();
  });
  return resolved;
}

} // namespace

DbLoweringInfo RtDbUtils::extractDbLoweringInfo(DbAcquireOp op) {
  return DbUtils::extractDbLoweringInfo(op);
}

DbLoweringInfo RtDbUtils::extractDbLoweringInfo(DbAllocOp op) {
  return DbUtils::extractDbLoweringInfo(op);
}

DbLoweringInfo RtDbUtils::extractDbLoweringInfo(DepDbAcquireOp op) {
  DbLoweringInfo info;
  info.sizes.assign(op.getSizes().begin(), op.getSizes().end());
  info.offsets.assign(op.getOffsets().begin(), op.getOffsets().end());
  info.indices.assign(op.getIndices().begin(), op.getIndices().end());
  info.isSingleElement = false;
  if (info.sizes.empty()) {
    info.isSingleElement = true;
    return info;
  }
  if (info.sizes.size() == 1)
    info.isSingleElement = ValueAnalysis::isOneConstant(info.sizes[0]);
  return info;
}

Operation *RtDbUtils::getUnderlyingDb(Value value, unsigned depth) {
  if (!value)
    return nullptr;
  if (depth > 20)
    return nullptr;

  if (auto depAcquire = value.getDefiningOp<DepDbAcquireOp>())
    return depAcquire.getOperation();

  if (Operation *defOp = value.getDefiningOp()) {
    if (Operation *rootAlloc = resolveRootAllocFromAttr(value))
      return rootAlloc;
    if (auto dbGep = dyn_cast<DbGepOp>(defOp))
      return getUnderlyingDb(dbGep.getBasePtr(), depth + 1);
  }

  return DbUtils::getUnderlyingDb(value, depth);
}

Operation *RtDbUtils::getUnderlyingDbAlloc(Value value) {
  Operation *underlyingDb = getUnderlyingDb(value);
  if (!underlyingDb)
    return nullptr;
  if (isa<DbAllocOp>(underlyingDb))
    return underlyingDb;
  if (auto dbAcquire = dyn_cast<DbAcquireOp>(underlyingDb))
    return getUnderlyingDbAlloc(dbAcquire.getSourcePtr());
  return nullptr;
}

DbAllocOp RtDbUtils::getAllocOpFromGuid(Value dbGuid) {
  if (!dbGuid)
    return nullptr;

  if (auto depDbAcquireOp = dbGuid.getDefiningOp<DepDbAcquireOp>())
    return depDbAcquireOp.getGuid()
               ? depDbAcquireOp.getGuid().getDefiningOp<DbAllocOp>()
               : nullptr;
  return DbUtils::getAllocOpFromGuid(dbGuid);
}

SmallVector<Value> RtDbUtils::getSizesFromDb(Operation *dbOp) {
  if (auto depDbAcquireOp = dyn_cast_or_null<DepDbAcquireOp>(dbOp))
    return SmallVector<Value>(depDbAcquireOp.getSizes().begin(),
                              depDbAcquireOp.getSizes().end());
  return DbUtils::getSizesFromDb(dbOp);
}

SmallVector<Value> RtDbUtils::getSizesFromDb(Value dbPtr) {
  return getSizesFromDb(getUnderlyingDb(dbPtr));
}

SmallVector<Value> RtDbUtils::getDepSizesFromDb(Operation *dbOp) {
  if (auto depAcquireOp = dyn_cast_or_null<DepDbAcquireOp>(dbOp))
    return SmallVector<Value>(depAcquireOp.getSizes().begin(),
                              depAcquireOp.getSizes().end());
  return DbUtils::getDepSizesFromDb(dbOp);
}

SmallVector<Value> RtDbUtils::getDepSizesFromDb(Value dbPtr) {
  return getDepSizesFromDb(getUnderlyingDb(dbPtr));
}

SmallVector<Value> RtDbUtils::getDepOffsetsFromDb(Operation *dbOp) {
  if (auto depAcquireOp = dyn_cast_or_null<DepDbAcquireOp>(dbOp))
    return SmallVector<Value>(depAcquireOp.getOffsets().begin(),
                              depAcquireOp.getOffsets().end());
  return DbUtils::getDepOffsetsFromDb(dbOp);
}

SmallVector<Value> RtDbUtils::getDepOffsetsFromDb(Value dbPtr) {
  return getDepOffsetsFromDb(getUnderlyingDb(dbPtr));
}

DbMode RtDbUtils::convertArtsModeToDbMode(ArtsMode mode) {
  return DbUtils::convertArtsModeToDbMode(mode);
}

void RtDbUtils::convertElementSliceToBlockSlice(
    OpBuilder &builder, Location loc, ValueRange elementOffsets,
    ValueRange elementSizes, ValueRange blockSpans,
    ValueRange totalBlockCounts, SmallVectorImpl<Value> &blockOffsets,
    SmallVectorImpl<Value> &blockSizes) {
  DbUtils::convertElementSliceToBlockSlice(builder, loc, elementOffsets,
                                           elementSizes, blockSpans,
                                           totalBlockCounts, blockOffsets,
                                           blockSizes);
}

void RtDbUtils::mergeNormalizedBlockSlice(
    OpBuilder &builder, Location loc, ValueRange existingOffsets,
    ValueRange existingSizes, ValueRange totalBlockCounts,
    ValueRange normalizedOffsets, ValueRange normalizedSizes,
    SmallVectorImpl<Value> &blockOffsets, SmallVectorImpl<Value> &blockSizes) {
  DbUtils::mergeNormalizedBlockSlice(builder, loc, existingOffsets,
                                     existingSizes, totalBlockCounts,
                                     normalizedOffsets, normalizedSizes,
                                     blockOffsets, blockSizes);
}
