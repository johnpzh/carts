///==========================================================================///
/// File: RtDbUtils.cpp
///
/// Runtime-dialect datablock provenance helpers.
///==========================================================================///
#include "carts/dialect/arts-rt/Utils/RtDbUtils.h"

#include "carts/dialect/arts/Utils/ValueAnalysisUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/ValueAnalysis.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
using namespace mlir::carts::arts_rt;

namespace {

static Value getUnderlyingRtValueImpl(Value value,
                                      llvm::SmallPtrSetImpl<Value> &visited,
                                      unsigned depth) {
  if (!value || depth > 64 || !visited.insert(value).second)
    return nullptr;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *owner = blockArg.getOwner();
    if (owner && owner->getParentOp()) {
      if (auto edt = dyn_cast<EdtOp>(owner->getParentOp())) {
        unsigned argIndex = blockArg.getArgNumber();
        ValueRange deps = edt.getDependencies();
        if (argIndex < deps.size())
          return getUnderlyingRtValueImpl(deps[argIndex], visited, depth + 1);
      }
    }
    return value;
  }

  Operation *op = value.getDefiningOp();
  if (!op)
    return nullptr;
  if (!op->getBlock())
    return nullptr;

  if (isa<DepDbAcquireOp, DbAllocOp, memref::AllocOp, memref::AllocaOp,
          memref::GetGlobalOp>(op))
    return value;

  auto trace = [&](Value source) -> Value {
    return getUnderlyingRtValueImpl(source, visited, depth + 1);
  };

  if (auto dbGep = dyn_cast<DbGepOp>(op))
    return trace(dbGep.getBasePtr());
  if (auto dbAcquire = dyn_cast<DbAcquireOp>(op))
    return trace(dbAcquire.getSourcePtr());
  if (auto dbRef = dyn_cast<DbRefOp>(op))
    return trace(dbRef.getSource());
  if (auto subview = dyn_cast<memref::SubViewOp>(op))
    return trace(subview.getSource());
  if (auto castOp = dyn_cast<memref::CastOp>(op))
    return trace(castOp.getSource());
  if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(op)) {
    ValueRange inputs = unrealized.getInputs();
    return inputs.empty() ? nullptr : trace(inputs.front());
  }
  if (auto view = dyn_cast<memref::ViewOp>(op))
    return trace(view.getSource());
  if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(op))
    return trace(reinterpret.getSource());
  if (auto p2m = dyn_cast<polygeist::Pointer2MemrefOp>(op))
    return trace(p2m.getSource());
  if (auto m2p = dyn_cast<polygeist::Memref2PointerOp>(op))
    return trace(m2p.getSource());
  if (auto gep = dyn_cast<LLVM::GEPOp>(op))
    return trace(gep.getBase());
  if (auto subindex = dyn_cast<polygeist::SubIndexOp>(op))
    return trace(subindex.getSource());

  return nullptr;
}

static bool isDerivedFromRtPtrImpl(Value value, Value source,
                                   llvm::SmallPtrSetImpl<Value> &visited,
                                   unsigned depth) {
  if (depth > 64)
    return false;
  if (value == source)
    return true;
  if (!visited.insert(value).second)
    return false;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *parentBlock = blockArg.getParentBlock();
    if (parentBlock && parentBlock->getParentOp()) {
      if (auto edt = dyn_cast<EdtOp>(parentBlock->getParentOp())) {
        unsigned argIndex = blockArg.getArgNumber();
        ValueRange deps = edt.getDependencies();
        if (argIndex < deps.size())
          return isDerivedFromRtPtrImpl(deps[argIndex], source, visited,
                                        depth + 1);
      }
    }
    return false;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;

  auto trace = [&](Value next) -> bool {
    return isDerivedFromRtPtrImpl(next, source, visited, depth + 1);
  };

  if (auto dbGep = dyn_cast<DbGepOp>(defOp))
    return trace(dbGep.getBasePtr());
  if (auto dbAcquire = dyn_cast<DbAcquireOp>(defOp))
    return trace(dbAcquire.getSourcePtr());
  if (auto dbRef = dyn_cast<DbRefOp>(defOp))
    return trace(dbRef.getSource());
  if (auto gepOp = dyn_cast<LLVM::GEPOp>(defOp))
    return trace(gepOp.getBase());
  if (auto ptr2memref = dyn_cast<polygeist::Pointer2MemrefOp>(defOp))
    return trace(ptr2memref.getSource());
  if (auto memref2ptr = dyn_cast<polygeist::Memref2PointerOp>(defOp))
    return trace(memref2ptr.getSource());
  if (auto subview = dyn_cast<memref::SubViewOp>(defOp))
    return trace(subview.getSource());
  if (auto cast = dyn_cast<memref::CastOp>(defOp))
    return trace(cast.getSource());
  if (auto view = dyn_cast<memref::ViewOp>(defOp))
    return trace(view.getSource());
  if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(defOp))
    return trace(reinterpretCast.getSource());

  if (defOp->getNumOperands() > 0)
    return trace(defOp->getOperand(0));

  return false;
}

} // namespace

namespace {

static void finalizeSingleElementState(DbLoweringInfo &info) {
  info.isSingleElement = false;
  if (info.sizes.empty()) {
    info.isSingleElement = true;
    return;
  }
  if (info.sizes.size() == 1)
    info.isSingleElement = ValueAnalysis::isOneConstant(info.sizes[0]);
}

} // namespace

DbLoweringInfo RtDbUtils::extractDbLoweringInfo(DbAcquireOp op) {
  DbLoweringInfo info;
  if (!op) {
    finalizeSingleElementState(info);
    return info;
  }
  info.sizes = DbUtils::getDepSizesFromDb(op.getOperation());
  info.offsets = DbUtils::getDepOffsetsFromDb(op.getOperation());
  info.indices.assign(op.getIndices().begin(), op.getIndices().end());
  finalizeSingleElementState(info);
  return info;
}

DbLoweringInfo RtDbUtils::extractDbLoweringInfo(DbAllocOp op) {
  DbLoweringInfo info;
  if (op)
    info.sizes.assign(op.getSizes().begin(), op.getSizes().end());
  finalizeSingleElementState(info);
  return info;
}

DbLoweringInfo RtDbUtils::extractDbLoweringInfo(DepDbAcquireOp op) {
  DbLoweringInfo info;
  if (!op) {
    finalizeSingleElementState(info);
    return info;
  }
  info.sizes.assign(op.getSizes().begin(), op.getSizes().end());
  info.offsets.assign(op.getOffsets().begin(), op.getOffsets().end());
  info.indices.assign(op.getIndices().begin(), op.getIndices().end());
  finalizeSingleElementState(info);
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

Value RtDbUtils::getUnderlyingValue(Value value) {
  llvm::SmallPtrSet<Value, 16> visited;
  return getUnderlyingRtValueImpl(value, visited, 0);
}

Operation *RtDbUtils::getUnderlyingOperation(Value value) {
  Value underlying = getUnderlyingValue(value);
  return underlying ? underlying.getDefiningOp() : nullptr;
}

bool RtDbUtils::isDerivedFromPtr(Value value, Value source) {
  llvm::SmallPtrSet<Value, 16> visited;
  return isDerivedFromRtPtrImpl(value, source, visited, 0);
}

std::optional<int64_t> RtDbUtils::tryFoldConstantIndex(Value value,
                                                       unsigned depth) {
  return arts::tryFoldConstantIndex(value, depth);
}

std::optional<int64_t> RtDbUtils::getConstantIndexStripped(Value value) {
  if (!value)
    return std::nullopt;
  return tryFoldConstantIndex(ValueAnalysis::stripNumericCasts(value));
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
    ValueRange elementSizes, ValueRange blockSpans, ValueRange totalBlockCounts,
    SmallVectorImpl<Value> &blockOffsets, SmallVectorImpl<Value> &blockSizes) {
  DbUtils::convertElementSliceToBlockSlice(
      builder, loc, elementOffsets, elementSizes, blockSpans, totalBlockCounts,
      blockOffsets, blockSizes);
}

void RtDbUtils::mergeNormalizedBlockSlice(
    OpBuilder &builder, Location loc, ValueRange existingOffsets,
    ValueRange existingSizes, ValueRange totalBlockCounts,
    ValueRange normalizedOffsets, ValueRange normalizedSizes,
    SmallVectorImpl<Value> &blockOffsets, SmallVectorImpl<Value> &blockSizes) {
  DbUtils::mergeNormalizedBlockSlice(
      builder, loc, existingOffsets, existingSizes, totalBlockCounts,
      normalizedOffsets, normalizedSizes, blockOffsets, blockSizes);
}
