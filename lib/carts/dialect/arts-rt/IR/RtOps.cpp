///==========================================================================///
/// File: RtOps.cpp
/// Defines runtime-only ARTS operation helpers.
///==========================================================================///

#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts_rt;

#define GET_OP_CLASSES
#include "carts/dialect/arts-rt/IR/RtOps.cpp.inc"

namespace {
Value createZeroI32(OpBuilder &builder, Location loc) {
  return arith::ConstantIntOp::create(builder, loc, 0, 32);
}

static LogicalResult verifyParamvScalarTypes(Operation *op, TypeRange types,
                                             StringRef role) {
  for (auto [index, type] : llvm::enumerate(types)) {
    if (type.isIntOrIndexOrFloat())
      continue;
    return op->emitOpError(role)
           << " must be scalar int/index/float values; entry #" << index
           << " has type " << type;
  }
  return success();
}
} // namespace

LogicalResult CreateEpochOp::verify() {
  bool hasGuid = getFinishEdtGuid() != nullptr;
  bool hasSlot = getFinishSlot() != nullptr;
  if (hasGuid != hasSlot)
    return emitOpError("finishEdtGuid and finishSlot must both be present or "
                       "both be absent");
  return success();
}

LogicalResult EdtParamPackOp::verify() {
  return verifyParamvScalarTypes(getOperation(), getParams().getTypes(),
                                 "operands");
}

LogicalResult EdtParamUnpackOp::verify() {
  return verifyParamvScalarTypes(getOperation(), getUnpacked().getTypes(),
                                 "results");
}

void EdtCreateOp::build(OpBuilder &builder, OperationState &state,
                        Value param_memref, Value depCount) {
  build(builder, state, param_memref, depCount,
        createZeroI32(builder, state.location));
}

void EdtCreateOp::build(OpBuilder &builder, OperationState &state,
                        Value param_memref, Value depCount, Value route) {
  state.addTypes(builder.getI64Type());
  state.addOperands({param_memref, depCount, route});
}

void EdtCreateOp::build(OpBuilder &builder, OperationState &state,
                        Value param_memref, Value depCount, Value route,
                        Value epochGuid) {
  state.addTypes(builder.getI64Type());
  state.addOperands({param_memref, depCount, route, epochGuid});
}

void DbGepOp::build(OpBuilder &builder, OperationState &state, Type ptr,
                    Value base_ptr, SmallVector<Value> indices,
                    SmallVector<Value> strides) {
  state.addOperands(base_ptr);
  state.addOperands(indices);
  state.addOperands(strides);
  SmallVector<int32_t, 3> segments = {1, static_cast<int32_t>(indices.size()),
                                      static_cast<int32_t>(strides.size())};
  state.addAttribute(DbGepOp::getOperandSegmentSizesAttrName(state.name),
                     builder.getDenseI32ArrayAttr(segments));
  state.addTypes(ptr);
}

LogicalResult RecordDepOp::verify() {
  const size_t dbCount = getDatablocks().size();
  if (auto modes = getAcquireModes()) {
    if (modes->size() != dbCount)
      return emitOpError("acquire_modes entries (")
             << modes->size() << ") must match datablocks (" << dbCount << ")";
  }

  if (!getByteOffsets().empty() && getByteOffsets().size() != dbCount)
    return emitOpError("byte_offsets entries (")
           << getByteOffsets().size() << ") must match datablocks (" << dbCount
           << ")";

  if (!getByteSizes().empty() && getByteSizes().size() != dbCount)
    return emitOpError("byte_sizes entries (")
           << getByteSizes().size() << ") must match datablocks (" << dbCount
           << ")";

  if (auto flags = getDepFlags()) {
    if (flags->size() != dbCount)
      return emitOpError("dep_flags entries (")
             << flags->size() << ") must match datablocks (" << dbCount << ")";
  }

  if (getByteOffsets().empty() != getByteSizes().empty())
    return emitOpError("byte_offsets and byte_sizes must be provided together");

  return success();
}
