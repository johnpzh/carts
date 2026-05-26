///==========================================================================///
/// File: PolygeistToSdeUtils.cpp
///
/// Shared helpers for the Polygeist-to-SDE conversion passes.
///==========================================================================///

#include "PolygeistToSdeUtils.h"
#include "carts/utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::carts::sde {

Value materializeDependView(OpBuilder &builder, Location loc, Value source,
                            ArrayRef<Value> indices, ArrayRef<Value> sizes) {
  if (indices.empty() && sizes.empty())
    return source;

  auto sourceType = dyn_cast<MemRefType>(source.getType());
  if (!sourceType)
    return source;

  unsigned rank = sourceType.getRank();
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> viewSizes;
  SmallVector<OpFoldResult> strides;
  offsets.reserve(rank);
  viewSizes.reserve(rank);
  strides.reserve(rank);

  for (unsigned dim = 0; dim < rank; ++dim) {
    offsets.push_back(dim < indices.size() ? OpFoldResult(indices[dim])
                                           : builder.getIndexAttr(0));
    if (dim < sizes.size()) {
      viewSizes.push_back(sizes[dim]);
    } else if (!sourceType.isDynamicDim(dim)) {
      viewSizes.push_back(builder.getIndexAttr(sourceType.getDimSize(dim)));
    } else {
      viewSizes.push_back(
          memref::DimOp::create(builder, loc, source, dim).getResult());
    }
    strides.push_back(builder.getIndexAttr(1));
  }

  return memref::SubViewOp::create(builder, loc, source, offsets, viewSizes,
                                   strides)
      .getResult();
}

SmallVector<Value> clampDepIndices(Value source, ArrayRef<Value> indices,
                                   OpBuilder &builder, Location loc,
                                   ArrayRef<Value> dimSizes) {
  SmallVector<Value> clamped;
  clamped.reserve(indices.size());

  auto sourceType = dyn_cast<MemRefType>(source.getType());
  if (!sourceType) {
    clamped.append(indices.begin(), indices.end());
    return clamped;
  }

  auto castToType = [&](Value v, Type targetTy) -> Value {
    if (v.getType() == targetTy)
      return v;
    if ((v.getType().isIndex() && isa<IntegerType>(targetTy)) ||
        (isa<IntegerType>(v.getType()) && targetTy.isIndex()))
      return arith::IndexCastOp::create(builder, loc, targetTy, v);
    return v;
  };

  auto oneForType = [&](Type ty) -> Value {
    if (ty.isIndex())
      return createOneIndex(builder, loc);
    if (auto intTy = dyn_cast<IntegerType>(ty))
      return arith::ConstantIntOp::create(builder, loc, intTy, 1);
    return Value();
  };

  auto zeroForType = [&](Type ty) -> Value {
    if (ty.isIndex())
      return createZeroIndex(builder, loc);
    if (auto intTy = dyn_cast<IntegerType>(ty))
      return arith::ConstantIntOp::create(builder, loc, intTy, 0);
    return Value();
  };

  for (auto [dim, idx] : llvm::enumerate(indices)) {
    Type idxTy = idx.getType();
    if (!idxTy.isIndex() && !isa<IntegerType>(idxTy)) {
      clamped.push_back(idx);
      continue;
    }

    if (dim >= static_cast<size_t>(sourceType.getRank())) {
      clamped.push_back(idx);
      continue;
    }

    Value zero = zeroForType(idxTy);
    Value one = oneForType(idxTy);
    if (!zero || !one) {
      clamped.push_back(idx);
      continue;
    }

    Value dimSize;
    if (dim < dimSizes.size()) {
      dimSize = dimSizes[dim];
    } else if (sourceType.isDynamicDim(dim)) {
      dimSize = memref::DimOp::create(builder, loc, source, dim);
    } else {
      dimSize = createConstantIndex(builder, loc, sourceType.getDimSize(dim));
    }

    Value dimSizeTyped = castToType(dimSize, idxTy);
    Value upper = arith::SubIOp::create(builder, loc, dimSizeTyped, one);

    Value belowZero = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::slt, idx, zero);
    Value afterLowerClamp =
        arith::SelectOp::create(builder, loc, belowZero, zero, idx);

    Value aboveUpper = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sgt, afterLowerClamp, upper);
    Value finalIdx = arith::SelectOp::create(builder, loc, aboveUpper, upper,
                                             afterLowerClamp);
    clamped.push_back(finalIdx);
  }

  return clamped;
}

bool isInsideOmpRegion(Operation *op) {
  for (Operation *ancestor = op ? op->getParentOp() : nullptr; ancestor;
       ancestor = ancestor->getParentOp()) {
    if (ancestor->getDialect() &&
        ancestor->getDialect()->getNamespace() == "omp")
      return true;
  }
  return false;
}

bool containsOmpOp(Operation *op) {
  bool found = false;
  op->walk([&](Operation *nested) {
    if (nested->getDialect() && nested->getDialect()->getNamespace() == "omp") {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

} // namespace mlir::carts::sde
