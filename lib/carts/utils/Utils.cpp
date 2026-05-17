///==========================================================================///
/// File: Utils.cpp
/// Defines utility functions for the ARTS dialect.
///==========================================================================///
#include "carts/utils/Utils.h"
#include "carts/Dialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include <cassert>

/// Debug
#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(utils);
[[maybe_unused]] static const auto *kArtsUtilsDebugType = ARTS_DEBUG_TYPE_STR;

namespace mlir {
namespace carts {
namespace {

bool hasFloatingPointType(Type type) {
  if (!type)
    return false;
  if (type.isF16() || type.isBF16() || type.isF32() || type.isF64() ||
      type.isF80() || type.isF128())
    return true;
  if (auto vectorType = dyn_cast<VectorType>(type))
    return hasFloatingPointType(vectorType.getElementType());
  return false;
}

} // namespace

///===----------------------------------------------------------------------===///
/// Dominance Utilities
///===----------------------------------------------------------------------===///

bool operationTouchesFloatingPoint(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (hasFloatingPointType(operand.getType()))
      return true;
  }
  for (Value result : op->getResults()) {
    if (hasFloatingPointType(result.getType()))
      return true;
  }
  return false;
}

bool dominatesOrInAncestor(Value v, Operation *op, DominanceInfo &domInfo) {
  if (!v || !op)
    return false;
  if (Operation *def = v.getDefiningOp()) {
    Region *defRegion = def->getParentRegion();
    Region *opRegion = op->getParentRegion();
    if (defRegion && opRegion && defRegion != opRegion &&
        defRegion->isAncestor(opRegion))
      return true;
  } else {
    /// Block arguments from ancestor regions are considered in-scope.
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (Region *argRegion = arg.getOwner()->getParent())
        if (Region *opRegion = op->getParentRegion())
          if (argRegion->isAncestor(opRegion))
            return true;
    }
  }
  return domInfo.dominates(v, op);
}

///===----------------------------------------------------------------------===///
/// Constant Index Creation Utilities
///===----------------------------------------------------------------------===///

Value createConstantIndex(OpBuilder &builder, Location loc, int64_t val) {
  return arith::ConstantIndexOp::create(builder, loc, val);
}

Value createZeroIndex(OpBuilder &builder, Location loc) {
  return createConstantIndex(builder, loc, 0);
}

Value createOneIndex(OpBuilder &builder, Location loc) {
  return createConstantIndex(builder, loc, 1);
}

bool isSideEffectFreeArithmeticLikeOp(Operation *op) {
  if (!op || op->hasTrait<OpTrait::IsTerminator>())
    return true;

  return isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
             arith::DivUIOp, arith::DivSIOp, arith::RemUIOp, arith::RemSIOp,
             arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::NegFOp, arith::CmpIOp, arith::CmpFOp, arith::SelectOp,
             arith::MaxUIOp, arith::MinUIOp, arith::MaximumFOp,
             arith::MinimumFOp, arith::IndexCastOp, arith::IndexCastUIOp,
             arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncIOp,
             arith::TruncFOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
             arith::FPToUIOp>(op);
}

///===----------------------------------------------------------------------===///
/// Dependency Index Clamping Utilities
///===----------------------------------------------------------------------===///

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

///===----------------------------------------------------------------------===///
/// Block Work Detection Utilities
///===----------------------------------------------------------------------===///

bool hasWorkAfterInParentBlock(Operation *op) {
  if (!op || !op->getBlock())
    return false;

  for (Operation *next = op->getNextNode(); next; next = next->getNextNode()) {
    if (next->hasTrait<OpTrait::IsTerminator>())
      continue;
    return true;
  }
  return false;
}

///===----------------------------------------------------------------------===///
/// OMP Region Utilities
///===----------------------------------------------------------------------===///

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

namespace arts {

///===----------------------------------------------------------------------===///
/// Type and Size Utilities
///===----------------------------------------------------------------------===///

/// Computes the byte size of the given element type, supporting integer and
/// floating-point types. When the type is a memref, all static dimensions must
/// be known; otherwise, the size is treated as unknown (returns 0).
uint64_t getElementTypeByteSize(Type elemTy) {
  /// Safety check: return 0 for null or invalid types
  if (!elemTy) {
    return 0;
  }

  if (auto memTy = dyn_cast<MemRefType>(elemTy)) {
    uint64_t elementBytes = getElementTypeByteSize(memTy.getElementType());
    if (elementBytes == 0)
      return 0;

    uint64_t total = elementBytes;
    for (int64_t dim : memTy.getShape()) {
      if (dim == ShapedType::kDynamic)
        return 0;
      total *= static_cast<uint64_t>(std::max<int64_t>(dim, 0));
    }
    return total;
  }
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return intTy.getWidth() / 8u;
  if (auto fTy = dyn_cast<FloatType>(elemTy))
    return fTy.getWidth() / 8u;
  /// Unknown type
  return 0;
}

/// Gets the element memref type for a given element type and sizes.
MemRefType getElementMemRefType(Type elementType,
                                ArrayRef<Value> elementSizes) {
  /// Enforce scalar payloads use a single trailing dimension of 1 instead of
  /// an empty shape to keep rank handling uniform across the pipeline.
  const size_t rank = elementSizes.empty() ? 1 : elementSizes.size();
  SmallVector<int64_t> elementShape(rank, ShapedType::kDynamic);
  return MemRefType::get(elementShape, elementType);
}

///===----------------------------------------------------------------------===///
/// Access Mode Utilities
///===----------------------------------------------------------------------===///

/// Combine two access modes and return the more permissive mode
ArtsMode combineAccessModes(ArtsMode mode1, ArtsMode mode2) {
  /// If either mode is uninitialized, return the other mode
  if (mode1 == ArtsMode::uninitialized)
    return mode2;
  if (mode2 == ArtsMode::uninitialized)
    return mode1;

  /// If either mode is inout, the result is inout (most permissive)
  if (mode1 == ArtsMode::inout || mode2 == ArtsMode::inout)
    return ArtsMode::inout;

  /// If one is 'in' and the other is 'out', the result is inout
  if ((mode1 == ArtsMode::in && mode2 == ArtsMode::out) ||
      (mode1 == ArtsMode::out && mode2 == ArtsMode::in))
    return ArtsMode::inout;

  /// If both are the same, return that mode
  if (mode1 == mode2)
    return mode1;

  /// Default to inout for any other combination (shouldn't happen)
  return ArtsMode::inout;
}

///===----------------------------------------------------------------------===///
/// Operation Replacement Utilities
///===----------------------------------------------------------------------===///

/// Replaces uses of a value within a specific region.
void replaceInRegion(Region &region, Value from, Value to) {
  from.replaceUsesWithIf(to, [&](OpOperand &operand) {
    return region.isAncestor(operand.getOwner()->getParentRegion());
  });
}

} // namespace arts
} // namespace carts
} // namespace mlir
