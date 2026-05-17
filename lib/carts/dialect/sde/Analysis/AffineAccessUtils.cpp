///==========================================================================///
/// File: AffineAccessUtils.cpp
///==========================================================================///

#include "carts/dialect/sde/Analysis/AffineAccessUtils.h"

#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

using namespace mlir;
using namespace mlir::carts;

namespace mlir::carts::sde {
namespace {

static bool isLoopInductionVar(Value value) {
  auto arg = dyn_cast_or_null<BlockArgument>(value);
  if (!arg)
    return false;

  Operation *parent = arg.getOwner()->getParentOp();
  if (!parent)
    return false;

  if (auto loopNest = dyn_cast<omp::LoopNestOp>(parent)) {
    for (BlockArgument iv : loopNest.getIVs())
      if (iv == arg)
        return true;
    return false;
  }

  if (auto loopLike = dyn_cast<LoopLikeOpInterface>(parent)) {
    if (auto ivs = loopLike.getLoopInductionVars()) {
      for (Value iv : *ivs)
        if (iv == value)
          return true;
    }
  }
  return false;
}

static std::optional<LinearizedAccess2D>
matchLinearizedMul(Value mulCandidate, Value innerCandidate,
                   Value requiredStride) {
  mulCandidate =
      ::mlir::carts::ValueAnalysis::stripNumericCasts(mulCandidate);
  innerCandidate =
      ::mlir::carts::ValueAnalysis::stripNumericCasts(innerCandidate);

  auto mulOp = mulCandidate.getDefiningOp<arith::MulIOp>();
  if (!mulOp)
    return std::nullopt;

  Value lhs =
      ::mlir::carts::ValueAnalysis::stripNumericCasts(mulOp.getLhs());
  Value rhs =
      ::mlir::carts::ValueAnalysis::stripNumericCasts(mulOp.getRhs());

  auto build = [&](Value outer,
                   Value stride) -> std::optional<LinearizedAccess2D> {
    outer = ::mlir::carts::ValueAnalysis::stripNumericCasts(outer);
    stride = ::mlir::carts::ValueAnalysis::stripNumericCasts(stride);
    if (!outer || !stride || !innerCandidate)
      return std::nullopt;
    if (!outer.getType().isIndex() || !stride.getType().isIndex() ||
        !innerCandidate.getType().isIndex())
      return std::nullopt;
    if (requiredStride &&
        !::mlir::carts::ValueAnalysis::sameValue(stride, requiredStride))
      return std::nullopt;
    return LinearizedAccess2D{outer, innerCandidate, stride};
  };

  if (requiredStride) {
    if (::mlir::carts::ValueAnalysis::sameValue(lhs, requiredStride))
      return build(rhs, lhs);
    if (::mlir::carts::ValueAnalysis::sameValue(rhs, requiredStride))
      return build(lhs, rhs);
    return std::nullopt;
  }

  bool lhsIv = isLoopInductionVar(lhs);
  bool rhsIv = isLoopInductionVar(rhs);
  if (lhsIv && !rhsIv)
    return build(lhs, rhs);
  if (rhsIv && !lhsIv)
    return build(rhs, lhs);

  std::optional<int64_t> lhsConst =
      ::mlir::carts::ValueAnalysis::tryFoldConstantIndex(lhs);
  std::optional<int64_t> rhsConst =
      ::mlir::carts::ValueAnalysis::tryFoldConstantIndex(rhs);
  if (lhsConst && !rhsConst)
    return build(rhs, lhs);
  if (rhsConst && !lhsConst)
    return build(lhs, rhs);

  return std::nullopt;
}

} // namespace

std::optional<LinearizedAccess2D>
decomposeRowMajorLinearizedIndex(Value index, Value requiredStride) {
  index = ::mlir::carts::ValueAnalysis::stripNumericCasts(index);
  auto addOp = index.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return std::nullopt;

  if (auto info =
          matchLinearizedMul(addOp.getLhs(), addOp.getRhs(), requiredStride))
    return info;
  return matchLinearizedMul(addOp.getRhs(), addOp.getLhs(), requiredStride);
}

std::optional<SmallVector<Value, 2>> inferRowMajorFlatShape(Value totalElements,
                                                            Value stride) {
  if (!totalElements || !stride)
    return std::nullopt;

  totalElements =
      ::mlir::carts::ValueAnalysis::stripNumericCasts(totalElements);
  stride = ::mlir::carts::ValueAnalysis::stripNumericCasts(stride);
  if (!totalElements.getType().isIndex() || !stride.getType().isIndex())
    return std::nullopt;

  auto mulOp = totalElements.getDefiningOp<arith::MulIOp>();
  if (!mulOp)
    return std::nullopt;

  Value lhs =
      ::mlir::carts::ValueAnalysis::stripNumericCasts(mulOp.getLhs());
  Value rhs =
      ::mlir::carts::ValueAnalysis::stripNumericCasts(mulOp.getRhs());
  if (::mlir::carts::ValueAnalysis::sameValue(lhs, stride))
    return SmallVector<Value, 2>{rhs, stride};
  if (::mlir::carts::ValueAnalysis::sameValue(rhs, stride))
    return SmallVector<Value, 2>{lhs, stride};

  return std::nullopt;
}

} // namespace mlir::carts::sde
