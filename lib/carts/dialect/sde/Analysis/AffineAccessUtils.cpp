///==========================================================================///
/// File: AffineAccessUtils.cpp
///==========================================================================///

#include "carts/dialect/sde/Analysis/AffineAccessUtils.h"

#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mlir::carts;

namespace mlir::carts::sde {
namespace {

static bool isLoopInductionLike(Value value) {
  auto blockArg = dyn_cast_or_null<BlockArgument>(value);
  if (!blockArg)
    return false;

  Operation *parent = blockArg.getOwner()->getParentOp();
  if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent))
    return forOp.getInductionVar() == value;
  if (auto loopNest = dyn_cast_or_null<omp::LoopNestOp>(parent))
    return llvm::is_contained(loopNest.getIVs(), blockArg);

  return false;
}

static std::optional<LinearizedAccess2D>
matchLinearizedMul(Value mulCandidate, Value innerCandidate,
                   Value requiredStride) {
  mulCandidate = ::mlir::carts::arts::ValueAnalysis::stripNumericCasts(mulCandidate);
  innerCandidate = ::mlir::carts::arts::ValueAnalysis::stripNumericCasts(innerCandidate);

  auto mulOp = mulCandidate.getDefiningOp<arith::MulIOp>();
  if (!mulOp)
    return std::nullopt;

  Value lhs = ::mlir::carts::arts::ValueAnalysis::stripNumericCasts(mulOp.getLhs());
  Value rhs = ::mlir::carts::arts::ValueAnalysis::stripNumericCasts(mulOp.getRhs());

  auto build = [&](Value outer, Value stride)
      -> std::optional<LinearizedAccess2D> {
    outer = ::mlir::carts::arts::ValueAnalysis::stripNumericCasts(outer);
    stride = ::mlir::carts::arts::ValueAnalysis::stripNumericCasts(stride);
    if (!outer || !stride || !innerCandidate)
      return std::nullopt;
    if (!outer.getType().isIndex() || !stride.getType().isIndex() ||
        !innerCandidate.getType().isIndex())
      return std::nullopt;
    if (requiredStride && !::mlir::carts::arts::ValueAnalysis::sameValue(stride, requiredStride))
      return std::nullopt;
    return LinearizedAccess2D{outer, innerCandidate, stride};
  };

  if (requiredStride) {
    if (::mlir::carts::arts::ValueAnalysis::sameValue(lhs, requiredStride))
      return build(rhs, lhs);
    if (::mlir::carts::arts::ValueAnalysis::sameValue(rhs, requiredStride))
      return build(lhs, rhs);
    return std::nullopt;
  }

  bool lhsIv = isLoopInductionLike(lhs);
  bool rhsIv = isLoopInductionLike(rhs);
  if (lhsIv && !rhsIv)
    return build(lhs, rhs);
  if (rhsIv && !lhsIv)
    return build(rhs, lhs);

  std::optional<int64_t> lhsConst = ::mlir::carts::arts::ValueAnalysis::tryFoldConstantIndex(lhs);
  std::optional<int64_t> rhsConst = ::mlir::carts::arts::ValueAnalysis::tryFoldConstantIndex(rhs);
  if (lhsConst && !rhsConst)
    return build(rhs, lhs);
  if (rhsConst && !lhsConst)
    return build(lhs, rhs);

  return std::nullopt;
}

} // namespace

std::optional<LinearizedAccess2D>
decomposeRowMajorLinearizedIndex(Value index, Value requiredStride) {
  index = ::mlir::carts::arts::ValueAnalysis::stripNumericCasts(index);
  auto addOp = index.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return std::nullopt;

  if (auto info =
          matchLinearizedMul(addOp.getLhs(), addOp.getRhs(), requiredStride))
    return info;
  return matchLinearizedMul(addOp.getRhs(), addOp.getLhs(), requiredStride);
}

std::optional<SmallVector<Value, 2>>
inferRowMajorFlatShape(Value totalElements, Value stride) {
  if (!totalElements || !stride)
    return std::nullopt;

  totalElements = ::mlir::carts::arts::ValueAnalysis::stripNumericCasts(totalElements);
  stride = ::mlir::carts::arts::ValueAnalysis::stripNumericCasts(stride);
  if (!totalElements.getType().isIndex() || !stride.getType().isIndex())
    return std::nullopt;

  auto mulOp = totalElements.getDefiningOp<arith::MulIOp>();
  if (!mulOp)
    return std::nullopt;

  Value lhs = ::mlir::carts::arts::ValueAnalysis::stripNumericCasts(mulOp.getLhs());
  Value rhs = ::mlir::carts::arts::ValueAnalysis::stripNumericCasts(mulOp.getRhs());
  if (::mlir::carts::arts::ValueAnalysis::sameValue(lhs, stride))
    return SmallVector<Value, 2>{rhs, stride};
  if (::mlir::carts::arts::ValueAnalysis::sameValue(rhs, stride))
    return SmallVector<Value, 2>{lhs, stride};

  return std::nullopt;
}

} // namespace mlir::carts::sde
