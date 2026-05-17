///==========================================================================///
/// File: BlockedAccessUtils.h
///
/// Shared helpers for recognizing block-base-relative access patterns.
/// These patterns show up both during DB index localization and in late
/// codegen cleanups that recover loop-local blocked slices.
///==========================================================================///

#ifndef ARTS_UTILS_BLOCKEDACCESSUTILS_H
#define ARTS_UTILS_BLOCKEDACCESSUTILS_H

#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/ValueAnalysisUtils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <optional>

namespace mlir {
namespace carts::arts {

namespace detail {

inline std::optional<bool> tryFoldBool(Value value) {
  auto folded = tryFoldConstantIndex(value);
  if (!folded)
    return std::nullopt;
  return *folded != 0;
}

inline bool isMulOf(Value value, Value lhsExpected, Value rhsExpected) {
  value = ValueAnalysis::stripClampOne(value);
  lhsExpected = ValueAnalysis::stripClampOne(lhsExpected);
  rhsExpected = ValueAnalysis::stripClampOne(rhsExpected);

  auto mul = value.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;

  Value lhs = ValueAnalysis::stripClampOne(mul.getLhs());
  Value rhs = ValueAnalysis::stripClampOne(mul.getRhs());
  return (ValueAnalysis::sameValue(lhs, lhsExpected) &&
          ValueAnalysis::sameValue(rhs, rhsExpected)) ||
         (ValueAnalysis::sameValue(lhs, rhsExpected) &&
          ValueAnalysis::sameValue(rhs, lhsExpected));
}

inline Value canonicalizeStartBlock(Value startBlock, Value blockSize) {
  Value sb = ValueAnalysis::stripClampOne(startBlock);
  Value bs = ValueAnalysis::stripClampOne(blockSize);
  auto div = sb.getDefiningOp<arith::DivUIOp>();
  if (!div)
    return sb;

  Value lhs = ValueAnalysis::stripClampOne(div.getLhs());
  Value rhs = ValueAnalysis::stripClampOne(div.getRhs());
  if (!ValueAnalysis::sameValue(rhs, bs))
    return sb;

  auto mul = lhs.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return sb;

  Value mulLhs = ValueAnalysis::stripClampOne(mul.getLhs());
  Value mulRhs = ValueAnalysis::stripClampOne(mul.getRhs());
  if (ValueAnalysis::sameValue(mulLhs, bs))
    return mulRhs;
  if (ValueAnalysis::sameValue(mulRhs, bs))
    return mulLhs;
  return sb;
}

inline bool isAddOf(Value value, Value lhsExpected, Value rhsExpected) {
  value = ValueAnalysis::stripNumericCasts(value);
  lhsExpected = ValueAnalysis::stripClampOne(lhsExpected);
  rhsExpected = ValueAnalysis::stripClampOne(rhsExpected);
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    Value lhs = ValueAnalysis::stripClampOne(add.getLhs());
    Value rhs = ValueAnalysis::stripClampOne(add.getRhs());
    return (ValueAnalysis::sameValue(lhs, lhsExpected) &&
            ValueAnalysis::sameValue(rhs, rhsExpected)) ||
           (ValueAnalysis::sameValue(lhs, rhsExpected) &&
            ValueAnalysis::sameValue(rhs, lhsExpected));
  }
  return false;
}

inline bool matchesBasePlusBlockSpan(Value candidate, Value base,
                                     Value blockSize) {
  candidate = ValueAnalysis::stripNumericCasts(candidate);
  base = ValueAnalysis::stripNumericCasts(base);
  blockSize = ValueAnalysis::stripClampOne(blockSize);

  if (isAddOf(candidate, base, blockSize))
    return true;

  auto blockSizeConst = tryFoldConstantIndex(blockSize);
  if (!blockSizeConst)
    return false;

  int64_t candidateConst = 0;
  int64_t baseConst = 0;
  Value candidateBase =
      ValueAnalysis::stripConstantOffset(candidate, &candidateConst);
  Value normalizedBase = ValueAnalysis::stripConstantOffset(base, &baseConst);

  candidateBase = ValueAnalysis::stripNumericCasts(candidateBase);
  normalizedBase = ValueAnalysis::stripNumericCasts(normalizedBase);

  if (!ValueAnalysis::sameValue(candidateBase, normalizedBase))
    return false;

  return (candidateConst - baseConst) == *blockSizeConst;
}

} // namespace detail

/// Match `iv` or `iv + invariant_base` after stripping numeric casts.
/// Returns `baseIndex = nullptr` when the expression is exactly the IV.
inline bool matchLoopInvariantAddend(Value value, Value iv, Value &baseIndex) {
  value = ValueAnalysis::stripNumericCasts(value);
  iv = ValueAnalysis::stripNumericCasts(iv);

  if (ValueAnalysis::sameValue(value, iv)) {
    baseIndex = Value();
    return true;
  }

  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  Value lhs = ValueAnalysis::stripNumericCasts(add.getLhs());
  Value rhs = ValueAnalysis::stripNumericCasts(add.getRhs());
  if (ValueAnalysis::sameValue(lhs, iv) && !ValueAnalysis::dependsOn(rhs, iv)) {
    baseIndex = rhs;
    return true;
  }
  if (ValueAnalysis::sameValue(rhs, iv) && !ValueAnalysis::dependsOn(lhs, iv)) {
    baseIndex = lhs;
    return true;
  }
  return false;
}

/// Return the local in-block expression when the global index already has the
/// shape `startBlock * blockSize + local`.
inline std::optional<Value> extractLocalFromBlockBase(Value globalIndex,
                                                      Value startBlock,
                                                      Value blockSize) {
  Value idx = ValueAnalysis::stripNumericCasts(globalIndex);
  Value sb = detail::canonicalizeStartBlock(startBlock, blockSize);

  auto add = idx.getDefiningOp<arith::AddIOp>();
  if (!add)
    return std::nullopt;

  Value lhs = ValueAnalysis::stripNumericCasts(add.getLhs());
  Value rhs = ValueAnalysis::stripNumericCasts(add.getRhs());
  if (detail::isMulOf(lhs, sb, blockSize))
    return rhs;
  if (detail::isMulOf(rhs, sb, blockSize))
    return lhs;
  return std::nullopt;
}

/// Check whether the value is known to be non-negative.
inline bool isKnownNonNegative(Value value, unsigned depth = 0) {
  if (!value || depth > 8)
    return false;
  value = ValueAnalysis::stripNumericCasts(value);

  if (auto folded = tryFoldConstantIndex(value))
    return *folded >= 0;

  if (auto barg = dyn_cast<BlockArgument>(value)) {
    auto forOp = dyn_cast_or_null<scf::ForOp>(barg.getOwner()->getParentOp());
    if (forOp && forOp.getInductionVar() == barg &&
        isKnownNonNegative(forOp.getLowerBound(), depth + 1) &&
        ValueAnalysis::isConstantAtLeastOne(forOp.getStep()))
      return true;
  }

  if (auto maxOp = value.getDefiningOp<arith::MaxUIOp>())
    return ValueAnalysis::isZeroConstant(maxOp.getLhs()) ||
           ValueAnalysis::isZeroConstant(maxOp.getRhs()) ||
           (isKnownNonNegative(maxOp.getLhs(), depth + 1) &&
            isKnownNonNegative(maxOp.getRhs(), depth + 1));

  if (auto addOp = value.getDefiningOp<arith::AddIOp>())
    return isKnownNonNegative(addOp.getLhs(), depth + 1) &&
           isKnownNonNegative(addOp.getRhs(), depth + 1);

  if (auto mulOp = value.getDefiningOp<arith::MulIOp>()) {
    Value lhs = ValueAnalysis::stripNumericCasts(mulOp.getLhs());
    Value rhs = ValueAnalysis::stripNumericCasts(mulOp.getRhs());
    if (auto lhsConst = tryFoldConstantIndex(lhs))
      return *lhsConst >= 0 && isKnownNonNegative(rhs, depth + 1);
    if (auto rhsConst = tryFoldConstantIndex(rhs))
      return *rhsConst >= 0 && isKnownNonNegative(lhs, depth + 1);
    return isKnownNonNegative(lhs, depth + 1) &&
           isKnownNonNegative(rhs, depth + 1);
  }

  if (value.getDefiningOp() &&
      isa<arith::DivUIOp, arith::RemUIOp>(value.getDefiningOp()))
    return true;

  auto select = value.getDefiningOp<arith::SelectOp>();
  if (!select)
    return false;

  if (auto cond = detail::tryFoldBool(select.getCondition()))
    return isKnownNonNegative(
        *cond ? select.getTrueValue() : select.getFalseValue(), depth + 1);

  Value trueValue = ValueAnalysis::stripNumericCasts(select.getTrueValue());
  Value falseValue = ValueAnalysis::stripNumericCasts(select.getFalseValue());
  if (!ValueAnalysis::isZeroConstant(trueValue))
    return isKnownNonNegative(trueValue, depth + 1) &&
           isKnownNonNegative(falseValue, depth + 1);

  auto cmp = ValueAnalysis::stripNumericCasts(select.getCondition())
                 .getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return false;

  Value lhs = ValueAnalysis::stripNumericCasts(cmp.getLhs());
  Value rhs = ValueAnalysis::stripNumericCasts(cmp.getRhs());
  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return ValueAnalysis::sameValue(lhs, falseValue) &&
           ValueAnalysis::isZeroConstant(rhs);
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return ValueAnalysis::sameValue(rhs, falseValue) &&
           ValueAnalysis::isZeroConstant(lhs);
  default:
    return false;
  }
}

/// Check whether the value is known to be non-positive.
inline bool isKnownNonPositive(Value value, unsigned depth = 0) {
  if (!value || depth > 8)
    return false;
  value = ValueAnalysis::stripNumericCasts(value);

  if (auto folded = tryFoldConstantIndex(value))
    return *folded <= 0;

  if (auto subOp = value.getDefiningOp<arith::SubIOp>())
    return ValueAnalysis::isZeroConstant(subOp.getLhs()) &&
           isKnownNonNegative(subOp.getRhs(), depth + 1);

  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    if (auto cond = detail::tryFoldBool(select.getCondition()))
      return isKnownNonPositive(
          *cond ? select.getTrueValue() : select.getFalseValue(), depth + 1);
    return isKnownNonPositive(select.getTrueValue(), depth + 1) &&
           isKnownNonPositive(select.getFalseValue(), depth + 1);
  }

  return false;
}

/// Prove that the value is bounded by the block span.
inline bool isValueBoundedByBlockSpan(Value value, Value blockSize,
                                      unsigned depth = 0) {
  if (!value || !blockSize || depth > 8)
    return false;

  value = ValueAnalysis::stripNumericCasts(value);
  blockSize = ValueAnalysis::stripClampOne(blockSize);

  if (ValueAnalysis::sameValue(value, blockSize))
    return true;

  auto valueConst = tryFoldConstantIndex(value);
  auto blockConst = tryFoldConstantIndex(blockSize);
  if (valueConst && blockConst)
    return *valueConst <= *blockConst;

  auto valueHint = arts::DbUtils::extractBlockSizeFromHint(value);
  auto blockHint = arts::DbUtils::extractBlockSizeFromHint(blockSize);
  if (valueHint && blockHint)
    return *valueHint <= *blockHint;

  if (auto minOp = value.getDefiningOp<arith::MinUIOp>())
    return isValueBoundedByBlockSpan(minOp.getLhs(), blockSize, depth + 1) ||
           isValueBoundedByBlockSpan(minOp.getRhs(), blockSize, depth + 1);

  if (auto selectOp = value.getDefiningOp<arith::SelectOp>()) {
    if (auto cond = detail::tryFoldBool(selectOp.getCondition())) {
      return isValueBoundedByBlockSpan(*cond ? selectOp.getTrueValue()
                                             : selectOp.getFalseValue(),
                                       blockSize, depth + 1);
    }
    return isValueBoundedByBlockSpan(selectOp.getTrueValue(), blockSize,
                                     depth + 1) &&
           isValueBoundedByBlockSpan(selectOp.getFalseValue(), blockSize,
                                     depth + 1);
  }

  /// Owned stencil slices commonly materialize as
  /// `haloExpandedSpan - haloWidth`. If the expanded span is already known to
  /// fit within the block and the subtrahend is non-negative, the result must
  /// also fit within the same block.
  if (auto subOp = value.getDefiningOp<arith::SubIOp>()) {
    if (isValueBoundedByBlockSpan(subOp.getLhs(), blockSize, depth + 1) &&
        isKnownNonNegative(subOp.getRhs(), depth + 1))
      return true;
  }

  if (auto mulOp = value.getDefiningOp<arith::MulIOp>()) {
    Value lhs = ValueAnalysis::stripClampOne(mulOp.getLhs());
    Value rhs = ValueAnalysis::stripClampOne(mulOp.getRhs());
    auto lhsConst = tryFoldConstantIndex(lhs);
    auto rhsConst = tryFoldConstantIndex(rhs);
    if (ValueAnalysis::sameValue(lhs, blockSize) && rhsConst)
      return *rhsConst >= 0 && *rhsConst <= 1;
    if (ValueAnalysis::sameValue(rhs, blockSize) && lhsConst)
      return *lhsConst >= 0 && *lhsConst <= 1;
  }

  return false;
}

/// Normalize common owned-slice extent expressions that show up after blocked
/// lowering, such as `maxui(blockSize, 1)` and redundant `minui` nesting from
/// tail-clamp rewrites. This keeps later proofs mode-agnostic while accepting
/// equivalent IR shapes.
inline Value normalizeOwnedSliceExtent(Value value, unsigned depth = 0) {
  if (!value || depth > 8)
    return value;

  value = ValueAnalysis::stripNumericCasts(value);
  value = ValueAnalysis::stripClampOne(value);
  value = ValueAnalysis::stripSelectClamp(value);
  value = ValueAnalysis::stripClampOne(value);

  auto normalizeMin = [&](arith::MinUIOp minOp) -> Value {
    Value lhs = normalizeOwnedSliceExtent(minOp.getLhs(), depth + 1);
    Value rhs = normalizeOwnedSliceExtent(minOp.getRhs(), depth + 1);
    auto same = [&](Value a, Value b) {
      return ValueAnalysis::sameValue(a, b) ||
             ValueAnalysis::areValuesEquivalent(a, b);
    };

    if (same(lhs, rhs))
      return lhs;

    if (auto nested = rhs.getDefiningOp<arith::MinUIOp>()) {
      Value nestedLhs = normalizeOwnedSliceExtent(nested.getLhs(), depth + 1);
      Value nestedRhs = normalizeOwnedSliceExtent(nested.getRhs(), depth + 1);
      if (same(lhs, nestedLhs) || same(lhs, nestedRhs))
        return rhs;
    }

    if (auto nested = lhs.getDefiningOp<arith::MinUIOp>()) {
      Value nestedLhs = normalizeOwnedSliceExtent(nested.getLhs(), depth + 1);
      Value nestedRhs = normalizeOwnedSliceExtent(nested.getRhs(), depth + 1);
      if (same(rhs, nestedLhs) || same(rhs, nestedRhs))
        return lhs;
    }

    return value;
  };

  if (auto minOp = value.getDefiningOp<arith::MinUIOp>())
    return normalizeMin(minOp);

  return value;
}

inline bool areEquivalentOwnedSliceExtents(Value lhs, Value rhs) {
  if (!lhs || !rhs)
    return false;

  lhs = normalizeOwnedSliceExtent(lhs);
  rhs = normalizeOwnedSliceExtent(rhs);
  return ValueAnalysis::sameValue(lhs, rhs) ||
         ValueAnalysis::areValuesEquivalent(lhs, rhs);
}

/// Check whether `idx` is the IV of a loop whose upper bound is clamped by the
/// given block size. This is enough to prove the local slice stays within one
/// block.
inline bool isLoopIvBoundedBy(Value idx, Value blockSize) {
  auto barg = dyn_cast<BlockArgument>(idx);
  if (!barg)
    return false;

  auto forOp = dyn_cast_or_null<scf::ForOp>(barg.getOwner()->getParentOp());
  if (!forOp || forOp.getInductionVar() != barg)
    return false;

  Value lb = ValueAnalysis::stripNumericCasts(forOp.getLowerBound());
  Value ub = ValueAnalysis::stripClampOne(forOp.getUpperBound());
  Value bs = ValueAnalysis::stripClampOne(blockSize);

  if (!isKnownNonNegative(lb))
    return false;
  return isValueBoundedByBlockSpan(ub, bs);
}

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_UTILS_BLOCKEDACCESSUTILS_H
