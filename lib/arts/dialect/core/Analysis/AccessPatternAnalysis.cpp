///==========================================================================///
/// File: AccessPatternAnalysis.cpp
///
/// Shared loop-relative memory access bounds analysis.
///==========================================================================///

#include "arts/dialect/core/Analysis/AccessPatternAnalysis.h"
#include "arts/Dialect.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::arts;

namespace {

static constexpr int64_t kMaxSupportedLoopOffsetSpan = 16;

struct RelativeOffsetRange {
  int64_t minOffset = 0;
  int64_t maxOffset = 0;
  bool hasBase = false;
};

static bool matchesBasePattern(Value value, Value loopIV, Value blockBase) {
  if (!value || !loopIV || !blockBase)
    return false;

  auto addOp =
      ValueAnalysis::stripNumericCasts(value).getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return false;

  Value lhs = ValueAnalysis::stripNumericCasts(addOp.getLhs());
  Value rhs = ValueAnalysis::stripNumericCasts(addOp.getRhs());
  return (ValueAnalysis::areValuesEquivalent(lhs, loopIV) &&
          ValueAnalysis::areValuesEquivalent(rhs, blockBase)) ||
         (ValueAnalysis::areValuesEquivalent(lhs, blockBase) &&
          ValueAnalysis::areValuesEquivalent(rhs, loopIV));
}

static std::optional<RelativeOffsetRange> getSmallLoopOffsetRange(Value value) {
  auto blockArg =
      dyn_cast_or_null<BlockArgument>(ValueAnalysis::stripNumericCasts(value));
  if (!blockArg)
    return std::nullopt;

  auto buildRange = [&](Value lbVal, Value ubVal,
                        Value stepVal) -> std::optional<RelativeOffsetRange> {
    auto lb = ValueAnalysis::tryFoldConstantIndex(lbVal);
    auto ub = ValueAnalysis::tryFoldConstantIndex(ubVal);
    auto step = ValueAnalysis::tryFoldConstantIndex(stepVal);
    if (!lb || !ub || !step || *step == 0)
      return std::nullopt;

    int64_t minVal = *lb;
    int64_t maxVal = *lb;
    if (*step > 0) {
      if (*lb >= *ub)
        return std::nullopt;
      maxVal = *ub - *step;
    } else {
      if (*lb <= *ub)
        return std::nullopt;
      minVal = *ub - *step;
    }

    if (maxVal < minVal)
      std::swap(minVal, maxVal);
    if (maxVal - minVal > kMaxSupportedLoopOffsetSpan)
      return std::nullopt;

    return RelativeOffsetRange{minVal, maxVal, /*hasBase=*/false};
  };

  Operation *parentOp = blockArg.getOwner()->getParentOp();
  if (!parentOp)
    return std::nullopt;

  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
    if (blockArg != forOp.getInductionVar())
      return std::nullopt;
    return buildRange(forOp.getLowerBound(), forOp.getUpperBound(),
                      forOp.getStep());
  }

  if (auto parallelOp = dyn_cast<scf::ParallelOp>(parentOp)) {
    for (auto [idx, iv] : llvm::enumerate(parallelOp.getInductionVars())) {
      if (blockArg != iv)
        continue;
      return buildRange(parallelOp.getLowerBound()[idx],
                        parallelOp.getUpperBound()[idx],
                        parallelOp.getStep()[idx]);
    }
    return std::nullopt;
  }

  return std::nullopt;
}

static std::optional<RelativeOffsetRange>
extractRelativeOffsetRange(Value value, Value loopIV, Value blockBase,
                           unsigned depth = 0) {
  if (!value || !loopIV || !blockBase || depth > 8)
    return std::nullopt;

  value =
      ValueAnalysis::stripNumericCasts(ValueAnalysis::stripSelectClamp(value));
  loopIV = ValueAnalysis::stripNumericCasts(loopIV);
  blockBase = ValueAnalysis::stripNumericCasts(blockBase);

  if (ValueAnalysis::areValuesEquivalent(value, loopIV) ||
      ValueAnalysis::areValuesEquivalent(value, blockBase) ||
      matchesBasePattern(value, loopIV, blockBase))
    return RelativeOffsetRange{0, 0, /*hasBase=*/true};

  if (auto constant = ValueAnalysis::tryFoldConstantIndex(value))
    return RelativeOffsetRange{*constant, *constant, /*hasBase=*/false};

  if (auto loopRange = getSmallLoopOffsetRange(value))
    return loopRange;

  auto combine = [](const RelativeOffsetRange &lhs,
                    const RelativeOffsetRange &rhs, bool subtractRhs) {
    RelativeOffsetRange result;
    result.hasBase = lhs.hasBase || rhs.hasBase;
    if (subtractRhs) {
      result.minOffset = lhs.minOffset - rhs.maxOffset;
      result.maxOffset = lhs.maxOffset - rhs.minOffset;
    } else {
      result.minOffset = lhs.minOffset + rhs.minOffset;
      result.maxOffset = lhs.maxOffset + rhs.maxOffset;
    }
    return result;
  };

  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs =
        extractRelativeOffsetRange(add.getLhs(), loopIV, blockBase, depth + 1);
    auto rhs =
        extractRelativeOffsetRange(add.getRhs(), loopIV, blockBase, depth + 1);
    if (lhs && rhs)
      return combine(*lhs, *rhs, /*subtractRhs=*/false);
    return std::nullopt;
  }

  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    auto lhs =
        extractRelativeOffsetRange(sub.getLhs(), loopIV, blockBase, depth + 1);
    auto rhs =
        extractRelativeOffsetRange(sub.getRhs(), loopIV, blockBase, depth + 1);
    if (lhs && rhs)
      return combine(*lhs, *rhs, /*subtractRhs=*/true);
    return std::nullopt;
  }

  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    auto lhsConst = ValueAnalysis::tryFoldConstantIndex(mul.getLhs());
    auto rhsConst = ValueAnalysis::tryFoldConstantIndex(mul.getRhs());
    if (lhsConst) {
      auto rhs = extractRelativeOffsetRange(mul.getRhs(), loopIV, blockBase,
                                            depth + 1);
      if (!rhs)
        return std::nullopt;
      int64_t minVal = rhs->minOffset * *lhsConst;
      int64_t maxVal = rhs->maxOffset * *lhsConst;
      if (maxVal < minVal)
        std::swap(minVal, maxVal);
      return RelativeOffsetRange{minVal, maxVal, rhs->hasBase};
    }
    if (rhsConst) {
      auto lhs = extractRelativeOffsetRange(mul.getLhs(), loopIV, blockBase,
                                            depth + 1);
      if (!lhs)
        return std::nullopt;
      int64_t minVal = lhs->minOffset * *rhsConst;
      int64_t maxVal = lhs->maxOffset * *rhsConst;
      if (maxVal < minVal)
        std::swap(minVal, maxVal);
      return RelativeOffsetRange{minVal, maxVal, lhs->hasBase};
    }
    return std::nullopt;
  }

  return std::nullopt;
}

/// Split blocked accesses often represent one logical owner-dimension index as
/// a db_ref selector plus an in-block remainder:
///   selector ~= divui(globalIdx, blockSpan)
///   localIdx  = remui(globalIdx, blockSpan)
///
/// Access-bound analysis cares about the original logical index (`globalIdx`),
/// not the implementation split. Recovering that dividend keeps stencil/halo
/// facts stable across blocked DB materialization and later rewrites.
static Value peelSelectorDividend(Value value, Value divisor,
                                  unsigned depth = 0) {
  if (!value || !divisor || depth > 8)
    return Value();

  value =
      ValueAnalysis::stripNumericCasts(ValueAnalysis::stripSelectClamp(value));
  divisor = ValueAnalysis::stripNumericCasts(divisor);

  auto divisorMatches = [&](Value candidate) {
    candidate = ValueAnalysis::stripNumericCasts(candidate);
    return ValueAnalysis::sameValue(candidate, divisor) ||
           ValueAnalysis::areValuesEquivalent(candidate, divisor);
  };

  if (auto div = value.getDefiningOp<arith::DivUIOp>())
    if (divisorMatches(div.getRhs()))
      return div.getLhs();
  if (auto div = value.getDefiningOp<arith::DivSIOp>())
    if (divisorMatches(div.getRhs()))
      return div.getLhs();

  auto recurseBinary = [&](Value lhs, Value rhs) -> Value {
    if (Value recovered = peelSelectorDividend(lhs, divisor, depth + 1))
      return recovered;
    return peelSelectorDividend(rhs, divisor, depth + 1);
  };

  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return recurseBinary(add.getLhs(), add.getRhs());
  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    return peelSelectorDividend(sub.getLhs(), divisor, depth + 1);
  if (auto min = value.getDefiningOp<arith::MinUIOp>())
    return recurseBinary(min.getLhs(), min.getRhs());
  if (auto min = value.getDefiningOp<arith::MinSIOp>())
    return recurseBinary(min.getLhs(), min.getRhs());
  if (auto max = value.getDefiningOp<arith::MaxUIOp>())
    return recurseBinary(max.getLhs(), max.getRhs());
  if (auto max = value.getDefiningOp<arith::MaxSIOp>())
    return recurseBinary(max.getLhs(), max.getRhs());
  if (auto select = value.getDefiningOp<arith::SelectOp>())
    return recurseBinary(select.getTrueValue(), select.getFalseValue());

  return Value();
}

static Value recoverLogicalMemrefIndex(const AccessIndexInfo &access,
                                       unsigned memrefDim) {
  if (access.dbRefPrefix == 0)
    return Value();

  unsigned chainIdx = access.dbRefPrefix + memrefDim;
  if (chainIdx >= access.indexChain.size())
    return Value();

  Value localIdx =
      ValueAnalysis::stripNumericCasts(access.indexChain[chainIdx]);
  Value dividend;
  Value divisor;
  if (auto rem = localIdx.getDefiningOp<arith::RemUIOp>()) {
    dividend = rem.getLhs();
    divisor = rem.getRhs();
  } else if (auto rem = localIdx.getDefiningOp<arith::RemSIOp>()) {
    dividend = rem.getLhs();
    divisor = rem.getRhs();
  } else {
    return Value();
  }

  for (unsigned i = 0; i < access.dbRefPrefix; ++i) {
    Value selectorDividend =
        peelSelectorDividend(access.indexChain[i], divisor);
    if (!selectorDividend)
      continue;
    if (ValueAnalysis::sameValue(selectorDividend, dividend) ||
        ValueAnalysis::areValuesEquivalent(selectorDividend, dividend))
      return dividend;
  }

  return Value();
}

static Value getMemrefIndexForBounds(const AccessIndexInfo &access,
                                     unsigned memrefDim) {
  if (Value logicalIdx = recoverLogicalMemrefIndex(access, memrefDim))
    return logicalIdx;

  unsigned chainIdx = access.dbRefPrefix + memrefDim;
  if (chainIdx >= access.indexChain.size())
    return Value();
  return access.indexChain[chainIdx];
}

} // namespace

/// Normalize stencil bounds to center them around zero.
///
/// This is useful for uniformly shifted stencils. For example:
///   - Original bounds: [1, 3] (accesses A[i+1], A[i+2], A[i+3])
///   - Normalized:      [-1, 1] with centerOffset=2
///
/// This normalization simplifies halo computation by expressing the pattern
/// as symmetric offsets from a shifted base.
void arts::normalizeAccessBounds(AccessBoundsResult &bounds) {
  if (!bounds.valid || !bounds.isStencil)
    return;

  /// Already centered around zero (e.g., [-1, 1] or [-2, 2])
  if (bounds.minOffset < 0 && bounds.maxOffset > 0)
    return;

  /// Compute center offset: (min + max) / 2
  /// Only normalize if the sum is even (ensures integer center)
  int64_t sum = bounds.minOffset + bounds.maxOffset;
  if (sum % 2 != 0)
    return;

  int64_t center = sum / 2;
  if (center == 0)
    return;

  /// Shift bounds to center them around zero
  bounds.centerOffset = center;
  bounds.minOffset -= center;
  bounds.maxOffset -= center;
}

/// Analyze memory access bounds relative to loop induction variable and block
/// base.
///
/// This function extracts constant offsets from memory access index chains to
/// determine the access pattern (uniform vs. stencil) and compute halo bounds.
///
/// Algorithm:
///   1. For each access, find the index that depends on loopIV/blockBase
///   2. Try to extract constant offset from that index (e.g., iv+1, iv-2)
///   3. Track min/max offsets across all accesses
///   4. Classify as stencil if minOffset != maxOffset
///
/// Example:
///   Accesses: A[i-1], A[i], A[i+1]
///   Result: minOffset=-1, maxOffset=1, isStencil=true
///
/// Parameters:
///   - accesses: Collection of index chains from memory operations
///   - loopIV: Loop induction variable to analyze relative to
///   - blockBase: Block base value (may be same as loopIV)
///   - partitionDim: Optional dimension to focus on (for multi-dim analysis)
///
/// Returns: AccessBoundsResult with offset bounds and pattern classification
AccessBoundsResult
arts::analyzeAccessBoundsFromIndices(ArrayRef<AccessIndexInfo> accesses,
                                     Value loopIV, Value blockBase,
                                     std::optional<unsigned> partitionDim) {
  AccessBoundsResult bounds;
  bounds.minOffset = std::numeric_limits<int64_t>::max();
  bounds.maxOffset = std::numeric_limits<int64_t>::min();

  /// Validate inputs
  if (!loopIV || !blockBase)
    return bounds;

  /// Empty access list is valid (uniform access with offset 0)
  if (accesses.empty()) {
    bounds.minOffset = 0;
    bounds.maxOffset = 0;
    bounds.valid = true;
    return bounds;
  }

  bool foundAny = false;
  for (const AccessIndexInfo &access : accesses) {
    if (access.indexChain.empty())
      continue;

    /// Select which index to analyze for offset extraction.
    /// Priority:
    ///   1. Specific partition dimension (if provided)
    ///   2. First index that depends on loopIV/blockBase
    ///   3. First non-constant index (as a fallback)
    Value idxForBounds;

    /// Priority 1: Use specific dimension if analyzing multi-dim accesses
    if (partitionDim)
      idxForBounds = getMemrefIndexForBounds(access, *partitionDim);

    /// Priority 2: Find index that depends on our analysis base
    /// (loopIV/blockBase)
    if (!idxForBounds) {
      for (unsigned i = access.dbRefPrefix; i < access.indexChain.size(); ++i) {
        Value idx = getMemrefIndexForBounds(access, i - access.dbRefPrefix);
        if (ValueAnalysis::dependsOn(idx, loopIV) ||
            ValueAnalysis::dependsOn(idx, blockBase)) {
          idxForBounds = idx;
          break;
        }
      }
    }

    /// Prefix db_ref selectors are a fallback only after we fail to recover a
    /// logical memref-dimension index. Using them first on blocked accesses can
    /// undercount halos by observing only the chunk selector delta.
    if (!idxForBounds) {
      for (unsigned i = 0;
           i < access.dbRefPrefix && i < access.indexChain.size(); ++i) {
        Value idx = access.indexChain[i];
        if (ValueAnalysis::dependsOn(idx, loopIV) ||
            ValueAnalysis::dependsOn(idx, blockBase)) {
          idxForBounds = idx;
          break;
        }
      }
    }

    /// Priority 3: Use any non-constant index as fallback
    if (!idxForBounds) {
      for (unsigned i = access.dbRefPrefix; i < access.indexChain.size(); ++i) {
        Value idx = getMemrefIndexForBounds(access, i - access.dbRefPrefix);
        int64_t constVal = 0;
        if (!ValueAnalysis::getConstantIndex(idx, constVal)) {
          idxForBounds = idx;
          break;
        }
      }
    }

    if (!idxForBounds) {
      for (Value idx : access.indexChain) {
        int64_t constVal = 0;
        if (!ValueAnalysis::getConstantIndex(idx, constVal)) {
          idxForBounds = idx;
          break;
        }
      }
    }

    /// Skip this access if we couldn't find a suitable index
    if (!idxForBounds)
      continue;

    /// Try to extract constant offset from the selected index
    /// Handles patterns like: iv, iv+c, iv-c, blockBase+c, etc.
    auto constOffset =
        ValueAnalysis::extractConstantOffset(idxForBounds, loopIV, blockBase);
    if (constOffset) {
      foundAny = true;
      bounds.minOffset = std::min(bounds.minOffset, *constOffset);
      bounds.maxOffset = std::max(bounds.maxOffset, *constOffset);
      continue;
    }

    if (auto offsetRange =
            extractRelativeOffsetRange(idxForBounds, loopIV, blockBase);
        offsetRange && offsetRange->hasBase) {
      foundAny = true;
      bounds.minOffset = std::min(bounds.minOffset, offsetRange->minOffset);
      bounds.maxOffset = std::max(bounds.maxOffset, offsetRange->maxOffset);
      continue;
    }

    /// If we couldn't extract a constant offset but the index depends on
    /// both loopIV and blockBase, it's a variable offset pattern.
    /// This indicates indirect or complex affine access that we can't
    /// statically analyze. Mark as variable and return early.
    if (ValueAnalysis::dependsOn(idxForBounds, loopIV) &&
        ValueAnalysis::dependsOn(idxForBounds, blockBase)) {
      bounds.hasVariableOffset = true;
      return bounds;
    }
  }

  /// Finalize bounds if we found any constant offsets
  if (foundAny) {
    /// Classify as stencil if we found multiple distinct offsets
    bounds.isStencil = (bounds.minOffset != bounds.maxOffset);
    bounds.valid = true;
    /// Normalize to center around zero for easier halo computation
    normalizeAccessBounds(bounds);
  }

  return bounds;
}
