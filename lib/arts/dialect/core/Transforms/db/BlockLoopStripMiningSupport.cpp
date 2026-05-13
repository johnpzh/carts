///==========================================================================///
/// File: BlockLoopStripMiningSupport.cpp
///
/// Shared local helpers for the split BlockLoopStripMining implementation.
///==========================================================================///

#include "arts/dialect/core/Transforms/BlockLoopStripMiningInternal.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/OperationAttributes.h"

ARTS_DEBUG_SETUP(block_loop_strip_mining);

using namespace mlir;
using namespace mlir::arts;

namespace mlir::arts::block_loop_strip_mining {

const llvm::StringLiteral kStripMiningGeneratedAttr =
    AttrNames::Operation::StripMiningGenerated;

bool isGeneratedByStripMining(scf::ForOp loop) {
  return loop->hasAttr(AttrNames::Operation::StripMiningGenerated);
}

void markGeneratedByStripMining(scf::ForOp loop) {
  loop->setAttr(AttrNames::Operation::StripMiningGenerated,
                UnitAttr::get(loop.getContext()));
}

void clearGeneratedStripMiningMarks(func::FuncOp func) {
  func.walk([](scf::ForOp loop) {
    if (loop->hasAttr(AttrNames::Operation::StripMiningGenerated))
      loop->removeAttr(AttrNames::Operation::StripMiningGenerated);
  });
}

static std::optional<int64_t> resolveBlockSizeHint(Value value) {
  if (!value)
    return std::nullopt;
  if (auto folded = ValueAnalysis::getConstantIndexStripped(value))
    return folded;
  return DbUtils::extractBlockSizeFromHint(value);
}

static bool sameBlockSizeFamily(Value lhsValue, std::optional<int64_t> lhsConst,
                                Value rhsValue,
                                std::optional<int64_t> rhsConst) {
  lhsValue = ValueAnalysis::stripNumericCasts(lhsValue);
  rhsValue = ValueAnalysis::stripNumericCasts(rhsValue);

  if (ValueAnalysis::sameValue(lhsValue, rhsValue) ||
      ValueAnalysis::areValuesEquivalent(lhsValue, rhsValue))
    return true;

  auto lhsResolved = lhsConst ? lhsConst : resolveBlockSizeHint(lhsValue);
  auto rhsResolved = rhsConst ? rhsConst : resolveBlockSizeHint(rhsValue);
  return lhsResolved && rhsResolved && *lhsResolved == *rhsResolved;
}

std::optional<Value> extractInvariantOffset(Value lhs, Value iv) {
  lhs = ValueAnalysis::stripNumericCasts(lhs);
  if (lhs == iv)
    return Value();
  if (auto addOp = lhs.getDefiningOp<arith::AddIOp>()) {
    Value lhsOp = ValueAnalysis::stripNumericCasts(addOp.getLhs());
    Value rhsOp = ValueAnalysis::stripNumericCasts(addOp.getRhs());
    if (lhsOp == iv && !ValueAnalysis::dependsOn(rhsOp, iv))
      return rhsOp;
    if (rhsOp == iv && !ValueAnalysis::dependsOn(lhsOp, iv))
      return lhsOp;
  }
  return std::nullopt;
}

bool mergeInvariantBase(Value &currentBase, Value candidateBase) {
  candidateBase = ValueAnalysis::stripNumericCasts(candidateBase);
  if (!candidateBase)
    return true;
  if (!currentBase) {
    currentBase = candidateBase;
    return true;
  }
  return ValueAnalysis::areValuesEquivalent(currentBase, candidateBase);
}

/// Match expressions of the form:
///   iv
///   iv + invariant_base
///   iv + invariant_base +/- constant
/// and recover both the loop-invariant base and the small neighborhood offset
/// around that base. This keeps the transform generic for worker-local loops
/// where the IV is relative to an aligned chunk base instead of being the
/// global block index directly.
std::optional<NeighborhoodExprInfo> extractNeighborhoodExprInfo(Value value,
                                                                Value iv) {
  value = ValueAnalysis::stripNumericCasts(value);
  if (!value)
    return std::nullopt;
  if (ValueAnalysis::sameValue(value, iv))
    return NeighborhoodExprInfo{};

  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    Value lhs = ValueAnalysis::stripNumericCasts(add.getLhs());
    Value rhs = ValueAnalysis::stripNumericCasts(add.getRhs());

    if (auto lhsConst = ValueAnalysis::tryFoldConstantIndex(lhs)) {
      auto info = extractNeighborhoodExprInfo(rhs, iv);
      if (!info)
        return std::nullopt;
      info->offsetConst += *lhsConst;
      return info;
    }
    if (auto rhsConst = ValueAnalysis::tryFoldConstantIndex(rhs)) {
      auto info = extractNeighborhoodExprInfo(lhs, iv);
      if (!info)
        return std::nullopt;
      info->offsetConst += *rhsConst;
      return info;
    }

    if (!ValueAnalysis::dependsOn(lhs, iv)) {
      auto info = extractNeighborhoodExprInfo(rhs, iv);
      if (!info || !mergeInvariantBase(info->invariantBase, lhs))
        return std::nullopt;
      return info;
    }
    if (!ValueAnalysis::dependsOn(rhs, iv)) {
      auto info = extractNeighborhoodExprInfo(lhs, iv);
      if (!info || !mergeInvariantBase(info->invariantBase, rhs))
        return std::nullopt;
      return info;
    }
    return std::nullopt;
  }

  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    Value lhs = ValueAnalysis::stripNumericCasts(sub.getLhs());
    Value rhs = ValueAnalysis::stripNumericCasts(sub.getRhs());

    if (auto rhsConst = ValueAnalysis::tryFoldConstantIndex(rhs)) {
      auto info = extractNeighborhoodExprInfo(lhs, iv);
      if (!info)
        return std::nullopt;
      info->offsetConst -= *rhsConst;
      return info;
    }
    return std::nullopt;
  }

  return std::nullopt;
}

OffsetPatternGroup &getOrCreateOffsetGroup(NeighborhoodLoopInfo &info,
                                           int64_t offsetConst) {
  for (OffsetPatternGroup &group : info.offsetGroups) {
    if (group.offsetConst == offsetConst)
      return group;
  }
  info.offsetGroups.push_back(OffsetPatternGroup{offsetConst, {}, {}, {}});
  return info.offsetGroups.back();
}

std::optional<NeighborhoodPattern>
matchNeighborhoodPattern(Value lhs, Value rhs,
                         std::optional<arith::DivUIOp> divOp, Value remResult,
                         Value iv, ArrayRef<Operation *> patternOps) {
  auto exprInfo = extractNeighborhoodExprInfo(lhs, iv);
  if (!exprInfo)
    return std::nullopt;

  Value normalizedRhs = ValueAnalysis::stripNumericCasts(rhs);
  if (ValueAnalysis::dependsOn(normalizedRhs, iv))
    return std::nullopt;

  auto rhsConst = ValueAnalysis::tryFoldConstantIndex(normalizedRhs);
  if (rhsConst && *rhsConst <= 1)
    return std::nullopt;
  if (!rhsConst && !ValueAnalysis::isProvablyNonZero(normalizedRhs))
    return std::nullopt;

  if (rhsConst && std::abs(exprInfo->offsetConst) >= *rhsConst)
    return std::nullopt;

  NeighborhoodPattern pattern;
  pattern.offsetConst = exprInfo->offsetConst;
  pattern.invariantBase = exprInfo->invariantBase;
  pattern.blockSizeVal = normalizedRhs;
  pattern.blockSizeConst = rhsConst;
  pattern.divOp = divOp;
  pattern.remResult = remResult;
  llvm::append_range(pattern.patternOps, patternOps);
  return pattern;
}

NeighborhoodLoopInfo &getOrCreateNeighborhoodFamily(
    SmallVectorImpl<NeighborhoodLoopInfo> &families, Value blockSizeVal,
    std::optional<int64_t> blockSizeConst, Value invariantBase) {
  for (NeighborhoodLoopInfo &family : families) {
    bool sameBlockSize =
        sameBlockSizeFamily(family.blockSizeVal, family.blockSizeConst,
                            blockSizeVal, blockSizeConst);
    bool sameBase =
        (!family.invariantBase && !invariantBase) ||
        (family.invariantBase && invariantBase &&
         (ValueAnalysis::sameValue(family.invariantBase, invariantBase) ||
          ValueAnalysis::areValuesEquivalent(family.invariantBase,
                                             invariantBase)));
    if (sameBlockSize && sameBase)
      return family;
  }
  NeighborhoodLoopInfo family;
  family.blockSizeVal = blockSizeVal;
  family.blockSizeConst = blockSizeConst;
  family.invariantBase = invariantBase;
  families.push_back(std::move(family));
  return families.back();
}

void recordNeighborhoodPattern(NeighborhoodLoopInfo &info,
                               const NeighborhoodPattern &pattern) {
  OffsetPatternGroup &group = getOrCreateOffsetGroup(info, pattern.offsetConst);
  if (info.blockSizeConst) {
    ARTS_DEBUG(
        "Neighborhood match: offset="
        << pattern.offsetConst << " blockSize=" << *info.blockSizeConst
        << (info.invariantBase ? " invariant_base=yes" : " invariant_base=no"));
  } else {
    ARTS_DEBUG(
        "Neighborhood match: offset="
        << pattern.offsetConst << " blockSize=<dynamic>"
        << (info.invariantBase ? " invariant_base=yes" : " invariant_base=no"));
  }
  if (pattern.divOp)
    group.divOps.push_back(*pattern.divOp);
  if (pattern.remResult)
    group.remValues.push_back(pattern.remResult);
  llvm::append_range(group.remPatternOps, pattern.patternOps);
}

std::optional<NeighborhoodPattern>
matchNormalizedSignedRemainderNeighborhood(arith::SelectOp selectOp, Value iv) {
  auto rem = selectOp.getFalseValue().getDefiningOp<arith::RemSIOp>();
  if (!rem)
    return std::nullopt;
  auto cmp = selectOp.getCondition().getDefiningOp<arith::CmpIOp>();
  auto add = selectOp.getTrueValue().getDefiningOp<arith::AddIOp>();
  if (!cmp || !add)
    return std::nullopt;
  if (cmp.getPredicate() != arith::CmpIPredicate::slt)
    return std::nullopt;

  Value remResult = rem.getResult();
  if (!ValueAnalysis::sameValue(ValueAnalysis::stripNumericCasts(cmp.getLhs()),
                                remResult) ||
      !ValueAnalysis::isZeroConstant(
          ValueAnalysis::stripNumericCasts(cmp.getRhs())))
    return std::nullopt;

  Value addLhs = ValueAnalysis::stripNumericCasts(add.getLhs());
  Value addRhs = ValueAnalysis::stripNumericCasts(add.getRhs());
  Value rhs = ValueAnalysis::stripNumericCasts(rem.getRhs());
  if (!ValueAnalysis::sameValue(addLhs, remResult) ||
      !ValueAnalysis::sameValue(addRhs, rhs))
    return std::nullopt;

  SmallVector<Operation *> patternOps = {rem.getOperation(), cmp.getOperation(),
                                         add.getOperation(),
                                         selectOp.getOperation()};
  return matchNeighborhoodPattern(rem.getLhs(), rem.getRhs(), std::nullopt,
                                  selectOp.getResult(), iv, patternOps);
}

bool recordRemPattern(Value lhs, Value rhs, Value remResult, Value iv,
                      LoopBlockInfo &info, ArrayRef<Operation *> patternOps) {
  auto offsetOpt = extractInvariantOffset(lhs, iv);
  if (!offsetOpt)
    return false;
  Value offset = *offsetOpt;
  if (offset) {
    if (!info.offsetVal)
      info.offsetVal = offset;
    else if (!ValueAnalysis::sameValue(info.offsetVal, offset))
      return false;
  } else if (info.offsetVal) {
    return false;
  }

  Value normalizedRhs = ValueAnalysis::stripNumericCasts(rhs);
  if (ValueAnalysis::dependsOn(normalizedRhs, iv))
    return false;
  auto rhsConst = ValueAnalysis::tryFoldConstantIndex(normalizedRhs);
  if (rhsConst && *rhsConst <= 1)
    return false;
  if (!info.blockSizeVal) {
    info.blockSizeVal = normalizedRhs;
    info.blockSizeConst =
        rhsConst ? rhsConst : resolveBlockSizeHint(normalizedRhs);
  } else if (!sameBlockSizeFamily(info.blockSizeVal, info.blockSizeConst,
                                  normalizedRhs, rhsConst)) {
    if (!info.blockSizeConst || !rhsConst || *info.blockSizeConst != *rhsConst)
      return false;
  }

  info.remValues.push_back(remResult);
  llvm::append_range(info.remPatternOps, patternOps);
  return true;
}

bool matchNormalizedSignedRemainder(arith::SelectOp selectOp, Value iv,
                                    LoopBlockInfo &info) {
  auto rem = selectOp.getFalseValue().getDefiningOp<arith::RemSIOp>();
  if (!rem)
    return false;
  auto cmp = selectOp.getCondition().getDefiningOp<arith::CmpIOp>();
  auto add = selectOp.getTrueValue().getDefiningOp<arith::AddIOp>();
  if (!cmp || !add)
    return false;
  if (cmp.getPredicate() != arith::CmpIPredicate::slt)
    return false;

  Value remResult = rem.getResult();
  if (!ValueAnalysis::sameValue(ValueAnalysis::stripNumericCasts(cmp.getLhs()),
                                remResult) ||
      !ValueAnalysis::isZeroConstant(
          ValueAnalysis::stripNumericCasts(cmp.getRhs())))
    return false;

  Value addLhs = ValueAnalysis::stripNumericCasts(add.getLhs());
  Value addRhs = ValueAnalysis::stripNumericCasts(add.getRhs());
  Value rhs = ValueAnalysis::stripNumericCasts(rem.getRhs());
  if (!ValueAnalysis::sameValue(addLhs, remResult) ||
      !ValueAnalysis::sameValue(addRhs, rhs))
    return false;

  SmallVector<Operation *> patternOps = {rem.getOperation(), cmp.getOperation(),
                                         add.getOperation(),
                                         selectOp.getOperation()};
  return recordRemPattern(rem.getLhs(), rem.getRhs(), selectOp.getResult(), iv,
                          info, patternOps);
}

bool isAlignedOffset(Value offset, Value blockSize,
                     const std::optional<int64_t> &blockSizeConst,
                     Value *mulOther) {
  if (!offset)
    return true;
  Value off = ValueAnalysis::stripClampOne(offset);
  Value bs = ValueAnalysis::stripClampOne(blockSize);
  if (auto offConst = ValueAnalysis::tryFoldConstantIndex(off)) {
    if (*offConst == 0)
      return true;
    if (blockSizeConst && *blockSizeConst > 0)
      return (*offConst % *blockSizeConst) == 0;
    return false;
  }
  if (auto mul = off.getDefiningOp<arith::MulIOp>()) {
    Value lhs = ValueAnalysis::stripClampOne(mul.getLhs());
    Value rhs = ValueAnalysis::stripClampOne(mul.getRhs());
    if (ValueAnalysis::sameValue(lhs, bs)) {
      if (mulOther)
        *mulOther = rhs;
      return true;
    }
    if (ValueAnalysis::sameValue(rhs, bs)) {
      if (mulOther)
        *mulOther = lhs;
      return true;
    }
  }
  return false;
}

bool isAlignedValue(Value value, Value blockSize,
                    const std::optional<int64_t> &blockSizeConst,
                    Value *divHint) {
  if (!value)
    return true;

  value = ValueAnalysis::stripClampOne(value);
  if (isAlignedOffset(value, blockSize, blockSizeConst, divHint))
    return true;

  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    Value lhs = ValueAnalysis::stripClampOne(add.getLhs());
    Value rhs = ValueAnalysis::stripClampOne(add.getRhs());
    return isAlignedValue(lhs, blockSize, blockSizeConst) &&
           isAlignedValue(rhs, blockSize, blockSizeConst);
  }

  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    Value lhs = ValueAnalysis::stripClampOne(sub.getLhs());
    Value rhs = ValueAnalysis::stripClampOne(sub.getRhs());
    return isAlignedValue(lhs, blockSize, blockSizeConst) &&
           isAlignedValue(rhs, blockSize, blockSizeConst);
  }

  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    Value trueValue = ValueAnalysis::stripClampOne(select.getTrueValue());
    Value falseValue = ValueAnalysis::stripClampOne(select.getFalseValue());
    return isAlignedValue(trueValue, blockSize, blockSizeConst) &&
           isAlignedValue(falseValue, blockSize, blockSizeConst);
  }

  return false;
}

/// Neighborhood analysis walks recursively so it can detect block-neighborhood
/// localization rooted in an enclosing loop even when the actual db_ref uses
/// live inside nested point loops. The rewrite must mirror that recursive
/// scope; otherwise cloning an enclosing loop leaves the nested matched div/rem
/// tree intact and the fixed-point driver keeps strip-mining the same family.
void rewriteNestedNeighborhoodClone(
    Operation *originalOp, Operation *clonedOp,
    const NeighborhoodReplacementMap &replacements,
    const NeighborhoodEraseSet &eraseIfDead) {
  if (!originalOp || !clonedOp)
    return;

  for (auto [origRegion, clonedRegion] :
       llvm::zip(originalOp->getRegions(), clonedOp->getRegions())) {
    auto origBlockIt = origRegion.begin();
    auto clonedBlockIt = clonedRegion.begin();
    for (;
         origBlockIt != origRegion.end() && clonedBlockIt != clonedRegion.end();
         ++origBlockIt, ++clonedBlockIt) {
      rewriteNestedNeighborhoodClone(*origBlockIt, *clonedBlockIt, replacements,
                                     eraseIfDead);
    }
  }

  if (auto it = replacements.find(originalOp); it != replacements.end()) {
    if (clonedOp->getNumResults() == 1)
      clonedOp->getResult(0).replaceAllUsesWith(it->second);
  }

  if ((replacements.contains(originalOp) || eraseIfDead.contains(originalOp)) &&
      clonedOp->use_empty()) {
    clonedOp->erase();
  }
}

void rewriteNestedNeighborhoodClone(
    Block &originalBlock, Block &clonedBlock,
    const NeighborhoodReplacementMap &replacements,
    const NeighborhoodEraseSet &eraseIfDead) {
  auto origIt = originalBlock.begin();
  auto clonedIt = clonedBlock.begin();
  for (; origIt != originalBlock.end() && clonedIt != clonedBlock.end();) {
    Operation *origOp = &*origIt++;
    Operation *clonedOp = &*clonedIt++;
    rewriteNestedNeighborhoodClone(origOp, clonedOp, replacements, eraseIfDead);
  }
}

LoopNeighborhoodSegment::Boundary makeConstantBoundary(int64_t constant) {
  return {LoopNeighborhoodSegment::BoundaryKind::constant, constant};
}

LoopNeighborhoodSegment::Boundary makeBlockSizeMinusBoundary(int64_t constant) {
  return {LoopNeighborhoodSegment::BoundaryKind::blockSizeMinusConstant,
          constant};
}

LoopNeighborhoodSegment::Boundary makeBlockSizeBoundary() {
  return {LoopNeighborhoodSegment::BoundaryKind::blockSize, 0};
}

SmallVector<LoopNeighborhoodSegment, 4>
buildConstantNeighborhoodSegments(const NeighborhoodLoopInfo &info) {
  SmallVector<int64_t, 8> boundaries = {0, *info.blockSizeConst};
  for (const OffsetPatternGroup &group : info.offsetGroups) {
    int64_t offset = group.offsetConst;
    if (offset < 0)
      boundaries.push_back(-offset);
    else if (offset > 0)
      boundaries.push_back(*info.blockSizeConst - offset);
  }

  llvm::sort(boundaries);
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                   boundaries.end());

  SmallVector<LoopNeighborhoodSegment, 4> segments;
  for (size_t idx = 0, e = boundaries.size(); idx + 1 < e; ++idx) {
    int64_t lowerConst = boundaries[idx];
    int64_t upperConst = boundaries[idx + 1];
    if (lowerConst >= upperConst)
      continue;

    LoopNeighborhoodSegment segment;
    segment.localLower = makeConstantBoundary(lowerConst);
    segment.localUpper = makeConstantBoundary(upperConst);
    segment.blockDeltas.reserve(info.offsetGroups.size());

    for (const OffsetPatternGroup &group : info.offsetGroups) {
      int64_t blockDelta = 0;
      if (group.offsetConst < 0 && lowerConst < -group.offsetConst) {
        blockDelta = -1;
      } else if (group.offsetConst > 0 &&
                 lowerConst >= *info.blockSizeConst - group.offsetConst) {
        blockDelta = 1;
      }
      segment.blockDeltas.push_back(blockDelta);
    }

    segments.push_back(std::move(segment));
  }
  return segments;
}

std::optional<int64_t>
getDynamicNeighborhoodOrderingThreshold(const NeighborhoodLoopInfo &info) {
  int64_t maxNegativeSpan = 0;
  int64_t maxPositiveSpan = 0;
  for (const OffsetPatternGroup &group : info.offsetGroups) {
    if (group.offsetConst < 0)
      maxNegativeSpan = std::max(maxNegativeSpan, -group.offsetConst);
    else if (group.offsetConst > 0)
      maxPositiveSpan = std::max(maxPositiveSpan, group.offsetConst);
  }
  if (maxNegativeSpan == 0 && maxPositiveSpan == 0)
    return std::nullopt;
  return maxNegativeSpan + maxPositiveSpan;
}

SmallVector<LoopNeighborhoodSegment, 4>
buildDynamicNeighborhoodSegments(const NeighborhoodLoopInfo &info) {
  SmallVector<int64_t, 4> negativeBoundaries;
  SmallVector<int64_t, 4> positiveOffsets;
  for (const OffsetPatternGroup &group : info.offsetGroups) {
    if (group.offsetConst < 0)
      negativeBoundaries.push_back(-group.offsetConst);
    else if (group.offsetConst > 0)
      positiveOffsets.push_back(group.offsetConst);
  }

  llvm::sort(negativeBoundaries);
  negativeBoundaries.erase(
      std::unique(negativeBoundaries.begin(), negativeBoundaries.end()),
      negativeBoundaries.end());
  llvm::sort(positiveOffsets,
             [](int64_t lhs, int64_t rhs) { return lhs > rhs; });
  positiveOffsets.erase(
      std::unique(positiveOffsets.begin(), positiveOffsets.end()),
      positiveOffsets.end());

  llvm::DenseMap<int64_t, unsigned> negativeRank;
  for (auto [index, value] : llvm::enumerate(negativeBoundaries))
    negativeRank[value] = index;

  llvm::DenseMap<int64_t, unsigned> positiveRank;
  for (auto [index, value] : llvm::enumerate(positiveOffsets))
    positiveRank[value] = index;

  SmallVector<LoopNeighborhoodSegment::Boundary, 8> boundaries;
  boundaries.push_back(makeConstantBoundary(0));
  for (int64_t value : negativeBoundaries)
    boundaries.push_back(makeConstantBoundary(value));
  for (int64_t value : positiveOffsets)
    boundaries.push_back(makeBlockSizeMinusBoundary(value));
  boundaries.push_back(makeBlockSizeBoundary());

  SmallVector<LoopNeighborhoodSegment, 4> segments;
  unsigned negativesPassed = 0;
  unsigned positivesPassed = 0;
  for (size_t index = 0, e = boundaries.size(); index + 1 < e; ++index) {
    LoopNeighborhoodSegment segment;
    segment.localLower = boundaries[index];
    segment.localUpper = boundaries[index + 1];
    segment.blockDeltas.reserve(info.offsetGroups.size());

    for (const OffsetPatternGroup &group : info.offsetGroups) {
      int64_t blockDelta = 0;
      if (group.offsetConst < 0) {
        unsigned rank = negativeRank[-group.offsetConst];
        if (rank >= negativesPassed)
          blockDelta = -1;
      } else if (group.offsetConst > 0) {
        unsigned rank = positiveRank[group.offsetConst];
        if (rank < positivesPassed)
          blockDelta = 1;
      }
      segment.blockDeltas.push_back(blockDelta);
    }

    segments.push_back(std::move(segment));

    auto upperKind = boundaries[index + 1].kind;
    if (upperKind == LoopNeighborhoodSegment::BoundaryKind::constant &&
        boundaries[index + 1].constant > 0) {
      ++negativesPassed;
    } else if (upperKind ==
               LoopNeighborhoodSegment::BoundaryKind::blockSizeMinusConstant) {
      ++positivesPassed;
    }
  }

  return segments;
}

SmallVector<LoopNeighborhoodSegment, 4>
buildNeighborhoodSegments(NeighborhoodLoopInfo &info) {
  if (info.blockSizeConst)
    return buildConstantNeighborhoodSegments(info);
  info.dynamicOrderingGuardThreshold =
      getDynamicNeighborhoodOrderingThreshold(info);
  return buildDynamicNeighborhoodSegments(info);
}

unsigned countTrackedNeighborhoodPatterns(const NeighborhoodLoopInfo &info) {
  unsigned count = 0;
  for (const OffsetPatternGroup &group : info.offsetGroups)
    count += group.divOps.size() + group.remValues.size();
  return count;
}

bool hasNonZeroNeighborhoodOffset(const NeighborhoodLoopInfo &info) {
  return llvm::any_of(info.offsetGroups, [](const OffsetPatternGroup &group) {
    return group.offsetConst != 0;
  });
}

unsigned countNeighborhoodDbRefUsers(scf::ForOp loop,
                                     const NeighborhoodLoopInfo &info) {
  SmallVector<Operation *, 8> matchedRefs;
  loop.getBody()->walk([&](DbRefOp ref) {
    for (Value idx : ref.getIndices()) {
      Value base = ValueAnalysis::stripNumericCasts(idx);
      for (const OffsetPatternGroup &group : info.offsetGroups) {
        for (auto div : group.divOps) {
          if (!ValueAnalysis::dependsOn(base, div.getResult()))
            continue;
          if (!llvm::is_contained(matchedRefs, ref.getOperation()))
            matchedRefs.push_back(ref.getOperation());
          return;
        }
      }
    }
  });
  return matchedRefs.size();
}

bool finalizeNeighborhoodCandidate(scf::ForOp loop, NeighborhoodLoopInfo &info,
                                   DominanceInfo &domInfo) {
  if (!info.blockSizeVal || info.offsetGroups.empty()) {
    ARTS_DEBUG("Neighborhood candidate rejected: missing block size or offset "
               "groups");
    return false;
  }
  if (info.blockSizeConst && *info.blockSizeConst <= 1) {
    ARTS_DEBUG("Neighborhood candidate rejected: block size <= 1");
    return false;
  }
  if (info.invariantBase &&
      !dominatesOrInAncestor(info.invariantBase, loop, domInfo)) {
    ARTS_DEBUG("Neighborhood candidate rejected: invariant base is not "
               "loop-invariant");
    return false;
  }
  /// Neighborhood strip-mining is only profitable when the loop carries
  /// halo-style offsets around the block-local center. Center-only families
  /// belong to the legacy block strip-miner or later hoisting; rewriting them
  /// here just adds outer-loop structure without removing any neighborhood
  /// localization work.
  if (!hasNonZeroNeighborhoodOffset(info)) {
    ARTS_DEBUG("Neighborhood candidate rejected: center-only family");
    return false;
  }

  unsigned divCount = 0;
  unsigned remCount = 0;
  for (const OffsetPatternGroup &group : info.offsetGroups) {
    divCount += group.divOps.size();
    remCount += group.remValues.size();
  }
  if (divCount == 0 || remCount == 0) {
    ARTS_DEBUG("Neighborhood candidate rejected: missing div/rem pair");
    return false;
  }

  llvm::sort(info.offsetGroups,
             [](const OffsetPatternGroup &lhs, const OffsetPatternGroup &rhs) {
               return lhs.offsetConst < rhs.offsetConst;
             });

  info.matchedDbRefCount = countNeighborhoodDbRefUsers(loop, info);
  if (info.matchedDbRefCount == 0) {
    ARTS_DEBUG("Neighborhood candidate rejected: no db_ref uses depend on "
               "matched block indices");
    return false;
  }

  info.segments = buildNeighborhoodSegments(info);
  if (info.segments.empty())
    ARTS_DEBUG("Neighborhood candidate rejected: no legal boundary segments");
  return !info.segments.empty();
}

bool isBetterNeighborhoodCandidate(const NeighborhoodLoopInfo &lhs,
                                   const NeighborhoodLoopInfo &rhs) {
  if (lhs.matchedDbRefCount != rhs.matchedDbRefCount)
    return lhs.matchedDbRefCount > rhs.matchedDbRefCount;
  if (lhs.offsetGroups.size() != rhs.offsetGroups.size())
    return lhs.offsetGroups.size() > rhs.offsetGroups.size();
  return countTrackedNeighborhoodPatterns(lhs) >
         countTrackedNeighborhoodPatterns(rhs);
}

std::optional<NeighborhoodLoopInfo>
analyzeNeighborhoodLoop(scf::ForOp loop, DominanceInfo &domInfo) {
  Value iv = loop.getInductionVar();
  SmallVector<NeighborhoodLoopInfo, 4> candidates;

  auto step = ValueAnalysis::getConstantIndexStripped(loop.getStep());
  if (!step || *step != 1) {
    ARTS_DEBUG("Neighborhood strip-mining skipped: non-unit step");
    return std::nullopt;
  }

  auto recordMatch = [&](const NeighborhoodPattern &pattern) {
    recordNeighborhoodPattern(
        getOrCreateNeighborhoodFamily(candidates, pattern.blockSizeVal,
                                      pattern.blockSizeConst,
                                      pattern.invariantBase),
        pattern);
  };

  /// One loop can carry multiple independent block decompositions, such as
  /// row and column block indices in a 2-D stencil. Collect families per
  /// block size and choose the family most tightly coupled to db_ref users
  /// instead of rejecting on the first unrelated div/rem encountered.
  loop.getBody()->walk([&](Operation *op) {
    if (auto div = dyn_cast<arith::DivUIOp>(op)) {
      if (auto pattern = matchNeighborhoodPattern(div.getLhs(), div.getRhs(),
                                                  div, Value(), iv, {}))
        recordMatch(*pattern);
      return;
    }
    if (auto rem = dyn_cast<arith::RemUIOp>(op)) {
      if (auto pattern = matchNeighborhoodPattern(rem.getLhs(), rem.getRhs(),
                                                  std::nullopt, rem.getResult(),
                                                  iv, {rem.getOperation()}))
        recordMatch(*pattern);
      return;
    }
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      if (auto pattern = matchNormalizedSignedRemainderNeighborhood(select, iv))
        recordMatch(*pattern);
      return;
    }
    if (isa<arith::RemSIOp, arith::CmpIOp>(op))
      return;
  });

  NeighborhoodLoopInfo *bestCandidate = nullptr;
  for (NeighborhoodLoopInfo &candidate : candidates) {
    if (!finalizeNeighborhoodCandidate(loop, candidate, domInfo))
      continue;
    if (!bestCandidate ||
        isBetterNeighborhoodCandidate(candidate, *bestCandidate))
      bestCandidate = &candidate;
  }

  if (!bestCandidate) {
    ARTS_DEBUG("Neighborhood strip-mining skipped: no valid candidate family");
    return std::nullopt;
  }

  if (bestCandidate->blockSizeConst) {
    ARTS_DEBUG(
        "Neighborhood candidate selected: blockSize="
        << *bestCandidate->blockSizeConst
        << " groups=" << bestCandidate->offsetGroups.size()
        << " dbRefs=" << bestCandidate->matchedDbRefCount
        << " segments=" << bestCandidate->segments.size()
        << " depth=" << getLoopDepth(loop.getOperation())
        << (bestCandidate->dynamicOrderingGuardThreshold ? " guarded" : ""));
  } else {
    ARTS_DEBUG(
        "Neighborhood candidate selected: blockSize=<dynamic> groups="
        << bestCandidate->offsetGroups.size()
        << " dbRefs=" << bestCandidate->matchedDbRefCount
        << " segments=" << bestCandidate->segments.size()
        << " depth=" << getLoopDepth(loop.getOperation())
        << (bestCandidate->dynamicOrderingGuardThreshold ? " guarded" : ""));
  }
  return *bestCandidate;
}

std::optional<LoopBlockInfo> analyzeLegacyLoop(scf::ForOp loop,
                                               DominanceInfo &domInfo) {
  LoopBlockInfo info;
  Value iv = loop.getInductionVar();
  Value loopLowerBound = loop.getLowerBound();

  /// Require constant step = 1; allow dynamic upper bound.
  auto step = ValueAnalysis::getConstantIndexStripped(loop.getStep());
  if (!step || *step != 1)
    return std::nullopt;
  info.lowerBoundConst = ValueAnalysis::tryFoldConstantIndex(loopLowerBound);
  auto ub = ValueAnalysis::getConstantIndexStripped(loop.getUpperBound());
  if (ub && info.lowerBoundConst && *ub <= *info.lowerBoundConst)
    return std::nullopt;

  bool invalid = false;
  loop.getBody()->walk([&](Operation *op) {
    if (auto div = dyn_cast<arith::DivUIOp>(op)) {
      auto offsetOpt = extractInvariantOffset(div.getLhs(), iv);
      if (!offsetOpt)
        return;
      Value offset = *offsetOpt;
      if (offset) {
        if (!info.offsetVal)
          info.offsetVal = offset;
        else if (!ValueAnalysis::sameValue(info.offsetVal, offset))
          invalid = true;
      } else if (info.offsetVal) {
        invalid = true;
      }
      Value rhs = ValueAnalysis::stripNumericCasts(div.getRhs());
      if (ValueAnalysis::dependsOn(rhs, iv))
        return;
      auto rhsConst = ValueAnalysis::tryFoldConstantIndex(rhs);
      if (rhsConst && *rhsConst <= 1)
        return;
      if (!info.blockSizeVal) {
        info.blockSizeVal = rhs;
        info.blockSizeConst = rhsConst ? rhsConst : resolveBlockSizeHint(rhs);
      } else if (!sameBlockSizeFamily(info.blockSizeVal, info.blockSizeConst,
                                      rhs, rhsConst)) {
        if (!info.blockSizeConst || !rhsConst ||
            *info.blockSizeConst != *rhsConst)
          invalid = true;
      }
      info.divOps.push_back(div);
      return;
    }
    if (auto rem = dyn_cast<arith::RemUIOp>(op)) {
      if (!recordRemPattern(rem.getLhs(), rem.getRhs(), rem.getResult(), iv,
                            info, {rem.getOperation()}))
        invalid = true;
      return;
    }
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      if (matchNormalizedSignedRemainder(select, iv, info))
        return;
      return;
    }
    if (isa<arith::RemSIOp, arith::CmpIOp>(op)) {
      return;
    }
  });

  if (invalid || !info.blockSizeVal)
    return std::nullopt;
  if (info.divOps.empty() || info.remValues.empty())
    return std::nullopt;

  if (info.blockSizeConst && *info.blockSizeConst <= 1)
    return std::nullopt;
  if (!info.blockSizeConst &&
      !dominatesOrInAncestor(info.blockSizeVal, loop, domInfo))
    return std::nullopt;
  if (!info.lowerBoundConst &&
      !dominatesOrInAncestor(loopLowerBound, loop, domInfo))
    return std::nullopt;
  if (info.offsetVal) {
    if (!dominatesOrInAncestor(info.offsetVal, loop, domInfo))
      return std::nullopt;
    if (!isAlignedOffset(info.offsetVal, info.blockSizeVal, info.blockSizeConst,
                         &info.offsetDivHint))
      return std::nullopt;
  }
  if (!info.lowerBoundConst &&
      !isAlignedValue(loopLowerBound, info.blockSizeVal, info.blockSizeConst,
                      &info.lowerBoundDivHint))
    return std::nullopt;
  if (info.lowerBoundConst && info.blockSizeConst &&
      (*info.lowerBoundConst % *info.blockSizeConst) != 0)
    return std::nullopt;
  if (info.offsetVal) {
    auto offsetConst = ValueAnalysis::tryFoldConstantIndex(info.offsetVal);
    if (info.lowerBoundConst && offsetConst && info.blockSizeConst) {
      int64_t effectiveLower = *info.lowerBoundConst + *offsetConst;
      if ((effectiveLower % *info.blockSizeConst) != 0)
        return std::nullopt;
    } else if (!isAlignedValue(info.offsetVal, info.blockSizeVal,
                               info.blockSizeConst)) {
      return std::nullopt;
    }
  }

  /// Require at least one db_ref that depends on a matching div op for block
  /// index.
  bool hasDbRef = false;
  loop.getBody()->walk([&](DbRefOp ref) {
    for (Value idx : ref.getIndices()) {
      Value base = ValueAnalysis::stripNumericCasts(idx);
      for (auto div : info.divOps) {
        if (ValueAnalysis::dependsOn(base, div.getResult())) {
          hasDbRef = true;
          return;
        }
      }
    }
  });

  if (!hasDbRef)
    return std::nullopt;

  return info;
}

Value buildNeighborhoodBoundaryValue(
    OpBuilder &builder, Location loc,
    const LoopNeighborhoodSegment::Boundary &boundary, Value blockSize) {
  switch (boundary.kind) {
  case LoopNeighborhoodSegment::BoundaryKind::constant:
    return arts::createConstantIndex(builder, loc, boundary.constant);
  case LoopNeighborhoodSegment::BoundaryKind::blockSizeMinusConstant: {
    Value constant = arts::createConstantIndex(builder, loc, boundary.constant);
    return arith::SubIOp::create(builder, loc, blockSize, constant);
  }
  case LoopNeighborhoodSegment::BoundaryKind::blockSize:
    return blockSize;
  }
  llvm_unreachable("unknown neighborhood boundary kind");
}

Value buildNeighborhoodRemValue(OpBuilder &builder, Location loc, Value local,
                                Value blockSize, int64_t offsetConst,
                                int64_t blockDelta) {
  Value remVal = local;
  if (offsetConst != 0) {
    Value offset = arts::createConstantIndex(builder, loc, offsetConst);
    remVal = arith::AddIOp::create(builder, loc, remVal, offset);
  }
  if (blockDelta > 0)
    remVal = arith::SubIOp::create(builder, loc, remVal, blockSize);
  else if (blockDelta < 0)
    remVal = arith::AddIOp::create(builder, loc, remVal, blockSize);
  return remVal;
}

bool stripMineNeighborhoodLoopImpl(scf::ForOp loop,
                                   const NeighborhoodLoopInfo &info,
                                   SmallVectorImpl<Value> *newResults) {
  Location loc = loop.getLoc();
  OpBuilder builder(loop);

  auto step = ValueAnalysis::getConstantIndexStripped(loop.getStep());
  if (!step || *step != 1)
    return false;

  Operation *origTerminator =
      loop.getBody()->empty() ? nullptr : &loop.getBody()->back();
  auto origYield = dyn_cast_or_null<scf::YieldOp>(origTerminator);
  if (!origYield)
    return false;

  Value zero = arts::createZeroIndex(builder, loc);
  Value one = arts::createOneIndex(builder, loc);
  Value lbVal =
      ValueAnalysis::ensureIndexType(loop.getLowerBound(), builder, loc);
  Value ubVal =
      ValueAnalysis::ensureIndexType(loop.getUpperBound(), builder, loc);
  func::FuncOp parentFunc = loop->getParentOfType<func::FuncOp>();
  DominanceInfo domInfo(parentFunc);
  Value bsSource = info.blockSizeVal;
  if (!info.blockSizeConst)
    bsSource = ValueAnalysis::traceValueToDominating(info.blockSizeVal, loop,
                                                     builder, domInfo, loc);
  Value bsVal =
      info.blockSizeConst
          ? arts::createConstantIndex(builder, loc, *info.blockSizeConst)
          : ValueAnalysis::ensureIndexType(bsSource, builder, loc);
  if (!lbVal || !ubVal || !bsVal)
    return false;
  Value baseVal;
  if (info.invariantBase) {
    Value baseSource = ValueAnalysis::traceValueToDominating(
        info.invariantBase, loop, builder, domInfo, loc);
    baseVal = ValueAnalysis::ensureIndexType(baseSource, builder, loc);
    if (!baseVal)
      return false;
  }

  Value effectiveLb = lbVal;
  Value effectiveUb = ubVal;
  if (baseVal) {
    effectiveLb = arith::AddIOp::create(builder, loc, effectiveLb, baseVal);
    effectiveUb = arith::AddIOp::create(builder, loc, effectiveUb, baseVal);
  }
  Value bsMinusOne = arith::SubIOp::create(builder, loc, bsVal, one);

  Value blockLower = arith::DivUIOp::create(builder, loc, effectiveLb, bsVal);
  Value ubPlus = arith::AddIOp::create(builder, loc, effectiveUb, bsMinusOne);
  Value blockUpperRaw = arith::DivUIOp::create(builder, loc, ubPlus, bsVal);
  Value hasWork = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt,
                                        effectiveUb, effectiveLb);
  Value blockUpper =
      arith::SelectOp::create(builder, loc, hasWork, blockUpperRaw, blockLower);

  auto outer = scf::ForOp::create(builder, loc, blockLower, blockUpper, one,
                                  loop.getInitArgs());
  /// The pass runs to fixed point. Mark implementation loops created here so
  /// the driver keeps analyzing user IR instead of re-strip-mining the
  /// synthetic control structure produced by this transform.
  markGeneratedByStripMining(outer);
  builder.setInsertionPointToStart(outer.getBody());

  Value blockStart =
      arith::MulIOp::create(builder, loc, outer.getInductionVar(), bsVal);

  auto clampToBlock = [&](Value bound) -> Value {
    Value before = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::uge, blockStart, bound);
    Value distance = arith::SubIOp::create(builder, loc, bound, blockStart);
    Value nonNegative =
        arith::SelectOp::create(builder, loc, before, zero, distance);
    return arith::MinUIOp::create(builder, loc, nonNegative, bsVal);
  };

  Value localLower = clampToBlock(effectiveLb);
  Value localUpper = clampToBlock(effectiveUb);

  SmallVector<Operation *> skipOps;
  skipOps.reserve(info.offsetGroups.size() * 4);
  for (const OffsetPatternGroup &group : info.offsetGroups) {
    for (auto div : group.divOps)
      skipOps.push_back(div);
    llvm::append_range(skipOps, group.remPatternOps);
  }

  SmallVector<Value> carriedValues(outer.getRegionIterArgs().begin(),
                                   outer.getRegionIterArgs().end());
  for (const LoopNeighborhoodSegment &segmentInfo : info.segments) {
    Value segLowerBound = buildNeighborhoodBoundaryValue(
        builder, loc, segmentInfo.localLower, bsVal);
    Value segUpperBound = buildNeighborhoodBoundaryValue(
        builder, loc, segmentInfo.localUpper, bsVal);
    Value segLower =
        arith::MaxUIOp::create(builder, loc, localLower, segLowerBound);
    Value segUpper =
        arith::MinUIOp::create(builder, loc, localUpper, segUpperBound);

    auto inner = scf::ForOp::create(builder, loc, segLower, segUpper, one,
                                    carriedValues);
    markGeneratedByStripMining(inner);
    builder.setInsertionPointToStart(inner.getBody());

    Value local = inner.getInductionVar();
    Value global = arith::AddIOp::create(builder, loc, blockStart, local);
    Value mappedIv = global;
    if (baseVal)
      mappedIv = arith::SubIOp::create(builder, loc, global, baseVal);

    IRMapping mapping;
    mapping.map(loop.getInductionVar(), mappedIv);

    auto oldArgs = loop.getRegionIterArgs();
    auto newArgs = inner.getRegionIterArgs();
    for (auto it : llvm::zip(oldArgs, newArgs))
      mapping.map(std::get<0>(it), std::get<1>(it));

    for (auto [groupIndex, group] : llvm::enumerate(info.offsetGroups)) {
      int64_t blockDelta = segmentInfo.blockDeltas[groupIndex];

      Value blockIdx = outer.getInductionVar();
      if (blockDelta != 0) {
        Value deltaConst = arts::createConstantIndex(builder, loc, blockDelta);
        blockIdx = arith::AddIOp::create(builder, loc, blockIdx, deltaConst);
      }
      for (auto div : group.divOps)
        mapping.map(div.getResult(), blockIdx);

      Value remVal = buildNeighborhoodRemValue(builder, loc, local, bsVal,
                                               group.offsetConst, blockDelta);
      for (Value rem : group.remValues)
        mapping.map(rem, remVal);
    }

    NeighborhoodReplacementMap nestedReplacements;
    NeighborhoodEraseSet nestedEraseIfDead;
    for (auto [groupIndex, group] : llvm::enumerate(info.offsetGroups)) {
      int64_t blockDelta = segmentInfo.blockDeltas[groupIndex];

      Value blockIdx = outer.getInductionVar();
      if (blockDelta != 0) {
        Value deltaConst = arts::createConstantIndex(builder, loc, blockDelta);
        blockIdx = arith::AddIOp::create(builder, loc, blockIdx, deltaConst);
      }
      for (auto div : group.divOps)
        nestedReplacements[div.getOperation()] = blockIdx;

      Value remVal = buildNeighborhoodRemValue(builder, loc, local, bsVal,
                                               group.offsetConst, blockDelta);
      for (Value rem : group.remValues)
        if (Operation *defOp = rem.getDefiningOp())
          nestedReplacements[defOp] = remVal;

      for (Operation *patternOp : group.remPatternOps)
        nestedEraseIfDead.insert(patternOp);
    }

    for (Operation &op : loop.getBody()->without_terminator()) {
      if (auto div = dyn_cast<arith::DivUIOp>(&op)) {
        bool skipped = false;
        for (const OffsetPatternGroup &group : info.offsetGroups) {
          if (llvm::is_contained(group.divOps, div)) {
            skipped = true;
            break;
          }
        }
        if (skipped)
          continue;
      }
      if (llvm::is_contained(skipOps, &op))
        continue;
      Operation *clonedOp = builder.clone(op, mapping);
      rewriteNestedNeighborhoodClone(&op, clonedOp, nestedReplacements,
                                     nestedEraseIfDead);
    }

    scf::YieldOp innerYield = nullptr;
    for (Operation &op : *inner.getBody()) {
      if (auto y = dyn_cast<scf::YieldOp>(&op)) {
        innerYield = y;
        break;
      }
    }
    if (innerYield)
      innerYield.erase();
    builder.setInsertionPointToEnd(inner.getBody());

    SmallVector<Value> newYieldOperands;
    newYieldOperands.reserve(origYield.getNumOperands());
    for (Value v : origYield.getOperands())
      newYieldOperands.push_back(mapping.lookupOrDefault(v));
    scf::YieldOp::create(builder, loc, newYieldOperands);

    builder.setInsertionPointAfter(inner);
    carriedValues.assign(inner.getResults().begin(), inner.getResults().end());
  }

  scf::YieldOp outerYield = nullptr;
  for (Operation &op : *outer.getBody()) {
    if (auto y = dyn_cast<scf::YieldOp>(&op)) {
      outerYield = y;
      break;
    }
  }
  if (outerYield)
    outerYield.erase();
  builder.setInsertionPointToEnd(outer.getBody());
  scf::YieldOp::create(builder, loc, carriedValues);

  if (newResults)
    newResults->assign(outer.getResults().begin(), outer.getResults().end());
  loop.replaceAllUsesWith(outer.getResults());
  loop.erase();
  return true;
}

bool stripMineNeighborhoodLoop(scf::ForOp loop,
                               const NeighborhoodLoopInfo &info) {
  if (!info.dynamicOrderingGuardThreshold)
    return stripMineNeighborhoodLoopImpl(loop, info);

  Location loc = loop.getLoc();
  OpBuilder builder(loop);
  func::FuncOp parentFunc = loop->getParentOfType<func::FuncOp>();
  DominanceInfo domInfo(parentFunc);
  Value bsSource = ValueAnalysis::traceValueToDominating(
      info.blockSizeVal, loop, builder, domInfo, loc);
  Value bsVal = ValueAnalysis::ensureIndexType(bsSource, builder, loc);
  if (!bsVal)
    return false;

  Value threshold = arts::createConstantIndex(
      builder, loc, *info.dynamicOrderingGuardThreshold);
  Value guard = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt,
                                      bsVal, threshold);
  auto ifOp = scf::IfOp::create(builder, loc, loop.getResultTypes(), guard,
                                /*withElseRegion=*/true);

  if (loop.getNumResults() > 0)
    loop.replaceAllUsesWith(ifOp.getResults());

  Block &elseBlock = ifOp.getElseRegion().front();
  auto elseYield = cast<scf::YieldOp>(elseBlock.getTerminator());
  builder.setInsertionPoint(elseYield);
  auto elseLoop = cast<scf::ForOp>(builder.clone(*loop));
  markGeneratedByStripMining(elseLoop);
  elseYield->setOperands(elseLoop.getResults());

  Block &thenBlock = ifOp.getThenRegion().front();
  auto thenYield = cast<scf::YieldOp>(thenBlock.getTerminator());
  loop->moveBefore(thenYield);

  SmallVector<Value> rewrittenResults;
  if (!stripMineNeighborhoodLoopImpl(loop, info, &rewrittenResults)) {
    ifOp.erase();
    return false;
  }

  thenYield->setOperands(rewrittenResults);
  return true;
}

bool stripMineLegacyLoop(scf::ForOp loop, const LoopBlockInfo &info) {
  Location loc = loop.getLoc();
  OpBuilder builder(loop);
  auto step = ValueAnalysis::getConstantIndexStripped(loop.getStep());
  if (!step || *step != 1)
    return false;

  if (info.blockSizeConst && *info.blockSizeConst <= 1)
    return false;

  Operation *origTerminator =
      loop.getBody()->empty() ? nullptr : &loop.getBody()->back();
  auto origYield = dyn_cast_or_null<scf::YieldOp>(origTerminator);
  if (!origYield)
    return false;

  Value zero = arts::createZeroIndex(builder, loc);
  Value one = arts::createOneIndex(builder, loc);
  Value lbVal =
      ValueAnalysis::ensureIndexType(loop.getLowerBound(), builder, loc);
  Value ubVal = loop.getUpperBound();
  Value bsVal =
      info.blockSizeConst
          ? arts::createConstantIndex(builder, loc, *info.blockSizeConst)
          : ValueAnalysis::ensureIndexType(info.blockSizeVal, builder, loc);
  Value ubGeLb = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::uge,
                                       ubVal, lbVal);
  Value tripCountRaw = arith::SubIOp::create(builder, loc, ubVal, lbVal);
  Value tripCount =
      arith::SelectOp::create(builder, loc, ubGeLb, tripCountRaw, zero);
  Value bsMinusOne = arith::SubIOp::create(builder, loc, bsVal, one);
  Value tripPlus = arith::AddIOp::create(builder, loc, tripCount, bsMinusOne);
  Value numBlocksVal = arith::DivUIOp::create(builder, loc, tripPlus, bsVal);
  auto ub = ValueAnalysis::getConstantIndexStripped(loop.getUpperBound());
  if (info.blockSizeConst && info.lowerBoundConst && ub) {
    int64_t tripCountConst = *ub - *info.lowerBoundConst;
    if (tripCountConst <= *info.blockSizeConst)
      return false;
    int64_t numBlocksConst =
        (tripCountConst + *info.blockSizeConst - 1) / *info.blockSizeConst;
    if (numBlocksConst <= 1)
      return false;
  }

  /// Create outer block loop.
  auto outer = scf::ForOp::create(builder, loc, zero, numBlocksVal, one,
                                  loop.getInitArgs());
  markGeneratedByStripMining(outer);

  builder.setInsertionPointToStart(outer.getBody());
  Value blockStart = arith::AddIOp::create(
      builder, loc, lbVal,
      arith::MulIOp::create(builder, loc, outer.getInductionVar(), bsVal));

  /// innerUpper = min(bs, max(0, ub - blockStart))
  Value remaining = arith::SubIOp::create(builder, loc, ubVal, blockStart);
  Value clampCond = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::uge, blockStart, ubVal);
  Value remainingNonNeg =
      arith::SelectOp::create(builder, loc, clampCond, zero, remaining);
  Value innerUpper =
      arith::MinUIOp::create(builder, loc, remainingNonNeg, bsVal);

  /// Create inner local loop, carrying iter_args from outer.
  auto inner = scf::ForOp::create(builder, loc, zero, innerUpper, one,
                                  outer.getRegionIterArgs());
  markGeneratedByStripMining(inner);

  /// Prepare mapping and clone body.
  builder.setInsertionPointToStart(inner.getBody());
  Value global =
      arith::AddIOp::create(builder, loc, blockStart, inner.getInductionVar());

  IRMapping mapping;
  mapping.map(loop.getInductionVar(), global);

  /// Map iter args.
  auto oldArgs = loop.getRegionIterArgs();
  auto newArgs = inner.getRegionIterArgs();
  for (auto it : llvm::zip(oldArgs, newArgs)) {
    mapping.map(std::get<0>(it), std::get<1>(it));
  }

  /// Map div/rem results to outer/inner ivs.
  Value blockIdx = outer.getInductionVar();
  Value effectiveLowerDiv = nullptr;
  if (info.lowerBoundDivHint) {
    effectiveLowerDiv =
        ValueAnalysis::ensureIndexType(info.lowerBoundDivHint, builder, loc);
  } else if (info.lowerBoundConst && info.blockSizeConst) {
    int64_t lowerDivConst = *info.lowerBoundConst / *info.blockSizeConst;
    effectiveLowerDiv = arts::createConstantIndex(builder, loc, lowerDivConst);
  } else {
    effectiveLowerDiv = arith::DivUIOp::create(builder, loc, lbVal, bsVal);
  }
  if (info.offsetVal) {
    Value offsetDiv = nullptr;
    if (info.offsetDivHint) {
      offsetDiv =
          ValueAnalysis::ensureIndexType(info.offsetDivHint, builder, loc);
    } else if (auto offConst =
                   ValueAnalysis::tryFoldConstantIndex(info.offsetVal)) {
      if (info.blockSizeConst) {
        int64_t offDivConst = *offConst / *info.blockSizeConst;
        offsetDiv = arts::createConstantIndex(builder, loc, offDivConst);
      }
    }
    if (!offsetDiv) {
      Value offsetIdx =
          ValueAnalysis::ensureIndexType(info.offsetVal, builder, loc);
      offsetDiv = arith::DivUIOp::create(builder, loc, offsetIdx, bsVal);
    }
    effectiveLowerDiv =
        arith::AddIOp::create(builder, loc, effectiveLowerDiv, offsetDiv);
  }
  auto effectiveLowerDivConst =
      ValueAnalysis::tryFoldConstantIndex(effectiveLowerDiv);
  if (!effectiveLowerDivConst || *effectiveLowerDivConst != 0)
    blockIdx = arith::AddIOp::create(builder, loc, blockIdx, effectiveLowerDiv);
  for (auto div : info.divOps)
    mapping.map(div.getResult(), blockIdx);
  for (Value rem : info.remValues)
    mapping.map(rem, inner.getInductionVar());

  /// Clone body ops (excluding terminator) into inner loop.
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (auto div = dyn_cast<arith::DivUIOp>(&op)) {
      if (llvm::is_contained(info.divOps, div))
        continue;
    }
    if (llvm::is_contained(info.remPatternOps, &op))
      continue;
    builder.clone(op, mapping);
  }

  /// Replace inner loop terminator with mapped yield.
  scf::YieldOp innerYield = nullptr;
  for (Operation &op : *inner.getBody()) {
    if (auto y = dyn_cast<scf::YieldOp>(&op)) {
      innerYield = y;
      break;
    }
  }
  if (innerYield)
    innerYield.erase();
  builder.setInsertionPointToEnd(inner.getBody());

  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(origYield.getNumOperands());
  for (Value v : origYield.getOperands()) {
    newYieldOperands.push_back(mapping.lookupOrDefault(v));
  }
  scf::YieldOp::create(builder, loc, newYieldOperands);

  /// Replace outer loop terminator with inner results.
  scf::YieldOp outerYield = nullptr;
  for (Operation &op : *outer.getBody()) {
    if (auto y = dyn_cast<scf::YieldOp>(&op)) {
      outerYield = y;
      break;
    }
  }
  if (outerYield)
    outerYield.erase();
  builder.setInsertionPointToEnd(outer.getBody());
  scf::YieldOp::create(builder, loc, inner.getResults());

  /// Replace uses of original loop with new outer loop results.
  loop.replaceAllUsesWith(outer.getResults());
  loop.erase();

  return true;
}

} // namespace mlir::arts::block_loop_strip_mining
