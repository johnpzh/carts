///==========================================================================///
/// File: IterationSizingUtils.cpp
///
/// SDE-owned iteration sizing helpers for source-level scheduling intent.
///==========================================================================///

#include "carts/dialect/sde/Utils/IterationSizingUtils.h"

#include "carts/utils/LoopUtils.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::carts;

namespace mlir::carts::sde {

int64_t ceilDivPositive(int64_t value, int64_t divisor) {
  return llvm::divideCeil(std::max<int64_t>(1, value),
                          std::max<int64_t>(1, divisor));
}

SmallVector<int64_t, 4> factorWorkersAcrossDims(int64_t workers,
                                                ArrayRef<int64_t> extents) {
  SmallVector<int64_t, 4> grid(extents.size(), 1);
  if (workers <= 1 || extents.empty())
    return grid;

  SmallVector<int64_t, 8> factors;
  int64_t remaining = workers;
  for (int64_t factor = 2; factor * factor <= remaining; ++factor) {
    while (remaining % factor == 0) {
      factors.push_back(factor);
      remaining /= factor;
    }
  }
  if (remaining > 1)
    factors.push_back(remaining);
  llvm::sort(factors, std::greater<int64_t>());

  for (int64_t factor : factors) {
    unsigned bestDim = 0;
    int64_t bestSpan = -1;
    for (auto [idx, extent] : llvm::enumerate(extents)) {
      int64_t span = ceilDivPositive(extent, grid[idx]);
      if (span > bestSpan) {
        bestSpan = span;
        bestDim = static_cast<unsigned>(idx);
      }
    }
    grid[bestDim] *= factor;
  }

  return grid;
}

static long double estimateStencilExpandedTileRatio(ArrayRef<int64_t> extents,
                                                    ArrayRef<int64_t> haloRadii,
                                                    ArrayRef<int64_t> grid) {
  long double ownedVolume = 1.0L;
  long double expandedVolume = 1.0L;
  for (auto [idx, extent] : llvm::enumerate(extents)) {
    int64_t tile = ceilDivPositive(extent, grid[idx]);
    int64_t halo =
        idx < haloRadii.size() ? std::max<int64_t>(0, haloRadii[idx]) : 0;
    ownedVolume *= static_cast<long double>(tile);
    expandedVolume *= static_cast<long double>(tile + 2 * halo);
  }
  if (ownedVolume <= 0.0L)
    return std::numeric_limits<long double>::infinity();
  return expandedVolume / ownedVolume;
}

SmallVector<int64_t, 4>
factorStencilWorkersAcrossDims(int64_t workers, ArrayRef<int64_t> extents,
                               ArrayRef<int64_t> haloRadii) {
  SmallVector<int64_t, 4> grid(extents.size(), 1);
  if (workers <= 1 || extents.empty())
    return grid;
  if (haloRadii.empty() ||
      llvm::all_of(haloRadii, [](int64_t halo) { return halo <= 0; }))
    return factorWorkersAcrossDims(workers, extents);

  SmallVector<int64_t, 8> factors;
  int64_t remaining = workers;
  for (int64_t factor = 2; factor * factor <= remaining; ++factor) {
    while (remaining % factor == 0) {
      factors.push_back(factor);
      remaining /= factor;
    }
  }
  if (remaining > 1)
    factors.push_back(remaining);
  llvm::sort(factors, std::greater<int64_t>());

  constexpr long double epsilon = 1.0e-12L;
  for (int64_t factor : factors) {
    unsigned bestDim = 0;
    long double bestRatio = std::numeric_limits<long double>::infinity();
    int64_t bestSpan = -1;

    for (auto [idx, extent] : llvm::enumerate(extents)) {
      SmallVector<int64_t, 4> candidate(grid.begin(), grid.end());
      candidate[idx] *= factor;
      long double ratio =
          estimateStencilExpandedTileRatio(extents, haloRadii, candidate);
      int64_t span = ceilDivPositive(extent, candidate[idx]);
      if (ratio + epsilon < bestRatio ||
          (std::abs(ratio - bestRatio) <= epsilon && span > bestSpan)) {
        bestRatio = ratio;
        bestSpan = span;
        bestDim = static_cast<unsigned>(idx);
      }
    }

    grid[bestDim] *= factor;
  }

  return grid;
}


Value buildLogicalWorkerCapacityValue(OpBuilder &builder, Location loc) {
  Value logicalWorkers = SdeResourceQueryOp::create(
                             builder, loc, SdeResourceQueryKind::logicalWorkers)
                             .getResult();
  return arith::MaxUIOp::create(builder, loc, logicalWorkers,
                                createConstantIndex(builder, loc, 1));
}

Value buildTripCountValue(OpBuilder &builder, Location loc, SdeSuIterateOp op) {
  if (std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation()))
    return createConstantIndex(builder, loc, *tripCount);

  if (op.getLowerBounds().size() != 1 || op.getUpperBounds().size() != 1 ||
      op.getSteps().size() != 1)
    return Value();

  Value lowerBound = op.getLowerBounds().front();
  Value upperBound = op.getUpperBounds().front();
  Value step = op.getSteps().front();
  Value zero = createConstantIndex(builder, loc, 0);
  Value one = createConstantIndex(builder, loc, 1);

  Value span = arith::SubIOp::create(builder, loc, upperBound, lowerBound);

  int64_t constantStep = 0;
  Value safeStep = step;
  if (ValueAnalysis::getConstantIndex(step, constantStep)) {
    if (constantStep <= 0)
      return Value();
  } else {
    Value stepIsTooSmall = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sle, step, zero);
    safeStep = arith::SelectOp::create(builder, loc, stepIsTooSmall, one, step);
  }

  Value spanIsNegative = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::slt, span, zero);
  Value nonNegativeSpan =
      arith::SelectOp::create(builder, loc, spanIsNegative, zero, span);
  return arith::CeilDivSIOp::create(builder, loc, nonNegativeSpan, safeStep);
}

} // namespace mlir::carts::sde
