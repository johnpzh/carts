///==========================================================================///
/// File: HeuristicUtils.cpp
///
/// Shared heuristic building blocks for H1 (Partitioning) and H2
/// (Distribution) heuristic families.
///==========================================================================///

#include "carts/dialect/arts/Analysis/heuristics/HeuristicUtils.h"
#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

int64_t mlir::carts::arts::ceilDivPositive(int64_t num, int64_t denom) {
  if (num <= 0 || denom <= 0)
    return 1;
  return 1 + (num - 1) / denom;
}

int64_t mlir::carts::arts::clampPositive(int64_t value, int64_t minValue,
                                  int64_t maxValue) {
  return std::clamp(std::max<int64_t>(1, value), std::max<int64_t>(1, minValue),
                    std::max<int64_t>(1, maxValue));
}

int64_t mlir::carts::arts::saturatingMulPositive(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0)
    return 1;
  if (lhs >= std::numeric_limits<int64_t>::max() / rhs)
    return std::numeric_limits<int64_t>::max();
  return lhs * rhs;
}

int64_t mlir::carts::arts::floorLog2Positive(int64_t value) {
  if (value <= 1)
    return 0;

  int64_t result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}
