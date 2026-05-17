///==========================================================================///
/// File: IterationSizingUtils.cpp
///
/// SDE-owned iteration sizing helpers for source-level scheduling intent.
///==========================================================================///

#include "carts/dialect/sde/Utils/IterationSizingUtils.h"

#include "carts/utils/LoopUtils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::carts;

namespace mlir::carts::sde {

int64_t ceilDivPositive(int64_t value, int64_t divisor) {
  return llvm::divideCeil(std::max<int64_t>(1, value),
                          std::max<int64_t>(1, divisor));
}

static Value getConstantIndex(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

Value buildLogicalWorkerCapacityValue(OpBuilder &builder, Location loc) {
  Value logicalWorkers = SdeResourceQueryOp::create(
                             builder, loc, SdeResourceQueryKind::logicalWorkers)
                             .getResult();
  return arith::MaxUIOp::create(builder, loc, logicalWorkers,
                                getConstantIndex(builder, loc, 1));
}

Value buildTripCountValue(OpBuilder &builder, Location loc, SdeSuIterateOp op) {
  if (std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation()))
    return getConstantIndex(builder, loc, *tripCount);

  if (op.getLowerBounds().size() != 1 || op.getUpperBounds().size() != 1 ||
      op.getSteps().size() != 1)
    return Value();

  Value lowerBound = op.getLowerBounds().front();
  Value upperBound = op.getUpperBounds().front();
  Value step = op.getSteps().front();
  Value zero = getConstantIndex(builder, loc, 0);
  Value one = getConstantIndex(builder, loc, 1);

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
