///==========================================================================///
/// File: IterationSizingUtils.h
///
/// SDE-owned iteration sizing helpers for source-level scheduling intent.
///==========================================================================///

#ifndef CARTS_DIALECT_SDE_UTILS_ITERATIONSIZINGUTILS_H
#define CARTS_DIALECT_SDE_UTILS_ITERATIONSIZINGUTILS_H

#include "carts/dialect/sde/IR/SdeDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include <cstdint>

namespace mlir::carts::sde {

int64_t ceilDivPositive(int64_t value, int64_t divisor);

Value buildLogicalWorkerCapacityValue(OpBuilder &builder, Location loc);

Value buildTripCountValue(OpBuilder &builder, Location loc, SdeSuIterateOp op);

} // namespace mlir::carts::sde

#endif // CARTS_DIALECT_SDE_UTILS_ITERATIONSIZINGUTILS_H
