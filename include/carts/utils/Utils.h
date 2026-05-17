///==========================================================================///
/// File: Utils.h
///
/// Dialect-neutral CARTS IR utility functions.
///==========================================================================///

#ifndef CARTS_UTILS_UTILS_H
#define CARTS_UTILS_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace carts {

/// Constant Index Creation Utilities
Value createConstantIndex(OpBuilder &builder, Location loc, int64_t val);
Value createZeroIndex(OpBuilder &builder, Location loc);
Value createOneIndex(OpBuilder &builder, Location loc);

/// Return true for regionless arithmetic-like operations that are safe to
/// duplicate when restructuring control flow around epochs/EDTs.
bool isSideEffectFreeArithmeticLikeOp(Operation *op);

} // namespace carts
} // namespace mlir

#endif // CARTS_UTILS_UTILS_H
