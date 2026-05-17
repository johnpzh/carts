///==========================================================================///
/// File: LoopStructureUtils.h
///
/// ARTS loop-structure helpers used by DB/EDT analysis and transforms.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_UTILS_LOOPSTRUCTUREUTILS_H
#define CARTS_DIALECT_ARTS_UTILS_LOOPSTRUCTUREUTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace carts::arts {

/// Collect upper bounds from a while-loop condition for the given iteration
/// argument. Recursively decomposes AND-ed conditions and extracts bounds from
/// less-than / greater-than comparisons.
void collectWhileBounds(Value cond, Value iterArg, SmallVector<Value> &bounds);

/// Compute the loop nesting depth of an operation by counting how many
/// enclosing loop operations surround it.
unsigned getLoopDepth(Operation *op);

} // namespace carts::arts
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_UTILS_LOOPSTRUCTUREUTILS_H
