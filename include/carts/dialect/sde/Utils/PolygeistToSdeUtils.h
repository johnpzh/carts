///==========================================================================///
/// File: PolygeistToSdeUtils.h
///
/// Shared helpers for the Polygeist-to-SDE conversion passes.
///==========================================================================///

#ifndef CARTS_DIALECT_SDE_UTILS_POLYGEISTTOSDEUTILS_H
#define CARTS_DIALECT_SDE_UTILS_POLYGEISTTOSDEUTILS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir::carts::sde {

/// Build a memref subview of `source` covering the region described by
/// `indices` (offsets) and `sizes`. Missing trailing dims default to offset 0
/// and the source dim size (DimOp for dynamic dims). Returns `source`
/// directly when no slicing is requested.
Value materializeDependView(OpBuilder &builder, Location loc, Value source,
                            ArrayRef<Value> indices, ArrayRef<Value> sizes);

/// Clamp dependency indices to valid memref bounds [0, dimSize - 1].
SmallVector<Value> clampDepIndices(Value source, ArrayRef<Value> indices,
                                   OpBuilder &builder, Location loc,
                                   ArrayRef<Value> dimSizes = {});

/// Return true when `op` is nested inside an OMP dialect region.
bool isInsideOmpRegion(Operation *op);

/// Return true when `op` transitively contains any OMP dialect operation.
bool containsOmpOp(Operation *op);

} // namespace mlir::carts::sde

#endif // CARTS_DIALECT_SDE_UTILS_POLYGEISTTOSDEUTILS_H
