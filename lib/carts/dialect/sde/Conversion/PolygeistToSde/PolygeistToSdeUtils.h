///==========================================================================///
/// File: PolygeistToSdeUtils.h
///
/// Shared helpers for the Polygeist-to-SDE conversion passes.
///==========================================================================///

#ifndef CARTS_DIALECT_SDE_CONVERSION_POLYGEISTTOSDE_UTILS_H
#define CARTS_DIALECT_SDE_CONVERSION_POLYGEISTTOSDE_UTILS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir::carts::arts {

/// Build a memref subview of `source` covering the region described by
/// `indices` (offsets) and `sizes`. Missing trailing dims default to offset 0
/// and the source dim size (DimOp for dynamic dims). Returns `source`
/// directly when no slicing is requested.
Value materializeDependView(OpBuilder &builder, Location loc, Value source,
                            ArrayRef<Value> indices, ArrayRef<Value> sizes);

} // namespace mlir::carts::arts

#endif // CARTS_DIALECT_SDE_CONVERSION_POLYGEISTTOSDE_UTILS_H
