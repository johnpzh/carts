///==========================================================================///
/// File: CodeletUtils.h
///
/// Utilities shared by SDE codelet lowering passes.
///==========================================================================///

#ifndef ARTS_DIALECT_SDE_TRANSFORMS_CODELETUTILS_H
#define ARTS_DIALECT_SDE_TRANSFORMS_CODELETUTILS_H

#include "arts/dialect/sde/IR/SdeDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::carts::sde {

struct ExternalCapturePlan {
  SmallVector<Value> captures;
  SmallVector<Value> pureExternal;
  DenseSet<Value> materializedMemrefs;

  DenseSet<Value> captureSet;
  DenseSet<Value> pureExternalSet;
};

/// Return true when a cu_region has tensor-typed iter_args.
bool hasTensorIterArgs(SdeCuRegionOp region);

/// Classify tensor block-argument uses as read, write, or readwrite.
SdeAccessMode classifyTensorAccess(BlockArgument blockArg);

/// Trace a tensor value at an SDE boundary back to the memref it represents.
Value traceTensorBoundaryMemref(Value tensor);

/// Collect root ops and all nested ops that will move into a codelet body.
DenseSet<Operation *> collectMovedOpTree(ArrayRef<Operation *> opsToMove);

/// Return true when a value is defined by, or scoped under, moved operations.
bool isValueInternalToMovedOps(Value value,
                               const DenseSet<Operation *> &movedOps);

/// Return true when the moved operation tree writes to the given memref value.
bool hasMemrefWriteInMovedOps(Value memref, ArrayRef<Operation *> opsToMove);

/// Plan the explicit captures needed before cloning into an isolated codelet.
LogicalResult planExternalCaptures(
    ArrayRef<Operation *> opsToMove, const DenseSet<Value> &preMappedValues,
    const DenseSet<Value> &materializableMemrefs,
    const DenseSet<Type> &materializableMemrefTypes,
    const DenseSet<Operation *> &movedOps, ExternalCapturePlan &plan);

} // namespace mlir::carts::sde

#endif // ARTS_DIALECT_SDE_TRANSFORMS_CODELETUTILS_H
