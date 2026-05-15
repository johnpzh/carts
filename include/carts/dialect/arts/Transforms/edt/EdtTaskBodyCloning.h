///==========================================================================///
/// File: EdtTaskBodyCloning.h
///
/// Helpers for cloning task bodies while preserving Core EDT capture and
/// task-local stack allocation semantics.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTTASKBODYCLONING_H
#define ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTTASKBODYCLONING_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace carts::arts {

/// Collect external values needed by operations in a block, including nested
/// regions, while respecting the task boundary being cloned into.
void collectExternalValues(Block &sourceBlock, Region *boundaryRegion,
                           llvm::SetVector<Value> &externalValues,
                           const llvm::DenseSet<Operation *> &opsToSkip);

/// Clone external stack allocations into the EDT region and remap uses.
void cloneExternalAllocasIntoEdt(Region *taskEdtRegion, Block &taskBlock,
                                 IRMapping &mapper, OpBuilder &builder);

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTTASKBODYCLONING_H
