///==========================================================================///
/// File: Utils.h
///
/// Utility functions for ARTS.
///==========================================================================///

#ifndef CARTS_UTILS_ARTSUTILS_H
#define CARTS_UTILS_ARTSUTILS_H

#include "carts/Dialect.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace carts {

/// Return true when any operand or result of an operation has a floating-point
/// type.
bool operationTouchesFloatingPoint(Operation *op);

/// Constant Index Creation Utilities
Value createConstantIndex(OpBuilder &builder, Location loc, int64_t val);
Value createZeroIndex(OpBuilder &builder, Location loc);
Value createOneIndex(OpBuilder &builder, Location loc);

/// Return true when `v` dominates `op` or is defined in an ancestor region
/// of `op`. Handles both op-result and block-argument values.
bool dominatesOrInAncestor(Value v, Operation *op, DominanceInfo &domInfo);

/// Return true for regionless arithmetic-like operations that are safe to
/// duplicate when restructuring control flow around epochs/EDTs.
bool isSideEffectFreeArithmeticLikeOp(Operation *op);

/// Return true when non-terminator operations exist after `op` in its parent
/// block. Used to decide whether a trailing synchronization barrier is needed.
bool hasWorkAfterInParentBlock(Operation *op);

/// Clamp dependency indices to valid memref bounds [0, dimSize-1].
SmallVector<Value> clampDepIndices(Value source, ArrayRef<Value> indices,
                                   OpBuilder &builder, Location loc,
                                   ArrayRef<Value> dimSizes = {});

/// Return true when `op` is nested inside an OMP dialect region.
bool isInsideOmpRegion(Operation *op);

/// Return true when `op` transitively contains any OMP dialect operation.
bool containsOmpOp(Operation *op);

namespace arts {

/// Type and Size Utilities
uint64_t getElementTypeByteSize(Type elementType);
MemRefType getElementMemRefType(Type elementType, ArrayRef<Value> elementSizes);

/// Access Mode Utilities
ArtsMode combineAccessModes(ArtsMode mode1, ArtsMode mode2);

/// Create the ARTS runtime route sentinel for "run/create on the current node"
/// when no explicit destination rank is requested.
Value createCurrentNodeRoute(OpBuilder &builder, Location loc);

/// ARTS Runtime Query Utilities
bool isArtsRuntimeQuery(Value val);

/// Region Replacement Utilities
void replaceInRegion(Region &region, Value from, Value to);
void replaceInRegion(Region &region, DenseMap<Value, Value> &rewireMap,
                     bool clear = true);

/// Return true when `op` is an undef-like operation (llvm.mlir.undef,
/// polygeist.undef, or arts.undef).
bool isUndefLikeOp(Operation *op);

} // namespace arts
} // namespace carts
} // namespace mlir

#endif // CARTS_UTILS_ARTSUTILS_H
