///==========================================================================///
/// File: Utils.h
///
/// Utility functions for ARTS.
///==========================================================================///

#ifndef CARTS_UTILS_ARTSUTILS_H
#define CARTS_UTILS_ARTSUTILS_H

#include "arts/Dialect.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include <optional>

namespace mlir {
namespace arts {

/// IR Simplification Utilities
bool simplifyIR(ModuleOp module, DominanceInfo &domInfo);

/// Type and Size Utilities
uint64_t getElementTypeByteSize(Type elementType);
MemRefType getElementMemRefType(Type elementType, ArrayRef<Value> elementSizes);

/// String Utilities
std::string sanitizeString(StringRef s);

/// Access Mode Utilities
ArtsMode combineAccessModes(ArtsMode mode1, ArtsMode mode2);

/// Constant Index Creation Utilities
Value createConstantIndex(OpBuilder &builder, Location loc, int64_t val);
Value createZeroIndex(OpBuilder &builder, Location loc);
Value createOneIndex(OpBuilder &builder, Location loc);
Value createCeilDivUI(OpBuilder &builder, Location loc, Value numerator,
                      Value denominator);

/// Route sentinel used by the ARTS runtime to mean "run/create on the current
/// node" when no explicit destination rank is requested.
inline constexpr int64_t kCurrentNodeRoute = -1;
Value createCurrentNodeRoute(OpBuilder &builder, Location loc);

/// ARTS Runtime Query Utilities
bool isArtsRuntimeQuery(Value val);

/// Operation Replacement Utilities
void replaceUses(Value from, Value to, DominanceInfo &domInfo,
                 Operation *dominatingOp);
void replaceUses(DenseMap<Value, Value> &rewireMap);
void replaceInRegion(Region &region, Value from, Value to);
void replaceInRegion(Region &region, DenseMap<Value, Value> &rewireMap,
                     bool clear = true);

/// Transfer attributes from source to destination operation.
void transferAttributes(Operation *src, Operation *dst,
                        ArrayRef<StringRef> excludes = {});

/// Return true when `v` dominates `op` or is defined in an ancestor region
/// of `op`. Handles both op-result and block-argument values.
bool dominatesOrInAncestor(Value v, Operation *op, DominanceInfo &domInfo);

/// Check if a value is defined inside a given region (either as a block
/// argument owned by a block in the region, or as the result of an operation
/// whose parent region is within the given region).
inline bool isDefinedInRegion(Value val, Region *region) {
  if (auto blockArg = dyn_cast<BlockArgument>(val))
    return region->isAncestor(blockArg.getOwner()->getParent());
  if (Operation *defOp = val.getDefiningOp())
    return region->isAncestor(defOp->getParentRegion());
  return false;
}

/// Whitelist check for pure arithmetic operations that are safe to clone
/// into EDT bodies (start-block cloning). Returns true for constant, div,
/// rem, add, sub, mul, max, min, select, and index_cast operations.
inline bool isStartBlockArithmeticOp(Operation *op) {
  return isa<arith::ConstantIndexOp, arith::ConstantIntOp, arith::DivUIOp,
             arith::DivSIOp, arith::RemUIOp, arith::RemSIOp, arith::AddIOp,
             arith::SubIOp, arith::MulIOp, arith::MaxUIOp, arith::MinUIOp,
             arith::SelectOp, arith::IndexCastOp>(op);
}

/// Return true for regionless arithmetic-like operations that are safe to
/// duplicate when restructuring control flow around epochs/EDTs.
bool isSideEffectFreeArithmeticLikeOp(Operation *op);

/// Return true when `op` is nested inside an OMP dialect region.
bool isInsideOmpRegion(Operation *op);

/// Return true when `op` transitively contains any OMP dialect operation.
bool containsOmpOp(Operation *op);

/// Walk up from `op` to `parentFunc` and return the immediate child of
/// `parentFunc` that contains `op`. Returns nullptr when unreachable.
Operation *getTopLevelFuncChild(Operation *op, func::FuncOp parentFunc);

/// Return true when `candidate` is positioned after `anchor` in the top-level
/// block of `parentFunc`.
bool isOrderedAfter(Operation *anchor, Operation *candidate,
                    func::FuncOp parentFunc);

/// Convert OMP task dependency mode to ARTS access mode.
ArtsMode convertOmpMode(omp::ClauseTaskDepend mode);

/// Clamp dependency indices to valid memref bounds [0, dimSize-1].
SmallVector<Value> clampDepIndices(Value source, ArrayRef<Value> indices,
                                   OpBuilder &builder, Location loc,
                                   ArrayRef<Value> dimSizes = {});

/// Return true when non-terminator operations exist after `op` in its parent
/// block. Used to decide whether a trailing synchronization barrier is needed.
bool hasWorkAfterInParentBlock(Operation *op);

/// Return true for operations considered "pure" for stencil/pattern analysis:
/// known loop/load/store ops, side-effect-free regionless ops, and ops
/// implementing MemoryEffectOpInterface with no effects. Returns false for
/// call-like operations.
bool isPureOp(Operation *op);

/// Sort memref::StoreOp operations by their position in the parent block.
/// Stores in the same block are ordered by `isBeforeInBlock`; stores in
/// different blocks fall back to pointer comparison for a stable order.
void sortStoresInProgramOrder(MutableArrayRef<memref::StoreOp> stores);

/// Return true when `op` is an undef-like operation (llvm.mlir.undef,
/// polygeist.undef, or arts.undef).
bool isUndefLikeOp(Operation *op);

} // namespace arts
} // namespace mlir

#endif // CARTS_UTILS_ARTSUTILS_H
