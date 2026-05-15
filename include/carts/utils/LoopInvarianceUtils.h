///==========================================================================///
/// File: LoopInvarianceUtils.h
///
/// Consolidated loop invariance checking and hoisting safety utilities.
/// Previously duplicated across Hoisting.cpp, DataPointerHoisting.cpp,
/// and ScalarReplacement.cpp.
///==========================================================================///

#ifndef ARTS_UTILS_LOOPINVARIANCEUTILS_H
#define ARTS_UTILS_LOOPINVARIANCEUTILS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace arts {

/// Check if a value is loop-invariant w.r.t. the given for-loop.
/// Handles numeric casts and constants via ValueUtils.
bool isLoopInvariant(scf::ForOp loop, Value v);

/// Check if a div/rem operation is safe to hoist out of a loop.
/// Verifies all operands are loop-invariant and denominator is provably
/// non-zero.
bool isSafeToHoistDivRem(scf::ForOp loop, Operation *op);

/// Check if a div/rem operation is safe to hoist, using dominance info.
/// Variant used by DataPtrHoisting where dominance checking is needed.
bool isSafeDivRemToHoist(Operation *op, scf::ForOp loop,
                         DominanceInfo &domInfo);

/// Check if a value is defined outside a region.
bool isDefinedOutside(Region &region, Value value);

/// Check if all operands of an operation are defined outside a region.
bool allOperandsDefinedOutside(Operation *op, Region &region);

/// Check if all operands of an operation dominate a given insertion point.
bool allOperandsDominate(Operation *op, Operation *insertionPoint,
                         DominanceInfo &domInfo);

/// Find the outermost enclosing scf::ForOp into which \p op can be legally
/// hoisted.  Walks from \p op outward, checking that all operands of
/// \p addrOp are defined outside each candidate loop and dominate it.
/// Returns nullptr if no hoist target exists.
scf::ForOp findHoistTarget(Operation *op, Operation *addrOp,
                           DominanceInfo &domInfo);

} // namespace arts
} // namespace mlir

#endif // ARTS_UTILS_LOOPINVARIANCEUTILS_H
