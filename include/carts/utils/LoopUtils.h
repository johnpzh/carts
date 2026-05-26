///==========================================================================///
/// File: LoopUtils.h
///
/// Utility functions for querying SCF loop properties.
/// Lightweight helpers for innermost-loop checks, loop IV recognition, and
/// trip-count estimates. Complements the heavier LoopAnalysis framework
/// without requiring an AnalysisManager.
///==========================================================================///

#ifndef CARTS_UTILS_LOOPUTILS_H
#define CARTS_UTILS_LOOPUTILS_H

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include <optional>

namespace mlir {
namespace carts {

/// Check whether a scf::ForOp is the innermost loop (contains no nested
/// scf::ForOp operations). Used by strip-mining and other loop transforms
/// that target leaf loops only.
inline bool isInnermostLoop(scf::ForOp loop) {
  bool hasNested = false;
  loop.getBody()->walk([&](scf::ForOp nested) {
    if (nested != loop) {
      hasNested = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !hasNested;
}

/// Resolve a constant trip count for a loop-like op when all bounds are static.
/// Returns std::nullopt when the trip count cannot be proven statically.
std::optional<int64_t> getStaticTripCount(Operation *loopOp);

/// Check if a value is loop-invariant w.r.t. the given for-loop.
/// Handles numeric casts and constants via ValueAnalysis.
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

} // namespace carts
} // namespace mlir

#endif // CARTS_UTILS_LOOPUTILS_H
