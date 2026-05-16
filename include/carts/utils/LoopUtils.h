///==========================================================================///
/// File: LoopUtils.h
///
/// Utility functions for querying SCF loop properties.
/// Lightweight inline helpers for worker-loop detection, innermost-loop
/// checks, and bound matching — complements the heavier LoopAnalysis
/// framework without requiring an AnalysisManager.
///==========================================================================///

#ifndef ARTS_UTILS_LOOPUTILS_H
#define ARTS_UTILS_LOOPUTILS_H

#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include <optional>

namespace mlir {
namespace carts::arts {

class EdtOp;
class LoopNode;

/// Check whether a scf::ForOp is a "worker loop" (i.e., contains at least one
/// arts.edt operation anywhere in its body).
/// Used by epoch-level passes to identify task-spawning loops.
bool isWorkerLoop(scf::ForOp loop);

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

/// Check whether two scf::ForOp loops have matching bounds (same lower bound,
/// upper bound, and step). Uses ValueAnalysis::sameValue for comparison.
inline bool haveCompatibleBounds(scf::ForOp a, scf::ForOp b) {
  return ValueAnalysis::sameValue(a.getLowerBound(), b.getLowerBound()) &&
         ValueAnalysis::sameValue(a.getUpperBound(), b.getUpperBound()) &&
         ValueAnalysis::sameValue(a.getStep(), b.getStep());
}

/// Return the induction variable of a loop operation. Returns a null Value for
/// unsupported loop types or if the loop body is empty.
Value getLoopInductionVar(Operation *op);

/// Check whether a value is a loop induction variable (i.e., a BlockArgument
/// listed as an induction variable by its parent loop construct).
inline bool isLoopInductionVar(Value value) {
  auto arg = dyn_cast_or_null<BlockArgument>(value);
  if (!arg)
    return false;

  Operation *parent = arg.getOwner()->getParentOp();
  if (!parent)
    return false;

  if (auto loopNest = dyn_cast<omp::LoopNestOp>(parent)) {
    for (BlockArgument iv : loopNest.getIVs())
      if (iv == arg)
        return true;
    return false;
  }

  if (auto loopLike = dyn_cast<LoopLikeOpInterface>(parent)) {
    if (auto ivs = loopLike.getLoopInductionVars()) {
      for (Value iv : *ivs)
        if (iv == value)
          return true;
    }
  }
  return false;
}

/// Collect upper bounds from a while-loop condition for the given iteration
/// argument. Recursively decomposes AND-ed conditions and extracts bounds
/// from less-than / greater-than comparisons.
void collectWhileBounds(Value cond, Value iterArg, SmallVector<Value> &bounds);

/// Compute the loop nesting depth of an operation by counting how many
/// enclosing loop operations surround it.
unsigned getLoopDepth(Operation *op);

/// Returns true if the EDT's body contains any loop operations
/// (scf::ForOp, scf::ParallelOp, affine::AffineForOp).
bool containsLoop(arts::EdtOp edt);

/// Return the nearest enclosing loop-like op that contains the given operation.
/// Searches for loop-like operations and omp.wsloop.
inline Operation *findNearestLoop(Operation *op) {
  for (Operation *cur = op->getParentOp(); cur; cur = cur->getParentOp()) {
    if (isa<LoopLikeOpInterface>(cur) || isa<omp::WsloopOp>(cur))
      return cur;
  }
  return nullptr;
}

/// Return true when a loop lower bound is provably zero, including through
/// select-based clamping patterns like max(0, expr).
bool isProvablyZeroLoopLowerBound(Value lb);

/// Return true when a LoopNode covers the full iteration range [0, dimSize)
/// with unit step.
bool isLoopFullRange(LoopNode *loop, Value dimSize);

/// Resolve a constant trip count for a loop-like op when all bounds are static.
/// Returns std::nullopt when the trip count cannot be proven statically.
std::optional<int64_t> getStaticTripCount(Operation *loopOp);

/// Compute a capped product of static enclosing loop trip counts.
/// Starts at the parent of `op`, so the operation's own trip count is excluded.
/// Useful for repeated-dispatch cost models that need a lightweight estimate
/// of how many times a region is re-entered.
int64_t getRepeatedParentTripProduct(Operation *op,
                                     int64_t maxProduct = 1 << 20);

/// Return true when a type is a floating-point type (F16, BF16, F32, F64, F80,
/// F128) or a vector of one.
bool hasFloatingPointType(Type type);

/// Return true when any operand or result of an operation has a floating-point
/// type.
bool operationTouchesFloatingPoint(Operation *op);

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_UTILS_LOOPUTILS_H
