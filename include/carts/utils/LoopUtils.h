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

/// Return the nearest enclosing loop-like op that contains the given operation.
/// Searches for loop-like operations and omp.wsloop.
inline Operation *findNearestLoop(Operation *op) {
  for (Operation *cur = op->getParentOp(); cur; cur = cur->getParentOp()) {
    if (isa<LoopLikeOpInterface>(cur) || isa<omp::WsloopOp>(cur))
      return cur;
  }
  return nullptr;
}

/// Resolve a constant trip count for a loop-like op when all bounds are static.
/// Returns std::nullopt when the trip count cannot be proven statically.
std::optional<int64_t> getStaticTripCount(Operation *loopOp);

} // namespace carts
} // namespace mlir

#endif // CARTS_UTILS_LOOPUTILS_H
