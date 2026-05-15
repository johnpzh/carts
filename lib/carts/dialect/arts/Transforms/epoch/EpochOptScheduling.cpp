///==========================================================================///
/// File: EpochOptScheduling.cpp
///
/// Scheduling-oriented epoch transforms used by EpochOpt.
///==========================================================================///

#include "carts/dialect/arts/Transforms/EpochOptInternal.h"
#include "carts/utils/Debug.h"

ARTS_DEBUG_SETUP(epoch_opt);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace mlir::carts::arts::epoch_opt {

LogicalResult
transformToContinuation(EpochOp epochOp,
                        const EpochContinuationDecision &decision) {
  (void)decision;
  ARTS_INFO("  Skipping epoch continuation: continuations must be represented "
            "as codelets before ARTS EDT materialization");
  return failure();
}

} // namespace mlir::carts::arts::epoch_opt
