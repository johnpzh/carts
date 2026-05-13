///==========================================================================///
/// File: EdtTaskLoopLowering.cpp
///
/// Factory for strategy-specific task loop lowerers.
///
/// Before:
///   arts.for (%i = %lb to %ub step %step) { ... }
///   // no worker-local slice yet
///
/// After:
///   // planner computes one worker slice
///   %worker_lb = ...
///   %worker_sz = ...
///   arts.db_acquire ... {partition_offsets = [%worker_lb],
///                        partition_sizes = [%worker_sz]}
///==========================================================================///

#include "arts/dialect/core/Transforms/edt/EdtTaskLoopLowering.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::arts;

TaskAcquirePlanningResult
EdtTaskLoopLowering::planAcquireRewrite(const TaskLoopLoweringInput &input,
                                        Value chunkOffset) const {
  TaskAcquirePlanningResult result;
  if (!input.AC)
    return result;

  Value one = input.AC->createIndexConstant(1, input.loc);
  Value stepVal = input.loopStep ? input.loopStep : one;
  int64_t stepConst = 0;
  bool stepIsUnit =
      !stepVal ||
      (ValueAnalysis::getConstantIndex(stepVal, stepConst) && stepConst == 1);

  Value lowerBoundVal =
      input.chunkLowerBound ? input.chunkLowerBound : input.lowerBound;
  Value workerOffsetVal = chunkOffset;
  if (!stepIsUnit)
    workerOffsetVal =
        input.AC->create<arith::MulIOp>(input.loc, workerOffsetVal, stepVal);
  workerOffsetVal = input.AC->create<arith::AddIOp>(input.loc, lowerBoundVal,
                                                    workerOffsetVal);

  result.workerBaseOffset = workerOffsetVal;
  result.acquireOffset = workerOffsetVal;
  result.acquireSize = input.bounds.iterCount;
  result.acquireHintSize = input.bounds.iterCountHint
                               ? input.bounds.iterCountHint
                               : result.acquireSize;
  result.step = stepVal;
  result.stepIsUnit = stepIsUnit;
  return result;
}

TaskLoopLoweringResult EdtTaskLoopLowering::lowerBlockStyle(
    TaskLoopLoweringInput &input,
    const TaskLoopLoweringMappedValues &mapped) const {
  TaskLoopLoweringResult result;

  Value zero = input.AC->createIndexConstant(0, input.loc);
  Value one = input.AC->createIndexConstant(1, input.loc);
  result.innerStripeLane = zero;
  result.innerStripeCount = one;

  result.insideBounds = WorkDistributionUtils::recomputeBoundsInside(
      input.AC, input.loc, input.strategy, input.taskWorkerId,
      input.totalWorkers, input.workersPerNode, input.upperBound,
      input.lowerBound, input.loopStep, input.blockSize,
      input.alignmentBlockSize, input.runtimeAlignmentBlockSize,
      input.useRuntimeBlockAlignment,
      input.distributionContract);
  result.iterStart = result.insideBounds.iterStart;
  result.globalBase = mapped.workerBaseOffset;

  /// Derive the task-local loop window from the true loop bounds and the
  /// worker's aligned base offset. This remains correct for the aligned-down
  /// chunking case and simplifies to the original [0, iterCount) window when
  /// the worker base already matches the logical lower bound.
  Value stepClamped =
      input.AC->create<arith::MaxUIOp>(input.loc, mapped.step, one);

  auto clampNonNeg = [&](Value v) -> Value {
    Value isNeg = input.AC->create<arith::CmpIOp>(
        input.loc, arith::CmpIPredicate::slt, v, zero);
    return input.AC->create<arith::SelectOp>(input.loc, isNeg, zero, v);
  };
  auto ceilDiv = [&](Value num, Value denom) -> Value {
    Value denomMinusOne =
        input.AC->create<arith::SubIOp>(input.loc, denom, one);
    Value adjusted =
        input.AC->create<arith::AddIOp>(input.loc, num, denomMinusOne);
    return input.AC->create<arith::DivUIOp>(input.loc, adjusted, denom);
  };

  Value diffLower = input.AC->create<arith::SubIOp>(
      input.loc, mapped.lowerBound, mapped.workerBaseOffset);
  Value diffUpper = input.AC->create<arith::SubIOp>(
      input.loc, mapped.upperBound, mapped.workerBaseOffset);
  Value diffLowerPos = clampNonNeg(diffLower);
  Value diffUpperPos = clampNonNeg(diffUpper);

  Value loopLower = ceilDiv(diffLowerPos, stepClamped);
  Value loopUpper = ceilDiv(diffUpperPos, stepClamped);
  loopUpper = input.AC->create<arith::MinUIOp>(input.loc, loopUpper,
                                               result.insideBounds.iterCount);

  result.iterLoop =
      input.AC->create<scf::ForOp>(input.loc, loopLower, loopUpper, one);
  return result;
}

std::unique_ptr<EdtTaskLoopLowering>
EdtTaskLoopLowering::create(DistributionKind kind) {
  switch (kind) {
  case DistributionKind::Flat:
  case DistributionKind::Replicated:
    /// TODO(DT-5): When DB replication lowering is implemented
    /// (arts_add_db_duplicate), replicated distributions should use a
    /// broadcast-aware task loop. Until then, use block lowering.
    return detail::createBlockTaskLoopLowering();
  case DistributionKind::TwoLevel:
    return detail::createBlockTaskLoopLowering();
  case DistributionKind::BlockCyclic:
    return detail::createBlockCyclicTaskLoopLowering();
  case DistributionKind::Tiling2D:
    return detail::createTile2DTaskLoopLowering();
  }
  llvm_unreachable("unknown DistributionKind");
}
