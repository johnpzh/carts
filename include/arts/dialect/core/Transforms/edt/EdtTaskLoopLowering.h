///==========================================================================///
/// File: EdtTaskLoopLowering.h
///
/// Strategy-specific task loop lowering for ForLowering.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTTASKLOOPLOWERING_H
#define ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTTASKLOOPLOWERING_H

#include "arts/dialect/core/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "arts/dialect/core/Transforms/edt/WorkDistributionUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SetVector.h"
#include <memory>
#include <optional>

namespace mlir {
namespace arts {

struct TaskLoopLoweringInput {
  ArtsCodegen *AC = nullptr;
  Location loc;

  DistributionStrategy strategy;
  DistributionBounds bounds;
  LoweringContractInfo distributionContract;
  Value taskWorkerId;
  Value totalWorkers;
  Value workersPerNode;

  Value upperBound;
  Value lowerBound;
  Value chunkLowerBound;
  Value loopStep;
  Value blockSize;
  Value totalIterations;
  Value totalChunks;

  std::optional<int64_t> alignmentBlockSize;
  Value runtimeAlignmentBlockSize;
  bool useRuntimeBlockAlignment = false;
  bool useAlignedLowerBound = false;
};

struct TaskLoopLoweringMappedValues {
  Value step;
  Value lowerBound;
  Value upperBound;
  Value workerBaseOffset;

  Value blockSize;
  Value totalIterations;
  Value totalChunks;
};

struct TaskLoopLoweringResult {
  DistributionBounds insideBounds;
  Value iterStart;
  Value globalBase;
  scf::ForOp iterLoop;
  Value innerStripeLane;
  Value innerStripeCount;
};

struct TaskAcquirePlanningResult {
  Value workerBaseOffset;
  Value acquireOffset;
  Value acquireSize;
  Value acquireHintSize;
  Value step;
  bool stepIsUnit = true;
  std::optional<Tiling2DWorkerGrid> tiling2DGrid;
};

struct TaskLoopPostCloneInput {
  ArtsCodegen *AC = nullptr;
  Location loc;
  scf::ForOp iterLoop;
  Value globalIter;
  Value innerStripeLane;
  Value innerStripeCount;
};

class EdtTaskLoopLowering {
public:
  virtual ~EdtTaskLoopLowering() = default;

  virtual TaskLoopLoweringResult
  lower(TaskLoopLoweringInput &input,
        const TaskLoopLoweringMappedValues &mapped) const = 0;

  virtual void
  collectExtraExternalValues(const TaskLoopLoweringInput &input,
                             llvm::SetVector<Value> &values) const {}

  /// Strategy hook for planning parent-acquire rewrite bounds.
  virtual TaskAcquirePlanningResult
  planAcquireRewrite(const TaskLoopLoweringInput &input,
                     Value chunkOffset) const;

  /// Strategy hook for optional post-clone rewrites in the generated task body.
  virtual void postCloneAdjust(TaskLoopPostCloneInput &input) const {}

  static std::unique_ptr<EdtTaskLoopLowering> create(DistributionKind kind);

protected:
  /// Shared lowering for block-based strategies (block, two-level, tiling2D).
  /// Calls recomputeBoundsInside and applies aligned-lower-bound adjustment.
  TaskLoopLoweringResult
  lowerBlockStyle(TaskLoopLoweringInput &input,
                  const TaskLoopLoweringMappedValues &mapped) const;
};

namespace detail {

std::unique_ptr<EdtTaskLoopLowering> createBlockTaskLoopLowering();
std::unique_ptr<EdtTaskLoopLowering> createBlockCyclicTaskLoopLowering();
std::unique_ptr<EdtTaskLoopLowering> createTile2DTaskLoopLowering();

} // namespace detail

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTTASKLOOPLOWERING_H
