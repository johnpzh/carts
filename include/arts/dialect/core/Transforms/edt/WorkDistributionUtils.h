///==========================================================================///
/// File: WorkDistributionUtils.h
///
/// Lowering-time helpers for emitting runtime work-distribution state.
///
/// Policy selection stays in EdtHeuristics / DistributionHeuristics. This
/// utility consumes the chosen policy and materializes worker counts, chunking
/// plans, and per-worker bounds as SSA values for lowering.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_EDT_WORKDISTRIBUTIONUTILS_H
#define ARTS_DIALECT_CORE_TRANSFORMS_EDT_WORKDISTRIBUTIONUTILS_H

#include "arts/dialect/core/Analysis/heuristics/DistributionHeuristics.h"
#include "arts/dialect/core/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "arts/utils/LoweringContractUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include <optional>
#include <utility>

namespace mlir {
namespace arts {

class DbAnalysis;

/// Runtime distribution bounds (SSA Values emitted during lowering).
struct DistributionBounds {
  /// Thread-level: inner loop bounds.
  Value iterStart;
  Value iterCount;
  Value iterCountHint;
  Value hasWork;

  /// Acquire-level: DB acquire bounds.
  Value acquireStart;
  Value acquireSize;
  Value acquireSizeHint;
};

/// 2D worker grid for tiling_2d strategy.
/// workerId = rowWorkerId * colWorkers + colWorkerId
struct Tiling2DWorkerGrid {
  Value rowWorkers;
  Value colWorkers;
  Value rowWorkerId;
  Value colWorkerId;
};

/// Shared chunking plan for lowering one distributed arts.for loop.
struct LoopChunkPlan {
  Value blockSize;
  Value chunkLowerBound;
  Value totalIterations;
  Value totalChunks;
  bool useAlignedLowerBound = false;
  bool useRuntimeBlockAlignment = false;
  std::optional<int64_t> alignmentBlockSize;
  Value runtimeAlignmentBlockSize;
};

class WorkDistributionUtils {
public:
  static LoopChunkPlan planLoopChunking(ArtsCodegen *AC, ForOp forOp,
                                        const LoweringContractInfo &contract,
                                        Value runtimeBlockSizeHint = Value());

  static DistributionBounds computeBounds(ArtsCodegen *AC, Location loc,
                                          const DistributionStrategy &strategy,
                                          Value workerId, Value totalWorkers,
                                          Value workersPerNode,
                                          Value totalIterations,
                                          Value totalChunks, Value blockSize,
                                          const LoweringContractInfo &contract);

  static DistributionBounds recomputeBoundsInside(
      ArtsCodegen *AC, Location loc, const DistributionStrategy &strategy,
      Value workerId, Value insideTotalWorkers, Value workersPerNode,
      Value upperBound, Value lowerBound, Value loopStep, Value blockSize,
      std::optional<int64_t> alignmentBlockSize,
      Value runtimeAlignmentBlockSize, bool useRuntimeBlockAlignment,
      const LoweringContractInfo &contract);

  static Value computeDbAlignmentBlockSize(ForOp forOp, EdtOp parallelEdt,
                                           Value numPartitions, ArtsCodegen *AC,
                                           Location loc,
                                           DbAnalysis &dbAnalysis);

  static Value getWorkersPerNode(OpBuilder &builder, Location loc,
                                 EdtOp parallelEdt);
  static Value getWorkersPerNode(ArtsCodegen *AC, Location loc,
                                 EdtOp parallelEdt);

  static Value getTotalWorkers(OpBuilder &builder, Location loc,
                               EdtOp parallelEdt);
  static Value getTotalWorkers(ArtsCodegen *AC, Location loc,
                               EdtOp parallelEdt);

  static Value getDispatchWorkerCount(OpBuilder &builder, Location loc,
                                      EdtOp parallelEdt);

  static Value getForDispatchWorkerCount(ArtsCodegen *AC, Location loc,
                                         EdtOp parallelEdt,
                                         const DistributionStrategy &strategy,
                                         Value totalChunks,
                                         const LoweringContractInfo &contract);

  static Tiling2DWorkerGrid getTiling2DWorkerGrid(
      ArtsCodegen *AC, Location loc, Value workerId, Value totalWorkers,
      const LoweringContractInfo &contract, Value workersPerNode = Value());

private:
  static std::pair<Value, Value>
  balancedDistribute(ArtsCodegen *AC, Location loc, Value totalChunks,
                     Value numParticipants, Value participantId);
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_EDT_WORKDISTRIBUTIONUTILS_H
