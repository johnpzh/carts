///==========================================================================///
/// File: WorkDistributionUtils.cpp
///
/// Lowering-time helpers for materializing runtime work-distribution state.
///==========================================================================///

#include "arts/dialect/core/Transforms/edt/WorkDistributionUtils.h"
#include "arts/dialect/core/Analysis/db/DbAnalysis.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/PartitionPredicates.h"
#include "arts/utils/Utils.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

using namespace mlir;
using namespace mlir::arts;

namespace {

/// ARTS runtime query ops can be cloned safely while rematerializing SSA
/// values into task-local regions.
static bool isSafeRuntimeQuery(Operation *op) {
  if (auto queryOp = dyn_cast<RuntimeQueryOp>(op)) {
    auto kind = queryOp.getKind();
    return kind == RuntimeQueryKind::totalNodes ||
           kind == RuntimeQueryKind::totalWorkers;
  }
  return false;
}

static std::optional<int64_t> getExplicitWorkerCount(EdtOp edt) {
  if (!edt)
    return std::nullopt;
  return arts::getWorkers(edt.getOperation());
}

static std::optional<int64_t> getExplicitWorkersPerNodeCount(EdtOp edt) {
  if (!edt)
    return std::nullopt;
  return arts::getWorkersPerNode(edt.getOperation());
}

static std::optional<int64_t> getExplicitLoopBlockHint(ForOp forOp) {
  if (!forOp)
    return std::nullopt;
  if (auto hint = getPartitioningHint(forOp.getOperation()))
    if (usesBlockLayout(hint->mode) && hint->blockSize && *hint->blockSize > 0)
      return hint->blockSize;
  return std::nullopt;
}

static bool
shouldHonorLoopBlockHintForDbAlignment(const LoweringContractInfo &contract,
                                       EdtOp parallelEdt) {
  if (!parallelEdt)
    return false;
  return parallelEdt.getConcurrency() == EdtConcurrency::intranode &&
         contract.shouldHonorLoopBlockHintForDbAlignment();
}

static std::optional<int64_t> getStaticRuntimeWorkersPerNode(ModuleOp module) {
  if (!module || !getRuntimeStaticWorkers(module))
    return std::nullopt;

  auto totalWorkers = getRuntimeTotalWorkers(module);
  if (!totalWorkers || *totalWorkers <= 0)
    return std::nullopt;

  int64_t totalNodes = getRuntimeTotalNodes(module).value_or(1);
  totalNodes = std::max<int64_t>(1, totalNodes);
  return std::max<int64_t>(1, *totalWorkers / totalNodes);
}

static std::optional<int64_t> getStaticRuntimeDispatchWorkers(ModuleOp module,
                                                              EdtOp edt) {
  if (!module || !edt || !getRuntimeStaticWorkers(module))
    return std::nullopt;

  if (edt.getConcurrency() == EdtConcurrency::internode) {
    int64_t totalNodes = getRuntimeTotalNodes(module).value_or(1);
    return std::max<int64_t>(1, totalNodes);
  }

  return getStaticRuntimeWorkersPerNode(module);
}

static std::optional<int64_t> getStaticRuntimeTotalWorkers(ModuleOp module,
                                                           EdtOp edt) {
  if (!module || !edt || !getRuntimeStaticWorkers(module))
    return std::nullopt;

  auto workersPerNode = getStaticRuntimeWorkersPerNode(module);
  if (!workersPerNode)
    return std::nullopt;

  if (edt.getConcurrency() == EdtConcurrency::internode) {
    int64_t totalNodes = getRuntimeTotalNodes(module).value_or(1);
    totalNodes = std::max<int64_t>(1, totalNodes);
    return totalNodes * *workersPerNode;
  }

  return workersPerNode;
}

static bool isValidTiling2DColumnDivisor(int64_t totalWorkers,
                                         std::optional<int64_t> workersPerNode,
                                         int64_t candidate) {
  if (candidate <= 1 || totalWorkers % candidate != 0)
    return false;
  if (!workersPerNode || *workersPerNode <= 0)
    return true;
  return (*workersPerNode % candidate) == 0;
}

static int64_t
chooseColumnWorkersNearTarget(int64_t totalWorkers,
                              std::optional<int64_t> workersPerNode,
                              double target) {
  if (totalWorkers < 4)
    return 1;

  int64_t best = 1;
  double bestDist = std::numeric_limits<double>::infinity();
  int64_t sqrtW =
      static_cast<int64_t>(std::sqrt(static_cast<double>(totalWorkers)));
  for (int64_t d = 2; d <= sqrtW; ++d) {
    if (totalWorkers % d != 0)
      continue;
    for (int64_t candidate : {d, totalWorkers / d}) {
      if (!isValidTiling2DColumnDivisor(totalWorkers, workersPerNode,
                                        candidate))
        continue;
      double dist = std::fabs(static_cast<double>(candidate) - target);
      if (dist < bestDist || (dist == bestDist && candidate > best)) {
        best = candidate;
        bestDist = dist;
      }
    }
  }
  return best;
}

static int64_t
chooseTiling2DColumnWorkers(int64_t totalWorkers,
                            std::optional<int64_t> workersPerNode) {
  double target = std::sqrt(static_cast<double>(totalWorkers));
  return chooseColumnWorkersNearTarget(totalWorkers, workersPerNode, target);
}

static int64_t
chooseWavefront2DColumnWorkers(int64_t totalWorkers,
                               std::optional<int64_t> workersPerNode) {
  /// Weighted wavefront frontiers expose less row-wise concurrency than a
  /// near-square tiling. Bias toward wider column decomposition while keeping
  /// the same node-aligned divisor legality checks as generic tiling_2d.
  double baseTarget = std::sqrt(static_cast<double>(totalWorkers));
  double widenedTarget = std::min(static_cast<double>(totalWorkers),
                                  std::max(2.0, baseTarget * 2.0));
  return chooseColumnWorkersNearTarget(totalWorkers, workersPerNode,
                                       widenedTarget);
}

static int64_t
chooseColumnWorkersForPattern(int64_t totalWorkers,
                              std::optional<int64_t> workersPerNode,
                              const LoweringContractInfo &contract) {
  if (contract.prefersWideTiling2DColumns())
    return chooseWavefront2DColumnWorkers(totalWorkers, workersPerNode);
  return chooseTiling2DColumnWorkers(totalWorkers, workersPerNode);
}

/// Cast a Value to index type if needed. Returns null Value for null input.
static Value castToIndexType(OpBuilder &builder, Location loc, Value v) {
  if (!v)
    return Value();
  if (v.getType().isIndex())
    return v;
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(), v);
}

} // namespace

std::pair<Value, Value> WorkDistributionUtils::balancedDistribute(
    ArtsCodegen *AC, Location loc, Value totalChunks, Value numParticipants,
    Value participantId) {
  Value oneIndex = AC->createIndexConstant(1, loc);

  Value base = AC->create<arith::DivUIOp>(loc, totalChunks, numParticipants);
  Value rem = AC->create<arith::RemUIOp>(loc, totalChunks, numParticipants);

  /// When the chunk count divides evenly, every participant receives exactly
  /// `base` chunks and the clipped remainder term is provably zero. Emitting
  /// the generic `participantId < rem ? base + 1 : base` shape in that case
  /// leaves dead symbolic control in worker-slice bounds, which then obscures
  /// single-block DB acquires downstream.
  auto totalChunksConst = ValueAnalysis::tryFoldConstantIndex(totalChunks);
  auto participantsConst = ValueAnalysis::tryFoldConstantIndex(numParticipants);
  bool evenlyDivisible = totalChunksConst && participantsConst &&
                         *participantsConst > 0 &&
                         (*totalChunksConst % *participantsConst) == 0;
  if (evenlyDivisible || ValueAnalysis::isZeroConstant(rem)) {
    Value start = AC->create<arith::MulIOp>(loc, participantId, base);
    return {start, base};
  }

  Value getsExtra = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                              participantId, rem);
  Value count = AC->create<arith::SelectOp>(
      loc, getsExtra, AC->create<arith::AddIOp>(loc, base, oneIndex), base);

  Value clipped = AC->create<arith::MinUIOp>(loc, participantId, rem);
  Value start = AC->create<arith::AddIOp>(
      loc, AC->create<arith::MulIOp>(loc, participantId, base), clipped);

  return {start, count};
}

LoopChunkPlan
WorkDistributionUtils::planLoopChunking(ArtsCodegen *AC, ForOp forOp,
                                        const LoweringContractInfo &contract,
                                        Value runtimeBlockSizeHint) {
  LoopChunkPlan plan;
  if (!AC || !forOp)
    return plan;

  Location loc = forOp.getLoc();
  Value one = AC->createIndexConstant(1, loc);
  Value lowerBound = forOp.getLowerBound()[0];
  Value upperBound = forOp.getUpperBound()[0];
  Value loopStep = forOp.getStep()[0];

  int64_t advisoryBlockSize = getExplicitLoopBlockHint(forOp).value_or(1);

  plan.blockSize = AC->createIndexConstant(advisoryBlockSize, loc);
  if (contract.allowsDbAlignedChunking()) {
    if (runtimeBlockSizeHint) {
      Value requiredBlockSize = AC->castToIndex(runtimeBlockSizeHint, loc);
      requiredBlockSize =
          AC->create<arith::MaxUIOp>(loc, requiredBlockSize, one);

      auto requiredConst =
          ValueAnalysis::tryFoldConstantIndex(requiredBlockSize);
      bool hasNonTrivialAlignmentRequirement =
          !requiredConst || *requiredConst > 1;

      if (hasNonTrivialAlignmentRequirement) {
        /// DB-aligned owner slices are a lowering requirement, not just a
        /// coarsening preference. Once lowering derives a non-trivial owner
        /// block span, use that span as the chunking unit so worker slices
        /// line up with the DB layout chosen later by DbPartitioning.
        plan.blockSize = requiredBlockSize;
        if (requiredConst)
          plan.alignmentBlockSize = *requiredConst;
        else
          plan.useRuntimeBlockAlignment = true;
      }
    }

    if (!plan.alignmentBlockSize && !plan.useRuntimeBlockAlignment &&
        advisoryBlockSize > 1)
      plan.alignmentBlockSize = advisoryBlockSize;
  }

  plan.chunkLowerBound = lowerBound;
  if (plan.alignmentBlockSize) {
    if (auto lbConst = ValueAnalysis::tryFoldConstantIndex(lowerBound)) {
      int64_t lbVal = *lbConst;
      int64_t aligned = lbVal - (lbVal % *plan.alignmentBlockSize);
      if (aligned != lbVal) {
        plan.useAlignedLowerBound = true;
        plan.chunkLowerBound = AC->createIndexConstant(aligned, loc);
      }
    }
  }

  if (plan.useRuntimeBlockAlignment) {
    Value rem = AC->create<arith::RemUIOp>(loc, lowerBound, plan.blockSize);
    plan.chunkLowerBound = AC->create<arith::SubIOp>(loc, lowerBound, rem);
    plan.useAlignedLowerBound = true;
  }

  Value range =
      AC->create<arith::SubIOp>(loc, upperBound, plan.chunkLowerBound);
  Value adjustedRange = AC->create<arith::AddIOp>(
      loc, range, AC->create<arith::SubIOp>(loc, loopStep, one));
  plan.totalIterations =
      AC->create<arith::DivUIOp>(loc, adjustedRange, loopStep);

  Value adjustedIterations = AC->create<arith::AddIOp>(
      loc, plan.totalIterations,
      AC->create<arith::SubIOp>(loc, plan.blockSize, one));
  plan.totalChunks =
      AC->create<arith::DivUIOp>(loc, adjustedIterations, plan.blockSize);
  return plan;
}

Tiling2DWorkerGrid WorkDistributionUtils::getTiling2DWorkerGrid(
    ArtsCodegen *AC, Location loc, Value workerId, Value totalWorkers,
    const LoweringContractInfo &contract, Value workersPerNode) {
  Tiling2DWorkerGrid grid;
  Value one = AC->createIndexConstant(1, loc);
  Value zero = AC->createIndexConstant(0, loc);

  Value totalWorkersIndex = AC->castToIndex(totalWorkers, loc);
  Value workerIdIndex = AC->castToIndex(workerId, loc);

  grid.rowWorkers = totalWorkersIndex;
  grid.colWorkers = one;
  grid.rowWorkerId = workerIdIndex;
  grid.colWorkerId = zero;

  if (auto totalWorkersConst =
          ValueAnalysis::tryFoldConstantIndex(totalWorkersIndex)) {
    std::optional<int64_t> workersPerNodeConst = std::nullopt;
    if (workersPerNode) {
      workersPerNodeConst = ValueAnalysis::tryFoldConstantIndex(
          AC->castToIndex(workersPerNode, loc));
    }

    int64_t colWorkersConst = chooseColumnWorkersForPattern(
        *totalWorkersConst, workersPerNodeConst, contract);
    if (colWorkersConst > 1) {
      grid.colWorkers = AC->createIndexConstant(colWorkersConst, loc);
      grid.rowWorkers =
          AC->create<arith::DivUIOp>(loc, totalWorkersIndex, grid.colWorkers);
      grid.rowWorkerId =
          AC->create<arith::DivUIOp>(loc, workerIdIndex, grid.colWorkers);
      grid.colWorkerId =
          AC->create<arith::RemUIOp>(loc, workerIdIndex, grid.colWorkers);
    }
  }

  return grid;
}

DistributionBounds WorkDistributionUtils::computeBounds(
    ArtsCodegen *AC, Location loc, const DistributionStrategy &strategy,
    Value workerId, Value runtimeTotalWorkers, Value runtimeWorkersPerNode,
    Value totalIterations, Value totalChunks, Value blockSize,
    const LoweringContractInfo &contract) {
  DistributionBounds bounds;
  Value zeroIndex = AC->createIndexConstant(0, loc);

  if (strategy.kind == DistributionKind::BlockCyclic) {
    bounds.iterStart = zeroIndex;
    bounds.iterCount = zeroIndex;
    bounds.iterCountHint = zeroIndex;
    bounds.hasWork = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                               workerId, totalChunks);
    bounds.acquireStart = zeroIndex;
    bounds.acquireSize = totalIterations;
    bounds.acquireSizeHint = totalIterations;
    return bounds;
  }

  if (strategy.kind == DistributionKind::Flat ||
      strategy.kind == DistributionKind::Tiling2D ||
      strategy.kind == DistributionKind::Replicated) {
    Value distWorkerId = workerId;
    Value distWorkers = runtimeTotalWorkers;
    if (strategy.kind == DistributionKind::Tiling2D) {
      Tiling2DWorkerGrid grid =
          getTiling2DWorkerGrid(AC, loc, workerId, runtimeTotalWorkers,
                                contract, runtimeWorkersPerNode);
      distWorkerId = grid.rowWorkerId;
      distWorkers = grid.rowWorkers;
    }

    auto [firstChunkIdx, chunkCount] =
        balancedDistribute(AC, loc, totalChunks, distWorkers, distWorkerId);

    bounds.iterStart = AC->create<arith::MulIOp>(loc, firstChunkIdx, blockSize);
    bounds.iterCountHint =
        AC->create<arith::MulIOp>(loc, chunkCount, blockSize);

    Value needZeroIters = AC->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, bounds.iterStart, totalIterations);
    Value remainingIters =
        AC->create<arith::SubIOp>(loc, totalIterations, bounds.iterStart);
    Value remainingItersNonNeg = AC->create<arith::SelectOp>(
        loc, needZeroIters, zeroIndex, remainingIters);
    bounds.iterCount = AC->create<arith::MinUIOp>(loc, bounds.iterCountHint,
                                                  remainingItersNonNeg);

    bounds.hasWork = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                               bounds.iterCount, zeroIndex);
    bounds.acquireStart = bounds.iterStart;
    bounds.acquireSize = bounds.iterCount;
    bounds.acquireSizeHint = bounds.iterCountHint;
    return bounds;
  }

  Value numNodes = AC->create<arith::DivUIOp>(loc, runtimeTotalWorkers,
                                              runtimeWorkersPerNode);
  Value nodeId =
      AC->create<arith::DivUIOp>(loc, workerId, runtimeWorkersPerNode);
  Value localThreadId =
      AC->create<arith::RemUIOp>(loc, workerId, runtimeWorkersPerNode);

  auto [nodeChunkStart, nodeChunkCount] =
      balancedDistribute(AC, loc, totalChunks, numNodes, nodeId);

  Value nodeStart = AC->create<arith::MulIOp>(loc, nodeChunkStart, blockSize);
  Value nodeMaxRows = AC->create<arith::MulIOp>(loc, nodeChunkCount, blockSize);

  Value needZeroNode = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                                 nodeStart, totalIterations);
  Value nodeRemaining =
      AC->create<arith::SubIOp>(loc, totalIterations, nodeStart);
  Value nodeRemainingNonNeg =
      AC->create<arith::SelectOp>(loc, needZeroNode, zeroIndex, nodeRemaining);
  Value nodeRows =
      AC->create<arith::MinUIOp>(loc, nodeMaxRows, nodeRemainingNonNeg);

  bounds.acquireStart = nodeStart;
  bounds.acquireSize = nodeRows;
  bounds.acquireSizeHint = nodeMaxRows;

  Value oneIndex = AC->createIndexConstant(1, loc);
  Value adjusted = AC->create<arith::AddIOp>(
      loc, nodeRows,
      AC->create<arith::SubIOp>(loc, runtimeWorkersPerNode, oneIndex));
  Value subBlock =
      AC->create<arith::DivUIOp>(loc, adjusted, runtimeWorkersPerNode);

  Value localStart = AC->create<arith::MulIOp>(loc, localThreadId, subBlock);

  Value needZeroLocal = AC->create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::uge, localStart, nodeRows);
  Value localRemaining = AC->create<arith::SubIOp>(loc, nodeRows, localStart);
  Value localRemainingNonNeg = AC->create<arith::SelectOp>(
      loc, needZeroLocal, zeroIndex, localRemaining);
  Value localCount =
      AC->create<arith::MinUIOp>(loc, subBlock, localRemainingNonNeg);

  bounds.iterStart = AC->create<arith::AddIOp>(loc, nodeStart, localStart);
  bounds.iterCount = localCount;
  bounds.iterCountHint = subBlock;
  bounds.hasWork = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                             localCount, zeroIndex);
  return bounds;
}

DistributionBounds WorkDistributionUtils::recomputeBoundsInside(
    ArtsCodegen *AC, Location loc, const DistributionStrategy &strategy,
    Value workerId, Value insideTotalWorkers, Value workersPerNode,
    Value upperBound, Value lowerBound, Value loopStep, Value blockSize,
    std::optional<int64_t> alignmentBlockSize, bool useRuntimeBlockAlignment,
    const LoweringContractInfo &contract) {
  Value workerIdIndex = AC->castToIndex(workerId, loc);
  Value totalWorkersIndex = AC->castToIndex(insideTotalWorkers, loc);

  Value oneIndex = AC->createIndexConstant(1, loc);
  Value zeroIndex = AC->createIndexConstant(0, loc);

  SetVector<Value> boundsToClone;
  Region *currentRegion = AC->getBuilder().getInsertionBlock()->getParent();

  std::function<void(Value)> collectWithDeps = [&](Value val) {
    if (boundsToClone.contains(val))
      return;
    if (Operation *defOp = val.getDefiningOp()) {
      if (!currentRegion->isAncestor(defOp->getParentRegion())) {
        for (Value operand : defOp->getOperands())
          collectWithDeps(operand);
        boundsToClone.insert(val);
      }
    }
  };

  collectWithDeps(upperBound);
  collectWithDeps(lowerBound);
  collectWithDeps(loopStep);
  collectWithDeps(blockSize);

  IRMapping boundsMapper;
  ValueAnalysis::cloneValuesIntoRegion(
      boundsToClone, currentRegion, boundsMapper, AC->getBuilder(),
      /*allowMemoryEffectFree=*/true, isSafeRuntimeQuery);

  Value localUpperBound = boundsMapper.lookupOrDefault(upperBound);
  Value localLowerBound = boundsMapper.lookupOrDefault(lowerBound);
  Value localLoopStep = boundsMapper.lookupOrDefault(loopStep);
  Value localBlockSize = boundsMapper.lookupOrDefault(blockSize);

  Value chunkLowerBound = localLowerBound;
  if (alignmentBlockSize) {
    if (auto lbConst = ValueAnalysis::tryFoldConstantIndex(localLowerBound)) {
      int64_t aligned = *lbConst - (*lbConst % *alignmentBlockSize);
      if (aligned != *lbConst)
        chunkLowerBound = AC->createIndexConstant(aligned, loc);
    }
  } else if (useRuntimeBlockAlignment) {
    Value rem =
        AC->create<arith::RemUIOp>(loc, localLowerBound, localBlockSize);
    chunkLowerBound = AC->create<arith::SubIOp>(loc, localLowerBound, rem);
  }

  Value range =
      AC->create<arith::SubIOp>(loc, localUpperBound, chunkLowerBound);
  Value adjustedRange = AC->create<arith::AddIOp>(
      loc, range, AC->create<arith::SubIOp>(loc, localLoopStep, oneIndex));
  Value localTotalIterations =
      AC->create<arith::DivUIOp>(loc, adjustedRange, localLoopStep);

  Value adjustedIterations = AC->create<arith::AddIOp>(
      loc, localTotalIterations,
      AC->create<arith::SubIOp>(loc, localBlockSize, oneIndex));
  Value localTotalChunks =
      AC->create<arith::DivUIOp>(loc, adjustedIterations, localBlockSize);

  if (strategy.kind == DistributionKind::BlockCyclic) {
    DistributionBounds bounds;
    bounds.iterStart = zeroIndex;
    bounds.iterCount = zeroIndex;
    bounds.iterCountHint = zeroIndex;
    bounds.hasWork = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                               workerIdIndex, localTotalChunks);
    bounds.acquireStart = zeroIndex;
    bounds.acquireSize = localTotalIterations;
    bounds.acquireSizeHint = localTotalIterations;
    return bounds;
  }

  if (strategy.kind == DistributionKind::TwoLevel) {
    Value wpnIndex = workersPerNode;
    if (wpnIndex) {
      if (Operation *defOp = wpnIndex.getDefiningOp()) {
        if (!currentRegion->isAncestor(defOp->getParentRegion())) {
          collectWithDeps(wpnIndex);
          ValueAnalysis::cloneValuesIntoRegion(
              boundsToClone, currentRegion, boundsMapper, AC->getBuilder(),
              /*allowMemoryEffectFree=*/true, isSafeRuntimeQuery);
          wpnIndex = boundsMapper.lookupOrDefault(wpnIndex);
        }
      }
      wpnIndex = AC->castToIndex(wpnIndex, loc);
    }

    Value numNodes =
        AC->create<arith::DivUIOp>(loc, totalWorkersIndex, wpnIndex);
    Value nodeId = AC->create<arith::DivUIOp>(loc, workerIdIndex, wpnIndex);
    Value localThreadId =
        AC->create<arith::RemUIOp>(loc, workerIdIndex, wpnIndex);

    auto [nodeChunkStart, nodeChunkCount] =
        balancedDistribute(AC, loc, localTotalChunks, numNodes, nodeId);

    Value nodeStart =
        AC->create<arith::MulIOp>(loc, nodeChunkStart, localBlockSize);
    Value nodeMaxRows =
        AC->create<arith::MulIOp>(loc, nodeChunkCount, localBlockSize);

    Value needZeroNode = AC->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, nodeStart, localTotalIterations);
    Value nodeRemaining =
        AC->create<arith::SubIOp>(loc, localTotalIterations, nodeStart);
    Value nodeRemainingNonNeg = AC->create<arith::SelectOp>(
        loc, needZeroNode, zeroIndex, nodeRemaining);
    Value nodeRows =
        AC->create<arith::MinUIOp>(loc, nodeMaxRows, nodeRemainingNonNeg);

    Value subAdjusted = AC->create<arith::AddIOp>(
        loc, nodeRows, AC->create<arith::SubIOp>(loc, wpnIndex, oneIndex));
    Value subBlock = AC->create<arith::DivUIOp>(loc, subAdjusted, wpnIndex);

    Value localStart = AC->create<arith::MulIOp>(loc, localThreadId, subBlock);

    Value needZeroLocal = AC->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, localStart, nodeRows);
    Value localRemaining = AC->create<arith::SubIOp>(loc, nodeRows, localStart);
    Value localRemainingNonNeg = AC->create<arith::SelectOp>(
        loc, needZeroLocal, zeroIndex, localRemaining);
    Value localCount =
        AC->create<arith::MinUIOp>(loc, subBlock, localRemainingNonNeg);

    DistributionBounds bounds;
    bounds.iterStart = AC->create<arith::AddIOp>(loc, nodeStart, localStart);
    bounds.iterCount = localCount;
    bounds.iterCountHint = subBlock;
    bounds.hasWork = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                               localCount, zeroIndex);
    bounds.acquireStart = nodeStart;
    bounds.acquireSize = nodeRows;
    bounds.acquireSizeHint = nodeMaxRows;
    return bounds;
  }

  Value distWorkerId = workerIdIndex;
  Value distWorkers = totalWorkersIndex;
  if (strategy.kind == DistributionKind::Tiling2D) {
    Tiling2DWorkerGrid grid = getTiling2DWorkerGrid(
        AC, loc, workerIdIndex, totalWorkersIndex, contract, workersPerNode);
    distWorkerId = grid.rowWorkerId;
    distWorkers = grid.rowWorkers;
  }

  auto [firstChunkIdx, chunkCount] =
      balancedDistribute(AC, loc, localTotalChunks, distWorkers, distWorkerId);

  DistributionBounds bounds;
  bounds.iterStart =
      AC->create<arith::MulIOp>(loc, firstChunkIdx, localBlockSize);
  bounds.iterCountHint =
      AC->create<arith::MulIOp>(loc, chunkCount, localBlockSize);

  Value needZeroIters = AC->create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::uge, bounds.iterStart, localTotalIterations);
  Value remainingIters =
      AC->create<arith::SubIOp>(loc, localTotalIterations, bounds.iterStart);
  Value remainingItersNonNeg = AC->create<arith::SelectOp>(
      loc, needZeroIters, zeroIndex, remainingIters);
  bounds.iterCount = AC->create<arith::MinUIOp>(loc, bounds.iterCountHint,
                                                remainingItersNonNeg);
  bounds.hasWork = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                             bounds.iterCount, zeroIndex);

  bounds.acquireStart = bounds.iterStart;
  bounds.acquireSize = bounds.iterCount;
  bounds.acquireSizeHint = bounds.iterCountHint;
  return bounds;
}

Value WorkDistributionUtils::computeDbAlignmentBlockSize(
    ForOp forOp, EdtOp parallelEdt, Value numPartitions, ArtsCodegen *AC,
    Location loc, DbAnalysis &dbAnalysis) {
  if (!forOp || !parallelEdt || !numPartitions)
    return Value();

  Value one = AC->createIndexConstant(1, loc);
  Value hintBlockSize;
  LoweringContractInfo loopContract =
      resolveLoopDistributionContract(forOp.getOperation());

  bool isTwoLevel = (parallelEdt.getConcurrency() == EdtConcurrency::internode);
  for (Value dep : parallelEdt.getDependencies()) {
    auto allocInfo = DbUtils::traceToDbAlloc(dep);
    if (!allocInfo)
      continue;
    auto [rootGuid, rootPtr] = *allocInfo;
    (void)rootPtr;
    auto allocOp = rootGuid.getDefiningOp<DbAllocOp>();
    if (!allocOp || allocOp.getElementSizes().empty())
      continue;

    std::optional<unsigned> mappedDim;
    if (auto acquire =
            dyn_cast_or_null<DbAcquireOp>(DbUtils::getUnderlyingDb(dep)))
      mappedDim = dbAnalysis.inferLoopMappedDim(acquire, forOp);
    if (!mappedDim)
      mappedDim = dbAnalysis.inferLoopMappedDim(dep, forOp);

    if (!mappedDim) {
      if (allocOp.getElementSizes().size() == 1)
        mappedDim = 0u;
      else
        continue;
    }
    if (*mappedDim >= allocOp.getElementSizes().size())
      continue;

    Value ownerExtent = allocOp.getElementSizes()[*mappedDim];
    if (auto constElem = ValueAnalysis::tryFoldConstantIndex(ownerExtent))
      if (*constElem <= 1)
        continue;

    if (!isTwoLevel) {
      auto pattern = getDbAccessPattern(allocOp.getOperation());
      bool isStencil = pattern && *pattern == DbAccessPattern::stencil;
      bool requiresWriteBlockOwnership = allocOp.getMode() != ArtsMode::in;
      if (!isStencil && !requiresWriteBlockOwnership)
        continue;
    }

    ownerExtent = AC->castToIndex(ownerExtent, loc);
    Value adjusted = AC->create<arith::AddIOp>(
        loc, ownerExtent, AC->create<arith::SubIOp>(loc, numPartitions, one));
    Value candidate = AC->create<arith::DivUIOp>(loc, adjusted, numPartitions);
    candidate = AC->create<arith::MaxUIOp>(loc, candidate, one);

    if (!hintBlockSize)
      hintBlockSize = candidate;
    else
      hintBlockSize = AC->create<arith::MinUIOp>(loc, hintBlockSize, candidate);
  }

  if (auto loopBlockHint = getExplicitLoopBlockHint(forOp);
      loopBlockHint &&
      shouldHonorLoopBlockHintForDbAlignment(loopContract, parallelEdt)) {
    /// This helper computes a fallback owner-alignment size from the current
    /// dependency layout. When ForOpt already chose a stronger stencil owner
    /// span, keep that explicit hint instead of shrinking the loop back to a
    /// one-block-per-worker fallback.
    Value advisory = AC->createIndexConstant(*loopBlockHint, loc);
    hintBlockSize =
        hintBlockSize ? AC->create<arith::MaxUIOp>(loc, hintBlockSize, advisory)
                      : advisory;
  }

  return hintBlockSize;
}

Value WorkDistributionUtils::getWorkersPerNode(OpBuilder &builder, Location loc,
                                               EdtOp parallelEdt) {
  Value one = arts::createOneIndex(builder, loc);
  Value workersPerNode;

  if (auto wpnConst = getExplicitWorkersPerNodeCount(parallelEdt)) {
    int64_t clamped = std::max<int64_t>(1, *wpnConst);
    return arts::createConstantIndex(builder, loc, clamped);
  }

  if (auto module =
          parallelEdt ? parallelEdt->getParentOfType<ModuleOp>() : ModuleOp();
      auto staticWorkersPerNode = getStaticRuntimeWorkersPerNode(module)) {
    return arts::createConstantIndex(builder, loc, *staticWorkersPerNode);
  }

  if (auto workers = getExplicitWorkerCount(parallelEdt)) {
    Value totalWorkers = arts::createConstantIndex(builder, loc, *workers);
    Value totalNodes = castToIndexType(
        builder, loc,
        RuntimeQueryOp::create(builder, loc, RuntimeQueryKind::totalNodes)
            .getResult());
    workersPerNode =
        arith::DivUIOp::create(builder, loc, totalWorkers, totalNodes);
  } else {
    workersPerNode = castToIndexType(
        builder, loc,
        RuntimeQueryOp::create(builder, loc, RuntimeQueryKind::totalWorkers)
            .getResult());
  }

  Value zero = arts::createZeroIndex(builder, loc);
  Value isZero = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                       workersPerNode, zero);
  return arith::SelectOp::create(builder, loc, isZero, one, workersPerNode);
}

Value WorkDistributionUtils::getWorkersPerNode(ArtsCodegen *AC, Location loc,
                                               EdtOp parallelEdt) {
  return getWorkersPerNode(AC->getBuilder(), loc, parallelEdt);
}

Value WorkDistributionUtils::getTotalWorkers(OpBuilder &builder, Location loc,
                                             EdtOp parallelEdt) {
  if (auto workers = getExplicitWorkerCount(parallelEdt))
    return arts::createConstantIndex(builder, loc, *workers);

  if (auto module =
          parallelEdt ? parallelEdt->getParentOfType<ModuleOp>() : ModuleOp();
      auto staticTotalWorkers =
          getStaticRuntimeTotalWorkers(module, parallelEdt)) {
    return arts::createConstantIndex(builder, loc, *staticTotalWorkers);
  }

  if (parallelEdt.getConcurrency() == EdtConcurrency::internode) {
    Value nodes = castToIndexType(
        builder, loc,
        RuntimeQueryOp::create(builder, loc, RuntimeQueryKind::totalNodes)
            .getResult());
    Value threads = castToIndexType(
        builder, loc,
        RuntimeQueryOp::create(builder, loc, RuntimeQueryKind::totalWorkers)
            .getResult());
    return arith::MulIOp::create(builder, loc, nodes, threads);
  }

  return castToIndexType(
      builder, loc,
      RuntimeQueryOp::create(builder, loc, RuntimeQueryKind::totalWorkers)
          .getResult());
}

Value WorkDistributionUtils::getTotalWorkers(ArtsCodegen *AC, Location loc,
                                             EdtOp parallelEdt) {
  return getTotalWorkers(AC->getBuilder(), loc, parallelEdt);
}

Value WorkDistributionUtils::getDispatchWorkerCount(OpBuilder &builder,
                                                    Location loc,
                                                    EdtOp parallelEdt) {
  if (auto workers = getExplicitWorkerCount(parallelEdt))
    return arts::createConstantIndex(builder, loc, *workers);

  if (auto module =
          parallelEdt ? parallelEdt->getParentOfType<ModuleOp>() : ModuleOp();
      auto staticDispatchWorkers =
          getStaticRuntimeDispatchWorkers(module, parallelEdt)) {
    return arts::createConstantIndex(builder, loc, *staticDispatchWorkers);
  }

  RuntimeQueryKind queryKind =
      parallelEdt.getConcurrency() == EdtConcurrency::internode
          ? RuntimeQueryKind::totalNodes
          : RuntimeQueryKind::totalWorkers;
  Value workerCount =
      RuntimeQueryOp::create(builder, loc, queryKind).getResult();
  return castToIndexType(builder, loc, workerCount);
}

Value WorkDistributionUtils::getForDispatchWorkerCount(
    ArtsCodegen *AC, Location loc, EdtOp parallelEdt,
    const DistributionStrategy &strategy, Value totalChunks,
    const LoweringContractInfo &contract) {
  Value one = AC->createIndexConstant(1, loc);
  Value totalWorkers = getTotalWorkers(AC, loc, parallelEdt);
  Value clampedDispatch = totalWorkers;

  switch (strategy.kind) {
  case DistributionKind::Flat:
  case DistributionKind::BlockCyclic:
  case DistributionKind::Replicated:
    clampedDispatch =
        AC->create<arith::MinUIOp>(loc, totalWorkers, totalChunks);
    break;
  case DistributionKind::TwoLevel: {
    Value workersPerNode = getWorkersPerNode(AC, loc, parallelEdt);
    Value numNodes =
        AC->create<arith::DivUIOp>(loc, totalWorkers, workersPerNode);
    Value activeNodes = AC->create<arith::MinUIOp>(loc, numNodes, totalChunks);
    clampedDispatch =
        AC->create<arith::MulIOp>(loc, activeNodes, workersPerNode);
    break;
  }
  case DistributionKind::Tiling2D: {
    Value totalWorkersIndex = AC->castToIndex(totalWorkers, loc);
    auto totalWorkersConst =
        ValueAnalysis::tryFoldConstantIndex(totalWorkersIndex);
    if (!totalWorkersConst)
      break;

    std::optional<int64_t> workersPerNodeConst = std::nullopt;
    if (parallelEdt.getConcurrency() == EdtConcurrency::internode) {
      Value workersPerNode = getWorkersPerNode(AC, loc, parallelEdt);
      workersPerNodeConst = ValueAnalysis::tryFoldConstantIndex(
          AC->castToIndex(workersPerNode, loc));
    }

    int64_t colWorkersConst = chooseColumnWorkersForPattern(
        *totalWorkersConst, workersPerNodeConst, contract);
    int64_t rowWorkersConst =
        std::max<int64_t>(1, *totalWorkersConst / colWorkersConst);

    Value rowWorkers = AC->createIndexConstant(rowWorkersConst, loc);
    Value activeRows = AC->create<arith::MinUIOp>(loc, rowWorkers, totalChunks);
    Value colWorkers = AC->createIndexConstant(colWorkersConst, loc);
    clampedDispatch = AC->create<arith::MulIOp>(loc, activeRows, colWorkers);
    break;
  }
  }

  return AC->create<arith::MaxUIOp>(loc, clampedDispatch, one);
}
