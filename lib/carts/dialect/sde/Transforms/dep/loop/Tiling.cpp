///==========================================================================///
/// File: Tiling.cpp
///
/// Pattern-driven memref tiling in SDE. The pass rewrites SDE scheduling units
/// and their executable memref loop bodies so CU/SU/MU tiling intent is
/// explicit before boundary materialization.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::carts::sde {
#define GEN_PASS_DEF_TILING
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::sde

#include "carts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "carts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "carts/dialect/sde/Utils/IterationSizingUtils.h"
#include "carts/dialect/sde/Utils/SDECostModel.h"
#include "carts/utils/ArrayAttrUtils.h"
#include "carts/utils/LoopUtils.h"
#include "carts/utils/ValueAnalysis.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::carts;

namespace {

static Value getConstantIndex(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

static bool usesOwnerLocalPipelineGrain(sde::SdeSuIterateOp op) {
  auto classification = op.getStructuredClassification();
  return classification &&
         *classification ==
             sde::SdeStructuredClassification::elementwise_pipeline &&
         sde::isOwnerLocalPipelineReduction(op);
}

static Value buildRelaxedOwnerLocalPipelineMinIterations(
    OpBuilder &builder, Location loc, sde::SdeSuIterateOp op,
    Value clampedTripCount, Value targetTasks, int64_t minIterations) {
  Value minValue =
      getConstantIndex(builder, loc, std::max<int64_t>(1, minIterations));
  if (!op || !clampedTripCount || !targetTasks ||
      !usesOwnerLocalPipelineGrain(op) || minIterations <= 1)
    return minValue;

  Value one = getConstantIndex(builder, loc, 1);
  Value relaxationThreshold =
      arith::MulIOp::create(builder, loc, targetTasks, minValue);
  Value underfilledOwnerDomain = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, clampedTripCount,
      relaxationThreshold);
  return arith::SelectOp::create(builder, loc, underfilledOwnerDomain, one,
                                 minValue);
}

static int64_t getMinTileIterations(sde::SdeSuIterateOp op,
                                    sde::SDECostModel &costModel) {
  if (usesOwnerLocalPipelineGrain(op))
    return std::max<int64_t>(1,
                             costModel.getMinPipelineOwnerIterationsPerTask());
  return std::max<int64_t>(1, costModel.getMinIterationsPerWorker());
}

static int64_t getTargetTaskWaves(sde::SdeSuIterateOp op,
                                  sde::SDECostModel &costModel) {
  if (usesOwnerLocalPipelineGrain(op))
    return std::max<int64_t>(1,
                             costModel.getOwnerLocalPipelineTargetTaskWaves());
  return std::max<int64_t>(1, costModel.getInterLocalityTaskWaves());
}

static int64_t getTargetTileTasks(sde::SdeSuIterateOp op,
                                  sde::SDECostModel &costModel) {
  int64_t workers = std::max<int64_t>(1, costModel.getLogicalWorkerCapacity());
  int64_t waves = getTargetTaskWaves(op, costModel);
  if (workers > std::numeric_limits<int64_t>::max() / waves)
    return std::numeric_limits<int64_t>::max();
  return workers * waves;
}

static Value buildTileIterationValue(OpBuilder &builder, Location loc,
                                     sde::SdeSuIterateOp op,
                                     sde::SDECostModel &costModel) {
  Value tripCount = sde::buildTripCountValue(builder, loc, op);
  if (!tripCount)
    return Value();

  Value one = getConstantIndex(builder, loc, 1);
  Value workerCountValue = sde::buildLogicalWorkerCapacityValue(builder, loc);
  int64_t targetWaves = getTargetTaskWaves(op, costModel);
  if (targetWaves > 1) {
    Value waveCountValue = getConstantIndex(builder, loc, targetWaves);
    workerCountValue =
        arith::MulIOp::create(builder, loc, workerCountValue, waveCountValue);
  }
  Value clampedTripCount = arith::MaxUIOp::create(builder, loc, tripCount, one);
  Value minIterationsValue = buildRelaxedOwnerLocalPipelineMinIterations(
      builder, loc, op, clampedTripCount, workerCountValue,
      getMinTileIterations(op, costModel));
  Value balancedTile = arith::CeilDivUIOp::create(
      builder, loc, clampedTripCount, workerCountValue);
  Value preferredTile =
      arith::MaxUIOp::create(builder, loc, balancedTile, minIterationsValue);
  return arith::MinUIOp::create(builder, loc, preferredTile, clampedTripCount);
}

static SmallVector<Value> buildPerDimTripCounts(OpBuilder &builder,
                                                Location loc,
                                                sde::SdeSuIterateOp op) {
  SmallVector<Value> tripCounts;
  unsigned numDims = op.getLowerBounds().size();

  for (unsigned d = 0; d < numDims; ++d) {
    Value lb = op.getLowerBounds()[d];
    Value ub = op.getUpperBounds()[d];
    Value step = op.getSteps()[d];
    Value zero = getConstantIndex(builder, loc, 0);
    Value one = getConstantIndex(builder, loc, 1);

    Value span = arith::SubIOp::create(builder, loc, ub, lb);

    int64_t constantStep = 0;
    Value safeStep = step;
    if (::mlir::carts::ValueAnalysis::getConstantIndex(step, constantStep)) {
      if (constantStep <= 0)
        return {};
    } else {
      Value stepIsTooSmall = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sle, step, zero);
      safeStep =
          arith::SelectOp::create(builder, loc, stepIsTooSmall, one, step);
    }

    Value spanIsNegative = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::slt, span, zero);
    Value nonNegativeSpan =
        arith::SelectOp::create(builder, loc, spanIsNegative, zero, span);
    tripCounts.push_back(
        arith::CeilDivSIOp::create(builder, loc, nonNegativeSpan, safeStep));
  }
  return tripCounts;
}

static SmallVector<Value>
buildPerDimTileIterations(OpBuilder &builder, Location loc,
                          sde::SdeSuIterateOp op,
                          sde::SDECostModel &costModel) {
  SmallVector<Value> tripCounts = buildPerDimTripCounts(builder, loc, op);
  if (tripCounts.empty())
    return {};

  unsigned numDims = tripCounts.size();
  int workersPerDim =
      std::max(1, static_cast<int>(std::ceil(std::pow(
                      getTargetTileTasks(op, costModel), 1.0 / numDims))));
  int64_t minIter = getMinTileIterations(op, costModel);

  SmallVector<Value> tileIterations;
  for (unsigned d = 0; d < numDims; ++d) {
    Value one = getConstantIndex(builder, loc, 1);
    Value workersVal = getConstantIndex(builder, loc, workersPerDim);
    Value clampedTrip =
        arith::MaxUIOp::create(builder, loc, tripCounts[d], one);
    Value minIterVal = buildRelaxedOwnerLocalPipelineMinIterations(
        builder, loc, op, clampedTrip, workersVal, minIter);
    Value balanced =
        arith::CeilDivUIOp::create(builder, loc, clampedTrip, workersVal);
    Value preferred =
        arith::MaxUIOp::create(builder, loc, balanced, minIterVal);
    Value tile = arith::MinUIOp::create(builder, loc, preferred, clampedTrip);
    tileIterations.push_back(tile);
  }
  return tileIterations;
}

static int64_t chooseMatmulColumnWorkers(int64_t workers) {
  workers = std::max<int64_t>(1, workers);
  return std::max<int64_t>(1, static_cast<int64_t>(std::ceil(
                                  std::sqrt(static_cast<double>(workers)))));
}

static int64_t chooseStaticMatmulTile(int64_t extent, int64_t participants,
                                      int64_t minIterations) {
  if (extent <= 1)
    return std::max<int64_t>(1, extent);
  int64_t balanced = sde::ceilDivPositive(extent, participants);
  int64_t preferred =
      std::max<int64_t>(balanced, std::max<int64_t>(1, minIterations));
  return std::clamp<int64_t>(preferred, 1, extent);
}

struct DirectMatmulTilePlan {
  sde::LoopIndexedOutputPlan output;
  int64_t rowTile = 1;
  int64_t columnTile = 1;
  Value rowTileValue;
  Value columnTileValue;
};

struct PhysicalTilePlan {
  SmallVector<int64_t, 4> ownerPhysicalDims;
  SmallVector<int64_t, 4> blockShape;
  SmallVector<int64_t, 4> haloShape;
  SmallVector<int64_t, 4> tileIterations;
  sde::SdeIterationTopology topology = sde::SdeIterationTopology::owner_strip;
};

static std::optional<DirectMatmulTilePlan>
buildDirectMatmulTilePlan(OpBuilder &builder, Location loc,
                          sde::SdeSuIterateOp op,
                          sde::SDECostModel &costModel) {
  if (!sde::hasDistinctExternalMatmulInputRoots(op))
    return std::nullopt;

  std::optional<sde::LoopIndexedOutputPlan> outputPlan =
      sde::findLoopIndexedOutputPlan(op);
  if (!outputPlan || outputPlan->shape.size() < 2)
    return std::nullopt;

  int64_t rows = outputPlan->shape[0];
  int64_t columns = outputPlan->shape[1];
  if (rows <= 1 || columns <= 1)
    return std::nullopt;

  int64_t workers = std::max<int64_t>(1, costModel.getLogicalWorkerCapacity());
  int64_t rowWorkers = std::max<int64_t>(1, getTargetTileTasks(op, costModel));
  int64_t columnWorkers = chooseMatmulColumnWorkers(workers);
  int64_t minIterations =
      std::max<int64_t>(1, costModel.getMinIterationsPerWorker());

  DirectMatmulTilePlan plan;
  plan.output = std::move(*outputPlan);
  /// Keep the SDE-owned row dimension exposed to the worker distributor.
  /// The column tile is an inner locality tile; row tiling must not collapse
  /// the distributed task count down to the square-root worker grid.
  plan.rowTile = chooseStaticMatmulTile(rows, rowWorkers, /*minIterations=*/1);
  plan.columnTile =
      chooseStaticMatmulTile(columns, columnWorkers, minIterations);
  if (plan.rowTile <= 1 && plan.columnTile <= 1)
    return std::nullopt;

  plan.rowTileValue = getConstantIndex(builder, loc, plan.rowTile);
  plan.columnTileValue = getConstantIndex(builder, loc, plan.columnTile);
  return plan;
}

static bool isScalarExecutableOp(Operation &op) {
  if (isa<memref::LoadOp, memref::StoreOp>(op))
    return true;
  return op.getNumRegions() == 0 && isMemoryEffectFree(&op);
}

static bool isExecutableInnermostBody(Block &body) {
  for (Operation &op : body.without_terminator()) {
    if (!isScalarExecutableOp(op))
      return false;
  }
  return true;
}

static bool hasPerfectNestedScalarLoopNest(Block &body, unsigned numLoops) {
  if (numLoops == 0)
    return false;
  if (numLoops == 1)
    return isExecutableInnermostBody(body);

  Block *current = &body;
  for (unsigned depth = 1; depth < numLoops; ++depth) {
    scf::ForOp nestedLoop;
    for (Operation &op : current->without_terminator()) {
      if (!isa<scf::ForOp>(op) || nestedLoop)
        return false;
      nestedLoop = cast<scf::ForOp>(op);
    }
    if (!nestedLoop || !nestedLoop.getInitArgs().empty())
      return false;
    current = nestedLoop.getBody();
  }

  return isExecutableInnermostBody(*current);
}

static bool loopWritesOwnerColumn(scf::ForOp loop, Value outputRoot,
                                  Value ownerIv) {
  if (!loop || !outputRoot || !ownerIv || loop.getNumResults() != 0 ||
      !loop.getInitArgs().empty())
    return false;

  Value loopIv = loop.getInductionVar();
  bool matched = false;
  loop.walk([&](memref::StoreOp store) {
    Value root =
        ::mlir::carts::ValueAnalysis::stripMemrefViewOps(store.getMemref());
    if (root != outputRoot)
      return WalkResult::advance();

    ValueRange indices = store.getIndices();
    if (indices.size() < 2)
      return WalkResult::advance();
    if (!::mlir::carts::ValueAnalysis::dependsOn(indices.front(), ownerIv))
      return WalkResult::advance();

    for (Value index : indices.drop_front()) {
      if (::mlir::carts::ValueAnalysis::dependsOn(index, loopIv)) {
        matched = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return matched;
}

static void
collectDirectMatmulColumnLoops(Block &body, Value outputRoot, Value ownerIv,
                               SmallVectorImpl<scf::ForOp> &columnLoops) {
  body.walk([&](scf::ForOp loop) {
    if (loopWritesOwnerColumn(loop, outputRoot, ownerIv))
      columnLoops.push_back(loop);
  });
}

static bool isDirectMemoryMatmulCandidate(sde::SdeSuIterateOp op, Block &body) {
  if (op.getLowerBounds().size() != 1 ||
      op.getReductionAccumulators().size() != 0)
    return false;

  std::optional<sde::LoopIndexedOutputPlan> outputPlan =
      sde::findLoopIndexedOutputPlan(op);
  if (!outputPlan || outputPlan->shape.size() < 2)
    return false;

  auto effects = sde::collectStructuredMemoryEffects(op.getBody());
  if (effects.hasUnknownEffects || !effects.writes.contains(outputPlan->root))
    return false;

  Block &suBody = op.getBody().front();
  Value ownerIv = suBody.getArgument(0);
  SmallVector<scf::ForOp, 4> columnLoops;
  collectDirectMatmulColumnLoops(body, outputPlan->root, ownerIv, columnLoops);
  return !columnLoops.empty();
}

/// Return a per-SDE-dim mask: true = parallel (should tile), false = reduction
/// (keep original step). Memref SDE tiling only tiles scheduling-unit owner
/// dimensions here; reduction-specific tiling is handled by dedicated SDE
/// planning passes.
static SmallVector<bool> getParallelDimMask(sde::SdeSuIterateOp op) {
  unsigned numDims = op.getLowerBounds().size();
  return SmallVector<bool>(numDims, true);
}

static bool isStencilCandidate(sde::SdeSuIterateOp op, Block &body) {
  if (!op.getAccessMinOffsetsAttr() || !op.getAccessMaxOffsetsAttr())
    return false;
  // Stencils always have at least one nested scf.for for the inner dimension.
  // Count the total loop depth: 1 SDE dim + inner scf.for loops.
  unsigned numSuDims = op.getLowerBounds().size();
  unsigned innerLoops = 0;
  Block *current = &body;
  while (true) {
    scf::ForOp nestedLoop;
    for (Operation &nested : current->without_terminator()) {
      if (auto forOp = dyn_cast<scf::ForOp>(nested)) {
        if (nestedLoop)
          return false; // imperfect nest
        nestedLoop = forOp;
      } else if (!isScalarExecutableOp(nested)) {
        return false;
      }
    }
    if (!nestedLoop)
      break;
    ++innerLoops;
    current = nestedLoop.getBody();
  }
  return isExecutableInnermostBody(*current) && (numSuDims + innerLoops) >= 1;
}

static SmallVector<int64_t> getStencilHaloWidths(sde::SdeSuIterateOp op) {
  SmallVector<int64_t> halos;
  ArrayAttr minArr = op.getAccessMinOffsetsAttr();
  ArrayAttr maxArr = op.getAccessMaxOffsetsAttr();
  if (!minArr || !maxArr || minArr.size() != maxArr.size())
    return {};
  for (unsigned d = 0; d < minArr.size(); ++d) {
    int64_t lo = cast<IntegerAttr>(minArr[d]).getInt();
    int64_t hi = cast<IntegerAttr>(maxArr[d]).getInt();
    halos.push_back(std::max<int64_t>(1, hi - lo + 1));
  }
  return halos;
}

static SmallVector<int64_t>
getStencilHaloRadiiForOwnerDims(sde::SdeSuIterateOp op,
                                unsigned ownerDimCount) {
  SmallVector<int64_t> halos;
  ArrayAttr minArr = op.getAccessMinOffsetsAttr();
  ArrayAttr maxArr = op.getAccessMaxOffsetsAttr();
  if (!minArr || !maxArr || minArr.size() != maxArr.size())
    return {};
  for (unsigned d = 0; d < ownerDimCount && d < minArr.size(); ++d) {
    int64_t lo = cast<IntegerAttr>(minArr[d]).getInt();
    int64_t hi = cast<IntegerAttr>(maxArr[d]).getInt();
    halos.push_back(std::max<int64_t>(0, std::max(-lo, hi)));
  }
  return halos;
}

static int64_t relaxOwnerLocalPipelineMinIterations(
    sde::SdeSuIterateOp op, int64_t tripCount, int64_t targetTasks,
    int64_t minIterations) {
  if (!usesOwnerLocalPipelineGrain(op) || targetTasks <= 1 ||
      minIterations <= 1)
    return minIterations;

  bool productOverflows =
      targetTasks > std::numeric_limits<int64_t>::max() / minIterations;
  if (productOverflows || tripCount < targetTasks * minIterations)
    return 1;
  return minIterations;
}

static std::optional<SmallVector<int64_t, 4>>
computeStaticTileIterations(sde::SdeSuIterateOp op,
                            sde::SDECostModel &costModel) {
  unsigned numDims = op.getLowerBounds().size();
  if (numDims == 0)
    return std::nullopt;

  SmallVector<int64_t, 4> tripCounts;
  tripCounts.reserve(numDims);
  for (unsigned d = 0; d < numDims; ++d) {
    int64_t lb = 0;
    int64_t ub = 0;
    int64_t step = 0;
    if (!::mlir::carts::ValueAnalysis::getConstantIndex(op.getLowerBounds()[d],
                                                        lb) ||
        !::mlir::carts::ValueAnalysis::getConstantIndex(op.getUpperBounds()[d],
                                                        ub) ||
        !::mlir::carts::ValueAnalysis::getConstantIndex(op.getSteps()[d],
                                                        step) ||
        step <= 0)
      return std::nullopt;
    tripCounts.push_back(
        sde::ceilDivPositive(std::max<int64_t>(0, ub - lb), step));
  }

  int64_t minIter = getMinTileIterations(op, costModel);
  SmallVector<int64_t, 4> tileIterations;
  tileIterations.reserve(numDims);
  if (numDims == 1) {
    int64_t tripCount = tripCounts.front();
    if (tripCount <= 0)
      return std::nullopt;
    int64_t targetTasks = getTargetTileTasks(op, costModel);
    minIter = relaxOwnerLocalPipelineMinIterations(op, tripCount, targetTasks,
                                                   minIter);
    int64_t balanced = sde::ceilDivPositive(tripCount, targetTasks);
    tileIterations.push_back(std::clamp(std::max<int64_t>(balanced, minIter),
                                        int64_t{1}, tripCount));
    return tileIterations;
  }

  int64_t workersPerDim = std::max<int64_t>(
      1, static_cast<int64_t>(std::ceil(
             std::pow(getTargetTileTasks(op, costModel), 1.0 / numDims))));
  for (int64_t tripCount : tripCounts) {
    if (tripCount <= 0)
      return std::nullopt;
    int64_t balanced = sde::ceilDivPositive(tripCount, workersPerDim);
    tileIterations.push_back(std::clamp(std::max<int64_t>(balanced, minIter),
                                        int64_t{1}, tripCount));
  }
  return tileIterations;
}

static void applyStencilTileGuardsToStaticPlan(
    sde::SdeSuIterateOp op, SmallVectorImpl<int64_t> &tileIterations,
    unsigned numDims, sde::SDECostModel &costModel) {
  SmallVector<int64_t> halos = getStencilHaloWidths(op);
  for (unsigned d = 0;
       d < numDims && d < halos.size() && d < tileIterations.size(); ++d)
    tileIterations[d] = std::max<int64_t>(tileIterations[d], halos[d]);

  int64_t elemSize = 8;
  bool foundElem = false;
  op.getBody().walk([&](memref::StoreOp storeOp) {
    Type elemType = storeOp.getValueToStore().getType();
    if (elemType.isF32() || elemType.isInteger(32))
      elemSize = 4;
    foundElem = true;
    return WalkResult::interrupt();
  });
  int64_t cacheLineTile =
      costModel.getL2CacheSize() / (elemSize * std::max<unsigned>(1, numDims));
  cacheLineTile = std::max<int64_t>(1, cacheLineTile);
  for (int64_t &tile : tileIterations)
    tile = std::min<int64_t>(tile, cacheLineTile);
}

static std::optional<PhysicalTilePlan>
buildStencilPhysicalTilePlan(sde::SdeSuIterateOp op,
                             ArrayRef<int64_t> tileIterations) {
  if (op.getPhysicalOwnerDimsAttr() || op.getPhysicalBlockShapeAttr() ||
      op.getInPlaceSharedStateAttr())
    return std::nullopt;
  // The loop-indexed output helper proves the current SDE owner IV only. For
  // multi-dimensional/component stencils the final ND owner plan still belongs
  // to DistributionPlanning, which consumes the PatternAnalysis facts.
  if (op.getLowerBounds().size() != 1)
    return std::nullopt;
  auto effects = sde::collectStructuredMemoryEffects(op.getBody());
  bool ownerLocalPipeline =
      op.getStructuredClassification() ==
          sde::SdeStructuredClassification::elementwise_pipeline &&
      sde::isOwnerLocalPipelineReduction(op);
  if (effects.hasUnknownEffects ||
      (sde::hasInPlaceSelfRead(effects) && !ownerLocalPipeline))
    return std::nullopt;

  std::optional<sde::LoopIndexedOutputPlan> outputPlan =
      sde::findLoopIndexedOutputPlan(op);
  if (!outputPlan || outputPlan->shape.empty() ||
      outputPlan->ownerPhysicalDims.empty())
    return std::nullopt;
  if (outputPlan->ownerPhysicalDims.size() > tileIterations.size())
    return std::nullopt;

  PhysicalTilePlan plan;
  plan.ownerPhysicalDims.assign(outputPlan->ownerPhysicalDims.begin(),
                                outputPlan->ownerPhysicalDims.end());
  plan.blockShape.assign(outputPlan->shape.begin(), outputPlan->shape.end());
  plan.tileIterations.assign(tileIterations.begin(), tileIterations.end());
  for (auto [idx, physicalDim] : llvm::enumerate(plan.ownerPhysicalDims)) {
    if (physicalDim < 0 ||
        static_cast<size_t>(physicalDim) >= plan.blockShape.size())
      return std::nullopt;
    plan.blockShape[physicalDim] = tileIterations[idx];
  }

  plan.haloShape =
      getStencilHaloRadiiForOwnerDims(op, plan.ownerPhysicalDims.size());
  plan.topology = plan.ownerPhysicalDims.size() > 1
                      ? sde::SdeIterationTopology::owner_tile
                      : sde::SdeIterationTopology::owner_strip;
  return plan;
}

static std::optional<PhysicalTilePlan>
buildNdStencilPhysicalTilePlan(sde::SdeSuIterateOp op,
                               sde::SDECostModel &costModel) {
  if (op.getLowerBounds().size() <= 1 || op.getPhysicalOwnerDimsAttr() ||
      op.getPhysicalBlockShapeAttr() || op.getInPlaceSharedStateAttr())
    return std::nullopt;
  auto pattern = op.getPattern();
  if (!pattern || (*pattern != sde::SdePattern::cross_dim_stencil_3d &&
                   *pattern != sde::SdePattern::stencil_tiling_nd &&
                   *pattern != sde::SdePattern::higher_order_stencil))
    return std::nullopt;

  auto effects = sde::collectStructuredMemoryEffects(op.getBody());
  if (effects.hasUnknownEffects || sde::hasInPlaceSelfRead(effects))
    return std::nullopt;

  std::optional<sde::StructuredOutputLayoutPlan> outputPlan =
      sde::findCompatibleOutputLayoutPlan(op);
  if (!outputPlan || outputPlan->shape.empty())
    return std::nullopt;

  auto ownerDims = readI64ArrayAttr(op.getOwnerDimsAttr());
  auto minOffsets = readI64ArrayAttr(op.getAccessMinOffsetsAttr());
  auto maxOffsets = readI64ArrayAttr(op.getAccessMaxOffsetsAttr());
  if (!ownerDims || !minOffsets || !maxOffsets ||
      minOffsets->size() != maxOffsets->size())
    return std::nullopt;

  PhysicalTilePlan plan;
  plan.blockShape.assign(outputPlan->shape.begin(), outputPlan->shape.end());
  plan.tileIterations.assign(outputPlan->shape.begin(),
                             outputPlan->shape.end());
  if (plan.tileIterations.size() < op.getLowerBounds().size())
    return std::nullopt;
  plan.tileIterations.resize(op.getLowerBounds().size());

  SmallVector<int64_t, 4> ownerExtents;
  SmallVector<unsigned, 4> ownerLoopDims;
  for (auto [idx, rawLoopDim] : llvm::enumerate(*ownerDims)) {
    if (rawLoopDim < 0 ||
        static_cast<size_t>(rawLoopDim) >= op.getLowerBounds().size() ||
        static_cast<size_t>(rawLoopDim) >=
            outputPlan->loopDimToPhysicalDim.size() ||
        idx >= minOffsets->size())
      continue;
    int64_t physicalDim = outputPlan->loopDimToPhysicalDim[rawLoopDim];
    if (physicalDim < 0 ||
        static_cast<size_t>(physicalDim) >= outputPlan->shape.size())
      continue;
    int64_t halo =
        std::max<int64_t>(0, std::max(-(*minOffsets)[idx], (*maxOffsets)[idx]));
    if (halo == 0)
      continue;
    ownerLoopDims.push_back(static_cast<unsigned>(rawLoopDim));
    plan.ownerPhysicalDims.push_back(physicalDim);
    plan.haloShape.push_back(halo);
    ownerExtents.push_back(outputPlan->shape[physicalDim]);
  }
  if (plan.ownerPhysicalDims.empty())
    return std::nullopt;

  SmallVector<int64_t, 4> grid = sde::factorStencilWorkersAcrossDims(
      std::max<int64_t>(1, getTargetTileTasks(op, costModel)), ownerExtents,
      plan.haloShape);
  for (auto [idx, physicalDim] : llvm::enumerate(plan.ownerPhysicalDims)) {
    int64_t tile =
        sde::ceilDivPositive(outputPlan->shape[physicalDim], grid[idx]);
    plan.blockShape[physicalDim] = tile;
    unsigned loopDim = ownerLoopDims[idx];
    if (loopDim < plan.tileIterations.size())
      plan.tileIterations[loopDim] = tile;
  }

  plan.topology = plan.ownerPhysicalDims.size() > 1
                      ? sde::SdeIterationTopology::owner_tile
                      : sde::SdeIterationTopology::owner_strip;
  return plan;
}

static void stampPhysicalTilePlan(sde::SdeSuIterateOp op,
                                  const PhysicalTilePlan &plan) {
  op.setPhysicalOwnerDimsAttr(
      buildI64ArrayAttr(op.getContext(), plan.ownerPhysicalDims));
  op.setPhysicalBlockShapeAttr(
      buildI64ArrayAttr(op.getContext(), plan.blockShape));
  op.setLogicalWorkerSliceAttr(
      buildI64ArrayAttr(op.getContext(), plan.blockShape));
  if (llvm::any_of(plan.haloShape, [](int64_t halo) { return halo > 0; }))
    op.setPhysicalHaloShapeAttr(
        buildI64ArrayAttr(op.getContext(), plan.haloShape));
  op.setIterationTopologyAttr(
      sde::SdeIterationTopologyAttr::get(op.getContext(), plan.topology));
}

static std::optional<int64_t> getPositiveConstantIndex(Value value) {
  int64_t constant = 0;
  if (::mlir::carts::ValueAnalysis::getConstantIndex(value, constant) &&
      constant > 0)
    return constant;
  std::optional<int64_t> folded =
      ::mlir::carts::ValueAnalysis::tryFoldConstantIndex(value);
  if (folded && *folded > 0)
    return folded;
  return std::nullopt;
}

static std::optional<unsigned> mapLoopDimToPhysicalDim(sde::SdeSuIterateOp op,
                                                       unsigned loopDim) {
  if (std::optional<SmallVector<int64_t, 4>> ownerDims =
          readI64ArrayAttr(op.getPhysicalOwnerDimsAttr())) {
    if (loopDim < ownerDims->size() && (*ownerDims)[loopDim] >= 0)
      return static_cast<unsigned>((*ownerDims)[loopDim]);
  }
  return loopDim;
}

static std::optional<SmallVector<int64_t, 4>>
alignStaticShapeAttrToSteps(sde::SdeSuIterateOp op, ArrayAttr attr,
                            ArrayRef<Value> tiledSteps,
                            ArrayRef<bool> parallelMask) {
  std::optional<SmallVector<int64_t, 4>> shape = readI64ArrayAttr(attr);
  if (!shape)
    return std::nullopt;

  for (unsigned dim = 0, e = std::min(tiledSteps.size(),
                                      static_cast<size_t>(parallelMask.size()));
       dim < e; ++dim) {
    if (!parallelMask[dim])
      continue;
    std::optional<int64_t> step = getPositiveConstantIndex(tiledSteps[dim]);
    if (!step || *step <= 1)
      continue;
    std::optional<unsigned> physicalDim = mapLoopDimToPhysicalDim(op, dim);
    if (!physicalDim || *physicalDim >= shape->size())
      continue;
    int64_t &extent = (*shape)[*physicalDim];
    if (extent <= 0)
      return std::nullopt;
    int64_t chunks = (extent + *step - 1) / *step;
    if (chunks > std::numeric_limits<int64_t>::max() / *step)
      return std::nullopt;
    extent = chunks * *step;
  }

  return shape;
}

static void
alignExistingStaticPhysicalPlanToSteps(sde::SdeSuIterateOp op,
                                       ArrayRef<Value> tiledSteps,
                                       ArrayRef<bool> parallelMask) {
  if (std::optional<SmallVector<int64_t, 4>> blockShape =
          alignStaticShapeAttrToSteps(op, op.getPhysicalBlockShapeAttr(),
                                      tiledSteps, parallelMask))
    op.setPhysicalBlockShapeAttr(
        buildI64ArrayAttr(op.getContext(), *blockShape));

  if (std::optional<SmallVector<int64_t, 4>> workerSlice =
          alignStaticShapeAttrToSteps(op, op.getLogicalWorkerSliceAttr(),
                                      tiledSteps, parallelMask))
    op.setLogicalWorkerSliceAttr(
        buildI64ArrayAttr(op.getContext(), *workerSlice));
}

static bool isTilingCandidate(sde::SdeSuIterateOp op, Block &body) {
  if (op.getChunkSize())
    return false;
  if (op->getParentOfType<sde::SdeSuIterateOp>())
    return false;
  if (op.getLowerBounds().empty())
    return false;
  if (!op.getStructuredClassificationAttr())
    return false;

  auto classification = *op.getStructuredClassification();
  switch (classification) {
  case sde::SdeStructuredClassification::stencil:
    return op.getReductionAccumulators().size() == 0 &&
           isStencilCandidate(op, body);
  case sde::SdeStructuredClassification::elementwise:
    if (op.getReductionAccumulators().size() != 0)
      return false;
    return isExecutableInnermostBody(body) ||
           hasPerfectNestedScalarLoopNest(body, /*numLoops=*/2);
  case sde::SdeStructuredClassification::elementwise_pipeline:
    if (op.getReductionAccumulators().size() != 0)
      return false;
    return isExecutableInnermostBody(body) ||
           hasPerfectNestedScalarLoopNest(body, /*numLoops=*/2) ||
           sde::isOwnerLocalPipelineReduction(op);
  case sde::SdeStructuredClassification::matmul:
    if (op.getReductionAccumulators().size() != 0)
      return false;
    return isDirectMemoryMatmulCandidate(op, body);
  case sde::SdeStructuredClassification::reduction:
    if (op.getReductionAccumulators().size() != 0)
      return false;
    if (!sde::isOwnerLocalPipelineReduction(op))
      return false;
    return isExecutableInnermostBody(body) ||
           hasPerfectNestedScalarLoopNest(body, /*numLoops=*/2);
  }
  return false;
}

static void cloneBodyIntoTileLoop(Block &srcBody, IRMapping &mapper,
                                  OpBuilder &builder) {
  for (Operation &op : srcBody.without_terminator()) {
    builder.clone(op, mapper);
  }
}

static bool stripMineLoop(scf::ForOp loop, Value tileIterations) {
  if (!loop || !tileIterations || loop.getNumResults() != 0 ||
      !loop.getInitArgs().empty())
    return false;

  int64_t tileConstant = 0;
  if (::mlir::carts::ValueAnalysis::getConstantIndex(tileIterations,
                                                     tileConstant) &&
      tileConstant <= 1)
    return false;

  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  Value originalStep = loop.getStep();
  Value tileStep =
      arith::MulIOp::create(builder, loc, originalStep, tileIterations);

  auto outerLoop = scf::ForOp::create(builder, loc, loop.getLowerBound(),
                                      loop.getUpperBound(), tileStep);
  outerLoop->setAttrs(loop->getAttrs());

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(outerLoop.getBody());
  Value tileBase = outerLoop.getInductionVar();
  Value tileLimit = arith::AddIOp::create(builder, loc, tileBase, tileStep);
  Value tileUpper =
      arith::MinUIOp::create(builder, loc, tileLimit, loop.getUpperBound());
  auto innerLoop =
      scf::ForOp::create(builder, loc, tileBase, tileUpper, originalStep);
  innerLoop->setAttrs(loop->getAttrs());

  IRMapping mapping;
  mapping.map(loop.getInductionVar(), innerLoop.getInductionVar());
  builder.setInsertionPointToStart(innerLoop.getBody());
  for (Operation &op : loop.getBody()->without_terminator())
    builder.clone(op, mapping);

  loop.erase();
  return true;
}

static unsigned stripMineDirectMatmulColumnLoops(Block &body, Value outputRoot,
                                                 Value ownerIv,
                                                 Value columnTileIterations) {
  SmallVector<scf::ForOp, 4> columnLoops;
  collectDirectMatmulColumnLoops(body, outputRoot, ownerIv, columnLoops);

  unsigned tiled = 0;
  for (scf::ForOp loop : llvm::reverse(columnLoops)) {
    if (!loop || loop->getParentRegion() == nullptr)
      continue;
    if (stripMineLoop(loop, columnTileIterations))
      ++tiled;
  }
  return tiled;
}

static void stampDirectMatmulTilePlan(sde::SdeSuIterateOp op,
                                      const DirectMatmulTilePlan &plan) {
  SmallVector<int64_t, 4> blockShape = plan.output.shape;
  if (blockShape.empty())
    return;
  blockShape[0] = plan.rowTile;

  // Direct-memory matmul keeps full output rows in one owner task until SDE
  // can also tile the input access windows. Splitting columns across owner
  // tasks duplicates the k-sweep against coarse A/B inputs and regresses large
  // GEMM.
  op.setPhysicalOwnerDimsAttr(
      buildI64ArrayAttr(op.getContext(), SmallVector<int64_t, 1>{0}));
  op.setPhysicalBlockShapeAttr(buildI64ArrayAttr(op.getContext(), blockShape));
  op.setLogicalWorkerSliceAttr(buildI64ArrayAttr(op.getContext(), blockShape));
  op.setIterationTopologyAttr(sde::SdeIterationTopologyAttr::get(
      op.getContext(), sde::SdeIterationTopology::owner_strip));
}

struct TilingPass : public sde::impl::TilingBase<TilingPass> {
  explicit TilingPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    if (!costModel)
      return;

    SmallVector<sde::SdeSuIterateOp> rewrites;
    getOperation().walk([&](sde::SdeSuIterateOp op) {
      Block *body = sde::getSuIterateComputeBlock(op);
      if (!isTilingCandidate(op, *body))
        return;

      std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation());
      if (tripCount && *tripCount <= 1)
        return;
      rewrites.push_back(op);
    });

    for (sde::SdeSuIterateOp op : rewrites) {
      PatternRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();
      unsigned numDims = op.getLowerBounds().size();
      bool directMatmul = false;
      std::optional<DirectMatmulTilePlan> directMatmulPlan;
      if (op.getStructuredClassification() ==
          sde::SdeStructuredClassification::matmul) {
        directMatmulPlan =
            buildDirectMatmulTilePlan(rewriter, loc, op, *costModel);
        if (!directMatmulPlan)
          continue;
        directMatmul = true;
      }

      // Determine which dims are parallel (should tile) vs reduction (skip).
      SmallVector<bool> parallelMask = getParallelDimMask(op);

      // Compute per-dim tile iterations.
      SmallVector<Value> perDimTileIter;
      if (directMatmul) {
        perDimTileIter.push_back(directMatmulPlan->rowTileValue);
      } else if (numDims == 1) {
        // 1-D fast path: preserves existing static trip count optimization.
        if (!parallelMask[0]) {
          // Single reduction dim -- nothing to tile.
          continue;
        }
        Value tileIterations;
        if (std::optional<int64_t> tripCount =
                getStaticTripCount(op.getOperation())) {
          int64_t targetTasks = getTargetTileTasks(op, *costModel);
          int64_t minIterations =
              relaxOwnerLocalPipelineMinIterations(
                  op, *tripCount, targetTasks,
                  getMinTileIterations(op, *costModel));
          int64_t balancedTile = llvm::divideCeil(*tripCount, targetTasks);
          int64_t tileCount = std::clamp(
              std::max<int64_t>(balancedTile, minIterations),
              int64_t{1}, *tripCount);
          if (tileCount <= 1)
            continue;
          tileIterations = getConstantIndex(rewriter, loc, tileCount);
        } else {
          tileIterations =
              buildTileIterationValue(rewriter, loc, op, *costModel);
        }
        if (!tileIterations)
          continue;
        perDimTileIter.push_back(tileIterations);
      } else {
        // N-dim path: distribute workers across dimensions.
        perDimTileIter =
            buildPerDimTileIterations(rewriter, loc, op, *costModel);
        if (perDimTileIter.empty())
          continue;
      }

      std::optional<PhysicalTilePlan> physicalTilePlan =
          buildNdStencilPhysicalTilePlan(op, *costModel);
      if (physicalTilePlan) {
        perDimTileIter.clear();
        for (int64_t tile : physicalTilePlan->tileIterations)
          perDimTileIter.push_back(getConstantIndex(rewriter, loc, tile));
      }

      if (!physicalTilePlan && !directMatmul &&
          sde::isOwnerLocalPipelineReduction(op)) {
        if (auto staticTileIterations =
                computeStaticTileIterations(op, *costModel))
          physicalTilePlan =
              buildStencilPhysicalTilePlan(op, *staticTileIterations);
      }

      // For stencils, enforce halo-aware minimum tile size per dimension.
      if (op.getStructuredClassification() ==
          sde::SdeStructuredClassification::stencil) {
        SmallVector<int64_t> halos = getStencilHaloWidths(op);
        for (unsigned d = 0; d < numDims && d < halos.size(); ++d) {
          Value haloVal = getConstantIndex(rewriter, loc, halos[d]);
          perDimTileIter[d] =
              arith::MaxUIOp::create(rewriter, loc, perDimTileIter[d], haloVal);
        }
        // Cache-friendly tile: prefer tiles that fit in L2 cache
        int64_t elemSize = 8; // default f64
        bool foundElem = false;
        op.getBody().walk([&](memref::StoreOp storeOp) {
          Type elemType = storeOp.getValueToStore().getType();
          if (elemType.isF32() || elemType.isInteger(32))
            elemSize = 4;
          foundElem = true;
          return WalkResult::interrupt();
        });
        int64_t l2Size = costModel->getL2CacheSize();
        int64_t cacheLineTile =
            l2Size / (elemSize * std::max<unsigned>(1, numDims));
        Value cacheVal = getConstantIndex(rewriter, loc,
                                          std::max<int64_t>(1, cacheLineTile));
        for (unsigned d = 0; d < numDims; ++d) {
          perDimTileIter[d] = arith::MinUIOp::create(
              rewriter, loc, perDimTileIter[d], cacheVal);
        }
      }

      // Validate steps and compute tiled steps per dim.
      // For reduction dims, keep the original step (no tiling).
      SmallVector<Value> tiledSteps;
      bool badStep = false;
      bool anyTiled = false;
      for (unsigned d = 0; d < numDims; ++d) {
        Value originalStep = op.getSteps()[d];
        int64_t constantStep = 0;
        if (::mlir::carts::ValueAnalysis::getConstantIndex(originalStep,
                                                           constantStep) &&
            constantStep <= 0) {
          badStep = true;
          break;
        }
        if (parallelMask[d]) {
          tiledSteps.push_back(arith::MulIOp::create(
              rewriter, loc, originalStep, perDimTileIter[d]));
          anyTiled = true;
        } else {
          tiledSteps.push_back(originalStep);
        }
      }
      if (badStep || !anyTiled)
        continue;

      if (!physicalTilePlan && !directMatmul &&
          op.getStructuredClassification() ==
              sde::SdeStructuredClassification::stencil) {
        if (auto staticTileIterations =
                computeStaticTileIterations(op, *costModel)) {
          applyStencilTileGuardsToStaticPlan(op, *staticTileIterations, numDims,
                                             *costModel);
          physicalTilePlan =
              buildStencilPhysicalTilePlan(op, *staticTileIterations);
        }
      }

      auto newOp = sde::SdeSuIterateOp::create(
          rewriter, loc, /*resultTypes=*/TypeRange{}, op.getLowerBounds(),
          op.getUpperBounds(), ValueRange{tiledSteps}, op.getScheduleAttr(),
          op.getChunkSize(), op.getNowaitAttr(), op.getReductionAccumulators(),
          op.getReductionKindsAttr(), op.getReductionStrategyAttr(),
          op.getPartialReductionAttr(), op.getPartialReductionDimsAttr(),
          op.getPartialReductionOwnerDimsAttr(),
          op.getStructuredClassificationAttr(), op.getPatternAttr(),
          op.getAccessMinOffsetsAttr(), op.getAccessMaxOffsetsAttr(),
          op.getOwnerDimsAttr(), op.getSpatialDimsAttr(),
          op.getWriteFootprintAttr(), op.getPhysicalOwnerDimsAttr(),
          op.getPhysicalBlockShapeAttr(), op.getLogicalWorkerSliceAttr(),
          op.getPhysicalHaloShapeAttr(), op.getIterationTopologyAttr(),
          op.getRepetitionStructureAttr(), op.getAsyncStrategyAttr(),
          op.getCpsGroupIdAttr(), op.getCpsStageIndexAttr(),
          op.getCpsStageCountAttr(), op.getDistributionKindAttr(),
          op.getInPlaceSafeAttr(), op.getInPlaceSharedStateAttr(),
          op.getVectorizeWidthAttr(), op.getUnrollFactorAttr(),
          op.getInterleaveCountAttr());
      newOp->setAttrs(sde::getRewrittenAttrs(op));
      if (!physicalTilePlan && !directMatmul)
        alignExistingStaticPhysicalPlanToSteps(newOp, tiledSteps, parallelMask);
      if (physicalTilePlan)
        stampPhysicalTilePlan(newOp, *physicalTilePlan);

      Block &newBody = sde::ensureBlock(newOp.getBody());
      for (unsigned d = newBody.getNumArguments(); d < numDims; ++d)
        newBody.addArgument(rewriter.getIndexType(), loc);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newBody);

      Block &srcBody = op.getBody().front();
      Block *computeBody = sde::getSuIterateComputeBlock(op);

      IRMapping mapper;
      SmallVector<scf::ForOp, 4> tileLoops;
      for (unsigned d = 0; d < numDims; ++d) {
        Value tileBase = newBody.getArgument(d);
        if (!parallelMask[d]) {
          mapper.map(srcBody.getArgument(d), tileBase);
          continue;
        }
        Value tileLimit =
            arith::AddIOp::create(rewriter, loc, tileBase, tiledSteps[d]);
        Value tileUpper = arith::MinUIOp::create(rewriter, loc, tileLimit,
                                                 op.getUpperBounds()[d]);
        Value originalStep = op.getSteps()[d];
        auto tileLoop = scf::ForOp::create(rewriter, loc, tileBase, tileUpper,
                                           originalStep);
        tileLoops.push_back(tileLoop);
        mapper.map(srcBody.getArgument(d), tileLoop.getInductionVar());
        rewriter.setInsertionPointToStart(tileLoop.getBody());
      }

      cloneBodyIntoTileLoop(*computeBody, mapper, rewriter);

      if (directMatmul && !tileLoops.empty()) {
        Value outputRoot =
            mapper.lookupOrDefault(directMatmulPlan->output.root);
        Value ownerIv = mapper.lookupOrDefault(srcBody.getArgument(0));
        unsigned tiledColumns = stripMineDirectMatmulColumnLoops(
            *tileLoops.back().getBody(), outputRoot, ownerIv,
            directMatmulPlan->columnTileValue);
        if (tiledColumns == 0) {
          rewriter.eraseOp(newOp);
          continue;
        }
        stampDirectMatmulTilePlan(newOp, *directMatmulPlan);
      }

      // Yield at the end of the su_iterate body (not inside nested loops).
      rewriter.setInsertionPointToEnd(&newBody);
      sde::SdeYieldOp::create(rewriter, loc, ValueRange{});

      rewriter.eraseOp(op);
    }
  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createTilingPass(sde::SDECostModel *costModel) {
  return std::make_unique<TilingPass>(costModel);
}

} // namespace mlir::carts::sde
