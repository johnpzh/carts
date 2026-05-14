///==========================================================================///
/// File: Tiling.cpp
///
/// Pattern-driven tiling in SDE. The current implementation focuses on
/// executable loop nests whose tensor-backed linalg carrier proves that
/// strip-mining the outer SDE loop preserves a disjoint write set. The pass
/// rewrites the surrounding sde.su_iterate so the tiled loop nest survives
/// SDE->ARTS lowering while tensor carriers remain an SDE-only concern.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_TILING
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "arts/utils/LoopUtils.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/ValueAnalysis.h"
#include "arts/utils/costs/SDECostModel.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>

using namespace mlir;
using namespace mlir::arts;

namespace {

static Value stripSimpleMemrefAlias(Value value) {
  while (auto castOp = value.getDefiningOp<memref::CastOp>())
    value = castOp.getSource();
  return value;
}

static Value getConstantIndex(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

static Value buildLogicalWorkerCapacityValue(OpBuilder &builder,
                                             Location loc) {
  Value logicalWorkers =
      sde::SdeResourceQueryOp::create(
          builder, loc, sde::SdeResourceQueryKind::logicalWorkers)
          .getResult();
  return arith::MaxUIOp::create(builder, loc, logicalWorkers,
                                getConstantIndex(builder, loc, 1));
}

static Value buildTripCountValue(OpBuilder &builder, Location loc,
                                 sde::SdeSuIterateOp op) {
  if (std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation()))
    return getConstantIndex(builder, loc, *tripCount);

  if (op.getLowerBounds().size() != 1 || op.getUpperBounds().size() != 1 ||
      op.getSteps().size() != 1)
    return Value();

  Value lowerBound = op.getLowerBounds().front();
  Value upperBound = op.getUpperBounds().front();
  Value step = op.getSteps().front();
  Value zero = getConstantIndex(builder, loc, 0);
  Value one = getConstantIndex(builder, loc, 1);

  Value span = arith::SubIOp::create(builder, loc, upperBound, lowerBound);

  int64_t constantStep = 0;
  Value safeStep = step;
  if (ValueAnalysis::getConstantIndex(step, constantStep)) {
    if (constantStep <= 0)
      return Value();
  } else {
    Value stepIsTooSmall = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sle, step, zero);
    safeStep = arith::SelectOp::create(builder, loc, stepIsTooSmall, one, step);
  }

  Value spanIsNegative = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::slt, span, zero);
  Value nonNegativeSpan =
      arith::SelectOp::create(builder, loc, spanIsNegative, zero, span);
  return arith::CeilDivSIOp::create(builder, loc, nonNegativeSpan, safeStep);
}

static Value buildTileIterationValue(OpBuilder &builder, Location loc,
                                     sde::SdeSuIterateOp op,
                                     sde::SDECostModel &costModel) {
  Value tripCount = buildTripCountValue(builder, loc, op);
  if (!tripCount)
    return Value();

  Value one = getConstantIndex(builder, loc, 1);
  Value workerCountValue = buildLogicalWorkerCapacityValue(builder, loc);
  Value minIterationsValue = getConstantIndex(
      builder, loc,
      std::max<int64_t>(1, costModel.getMinIterationsPerWorker()));

  Value clampedTripCount = arith::MaxUIOp::create(builder, loc, tripCount, one);
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
    if (ValueAnalysis::getConstantIndex(step, constantStep)) {
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
      std::max(1, static_cast<int>(std::ceil(
                      std::pow(costModel.getLogicalWorkerCapacity(),
                               1.0 / numDims))));
  int64_t minIter = costModel.getMinIterationsPerWorker();

  SmallVector<Value> tileIterations;
  for (unsigned d = 0; d < numDims; ++d) {
    Value one = getConstantIndex(builder, loc, 1);
    Value workersVal = getConstantIndex(builder, loc, workersPerDim);
    Value minIterVal =
        getConstantIndex(builder, loc, std::max<int64_t>(1, minIter));
    Value clampedTrip =
        arith::MaxUIOp::create(builder, loc, tripCounts[d], one);
    Value balanced =
        arith::CeilDivUIOp::create(builder, loc, clampedTrip, workersVal);
    Value preferred =
        arith::MaxUIOp::create(builder, loc, balanced, minIterVal);
    Value tile = arith::MinUIOp::create(builder, loc, preferred, clampedTrip);
    tileIterations.push_back(tile);
  }
  return tileIterations;
}

static int64_t ceilDivPositive(int64_t value, int64_t divisor) {
  return llvm::divideCeil(std::max<int64_t>(1, value),
                          std::max<int64_t>(1, divisor));
}

static int64_t chooseMatmulColumnWorkers(int64_t workers) {
  workers = std::max<int64_t>(1, workers);
  return std::max<int64_t>(
      1, static_cast<int64_t>(
             std::ceil(std::sqrt(static_cast<double>(workers)))));
}

static int64_t chooseStaticMatmulTile(int64_t extent, int64_t participants,
                                      int64_t minIterations) {
  if (extent <= 1)
    return std::max<int64_t>(1, extent);
  int64_t balanced = ceilDivPositive(extent, participants);
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
  std::optional<sde::LoopIndexedOutputPlan> outputPlan =
      sde::findLoopIndexedOutputPlan(op);
  if (!outputPlan || outputPlan->shape.size() < 2)
    return std::nullopt;

  int64_t rows = outputPlan->shape[0];
  int64_t columns = outputPlan->shape[1];
  if (rows <= 1 || columns <= 1)
    return std::nullopt;

  int64_t workers =
      std::max<int64_t>(1, costModel.getLogicalWorkerCapacity());
  int64_t columnWorkers = chooseMatmulColumnWorkers(workers);
  int64_t minIterations =
      std::max<int64_t>(1, costModel.getMinIterationsPerWorker());

  DirectMatmulTilePlan plan;
  plan.output = std::move(*outputPlan);
  /// Keep the SDE-owned row dimension exposed to the worker distributor.
  /// The column tile is an inner locality tile; row tiling must not collapse
  /// the distributed task count down to the square-root worker grid.
  plan.rowTile = chooseStaticMatmulTile(rows, workers, /*minIterations=*/1);
  plan.columnTile =
      chooseStaticMatmulTile(columns, columnWorkers, minIterations);
  if (plan.rowTile <= 1 && plan.columnTile <= 1)
    return std::nullopt;

  plan.rowTileValue = getConstantIndex(builder, loc, plan.rowTile);
  plan.columnTileValue = getConstantIndex(builder, loc, plan.columnTile);
  return plan;
}

static bool isCarrierOp(Operation &op) {
  return isa<bufferization::ToTensorOp, sde::SdeMuMemrefToTensorOp,
             tensor::EmptyOp, linalg::GenericOp, tensor::ExtractSliceOp,
             tensor::InsertSliceOp>(op);
}

static bool isScalarExecutableOp(Operation &op) {
  if (isa<memref::LoadOp, memref::StoreOp>(op))
    return true;
  return op.getNumRegions() == 0 && isMemoryEffectFree(&op);
}

static bool isExecutableInnermostBody(Block &body) {
  for (Operation &op : body.without_terminator()) {
    if (isCarrierOp(op))
      continue;
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
      if (isCarrierOp(op))
        continue;
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

static linalg::GenericOp findTensorCarrier(Block &body) {
  linalg::GenericOp tensorGeneric;
  for (Operation &op : body) {
    auto generic = dyn_cast<linalg::GenericOp>(op);
    if (!generic)
      continue;
    if (!llvm::all_of(generic.getDpsInputs(), [](Value operand) {
          return isa<TensorType>(operand.getType());
        }))
      continue;
    if (!llvm::all_of(generic.getDpsInits(), [](Value operand) {
          return isa<TensorType>(operand.getType());
        }))
      continue;
    if (tensorGeneric)
      return nullptr;
    tensorGeneric = generic;
  }
  return tensorGeneric;
}

static bool hasDisjointWriteSet(Block &body, linalg::GenericOp tensorGeneric) {
  unsigned loadCount = 0;
  unsigned storeCount = 0;
  SmallVector<Value> writtenMemrefs;

  body.walk([&](memref::LoadOp) { ++loadCount; });
  body.walk([&](memref::StoreOp storeOp) {
    ++storeCount;
    Value base = stripSimpleMemrefAlias(storeOp.getMemref());
    if (!llvm::is_contained(writtenMemrefs, base))
      writtenMemrefs.push_back(base);
  });

  if (loadCount != tensorGeneric.getNumDpsInputs() ||
      storeCount != tensorGeneric.getNumDpsInits() ||
      writtenMemrefs.size() != static_cast<size_t>(storeCount))
    return false;

  for (auto [outputIndex, output] :
       llvm::enumerate(tensorGeneric.getDpsInits())) {
    AffineMap indexingMap = tensorGeneric.getMatchingIndexingMap(
        tensorGeneric.getDpsInitOperand(outputIndex));
    if (!indexingMap.isProjectedPermutation())
      return false;
  }

  return true;
}

static bool isElementwiseTensorCandidate(Block &body,
                                         linalg::GenericOp tensorGeneric,
                                         unsigned numSuDims) {
  if (tensorGeneric.getNumLoops() == 0)
    return false;
  if (!llvm::all_of(tensorGeneric.getIteratorTypesArray(),
                    [](utils::IteratorType type) {
                      return type == utils::IteratorType::parallel;
                    }))
    return false;
  if (tensorGeneric.getNumDpsInits() == 0)
    return false;
  if (!hasDisjointWriteSet(body, tensorGeneric))
    return false;
  // The su_iterate provides numSuDims induction variables directly as block
  // args. The remaining dimensions must appear as nested scf.for loops.
  unsigned nestedLoops = tensorGeneric.getNumLoops() >= numSuDims
                             ? tensorGeneric.getNumLoops() - numSuDims + 1
                             : 1;
  return hasPerfectNestedScalarLoopNest(body, nestedLoops);
}

static bool isMatmulTensorCandidate(Block &body,
                                    linalg::GenericOp tensorGeneric) {
  if (tensorGeneric.getNumLoops() != 3 || tensorGeneric.getNumDpsInits() != 1)
    return false;

  unsigned numParallel = 0;
  unsigned numReduction = 0;
  for (utils::IteratorType type : tensorGeneric.getIteratorTypesArray()) {
    if (type == utils::IteratorType::parallel)
      ++numParallel;
    else if (type == utils::IteratorType::reduction)
      ++numReduction;
  }
  if (numParallel != 2 || numReduction != 1)
    return false;

  unsigned nonCarrierOps = 0;
  for (Operation &nested : body.without_terminator()) {
    if (isCarrierOp(nested))
      continue;
    if (!isa<scf::ForOp>(nested))
      return false;
    ++nonCarrierOps;
  }
  return nonCarrierOps == 1;
}

static bool isReductionTensorCandidate(Block &body,
                                       linalg::GenericOp tensorGeneric) {
  if (tensorGeneric.getNumLoops() == 0 || tensorGeneric.getNumDpsInits() == 0)
    return false;
  // Must have at least one parallel dim to tile.
  bool hasParallel = false;
  bool hasReduction = false;
  for (utils::IteratorType type : tensorGeneric.getIteratorTypesArray()) {
    if (type == utils::IteratorType::parallel)
      hasParallel = true;
    else if (type == utils::IteratorType::reduction)
      hasReduction = true;
  }
  if (!hasParallel || !hasReduction)
    return false;
  // Must have a single nested scf.for (the loop nest).
  unsigned nonCarrierOps = 0;
  for (Operation &nested : body.without_terminator()) {
    if (isCarrierOp(nested))
      continue;
    if (!isa<scf::ForOp>(nested))
      return false;
    ++nonCarrierOps;
  }
  return nonCarrierOps == 1;
}

static bool loopWritesOwnerColumn(scf::ForOp loop, Value outputRoot,
                                  Value ownerIv) {
  if (!loop || !outputRoot || !ownerIv || loop.getNumResults() != 0 ||
      !loop.getInitArgs().empty())
    return false;

  Value loopIv = loop.getInductionVar();
  bool matched = false;
  loop.walk([&](memref::StoreOp store) {
    Value root = ValueAnalysis::stripMemrefViewOps(store.getMemref());
    if (root != outputRoot)
      return WalkResult::advance();

    ValueRange indices = store.getIndices();
    if (indices.size() < 2)
      return WalkResult::advance();
    if (!ValueAnalysis::dependsOn(indices.front(), ownerIv))
      return WalkResult::advance();

    for (Value index : indices.drop_front()) {
      if (ValueAnalysis::dependsOn(index, loopIv)) {
        matched = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return matched;
}

static void collectDirectMatmulColumnLoops(
    Block &body, Value outputRoot, Value ownerIv,
    SmallVectorImpl<scf::ForOp> &columnLoops) {
  body.walk([&](scf::ForOp loop) {
    if (loopWritesOwnerColumn(loop, outputRoot, ownerIv))
      columnLoops.push_back(loop);
  });
}

static bool isDirectMemoryMatmulCandidate(sde::SdeSuIterateOp op,
                                          Block &body) {
  if (op.getLowerBounds().size() != 1 || op.getReductionAccumulators().size() != 0)
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
/// (keep original step). The su_iterate's first dim maps to the first carrier
/// iterator type, etc.
static SmallVector<bool> getParallelDimMask(sde::SdeSuIterateOp op,
                                            linalg::GenericOp tensorGeneric) {
  unsigned numDims = op.getLowerBounds().size();
  SmallVector<bool> mask(numDims, true); // default: tile everything
  if (!tensorGeneric)
    return mask;
  auto iterTypes = tensorGeneric.getIteratorTypesArray();
  for (unsigned d = 0; d < numDims && d < iterTypes.size(); ++d) {
    mask[d] = (iterTypes[d] == utils::IteratorType::parallel);
  }
  return mask;
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
      if (isCarrierOp(nested))
        continue;
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

static SmallVector<int64_t, 4>
factorWorkersAcrossDims(int64_t workers, ArrayRef<int64_t> extents) {
  SmallVector<int64_t, 4> grid(extents.size(), 1);
  if (workers <= 1 || extents.empty())
    return grid;

  SmallVector<int64_t, 8> factors;
  int64_t remaining = workers;
  for (int64_t factor = 2; factor * factor <= remaining; ++factor) {
    while (remaining % factor == 0) {
      factors.push_back(factor);
      remaining /= factor;
    }
  }
  if (remaining > 1)
    factors.push_back(remaining);
  llvm::sort(factors, std::greater<int64_t>());

  for (int64_t factor : factors) {
    unsigned bestDim = 0;
    int64_t bestSpan = -1;
    for (auto [idx, extent] : llvm::enumerate(extents)) {
      int64_t span = ceilDivPositive(extent, grid[idx]);
      if (span > bestSpan) {
        bestSpan = span;
        bestDim = static_cast<unsigned>(idx);
      }
    }
    grid[bestDim] *= factor;
  }
  return grid;
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
    if (!ValueAnalysis::getConstantIndex(op.getLowerBounds()[d], lb) ||
        !ValueAnalysis::getConstantIndex(op.getUpperBounds()[d], ub) ||
        !ValueAnalysis::getConstantIndex(op.getSteps()[d], step) || step <= 0)
      return std::nullopt;
    tripCounts.push_back(ceilDivPositive(std::max<int64_t>(0, ub - lb), step));
  }

  int64_t minIter = std::max<int64_t>(1, costModel.getMinIterationsPerWorker());
  SmallVector<int64_t, 4> tileIterations;
  tileIterations.reserve(numDims);
  if (numDims == 1) {
    int64_t tripCount = tripCounts.front();
    if (tripCount <= 0)
      return std::nullopt;
    int64_t balanced =
        ceilDivPositive(
            tripCount,
            std::max<int64_t>(1, costModel.getLogicalWorkerCapacity()));
    tileIterations.push_back(
        std::clamp(std::max<int64_t>(balanced, minIter), int64_t{1},
                   tripCount));
    return tileIterations;
  }

  int64_t workersPerDim =
      std::max<int64_t>(1, static_cast<int64_t>(std::ceil(std::pow(
                              costModel.getLogicalWorkerCapacity(),
                              1.0 / numDims))));
  for (int64_t tripCount : tripCounts) {
    if (tripCount <= 0)
      return std::nullopt;
    int64_t balanced = ceilDivPositive(tripCount, workersPerDim);
    tileIterations.push_back(
        std::clamp(std::max<int64_t>(balanced, minIter), int64_t{1},
                   tripCount));
  }
  return tileIterations;
}

static void applyStencilTileGuardsToStaticPlan(
    sde::SdeSuIterateOp op, linalg::GenericOp tensorGeneric,
    SmallVectorImpl<int64_t> &tileIterations, unsigned numDims,
    sde::SDECostModel &costModel) {
  SmallVector<int64_t> halos = getStencilHaloWidths(op);
  for (unsigned d = 0; d < numDims && d < halos.size() &&
                       d < tileIterations.size();
       ++d)
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
  if (!foundElem && tensorGeneric && tensorGeneric.getNumDpsInits() > 0) {
    Type outTy = tensorGeneric.getDpsInits()[0].getType();
    if (auto shapedTy = dyn_cast<ShapedType>(outTy)) {
      Type et = shapedTy.getElementType();
      if (et.isF32() || et.isInteger(32))
        elemSize = 4;
    }
  }

  int64_t cacheLineTile = costModel.getL2CacheSize() /
                          (elemSize * std::max<unsigned>(1, numDims));
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
  if (effects.hasUnknownEffects || sde::hasInPlaceSelfRead(effects))
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
  if (!pattern ||
      (*pattern != sde::SdePattern::cross_dim_stencil_3d &&
       *pattern != sde::SdePattern::stencil_tiling_nd &&
       *pattern != sde::SdePattern::higher_order_stencil))
    return std::nullopt;

  auto effects = sde::collectStructuredMemoryEffects(op.getBody());
  if (effects.hasUnknownEffects || sde::hasInPlaceSelfRead(effects))
    return std::nullopt;

  std::optional<sde::LoopIndexedOutputPlan> outputPlan =
      sde::findLoopIndexedOutputPlan(op);
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
  plan.tileIterations.assign(outputPlan->shape.begin(), outputPlan->shape.end());
  if (plan.tileIterations.size() < op.getLowerBounds().size())
    return std::nullopt;
  plan.tileIterations.resize(op.getLowerBounds().size());

  SmallVector<int64_t, 4> ownerExtents;
  for (auto [idx, rawDim] : llvm::enumerate(*ownerDims)) {
    if (idx >= op.getLowerBounds().size() || rawDim < 0 ||
        static_cast<size_t>(rawDim) >= outputPlan->shape.size() ||
        idx >= minOffsets->size())
      continue;
    int64_t halo = std::max<int64_t>(0, std::max(-(*minOffsets)[idx],
                                                 (*maxOffsets)[idx]));
    if (halo == 0)
      continue;
    plan.ownerPhysicalDims.push_back(rawDim);
    plan.haloShape.push_back(halo);
    ownerExtents.push_back(outputPlan->shape[rawDim]);
  }
  if (plan.ownerPhysicalDims.empty())
    return std::nullopt;

  SmallVector<int64_t, 4> grid =
      factorWorkersAcrossDims(
          std::max<int64_t>(1, costModel.getLogicalWorkerCapacity()),
          ownerExtents);
  for (auto [idx, physicalDim] : llvm::enumerate(plan.ownerPhysicalDims)) {
    int64_t tile =
        ceilDivPositive(outputPlan->shape[physicalDim], grid[idx]);
    plan.blockShape[physicalDim] = tile;
    if (static_cast<size_t>(physicalDim) < plan.tileIterations.size())
      plan.tileIterations[physicalDim] = tile;
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

/// Check whether a body block is carrier-authoritative: it contains a
/// linalg.generic carrier but NO scalar executable ops (memref.load/store)
/// anywhere in the body, including nested regions (except inside the carrier
/// itself). This distinguishes carrier-authoritative IR (where the carrier IS
/// the computation) from dual-representation IR (where scalar ops coexist).
static bool isCarrierAuthoritative(Block &body) {
  bool hasCarrier = false;
  bool hasScalar = false;
  for (Operation &op : body.without_terminator()) {
    if (isa<linalg::GenericOp>(op)) {
      hasCarrier = true;
      continue; // Don't walk into the carrier's own body.
    }
    if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      hasScalar = true;
      continue;
    }
    // Walk into nested regions (scf.for, etc.) to find scalar ops.
    op.walk([&](Operation *nested) {
      if (isa<memref::LoadOp, memref::StoreOp>(nested))
        hasScalar = true;
    });
  }
  return hasCarrier && !hasScalar;
}

/// Check carrier-authoritative eligibility purely from the carrier's
/// properties. In authoritative mode, the carrier IS the computation — no
/// scalar body to inspect. We check iterator types, output projections, and
/// disjointness directly from the linalg.generic.
static bool
isCarrierAuthoritativeCandidate(linalg::GenericOp carrier,
                                sde::SdeStructuredClassification cls) {
  if (carrier.getNumDpsInits() == 0 || carrier.getNumLoops() == 0)
    return false;

  auto iterTypes = carrier.getIteratorTypesArray();
  switch (cls) {
  case sde::SdeStructuredClassification::elementwise:
    // All iterators must be parallel, all output maps must be projected perms.
    if (!llvm::all_of(iterTypes, [](utils::IteratorType t) {
          return t == utils::IteratorType::parallel;
        }))
      return false;
    for (unsigned i = 0; i < carrier.getNumDpsInits(); ++i) {
      AffineMap map =
          carrier.getMatchingIndexingMap(carrier.getDpsInitOperand(i));
      if (!map.isProjectedPermutation())
        return false;
    }
    return true;

  case sde::SdeStructuredClassification::matmul:
    return carrier.getNumLoops() == 3 && carrier.getNumDpsInits() == 1;

  case sde::SdeStructuredClassification::reduction: {
    bool hasParallel = false, hasReduction = false;
    for (auto t : iterTypes) {
      if (t == utils::IteratorType::parallel)
        hasParallel = true;
      if (t == utils::IteratorType::reduction)
        hasReduction = true;
    }
    return hasParallel && hasReduction;
  }

  default:
    return false;
  }
}

static bool isTilingCandidate(sde::SdeSuIterateOp op, Block &body,
                              linalg::GenericOp tensorGeneric) {
  if (op.getChunkSize())
    return false;
  if (op->getParentOfType<scf::ForOp>())
    return false;
  if (op.getLowerBounds().empty())
    return false;
  if (!op.getStructuredClassificationAttr())
    return false;

  auto classification = *op.getStructuredClassification();

  // Carrier-authoritative path: body has carrier but no scalar ops.
  if (tensorGeneric && isCarrierAuthoritative(body))
    return isCarrierAuthoritativeCandidate(tensorGeneric, classification);

  // Dual-representation path: scalar body alongside carrier.
  unsigned numSuDims = op.getLowerBounds().size();
  switch (classification) {
  case sde::SdeStructuredClassification::stencil:
    return op.getReductionAccumulators().size() == 0 &&
           isStencilCandidate(op, body);
  case sde::SdeStructuredClassification::elementwise:
    if (!tensorGeneric || op.getReductionAccumulators().size() != 0)
      return false;
    return isElementwiseTensorCandidate(body, tensorGeneric, numSuDims);
  case sde::SdeStructuredClassification::matmul:
    if (op.getReductionAccumulators().size() != 0)
      return false;
    if (!tensorGeneric)
      return isDirectMemoryMatmulCandidate(op, body);
    return isMatmulTensorCandidate(body, tensorGeneric);
  case sde::SdeStructuredClassification::reduction:
    if (!tensorGeneric)
      return false;
    return isReductionTensorCandidate(body, tensorGeneric);
  default:
    return false;
  }
}

/// Tile a carrier-authoritative body. Instead of creating scf.for tile loops
/// and cloning scalar ops, construct extract_slice → tiled linalg.generic →
/// insert_slice at the su_iterate body level.
///
/// For each carrier operand, the slicing is driven by the operand's indexing
/// map: dimensions that reference a tiled parallel dim get sliced at
/// [tileBase][tileSize], while untiled (reduction/full-range) dimensions
/// keep their full range.
static bool tileCarrierAuthoritative(linalg::GenericOp carrier, Block &srcBody,
                                     OpBuilder &builder, Value tileBase,
                                     Value tileSize, unsigned tileDim,
                                     IRMapping &mapper) {
  Location loc = carrier.getLoc();

  // 1. Clone carrier prep ops (mu_memref_to_tensor, bufferization.to_tensor,
  //    tensor.empty, etc.) but NOT the carrier itself.
  for (Operation &op : srcBody.without_terminator()) {
    if (isa<linalg::GenericOp>(op))
      continue;
    builder.clone(op, mapper);
  }

  auto iterTypes = carrier.getIteratorTypesArray();
  auto indexingMaps = carrier.getIndexingMapsArray();

  // Build a map: carrier loop dim → (isTiled, tileBase, tileSize).
  // Only the tileDim is tiled; others keep full range.
  // For N-dim carriers where the su_iterate provides dim 0 and inner dims
  // are additional parallel dims, only dim tileDim is sliced here.

  // Helper: for a given operand, compute extract_slice offsets/sizes/strides
  // based on its indexing map.
  auto sliceOperand = [&](Value operand, AffineMap map) -> Value {
    Value mapped = mapper.lookupOrDefault(operand);
    auto tensorTy = cast<RankedTensorType>(mapped.getType());
    unsigned rank = tensorTy.getRank();

    SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

    // Initialize sizes to full static extents.
    for (unsigned d = 0; d < rank; ++d)
      sizes.push_back(builder.getIndexAttr(tensorTy.getDimSize(d)));

    // For each result dim of the map, check if it references the tiled
    // carrier dim. If so, slice that operand dim.
    for (unsigned r = 0; r < map.getNumResults(); ++r) {
      AffineExpr expr = map.getResult(r);
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        if (dimExpr.getPosition() == tileDim) {
          offsets[r] = tileBase;
          sizes[r] = tileSize;
        }
      }
    }

    // Check if any dim is actually dynamic (tiled).
    bool needsSlice = false;
    for (unsigned r = 0; r < rank; ++r) {
      if (auto val = dyn_cast<Value>(offsets[r])) {
        needsSlice = true;
        break;
      }
      if (auto val = dyn_cast<Value>(sizes[r])) {
        needsSlice = true;
        break;
      }
    }
    if (!needsSlice)
      return mapped;

    return tensor::ExtractSliceOp::create(builder, loc, mapped, offsets, sizes,
                                          strides);
  };

  // 2. Slice each carrier input.
  SmallVector<Value> slicedInputs;
  for (auto [idx, input] : llvm::enumerate(carrier.getDpsInputs())) {
    AffineMap map = indexingMaps[idx];
    Value sliced = sliceOperand(input, map);
    slicedInputs.push_back(sliced);
  }

  // 3. Slice each carrier output.
  unsigned inputCount = carrier.getNumDpsInputs();
  SmallVector<Value> slicedOutputs;
  SmallVector<Value> fullOutputs; // for insert_slice targets
  for (auto [idx, output] : llvm::enumerate(carrier.getDpsInits())) {
    AffineMap map = indexingMaps[inputCount + idx];
    Value mappedFull = mapper.lookupOrDefault(output);
    fullOutputs.push_back(mappedFull);
    Value sliced = sliceOperand(output, map);
    slicedOutputs.push_back(sliced);
  }

  // 4. Compute result types from sliced outputs.
  SmallVector<Type> resultTypes;
  for (Value out : slicedOutputs)
    resultTypes.push_back(out.getType());

  // 5. Create tiled linalg.generic with sliced operands.
  // The carrier body may reference values outside the carrier (e.g., su_iterate
  // block args). Start from the parent mapper so those external references are
  // remapped to the new body's arguments.
  auto tiledGeneric = linalg::GenericOp::create(
      builder, loc, resultTypes, slicedInputs, slicedOutputs, indexingMaps,
      iterTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        IRMapping bodyMapper(mapper);
        Block &carrierBody = carrier.getRegion().front();
        for (auto [oldArg, newArg] :
             llvm::zip(carrierBody.getArguments(), blockArgs))
          bodyMapper.map(oldArg, newArg);
        for (Operation &op : carrierBody.without_terminator())
          nestedBuilder.clone(op, bodyMapper);
        auto carrierYield = cast<linalg::YieldOp>(carrierBody.getTerminator());
        SmallVector<Value> yieldValues;
        for (Value v : carrierYield.getValues())
          yieldValues.push_back(bodyMapper.lookupOrDefault(v));
        linalg::YieldOp::create(nestedBuilder, nestedLoc, yieldValues);
      });

  // 6. Insert_slice each output result back into the full tensor.
  for (auto [idx, result] : llvm::enumerate(tiledGeneric.getResults())) {
    AffineMap map = indexingMaps[inputCount + idx];
    auto fullTensorTy = cast<RankedTensorType>(fullOutputs[idx].getType());
    unsigned rank = fullTensorTy.getRank();

    SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
    for (unsigned d = 0; d < rank; ++d)
      sizes.push_back(builder.getIndexAttr(fullTensorTy.getDimSize(d)));

    for (unsigned r = 0; r < map.getNumResults(); ++r) {
      AffineExpr expr = map.getResult(r);
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        if (dimExpr.getPosition() == tileDim) {
          offsets[r] = tileBase;
          sizes[r] = tileSize;
        }
      }
    }

    tensor::InsertSliceOp::create(builder, loc, result, fullOutputs[idx],
                                  offsets, sizes, strides);
  }

  return true;
}

static void cloneBodyIntoTileLoop(Block &srcBody, IRMapping &mapper,
                                  OpBuilder &builder) {
  bool authoritative = isCarrierAuthoritative(srcBody);
  for (Operation &op : srcBody.without_terminator()) {
    if (!authoritative && isCarrierOp(op))
      continue;
    builder.clone(op, mapper);
  }
}

static bool stripMineLoop(scf::ForOp loop, Value tileIterations) {
  if (!loop || !tileIterations || loop.getNumResults() != 0 ||
      !loop.getInitArgs().empty())
    return false;

  int64_t tileConstant = 0;
  if (ValueAnalysis::getConstantIndex(tileIterations, tileConstant) &&
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
  // can also tile the input DBs. Splitting columns across owner tasks duplicates
  // the k-sweep against coarse A/B inputs and regresses large GEMM.
  op.setPhysicalOwnerDimsAttr(
      buildI64ArrayAttr(op.getContext(), SmallVector<int64_t, 1>{0}));
  op.setPhysicalBlockShapeAttr(buildI64ArrayAttr(op.getContext(), blockShape));
  op.setLogicalWorkerSliceAttr(buildI64ArrayAttr(op.getContext(), blockShape));
  op.setIterationTopologyAttr(sde::SdeIterationTopologyAttr::get(
      op.getContext(), sde::SdeIterationTopology::owner_strip));
}

struct TilingPass : public arts::impl::TilingBase<TilingPass> {
  explicit TilingPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    if (!costModel)
      return;

    SmallVector<std::pair<sde::SdeSuIterateOp, linalg::GenericOp>> rewrites;
    getOperation().walk([&](sde::SdeSuIterateOp op) {
      Block *body = sde::getSuIterateComputeBlock(op);
      linalg::GenericOp tensorGeneric = findTensorCarrier(*body);
      if (!isTilingCandidate(op, *body, tensorGeneric))
        return;

      std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation());
      if (tripCount && *tripCount <= 1)
        return;
      rewrites.push_back({op, tensorGeneric});
    });

    for (auto [op, tensorGeneric] : rewrites) {
      PatternRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();
      unsigned numDims = op.getLowerBounds().size();
      bool directMatmul = false;
      std::optional<DirectMatmulTilePlan> directMatmulPlan;
      if (!tensorGeneric && op.getStructuredClassification() ==
                                sde::SdeStructuredClassification::matmul) {
        directMatmulPlan =
            buildDirectMatmulTilePlan(rewriter, loc, op, *costModel);
        if (!directMatmulPlan)
          continue;
        directMatmul = true;
      }

      // Determine which dims are parallel (should tile) vs reduction (skip).
      SmallVector<bool> parallelMask = getParallelDimMask(op, tensorGeneric);

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
          int64_t balancedTile = llvm::divideCeil(
              *tripCount,
              std::max<int64_t>(1, costModel->getLogicalWorkerCapacity()));
          int64_t tileCount = std::clamp(
              std::max<int64_t>(balancedTile,
                                costModel->getMinIterationsPerWorker()),
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
        // Try memref stores first; fall back to carrier output type.
        bool foundElem = false;
        op.getBody().walk([&](memref::StoreOp storeOp) {
          Type elemType = storeOp.getValueToStore().getType();
          if (elemType.isF32() || elemType.isInteger(32))
            elemSize = 4;
          foundElem = true;
          return WalkResult::interrupt();
        });
        if (!foundElem && tensorGeneric && tensorGeneric.getNumDpsInits() > 0) {
          Type outTy = tensorGeneric.getDpsInits()[0].getType();
          if (auto shapedTy = dyn_cast<ShapedType>(outTy)) {
            Type et = shapedTy.getElementType();
            if (et.isF32() || et.isInteger(32))
              elemSize = 4;
          }
        }
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
        if (ValueAnalysis::getConstantIndex(originalStep, constantStep) &&
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
          applyStencilTileGuardsToStaticPlan(op, tensorGeneric,
                                             *staticTileIterations, numDims,
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
          op.getStructuredClassificationAttr(), op.getPatternAttr(),
          op.getAccessMinOffsetsAttr(), op.getAccessMaxOffsetsAttr(),
          op.getOwnerDimsAttr(), op.getSpatialDimsAttr(),
          op.getWriteFootprintAttr(),
          op.getPhysicalOwnerDimsAttr(), op.getPhysicalBlockShapeAttr(),
          op.getLogicalWorkerSliceAttr(), op.getPhysicalHaloShapeAttr(),
          op.getIterationTopologyAttr(), op.getRepetitionStructureAttr(),
          op.getAsyncStrategyAttr(), op.getCpsGroupIdAttr(),
          op.getCpsStageIndexAttr(), op.getCpsStageCountAttr(),
          op.getDistributionKindAttr(),
          op.getInPlaceSafeAttr(), op.getInPlaceSharedStateAttr(),
          op.getVectorizeWidthAttr(), op.getUnrollFactorAttr(),
          op.getInterleaveCountAttr());
      newOp->setAttrs(sde::getRewrittenAttrs(op));
      if (physicalTilePlan)
        stampPhysicalTilePlan(newOp, *physicalTilePlan);

      Block &newBody = sde::ensureBlock(newOp.getBody());
      for (unsigned d = newBody.getNumArguments(); d < numDims; ++d)
        newBody.addArgument(rewriter.getIndexType(), loc);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newBody);

      Block &srcBody = op.getBody().front();
      Block *computeBody = sde::getSuIterateComputeBlock(op);

      // Carrier-authoritative path: tile the carrier directly via
      // extract_slice → tiled linalg.generic → insert_slice.
      // No scf.for tile loop — the carrier encodes the iteration.
      if (tensorGeneric && isCarrierAuthoritative(*computeBody)) {
        // For 1-D: tile dim 0 (the su_iterate dim).
        // Compute dynamic tile size: min(tileStep, ub - base).
        Value tileBase = newBody.getArgument(0);
        Value rem = arith::SubIOp::create(rewriter, loc, op.getUpperBounds()[0],
                                          tileBase);
        Value tileSizeVal =
            arith::MinUIOp::create(rewriter, loc, perDimTileIter[0], rem);
        // Map old body's block args → new body's block args so that any
        // ops cloned from the old body that reference iteration variables
        // will point to the new body's arguments (not the old, soon-erased
        // ones).
        IRMapping carrierMapper;
        for (unsigned d = 0; d < numDims; ++d)
          carrierMapper.map(srcBody.getArgument(d), newBody.getArgument(d));
        tileCarrierAuthoritative(tensorGeneric, *computeBody, rewriter,
                                 tileBase, tileSizeVal, /*tileDim=*/0,
                                 carrierMapper);
      } else {
        // Dual-representation path: build nested scf.for tile loops for
        // parallel dims only. Reduction dims pass through directly.
        IRMapping mapper;
        SmallVector<scf::ForOp, 4> tileLoops;
        for (unsigned d = 0; d < numDims; ++d) {
          Value tileBase = newBody.getArgument(d);
          if (!parallelMask[d]) {
            // Reduction dim: map directly, no tile loop.
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

        // Clone body into the innermost tile loop. In dual-representation
        // mode, skips carrier ops.
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

namespace mlir::arts::sde {

std::unique_ptr<Pass> createTilingPass(sde::SDECostModel *costModel) {
  return std::make_unique<TilingPass>(costModel);
}

} // namespace mlir::arts::sde
