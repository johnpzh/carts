///==========================================================================///
/// File: DistributionPlanning.cpp
///
/// SDE distribution planning. This pass keeps distribution intent on the SDE
/// side of the boundary by wrapping eligible `sde.su_iterate` operations in
/// `sde.su_distribute`; it uses SDE pattern/effect facts and generic logical
/// worker capacity, not local/distributed machine scope.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::carts::sde {
#define GEN_PASS_DEF_DISTRIBUTIONPLANNING
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::sde

#include "carts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "carts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "carts/utils/LoopUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/StencilAttributes.h"
#include "carts/utils/ValueAnalysis.h"
#include "carts/dialect/sde/Utils/SDECostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

using namespace mlir;
using namespace mlir::carts::arts;
using namespace mlir::carts;

namespace {

struct PlannedDistribution {
  sde::SdeSuIterateOp op;
  sde::SdeDistributionKind kind = sde::SdeDistributionKind::blocked;
};

static int64_t readStencilHaloForOwnerDim(sde::SdeSuIterateOp op,
                                          unsigned ownerDim) {
  auto minOffsets = readI64ArrayAttr(op.getAccessMinOffsetsAttr());
  auto maxOffsets = readI64ArrayAttr(op.getAccessMaxOffsetsAttr());
  auto ownerDims = readI64ArrayAttr(op.getOwnerDimsAttr());
  if (!minOffsets || !maxOffsets || !ownerDims)
    return 0;

  for (auto [idx, rawDim] : llvm::enumerate(*ownerDims)) {
    if (rawDim < 0 || static_cast<unsigned>(rawDim) != ownerDim)
      continue;
    if (idx >= minOffsets->size() || idx >= maxOffsets->size())
      return 0;
    return std::max<int64_t>(0,
                             std::max(-(*minOffsets)[idx], (*maxOffsets)[idx]));
  }
  return 0;
}

static bool isInPlaceSelfReadStencil(sde::SdeSuIterateOp op) {
  auto classification = op.getStructuredClassification();
  if (!classification ||
      *classification != sde::SdeStructuredClassification::stencil)
    return false;

  auto effects = sde::collectStructuredMemoryEffects(op.getBody());
  return !effects.hasUnknownEffects && sde::hasInPlaceSelfRead(effects);
}

static int64_t ceilDivPositive(int64_t value, int64_t divisor) {
  return llvm::divideCeil(std::max<int64_t>(1, value),
                          std::max<int64_t>(1, divisor));
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

static SmallVector<unsigned, 4>
chooseMappedSdeOwnerLoopDims(sde::SdeSuIterateOp op,
                             const sde::StructuredOutputLayoutPlan &plan) {
  unsigned sdeRank = op.getLowerBounds().size();
  SmallVector<unsigned, 4> mappedLoopDims;
  mappedLoopDims.reserve(sdeRank);
  for (unsigned loopDim = 0; loopDim < sdeRank; ++loopDim) {
    if (loopDim >= plan.loopDimToPhysicalDim.size())
      break;
    int64_t physicalDim = plan.loopDimToPhysicalDim[loopDim];
    if (physicalDim < 0 ||
        static_cast<size_t>(physicalDim) >= plan.shape.size())
      continue;
    mappedLoopDims.push_back(loopDim);
  }
  if (mappedLoopDims.empty())
    return {};

  SmallVector<unsigned, 4> haloLoopDims;
  for (unsigned loopDim : mappedLoopDims)
    if (readStencilHaloForOwnerDim(op, loopDim) > 0)
      haloLoopDims.push_back(loopDim);
  if (!haloLoopDims.empty())
    return haloLoopDims;

  return mappedLoopDims;
}

static bool buildOwnerDimPlan(
    sde::SdeSuIterateOp op, const sde::StructuredOutputLayoutPlan &outputPlan,
    ArrayRef<unsigned> ownerLoopDims, int64_t workers,
    SmallVectorImpl<int64_t> &ownerPhysicalDims,
    SmallVectorImpl<int64_t> &physicalBlockShape,
    SmallVectorImpl<int64_t> &haloShape) {
  if (ownerLoopDims.empty() || outputPlan.shape.empty())
    return false;

  physicalBlockShape.assign(outputPlan.shape.begin(), outputPlan.shape.end());

  SmallVector<int64_t, 4> ownerExtents;
  ownerPhysicalDims.clear();
  haloShape.clear();
  for (unsigned loopDim : ownerLoopDims) {
    if (loopDim >= outputPlan.loopDimToPhysicalDim.size())
      return false;
    int64_t physicalDim = outputPlan.loopDimToPhysicalDim[loopDim];
    if (physicalDim < 0 ||
        static_cast<size_t>(physicalDim) >= outputPlan.shape.size())
      return false;
    if (outputPlan.shape[physicalDim] <= 0)
      return false;

    ownerPhysicalDims.push_back(physicalDim);
    ownerExtents.push_back(outputPlan.shape[physicalDim]);
    haloShape.push_back(
        std::max<int64_t>(0, readStencilHaloForOwnerDim(op, loopDim)));
  }

  SmallVector<int64_t, 4> workerGrid =
      factorWorkersAcrossDims(std::max<int64_t>(1, workers), ownerExtents);
  for (auto [idx, physicalDim] : llvm::enumerate(ownerPhysicalDims)) {
    physicalBlockShape[physicalDim] =
        ceilDivPositive(outputPlan.shape[physicalDim], workerGrid[idx]);
  }

  return true;
}

static bool buildOwnerDimPlan(
    const sde::LoopIndexedOutputPlan &outputPlan, int64_t workers,
    SmallVectorImpl<int64_t> &ownerPhysicalDims,
    SmallVectorImpl<int64_t> &physicalBlockShape) {
  if (outputPlan.ownerPhysicalDims.empty() || outputPlan.shape.empty())
    return false;

  physicalBlockShape.assign(outputPlan.shape.begin(), outputPlan.shape.end());
  ownerPhysicalDims.assign(outputPlan.ownerPhysicalDims.begin(),
                           outputPlan.ownerPhysicalDims.end());

  SmallVector<int64_t, 4> ownerExtents;
  ownerExtents.reserve(ownerPhysicalDims.size());
  for (int64_t physicalDim : ownerPhysicalDims) {
    if (physicalDim < 0 ||
        static_cast<size_t>(physicalDim) >= outputPlan.shape.size())
      return false;
    if (outputPlan.shape[physicalDim] <= 0)
      return false;
    ownerExtents.push_back(outputPlan.shape[physicalDim]);
  }

  SmallVector<int64_t, 4> workerGrid =
      factorWorkersAcrossDims(std::max<int64_t>(1, workers), ownerExtents);
  for (auto [idx, physicalDim] : llvm::enumerate(ownerPhysicalDims)) {
    physicalBlockShape[physicalDim] =
        ceilDivPositive(outputPlan.shape[physicalDim], workerGrid[idx]);
  }

  return true;
}

static void applyPhysicalPlan(sde::SdeSuIterateOp op,
                              ArrayRef<int64_t> ownerDims,
                              ArrayRef<int64_t> physicalBlockShape,
                              ArrayRef<int64_t> haloShape = {}) {
  op.setPhysicalOwnerDimsAttr(buildI64ArrayAttr(op.getContext(), ownerDims));
  op.setPhysicalBlockShapeAttr(
      buildI64ArrayAttr(op.getContext(), physicalBlockShape));
  if (llvm::any_of(haloShape, [](int64_t halo) { return halo > 0; }))
    op.setPhysicalHaloShapeAttr(buildI64ArrayAttr(op.getContext(), haloShape));
  op.setLogicalWorkerSliceAttr(
      buildI64ArrayAttr(op.getContext(), physicalBlockShape));
  op.setIterationTopologyAttr(sde::SdeIterationTopologyAttr::get(
      op.getContext(), ownerDims.size() > 1
                           ? sde::SdeIterationTopology::owner_tile
                           : sde::SdeIterationTopology::owner_strip));
}

static void stampStencilPhysicalPlan(sde::SdeSuIterateOp op,
                                     sde::SDECostModel &costModel) {
  if ((op.getPhysicalOwnerDimsAttr() && op.getPhysicalBlockShapeAttr()) ||
      costModel.getLogicalWorkerCapacity() <= 1)
    return;
  auto classification = op.getStructuredClassification();
  bool isStencil =
      classification &&
      *classification == sde::SdeStructuredClassification::stencil;
  if (op.getInPlaceSafe() && !isStencil)
    return;
  if (classification && !isStencil)
    return;

  /// First-dimension unclassified loops are owned by the uniform/matmul
  /// planners below. This secondary owner-dim path handles imperfect local
  /// stencil/update nests whose owner IV indexes a later physical output
  /// dimension.
  if (!isStencil && sde::findLoopIndexedOutputPlan(op))
    return;

  if (isStencil) {
    std::optional<sde::StructuredOutputLayoutPlan> outputPlan =
        sde::findCompatibleOutputLayoutPlan(op);
    if (outputPlan) {
      SmallVector<unsigned, 4> ownerLoopDims =
          chooseMappedSdeOwnerLoopDims(op, *outputPlan);
      int64_t workers =
          std::max<int64_t>(1, costModel.getLogicalWorkerCapacity());
      // One-dimensional halo stencils form a dependency pipeline between
      // neighboring owner blocks. Planning modestly more owner blocks than
      // logical worker capacity gives later ARTS scheduling enough ready tasks
      // without flooding the runtime with tiny stencil slices.
      if (ownerLoopDims.size() == 1)
        workers *= 2;
      SmallVector<int64_t, 4> ownerDims;
      SmallVector<int64_t, 4> physicalBlockShape;
      SmallVector<int64_t, 4> haloShape;
      if (!buildOwnerDimPlan(op, *outputPlan, ownerLoopDims, workers, ownerDims,
                             physicalBlockShape, haloShape))
        return;

      if (isInPlaceSelfReadStencil(op)) {
        auto effects = sde::collectStructuredMemoryEffects(op.getBody());
        if (effects.hasUnknownEffects || effects.writes.empty())
          return;
        for (Value written : effects.writes) {
          if (sde::isDefinedInside(op.getOperation(), written))
            continue;
          if (!effects.reads.contains(written))
            continue;
          if (!sde::allRootAccessesStayWithinOwnerSlice(op, written, ownerDims))
            return;
        }
      }

      applyPhysicalPlan(op, ownerDims, physicalBlockShape, haloShape);
      return;
    }
  }

  std::optional<sde::LoopIndexedOutputPlan> secondaryPlan =
      sde::findConsistentLoopIndexedOutputPlanWithOwnerDims(op);
  if (!secondaryPlan || secondaryPlan->ownerPhysicalDims.size() != 1)
    return;

  auto effects = sde::collectStructuredMemoryEffects(op.getBody());
  if (effects.hasUnknownEffects || effects.writes.empty())
    return;
  for (Value written : effects.writes) {
    if (sde::isDefinedInside(op.getOperation(), written))
      continue;
    if (!effects.reads.contains(written))
      continue;
    if (!sde::allRootAccessesStayWithinOwnerSlice(
            op, written, secondaryPlan->ownerPhysicalDims))
      return;
  }

  int64_t workers =
      std::max<int64_t>(1, costModel.getLogicalWorkerCapacity());
  SmallVector<int64_t, 4> ownerDims;
  SmallVector<int64_t, 4> physicalBlockShape;
  if (!buildOwnerDimPlan(*secondaryPlan, workers, ownerDims,
                         physicalBlockShape))
    return;
  applyPhysicalPlan(op, ownerDims, physicalBlockShape);
}

static void stampUniformPhysicalPlan(sde::SdeSuIterateOp op,
                                     sde::SDECostModel &costModel) {
  if ((op.getPhysicalOwnerDimsAttr() && op.getPhysicalBlockShapeAttr()) ||
      costModel.getLogicalWorkerCapacity() <= 1)
    return;
  auto classification = op.getStructuredClassification();
  if (classification) {
    if (*classification == sde::SdeStructuredClassification::stencil ||
        *classification == sde::SdeStructuredClassification::matmul)
      return;
  }
  std::optional<sde::LoopIndexedOutputPlan> outputPlan =
      sde::findLoopIndexedOutputPlan(op);
  if (!outputPlan || outputPlan->shape.empty() || outputPlan->shape.front() <= 0)
    return;
  if (!classification) {
    auto effects = sde::collectStructuredMemoryEffects(op.getBody());
    bool selfRead = effects.reads.contains(outputPlan->root);
    bool ownerLocalSelfRead =
        selfRead &&
        sde::allRootAccessesStayWithinOwnerSlice(op, outputPlan->root);
    if (effects.hasUnknownEffects || (selfRead && !ownerLocalSelfRead))
      return;
  }

  int64_t workers =
      std::max<int64_t>(1, costModel.getLogicalWorkerCapacity());
  if (outputPlan->shape.front() >= workers * 4LL * 1024LL * 1024LL)
    workers *= 2;
  SmallVector<int64_t, 4> physicalBlockShape(outputPlan->shape);
  physicalBlockShape.front() =
      std::max<int64_t>(1,
                        llvm::divideCeil(outputPlan->shape.front(), workers));

  if (!classification)
    op.setStructuredClassificationAttr(sde::SdeStructuredClassificationAttr::get(
        op.getContext(), sde::SdeStructuredClassification::elementwise));
  op.setPhysicalOwnerDimsAttr(
      buildI64ArrayAttr(op.getContext(), SmallVector<int64_t, 1>{0}));
  op.setPhysicalBlockShapeAttr(
      buildI64ArrayAttr(op.getContext(), physicalBlockShape));
  op.setLogicalWorkerSliceAttr(
      buildI64ArrayAttr(op.getContext(), physicalBlockShape));
  op.setIterationTopologyAttr(sde::SdeIterationTopologyAttr::get(
      op.getContext(), sde::SdeIterationTopology::owner_strip));
}

static void stampMatmulPhysicalPlan(sde::SdeSuIterateOp op,
                                    sde::SDECostModel &costModel) {
  if ((op.getPhysicalOwnerDimsAttr() && op.getPhysicalBlockShapeAttr()) ||
      costModel.getLogicalWorkerCapacity() <= 1)
    return;
  auto classification = op.getStructuredClassification();
  if (!classification ||
      *classification != sde::SdeStructuredClassification::matmul)
    return;
  std::optional<sde::LoopIndexedOutputPlan> outputPlan =
      sde::findLoopIndexedOutputPlan(op);
  if (!outputPlan || outputPlan->shape.size() < 2 || outputPlan->shape[0] <= 0 ||
      outputPlan->shape[1] <= 0)
    return;

  auto effects = sde::collectStructuredMemoryEffects(op.getBody());
  if (effects.hasUnknownEffects || !effects.writes.contains(outputPlan->root))
    return;

  int64_t workers =
      std::max<int64_t>(1, costModel.getLogicalWorkerCapacity());
  SmallVector<int64_t, 4> physicalBlockShape(outputPlan->shape);
  SmallVector<int64_t, 4> ownerExtents{outputPlan->shape[0],
                                       outputPlan->shape[1]};
  SmallVector<int64_t, 4> workerGrid =
      factorWorkersAcrossDims(workers, ownerExtents);
  physicalBlockShape[0] =
      ceilDivPositive(outputPlan->shape[0], workerGrid[0]);
  physicalBlockShape[1] =
      ceilDivPositive(outputPlan->shape[1], workerGrid[1]);

  op.setPhysicalOwnerDimsAttr(
      buildI64ArrayAttr(op.getContext(), SmallVector<int64_t, 2>{0, 1}));
  op.setPhysicalBlockShapeAttr(
      buildI64ArrayAttr(op.getContext(), physicalBlockShape));
  op.setLogicalWorkerSliceAttr(
      buildI64ArrayAttr(op.getContext(), physicalBlockShape));
  op.setIterationTopologyAttr(sde::SdeIterationTopologyAttr::get(
      op.getContext(), sde::SdeIterationTopology::owner_tile_2d));
}

static void stampReductionTaskShapePlan(sde::SdeSuIterateOp op,
                                        sde::SDECostModel &costModel) {
  if (op.getLogicalWorkerSliceAttr() ||
      costModel.getLogicalWorkerCapacity() <= 1)
    return;
  auto classification = op.getStructuredClassification();
  if (!classification ||
      *classification != sde::SdeStructuredClassification::reduction)
    return;
  std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation());
  if (!tripCount || *tripCount <= 0)
    return;

  int64_t workers =
      std::max<int64_t>(1, costModel.getLogicalWorkerCapacity());
  int64_t slice = ceilDivPositive(*tripCount, workers);
  op.setLogicalWorkerSliceAttr(
      buildI64ArrayAttr(op.getContext(), SmallVector<int64_t, 1>{slice}));
  op.setIterationTopologyAttr(sde::SdeIterationTopologyAttr::get(
      op.getContext(), sde::SdeIterationTopology::owner_strip));
}

static void stampInPlaceSharedStencilSerialSlice(
    sde::SdeSuIterateOp op, sde::SDECostModel &costModel) {
  if (op.getLogicalWorkerSliceAttr() ||
      costModel.getLogicalWorkerCapacity() <= 1)
    return;
  auto classification = op.getStructuredClassification();
  if (!classification ||
      *classification != sde::SdeStructuredClassification::stencil)
    return;
  if (!op.getInPlaceSharedStateAttr())
    return;
  if (op.getLowerBounds().size() != 1 || op.getUpperBounds().size() != 1 ||
      op.getSteps().size() != 1)
    return;

  std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation());
  if (!tripCount || *tripCount <= 1)
    return;

  int64_t targetTasks = std::min<int64_t>(
      64, std::max<int64_t>(1, costModel.getLogicalWorkerCapacity()));
  int64_t slice = ceilDivPositive(*tripCount, targetTasks);
  if (slice <= 1)
    return;

  op.setPhysicalOwnerDimsAttr(
      buildI64ArrayAttr(op.getContext(), SmallVector<int64_t, 1>{0}));
  op.setPhysicalBlockShapeAttr(
      buildI64ArrayAttr(op.getContext(), SmallVector<int64_t, 1>{slice}));
  op.setLogicalWorkerSliceAttr(
      buildI64ArrayAttr(op.getContext(), SmallVector<int64_t, 1>{slice}));
  op.setIterationTopologyAttr(sde::SdeIterationTopologyAttr::get(
      op.getContext(), sde::SdeIterationTopology::owner_strip));
}

static int64_t saturatingMultiplyPositive(int64_t lhs, int64_t rhs) {
  lhs = std::max<int64_t>(1, lhs);
  rhs = std::max<int64_t>(1, rhs);
  if (lhs > std::numeric_limits<int64_t>::max() / rhs)
    return std::numeric_limits<int64_t>::max();
  return lhs * rhs;
}

static bool hasEnoughWorkForDistribution(sde::SdeSuIterateOp op,
                                         sde::SDECostModel &costModel) {
  std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation());
  if (!tripCount)
    return true;

  int64_t threshold = saturatingMultiplyPositive(
      costModel.getLogicalWorkerCapacity(),
      costModel.getMinIterationsPerWorker());
  return *tripCount >= threshold;
}

static std::optional<sde::SdeDistributionKind>
chooseDistributionKind(sde::SdeSuIterateOp op, sde::SDECostModel &costModel) {
  if (op->getParentOfType<sde::SdeSuDistributeOp>())
    return std::nullopt;
  if (costModel.getLogicalWorkerCapacity() <= 1)
    return std::nullopt;

  auto classificationAttr = op.getStructuredClassificationAttr();
  if (!classificationAttr) {
    std::optional<sde::LoopIndexedOutputPlan> outputPlan =
        sde::findLoopIndexedOutputPlan(op);
    if (!outputPlan)
      return std::nullopt;
    auto effects = sde::collectStructuredMemoryEffects(op.getBody());
    if (effects.hasUnknownEffects || effects.reads.contains(outputPlan->root))
      return std::nullopt;
    if (hasEnoughWorkForDistribution(op, costModel))
      return sde::SdeDistributionKind::blocked;
    return std::nullopt;
  }

  if (op.getNumResults() > 0 &&
      classificationAttr.getValue() !=
          sde::SdeStructuredClassification::reduction)
    return std::nullopt;

  switch (classificationAttr.getValue()) {
  case sde::SdeStructuredClassification::elementwise:
  case sde::SdeStructuredClassification::elementwise_pipeline:
    return sde::SdeDistributionKind::blocked;
  case sde::SdeStructuredClassification::stencil:
    if (isInPlaceSelfReadStencil(op))
      return std::nullopt;
    if (hasEnoughWorkForDistribution(op, costModel))
      return sde::SdeDistributionKind::owner_compute;
    return std::nullopt;
  case sde::SdeStructuredClassification::matmul:
    return sde::SdeDistributionKind::blocked;
  case sde::SdeStructuredClassification::reduction:
    if (op.getReductionStrategyAttr() || op.getReductionAccumulators().empty())
      return sde::SdeDistributionKind::blocked;
    return std::nullopt;
  }
  return std::nullopt;
}

struct DistributionPlanningPass
    : public sde::impl::DistributionPlanningBase<DistributionPlanningPass> {
  explicit DistributionPlanningPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    if (!costModel)
      return;

    SmallVector<PlannedDistribution> rewrites;
    getOperation().walk([&](sde::SdeSuIterateOp op) {
      stampStencilPhysicalPlan(op, *costModel);
      stampMatmulPhysicalPlan(op, *costModel);
      stampUniformPhysicalPlan(op, *costModel);
      stampReductionTaskShapePlan(op, *costModel);
      stampInPlaceSharedStencilSerialSlice(op, *costModel);
      if (auto kind = chooseDistributionKind(op, *costModel))
        rewrites.push_back({op, *kind});
    });

    for (PlannedDistribution rewrite : rewrites) {
      if (rewrite.op.getNumResults() > 0) {
        // su_iterate with iter_arg results: can't wrap in su_distribute
        // (NoTerminator op can't forward results). Stamp distribution kind
        // directly on the su_iterate; SDE-to-CODIR carries it forward and the
        // ARTS materialization boundary consumes the CODIR plan fact.
        rewrite.op.setDistributionKindAttr(
            sde::SdeDistributionKindAttr::get(&getContext(), rewrite.kind));
        continue;
      }

      IRRewriter rewriter(rewrite.op.getContext());
      rewriter.setInsertionPoint(rewrite.op);

      auto distributeOp = sde::SdeSuDistributeOp::create(
          rewriter, rewrite.op.getLoc(),
          sde::SdeDistributionKindAttr::get(&getContext(), rewrite.kind));
      Block &body = sde::ensureBlock(distributeOp.getBody());
      rewrite.op->moveBefore(&body, body.end());
    }
  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass>
createDistributionPlanningPass(sde::SDECostModel *costModel) {
  return std::make_unique<DistributionPlanningPass>(costModel);
}

} // namespace mlir::carts::sde
