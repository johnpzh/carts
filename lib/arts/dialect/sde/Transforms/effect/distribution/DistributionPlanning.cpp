///==========================================================================///
/// File: DistributionPlanning.cpp
///
/// Cost-model-backed SDE distribution planning. This pass keeps distribution
/// ownership on the SDE side of the boundary by wrapping eligible
/// `arts_sde.su_iterate` operations in `arts_sde.su_distribute`.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_DISTRIBUTIONPLANNING
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "arts/utils/LoopUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/PlanContract.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/ValueAnalysis.h"
#include "arts/utils/costs/SDECostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::arts;

namespace {

struct PlannedDistribution {
  sde::SdeSuIterateOp op;
  sde::SdeDistributionKind kind = sde::SdeDistributionKind::blocked;
};

static std::optional<SmallVector<int64_t, 4>>
findStaticMemrefShape(sde::SdeSuIterateOp op, unsigned minRank) {
  std::optional<SmallVector<int64_t, 4>> selectedShape;

  op.getBody().walk([&](Operation *nested) {
    Value memref;
    if (auto load = dyn_cast<memref::LoadOp>(nested))
      memref = load.getMemref();
    else if (auto store = dyn_cast<memref::StoreOp>(nested))
      memref = store.getMemref();
    else
      return WalkResult::advance();

    Value base = ValueAnalysis::stripMemrefViewOps(memref);
    auto memRefType = dyn_cast<MemRefType>(base.getType());
    if (!memRefType || memRefType.getRank() < static_cast<int64_t>(minRank))
      return WalkResult::advance();

    SmallVector<int64_t, 4> shape;
    shape.reserve(memRefType.getRank());
    for (int64_t dim : memRefType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return WalkResult::advance();
      shape.push_back(dim);
    }

    selectedShape = std::move(shape);
    return WalkResult::interrupt();
  });

  return selectedShape;
}

static int64_t readStencilHaloForOwnerDim(sde::SdeSuIterateOp op,
                                          unsigned ownerDim) {
  auto minOffsets = readI64ArrayAttr(op.getOperation(), "accessMinOffsets");
  auto maxOffsets = readI64ArrayAttr(op.getOperation(), "accessMaxOffsets");
  auto ownerDims = readI64ArrayAttr(op.getOperation(), "ownerDims");
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

static void stampStencilPhysicalPlan(sde::SdeSuIterateOp op,
                                     sde::SDECostModel &costModel) {
  if (costModel.getWorkerCount() <= 1)
    return;
  if (auto plannedWorkers = getWorkers(op.getOperation());
      plannedWorkers && *plannedWorkers <= 1)
    return;
  if (op->hasAttr(AttrNames::Operation::InPlaceSafe))
    return;
  auto classification = op.getStructuredClassification();
  if (!classification ||
      *classification != sde::SdeStructuredClassification::stencil)
    return;
  if (isInPlaceSelfReadStencil(op))
    return;

  unsigned spatialRank = 1;
  if (auto spatialDims = readI64ArrayAttr(op.getOperation(), "spatialDims"))
    spatialRank = std::max<unsigned>(1, spatialDims->size());
  else if (auto minOffsets =
               readI64ArrayAttr(op.getOperation(), "accessMinOffsets"))
    spatialRank = std::max<unsigned>(1, minOffsets->size());

  std::optional<SmallVector<int64_t, 4>> shape =
      findStaticMemrefShape(op, spatialRank);
  if (!shape || shape->empty() || shape->front() <= 0)
    return;

  int64_t workers = std::max<int64_t>(1, costModel.getWorkerCount());
  SmallVector<int64_t, 4> physicalBlockShape(*shape);
  physicalBlockShape.front() =
      std::max<int64_t>(1, llvm::divideCeil(shape->front(), workers));

  SmallVector<int64_t, 1> ownerDims{0};
  SmallVector<int64_t, 1> haloShape{
      std::max<int64_t>(0, readStencilHaloForOwnerDim(op, 0))};

  MLIRContext *ctx = op.getContext();
  op->setAttr(
      AttrNames::Operation::Plan::KernelFamily,
      StringAttr::get(ctx, kernelFamilyToString(KernelFamily::Stencil)));
  op->setAttr(AttrNames::Operation::Plan::OwnerDims,
              buildI64ArrayAttr(ctx, ownerDims));
  op->setAttr(AttrNames::Operation::Plan::PhysicalBlockShape,
              buildI64ArrayAttr(ctx, physicalBlockShape));
  if (haloShape.front() > 0)
    op->setAttr(AttrNames::Operation::Plan::HaloShape,
                buildI64ArrayAttr(ctx, haloShape));
  op->setAttr(AttrNames::Operation::Plan::IterationTopology,
              StringAttr::get(ctx, "owner_strip"));
}

static void stampUniformPhysicalPlan(sde::SdeSuIterateOp op,
                                     sde::SDECostModel &costModel) {
  if ((op->hasAttr(AttrNames::Operation::Plan::OwnerDims) &&
       op->hasAttr(AttrNames::Operation::Plan::PhysicalBlockShape)) ||
      costModel.getWorkerCount() <= 1)
    return;
  auto classification = op.getStructuredClassification();
  if (classification) {
    if (*classification == sde::SdeStructuredClassification::stencil ||
        *classification == sde::SdeStructuredClassification::matmul)
      return;
  }
  if (auto plannedWorkers = getWorkers(op.getOperation());
      plannedWorkers && *plannedWorkers <= 1)
    return;
  if (op->hasAttr(AttrNames::Operation::InPlaceSafe))
    return;

  std::optional<sde::LoopIndexedOutputPlan> outputPlan =
      sde::findLoopIndexedOutputPlan(op);
  if (!outputPlan || outputPlan->shape.empty() || outputPlan->shape.front() <= 0)
    return;
  if (!classification) {
    auto effects = sde::collectStructuredMemoryEffects(op.getBody());
    if (effects.hasUnknownEffects || effects.reads.contains(outputPlan->root))
      return;
  }

  int64_t workers = std::max<int64_t>(1, costModel.getWorkerCount());
  SmallVector<int64_t, 4> physicalBlockShape(outputPlan->shape);
  physicalBlockShape.front() =
      std::max<int64_t>(1,
                        llvm::divideCeil(outputPlan->shape.front(), workers));

  MLIRContext *ctx = op.getContext();
  op->setAttr(
      AttrNames::Operation::Plan::KernelFamily,
      StringAttr::get(ctx, kernelFamilyToString(KernelFamily::Uniform)));
  op->setAttr(AttrNames::Operation::Plan::OwnerDims,
              buildI64ArrayAttr(ctx, SmallVector<int64_t, 1>{0}));
  op->setAttr(AttrNames::Operation::Plan::PhysicalBlockShape,
              buildI64ArrayAttr(ctx, physicalBlockShape));
  op->setAttr(AttrNames::Operation::Plan::IterationTopology,
              StringAttr::get(ctx, "owner_strip"));
}

static std::optional<sde::SdeConcurrencyScope>
getEnclosingParallelScope(sde::SdeSuIterateOp op) {
  for (Operation *current = op->getParentOp(); current;
       current = current->getParentOp()) {
    auto cuRegion = dyn_cast<sde::SdeCuRegionOp>(current);
    if (!cuRegion || cuRegion.getKind() != sde::SdeCuKind::parallel)
      continue;
    if (auto scope = cuRegion.getConcurrencyScope())
      return *scope;
    return sde::SdeConcurrencyScope::local;
  }
  return std::nullopt;
}

static bool hasEnoughDistributedWork(sde::SdeSuIterateOp op,
                                     sde::SDECostModel &costModel) {
  std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation());
  if (!tripCount)
    return true;

  int64_t baseThreshold =
      std::max<int64_t>(1, costModel.getMinIterationsPerWorker()) *
      std::max<int64_t>(1, costModel.getNodeCount());

  // For stencils, account for halo exchange overhead.
  int64_t threshold = baseThreshold;
  auto classification = op.getStructuredClassification();
  if (classification &&
      *classification == sde::SdeStructuredClassification::stencil) {
    ArrayAttr minArr = op.getAccessMinOffsetsAttr();
    ArrayAttr maxArr = op.getAccessMaxOffsetsAttr();
    if (minArr && maxArr && minArr.size() == maxArr.size()) {
      int64_t haloVolume = 0;
      unsigned numDims = minArr.size();

      for (unsigned d = 0; d < numDims; ++d) {
        int64_t lo = cast<IntegerAttr>(minArr[d]).getInt();
        int64_t hi = cast<IntegerAttr>(maxArr[d]).getInt();
        int64_t haloWidth = hi - lo;
        haloVolume += haloWidth;
      }

      // Scale threshold by estimated halo cost relative to compute cost.
      double haloCost =
          haloVolume * costModel.getHaloExchangeCostPerByte() * 8; // assume f64
      double computeCost = costModel.getLocalDataAccessCost();
      if (computeCost > 0) {
        double costRatio = 1.0 + haloCost / computeCost;
        threshold = static_cast<int64_t>(baseThreshold * costRatio);
      }
    }
  }

  return *tripCount >= threshold;
}

static std::optional<sde::SdeDistributionKind>
chooseDistributionKind(sde::SdeSuIterateOp op, sde::SDECostModel &costModel) {
  if (op->getParentOfType<sde::SdeSuDistributeOp>())
    return std::nullopt;
  if (costModel.getWorkerCount() <= 1)
    return std::nullopt;

  auto scope = getEnclosingParallelScope(op);
  if (!scope)
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
    if (*scope == sde::SdeConcurrencyScope::local)
      return sde::SdeDistributionKind::blocked;
    if (*scope == sde::SdeConcurrencyScope::distributed &&
        hasEnoughDistributedWork(op, costModel))
      return sde::SdeDistributionKind::blocked;
    return std::nullopt;
  }

  switch (classificationAttr.getValue()) {
  case sde::SdeStructuredClassification::elementwise:
  case sde::SdeStructuredClassification::elementwise_pipeline:
    if (*scope == sde::SdeConcurrencyScope::local)
      return sde::SdeDistributionKind::blocked;
    if (*scope == sde::SdeConcurrencyScope::distributed &&
        hasEnoughDistributedWork(op, costModel))
      return sde::SdeDistributionKind::blocked;
    return std::nullopt;
  case sde::SdeStructuredClassification::stencil:
    if (isInPlaceSelfReadStencil(op))
      return std::nullopt;
    if (*scope == sde::SdeConcurrencyScope::distributed &&
        hasEnoughDistributedWork(op, costModel))
      return sde::SdeDistributionKind::owner_compute;
    return std::nullopt;
  case sde::SdeStructuredClassification::matmul:
    if (*scope == sde::SdeConcurrencyScope::local)
      return sde::SdeDistributionKind::blocked;
    if (*scope == sde::SdeConcurrencyScope::distributed &&
        hasEnoughDistributedWork(op, costModel))
      return sde::SdeDistributionKind::blocked;
    return std::nullopt;
  case sde::SdeStructuredClassification::reduction:
    if (*scope == sde::SdeConcurrencyScope::local &&
        op.getReductionStrategyAttr())
      return sde::SdeDistributionKind::blocked;
    if (*scope == sde::SdeConcurrencyScope::distributed &&
        op.getReductionStrategyAttr() &&
        hasEnoughDistributedWork(op, costModel))
      return sde::SdeDistributionKind::blocked;
    return std::nullopt;
  }
  return std::nullopt;
}

struct DistributionPlanningPass
    : public arts::impl::DistributionPlanningBase<DistributionPlanningPass> {
  explicit DistributionPlanningPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    if (!costModel)
      return;

    SmallVector<PlannedDistribution> rewrites;
    getOperation().walk([&](sde::SdeSuIterateOp op) {
      stampStencilPhysicalPlan(op, *costModel);
      stampUniformPhysicalPlan(op, *costModel);
      if (auto kind = chooseDistributionKind(op, *costModel))
        rewrites.push_back({op, *kind});
    });

    for (PlannedDistribution rewrite : rewrites) {
      if (rewrite.op.getNumResults() > 0) {
        // su_iterate with iter_arg results: can't wrap in su_distribute
        // (NoTerminator op can't forward results). Stamp distribution kind
        // directly on the su_iterate — SuIterateToArtsPattern reads it.
        rewrite.op->setAttr(
            "distributionKind",
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

namespace mlir::arts::sde {

std::unique_ptr<Pass>
createDistributionPlanningPass(sde::SDECostModel *costModel) {
  return std::make_unique<DistributionPlanningPass>(costModel);
}

} // namespace mlir::arts::sde
