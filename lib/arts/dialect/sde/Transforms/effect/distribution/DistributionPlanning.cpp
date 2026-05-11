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

#include "arts/utils/LoopUtils.h"
#include "arts/utils/costs/SDECostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::arts;

namespace {

struct PlannedDistribution {
  sde::SdeSuIterateOp op;
  sde::SdeDistributionKind kind = sde::SdeDistributionKind::blocked;
};

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

  auto classificationAttr = op.getStructuredClassificationAttr();
  if (!classificationAttr)
    return std::nullopt;

  auto scope = getEnclosingParallelScope(op);
  if (!scope)
    return std::nullopt;

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
