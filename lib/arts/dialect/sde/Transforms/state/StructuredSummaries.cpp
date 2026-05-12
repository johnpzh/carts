///==========================================================================///
/// File: StructuredSummaries.cpp
///
/// Stamp SDE-owned structured summaries before crossing into ARTS IR.
///==========================================================================///

#include "arts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "arts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "arts/dialect/sde/Transforms/Passes.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/costs/SDECostModel.h"

namespace mlir::arts {
#define GEN_PASS_DEF_STRUCTUREDSUMMARIES
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(semantic_contracts);

using namespace mlir;
using namespace mlir::arts;

namespace {

static bool hasLocalParallelScope(sde::SdeSuIterateOp op) {
  for (Operation *current = op->getParentOp(); current;
       current = current->getParentOp()) {
    auto cuRegion = dyn_cast<sde::SdeCuRegionOp>(current);
    if (!cuRegion || cuRegion.getKind() != sde::SdeCuKind::parallel)
      continue;
    auto scope = cuRegion.getConcurrencyScope();
    return !scope || *scope == sde::SdeConcurrencyScope::local;
  }
  return false;
}

struct StructuredSummariesPass
    : public arts::impl::StructuredSummariesBase<StructuredSummariesPass> {
  explicit StructuredSummariesPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    getOperation().walk([&](sde::SdeSuIterateOp op) {
      std::optional<sde::StructuredLoopSummary> summary =
          sde::analyzeStructuredLoop(op);
      if (!summary)
        return;

      op->removeAttr(AttrNames::Operation::InPlaceSafe);
      op->removeAttr(AttrNames::Operation::InPlaceSharedState);

      sde::SdeStructuredClassification classification = summary->classification;
      if (auto existingClassification = op.getStructuredClassification();
          existingClassification &&
          *existingClassification ==
              sde::SdeStructuredClassification::elementwise_pipeline &&
          classification == sde::SdeStructuredClassification::elementwise) {
        classification = sde::SdeStructuredClassification::elementwise_pipeline;
      } else if (auto existingClassification = op.getStructuredClassification();
                 existingClassification &&
                 classification ==
                     sde::SdeStructuredClassification::reduction &&
                 *existingClassification !=
                     sde::SdeStructuredClassification::reduction &&
                 op.getReductionAccumulators().empty()) {
        // Tensor strip-mining preserves the original contract even when the
        // scalarized tiled body looks reduction-shaped after carrier removal.
        classification = *existingClassification;
      }

      op.setStructuredClassificationAttr(
          sde::SdeStructuredClassificationAttr::get(&getContext(),
                                                    classification));

      if (classification == sde::SdeStructuredClassification::stencil) {
        std::optional<sde::StructuredNeighborhoodInfo> neighborhoodSummary =
            sde::extractNeighborhoodSummary(*summary);
        if (!neighborhoodSummary)
          return;

        op.setAccessMinOffsetsAttr(buildI64ArrayAttr(
            op.getContext(), neighborhoodSummary->minOffsets));
        op.setAccessMaxOffsetsAttr(buildI64ArrayAttr(
            op.getContext(), neighborhoodSummary->maxOffsets));
        op.setOwnerDimsAttr(
            buildI64ArrayAttr(op.getContext(), neighborhoodSummary->ownerDims));
        op.setSpatialDimsAttr(buildI64ArrayAttr(
            op.getContext(), neighborhoodSummary->spatialDims));
        op.setWriteFootprintAttr(buildI64ArrayAttr(
            op.getContext(), neighborhoodSummary->writeFootprint));

        auto memoryEffects = sde::collectStructuredMemoryEffects(op.getBody());
        if (!memoryEffects.hasUnknownEffects &&
            sde::hasInPlaceSelfRead(memoryEffects) &&
            hasLocalParallelScope(op))
          op->setAttr(AttrNames::Operation::InPlaceSharedState,
                      UnitAttr::get(op.getContext()));

        ARTS_DEBUG("stamped generic SDE structured summary on su_iterate");
        return;
      }

      if (classification == sde::SdeStructuredClassification::elementwise ||
          classification ==
              sde::SdeStructuredClassification::elementwise_pipeline ||
          classification == sde::SdeStructuredClassification::matmul) {
        int64_t baseWidth = costModel ? costModel->getVectorWidth() : 4;
        int64_t vectorWidth = baseWidth; // default for f64/i64
        // Try scalar stores first (dual-rep bodies).
        bool foundType = false;
        op.getBody().walk([&](memref::StoreOp storeOp) {
          Type elemType = storeOp.getValueToStore().getType();
          if (elemType.isF32() || elemType.isInteger(32))
            vectorWidth = baseWidth * 2;
          else if (elemType.isF64() || elemType.isInteger(64))
            vectorWidth = baseWidth;
          foundType = true;
          return WalkResult::interrupt();
        });
        // Carrier-authoritative fallback: walk linalg.generic DPS inits.
        if (!foundType) {
          op.getBody().walk([&](linalg::GenericOp generic) {
            for (Value init : generic.getDpsInits()) {
              if (auto ty = dyn_cast<RankedTensorType>(init.getType())) {
                Type elemType = ty.getElementType();
                if (elemType.isF32() || elemType.isInteger(32))
                  vectorWidth = baseWidth * 2;
                else if (elemType.isF64() || elemType.isInteger(64))
                  vectorWidth = baseWidth;
                foundType = true;
                break;
              }
            }
            return WalkResult::interrupt();
          });
        }

        op->setAttr(
            AttrNames::Operation::Sde::VectorizeWidth,
            IntegerAttr::get(IndexType::get(&getContext()), vectorWidth));
        op->setAttr(AttrNames::Operation::Sde::UnrollFactor,
                    IntegerAttr::get(IndexType::get(&getContext()), 2));
        op->setAttr(AttrNames::Operation::Sde::InterleaveCount,
                    IntegerAttr::get(IndexType::get(&getContext()), 4));
      }

      auto memoryEffects = sde::collectStructuredMemoryEffects(op.getBody());
      if (!memoryEffects.hasUnknownEffects && memoryEffects.allWritesAreRead())
        op->setAttr(AttrNames::Operation::InPlaceSafe,
                    UnitAttr::get(op.getContext()));

      ARTS_DEBUG("refreshed SDE structured classification on su_iterate");
    });
  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::arts::sde {

std::unique_ptr<Pass>
createStructuredSummariesPass(sde::SDECostModel *costModel) {
  return std::make_unique<StructuredSummariesPass>(costModel);
}

} // namespace mlir::arts::sde
