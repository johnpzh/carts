///==========================================================================///
/// File: StructuredSummaries.cpp
///
/// Stamp SDE-owned structured summaries before crossing into ARTS IR.
///==========================================================================///

#include "arts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "arts/dialect/sde/Transforms/Passes.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/ValueAnalysis.h"
#include "arts/utils/costs/SDECostModel.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::arts {
#define GEN_PASS_DEF_STRUCTUREDSUMMARIES
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(semantic_contracts);

using namespace mlir;
using namespace mlir::arts;

namespace {

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

        // Phase 15: In-place operation detection (stencil path)
        {
          llvm::DenseSet<Value> writtenBases;
          llvm::DenseSet<Value> readBases;

          op.getBody().walk([&](memref::StoreOp storeOp) {
            writtenBases.insert(
                ValueAnalysis::stripMemrefViewOps(storeOp.getMemref()));
          });
          op.getBody().walk([&](memref::LoadOp loadOp) {
            readBases.insert(
                ValueAnalysis::stripMemrefViewOps(loadOp.getMemref()));
          });

          bool inPlaceSafe = !writtenBases.empty();
          for (Value written : writtenBases) {
            if (!readBases.contains(written)) {
              inPlaceSafe = false;
              break;
            }
          }

          if (inPlaceSafe) {
            op->setAttr(AttrNames::Operation::InPlaceSafe,
                        UnitAttr::get(op.getContext()));
          }
        }

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

      // Phase 15: In-place operation detection
      // Check if any output memref is the same as an input memref
      // (read-modify-write pattern that can be done in-place)
      {
        llvm::DenseSet<Value> writtenBases;
        llvm::DenseSet<Value> readBases;

        op.getBody().walk([&](memref::StoreOp storeOp) {
          writtenBases.insert(
              ValueAnalysis::stripMemrefViewOps(storeOp.getMemref()));
        });
        op.getBody().walk([&](memref::LoadOp loadOp) {
          readBases.insert(
              ValueAnalysis::stripMemrefViewOps(loadOp.getMemref()));
        });

        // In-place safe if ALL written memrefs are also read (RMW pattern)
        // and no other memref is aliased with the output
        bool inPlaceSafe = !writtenBases.empty();
        for (Value written : writtenBases) {
          if (!readBases.contains(written)) {
            inPlaceSafe = false;
            break;
          }
        }

        if (inPlaceSafe) {
          op->setAttr(AttrNames::Operation::InPlaceSafe,
                      UnitAttr::get(op.getContext()));
        }
      }

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
