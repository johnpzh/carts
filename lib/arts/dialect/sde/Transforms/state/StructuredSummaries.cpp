///==========================================================================///
/// File: StructuredSummaries.cpp
///
/// Stamp SDE-owned structured summaries before crossing into ARTS IR.
///==========================================================================///

#include "arts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "arts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "arts/dialect/sde/Transforms/Passes.h"
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

static Value accessRoot(Value value) {
  if (!value)
    return {};
  if (isa<BaseMemRefType>(value.getType()))
    return ValueAnalysis::stripMemrefViewOps(value);
  return value;
}

static bool sameAccessRoot(Value lhs, Value rhs) {
  Value lhsRoot = accessRoot(lhs);
  Value rhsRoot = accessRoot(rhs);
  return lhsRoot && rhsRoot && lhsRoot == rhsRoot;
}

static bool hasSelfRead(const sde::StructuredLoopSummary &summary) {
  for (const sde::MemrefAccessEntry &write : summary.writes)
    for (const sde::MemrefAccessEntry &read : summary.reads)
      if (sameAccessRoot(write.memref, read.memref))
        return true;
  return false;
}

static unsigned countHaloDims(const sde::StructuredNeighborhoodInfo &info) {
  unsigned count = 0;
  for (auto [minOffset, maxOffset] :
       llvm::zip(info.minOffsets, info.maxOffsets))
    if (minOffset != 0 || maxOffset != 0)
      ++count;
  return count;
}

static bool hasHigherOrderHalo(const sde::StructuredNeighborhoodInfo &info) {
  for (auto [minOffset, maxOffset] :
       llvm::zip(info.minOffsets, info.maxOffsets))
    if (minOffset < -1 || maxOffset > 1)
      return true;
  return false;
}

static bool isWavefront2D(const sde::StructuredLoopSummary &summary,
                          const sde::StructuredNeighborhoodInfo &info) {
  if (info.ownerDims.size() != 2 || !hasSelfRead(summary))
    return false;

  SmallVector<bool, 4> sawNegative(summary.nest.ivs.size(), false);
  bool sawPositiveSelfReadOffset = false;
  for (const sde::MemrefAccessEntry &write : summary.writes) {
    for (const sde::MemrefAccessEntry &read : summary.reads) {
      if (!sameAccessRoot(write.memref, read.memref))
        continue;
      for (AffineExpr result : read.indexingMap.getResults()) {
        auto dimOffset = sde::extractDimOffset(result);
        if (!dimOffset || !dimOffset->dim)
          continue;
        unsigned dim = *dimOffset->dim;
        if (dim >= sawNegative.size())
          continue;
        if (dimOffset->offset < 0)
          sawNegative[dim] = true;
        if (dimOffset->offset > 0)
          sawPositiveSelfReadOffset = true;
      }
    }
  }

  if (sawPositiveSelfReadOffset)
    return false;

  unsigned negativeOwnerDims = 0;
  for (int64_t dim : info.ownerDims)
    if (dim >= 0 && static_cast<size_t>(dim) < sawNegative.size() &&
        sawNegative[dim])
      ++negativeOwnerDims;
  return negativeOwnerDims == 2;
}

static sde::SdeDepFamily
deriveDepFamily(const sde::StructuredLoopSummary &summary,
                sde::SdeStructuredClassification classification,
                std::optional<sde::StructuredNeighborhoodInfo> neighborhood) {
  switch (classification) {
  case sde::SdeStructuredClassification::elementwise:
    return sde::SdeDepFamily::uniform;
  case sde::SdeStructuredClassification::elementwise_pipeline:
    return sde::SdeDepFamily::elementwise_pipeline;
  case sde::SdeStructuredClassification::matmul:
    return sde::SdeDepFamily::matmul;
  case sde::SdeStructuredClassification::reduction:
    return sde::SdeDepFamily::reduction;
  case sde::SdeStructuredClassification::stencil:
    break;
  }

  if (!neighborhood)
    return sde::SdeDepFamily::stencil_tiling_nd;

  if (isWavefront2D(summary, *neighborhood))
    return sde::SdeDepFamily::wavefront_2d;
  if (hasHigherOrderHalo(*neighborhood))
    return sde::SdeDepFamily::higher_order_stencil;
  if (countHaloDims(*neighborhood) >= 3)
    return sde::SdeDepFamily::cross_dim_stencil_3d;
  return sde::SdeDepFamily::stencil_tiling_nd;
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

      op->removeAttr(op.getInPlaceSafeAttrName());
      op->removeAttr(op.getInPlaceSharedStateAttrName());

      sde::SdeStructuredClassification classification = summary->classification;
      bool hasExplicitStencilContract = false;
      if (auto existingClassification = op.getStructuredClassification();
          existingClassification &&
          *existingClassification ==
              sde::SdeStructuredClassification::stencil &&
          op.getAccessMinOffsetsAttr() && op.getAccessMaxOffsetsAttr() &&
          op.getOwnerDimsAttr() && op.getWriteFootprintAttr()) {
        // An explicit SDE stencil stamp carries semantic intent plus the
        // neighborhood contract. Do not let a later scalar-shape refresh
        // rediscover a different family and lose the authored stencil meaning.
        classification = *existingClassification;
        hasExplicitStencilContract = true;
      } else if (existingClassification &&
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

      std::optional<sde::StructuredNeighborhoodInfo> neighborhoodSummary;
      if (classification == sde::SdeStructuredClassification::stencil) {
        neighborhoodSummary = sde::extractNeighborhoodSummary(*summary);
        if (!neighborhoodSummary) {
          if (hasExplicitStencilContract) {
            if (!op.getDepFamilyAttr())
              op.setDepFamilyAttr(sde::SdeDepFamilyAttr::get(
                  &getContext(), sde::SdeDepFamily::stencil_tiling_nd));
            ARTS_DEBUG("preserved explicit SDE stencil summary on su_iterate");
          }
          return;
        }

        op.setDepFamilyAttr(sde::SdeDepFamilyAttr::get(
            &getContext(),
            deriveDepFamily(*summary, classification, neighborhoodSummary)));

        if (hasExplicitStencilContract) {
          ARTS_DEBUG("preserved explicit SDE stencil summary on su_iterate");
          return;
        }

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
          op.setInPlaceSharedStateAttr(UnitAttr::get(op.getContext()));

        ARTS_DEBUG("stamped generic SDE structured summary on su_iterate");
        return;
      }

      op.setDepFamilyAttr(sde::SdeDepFamilyAttr::get(
          &getContext(),
          deriveDepFamily(*summary, classification, neighborhoodSummary)));

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

        Type i64 = IntegerType::get(&getContext(), 64);
        op.setVectorizeWidthAttr(IntegerAttr::get(i64, vectorWidth));
        op.setUnrollFactorAttr(
            IntegerAttr::get(i64, 2));
        op.setInterleaveCountAttr(
            IntegerAttr::get(i64, 4));
      }

      auto memoryEffects = sde::collectStructuredMemoryEffects(op.getBody());
      if (!memoryEffects.hasUnknownEffects && memoryEffects.allWritesAreRead())
        op.setInPlaceSafeAttr(UnitAttr::get(op.getContext()));

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
