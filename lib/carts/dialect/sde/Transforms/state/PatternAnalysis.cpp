///==========================================================================///
/// File: PatternAnalysis.cpp
///
/// Identify memref-backed SDE patterns and stamp approved SDE facts.
///==========================================================================///

#include "carts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "carts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "carts/dialect/sde/Transforms/Passes.h"
#include "carts/utils/ArrayAttrUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::carts::sde {
#define GEN_PASS_DEF_PATTERNANALYSIS
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::sde

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(semantic_contracts);

using namespace mlir;
using namespace mlir::carts;

namespace {

static bool isInsideParallelRegion(sde::SdeSuIterateOp op) {
  for (Operation *current = op->getParentOp(); current;
       current = current->getParentOp()) {
    auto cuRegion = dyn_cast<sde::SdeCuRegionOp>(current);
    if (!cuRegion || cuRegion.getKind() != sde::SdeCuKind::parallel)
      continue;
    return true;
  }
  return false;
}

static Value accessRoot(Value value) {
  if (!value)
    return {};
  if (isa<BaseMemRefType>(value.getType()))
    return ::mlir::carts::ValueAnalysis::stripMemrefViewOps(value);
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

static bool hasNonZeroHaloOnAllOwnerDims(
    const sde::StructuredNeighborhoodInfo &info) {
  if (info.ownerDims.empty() || info.minOffsets.size() != info.maxOffsets.size())
    return false;
  for (int64_t rawDim : info.ownerDims) {
    if (rawDim < 0 || static_cast<size_t>(rawDim) >= info.minOffsets.size())
      return false;
    unsigned dim = static_cast<unsigned>(rawDim);
    if (info.minOffsets[dim] == 0 && info.maxOffsets[dim] == 0)
      return false;
  }
  return true;
}

static bool hasSingleExternalWriteRoot(sde::SdeSuIterateOp op,
                                       const sde::StructuredLoopSummary &summary) {
  Value selectedRoot;
  for (const sde::MemrefAccessEntry &write : summary.writes) {
    Value root = accessRoot(write.memref);
    if (!root || sde::isDefinedInside(op.getOperation(), root))
      continue;
    if (!selectedRoot) {
      selectedRoot = root;
      continue;
    }
    if (selectedRoot != root)
      return false;
  }
  return selectedRoot != nullptr;
}

static scf::ForOp findSinglePromotableInnerFor(Block *computeBlock) {
  if (!computeBlock)
    return nullptr;

  scf::ForOp innerFor;
  bool seenInnerFor = false;
  bool sawUnsupportedSibling = false;
  for (Operation &op : computeBlock->without_terminator()) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (seenInnerFor) {
        sawUnsupportedSibling = true;
        break;
      }
      innerFor = forOp;
      seenInnerFor = true;
      continue;
    }

    if (seenInnerFor || !isMemoryEffectFree(&op)) {
      sawUnsupportedSibling = true;
      break;
    }
  }

  if (sawUnsupportedSibling)
    return nullptr;
  return innerFor;
}

static bool isSafeRank2OutOfPlaceStencilPromotion(
    sde::SdeSuIterateOp op, const sde::StructuredLoopSummary &summary,
    const sde::StructuredNeighborhoodInfo &neighborhood,
    scf::ForOp innerFor) {
  if (!op || !innerFor)
    return false;
  if (op.getLowerBounds().size() != 1 || op.getUpperBounds().size() != 1 ||
      op.getSteps().size() != 1)
    return false;
  if (op.getChunkSize() || op.getNumResults() != 0 ||
      !op.getReductionAccumulators().empty() || op.getReductionKindsAttr())
    return false;
  if (summary.classification != sde::SdeStructuredClassification::stencil)
    return false;
  if (summary.nest.ivs.size() != 2 || summary.iterTypes.size() != 2)
    return false;
  if (!llvm::all_of(summary.iterTypes, [](utils::IteratorType iteratorType) {
        return iteratorType == utils::IteratorType::parallel;
      }))
    return false;
  if (neighborhood.ownerDims.size() != 2 ||
      neighborhood.spatialDims.size() < 2 ||
      !hasNonZeroHaloOnAllOwnerDims(neighborhood))
    return false;
  if (!sde::findCompatibleOutputLayoutPlan(summary))
    return false;
  if (!hasSingleExternalWriteRoot(op, summary))
    return false;

  auto effects = sde::collectStructuredMemoryEffects(op.getBody());
  if (effects.hasUnknownEffects || effects.writes.empty() ||
      sde::hasInPlaceSelfRead(effects))
    return false;

  Block *computeBlock = sde::getSuIterateComputeBlock(op);
  if (!computeBlock)
    return false;
  if (auto cuRegion = dyn_cast_or_null<sde::SdeCuRegionOp>(
          computeBlock->getParentOp()))
    if (!cuRegion.getIterArgs().empty() || cuRegion.getNumResults() != 0)
      return false;

  if (!innerFor.getInitArgs().empty() || innerFor.getNumResults() != 0)
    return false;
  Value outerIv = op.getBody().front().getArgument(0);
  if (::mlir::carts::ValueAnalysis::dependsOn(innerFor.getLowerBound(), outerIv) ||
      ::mlir::carts::ValueAnalysis::dependsOn(innerFor.getUpperBound(), outerIv) ||
      ::mlir::carts::ValueAnalysis::dependsOn(innerFor.getStep(), outerIv))
    return false;

  bool nestedLoop = false;
  innerFor.getBody()->walk([&](scf::ForOp nested) {
    if (nested != innerFor)
      nestedLoop = true;
  });
  return !nestedLoop;
}

static void removeStaleShapePlanAttrs(sde::SdeSuIterateOp op) {
  op.removePhysicalOwnerDimsAttr();
  op.removePhysicalBlockShapeAttr();
  op.removeLogicalWorkerSliceAttr();
  op.removePhysicalHaloShapeAttr();
  op.removeIterationTopologyAttr();
  op.removeDistributionKindAttr();
}

static sde::SdeSuIterateOp promoteRank2OutOfPlaceStencilOwnerLoop(
    sde::SdeSuIterateOp op, scf::ForOp innerFor) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  SmallVector<Value, 2> lowerBounds{op.getLowerBounds().front(),
                                    innerFor.getLowerBound()};
  SmallVector<Value, 2> upperBounds{op.getUpperBounds().front(),
                                    innerFor.getUpperBound()};
  SmallVector<Value, 2> steps{op.getSteps().front(), innerFor.getStep()};

  auto newOp = sde::SdeSuIterateOp::create(
      builder, loc, /*resultTypes=*/TypeRange{}, ValueRange(lowerBounds),
      ValueRange(upperBounds), ValueRange(steps), op.getScheduleAttr(),
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
      op.getCpsStageCountAttr(),
      op.getDistributionKindAttr(), op.getInPlaceSafeAttr(),
      op.getInPlaceSharedStateAttr(), op.getVectorizeWidthAttr(),
      op.getUnrollFactorAttr(), op.getInterleaveCountAttr());
  newOp->setAttrs(sde::getRewrittenAttrs(op));
  removeStaleShapePlanAttrs(newOp);

  Block &newBody = sde::ensureBlock(newOp.getBody());
  while (newBody.getNumArguments() < 2)
    newBody.addArgument(builder.getIndexType(), loc);

  IRMapping mapping;
  mapping.map(op.getBody().front().getArgument(0), newBody.getArgument(0));
  mapping.map(innerFor.getInductionVar(), newBody.getArgument(1));

  Block *oldComputeBlock = sde::getSuIterateComputeBlock(op);
  auto oldCuRegion =
      dyn_cast_or_null<sde::SdeCuRegionOp>(oldComputeBlock->getParentOp());

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&newBody);
  Block *cloneBlock = &newBody;
  if (oldCuRegion) {
    auto newCuRegion = sde::SdeCuRegionOp::create(
        builder, loc, /*resultTypes=*/TypeRange{}, oldCuRegion.getKindAttr(),
        oldCuRegion.getNowaitAttr(), /*iterArgs=*/ValueRange{});
    cloneBlock = &sde::ensureBlock(newCuRegion.getBody());
    builder.setInsertionPointToStart(cloneBlock);
  }

  for (Operation &bodyOp : oldComputeBlock->without_terminator()) {
    if (&bodyOp == innerFor.getOperation())
      break;
    builder.clone(bodyOp, mapping);
  }
  for (Operation &bodyOp : innerFor.getBody()->without_terminator())
    builder.clone(bodyOp, mapping);
  sde::SdeYieldOp::create(builder, loc, ValueRange{});

  if (oldCuRegion) {
    builder.setInsertionPointAfter(cloneBlock->getParentOp());
    sde::SdeYieldOp::create(builder, loc, ValueRange{});
  }

  op.erase();
  return newOp;
}

static sde::SdeSuIterateOp tryPromoteRank2OutOfPlaceStencilOwnerLoop(
    sde::SdeSuIterateOp op, const sde::StructuredLoopSummary &summary,
    const sde::StructuredNeighborhoodInfo &neighborhood) {
  Block *computeBlock = sde::getSuIterateComputeBlock(op);
  scf::ForOp innerFor = findSinglePromotableInnerFor(computeBlock);
  if (!isSafeRank2OutOfPlaceStencilPromotion(op, summary, neighborhood,
                                             innerFor))
    return op;
  return promoteRank2OutOfPlaceStencilOwnerLoop(op, innerFor);
}

static sde::SdePattern
derivePattern(const sde::StructuredLoopSummary &summary,
                sde::SdeStructuredClassification classification,
                std::optional<sde::StructuredNeighborhoodInfo> neighborhood) {
  switch (classification) {
  case sde::SdeStructuredClassification::elementwise:
    return sde::SdePattern::uniform;
  case sde::SdeStructuredClassification::elementwise_pipeline:
    return sde::SdePattern::elementwise_pipeline;
  case sde::SdeStructuredClassification::matmul:
    return sde::SdePattern::matmul;
  case sde::SdeStructuredClassification::reduction:
    return sde::SdePattern::reduction;
  case sde::SdeStructuredClassification::stencil:
    break;
  }

  if (!neighborhood)
    return sde::SdePattern::stencil_tiling_nd;

  if (isWavefront2D(summary, *neighborhood))
    return sde::SdePattern::wavefront_2d;
  if (hasHigherOrderHalo(*neighborhood))
    return sde::SdePattern::higher_order_stencil;
  if (countHaloDims(*neighborhood) >= 3)
    return sde::SdePattern::cross_dim_stencil_3d;
  return sde::SdePattern::stencil_tiling_nd;
}

static void clearPartialReductionIntent(sde::SdeSuIterateOp op) {
  op->removeAttr(op.getPartialReductionAttrName());
  op->removeAttr(op.getPartialReductionDimsAttrName());
  op->removeAttr(op.getPartialReductionOwnerDimsAttrName());
}

static void stampPartialReductionIntent(
    sde::SdeSuIterateOp op, const sde::StructuredLoopSummary &summary,
    sde::SdeStructuredClassification classification) {
  clearPartialReductionIntent(op);

  if (classification != sde::SdeStructuredClassification::elementwise_pipeline)
    return;
  if (!sde::isOwnerLocalPipelineReduction(op))
    return;

  SmallVector<int64_t, 4> reductionDims;
  for (auto [dim, iteratorType] : llvm::enumerate(summary.iterTypes)) {
    if (iteratorType == utils::IteratorType::reduction)
      reductionDims.push_back(static_cast<int64_t>(dim));
  }
  if (reductionDims.empty())
    return;

  std::optional<sde::StructuredOutputLayoutPlan> outputPlan =
      sde::findCompatibleOutputLayoutPlan(summary);
  if (!outputPlan)
    return;

  SmallVector<int64_t, 4> ownerDims;
  for (auto [physicalDim, loopDim] :
       llvm::enumerate(outputPlan->physicalDimToLoopDim)) {
    if (loopDim < 0 || static_cast<size_t>(loopDim) >= summary.iterTypes.size())
      continue;
    if (summary.iterTypes[loopDim] == utils::IteratorType::parallel)
      ownerDims.push_back(static_cast<int64_t>(physicalDim));
  }
  if (ownerDims.empty())
    return;

  op.setPartialReductionAttr(UnitAttr::get(op.getContext()));
  op.setPartialReductionDimsAttr(
      buildI64ArrayAttr(op.getContext(), reductionDims));
  op.setPartialReductionOwnerDimsAttr(
      buildI64ArrayAttr(op.getContext(), ownerDims));
}

struct PatternAnalysisPass
    : public sde::impl::PatternAnalysisBase<PatternAnalysisPass> {
  using PatternAnalysisBase::PatternAnalysisBase;

  void runOnOperation() override {
    getOperation().walk([&](sde::SdeSuIterateOp op) {
      std::optional<sde::StructuredLoopSummary> summary =
          sde::analyzeStructuredLoop(op);
      if (!summary)
        return;

      op->removeAttr(op.getInPlaceSafeAttrName());
      op->removeAttr(op.getInPlaceSharedStateAttrName());
      clearPartialReductionIntent(op);

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
        // Loop strip-mining preserves the original contract even when the
        // tiled body looks reduction-shaped after block-local rewriting.
        classification = *existingClassification;
      }

      op.setStructuredClassificationAttr(
          sde::SdeStructuredClassificationAttr::get(&getContext(),
                                                    classification));
      stampPartialReductionIntent(op, *summary, classification);

      std::optional<sde::StructuredNeighborhoodInfo> neighborhoodSummary;
      if (classification == sde::SdeStructuredClassification::stencil) {
        neighborhoodSummary = sde::extractNeighborhoodSummary(*summary);
        if (!neighborhoodSummary) {
          if (hasExplicitStencilContract) {
            if (!op.getPatternAttr())
              op.setPatternAttr(sde::SdePatternAttr::get(
                  &getContext(), sde::SdePattern::stencil_tiling_nd));
            ARTS_DEBUG("preserved explicit SDE stencil pattern on su_iterate");
          }
          return;
        }

        if (!hasExplicitStencilContract) {
          sde::SdeSuIterateOp promoted =
              tryPromoteRank2OutOfPlaceStencilOwnerLoop(
                  op, *summary, *neighborhoodSummary);
          if (promoted != op) {
            op = promoted;
            summary = sde::analyzeStructuredLoop(op);
            if (!summary)
              return;
            classification = summary->classification;
            if (classification != sde::SdeStructuredClassification::stencil)
              return;
            op.setStructuredClassificationAttr(
                sde::SdeStructuredClassificationAttr::get(&getContext(),
                                                          classification));
            neighborhoodSummary = sde::extractNeighborhoodSummary(*summary);
            if (!neighborhoodSummary)
              return;
          }
        }

        op.setPatternAttr(sde::SdePatternAttr::get(
            &getContext(),
            derivePattern(*summary, classification, neighborhoodSummary)));

        if (hasExplicitStencilContract) {
          ARTS_DEBUG("preserved explicit SDE stencil pattern on su_iterate");
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
            isInsideParallelRegion(op))
          op.setInPlaceSharedStateAttr(UnitAttr::get(op.getContext()));

        ARTS_DEBUG("stamped generic SDE pattern facts on su_iterate");
        return;
      }

      op.setPatternAttr(sde::SdePatternAttr::get(
          &getContext(),
          derivePattern(*summary, classification, neighborhoodSummary)));

      auto memoryEffects = sde::collectStructuredMemoryEffects(op.getBody());
      if (!memoryEffects.hasUnknownEffects && memoryEffects.allWritesAreRead())
        op.setInPlaceSafeAttr(UnitAttr::get(op.getContext()));

      ARTS_DEBUG("refreshed SDE pattern classification on su_iterate");
    });
  }
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createPatternAnalysisPass() {
  return std::make_unique<PatternAnalysisPass>();
}

} // namespace mlir::carts::sde
