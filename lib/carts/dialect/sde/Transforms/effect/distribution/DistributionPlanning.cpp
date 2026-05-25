///==========================================================================///
/// File: DistributionPlanning.cpp
///
/// SDE distribution planning. This pass keeps distribution intent on the SDE
/// side of the boundary by wrapping eligible `sde.su_iterate` operations in
/// `sde.su_distribute`; it uses SDE pattern/effect facts plus abstract worker
/// capacity/locality. Concrete DB ownership, EDT placement, routes, and runtime
/// memory-model choices remain ARTS decisions.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::carts::sde {
#define GEN_PASS_DEF_DISTRIBUTIONPLANNING
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::sde

#include "carts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "carts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "carts/dialect/sde/Utils/IterationSizingUtils.h"
#include "carts/dialect/sde/Utils/SDECostModel.h"
#include "carts/utils/ArrayAttrUtils.h"
#include "carts/utils/LoopUtils.h"
#include "carts/utils/ValueAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::carts;

namespace {

struct PlannedDistribution {
  sde::SdeSuIterateOp op;
  sde::SdeDistributionKind kind = sde::SdeDistributionKind::blocked;
};

static int64_t saturatingMultiplyPositive(int64_t lhs, int64_t rhs);

static int64_t getInterLocalityTargetWorkers(sde::SDECostModel &costModel) {
  return saturatingMultiplyPositive(costModel.getLogicalWorkerCapacity(),
                                    costModel.getInterLocalityTaskWaves());
}

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

static bool buildOwnerDimPlan(sde::SdeSuIterateOp op,
                              const sde::StructuredOutputLayoutPlan &outputPlan,
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

  SmallVector<int64_t, 4> workerGrid = sde::factorStencilWorkersAcrossDims(
      std::max<int64_t>(1, workers), ownerExtents, haloShape);
  for (auto [idx, physicalDim] : llvm::enumerate(ownerPhysicalDims)) {
    physicalBlockShape[physicalDim] =
        sde::ceilDivPositive(outputPlan.shape[physicalDim], workerGrid[idx]);
  }

  return true;
}

static bool buildOwnerDimPlan(const sde::LoopIndexedOutputPlan &outputPlan,
                              int64_t workers,
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
      sde::factorWorkersAcrossDims(std::max<int64_t>(1, workers), ownerExtents);
  for (auto [idx, physicalDim] : llvm::enumerate(ownerPhysicalDims)) {
    physicalBlockShape[physicalDim] =
        sde::ceilDivPositive(outputPlan.shape[physicalDim], workerGrid[idx]);
  }

  return true;
}

static SmallVector<Value, 4>
collectAllSuLoopIndexValues(sde::SdeSuIterateOp op) {
  SmallVector<Value, 4> ownerIndexValues;
  if (!op || op.getBody().empty())
    return ownerIndexValues;
  auto ivs = op.getLoopInductionVars();
  if (!ivs)
    return ownerIndexValues;
  ownerIndexValues.append(ivs->begin(), ivs->end());
  return ownerIndexValues;
}

static bool allRootAccessesStayWithinOwnerTile(
    sde::SdeSuIterateOp op, Value root, ArrayRef<int64_t> ownerDims) {
  if (!op || !root || ownerDims.empty())
    return false;

  SmallVector<Value, 4> ownerIndexValues = collectAllSuLoopIndexValues(op);
  if (ownerIndexValues.empty())
    return false;

  bool sawRootAccess = false;
  auto checkIndices = [&](Value memref, OperandRange indices) {
    Value base = ::mlir::carts::ValueAnalysis::stripMemrefViewOps(memref);
    if (base != root)
      return WalkResult::advance();

    sawRootAccess = true;
    auto memRefType = dyn_cast<MemRefType>(base.getType());
    if (!memRefType || memRefType.getRank() == 0 || indices.empty())
      return WalkResult::interrupt();
    for (int64_t rawDim : ownerDims) {
      if (rawDim < 0 || static_cast<size_t>(rawDim) >= indices.size())
        return WalkResult::interrupt();
      if (!sde::isOwnerDependentIndex(indices[rawDim], ownerIndexValues))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };

  WalkResult result = op.getBody().walk([&](Operation *nested) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(nested)) {
      if (isa<MemRefType>(loadOp.getResult().getType()))
        return WalkResult::advance();
      return checkIndices(loadOp.getMemref(), loadOp.getIndices());
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(nested)) {
      if (isa<MemRefType>(storeOp.getValueToStore().getType()))
        return WalkResult::advance();
      return checkIndices(storeOp.getMemref(), storeOp.getIndices());
    }
    return WalkResult::advance();
  });

  return !result.wasInterrupted() && sawRootAccess;
}

static std::optional<sde::LoopIndexedOutputPlan>
findConsistentMultiOwnerOutputPlan(sde::SdeSuIterateOp op) {
  if (!op || op.getBody().empty())
    return std::nullopt;

  SmallVector<Value, 4> ownerIndexValues = collectAllSuLoopIndexValues(op);
  if (ownerIndexValues.size() < 2)
    return std::nullopt;

  Block *computeBlock = sde::getSuIterateComputeBlock(op);
  if (!computeBlock)
    return std::nullopt;

  bool rejected = false;
  std::optional<sde::LoopIndexedOutputPlan> selectedPlan;
  auto visitStore = [&](memref::StoreOp storeOp) {
    Value base =
        ::mlir::carts::ValueAnalysis::stripMemrefViewOps(storeOp.getMemref());
    if (sde::isDefinedInside(op.getOperation(), base))
      return WalkResult::advance();

    auto memRefType = dyn_cast<MemRefType>(base.getType());
    if (!memRefType || memRefType.getRank() == 0 ||
        storeOp.getIndices().empty()) {
      rejected = true;
      return WalkResult::interrupt();
    }

    SmallVector<int64_t, 4> ownerPhysicalDims =
        sde::collectExactOwnerIndexedPhysicalDims(storeOp.getIndices(),
                                                  ownerIndexValues);
    if (ownerPhysicalDims.size() < 2) {
      rejected = true;
      return WalkResult::interrupt();
    }

    SmallVector<int64_t, 4> shape;
    shape.reserve(memRefType.getRank());
    for (int64_t dim : memRefType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        rejected = true;
        return WalkResult::interrupt();
      }
      shape.push_back(dim);
    }

    sde::LoopIndexedOutputPlan candidate{base, std::move(shape),
                                         std::move(ownerPhysicalDims)};
    if (!selectedPlan) {
      selectedPlan = std::move(candidate);
      return WalkResult::advance();
    }

    if (candidate.root != selectedPlan->root ||
        candidate.shape != selectedPlan->shape ||
        candidate.ownerPhysicalDims != selectedPlan->ownerPhysicalDims) {
      rejected = true;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  };

  for (Operation &nested : computeBlock->without_terminator()) {
    if (nested.walk(visitStore).wasInterrupted())
      break;
  }

  if (rejected)
    return std::nullopt;
  return selectedPlan;
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

static void alignLateOwnerPlanToExistingStep(
    sde::SdeSuIterateOp op, ArrayRef<int64_t> ownerDims,
    MutableArrayRef<int64_t> physicalBlockShape) {
  if (ownerDims.size() != 1 || op.getSteps().empty())
    return;

  int64_t ownerPhysicalDim = ownerDims.front();
  if (ownerPhysicalDim < 0 ||
      static_cast<size_t>(ownerPhysicalDim) >= physicalBlockShape.size())
    return;

  std::optional<int64_t> ownerStep = getPositiveConstantIndex(op.getSteps()[0]);
  if (!ownerStep || *ownerStep <= physicalBlockShape[ownerPhysicalDim])
    return;

  // DistributionPlanning runs after SDE loop tiling. When it authors a physical
  // owner plan late, the DB block for the owner dimension must cover the
  // already-existing owner-loop step; otherwise a single EDT can index across
  // multiple physical DB blocks after acquiring only one dependency.
  physicalBlockShape[ownerPhysicalDim] = *ownerStep;
}

static void stampStencilPhysicalPlan(sde::SdeSuIterateOp op,
                                     sde::SDECostModel &costModel) {
  if ((op.getPhysicalOwnerDimsAttr() && op.getPhysicalBlockShapeAttr()) ||
      costModel.getLogicalWorkerCapacity() <= 1)
    return;
  auto classification = op.getStructuredClassification();
  bool isStencil = classification &&
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
      int64_t workers = std::max<int64_t>(
          1, getInterLocalityTargetWorkers(costModel));
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
      std::max<int64_t>(1, getInterLocalityTargetWorkers(costModel));
  SmallVector<int64_t, 4> ownerDims;
  SmallVector<int64_t, 4> physicalBlockShape;
  if (!buildOwnerDimPlan(*secondaryPlan, workers, ownerDims,
                         physicalBlockShape))
    return;
  alignLateOwnerPlanToExistingStep(op, ownerDims, physicalBlockShape);
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
    if (*classification == sde::SdeStructuredClassification::reduction &&
        !sde::isOwnerLocalPipelineReduction(op))
      return;
  }
  std::optional<sde::LoopIndexedOutputPlan> outputPlan =
      sde::findLoopIndexedOutputPlan(op);
  bool usedConsistentOwnerFallback = false;
  if (!outputPlan) {
    if (classification &&
        *classification != sde::SdeStructuredClassification::elementwise &&
        *classification !=
            sde::SdeStructuredClassification::elementwise_pipeline)
      return;
    outputPlan = sde::findConsistentLoopIndexedOutputPlanWithOwnerDims(op);
    if (outputPlan && outputPlan->ownerPhysicalDims.size() != 1)
      outputPlan.reset();
    usedConsistentOwnerFallback = outputPlan.has_value();
  }
  if (!outputPlan || outputPlan->shape.empty() ||
      outputPlan->ownerPhysicalDims.empty())
    return;

  if (classification &&
      (*classification == sde::SdeStructuredClassification::elementwise ||
       *classification ==
           sde::SdeStructuredClassification::elementwise_pipeline)) {
    std::optional<sde::LoopIndexedOutputPlan> multiOwnerPlan =
        findConsistentMultiOwnerOutputPlan(op);
    if (multiOwnerPlan && multiOwnerPlan->ownerPhysicalDims.size() == 2 &&
        op.getLowerBounds().size() >= 2 && op.getUpperBounds().size() >= 2 &&
        op.getSteps().size() >= 2) {
      auto effects = sde::collectStructuredMemoryEffects(op.getBody());
      if (!effects.hasUnknownEffects) {
        bool ownerLocal = true;
        for (Value written : effects.writes) {
          if (sde::isDefinedInside(op.getOperation(), written))
            continue;
          if (!effects.reads.contains(written))
            continue;
          if (!allRootAccessesStayWithinOwnerTile(
                  op, written, multiOwnerPlan->ownerPhysicalDims)) {
            ownerLocal = false;
            break;
          }
        }
        if (ownerLocal) {
          SmallVector<int64_t, 4> ownerDims;
          SmallVector<int64_t, 4> physicalBlockShape;
          int64_t workers =
              std::max<int64_t>(1, getInterLocalityTargetWorkers(costModel));
          if (buildOwnerDimPlan(*multiOwnerPlan, workers, ownerDims,
                                physicalBlockShape)) {
            applyPhysicalPlan(op, ownerDims, physicalBlockShape);
            return;
          }
        }
      }
    }
  }

  int64_t ownerPhysicalDim = outputPlan->ownerPhysicalDims.front();
  if (ownerPhysicalDim < 0 ||
      static_cast<size_t>(ownerPhysicalDim) >= outputPlan->shape.size() ||
      outputPlan->shape[ownerPhysicalDim] <= 0)
    return;
  if (!classification || usedConsistentOwnerFallback) {
    auto effects = sde::collectStructuredMemoryEffects(op.getBody());
    bool selfRead = effects.reads.contains(outputPlan->root);
    bool ownerLocalSelfRead =
        selfRead &&
        sde::allRootAccessesStayWithinOwnerSlice(
            op, outputPlan->root, outputPlan->ownerPhysicalDims);
    if (effects.hasUnknownEffects || (selfRead && !ownerLocalSelfRead))
      return;
  }

  int64_t workers =
      std::max<int64_t>(1, getInterLocalityTargetWorkers(costModel));
  if (outputPlan->shape[ownerPhysicalDim] >= workers * 4LL * 1024LL * 1024LL)
    workers *= 2;
  SmallVector<int64_t, 4> physicalBlockShape(outputPlan->shape);
  bool ownerLocalPipeline =
      classification &&
      *classification == sde::SdeStructuredClassification::elementwise_pipeline &&
      sde::isOwnerLocalPipelineReduction(op);
  int64_t minOwnerIterations =
      ownerLocalPipeline
          ? 1
          : std::max<int64_t>(1, costModel.getMinIterationsPerWorker());
  int64_t balancedOwnerIterations =
      sde::ceilDivPositive(outputPlan->shape[ownerPhysicalDim], workers);
  physicalBlockShape[ownerPhysicalDim] =
      std::clamp(std::max(balancedOwnerIterations, minOwnerIterations),
                 int64_t{1}, outputPlan->shape[ownerPhysicalDim]);
  alignLateOwnerPlanToExistingStep(op, outputPlan->ownerPhysicalDims,
                                   physicalBlockShape);

  if (!classification)
    op.setStructuredClassificationAttr(
        sde::SdeStructuredClassificationAttr::get(
            op.getContext(), sde::SdeStructuredClassification::elementwise));
  op.setPhysicalOwnerDimsAttr(
      buildI64ArrayAttr(op.getContext(), outputPlan->ownerPhysicalDims));
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
  if (!outputPlan || outputPlan->shape.size() < 2 ||
      outputPlan->shape[0] <= 0 || outputPlan->shape[1] <= 0)
    return;

  auto effects = sde::collectStructuredMemoryEffects(op.getBody());
  if (effects.hasUnknownEffects || !effects.writes.contains(outputPlan->root))
    return;
  if (!sde::hasDistinctExternalMatmulInputRoots(op))
    return;

  int64_t workers =
      std::max<int64_t>(1, getInterLocalityTargetWorkers(costModel));
  SmallVector<int64_t, 4> physicalBlockShape(outputPlan->shape);
  SmallVector<int64_t, 4> ownerExtents{outputPlan->shape[0],
                                       outputPlan->shape[1]};
  SmallVector<int64_t, 4> workerGrid =
      sde::factorWorkersAcrossDims(workers, ownerExtents);
  physicalBlockShape[0] =
      sde::ceilDivPositive(outputPlan->shape[0], workerGrid[0]);
  physicalBlockShape[1] =
      sde::ceilDivPositive(outputPlan->shape[1], workerGrid[1]);

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
  if (op.getLogicalWorkerSliceAttr() &&
      (op.getPhysicalOwnerDimsAttr() || op.getPhysicalBlockShapeAttr()))
    return;
  auto classification = op.getStructuredClassification();
  if (!classification ||
      *classification != sde::SdeStructuredClassification::reduction)
    return;
  std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation());
  if (!tripCount || *tripCount <= 0)
    return;

  int64_t workers = std::max<int64_t>(1, costModel.getLogicalWorkerCapacity());
  int64_t targetTasks = workers;
  if (op.getReductionAccumulators().empty())
    targetTasks = saturatingMultiplyPositive(
        workers, costModel.getOwnerLocalPipelineTargetTaskWaves());
  int64_t slice = std::max<int64_t>(
      costModel.getMinIterationsPerWorker(),
      sde::ceilDivPositive(*tripCount, targetTasks));
  slice = std::clamp<int64_t>(slice, 1, *tripCount);

  if (op.getReductionAccumulators().empty()) {
    std::optional<sde::LoopIndexedOutputPlan> outputPlan =
        sde::findLoopIndexedOutputPlan(op);
    if (outputPlan && !outputPlan->ownerPhysicalDims.empty() &&
        !outputPlan->shape.empty()) {
      SmallVector<int64_t, 4> physicalBlockShape(outputPlan->shape);
      if (auto logicalSlice = readI64ArrayAttr(op.getLogicalWorkerSliceAttr());
          logicalSlice && logicalSlice->size() == physicalBlockShape.size()) {
        physicalBlockShape.assign(logicalSlice->begin(), logicalSlice->end());
      } else {
        for (int64_t rawDim : outputPlan->ownerPhysicalDims) {
          if (rawDim < 0 ||
              static_cast<size_t>(rawDim) >= physicalBlockShape.size())
            return;
          physicalBlockShape[rawDim] = slice;
        }
      }
      for (int64_t rawDim : outputPlan->ownerPhysicalDims)
        if (rawDim < 0 ||
            static_cast<size_t>(rawDim) >= physicalBlockShape.size())
          return;
      op.setPhysicalOwnerDimsAttr(
          buildI64ArrayAttr(op.getContext(), outputPlan->ownerPhysicalDims));
      op.setPhysicalBlockShapeAttr(
          buildI64ArrayAttr(op.getContext(), physicalBlockShape));
      op.setLogicalWorkerSliceAttr(
          buildI64ArrayAttr(op.getContext(), physicalBlockShape));
      op.setIterationTopologyAttr(sde::SdeIterationTopologyAttr::get(
          op.getContext(), sde::SdeIterationTopology::owner_strip));
      return;
    }
  }

  op.setLogicalWorkerSliceAttr(
      buildI64ArrayAttr(op.getContext(), SmallVector<int64_t, 1>{slice}));
  op.setIterationTopologyAttr(sde::SdeIterationTopologyAttr::get(
      op.getContext(), sde::SdeIterationTopology::owner_strip));
}

static void stampInPlaceSharedStencilSerialSlice(sde::SdeSuIterateOp op,
                                                 sde::SDECostModel &costModel) {
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

  int64_t targetTasks =
      std::max<int64_t>(1, getInterLocalityTargetWorkers(costModel));
  int64_t slice = sde::ceilDivPositive(*tripCount, targetTasks);
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

  int64_t threshold =
      saturatingMultiplyPositive(costModel.getLogicalWorkerCapacity(),
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

  if (op.getNumResults() > 0 && classificationAttr.getValue() !=
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
