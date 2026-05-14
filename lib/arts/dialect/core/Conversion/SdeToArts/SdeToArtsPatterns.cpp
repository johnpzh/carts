///==========================================================================///
/// File: SdeToArtsPatterns.cpp
///
/// Converts SDE dialect ops into ARTS dialect ops, producing the same IR
/// that ConvertOpenMPToArts currently generates.
///
/// Mapping:
///   sde.cu_region parallel  ->  inline generic Core work units
///   sde.cu_region single    ->  arts.edt <single, intranode>
///   sde.cu_task             ->  arts.edt <task, intranode>
///   sde.su_iterate          ->  scf.for dispatch lanes + arts.edt <task>
///   sde.su_barrier          ->  arts.barrier
///   sde.cu_atomic           ->  arts.atomic_add
///   sde.resource_query      ->  arts.runtime_query
///   sde.mu_dep              ->  transient arts.db_control fallback
///   sde.mu_token + codelet  ->  arts.db_alloc/acquire + arts.edt directly
///   sde.yield               ->  arts.yield
///==========================================================================///

#include "arts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_CONVERTSDETOARTS
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts
#include "arts/passes/Passes.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/LoweringContractUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/Utils.h"
#include "arts/utils/ValueAnalysis.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(convert_sde_to_arts);

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
static llvm::Statistic numCuRegionConverted{
    "convert_sde_to_arts", "NumCuRegionConverted",
    "Number of sde.cu_region converted to arts.edt"};
static llvm::Statistic numSuIterateConverted{
    "convert_sde_to_arts", "NumSuIterateConverted",
    "Number of sde.su_iterate converted to generic task dispatch"};
static llvm::Statistic numCuTaskConverted{
    "convert_sde_to_arts", "NumCuTaskConverted",
    "Number of sde.cu_task converted to arts.edt"};

using namespace mlir;
using namespace mlir::arts;

static std::optional<ArtsDepPattern>
mapStructuredClassificationToArtsDepPattern(
    sde::SdeStructuredClassification classification) {
  switch (classification) {
  case sde::SdeStructuredClassification::stencil:
    return ArtsDepPattern::stencil_tiling_nd;
  case sde::SdeStructuredClassification::matmul:
    return ArtsDepPattern::matmul;
  case sde::SdeStructuredClassification::elementwise:
    return ArtsDepPattern::uniform;
  case sde::SdeStructuredClassification::elementwise_pipeline:
    return ArtsDepPattern::elementwise_pipeline;
  case sde::SdeStructuredClassification::reduction:
    break;
  }
  return std::nullopt;
}

static ArtsDepPattern mapSdePatternToArtsDepPattern(
    sde::SdePattern pattern) {
  switch (pattern) {
  case sde::SdePattern::uniform:
    return ArtsDepPattern::uniform;
  case sde::SdePattern::stencil_tiling_nd:
    return ArtsDepPattern::stencil_tiling_nd;
  case sde::SdePattern::cross_dim_stencil_3d:
    return ArtsDepPattern::cross_dim_stencil_3d;
  case sde::SdePattern::higher_order_stencil:
    return ArtsDepPattern::higher_order_stencil;
  case sde::SdePattern::wavefront_2d:
    return ArtsDepPattern::wavefront_2d;
  case sde::SdePattern::jacobi_alternating_buffers:
    return ArtsDepPattern::jacobi_alternating_buffers;
  case sde::SdePattern::matmul:
    return ArtsDepPattern::matmul;
  case sde::SdePattern::elementwise_pipeline:
    return ArtsDepPattern::elementwise_pipeline;
  case sde::SdePattern::reduction:
    return ArtsDepPattern::uniform;
  }
  return ArtsDepPattern::unknown;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Map SDE access mode to ARTS ArtsMode.
static ArtsMode convertAccessMode(sde::SdeAccessMode mode) {
  switch (mode) {
  case sde::SdeAccessMode::read:
    return ArtsMode::in;
  case sde::SdeAccessMode::write:
    return ArtsMode::out;
  case sde::SdeAccessMode::readwrite:
    return ArtsMode::inout;
  }
  return ArtsMode::inout;
}

static std::optional<EdtDistributionKind>
convertDistributionKind(sde::SdeDistributionKindAttr kind) {
  if (!kind)
    return std::nullopt;

  switch (kind.getValue()) {
  case sde::SdeDistributionKind::owner_compute:
  case sde::SdeDistributionKind::blocked:
    return EdtDistributionKind::block;
  case sde::SdeDistributionKind::cyclic:
    return EdtDistributionKind::block_cyclic;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// SDE contract classification helpers
//===----------------------------------------------------------------------===//

using StructuredNeighborhoodSummary = sde::StructuredNeighborhoodInfo;

static std::optional<StructuredNeighborhoodSummary>
getSdeNeighborhoodSummary(sde::SdeSuIterateOp op) {
  if (!op.getAccessMinOffsetsAttr() || !op.getAccessMaxOffsetsAttr() ||
      !op.getOwnerDimsAttr() || !op.getWriteFootprintAttr())
    return std::nullopt;

  auto readArrayAttr =
      [](ArrayAttr attr) -> std::optional<SmallVector<int64_t, 4>> {
    if (!attr)
      return std::nullopt;
    SmallVector<int64_t, 4> values;
    values.reserve(attr.size());
    for (Attribute element : attr) {
      auto intAttr = dyn_cast<IntegerAttr>(element);
      if (!intAttr)
        return std::nullopt;
      values.push_back(intAttr.getInt());
    }
    return values;
  };

  auto minOffsets = readArrayAttr(op.getAccessMinOffsetsAttr());
  auto maxOffsets = readArrayAttr(op.getAccessMaxOffsetsAttr());
  auto ownerDims = readArrayAttr(op.getOwnerDimsAttr());
  auto writeFootprint = readArrayAttr(op.getWriteFootprintAttr());
  if (!minOffsets || !maxOffsets || !ownerDims || !writeFootprint)
    return std::nullopt;

  StructuredNeighborhoodSummary info;
  info.minOffsets = std::move(*minOffsets);
  info.maxOffsets = std::move(*maxOffsets);
  info.ownerDims = std::move(*ownerDims);
  info.writeFootprint = std::move(*writeFootprint);
  if (auto spatialDims = readArrayAttr(op.getSpatialDimsAttr()))
    info.spatialDims = std::move(*spatialDims);
  else
    info.spatialDims = info.ownerDims;
  return info;
}

/// Stamp a simple (non-stencil) pattern contract on an op.
static void stampSimple(Operation *op, ArtsDepPattern family, int64_t rev) {
  if (!op)
    return;
  setDepPattern(op, family);
  setPatternRevision(op, rev);
  if (auto dist = getDistributionPatternForDepPattern(family))
    setEdtDistributionPattern(op, *dist);
}

static unsigned countTopLevelSchedulingUnits(Operation *op,
                                             Operation *sourceBeingRewritten) {
  if (!op || op == sourceBeingRewritten)
    return 0;
  if (isa<scf::ForOp, sde::SdeSuIterateOp>(op))
    return 1;

  auto distribute = dyn_cast<sde::SdeSuDistributeOp>(op);
  if (!distribute || distribute.getBody().empty())
    return 0;

  unsigned count = 0;
  for (Operation &nested : distribute.getBody().front().without_terminator())
    count += countTopLevelSchedulingUnits(&nested, sourceBeingRewritten);
  return count;
}

/// Stamp a stencil-family contract on an op with spatial metadata.
static void stampStencilOnOp(Operation *op, ArtsDepPattern family,
                             ArrayRef<int64_t> ownerDims,
                             ArrayRef<int64_t> minOffsets,
                             ArrayRef<int64_t> maxOffsets,
                             ArrayRef<int64_t> writeFootprint,
                             ArrayRef<int64_t> spatialDims,
                             ArrayRef<int64_t> blockShape, int64_t rev) {
  if (!op)
    return;
  setDepPattern(op, family);
  setPatternRevision(op, rev);
  setEdtDistributionPattern(op, EdtDistributionPattern::stencil);
  setSupportedBlockHalo(op);
  ArrayRef<int64_t> dims = spatialDims.empty() ? ownerDims : spatialDims;
  setStencilSpatialDims(op, dims);
  setStencilOwnerDims(op, ownerDims);
  setStencilMinOffsets(op, minOffsets);
  setStencilMaxOffsets(op, maxOffsets);
  setStencilWriteFootprint(op, writeFootprint);
  if (!blockShape.empty())
    setStencilBlockShape(op, blockShape);
}

static ArtsPlanIterationTopologyAttr
convertSdeIterationTopology(MLIRContext *ctx,
                            sde::SdeIterationTopology topology) {
  switch (topology) {
  case sde::SdeIterationTopology::owner_strip:
    return ArtsPlanIterationTopologyAttr::get(
        ctx, ArtsPlanIterationTopology::owner_strip);
  case sde::SdeIterationTopology::owner_tile:
    return ArtsPlanIterationTopologyAttr::get(
        ctx, ArtsPlanIterationTopology::owner_tile);
  case sde::SdeIterationTopology::owner_tile_2d:
    return ArtsPlanIterationTopologyAttr::get(
        ctx, ArtsPlanIterationTopology::owner_tile_2d);
  }
  return nullptr;
}

static ArtsPlanRepetitionStructureAttr
convertSdeRepetitionStructure(MLIRContext *ctx,
                              sde::SdeRepetitionStructure repetition) {
  switch (repetition) {
  case sde::SdeRepetitionStructure::none:
    return ArtsPlanRepetitionStructureAttr::get(
        ctx, ArtsPlanRepetitionStructure::none);
  case sde::SdeRepetitionStructure::pair_step:
    return ArtsPlanRepetitionStructureAttr::get(
        ctx, ArtsPlanRepetitionStructure::pair_step);
  case sde::SdeRepetitionStructure::k_step:
    return ArtsPlanRepetitionStructureAttr::get(
        ctx, ArtsPlanRepetitionStructure::k_step);
  case sde::SdeRepetitionStructure::full_timestep:
    return ArtsPlanRepetitionStructureAttr::get(
        ctx, ArtsPlanRepetitionStructure::full_timestep);
  }
  return nullptr;
}

static ArtsPlanAsyncStrategyAttr
convertSdeAsyncStrategy(MLIRContext *ctx, sde::SdeAsyncStrategy strategy) {
  if (strategy == sde::SdeAsyncStrategy::blocking)
    return ArtsPlanAsyncStrategyAttr::get(ctx, ArtsPlanAsyncStrategy::blocking);
  if (strategy == sde::SdeAsyncStrategy::advance_edt)
    return ArtsPlanAsyncStrategyAttr::get(ctx,
                                          ArtsPlanAsyncStrategy::advance_edt);
  return nullptr;
}

static ArtsBarrierReasonAttr
convertSdeBarrierReason(MLIRContext *ctx, sde::SdeBarrierReason reason) {
  switch (reason) {
  case sde::SdeBarrierReason::redundant:
    return ArtsBarrierReasonAttr::get(ctx, ArtsBarrierReason::redundant);
  case sde::SdeBarrierReason::required_memory:
    return ArtsBarrierReasonAttr::get(ctx, ArtsBarrierReason::required_memory);
  case sde::SdeBarrierReason::timestep_stage_boundary:
    return ArtsBarrierReasonAttr::get(
        ctx, ArtsBarrierReason::timestep_stage_boundary);
  case sde::SdeBarrierReason::wavefront_frontier:
    return ArtsBarrierReasonAttr::get(ctx, ArtsBarrierReason::wavefront_frontier);
  case sde::SdeBarrierReason::unknown_required:
    return ArtsBarrierReasonAttr::get(ctx, ArtsBarrierReason::unknown_required);
  }
  return nullptr;
}

static void stampDistributionKind(Operation *op, EdtDistributionKind kind,
                                  int64_t version = 1) {
  if (!op)
    return;

  setEdtDistributionKind(op, kind);
  setDistributionVersion(op, version);
}

static bool hasProjectedContractAttrs(Operation *op) {
  return getDepPattern(op) || getEdtDistributionKind(op) ||
         getEdtDistributionPattern(op) || hasStructuredPlanAttrs(op);
}

static void translateSdePhysicalPlanToCoreOp(sde::SdeSuIterateOp source,
                                             Operation *target) {
  if (!source || !target)
    return;

  MLIRContext *ctx = target->getContext();
  if (auto ownerDims = source.getPhysicalOwnerDimsAttr())
    setPlanOwnerDimsAttr(target, ownerDims);
  if (auto blockShape = source.getPhysicalBlockShapeAttr())
    setPlanPhysicalBlockShapeAttr(target, blockShape);
  if (auto workerSlice = source.getLogicalWorkerSliceAttr())
    setPlanLogicalWorkerSliceAttr(target, workerSlice);
  if (auto haloShape = source.getPhysicalHaloShapeAttr())
    setPlanHaloShapeAttr(target, haloShape);
  if (auto topology = source.getIterationTopologyAttr())
    setPlanIterationTopologyAttr(
        target, convertSdeIterationTopology(ctx, topology.getValue()));
  if (auto repetition = source.getRepetitionStructureAttr())
    setPlanRepetitionStructureAttr(
        target, convertSdeRepetitionStructure(ctx, repetition.getValue()));
  if (auto strategy = source.getAsyncStrategyAttr())
    if (auto converted = convertSdeAsyncStrategy(ctx, strategy.getValue()))
      setPlanAsyncStrategyAttr(target, converted);
}

static void translateSdeExecutionHintsToCoreEdt(sde::SdeSuIterateOp source,
                                                EdtOp target) {
  if (!source || !target)
    return;

  if (auto attr = source.getInPlaceSafeAttr())
    target.setInPlaceSafeAttr(attr);
  if (auto attr = source.getInPlaceSharedStateAttr())
    target.setInPlaceSharedStateAttr(attr);
  if (auto attr = source.getVectorizeWidthAttr())
    target.setVectorizeWidthAttr(attr);
  if (auto attr = source.getUnrollFactorAttr())
    target.setUnrollFactorAttr(attr);
  if (auto attr = source.getInterleaveCountAttr())
    target.setInterleaveCountAttr(attr);
}

static void stampSdeContractOnCoreOp(sde::SdeSuIterateOp source,
                                     Operation *target) {
  if (!source || !target)
    return;

  std::optional<ArtsDepPattern> selectedPattern;
  std::optional<StructuredNeighborhoodSummary> selectedStencilContract;
  int64_t selectedRevision = getPatternRevision(source.getOperation()).value_or(1);

  if (auto pattern = source.getPatternAttr()) {
    selectedPattern = mapSdePatternToArtsDepPattern(pattern.getValue());
    if (selectedPattern && isStencilFamilyDepPattern(*selectedPattern))
      selectedStencilContract = getSdeNeighborhoodSummary(source);
  } else if (auto classAttr = source.getStructuredClassificationAttr()) {
    selectedPattern =
        mapStructuredClassificationToArtsDepPattern(classAttr.getValue());
    if (selectedPattern && isStencilFamilyDepPattern(*selectedPattern))
      selectedStencilContract = getSdeNeighborhoodSummary(source);
  }

  if (selectedPattern) {
    if (selectedStencilContract && isStencilFamilyDepPattern(*selectedPattern)) {
      SmallVector<int64_t, 4> planOwnerDims;
      SmallVector<int64_t, 4> planBlockShape;
      if (auto ownerDims = readI64ArrayAttr(source.getPhysicalOwnerDimsAttr()))
        planOwnerDims.assign(ownerDims->begin(), ownerDims->end());
      if (auto blockShape =
              readI64ArrayAttr(source.getPhysicalBlockShapeAttr()))
        planBlockShape.assign(blockShape->begin(), blockShape->end());
      stampStencilOnOp(target, *selectedPattern, planOwnerDims.empty()
                                                    ? selectedStencilContract->ownerDims
                                                    : planOwnerDims,
                       selectedStencilContract->minOffsets,
                       selectedStencilContract->maxOffsets,
                       selectedStencilContract->writeFootprint,
                       selectedStencilContract->spatialDims, planBlockShape,
                       selectedRevision);
    } else {
      stampSimple(target, *selectedPattern, selectedRevision);
    }
  } else if (auto classAttr = source.getStructuredClassificationAttr();
             classAttr &&
             classAttr.getValue() ==
                 sde::SdeStructuredClassification::reduction) {
    setEdtDistributionPattern(target, EdtDistributionPattern::uniform);
  }

  if (auto distributionKind =
          convertDistributionKind(source.getDistributionKindAttr()))
    stampDistributionKind(target, *distributionKind);

  translateSdePhysicalPlanToCoreOp(source, target);
}

static void
collectTopLevelForContractCandidates(Operation *op,
                                     SmallVectorImpl<Operation *> &candidates) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (hasProjectedContractAttrs(forOp.getOperation()))
      candidates.push_back(forOp.getOperation());
    return;
  }

  auto distribute = dyn_cast<sde::SdeSuDistributeOp>(op);
  if (!distribute || distribute.getBody().empty())
    return;

  for (Operation &nested : distribute.getBody().front().without_terminator())
    collectTopLevelForContractCandidates(&nested, candidates);
}

static Value materializeGenericWorkerCount(OpBuilder &builder, Location loc) {
  Value runtimeWorkers =
      RuntimeQueryOp::create(builder, loc, RuntimeQueryKind::totalWorkers)
          .getResult();
  Value workers = ValueAnalysis::castToIndex(runtimeWorkers, builder, loc);
  return arith::MaxUIOp::create(builder, loc, workers,
                                createOneIndex(builder, loc));
}

static std::optional<ArtsMode>
getExternalRootAccessMode(sde::SdeSuIterateOp source, Value root) {
  if (!source || !root)
    return std::nullopt;

  sde::StructuredMemoryEffectSummary effects =
      sde::collectStructuredMemoryEffects(source.getBody());
  if (effects.hasUnknownEffects)
    return std::nullopt;

  bool reads = effects.reads.contains(root);
  bool writes = effects.writes.contains(root);
  if (!reads && !writes)
    return std::nullopt;
  if (reads && writes)
    return ArtsMode::inout;
  if (writes)
    return ArtsMode::out;
  return ArtsMode::in;
}

static SmallVector<Value> materializeSdeOwnerSliceTaskDependencies(
    sde::SdeSuIterateOp source, PatternRewriter &rewriter,
    const IRMapping &outerMapping, Value globalBase, Value iterCount,
    ArrayRef<Value> upperBounds, ArrayRef<Value> steps) {
  SmallVector<Value> deps;
  if (!source || !globalBase || !iterCount || upperBounds.empty() ||
      steps.empty())
    return deps;

  /// The generic SU materialization currently decomposes the first SDE
  /// iteration dimension into dispatch lanes. Only create explicit MU slice
  /// dependencies when the SDE physical plan also owns exactly that dimension.
  auto ownerDims = readI64ArrayAttr(source.getPhysicalOwnerDimsAttr());
  if (!ownerDims || ownerDims->size() != 1 || (*ownerDims)[0] != 0)
    return deps;

  sde::StructuredMemoryEffectSummary effects =
      sde::collectStructuredMemoryEffects(source.getBody());
  if (effects.hasUnknownEffects)
    return deps;

  SmallVector<Value, 4> writableRoots;
  for (Value root : effects.writes) {
    if (!root || sde::isDefinedInside(source.getOperation(), root))
      continue;
    if (!isa<MemRefType>(root.getType()))
      continue;
    writableRoots.push_back(root);
  }
  if (writableRoots.size() != 1)
    return deps;

  Value sourceRoot = writableRoots.front();
  std::optional<ArtsMode> mode = getExternalRootAccessMode(source, sourceRoot);
  if (!mode || !DbUtils::isWriterMode(*mode))
    return deps;

  Value root = outerMapping.lookupOrDefault(sourceRoot);
  if (!root || !isa<MemRefType>(root.getType()))
    return deps;

  Location loc = source.getLoc();
  Value rawSize = arith::MulIOp::create(rewriter, loc, iterCount, steps.front());
  Value remaining = arith::SubIOp::create(rewriter, loc, upperBounds.front(),
                                          globalBase);
  Value sliceSize = arith::MinUIOp::create(rewriter, loc, rawSize, remaining);

  /// This is the SDE-authored MU address-space slice for the task. Core will
  /// turn it into DB-space offsets after it creates the blocked DB layout.
  auto control = DbControlOp::create(
      rewriter, loc, *mode, root,
      /*indices=*/SmallVector<Value>{},
      /*offsets=*/SmallVector<Value>{globalBase},
      /*sizes=*/SmallVector<Value>{sliceSize});
  deps.push_back(control.getSubview());
  return deps;
}

struct SuReductionPlan {
  SmallVector<Value> sourceAccumulators;
  SmallVector<Value> accumulators;
  SmallVector<sde::SdeReductionKind> kinds;
  sde::SdeReductionStrategy strategy = sde::SdeReductionStrategy::local_accumulate;

  bool empty() const { return accumulators.empty(); }
};

struct MaterializedReduction {
  Value accumulator;
  sde::SdeReductionKind kind = sde::SdeReductionKind::custom;
  Type elementType;
  Value partialGuid;
  Value partialPtr;
};

static bool isScalarReductionCarrierType(MemRefType memrefType) {
  return memrefType && memrefType.getRank() <= 1;
}

static SmallVector<Value, 1>
getScalarReductionIndices(OpBuilder &builder, Location loc, Value memref) {
  SmallVector<Value, 1> indices;
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (memrefType && memrefType.getRank() == 1)
    indices.push_back(createZeroIndex(builder, loc));
  return indices;
}

static Value loadScalarReductionValue(OpBuilder &builder, Location loc,
                                      Value memref) {
  return memref::LoadOp::create(
      builder, loc, memref, getScalarReductionIndices(builder, loc, memref));
}

static void storeScalarReductionValue(OpBuilder &builder, Location loc,
                                      Value value, Value memref) {
  memref::StoreOp::create(builder, loc, value, memref,
                          getScalarReductionIndices(builder, loc, memref));
}

static Type getReductionElementType(Value accumulator) {
  if (auto memrefType = dyn_cast<MemRefType>(accumulator.getType()))
    return memrefType.getElementType();
  return accumulator.getType();
}

static Value createZeroLike(OpBuilder &builder, Location loc, Type elementType) {
  if (auto intType = dyn_cast<IntegerType>(elementType))
    return arith::ConstantIntOp::create(builder, loc, 0, intType.getWidth());
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return arith::ConstantFloatOp::create(
        builder, loc, floatType,
        APFloat::getZero(floatType.getFloatSemantics()));
  return {};
}

static Value createReductionIdentity(OpBuilder &builder, Location loc,
                                     Type elementType,
                                     sde::SdeReductionKind kind) {
  if (kind == sde::SdeReductionKind::add ||
      kind == sde::SdeReductionKind::lor ||
      kind == sde::SdeReductionKind::lxor)
    return createZeroLike(builder, loc, elementType);

  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    unsigned width = intType.getWidth();
    switch (kind) {
    case sde::SdeReductionKind::mul:
      return arith::ConstantIntOp::create(builder, loc, 1, width);
    case sde::SdeReductionKind::min:
      return arith::ConstantOp::create(
          builder, loc, elementType,
          IntegerAttr::get(elementType, APInt::getSignedMaxValue(width)));
    case sde::SdeReductionKind::max:
      return arith::ConstantOp::create(
          builder, loc, elementType,
          IntegerAttr::get(elementType, APInt::getSignedMinValue(width)));
    case sde::SdeReductionKind::land:
      return arith::ConstantOp::create(
          builder, loc, elementType,
          IntegerAttr::get(elementType, APInt::getAllOnes(width)));
    default:
      return createZeroLike(builder, loc, elementType);
    }
  }

  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
    switch (kind) {
    case sde::SdeReductionKind::mul:
      return arith::ConstantFloatOp::create(builder, loc, floatType,
                                            APFloat(semantics, 1));
    case sde::SdeReductionKind::min:
      return arith::ConstantFloatOp::create(
          builder, loc, floatType,
          APFloat::getInf(semantics, /*Negative=*/false));
    case sde::SdeReductionKind::max:
      return arith::ConstantFloatOp::create(
          builder, loc, floatType,
          APFloat::getInf(semantics, /*Negative=*/true));
    default:
      return createZeroLike(builder, loc, elementType);
    }
  }

  return {};
}

static Value createReductionCombiner(OpBuilder &builder, Location loc,
                                     Type elementType, Value lhs, Value rhs,
                                     sde::SdeReductionKind kind) {
  bool isFloat = isa<FloatType>(elementType);
  switch (kind) {
  case sde::SdeReductionKind::add:
    return isFloat ? arith::AddFOp::create(builder, loc, lhs, rhs).getResult()
                   : arith::AddIOp::create(builder, loc, lhs, rhs).getResult();
  case sde::SdeReductionKind::mul:
    return isFloat ? arith::MulFOp::create(builder, loc, lhs, rhs).getResult()
                   : arith::MulIOp::create(builder, loc, lhs, rhs).getResult();
  case sde::SdeReductionKind::min:
    return isFloat ? arith::MinimumFOp::create(builder, loc, lhs, rhs)
                         .getResult()
                   : arith::MinSIOp::create(builder, loc, lhs, rhs)
                         .getResult();
  case sde::SdeReductionKind::max:
    return isFloat ? arith::MaximumFOp::create(builder, loc, lhs, rhs)
                         .getResult()
                   : arith::MaxSIOp::create(builder, loc, lhs, rhs)
                         .getResult();
  case sde::SdeReductionKind::land:
    return arith::AndIOp::create(builder, loc, lhs, rhs).getResult();
  case sde::SdeReductionKind::lor:
    return arith::OrIOp::create(builder, loc, lhs, rhs).getResult();
  case sde::SdeReductionKind::lxor:
    return arith::XOrIOp::create(builder, loc, lhs, rhs).getResult();
  case sde::SdeReductionKind::custom:
    return {};
  }
  return {};
}

static LogicalResult validateReductionAccumulator(sde::SdeSuIterateOp source,
                                                  Value accumulator) {
  auto memrefType = dyn_cast<MemRefType>(accumulator.getType());
  if (!isScalarReductionCarrierType(memrefType))
    return source.emitOpError("requires scalar memref reduction accumulators "
                              "at the SDE/Core boundary");
  return success();
}

static FailureOr<SuReductionPlan>
getSuReductionPlan(sde::SdeSuIterateOp source,
                   const IRMapping &outerMapping) {
  SuReductionPlan plan;
  for (Value accumulator : source.getReductionAccumulators()) {
    plan.sourceAccumulators.push_back(accumulator);
    plan.accumulators.push_back(outerMapping.lookupOrDefault(accumulator));
  }

  if (plan.accumulators.empty())
    return plan;

  if (!source.getResults().empty()) {
    source.emitOpError("cannot directly materialize result-producing "
                       "su_iterate reductions; SDE must expose reductions as "
                       "explicit accumulator updates before Core");
    return failure();
  }

  auto kindsAttr = source.getReductionKindsAttr();
  if (!kindsAttr || kindsAttr.size() != plan.accumulators.size()) {
    source.emitOpError("requires one SDE reduction kind per accumulator");
    return failure();
  }

  auto strategyAttr = source.getReductionStrategyAttr();
  if (!strategyAttr) {
    source.emitOpError("requires an SDE reduction strategy before Core "
                       "materialization");
    return failure();
  }
  plan.strategy = strategyAttr.getValue();

  for (Attribute attr : kindsAttr) {
    auto kindAttr = dyn_cast<sde::SdeReductionKindAttr>(attr);
    if (!kindAttr || kindAttr.getValue() == sde::SdeReductionKind::custom) {
      source.emitOpError("requires concrete non-custom reduction kinds before "
                         "Core materialization");
      return failure();
    }
    plan.kinds.push_back(kindAttr.getValue());
  }

  for (auto [accumulator, kind] : llvm::zip(plan.accumulators, plan.kinds)) {
    if (failed(validateReductionAccumulator(source, accumulator)))
      return failure();
    Type elementType = getReductionElementType(accumulator);
    if (plan.strategy == sde::SdeReductionStrategy::atomic &&
        (kind != sde::SdeReductionKind::add || !isa<IntegerType>(elementType))) {
      source.emitOpError("atomic reduction strategy requires integer add "
                         "accumulators");
      return failure();
    }
  }

  return plan;
}

static void mapReductionAccumulatorAliases(IRMapping &mapper,
                                           Value sourceAccumulator,
                                           Value materializedAccumulator,
                                           Value replacement) {
  mapper.map(sourceAccumulator, replacement);
  mapper.map(materializedAccumulator, replacement);

  Value sourceRoot = ValueAnalysis::stripMemrefViewOps(sourceAccumulator);
  if (sourceRoot && sourceRoot != sourceAccumulator)
    mapper.map(sourceRoot, replacement);

  Value materializedRoot =
      ValueAnalysis::stripMemrefViewOps(materializedAccumulator);
  if (materializedRoot && materializedRoot != materializedAccumulator)
    mapper.map(materializedRoot, replacement);
}

static void preparePartialReductionSlots(
    OpBuilder &builder, Location loc, const SuReductionPlan &plan,
    EdtOp task, ArrayRef<BlockArgument> dependencyArgs, IRMapping &mapper) {
  Value zero = createZeroIndex(builder, loc);
  for (auto [idx, tuple] :
       llvm::enumerate(llvm::zip(plan.sourceAccumulators, plan.accumulators,
                                  plan.kinds, dependencyArgs))) {
    auto [sourceAccumulator, accumulator, kind, depArg] = tuple;
    (void)idx;
    Value slot =
        DbRefOp::create(builder, loc, depArg, SmallVector<Value>{zero});
    Value identity =
        createReductionIdentity(builder, loc, getReductionElementType(accumulator),
                                kind);
    if (identity)
      storeScalarReductionValue(builder, loc, identity, slot);
    mapReductionAccumulatorAliases(mapper, sourceAccumulator, accumulator, slot);
  }
}

static SmallVector<MaterializedReduction>
allocatePartialReductionDbs(OpBuilder &builder, Location loc,
                            const SuReductionPlan &plan, Value dispatchLanes,
                            Value one) {
  SmallVector<MaterializedReduction> materialized;
  materialized.reserve(plan.accumulators.size());
  if (plan.empty())
    return materialized;

  Value route = createCurrentNodeRoute(builder, loc);
  for (auto [accumulator, kind] : llvm::zip(plan.accumulators, plan.kinds)) {
    Type elementType = getReductionElementType(accumulator);
    auto alloc = DbAllocOp::create(
        builder, loc, ArtsMode::inout, route, DbAllocType::heap, DbMode::write,
        elementType, Value{}, SmallVector<Value>{dispatchLanes},
        SmallVector<Value>{one});
    setPartitionMode(alloc.getOperation(), PartitionMode::fine_grained);
    materialized.push_back(
        MaterializedReduction{accumulator, kind, elementType, alloc.getGuid(),
                              alloc.getPtr()});
  }

  return materialized;
}

static SmallVector<Value> createPartialReductionTaskDependencies(
    OpBuilder &builder, Location loc, ArrayRef<MaterializedReduction> reductions,
    Value lane) {
  SmallVector<Value> dependencies;
  dependencies.reserve(reductions.size());
  for (const MaterializedReduction &reduction : reductions) {
    auto acquire = DbAcquireOp::create(
        builder, loc, ArtsMode::inout, reduction.partialGuid,
        reduction.partialPtr, PartitionMode::fine_grained,
        /*indices=*/SmallVector<Value>{lane},
        /*offsets=*/SmallVector<Value>{},
        /*sizes=*/SmallVector<Value>{},
        /*partitionIndices=*/SmallVector<Value>{lane},
        /*partitionOffsets=*/SmallVector<Value>{},
        /*partitionSizes=*/SmallVector<Value>{},
        /*boundsValid=*/Value{},
        /*elementOffsets=*/SmallVector<Value>{},
        /*elementSizes=*/SmallVector<Value>{});
    acquire.setExplicitDepContract();
    dependencies.push_back(acquire.getPtr());
  }
  return dependencies;
}

static void combineReductionLinear(OpBuilder &builder, Location loc,
                                   const MaterializedReduction &reduction,
                                   Value zero, Value one,
                                   Value dispatchLanes) {
  Value identity =
      createReductionIdentity(builder, loc, reduction.elementType,
                              reduction.kind);
  auto loop = scf::ForOp::create(builder, loc, zero, dispatchLanes, one,
                                 ValueRange{identity});

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  Value workerIdx = loop.getInductionVar();
  Value accumulator = loop.getRegionIterArg(0);
  Value slot =
      DbRefOp::create(builder, loc, reduction.partialPtr,
                      SmallVector<Value>{workerIdx});
  Value partial = loadScalarReductionValue(builder, loc, slot);
  Value combined = createReductionCombiner(builder, loc, reduction.elementType,
                                           accumulator, partial,
                                           reduction.kind);
  scf::YieldOp::create(builder, loc, combined);

  builder.setInsertionPointAfter(loop);
  storeScalarReductionValue(builder, loc, loop.getResult(0),
                            reduction.accumulator);
}

static void combineReductionTree(OpBuilder &builder, Location loc,
                                 const MaterializedReduction &reduction,
                                 Value zero, Value one, Value dispatchLanes) {
  Value identity =
      createReductionIdentity(builder, loc, reduction.elementType,
                              reduction.kind);
  Value two = createConstantIndex(builder, loc, 2);
  auto loop = scf::ForOp::create(builder, loc, zero, dispatchLanes, two,
                                 ValueRange{identity});

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  Value pairIdx = loop.getInductionVar();
  Value accumulator = loop.getRegionIterArg(0);
  Value rhsIdx = arith::AddIOp::create(builder, loc, pairIdx, one);
  Value hasRhs = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                       rhsIdx, dispatchLanes);

  Value lhsSlot =
      DbRefOp::create(builder, loc, reduction.partialPtr,
                      SmallVector<Value>{pairIdx});
  Value lhs = loadScalarReductionValue(builder, loc, lhsSlot);
  auto rhsIf = scf::IfOp::create(builder, loc, TypeRange{reduction.elementType},
                                 hasRhs, /*withElseRegion=*/true);

  builder.setInsertionPointToStart(&rhsIf.getThenRegion().front());
  Value rhsSlot =
      DbRefOp::create(builder, loc, reduction.partialPtr,
                      SmallVector<Value>{rhsIdx});
  Value rhs = loadScalarReductionValue(builder, loc, rhsSlot);
  scf::YieldOp::create(builder, loc, rhs);

  builder.setInsertionPointToStart(&rhsIf.getElseRegion().front());
  scf::YieldOp::create(builder, loc, identity);

  builder.setInsertionPointToEnd(loop.getBody());
  Value pairValue = createReductionCombiner(builder, loc, reduction.elementType,
                                            lhs, rhsIf.getResult(0),
                                            reduction.kind);
  Value combined = createReductionCombiner(builder, loc, reduction.elementType,
                                           accumulator, pairValue,
                                           reduction.kind);
  scf::YieldOp::create(builder, loc, combined);

  builder.setInsertionPointAfter(loop);
  storeScalarReductionValue(builder, loc, loop.getResult(0),
                            reduction.accumulator);
}

static void combinePartialReductionsAfterEpoch(
    OpBuilder &builder, Location loc, const SuReductionPlan &plan,
    ArrayRef<MaterializedReduction> reductions, Value zero, Value one,
    Value dispatchLanes) {
  if (reductions.empty())
    return;

  for (const MaterializedReduction &reduction : reductions) {
    if (plan.strategy == sde::SdeReductionStrategy::tree)
      combineReductionTree(builder, loc, reduction, zero, one, dispatchLanes);
    else
      combineReductionLinear(builder, loc, reduction, zero, one, dispatchLanes);
  }

  for (const MaterializedReduction &reduction : reductions) {
    DbFreeOp::create(builder, loc, reduction.partialGuid);
    DbFreeOp::create(builder, loc, reduction.partialPtr);
  }
}

static LogicalResult cloneSuIterateBodyIntoLocalLoop(
    sde::SdeSuIterateOp source, scf::ForOp localLoop, Value globalBase,
    ArrayRef<Value> lowerBounds, ArrayRef<Value> upperBounds,
    ArrayRef<Value> steps, IRMapping outerMapping, PatternRewriter &rewriter);

static LogicalResult buildTrailingLoopNestAndClone(
    sde::SdeSuIterateOp source, unsigned dim, ArrayRef<Value> lowerBounds,
    ArrayRef<Value> upperBounds, ArrayRef<Value> steps, IRMapping &mapper,
    PatternRewriter &rewriter, Block *computeBlock) {
  Location loc = source.getLoc();
  if (dim == lowerBounds.size()) {
    for (Operation &srcOp : computeBlock->without_terminator())
      rewriter.clone(srcOp, mapper);
    return success();
  }

  auto loop = scf::ForOp::create(rewriter, loc, lowerBounds[dim],
                                 upperBounds[dim], steps[dim]);
  if (source.getBody().front().getNumArguments() > dim)
    mapper.map(source.getBody().front().getArgument(dim),
               loop.getInductionVar());

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(loop.getBody());
  return buildTrailingLoopNestAndClone(source, dim + 1, lowerBounds,
                                       upperBounds, steps, mapper, rewriter,
                                       computeBlock);
}

static LogicalResult cloneSuIterateBodyIntoLocalLoop(
    sde::SdeSuIterateOp source, scf::ForOp localLoop, Value globalBase,
    ArrayRef<Value> lowerBounds, ArrayRef<Value> upperBounds,
    ArrayRef<Value> steps, IRMapping outerMapping, PatternRewriter &rewriter) {
  Block *computeBlock = sde::getSuIterateComputeBlock(source);
  if (!computeBlock)
    return failure();

  Location loc = source.getLoc();
  IRMapping mapper = outerMapping;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(localLoop.getBody());

  Value localIter = localLoop.getInductionVar();
  Value scaled = arith::MulIOp::create(rewriter, loc, localIter, steps.front());
  Value globalIv = arith::AddIOp::create(rewriter, loc, globalBase, scaled);
  if (source.getBody().front().getNumArguments() > 0)
    mapper.map(source.getBody().front().getArgument(0), globalIv);

  return buildTrailingLoopNestAndClone(source, /*dim=*/1, lowerBounds,
                                       upperBounds, steps, mapper, rewriter,
                                       computeBlock);
}

static LogicalResult materializeSuIterateAsGenericTaskDispatch(
    sde::SdeSuIterateOp source, PatternRewriter &rewriter,
    const IRMapping &outerMapping = IRMapping()) {
  if (source.getLowerBounds().empty() || source.getUpperBounds().empty() ||
      source.getSteps().empty())
    return source.emitOpError("cannot materialize su_iterate without bounds");
  if (source.getLowerBounds().size() != source.getUpperBounds().size() ||
      source.getLowerBounds().size() != source.getSteps().size())
    return source.emitOpError("cannot materialize su_iterate with mismatched "
                              "bound ranks");

  Location loc = source.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  SmallVector<Value> lowerBounds, upperBounds, steps;
  for (Value value : source.getLowerBounds())
    lowerBounds.push_back(outerMapping.lookupOrDefault(value));
  for (Value value : source.getUpperBounds())
    upperBounds.push_back(outerMapping.lookupOrDefault(value));
  for (Value value : source.getSteps())
    steps.push_back(outerMapping.lookupOrDefault(value));

  FailureOr<SuReductionPlan> maybeReductionPlan =
      getSuReductionPlan(source, outerMapping);
  if (failed(maybeReductionPlan))
    return failure();
  SuReductionPlan reductionPlan = std::move(*maybeReductionPlan);
  if (reductionPlan.empty() && !source.getResults().empty())
    return source.emitOpError("cannot materialize result-producing su_iterate "
                              "without explicit SDE reduction accumulators");

  Value zero = createZeroIndex(rewriter, loc);
  Value one = createOneIndex(rewriter, loc);
  Value workers = materializeGenericWorkerCount(rewriter, loc);

  Value range = arith::SubIOp::create(rewriter, loc, upperBounds.front(),
                                      lowerBounds.front());
  Value totalIterations = createCeilDivUI(rewriter, loc, range, steps.front());
  Value rawBlockSize = createCeilDivUI(rewriter, loc, totalIterations, workers);
  Value blockSize =
      arith::MaxUIOp::create(rewriter, loc, rawBlockSize, one);
  Value totalChunks = createCeilDivUI(rewriter, loc, totalIterations, blockSize);
  Value dispatchLanes =
      arith::MinUIOp::create(rewriter, loc, workers, totalChunks);

  SmallVector<MaterializedReduction> partialReductions =
      allocatePartialReductionDbs(rewriter, loc, reductionPlan, dispatchLanes,
                                  one);

  auto epoch = EpochOp::create(rewriter, loc, ValueRange{});
  stampSdeContractOnCoreOp(source, epoch.getOperation());
  copyArtsMetadataAttrs(source, epoch);
  Region &epochRegion = epoch.getRegion();
  if (epochRegion.empty())
    epochRegion.push_back(new Block());
  Block *epochBlock = &epochRegion.front();

  OpBuilder::InsertionGuard epochGuard(rewriter);
  rewriter.setInsertionPointToStart(epochBlock);

  auto dispatchLoop = scf::ForOp::create(rewriter, loc, zero, dispatchLanes,
                                         one);
  rewriter.setInsertionPointToStart(dispatchLoop.getBody());
  Value lane = dispatchLoop.getInductionVar();
  Value chunkStart =
      arith::MulIOp::create(rewriter, loc, lane, blockSize);
  Value remaining =
      arith::SubIOp::create(rewriter, loc, totalIterations, chunkStart);
  Value iterCount =
      arith::MinUIOp::create(rewriter, loc, blockSize, remaining);

  Value globalBaseOffset =
      arith::MulIOp::create(rewriter, loc, chunkStart, steps.front());
  Value globalBase =
      arith::AddIOp::create(rewriter, loc, lowerBounds.front(),
                            globalBaseOffset);

  SmallVector<Value> taskDependencies =
      createPartialReductionTaskDependencies(rewriter, loc, partialReductions,
                                             lane);
  SmallVector<Value> ownerSliceDependencies =
      materializeSdeOwnerSliceTaskDependencies(source, rewriter, outerMapping,
                                               globalBase, iterCount,
                                               upperBounds, steps);
  taskDependencies.append(ownerSliceDependencies.begin(),
                          ownerSliceDependencies.end());
  auto task = EdtOp::create(rewriter, loc, EdtType::task,
                            EdtConcurrency::intranode, taskDependencies);
  task.setNoVerifyAttr(NoVerifyAttr::get(ctx));
  stampSdeContractOnCoreOp(source, task.getOperation());
  translateSdeExecutionHintsToCoreEdt(source, task);
  copyArtsMetadataAttrs(source, task);

  Block &taskBlock = task.getBody().front();
  SmallVector<BlockArgument> dependencyArgs;
  dependencyArgs.reserve(taskDependencies.size());
  for (Value dep : taskDependencies)
    dependencyArgs.push_back(taskBlock.addArgument(dep.getType(), loc));

  rewriter.setInsertionPointToStart(&taskBlock);
  IRMapping taskMapping = outerMapping;
  if (!partialReductions.empty()) {
    preparePartialReductionSlots(rewriter, loc, reductionPlan, task,
                                 dependencyArgs, taskMapping);
  }

  auto localLoop = scf::ForOp::create(rewriter, loc, zero, iterCount, one);
  if (failed(cloneSuIterateBodyIntoLocalLoop(source, localLoop, globalBase,
                                             lowerBounds, upperBounds, steps,
                                             taskMapping, rewriter)))
    return failure();

  rewriter.setInsertionPointToEnd(&taskBlock);
  if (taskBlock.empty() || !taskBlock.back().hasTrait<OpTrait::IsTerminator>())
    YieldOp::create(rewriter, loc);

  rewriter.setInsertionPointToEnd(epochBlock);
  if (epochBlock->empty() ||
      !epochBlock->back().hasTrait<OpTrait::IsTerminator>())
    YieldOp::create(rewriter, loc);

  rewriter.setInsertionPointAfter(epoch);
  combinePartialReductionsAfterEpoch(rewriter, loc, reductionPlan,
                                     partialReductions, zero, one,
                                     dispatchLanes);
  return success();
}

static void projectSingleNestedForContractToEdt(EdtOp edtOp) {
  if (!edtOp || edtOp.getBody().empty())
    return;

  unsigned schedulingUnits = 0;
  for (Operation &nested : edtOp.getBody().front().without_terminator()) {
    schedulingUnits += countTopLevelSchedulingUnits(&nested, nullptr);
    if (schedulingUnits > 1)
      return;
  }
  if (schedulingUnits != 1)
    return;

  SmallVector<Operation *, 2> candidates;
  for (Operation &nested : edtOp.getBody().front().without_terminator())
    collectTopLevelForContractCandidates(&nested, candidates);
  if (candidates.size() != 1)
    return;

  Operation *source = candidates.front();
  copySemanticContractAttrs(source, edtOp.getOperation());
  copyPlanAttrs(source, edtOp.getOperation());
  copyCoreExecutionHintAttrs(source, edtOp.getOperation());
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

static LogicalResult materializeSdeBlock(Operation *container, Block &block,
                                         PatternRewriter &rewriter,
                                         IRMapping &mapper);
static LogicalResult materializeSdeBlockAtCurrentInsertion(
    Block &block, PatternRewriter &rewriter, IRMapping &mapper);

static SmallVector<Value> materializeCuTaskDeps(sde::SdeCuTaskOp op,
                                                PatternRewriter &rewriter,
                                                IRMapping &mapper) {
  SmallVector<Value> artsDeps;
  for (Value dep : op.getDeps()) {
    auto muDep = dep.getDefiningOp<sde::SdeMuDepOp>();
    if (!muDep)
      continue;

    ArtsMode mode = convertAccessMode(muDep.getMode());
    SmallVector<Value> offsets;
    SmallVector<Value> sizes;
    offsets.reserve(muDep.getOffsets().size());
    sizes.reserve(muDep.getSizes().size());
    for (Value offset : muDep.getOffsets())
      offsets.push_back(mapper.lookupOrDefault(offset));
    for (Value size : muDep.getSizes())
      sizes.push_back(mapper.lookupOrDefault(size));

    SmallVector<Value> pinnedIndices, chunkOffsets, blockSizes;
    if (!offsets.empty() && sizes.empty()) {
      pinnedIndices = offsets;
    } else {
      for (size_t i = 0; i < offsets.size() && i < sizes.size(); ++i) {
        if (ValueAnalysis::isOneConstant(sizes[i])) {
          pinnedIndices.push_back(offsets[i]);
        } else {
          chunkOffsets.push_back(offsets[i]);
          blockSizes.push_back(sizes[i]);
        }
      }
    }

    /// Raw-memref fallback only.  Canonical SDE codelets use `sde.mu_token`,
    /// and `lowerCuCodelet` lowers those tokens directly to `arts.db_acquire`
    /// without routing through DbControlOp/CreateDbs.
    Value source = mapper.lookupOrDefault(muDep.getSource());
    auto dbControl = DbControlOp::create(rewriter, op.getLoc(), mode, source,
                                         pinnedIndices, chunkOffsets,
                                         blockSizes);
    artsDeps.push_back(dbControl);
  }
  return artsDeps;
}

static LogicalResult materializeCuTaskAsEdt(sde::SdeCuTaskOp op,
                                            PatternRewriter &rewriter,
                                            IRMapping &mapper) {
  SmallVector<Value> artsDeps = materializeCuTaskDeps(op, rewriter, mapper);
  auto edtOp = EdtOp::create(rewriter, op.getLoc(), EdtType::task,
                             EdtConcurrency::intranode, artsDeps);
  edtOp.setNoVerifyAttr(NoVerifyAttr::get(rewriter.getContext()));
  if (auto pattern = op.getPatternAttr())
    stampSimple(edtOp.getOperation(),
                mapSdePatternToArtsDepPattern(pattern.getValue()),
                /*rev=*/1);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block &edtBlock = edtOp.getBody().front();
    rewriter.setInsertionPointToStart(&edtBlock);
    if (failed(materializeSdeBlockAtCurrentInsertion(op.getBody().front(),
                                                     rewriter, mapper)))
      return failure();
    if (edtBlock.empty() ||
        !edtBlock.back().hasTrait<OpTrait::IsTerminator>())
      YieldOp::create(rewriter, op.getLoc());
  }

  rewriter.setInsertionPointAfter(edtOp);
  return success();
}

static LogicalResult materializeSdeOp(Operation *op, PatternRewriter &rewriter,
                                      IRMapping &mapper) {
  if (auto iterate = dyn_cast<sde::SdeSuIterateOp>(op))
    return materializeSuIterateAsGenericTaskDispatch(iterate, rewriter, mapper);

  if (auto distribute = dyn_cast<sde::SdeSuDistributeOp>(op)) {
    if (distribute.getBody().empty() ||
        distribute.getBody().front().getNumArguments() != 0)
      return distribute.emitOpError("cannot directly materialize non-empty "
                                    "su_distribute block arguments");
    for (Operation &nested : distribute.getBody().front().without_terminator()) {
      auto iterate = dyn_cast<sde::SdeSuIterateOp>(nested);
      if (iterate && !iterate.getDistributionKindAttr())
        iterate.setDistributionKindAttr(distribute.getKindAttr());
    }
    return materializeSdeBlockAtCurrentInsertion(distribute.getBody().front(),
                                                 rewriter, mapper);
  }

  if (auto task = dyn_cast<sde::SdeCuTaskOp>(op))
    return materializeCuTaskAsEdt(task, rewriter, mapper);

  if (auto barrier = dyn_cast<sde::SdeSuBarrierOp>(op)) {
    if (!barrier.getBarrierEliminated()) {
      ArtsBarrierReasonAttr reason;
      if (auto sdeReason = barrier.getBarrierReasonAttr())
        reason = convertSdeBarrierReason(rewriter.getContext(),
                                         sdeReason.getValue());
      BarrierOp::create(rewriter, barrier.getLoc(), reason);
    }
    return success();
  }

  if (isa<sde::SdeControlTokenOp>(op))
    return success();

  if (isa<sde::SdeYieldOp>(op))
    return success();

  if (op->getDialect()->getNamespace() ==
      sde::ArtsSdeDialect::getDialectNamespace())
    return op->emitError("unsupported SDE operation at direct Core "
                         "materialization boundary");

  Operation *cloned = rewriter.clone(*op, mapper);
  (void)cloned;
  return success();
}

static LogicalResult materializeSdeBlockAtCurrentInsertion(
    Block &block, PatternRewriter &rewriter, IRMapping &mapper) {
  for (Operation &nested : block.without_terminator())
    if (failed(materializeSdeOp(&nested, rewriter, mapper)))
      return failure();
  return success();
}

static LogicalResult materializeSdeBlock(Operation *container, Block &block,
                                         PatternRewriter &rewriter,
                                         IRMapping &mapper) {
  rewriter.setInsertionPoint(container);
  return materializeSdeBlockAtCurrentInsertion(block, rewriter, mapper);
}

/// sde.cu_region <parallel> -> generic Core work units.
struct ParallelCuRegionToCorePattern
    : public OpRewritePattern<sde::SdeCuRegionOp> {
  explicit ParallelCuRegionToCorePattern(MLIRContext *context)
      : OpRewritePattern<sde::SdeCuRegionOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(sde::SdeCuRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != sde::SdeCuKind::parallel)
      return failure();
    if (op.getNumResults() != 0 || op.getBody().empty())
      return failure();

    Block &body = op.getBody().front();
    IRMapping mapper;
    if (failed(materializeSdeBlock(op.getOperation(), body, rewriter, mapper)))
      return failure();

    rewriter.eraseOp(op);
    ++numCuRegionConverted;
    return success();
  }
};

/// sde.cu_region -> arts.edt
struct CuRegionToArtsPattern : public OpRewritePattern<sde::SdeCuRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeCuRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() == sde::SdeCuKind::parallel)
      return failure();

    // cu_regions with tensor iter_args (results) need ConvertToCodelet first.
    if (op.getNumResults() > 0)
      return failure();

    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    // Map SDE kind to ARTS EdtType and EdtConcurrency
    EdtType edtType;
    EdtConcurrency concurrency;
    switch (op.getKind()) {
    case sde::SdeCuKind::parallel:
      return failure();
    case sde::SdeCuKind::single:
      edtType = EdtType::single;
      concurrency = EdtConcurrency::intranode;
      break;
    case sde::SdeCuKind::task:
      edtType = EdtType::task;
      concurrency = EdtConcurrency::intranode;
      break;
    }

    // Override concurrency if SDE scope is explicitly local
    if (auto scope = op.getConcurrencyScope()) {
      if (*scope == sde::SdeConcurrencyScope::local)
        concurrency = EdtConcurrency::intranode;
    }

    auto edtOp = EdtOp::create(rewriter, loc, edtType, concurrency);
    edtOp.setNoVerifyAttr(NoVerifyAttr::get(ctx));

    Block &old = op.getBody().front();
    Block &blk = edtOp.getBody().front();
    blk.getOperations().splice(blk.end(), old.getOperations());
    projectSingleNestedForContractToEdt(edtOp);

    ++numCuRegionConverted;
    rewriter.eraseOp(op);
    return success();
  }
};

/// sde.cu_task -> arts.edt <task>.
///
/// Current raw-memref fallback: task deps are represented as `sde.mu_dep` and
/// temporarily materialized as `arts.db_control` so CreateDbs can allocate DBs,
/// acquire the requested slices, and rewrite the still-raw memref body.
/// Target path: SDE converts those deps to `mu_data`/`mu_token`/`cu_codelet`
/// before this boundary, so this pattern can wire real `arts.db_acquire`
/// dependencies directly and the Core bridge marker disappears.
struct CuTaskToArtsPattern : public OpRewritePattern<sde::SdeCuTaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeCuTaskOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    rewriter.setInsertionPoint(op);
    IRMapping mapper;
    SmallVector<Value> artsDeps = materializeCuTaskDeps(op, rewriter, mapper);

    auto edtOp = EdtOp::create(rewriter, loc, EdtType::task,
                               EdtConcurrency::intranode, artsDeps);
    edtOp.setNoVerifyAttr(NoVerifyAttr::get(ctx));
    if (auto pattern = op.getPatternAttr())
      stampSimple(edtOp.getOperation(),
                  mapSdePatternToArtsDepPattern(pattern.getValue()),
                  /*rev=*/1);

    Block &old = op.getBody().front();
    Block &blk = edtOp.getBody().front();
    blk.getOperations().splice(blk.end(), old.getOperations());

    ++numCuTaskConverted;
    rewriter.eraseOp(op);
    return success();
  }
};

/// sde.su_distribute -> inline wrapped body
struct SuDistributeToArtsPattern
    : public OpRewritePattern<sde::SdeSuDistributeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeSuDistributeOp op,
                                PatternRewriter &rewriter) const override {
    Region &body = op.getBody();
    if (body.empty())
      return failure();

    bool hasNestedSdeIterate = false;
    body.walk([&](sde::SdeSuIterateOp) -> WalkResult {
      hasNestedSdeIterate = true;
      return WalkResult::interrupt();
    });
    if (hasNestedSdeIterate)
      return failure();

    Block &block = body.front();
    if (block.getNumArguments() != 0)
      return failure();

    if (!block.empty())
      if (auto yield = dyn_cast<sde::SdeYieldOp>(&block.back()))
        rewriter.eraseOp(yield);

    rewriter.inlineBlockBefore(&block, op, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

/// sde.su_iterate -> generic Core task dispatch with local scf.for.
struct SuIterateToArtsPattern : public OpRewritePattern<sde::SdeSuIterateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeSuIterateOp op,
                                PatternRewriter &rewriter) const override {
    if (auto strategy = op.getAsyncStrategy();
        strategy && *strategy == sde::SdeAsyncStrategy::cps_chain)
      return op.emitOpError()
             << "sde.async_strategy cps_chain requires SDE CPS dataflow "
                "transformation before Core materialization";

    if (failed(materializeSuIterateAsGenericTaskDispatch(op, rewriter)))
      return failure();
    ++numSuIterateConverted;
    rewriter.eraseOp(op);
    return success();
  }
};

/// sde.su_barrier -> arts.barrier
struct SuBarrierToArtsPattern : public OpRewritePattern<sde::SdeSuBarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeSuBarrierOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getBarrierEliminated()) {
      ArtsBarrierReasonAttr reason;
      if (auto sdeReason = op.getBarrierReasonAttr())
        reason =
            convertSdeBarrierReason(rewriter.getContext(), sdeReason.getValue());
      BarrierOp::create(rewriter, op.getLoc(), reason);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ControlTokenToArtsPattern
    : public OpRewritePattern<sde::SdeControlTokenOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeControlTokenOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->use_empty())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

/// sde.cu_atomic -> arts.atomic_add
struct CuAtomicToArtsPattern : public OpRewritePattern<sde::SdeCuAtomicOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeCuAtomicOp op,
                                PatternRewriter &rewriter) const override {
    // Currently ARTS only supports atomic_add
    if (op.getReductionKind() != sde::SdeReductionKind::add)
      return failure();
    rewriter.replaceOpWithNewOp<AtomicAddOp>(op, op.getAddr(), op.getValue());
    return success();
  }
};

/// sde.yield -> arts.yield
struct SdeYieldToArtsPattern : public OpRewritePattern<sde::SdeYieldOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeYieldOp op,
                                PatternRewriter &rewriter) const override {
    YieldOp::create(rewriter, op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

/// sde.mu_reduction_decl -> semantic declaration only
struct MuReductionDeclToArtsPattern
    : public OpRewritePattern<sde::SdeMuReductionDeclOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeMuReductionDeclOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tensor-path SDE → ARTS lowering patterns.
//
// Three tensor-path SDE ops are lowered here:
//   - sde.mu_data     : tensor<...>            -> arts.db_alloc
//   - sde.mu_token    <mode>                    -> arts.db_acquire <mode>
//   - sde.cu_codelet (tokens...) -> (...)        -> arts.edt
//
// RaiseMemrefToTensor generates these ops from `sde.cu_task` / shared-memref
// IR; this file handles them whenever present.
//
// DB shape convention: one logical DB per `mu_data` with `sizes=[1]`,
// `elementSizes=SHAPE`, `elementType=T`. The ptr type is
// `memref<?xmemref<SHAPE x T>>`. Downstream code recovers the inner
// `memref<SHAPE x T>` via `arts.db_ref %ptr[0]`, then projects into tensor
// land with `bufferization.to_tensor`.
//===----------------------------------------------------------------------===//

/// Trace the DbAllocOp that backs a tensor value. Expected chain:
///   `%ptr = arts.db_alloc ...`
///   `%elem = arts.db_ref %ptr[0] : memref<?xmemref<SHAPExT>> ->
///   memref<SHAPExT>`
///   `%tensor = bufferization.to_tensor %elem`
/// Returns nullptr when the tensor is not backed by a DbAllocOp chain.
static DbAllocOp findBackingDbAlloc(Value tensor) {
  // Walk past any intervening tensor.cast ops. The raise / codelet lowering
  // chain inserts these to reconcile dynamic-shape DB tensors with the
  // (static) codelet operand / yielded types, so a token whose source is a
  // codelet result typically has the chain:
  //   `%cast = tensor.cast %to_tensor : tensor<?xT> to tensor<SHAPExT>`
  //   `%to_tensor = bufferization.to_tensor %db_ref`.
  Value cur = tensor;
  while (auto cast = cur.getDefiningOp<tensor::CastOp>())
    cur = cast.getSource();
  if (auto fromElements = cur.getDefiningOp<tensor::FromElementsOp>()) {
    if (fromElements.getElements().size() == 1) {
      Value element = fromElements.getElements().front();
      if (auto load = element.getDefiningOp<memref::LoadOp>()) {
        Value buffer = load.getMemref();
        if (auto alloc = buffer.getDefiningOp<DbAllocOp>())
          return alloc;
        if (auto ref = buffer.getDefiningOp<DbRefOp>())
          return ref.getSource().getDefiningOp<DbAllocOp>();
      }
    }
  }
  auto toTensor = cur.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor)
    return nullptr;
  Value buffer = toTensor.getBuffer();
  if (auto alloc = buffer.getDefiningOp<DbAllocOp>())
    return alloc;
  if (auto ref = buffer.getDefiningOp<DbRefOp>())
    return ref.getSource().getDefiningOp<DbAllocOp>();
  return nullptr;
}

/// Build a `bufferization.to_tensor` for a memref, producing a ranked tensor
/// of the same shape/element type.
static Value makeToTensor(OpBuilder &builder, Location loc, Value memref,
                          bool writable = false) {
  auto mrType = dyn_cast<MemRefType>(memref.getType());
  assert(mrType && "expected memref type");
  auto tensorType =
      RankedTensorType::get(mrType.getShape(), mrType.getElementType());
  UnitAttr writableAttr =
      writable ? UnitAttr::get(builder.getContext()) : UnitAttr{};
  return bufferization::ToTensorOp::create(
      builder, loc, tensorType, memref, /*restrict=*/UnitAttr{}, writableAttr);
}

/// Given a DB `sourcePtr` whose element type is the payload memref
/// `memref<SHAPExT>`, materialize that inner memref via `arts.db_ref[0]`.
static Value materializeInnerPayload(OpBuilder &builder, Location loc,
                                     Value sourcePtr) {
  Value zero = arts::createZeroIndex(builder, loc);
  return DbRefOp::create(builder, loc, sourcePtr, SmallVector<Value>{zero});
}

/// sde.mu_data : tensor<SHAPE x T>  ->  arts.db_alloc + db_ref + to_tensor
///
/// The tensor SSA value that downstream code still expects is preserved via
/// a `bufferization.to_tensor` over the DB's inner element memref (obtained
/// via `arts.db_ref %ptr[0]`).
static LogicalResult lowerMuData(sde::SdeMuDataOp op) {
  auto tensorType = dyn_cast<RankedTensorType>(op.getHandle().getType());
  if (!tensorType || !tensorType.hasStaticShape()) {
    // Memref-typed mu_data belongs to the v3+ fallback path; leave it alone.
    return success();
  }

  OpBuilder builder(op);
  Location loc = op.getLoc();

  // One DB holds the entire tensor. Outer `sizes=[1]` + inner
  // `elementSizes=SHAPE` keeps the payload addressable as a single
  // `memref<SHAPE x T>` via `arts.db_ref %ptr[0]`.
  SmallVector<Value> sizes{arts::createOneIndex(builder, loc)};
  SmallVector<Value> elementSizes;
  elementSizes.reserve(tensorType.getRank());
  for (int64_t dim : tensorType.getShape())
    elementSizes.push_back(arts::createConstantIndex(builder, loc, dim));

  Value route = arts::createCurrentNodeRoute(builder, loc);

  // TODO: honor `init` attribute. Until `arts.db_alloc` grows a first-class
  // init operand, the init attribute is dropped and the DB comes up
  // uninitialized. RaiseMemrefToTensor can synthesize an initializer EDT.
  auto dbAlloc = DbAllocOp::create(
      builder, loc, ArtsMode::inout, route, DbAllocType::heap, DbMode::write,
      tensorType.getElementType(),
      /*address=*/Value{}, std::move(sizes), std::move(elementSizes));

  Value inner = materializeInnerPayload(builder, loc, dbAlloc.getPtr());
  Value replacementTensor;
  if (tensorType.getRank() == 0) {
    Value zero = arts::createZeroIndex(builder, loc);
    Value scalar =
        memref::LoadOp::create(builder, loc, inner, SmallVector<Value>{zero});
    replacementTensor =
        tensor::FromElementsOp::create(builder, loc, tensorType, scalar);
  } else {
    replacementTensor = makeToTensor(builder, loc, inner, /*writable=*/true);
  }
  op.getResult().replaceAllUsesWith(replacementTensor);
  op.erase();
  return success();
}

/// sde.cu_codelet (+ associated sde.mu_token operands) -> arts.edt
///
/// Each token operand becomes an `arts.db_acquire` whose ptr is carried as
/// an EDT dependency. The codelet body runs inside the EDT, with
/// `bufferization.to_tensor` bridging the memref block arg back to the
/// tensor-typed value the codelet body expects. Writable tokens surface as
/// destination-passing-style results: the yielded tensor is materialized
/// into the acquired DB memref, and the codelet's SSA result becomes a
/// fresh `bufferization.to_tensor` over the same DB ptr (the update is
/// observable through the DB handle, not through a new tensor SSA value).
static LogicalResult lowerCuCodelet(sde::SdeCuCodeletOp codelet) {
  Location loc = codelet.getLoc();
  auto *ctx = codelet.getContext();
  OpBuilder rewriter(codelet);

  // Each codelet operand must be produced by a `sde.mu_token` whose source
  // tensor is backed by a concrete DbAllocOp. If any token is not DB-backed
  // we cannot lower this codelet — bail out so the invariant is enforced at
  // VerifySdeLowered time rather than producing half-converted IR.
  SmallVector<sde::SdeMuTokenOp> muTokens;
  muTokens.reserve(codelet.getTokens().size());
  SmallVector<DbAllocOp> backingAllocs;
  backingAllocs.reserve(codelet.getTokens().size());
  for (Value token : codelet.getTokens()) {
    auto muToken = token.getDefiningOp<sde::SdeMuTokenOp>();
    if (!muToken)
      return codelet.emitOpError(
          "token operand is not produced by sde.mu_token");
    DbAllocOp alloc = findBackingDbAlloc(muToken.getSource());
    if (!alloc)
      return codelet.emitOpError(
          "token source is not backed by arts.db_alloc; RaiseMemrefToTensor "
          "must produce db-backed tensors for codelets to lower");
    muTokens.push_back(muToken);
    backingAllocs.push_back(alloc);
  }

  // Build one db_acquire per token. Read-only tokens surface as
  // arts.db_acquire <in>; write tokens as <out>; readwrite as <inout>.
  //
  // Slice tokens encode their element-space offsets/sizes on the acquire's
  // `partition_*` channels (DB-space `offsets`/`sizes` are single-DB
  // coordinates — always trivial for the sizes=[1] convention we use).
  rewriter.setInsertionPoint(codelet);
  SmallVector<Value> acquirePtrs;
  SmallVector<Type> blockArgTypes;
  SmallVector<Location> blockArgLocs;
  SmallVector<bool> writableFlags;
  acquirePtrs.reserve(muTokens.size());
  blockArgTypes.reserve(muTokens.size());
  blockArgLocs.reserve(muTokens.size());
  writableFlags.reserve(muTokens.size());

  for (auto [idx, muToken] : llvm::enumerate(muTokens)) {
    ArtsMode mode = convertAccessMode(muToken.getMode());
    bool writable = muToken.getMode() == sde::SdeAccessMode::write ||
                    muToken.getMode() == sde::SdeAccessMode::readwrite;
    writableFlags.push_back(writable);

    SmallVector<Value> partitionOffsets(muToken.getOffsets().begin(),
                                        muToken.getOffsets().end());
    SmallVector<Value> partitionSizes(muToken.getSizes().begin(),
                                      muToken.getSizes().end());

    DbAllocOp alloc = backingAllocs[idx];
    std::optional<PartitionMode> partitionMode;
    if (!partitionOffsets.empty() || !partitionSizes.empty())
      partitionMode = PartitionMode::block;
    // The DB's outer rank is 1 (one partition holding the whole tensor);
    // mirror that on the acquire so db_ref can index into it later.
    SmallVector<Value> dbOffsets{
        arts::createZeroIndex(rewriter, muToken.getLoc())};
    SmallVector<Value> dbSizes{
        arts::createOneIndex(rewriter, muToken.getLoc())};
    auto acq = DbAcquireOp::create(
        rewriter, muToken.getLoc(), mode, alloc.getGuid(), alloc.getPtr(),
        partitionMode,
        /*indices=*/SmallVector<Value>{}, std::move(dbOffsets),
        std::move(dbSizes),
        /*partitionIndices=*/SmallVector<Value>{}, std::move(partitionOffsets),
        std::move(partitionSizes),
        /*boundsValid=*/Value{},
        /*elementOffsets=*/SmallVector<Value>{},
        /*elementSizes=*/SmallVector<Value>{});
    acquirePtrs.push_back(acq.getPtr());
    blockArgTypes.push_back(acq.getPtr().getType());
    blockArgLocs.push_back(muToken.getLoc());
  }

  // Create the EDT. Task-intranode is the default codelet concurrency per
  // The raise does not encode cross-node placement yet.
  auto edtOp = EdtOp::create(rewriter, loc, EdtType::task,
                             EdtConcurrency::intranode, acquirePtrs);

  Block &edtBlock = edtOp.getBody().front();
  for (auto [argTy, argLoc] : llvm::zip(blockArgTypes, blockArgLocs))
    edtBlock.addArgument(argTy, argLoc);

  // Inside the EDT, materialize a tensor view for each block argument.
  // The block arg is the acquire's ptr (memref<?xmemref<SHAPExT>>); we use
  // `arts.db_ref[0]` + `bufferization.to_tensor` to land in tensor space
  // with shape `SHAPE`. Slice tokens then take `tensor.extract_slice` to
  // reach the codelet block arg's slice_type.
  OpBuilder::InsertionGuard bodyGuard(rewriter);
  rewriter.setInsertionPointToStart(&edtBlock);

  IRMapping mapper;
  SmallVector<Value> innerPayloads; // per-token inner memref
  innerPayloads.reserve(muTokens.size());
  Block &codeletBlock = codelet.getBody().front();
  unsigned numTokens = codelet.getTokens().size();
  for (unsigned idx = 0; idx < numTokens; ++idx) {
    BlockArgument codeletArg = codeletBlock.getArgument(idx);
    Value edtArg = edtBlock.getArgument(idx);
    Value inner = materializeInnerPayload(rewriter, codelet.getLoc(), edtArg);
    innerPayloads.push_back(inner);

    Value view = makeToTensor(rewriter, codelet.getLoc(), inner,
                              /*writable=*/writableFlags[idx]);

    // Map the codelet block arg to the tensor view, narrowing to the
    // slice_type when the token addressed a sub-region.
    auto slicedType = dyn_cast<RankedTensorType>(codeletArg.getType());
    auto viewType = cast<RankedTensorType>(view.getType());
    if (slicedType && slicedType != viewType) {
      sde::SdeMuTokenOp muToken = muTokens[idx];
      if (slicedType.getRank() == 0 && viewType.getRank() == 1) {
        // Rank-0 tensor backed by rank-1 memref (DB scalar convention).
        // Load the scalar from memref[0] and create a rank-0 tensor.
        Value zero = arts::createZeroIndex(rewriter, codelet.getLoc());
        Value scalar = memref::LoadOp::create(rewriter, codelet.getLoc(), inner,
                                              SmallVector<Value>{zero});
        view = tensor::FromElementsOp::create(rewriter, codelet.getLoc(),
                                              slicedType, scalar);
      } else if (muToken.getOffsets().empty() && muToken.getSizes().empty()) {
        // Whole-tensor token; the only mismatch is dynamic vs. static
        // shape, which is a plain tensor.cast.
        view = tensor::CastOp::create(rewriter, codelet.getLoc(), slicedType,
                                      view);
      } else {
        // Slice token: carry explicit offsets/sizes through extract_slice.
        // Offsets/sizes are dynamic SSA values, so ExtractSliceOp infers a
        // fully dynamic result type. Cast to the codelet block arg's
        // (possibly static) slice type afterwards.
        SmallVector<OpFoldResult> offsets, sizes, strides;
        offsets.reserve(slicedType.getRank());
        sizes.reserve(slicedType.getRank());
        strides.reserve(slicedType.getRank());
        Value oneV = arts::createOneIndex(rewriter, codelet.getLoc());
        for (int dim = 0; dim < slicedType.getRank(); ++dim) {
          offsets.push_back(OpFoldResult(muToken.getOffsets()[dim]));
          sizes.push_back(OpFoldResult(muToken.getSizes()[dim]));
          strides.push_back(OpFoldResult(oneV));
        }
        Value sliced = tensor::ExtractSliceOp::create(
            rewriter, codelet.getLoc(), view, offsets, sizes, strides);
        if (sliced.getType() != slicedType)
          sliced = tensor::CastOp::create(rewriter, codelet.getLoc(),
                                          slicedType, sliced);
        view = sliced;
      }
    }
    mapper.map(codeletArg, view);
  }

  for (auto [idx, capture] : llvm::enumerate(codelet.getCaptures())) {
    BlockArgument codeletArg = codeletBlock.getArgument(numTokens + idx);
    mapper.map(codeletArg, capture);
  }

  // Clone codelet body ops (except the terminator) into the EDT body. The
  // terminator is a `sde.yield` whose operands drive destination-passing
  // materialization; we handle it explicitly below.
  Operation *terminator = codeletBlock.getTerminator();
  for (Operation &nested : codeletBlock.without_terminator())
    rewriter.insert(nested.clone(mapper));

  // For each writable token, materialize the yielded tensor into the
  // corresponding block-arg memref. `sde.cu_codelet` has one result per
  // writable token, and the terminator is `sde.yield`
  // with one operand per result.
  SmallVector<Value> yieldedValues;
  if (auto yieldOp = dyn_cast<sde::SdeYieldOp>(terminator)) {
    for (Value operand : yieldOp.getOperands())
      yieldedValues.push_back(mapper.lookupOrDefault(operand));
  }

  // Pair each yielded tensor with the writable inner payload in positional
  // order. If the yielded tensor is a slice, insert it
  // back into a whole-tensor view via `tensor.insert_slice` so the final
  // materialization target has the full shape.
  unsigned yieldIdx = 0;
  for (auto [idx, writable] : llvm::enumerate(writableFlags)) {
    if (!writable)
      continue;
    if (yieldIdx >= yieldedValues.size())
      break;
    Value src = yieldedValues[yieldIdx++];
    Value destMemref = innerPayloads[idx];
    auto destMrType = cast<MemRefType>(destMemref.getType());
    auto destTensorType = RankedTensorType::get(destMrType.getShape(),
                                                destMrType.getElementType());
    auto srcTensorType = cast<RankedTensorType>(src.getType());
    if (srcTensorType != destTensorType) {
      sde::SdeMuTokenOp muToken = muTokens[idx];
      if (srcTensorType.getRank() == 0 && destMrType.getRank() == 1) {
        // Rank-0 tensor yielded into rank-1 memref (DB scalar convention).
        // Extract the scalar and store directly — skip materialize.
        Value scalar = tensor::ExtractOp::create(rewriter, codelet.getLoc(),
                                                 src, ValueRange{});
        Value zero = arts::createZeroIndex(rewriter, codelet.getLoc());
        memref::StoreOp::create(rewriter, codelet.getLoc(), scalar, destMemref,
                                SmallVector<Value>{zero});
        continue; // No materialize_in_destination needed.
      } else if (muToken.getOffsets().empty() && muToken.getSizes().empty()) {
        // Whole-tensor token: static-vs-dynamic mismatch only.
        src = tensor::CastOp::create(rewriter, codelet.getLoc(), destTensorType,
                                     src);
      } else {
        // Slice update: stitch the slice back into a whole-tensor view
        // before materializing, preserving unchanged elements.
        Value viewTensor = makeToTensor(rewriter, codelet.getLoc(), destMemref,
                                        /*writable=*/true);
        SmallVector<OpFoldResult> offsets, sizes, strides;
        offsets.reserve(destTensorType.getRank());
        sizes.reserve(destTensorType.getRank());
        strides.reserve(destTensorType.getRank());
        Value oneV = arts::createOneIndex(rewriter, codelet.getLoc());
        for (int dim = 0; dim < destTensorType.getRank(); ++dim) {
          offsets.push_back(OpFoldResult(muToken.getOffsets()[dim]));
          sizes.push_back(OpFoldResult(muToken.getSizes()[dim]));
          strides.push_back(OpFoldResult(oneV));
        }
        src =
            tensor::InsertSliceOp::create(rewriter, codelet.getLoc(), src,
                                          viewTensor, offsets, sizes, strides);
      }
    }
    bufferization::MaterializeInDestinationOp::create(
        rewriter, codelet.getLoc(), /*result=*/TypeRange{}, src, destMemref,
        /*restrict=*/UnitAttr{}, /*writable=*/UnitAttr::get(ctx));
  }

  YieldOp::create(rewriter, loc);

  // Rewire codelet results: each SSA result corresponds to a writable
  // token's parent tensor. Re-materialize the parent tensor from the DB's
  // inner payload so tensor users continue to type-check.
  rewriter.setInsertionPointAfter(edtOp);
  SmallVector<Value> newResults;
  newResults.reserve(codelet.getNumResults());
  unsigned writableIdx = 0;
  for (auto [idx, writable] : llvm::enumerate(writableFlags)) {
    if (!writable)
      continue;
    if (writableIdx >= codelet.getNumResults())
      break;
    DbAllocOp alloc = backingAllocs[idx];
    Value inner =
        materializeInnerPayload(rewriter, codelet.getLoc(), alloc.getPtr());
    Value parentTensor =
        makeToTensor(rewriter, codelet.getLoc(), inner, /*writable=*/true);
    Value result = codelet.getResult(writableIdx);
    if (parentTensor.getType() != result.getType()) {
      auto dstType = dyn_cast<RankedTensorType>(result.getType());
      auto srcType = cast<RankedTensorType>(parentTensor.getType());
      if (dstType && dstType.getRank() == 0 && srcType.getRank() == 1) {
        // Rank-0 result from rank-1 memref: load scalar, wrap in rank-0.
        Value zero = arts::createZeroIndex(rewriter, codelet.getLoc());
        Value scalar = memref::LoadOp::create(rewriter, codelet.getLoc(), inner,
                                              SmallVector<Value>{zero});
        parentTensor = tensor::FromElementsOp::create(
            rewriter, codelet.getLoc(), dstType, scalar);
      } else if (dstType) {
        parentTensor = tensor::CastOp::create(rewriter, codelet.getLoc(),
                                              dstType, parentTensor);
      }
    }
    newResults.push_back(parentTensor);
    ++writableIdx;
  }

  // Erase the codelet and its mu_token producers.
  codelet.getResults().replaceAllUsesWith(newResults);
  codelet.erase();
  for (sde::SdeMuTokenOp tok : muTokens)
    if (tok->use_empty())
      tok.erase();
  return success();
}

struct TensorMemrefView {
  Value memref;
  SmallVector<Value> offsets;
  SmallVector<Value> strides;
};

static Value materializeIndexValue(OpBuilder &builder, Location loc,
                                   OpFoldResult value) {
  if (auto dynamic = value.dyn_cast<Value>())
    return dynamic;
  auto attr = cast<IntegerAttr>(value.dyn_cast<Attribute>());
  return arts::createConstantIndex(builder, loc, attr.getInt());
}

static Value mulIndexIfNeeded(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs) {
  if (ValueAnalysis::isOneConstant(rhs))
    return lhs;
  if (ValueAnalysis::isOneConstant(lhs))
    return rhs;
  if (ValueAnalysis::isZeroConstant(lhs) || ValueAnalysis::isZeroConstant(rhs))
    return arts::createZeroIndex(builder, loc);
  return arith::MulIOp::create(builder, loc, lhs, rhs).getResult();
}

static Value addIndexIfNeeded(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs) {
  if (ValueAnalysis::isZeroConstant(lhs))
    return rhs;
  if (ValueAnalysis::isZeroConstant(rhs))
    return lhs;
  return arith::AddIOp::create(builder, loc, lhs, rhs).getResult();
}

static void composeSliceMap(TensorMemrefView &view, tensor::ExtractSliceOp slice,
                            OpBuilder &builder, Location loc) {
  SmallVector<Value> sliceOffsets;
  SmallVector<Value> sliceStrides;
  for (OpFoldResult offset : slice.getMixedOffsets())
    sliceOffsets.push_back(materializeIndexValue(builder, loc, offset));
  for (OpFoldResult stride : slice.getMixedStrides())
    sliceStrides.push_back(materializeIndexValue(builder, loc, stride));

  if (view.offsets.empty()) {
    view.offsets = std::move(sliceOffsets);
    view.strides = std::move(sliceStrides);
    return;
  }

  for (size_t idx = 0, e = view.offsets.size(); idx < e; ++idx) {
    Value scaledOffset =
        mulIndexIfNeeded(builder, loc, view.offsets[idx], sliceStrides[idx]);
    view.offsets[idx] =
        addIndexIfNeeded(builder, loc, sliceOffsets[idx], scaledOffset);
    view.strides[idx] =
        mulIndexIfNeeded(builder, loc, view.strides[idx], sliceStrides[idx]);
  }
}

/// Trace `tensor` back through `tensor.cast` / `tensor.extract_slice` /
/// `tensor.insert` ops to the underlying memref that backs the tensor view.
/// Slice offsets and strides are preserved so scalar tensor.extract/insert ops
/// fold to the exact element addressed by the SDE token, not a coarse whole-DB
/// copy. Returns std::nullopt if the tensor is not backed by a memref-view chain
/// we can lower (e.g., it came from a non-tensor-path producer).
static std::optional<TensorMemrefView>
traceTensorToBackingMemref(Value tensor, OpBuilder &builder, Location loc) {
  Value cur = tensor;
  TensorMemrefView view;
  for (;;) {
    if (auto cast = cur.getDefiningOp<tensor::CastOp>()) {
      cur = cast.getSource();
      continue;
    }
    if (auto insert = cur.getDefiningOp<tensor::InsertOp>()) {
      cur = insert.getDest();
      continue;
    }
    if (auto slice = cur.getDefiningOp<tensor::ExtractSliceOp>()) {
      auto sourceType = dyn_cast<RankedTensorType>(slice.getSource().getType());
      auto resultType = dyn_cast<RankedTensorType>(slice.getResult().getType());
      if (!sourceType || !resultType ||
          sourceType.getRank() != resultType.getRank())
        return std::nullopt;
      composeSliceMap(view, slice, builder, loc);
      cur = slice.getSource();
      continue;
    }
    if (auto toTensor = cur.getDefiningOp<bufferization::ToTensorOp>()) {
      view.memref = toTensor.getBuffer();
      return view;
    }
    if (auto muToTensor = cur.getDefiningOp<sde::SdeMuMemrefToTensorOp>()) {
      view.memref = muToTensor.getMemref();
      return view;
    }
    return std::nullopt;
  }
}

/// Fold the tensor-path ops introduced by codelet lowering down to memref
/// load/store semantics. See `lowerTensorPathOps` Step 4 for context.
///
/// We walk the module in program order and transform each op in place:
///   - `tensor.extract %t[idx]` becomes `memref.load %m[idx]` when %t
///     traces to a memref view.
///   - `tensor.insert %val into %t[idx]` becomes `memref.store %val, %m[idx]`
///     and the insert's SSA result is RAUW'd with its dest (so downstream
///     extracts re-trace to the SAME backing memref, reading the store we
///     just emitted).
///   - `bufferization.materialize_in_destination` whose source tensor traces
///     back to the same memref is a no-op (the stores already happened).
///   - Dead tensor.cast / tensor.extract_slice / bufferization.to_tensor are
///     erased.
///
/// Walk order matters: a later `tensor.extract` on a value produced by an
/// earlier `tensor.insert` must be translated AFTER that insert has become a
/// memref.store, so the emitted `memref.load` is placed AFTER the store and
/// reads the fresh value.
/// Rank-0 tensor backed by rank-1 memref: supply index 0 for the extra
/// dimension (DB convention: elementSizes=[1] for scalars).
/// Returns false if indices don't match memref rank after fixup.
static bool fixupRank0Indices(SmallVectorImpl<Value> &indices, MemRefType mrType,
                              OpBuilder &builder, Location loc) {
  if (indices.empty() && mrType.getRank() == 1)
    indices.push_back(arts::createZeroIndex(builder, loc));
  return static_cast<int64_t>(indices.size()) == mrType.getRank();
}

static FailureOr<SmallVector<Value>>
buildMemrefIndices(const TensorMemrefView &view, ValueRange tensorIndices,
                   MemRefType mrType, OpBuilder &builder, Location loc) {
  SmallVector<Value> indices(tensorIndices.begin(), tensorIndices.end());
  if (!view.offsets.empty()) {
    if (view.offsets.size() != indices.size() ||
        view.strides.size() != indices.size())
      return failure();
    for (size_t i = 0, e = indices.size(); i < e; ++i) {
      Value scaled = mulIndexIfNeeded(builder, loc, indices[i], view.strides[i]);
      indices[i] = addIndexIfNeeded(builder, loc, view.offsets[i], scaled);
    }
  }
  if (!fixupRank0Indices(indices, mrType, builder, loc))
    return failure();
  return indices;
}

static void eraseDeadTensorPathCarrierOps(ModuleOp module) {
  // Iteratively erase dead tensor-path carriers. These ops are transient
  // SDE/Core glue; after scalar tensor ops are folded to memref operations,
  // any remaining dead carrier must be removed before VerifySdeLowered runs.
  for (;;) {
    SmallVector<Operation *> dead;
    module.walk([&](Operation *op) {
      if (isa<bufferization::ToTensorOp, tensor::CastOp,
              tensor::ExtractSliceOp, tensor::InsertSliceOp, tensor::EmptyOp,
              tensor::FromElementsOp>(op) &&
          op->use_empty()) {
        dead.push_back(op);
      }
    });
    if (dead.empty())
      return;
    for (Operation *op : llvm::reverse(dead))
      op->erase();
  }
}

static LogicalResult emitMemrefCopy(OpBuilder &builder, Location loc,
                                    Value source, Value dest) {
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  auto destType = dyn_cast<MemRefType>(dest.getType());
  if (!sourceType || !destType)
    return failure();

  unsigned destRank = destType.getRank();
  unsigned sourceRank = sourceType.getRank();
  if (sourceRank != destRank &&
      !(destRank == 0 && sourceRank == 1 && sourceType.getDimSize(0) == 1))
    return failure();

  SmallVector<Value> destIndices;
  std::function<void(unsigned)> emitLoopNest = [&](unsigned dim) {
    if (dim == destRank) {
      SmallVector<Value> sourceIndices(destIndices.begin(), destIndices.end());
      if (destRank == 0 && sourceRank == 1)
        sourceIndices.push_back(arts::createZeroIndex(builder, loc));
      Value loaded =
          memref::LoadOp::create(builder, loc, source, sourceIndices);
      memref::StoreOp::create(builder, loc, loaded, dest, destIndices);
      return;
    }

    Value lower = arts::createZeroIndex(builder, loc);
    Value upper;
    if (destType.isDynamicDim(dim))
      upper = memref::DimOp::create(builder, loc, dest, dim);
    else
      upper = arts::createConstantIndex(builder, loc, destType.getDimSize(dim));
    Value step = arts::createOneIndex(builder, loc);
    auto loop = scf::ForOp::create(builder, loc, lower, upper, step);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    destIndices.push_back(loop.getInductionVar());
    emitLoopNest(dim + 1);
    destIndices.pop_back();
  };

  emitLoopNest(0);
  return success();
}

static LogicalResult emitFromElementsStore(OpBuilder &builder, Location loc,
                                           tensor::FromElementsOp source,
                                           Value dest) {
  auto tensorType = dyn_cast<RankedTensorType>(source.getType());
  auto destType = dyn_cast<MemRefType>(dest.getType());
  if (!tensorType || !destType || tensorType.getRank() != 0 ||
      source.getElements().size() != 1)
    return failure();

  Value scalar = source.getElements().front();
  SmallVector<Value> indices;
  if (destType.getRank() == 1 && destType.getDimSize(0) == 1)
    indices.push_back(arts::createZeroIndex(builder, loc));
  else if (destType.getRank() != 0)
    return failure();

  memref::StoreOp::create(builder, loc, scalar, dest, indices);
  return success();
}

static void foldTensorPathToMemref(ModuleOp module) {
  // Collect target ops in IR pre-order so we can process them as they
  // appear in the instruction stream. We intentionally do not use
  // `module.walk` inline because we mutate the IR.
  SmallVector<Operation *> ordered;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<tensor::ExtractOp, tensor::InsertOp,
            bufferization::MaterializeInDestinationOp>(op))
      ordered.push_back(op);
  });

  for (Operation *op : ordered) {
    if (auto extract = dyn_cast<tensor::ExtractOp>(op)) {
      OpBuilder builder(extract);
      std::optional<TensorMemrefView> view =
          traceTensorToBackingMemref(extract.getTensor(), builder,
                                     extract.getLoc());
      if (!view || !view->memref)
        continue;
      auto mrType = dyn_cast<MemRefType>(view->memref.getType());
      if (!mrType)
        continue;
      FailureOr<SmallVector<Value>> indices =
          buildMemrefIndices(*view, extract.getIndices(), mrType, builder,
                             extract.getLoc());
      if (failed(indices))
        continue;
      Value loaded = memref::LoadOp::create(builder, extract.getLoc(),
                                            view->memref, *indices);
      extract.getResult().replaceAllUsesWith(loaded);
      extract.erase();
    } else if (auto insert = dyn_cast<tensor::InsertOp>(op)) {
      OpBuilder builder(insert);
      std::optional<TensorMemrefView> view =
          traceTensorToBackingMemref(insert.getDest(), builder,
                                     insert.getLoc());
      if (!view || !view->memref)
        continue;
      auto mrType = dyn_cast<MemRefType>(view->memref.getType());
      if (!mrType)
        continue;
      FailureOr<SmallVector<Value>> indices = buildMemrefIndices(
          *view, insert.getIndices(), mrType, builder, insert.getLoc());
      if (failed(indices))
        continue;
      memref::StoreOp::create(builder, insert.getLoc(), insert.getScalar(),
                              view->memref, *indices);
      insert.getResult().replaceAllUsesWith(insert.getDest());
      insert.erase();
    } else if (auto mat =
                   dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
      if (auto fromElements =
              mat.getSource().getDefiningOp<tensor::FromElementsOp>()) {
        OpBuilder builder(mat);
        if (succeeded(emitFromElementsStore(builder, mat.getLoc(), fromElements,
                                            mat.getDest()))) {
          mat.erase();
          continue;
        }
      }

      OpBuilder builder(mat);
      std::optional<TensorMemrefView> source =
          traceTensorToBackingMemref(mat.getSource(), builder, mat.getLoc());
      if (!source || !source->memref)
        continue;
      Value dest = mat.getDest();
      if (source->memref != dest) {
        if (failed(emitMemrefCopy(builder, mat.getLoc(), source->memref, dest)))
          continue;
      }
      mat.erase();
    }
  }

  eraseDeadTensorPathCarrierOps(module);
}

static void canonicalizeTensorPathRegionResults(ModuleOp module) {
  RewritePatternSet patterns(module.getContext());
  scf::IfOp::getCanonicalizationPatterns(patterns, module.getContext());
  scf::ForOp::getCanonicalizationPatterns(patterns, module.getContext());

  GreedyRewriteConfig config;
  config.setMaxIterations(4);
  (void)applyPatternsGreedily(module, std::move(patterns), config);

  eraseDeadTensorPathCarrierOps(module);
}

/// Drive the tensor-path lowerings ahead of the greedy rewriter. Order:
///   1. `sde.cu_codelet`  ->  arts.edt (acquires + body + materialize).
///   2. `sde.mu_token` leftovers (orphan or producer-less) -> erase.
///   3. `sde.mu_data`     ->  arts.db_alloc + db_ref + to_tensor chain.
///   4. Fold tensor-path ops to memref ops.
///
/// Steps 1/2 before step 3 ensures that when we tear down a codelet, all of
/// its `sde.mu_token` operands are still resolvable back to their originating
/// `sde.mu_data` — which remains an SSA producer until step 3 replaces it
/// with the ARTS-side chain.
static LogicalResult lowerTensorPathOps(ModuleOp module) {
  // Step 1: lower codelets (+ their tokens).
  //
  // `lowerMuData` erases its input op and RAUWs the tensor result with a
  // `bufferization.to_tensor` over the new DB chain. When two codelets (or
  // two tokens of the same codelet) share a single `sde.mu_data`, the first
  // lowering erases it and subsequent `muToken.getSource()` lookups see the
  // replacement `to_tensor` instead of the original `mu_data` — so
  // `getDefiningOp<SdeMuDataOp>()` naturally returns null. We still guard
  // with a seen-set to make the intent explicit and to prevent any future
  // re-entrancy (e.g., if lookup traversal ever changes) from dereferencing
  // an erased op.
  DenseSet<sde::SdeMuDataOp> loweredMuData;
  SmallVector<sde::SdeCuCodeletOp> codelets;
  module.walk([&](sde::SdeCuCodeletOp op) { codelets.push_back(op); });
  for (sde::SdeCuCodeletOp codelet : codelets) {
    // Lower mu_data producers feeding this codelet's tokens first so that
    // `findBackingDbAlloc` can see the DB chain when codelet lowering runs.
    for (Value token : codelet.getTokens()) {
      auto muToken = token.getDefiningOp<sde::SdeMuTokenOp>();
      if (!muToken)
        continue;
      auto muData = muToken.getSource().getDefiningOp<sde::SdeMuDataOp>();
      if (!muData)
        continue;
      if (!loweredMuData.insert(muData).second)
        continue;
      if (failed(lowerMuData(muData)))
        return failure();
    }
    if (failed(lowerCuCodelet(codelet)))
      return failure();
  }

  // Step 2: drop orphan `sde.mu_token`s. After step 1 they should be
  // use-less; anything else is malformed input.
  SmallVector<sde::SdeMuTokenOp> orphanTokens;
  module.walk([&](sde::SdeMuTokenOp op) { orphanTokens.push_back(op); });
  for (sde::SdeMuTokenOp op : orphanTokens) {
    if (!op.getToken().use_empty())
      return op.emitOpError(
          "sde.mu_token still has users after codelet lowering");
    op.erase();
  }

  // Step 3: any remaining `sde.mu_data` (unused by codelets) still needs
  // lowering so VerifySdeLowered passes.
  SmallVector<sde::SdeMuDataOp> muDatas;
  module.walk([&](sde::SdeMuDataOp op) { muDatas.push_back(op); });
  for (sde::SdeMuDataOp op : muDatas) {
    if (failed(lowerMuData(op)))
      return failure();
  }

  // Step 3b: lower sde.mu_alloc → tensor.empty,
  //          sde.mu_memref_to_tensor → bufferization.to_tensor,
  //          sde.mu_tensor_to_memref → bufferization.materialize_in_destination
  {
    SmallVector<sde::SdeMuAllocOp> muAllocs;
    SmallVector<sde::SdeMuMemrefToTensorOp> muToTensors;
    SmallVector<sde::SdeMuTensorToMemrefOp> muToMemrefs;
    module.walk([&](Operation *op) {
      if (auto a = dyn_cast<sde::SdeMuAllocOp>(op))
        muAllocs.push_back(a);
      else if (auto t = dyn_cast<sde::SdeMuMemrefToTensorOp>(op))
        muToTensors.push_back(t);
      else if (auto m = dyn_cast<sde::SdeMuTensorToMemrefOp>(op))
        muToMemrefs.push_back(m);
    });

    for (sde::SdeMuAllocOp op : muAllocs) {
      OpBuilder b(op);
      auto tensorTy = cast<RankedTensorType>(op.getTensor().getType());
      SmallVector<int64_t> staticShape(tensorTy.getShape());
      Value empty = tensor::EmptyOp::create(b, op.getLoc(), staticShape,
                                            tensorTy.getElementType(),
                                            op.getDynamicSizes());
      op.getTensor().replaceAllUsesWith(empty);
      op.erase();
    }

    for (sde::SdeMuMemrefToTensorOp op : muToTensors) {
      OpBuilder b(op);
      auto tensorTy = cast<RankedTensorType>(op.getTensor().getType());
      Value tensor = bufferization::ToTensorOp::create(b, op.getLoc(), tensorTy,
                                                       op.getMemref());
      op.getTensor().replaceAllUsesWith(tensor);
      op.erase();
    }

    for (sde::SdeMuTensorToMemrefOp op : muToMemrefs) {
      OpBuilder b(op);
      auto matOp = bufferization::MaterializeInDestinationOp::create(
          b, op.getLoc(), op.getSource(), op.getDest());
      matOp.setWritable(true);
      op.erase();
    }
  }

  // Step 4: fold the tensor-op chain produced by codelet lowering down to
  // pure memref loads/stores. ConvertArtsToLLVM does not know how to
  // consume `bufferization.to_tensor` / `materialize_in_destination` /
  // `tensor.insert` / `tensor.extract`, so we translate each of them to
  // the equivalent memref.load / memref.store against the backing DB
  // memref. The folds are only valid for the tensor-path pattern
  // RaiseMemrefToTensor emits (to_tensor of a db_ref, scalar extract /
  // insert, final materialize_in_destination on the same memref); other
  // tensor IR in the module is left untouched.
  foldTensorPathToMemref(module);
  canonicalizeTensorPathRegionResults(module);

  return success();
}

/// sde.cu_reduce -> handled inline (reduce to atomic_add for now)
struct CuReduceToArtsPattern : public OpRewritePattern<sde::SdeCuReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeCuReduceOp op,
                                PatternRewriter &rewriter) const override {
    // For add reductions on memref accumulators, lower to atomic_add
    if (op.getReductionKind() != sde::SdeReductionKind::add)
      return failure();
    auto acc = op.getAccumulator();
    if (!isa<MemRefType>(acc.getType()))
      return failure();

    (void)AtomicAddOp::create(rewriter, op.getLoc(), acc, op.getPartial());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ResourceQueryToArtsPattern
    : public OpRewritePattern<sde::SdeResourceQueryOp> {
  using OpRewritePattern<sde::SdeResourceQueryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeResourceQueryOp op,
                                PatternRewriter &rewriter) const override {
    RuntimeQueryKind kind;
    switch (op.getKind()) {
    case sde::SdeResourceQueryKind::logicalWorkers:
      kind = RuntimeQueryKind::totalWorkers;
      break;
    }

    Value runtimeQuery =
        RuntimeQueryOp::create(rewriter, op.getLoc(), kind).getResult();
    Value indexValue = ValueAnalysis::castToIndex(runtimeQuery, rewriter,
                                                  op.getLoc());
    rewriter.replaceOp(op, indexValue);
    return success();
  }
};

static LogicalResult rejectUnmaterializedCpsChain(ModuleOp module) {
  WalkResult result = module.walk([&](sde::SdeSuIterateOp op) {
    auto strategy = op.getAsyncStrategy();
    if (!strategy || *strategy != sde::SdeAsyncStrategy::cps_chain)
      return WalkResult::advance();

    op.emitOpError()
        << "sde.async_strategy cps_chain requires SDE CPS dataflow "
           "transformation before Core materialization";
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct ConvertSdeToArtsPass
    : public arts::impl::ConvertSdeToArtsBase<ConvertSdeToArtsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ARTS_INFO_HEADER(ConvertSdeToArtsPass);
    MLIRContext *context = &getContext();

    if (failed(rejectUnmaterializedCpsChain(module))) {
      signalPassFailure();
      return;
    }

    // Tensor-path lowering must run before the greedy driver
    // so its tensor.insert / tensor.extract body ops are cloned into EDTs
    // before region simplification can fold them away.
    if (failed(lowerTensorPathOps(module))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(context);
    // Process cu_task before mu_dep since cu_task consumes mu_dep results
    patterns.add<CuTaskToArtsPattern>(context);
    patterns.add<ParallelCuRegionToCorePattern>(context);
    patterns.add<CuRegionToArtsPattern>(context);
    patterns.add<SuDistributeToArtsPattern>(context);
    patterns.add<SuIterateToArtsPattern>(context);
    patterns.add<SuBarrierToArtsPattern>(context);
    patterns.add<ControlTokenToArtsPattern>(context);
    patterns.add<CuAtomicToArtsPattern>(context);
    patterns.add<CuReduceToArtsPattern>(context);
    patterns.add<ResourceQueryToArtsPattern>(context);
    patterns.add<SdeYieldToArtsPattern>(context);
    patterns.add<MuReductionDeclToArtsPattern>(context);
    // Tensor-path lowerings (mu_data, mu_token, cu_codelet).
    //
    // The codelet/mu_token/mu_data triple is handled ahead of the greedy
    // driver by `lowerTensorPathOps()` below. Running them as plain
    // OpRewritePatterns inside the greedy driver lets its region-
    // simplification pass DCE the codelet bodies between matches
    // (tensor-world ops are pure and look dead to the simplifier), which
    // would drop the very IR we're trying to lower.
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
    (void)applyPatternsGreedily(module, std::move(patterns), config);

    ARTS_INFO_FOOTER(ConvertSdeToArtsPass);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//

namespace mlir {
namespace arts {
namespace sde {
std::unique_ptr<Pass> createConvertSdeToArtsPass() {
  return std::make_unique<ConvertSdeToArtsPass>();
}
} // namespace sde
} // namespace arts
} // namespace mlir
