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
///   sde.mu_dep              ->  invalid at the SDE/Core boundary
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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(convert_sde_to_arts);

#include "llvm/ADT/DenseMap.h"
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
using namespace mlir::carts;

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

  sde::SdeDistributionKindAttr distributionKindAttr =
      source.getDistributionKindAttr();
  if (!distributionKindAttr) {
    if (auto parentDistribute =
            source->getParentOfType<sde::SdeSuDistributeOp>())
      distributionKindAttr = parentDistribute.getKindAttr();
  }
  if (auto distributionKind = convertDistributionKind(distributionKindAttr))
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
  Value workers = arts::ValueAnalysis::castToIndex(runtimeWorkers, builder, loc);
  return arith::MaxUIOp::create(builder, loc, workers,
                                createOneIndex(builder, loc));
}

//===----------------------------------------------------------------------===//
// Boundary DB dependency helpers.
//
// SDE owns the decision that a scheduling unit/codelet communicates through
// MU-backed memrefs. At this boundary we only consume memrefs that are already
// backed by `sde.mu_alloc`/`sde.mu_data` lowering and pass acquired payload
// views into generated EDTs. Raw memref captures stay on the compatibility
// bridge until SDE has rewritten their MU storage and access shape explicitly.
//===----------------------------------------------------------------------===//

/// Trace the DbAllocOp that backs a storage value. Sources may already be
/// DB-backed payloads or may pass through memref view/cast operations.
static DbAllocOp findBackingDbAlloc(Value storage) {
  if (Operation *allocOp = DbUtils::getUnderlyingDbAlloc(storage))
    if (auto dbAlloc = dyn_cast<DbAllocOp>(allocOp))
      return dbAlloc;
  return nullptr;
}

/// Given a DB `sourcePtr` whose element type is the payload memref
/// `memref<SHAPExT>`, materialize that inner memref via `arts.db_ref[0]`.
static Value materializeInnerPayload(OpBuilder &builder, Location loc,
                                     Value sourcePtr) {
  Value zero = arts::createZeroIndex(builder, loc);
  return DbRefOp::create(builder, loc, sourcePtr, SmallVector<Value>{zero});
}

static FailureOr<SmallVector<Value>>
buildElementSizes(OpBuilder &builder, Location loc, MemRefType memrefType,
                  ValueRange dynamicSizes) {
  SmallVector<Value> elementSizes;
  if (memrefType.getRank() == 0) {
    if (!dynamicSizes.empty())
      return failure();
    elementSizes.push_back(arts::createOneIndex(builder, loc));
    return elementSizes;
  }

  elementSizes.reserve(memrefType.getRank());
  unsigned dynamicIdx = 0;
  for (int64_t dim = 0, rank = memrefType.getRank(); dim < rank; ++dim) {
    if (memrefType.isDynamicDim(dim)) {
      if (dynamicIdx >= dynamicSizes.size())
        return failure();
      elementSizes.push_back(dynamicSizes[dynamicIdx++]);
      continue;
    }
    elementSizes.push_back(
        arts::createConstantIndex(builder, loc, memrefType.getDimSize(dim)));
  }
  if (dynamicIdx != dynamicSizes.size())
    return failure();
  return elementSizes;
}

static LogicalResult
createDbBackedMemref(OpBuilder &builder, Location loc, MemRefType memrefType,
                     ValueRange dynamicSizes, Value &memref) {
  FailureOr<SmallVector<Value>> elementSizes =
      buildElementSizes(builder, loc, memrefType, dynamicSizes);
  if (failed(elementSizes))
    return failure();

  SmallVector<Value> sizes{arts::createOneIndex(builder, loc)};
  Value route = arts::createCurrentNodeRoute(builder, loc);
  Type pointerType = MemRefType::get({ShapedType::kDynamic}, memrefType);

  auto dbAlloc = DbAllocOp::create(
      builder, loc, ArtsMode::inout, route, DbAllocType::heap, DbMode::write,
      memrefType.getElementType(), pointerType, std::move(sizes),
      std::move(*elementSizes));

  memref = materializeInnerPayload(builder, loc, dbAlloc.getPtr());
  return success();
}

struct BoundaryMemrefAccess {
  Value payload;
  DbAllocOp dbAlloc;
  ArtsMode mode = ArtsMode::uninitialized;
};

struct BoundarySlicePlan {
  bool enabled = false;
  SmallVector<Value, 4> offsets;
  SmallVector<Value, 4> sizes;
};

struct LocalizedMemrefView {
  Value view;
  SmallVector<Value, 4> offsets;
};

static bool isDefinedInside(Operation *container, Value value) {
  if (!container || !value)
    return false;
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return container->isAncestor(blockArg.getOwner()->getParentOp());
  Operation *def = value.getDefiningOp();
  return def && container->isAncestor(def);
}

static void combineMemrefMode(DenseMap<Value, ArtsMode> &modes, Value root,
                              ArtsMode mode) {
  auto it = modes.find(root);
  if (it == modes.end()) {
    modes[root] = mode;
    return;
  }
  it->second = combineAccessModes(it->second, mode);
}

static DenseMap<Value, ArtsMode>
collectExternalMemrefAccessModes(Operation *container, Region &region) {
  DenseMap<Value, ArtsMode> modes;
  region.walk([&](Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      Value root = arts::ValueAnalysis::stripMemrefViewOps(load.getMemref());
      if (root && isa<MemRefType>(root.getType()) &&
          !isDefinedInside(container, root))
        combineMemrefMode(modes, root, ArtsMode::in);
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      Value root = arts::ValueAnalysis::stripMemrefViewOps(store.getMemref());
      if (root && isa<MemRefType>(root.getType()) &&
          !isDefinedInside(container, root))
        combineMemrefMode(modes, root, ArtsMode::out);
    }
  });
  return modes;
}

static FailureOr<SmallVector<BoundaryMemrefAccess>>
materializeBoundaryMemrefs(Operation *container, Region &region,
                           PatternRewriter &rewriter) {
  DenseMap<Value, ArtsMode> rootModes =
      collectExternalMemrefAccessModes(container, region);
  SmallVector<BoundaryMemrefAccess> accesses;
  accesses.reserve(rootModes.size());

  for (auto &entry : rootModes) {
    Value root = entry.first;
    ArtsMode mode = entry.second;

    DbAllocOp dbAlloc = findBackingDbAlloc(root);
    if (!dbAlloc)
      continue;

    accesses.push_back(BoundaryMemrefAccess{root, dbAlloc, mode});
  }

  return accesses;
}

static bool isZero(Value value) {
  int64_t constant = 0;
  return arts::ValueAnalysis::getConstantIndex(value, constant) && constant == 0;
}

static Value getMemrefDimValue(OpBuilder &builder, Location loc, Value memref,
                               unsigned dim) {
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType || dim >= static_cast<unsigned>(memrefType.getRank()))
    return Value();
  if (!memrefType.isDynamicDim(dim))
    return createConstantIndex(builder, loc, memrefType.getDimSize(dim));
  return memref::DimOp::create(builder, loc, memref, dim);
}

static std::optional<sde::SdeStructuredClassification>
getSdeClassification(sde::SdeSuIterateOp op) {
  if (auto classification = op.getStructuredClassification())
    return *classification;
  return std::nullopt;
}

static bool canUseOwnerSliceBoundaryPlan(sde::SdeSuIterateOp source) {
  if (!source || source.getLowerBounds().size() != 1 ||
      source.getUpperBounds().size() != 1 || source.getSteps().size() != 1)
    return false;

  auto classification = getSdeClassification(source);
  if (!classification)
    return false;

  switch (*classification) {
  case sde::SdeStructuredClassification::matmul:
  case sde::SdeStructuredClassification::elementwise:
  case sde::SdeStructuredClassification::elementwise_pipeline:
    return true;
  case sde::SdeStructuredClassification::stencil:
  case sde::SdeStructuredClassification::reduction:
    return false;
  }
  return false;
}

static bool indexSelectsOwnerSlice(Value index, Value ownerIv) {
  if (arts::ValueAnalysis::dependsOn(index, ownerIv))
    return true;

  auto blockArg = dyn_cast<BlockArgument>(index);
  if (!blockArg)
    return false;

  auto loop = dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!loop || loop.getInductionVar() != index)
    return false;

  return arts::ValueAnalysis::dependsOn(loop.getLowerBound(), ownerIv) ||
         arts::ValueAnalysis::dependsOn(loop.getUpperBound(), ownerIv);
}

static bool allRootAccessesUseOwnerFirstDim(sde::SdeSuIterateOp source,
                                            Value root) {
  if (!source || !root || source.getBody().empty())
    return false;

  auto rootType = dyn_cast<MemRefType>(root.getType());
  if (!rootType || rootType.getRank() == 0)
    return false;

  Block &body = source.getBody().front();
  if (body.getNumArguments() == 0)
    return false;
  Value ownerIv = body.getArgument(0);

  bool sawRootAccess = false;
  bool rejected = false;
  auto checkAccess = [&](Value memref, OperandRange indices) {
    if (arts::ValueAnalysis::stripMemrefViewOps(memref) != root)
      return;
    sawRootAccess = true;
    if (indices.empty() || !indexSelectsOwnerSlice(indices.front(), ownerIv))
      rejected = true;
  };

  source.getBody().walk([&](Operation *op) {
    if (rejected)
      return WalkResult::interrupt();
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (isa<MemRefType>(load.getResult().getType()))
        return WalkResult::advance();
      checkAccess(load.getMemref(), load.getIndices());
      return rejected ? WalkResult::interrupt() : WalkResult::advance();
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (isa<MemRefType>(store.getValueToStore().getType()))
        return WalkResult::advance();
      checkAccess(store.getMemref(), store.getIndices());
      return rejected ? WalkResult::interrupt() : WalkResult::advance();
    }
    return WalkResult::advance();
  });

  return sawRootAccess && !rejected;
}

static BoundarySlicePlan
buildOwnerSlicePlan(sde::SdeSuIterateOp source, BoundaryMemrefAccess access,
                    OpBuilder &builder, Location loc, Value globalBase,
                    Value iterCount) {
  BoundarySlicePlan plan;
  if (!canUseOwnerSliceBoundaryPlan(source) ||
      !allRootAccessesUseOwnerFirstDim(source, access.payload))
    return plan;

  auto memrefType = dyn_cast<MemRefType>(access.payload.getType());
  if (!memrefType || memrefType.getRank() == 0)
    return plan;

  plan.enabled = true;
  plan.offsets.reserve(memrefType.getRank());
  plan.sizes.reserve(memrefType.getRank());
  Value zero = createZeroIndex(builder, loc);
  Value ownerSpan =
      arith::MulIOp::create(builder, loc, iterCount, source.getSteps().front());
  Value remaining =
      arith::SubIOp::create(builder, loc, source.getUpperBounds().front(),
                            globalBase);
  Value ownerSize = arith::MinUIOp::create(builder, loc, ownerSpan, remaining);
  for (unsigned dim = 0; dim < static_cast<unsigned>(memrefType.getRank());
       ++dim) {
    if (dim == 0) {
      plan.offsets.push_back(globalBase);
      plan.sizes.push_back(ownerSize);
      continue;
    }
    plan.offsets.push_back(zero);
    Value dimSize = getMemrefDimValue(builder, loc, access.payload, dim);
    if (!dimSize)
      return BoundarySlicePlan{};
    plan.sizes.push_back(dimSize);
  }
  return plan;
}

static DbAcquireOp createBoundaryAcquire(OpBuilder &builder, Location loc,
                                         BoundaryMemrefAccess access,
                                         const BoundarySlicePlan &slicePlan) {
  SmallVector<Value> offsets{createZeroIndex(builder, loc)};
  SmallVector<Value> sizes{createOneIndex(builder, loc)};
  SmallVector<Value> partitionOffsets;
  SmallVector<Value> partitionSizes;
  std::optional<PartitionMode> partitionMode;
  if (slicePlan.enabled) {
    partitionMode = PartitionMode::block;
    partitionOffsets.assign(slicePlan.offsets.begin(), slicePlan.offsets.end());
    partitionSizes.assign(slicePlan.sizes.begin(), slicePlan.sizes.end());
  }

  return DbAcquireOp::create(
      builder, loc, access.mode == ArtsMode::uninitialized ? ArtsMode::inout
                                                           : access.mode,
      access.dbAlloc.getGuid(), access.dbAlloc.getPtr(), partitionMode,
      /*indices=*/SmallVector<Value>{}, std::move(offsets), std::move(sizes),
      /*partitionIndices=*/SmallVector<Value>{}, std::move(partitionOffsets),
      std::move(partitionSizes),
      /*boundsValid=*/Value{},
      /*elementOffsets=*/SmallVector<Value>{},
      /*elementSizes=*/SmallVector<Value>{});
}

static Value materializeBoundaryView(OpBuilder &builder, Location loc,
                                     Value payload,
                                     const BoundarySlicePlan &slicePlan) {
  (void)builder;
  (void)loc;
  (void)slicePlan;
  return payload;
}

struct TokenPayloadAccess {
  Value payload;
  SmallVector<Value, 4> offsets;
  SmallVector<Value, 4> sizes;
};

static void addOffsetsToMemrefIndices(OpBuilder &builder, Operation *op,
                                      MutableOperandRange indices,
                                      ArrayRef<Value> offsets) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);
  for (auto [idx, offset] : llvm::enumerate(offsets)) {
    if (idx >= indices.size() || !offset || isZero(offset))
      continue;
    Value current = indices[idx].get();
    indices[idx].assign(
        arith::AddIOp::create(builder, op->getLoc(), current, offset));
  }
}

static void rewriteSlicedTokenPayloadUses(
    ArrayRef<Operation *> clonedOps, ArrayRef<TokenPayloadAccess> payloads,
    OpBuilder &builder) {
  if (clonedOps.empty() || payloads.empty())
    return;

  auto findPayload = [&](Value memref) -> const TokenPayloadAccess * {
    for (const TokenPayloadAccess &payload : payloads)
      if (arts::ValueAnalysis::isDerivedFromPtr(memref, payload.payload))
        return &payload;
    return nullptr;
  };

  auto rewriteDim = [&](memref::DimOp dimOp,
                        const TokenPayloadAccess &payload) -> bool {
    std::optional<int64_t> constantIndex = dimOp.getConstantIndex();
    if (!constantIndex)
      return false;
    int64_t dim = *constantIndex;
    if (dim < 0 || static_cast<size_t>(dim) >= payload.sizes.size() ||
        !payload.sizes[dim])
      return false;
    dimOp.getResult().replaceAllUsesWith(payload.sizes[dim]);
    return true;
  };

  SmallVector<Operation *> erase;
  for (Operation *cloned : clonedOps) {
    cloned->walk([&](Operation *op) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        if (const TokenPayloadAccess *payload = findPayload(load.getMemref()))
          addOffsetsToMemrefIndices(builder, op, load.getIndicesMutable(),
                                    payload->offsets);
        return;
      }
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (const TokenPayloadAccess *payload = findPayload(store.getMemref()))
          addOffsetsToMemrefIndices(builder, op, store.getIndicesMutable(),
                                    payload->offsets);
        return;
      }
      if (auto dim = dyn_cast<memref::DimOp>(op)) {
        if (const TokenPayloadAccess *payload = findPayload(dim.getSource()))
          if (rewriteDim(dim, *payload))
            erase.push_back(dim);
      }
    });
  }

  for (Operation *op : llvm::reverse(erase))
    op->erase();
}

static void localizeClonedMemrefAccesses(
    Operation *rootOp, ArrayRef<LocalizedMemrefView> localizedViews,
    OpBuilder &builder) {
  if (!rootOp || localizedViews.empty())
    return;

  auto findView = [&](Value memref) -> const LocalizedMemrefView * {
    for (const LocalizedMemrefView &view : localizedViews)
      if (arts::ValueAnalysis::isDerivedFromPtr(memref, view.view))
        return &view;
    return nullptr;
  };

  auto localize = [&](Operation *op, MutableOperandRange indices,
                      Value memref) {
    const LocalizedMemrefView *view = findView(memref);
    if (!view)
      return;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);
    for (auto [idx, offset] : llvm::enumerate(view->offsets)) {
      if (idx >= indices.size() || !offset || isZero(offset))
        continue;
      Value current = indices[idx].get();
      indices[idx].assign(
          arith::SubIOp::create(builder, op->getLoc(), current, offset));
    }
  };

  rootOp->walk([&](Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      localize(op, load.getIndicesMutable(), load.getMemref());
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(op))
      localize(op, store.getIndicesMutable(), store.getMemref());
  });
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

  Value sourceRoot = arts::ValueAnalysis::stripMemrefViewOps(sourceAccumulator);
  if (sourceRoot && sourceRoot != sourceAccumulator)
    mapper.map(sourceRoot, replacement);

  Value materializedRoot =
      arts::ValueAnalysis::stripMemrefViewOps(materializedAccumulator);
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
    ArrayRef<Value> steps, IRMapping outerMapping, PatternRewriter &rewriter,
    ArrayRef<LocalizedMemrefView> localizedViews);

static LogicalResult buildTrailingLoopNestAndClone(
    sde::SdeSuIterateOp source, unsigned dim, ArrayRef<Value> lowerBounds,
    ArrayRef<Value> upperBounds, ArrayRef<Value> steps, IRMapping &mapper,
    PatternRewriter &rewriter, Block *computeBlock,
    ArrayRef<LocalizedMemrefView> localizedViews) {
  Location loc = source.getLoc();
  if (dim == lowerBounds.size()) {
    for (Operation &srcOp : computeBlock->without_terminator()) {
      Operation *cloned = rewriter.clone(srcOp, mapper);
      localizeClonedMemrefAccesses(cloned, localizedViews, rewriter);
    }
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
                                       computeBlock, localizedViews);
}

static LogicalResult cloneSuIterateBodyIntoLocalLoop(
    sde::SdeSuIterateOp source, scf::ForOp localLoop, Value globalBase,
    ArrayRef<Value> lowerBounds, ArrayRef<Value> upperBounds,
    ArrayRef<Value> steps, IRMapping outerMapping, PatternRewriter &rewriter,
    ArrayRef<LocalizedMemrefView> localizedViews) {
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
                                       computeBlock, localizedViews);
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

  FailureOr<SmallVector<BoundaryMemrefAccess>> maybeBoundaryMemrefs =
      materializeBoundaryMemrefs(source.getOperation(), source.getBody(),
                                 rewriter);
  if (failed(maybeBoundaryMemrefs))
    return failure();
  SmallVector<BoundaryMemrefAccess> boundaryMemrefs =
      std::move(*maybeBoundaryMemrefs);

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
  unsigned partialDependencyCount = taskDependencies.size();
  SmallVector<BoundarySlicePlan> boundarySlicePlans;
  boundarySlicePlans.reserve(boundaryMemrefs.size());
  for (const BoundaryMemrefAccess &access : boundaryMemrefs) {
    BoundarySlicePlan slicePlan =
        buildOwnerSlicePlan(source, access, rewriter, loc, globalBase,
                            iterCount);
    DbAcquireOp acquire =
        createBoundaryAcquire(rewriter, loc, access, slicePlan);
    boundarySlicePlans.push_back(std::move(slicePlan));
    taskDependencies.push_back(acquire.getPtr());
  }

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
    ArrayRef<BlockArgument> partialDependencyArgs(
        dependencyArgs.data(), partialDependencyCount);
    preparePartialReductionSlots(rewriter, loc, reductionPlan, task,
                                 partialDependencyArgs, taskMapping);
  }

  SmallVector<LocalizedMemrefView> localizedViews;
  for (auto [idx, access] : llvm::enumerate(boundaryMemrefs)) {
    BlockArgument depArg =
        dependencyArgs[partialDependencyCount + static_cast<unsigned>(idx)];
    Value payload = materializeInnerPayload(rewriter, loc, depArg);
    const BoundarySlicePlan &slicePlan = boundarySlicePlans[idx];
    Value view = materializeBoundaryView(rewriter, loc, payload, slicePlan);
    taskMapping.map(access.payload, view);
  }

  auto localLoop = scf::ForOp::create(rewriter, loc, zero, iterCount, one);
  if (failed(cloneSuIterateBodyIntoLocalLoop(source, localLoop, globalBase,
                                             lowerBounds, upperBounds, steps,
                                             taskMapping, rewriter,
                                             localizedViews)))
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

static LogicalResult rejectUnmaterializedCuTaskDeps(sde::SdeCuTaskOp op) {
  for (Value dep : op.getDeps()) {
    if (auto muDep = dep.getDefiningOp<sde::SdeMuDepOp>()) {
      InFlightDiagnostic diag =
          op.emitOpError("has unmaterialized sde.mu_dep at the SDE/Core "
                         "boundary; SDE must rewrite task dependencies into "
                         "mu_data/mu_token/cu_codelet form before conversion");
      diag.attachNote(muDep.getLoc()) << "dependency declared here";
      return failure();
    }
    return op.emitOpError()
           << "has unsupported dependency value " << dep
           << "; only SDE-materialized codelet tokens may reach Core";
  }
  return success();
}

static LogicalResult materializeCuTaskAsEdt(sde::SdeCuTaskOp op,
                                            PatternRewriter &rewriter,
                                            IRMapping &mapper) {
  if (failed(rejectUnmaterializedCuTaskDeps(op)))
    return failure();
  auto edtOp = EdtOp::create(rewriter, op.getLoc(), EdtType::task,
                             EdtConcurrency::intranode);
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

  if (isa<sde::SdeMuDepOp>(op))
    return success();

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
      sde::CartsSdeDialect::getDialectNamespace())
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
/// Raw `sde.mu_dep` values are not a Core contract. SDE must consume task
/// dependence declarations before this boundary and materialize
/// `mu_data`/`mu_token`/`cu_codelet` form when memory dependencies are needed.
struct CuTaskToArtsPattern : public OpRewritePattern<sde::SdeCuTaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeCuTaskOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    if (failed(rejectUnmaterializedCuTaskDeps(op)))
      return failure();

    rewriter.setInsertionPoint(op);
    auto edtOp = EdtOp::create(rewriter, loc, EdtType::task,
                               EdtConcurrency::intranode);
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

struct MuDepToArtsPattern : public OpRewritePattern<sde::SdeMuDepOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeMuDepOp op,
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
// MU/token SDE -> ARTS lowering.
//
// SDE is memref-native at this boundary:
//   - sde.mu_data     : memref<...>       -> arts.db_alloc + arts.db_ref
//   - sde.mu_alloc    : memref<...>       -> arts.db_alloc + arts.db_ref
//   - sde.mu_token    : !sde.token<memref<...>> -> arts.db_acquire
//   - sde.cu_codelet  : memref-token body -> arts.edt
//
// The SDE layer owns CU/SU/MU planning and any tiling/view rewrites. Core does
// not rediscover alternate state carriers or destination-passing results here; it only
// materializes the DB/acquire/EDT objects requested by SDE.
//
// DB shape convention: one logical DB per MU allocation with outer `sizes=[1]`
// and `elementSizes=<memref shape>`. The DB pointer type is explicit so
// `arts.db_ref %ptr[0]` has exactly the MU memref type selected by SDE, even
// when the element sizes are dynamic.
//===----------------------------------------------------------------------===//

static LogicalResult lowerMuData(sde::SdeMuDataOp op) {
  auto memrefType = dyn_cast<MemRefType>(op.getHandle().getType());
  if (!memrefType)
    return op.emitOpError()
           << "expects a memref handle before DB materialization, got "
           << op.getHandle().getType();
  if (memrefType.getNumDynamicDims() != 0)
    return op.emitOpError()
           << "has dynamic dimensions but carries no dynamic size operands; "
              "use sde.mu_alloc for dynamic MU storage";

  OpBuilder builder(op);
  Value replacement;
  if (failed(createDbBackedMemref(builder, op.getLoc(), memrefType,
                                  ValueRange{}, replacement)))
    return op.emitOpError() << "failed to build DB-backed memref";

  op.getHandle().replaceAllUsesWith(replacement);
  op.erase();
  return success();
}

static LogicalResult lowerMuAlloc(sde::SdeMuAllocOp op) {
  auto memrefType = dyn_cast<MemRefType>(op.getMemref().getType());
  if (!memrefType)
    return op.emitOpError()
           << "expects a memref result before DB materialization, got "
           << op.getMemref().getType();

  OpBuilder builder(op);
  Value replacement;
  if (failed(createDbBackedMemref(builder, op.getLoc(), memrefType,
                                  op.getDynamicSizes(), replacement)))
    return op.emitOpError()
           << "dynamic size operands do not match result memref shape";

  op.getMemref().replaceAllUsesWith(replacement);
  op.erase();
  return success();
}

static LogicalResult lowerCuCodelet(sde::SdeCuCodeletOp codelet) {
  Location loc = codelet.getLoc();
  OpBuilder rewriter(codelet);

  // Each codelet operand must be produced by a `sde.mu_token` whose source is
  // backed by a concrete DbAllocOp. If any token is not DB-backed we cannot
  // lower this codelet — bail out rather than producing half-converted IR.
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
          "token source is not backed by arts.db_alloc; SDE MU planning must "
          "produce db-backed memref storage for codelets to lower");
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
  acquirePtrs.reserve(muTokens.size());
  blockArgTypes.reserve(muTokens.size());
  blockArgLocs.reserve(muTokens.size());

  for (auto [idx, muToken] : llvm::enumerate(muTokens)) {
    ArtsMode mode = convertAccessMode(muToken.getMode());
    SmallVector<Value> partitionOffsets(muToken.getOffsets().begin(),
                                        muToken.getOffsets().end());
    SmallVector<Value> partitionSizes(muToken.getSizes().begin(),
                                      muToken.getSizes().end());

    DbAllocOp alloc = backingAllocs[idx];
    std::optional<PartitionMode> partitionMode;
    if (!partitionOffsets.empty() || !partitionSizes.empty())
      partitionMode = PartitionMode::block;
    // The DB's outer rank is 1 (one partition holding the whole storage);
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

  // Create the EDT. Task-intranode is the default codelet concurrency; SDE
  // does not encode runtime placement here.
  auto edtOp = EdtOp::create(rewriter, loc, EdtType::task,
                             EdtConcurrency::intranode, acquirePtrs);

  Block &edtBlock = edtOp.getBody().front();
  for (auto [argTy, argLoc] : llvm::zip(blockArgTypes, blockArgLocs))
    edtBlock.addArgument(argTy, argLoc);

  // Inside the EDT, materialize the storage view for each token block
  // argument. Memref tokens map directly to the DB payload or a memref.subview.
  OpBuilder::InsertionGuard bodyGuard(rewriter);
  rewriter.setInsertionPointToStart(&edtBlock);

  IRMapping mapper;
  Block &codeletBlock = codelet.getBody().front();
  unsigned numTokens = codelet.getTokens().size();
  SmallVector<TokenPayloadAccess> tokenPayloadAccesses;
  for (unsigned idx = 0; idx < numTokens; ++idx) {
    BlockArgument codeletArg = codeletBlock.getArgument(idx);
    auto memrefArgType = dyn_cast<MemRefType>(codeletArg.getType());
    if (!memrefArgType)
      return codelet.emitOpError()
             << "codelet block argument #" << idx
             << " must be a memref, got " << codeletArg.getType();
    Value edtArg = edtBlock.getArgument(idx);
    Value inner = materializeInnerPayload(rewriter, codelet.getLoc(), edtArg);
    mapper.map(codeletArg, inner);
    if (!muTokens[idx].getOffsets().empty() ||
        !muTokens[idx].getSizes().empty()) {
      TokenPayloadAccess access;
      access.payload = inner;
      llvm::append_range(access.offsets, muTokens[idx].getOffsets());
      llvm::append_range(access.sizes, muTokens[idx].getSizes());
      tokenPayloadAccesses.push_back(std::move(access));
    }
  }

  for (auto [idx, capture] : llvm::enumerate(codelet.getCaptures())) {
    BlockArgument codeletArg = codeletBlock.getArgument(numTokens + idx);
    mapper.map(codeletArg, capture);
  }

  // Clone codelet body ops (except the terminator) into the EDT body. The
  // terminator is a value-less `sde.yield`; memref token writes happen
  // in-place through the mapped block arguments.
  Operation *terminator = codeletBlock.getTerminator();
  SmallVector<Operation *> clonedOps;
  for (Operation &nested : codeletBlock.without_terminator()) {
    Operation *cloned = nested.clone(mapper);
    rewriter.insert(cloned);
    clonedOps.push_back(cloned);
  }

  rewriteSlicedTokenPayloadUses(clonedOps, tokenPayloadAccesses, rewriter);

  if (auto yieldOp = dyn_cast<sde::SdeYieldOp>(terminator)) {
    if (!yieldOp.getOperands().empty())
      return codelet.emitOpError()
             << "memref codelet yield must not carry replacement values";
  }

  YieldOp::create(rewriter, loc);

  // Erase the codelet and its mu_token producers.
  codelet.erase();
  for (sde::SdeMuTokenOp tok : muTokens)
    if (tok->use_empty())
      tok.erase();
  return success();
}

static LogicalResult lowerTokenSource(Value source,
                                      DenseSet<Operation *> &loweredMuOps) {
  Value root = arts::ValueAnalysis::stripMemrefViewOps(source);
  Operation *rootOp = root ? root.getDefiningOp() : nullptr;
  if (!rootOp || !loweredMuOps.insert(rootOp).second)
    return success();

  if (auto muData = dyn_cast<sde::SdeMuDataOp>(rootOp))
    return lowerMuData(muData);
  if (auto muAlloc = dyn_cast<sde::SdeMuAllocOp>(rootOp))
    return lowerMuAlloc(muAlloc);
  return success();
}

/// Drive memref MU lowerings ahead of the greedy rewriter. Order:
///   1. `sde.cu_codelet`  ->  arts.edt (acquires + body).
///   2. `sde.mu_token` leftovers (orphan or producer-less) -> erase.
///   3. `sde.mu_data` / `sde.mu_alloc` -> arts.db_alloc + db_ref.
///
/// Steps 1/2 before step 3 ensures that when we tear down a codelet, all of
/// its `sde.mu_token` operands are still resolvable back to their originating
/// `sde.mu_data` / `sde.mu_alloc` storage producer. Those producers are lowered
/// first for codelet operands so `findBackingDbAlloc` sees the DB chain.
static LogicalResult lowerMemrefMuOps(ModuleOp module) {
  DenseSet<Operation *> loweredMuOps;
  SmallVector<sde::SdeCuCodeletOp> codelets;
  module.walk([&](sde::SdeCuCodeletOp op) { codelets.push_back(op); });
  for (sde::SdeCuCodeletOp codelet : codelets) {
    for (Value token : codelet.getTokens()) {
      auto muToken = token.getDefiningOp<sde::SdeMuTokenOp>();
      if (!muToken)
        continue;
      if (failed(lowerTokenSource(muToken.getSource(), loweredMuOps)))
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

  // Step 3: any remaining MU storage ops still need lowering so
  // VerifySdeLowered rejects no residual SDE storage surface.
  SmallVector<sde::SdeMuDataOp> muDatas;
  module.walk([&](sde::SdeMuDataOp op) { muDatas.push_back(op); });
  for (sde::SdeMuDataOp op : muDatas) {
    if (failed(lowerMuData(op)))
      return failure();
  }

  SmallVector<sde::SdeMuAllocOp> muAllocs;
  module.walk([&](sde::SdeMuAllocOp op) { muAllocs.push_back(op); });
  for (sde::SdeMuAllocOp op : muAllocs) {
    if (failed(lowerMuAlloc(op)))
      return failure();
  }

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
    Value indexValue = arts::ValueAnalysis::castToIndex(runtimeQuery, rewriter,
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

    // Memref MU/codelet lowering must run before the greedy driver so SDE
    // token/codelet storage surfaces are materialized as DB/acquire/EDT
    // objects before generic region rewrites see them.
    if (failed(lowerMemrefMuOps(module))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(context);
    // Process cu_task before mu_dep since cu_task consumes mu_dep results.
    patterns.add<CuTaskToArtsPattern>(context);
    patterns.add<MuDepToArtsPattern>(context);
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
    // MU/codelet lowerings (mu_data, mu_alloc, mu_token, cu_codelet) are
    // handled ahead of the greedy driver by `lowerMemrefMuOps()`, because
    // they need to materialize a coordinated DB/acquire/EDT shape.
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
namespace carts {
namespace sde {
std::unique_ptr<Pass> createConvertSdeToArtsPass() {
  return std::make_unique<ConvertSdeToArtsPass>();
}
} // namespace sde
} // namespace carts
} // namespace mlir
