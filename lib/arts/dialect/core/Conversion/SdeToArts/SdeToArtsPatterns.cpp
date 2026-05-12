///==========================================================================///
/// File: SdeToArtsPatterns.cpp
///
/// Converts SDE dialect ops into ARTS dialect ops, producing the same IR
/// that ConvertOpenMPToArts currently generates.
///
/// Mapping:
///   sde.cu_region parallel  ->  arts.edt <parallel, internode>
///   sde.cu_region single    ->  arts.edt <single, intranode>
///   sde.cu_task             ->  arts.edt <task, intranode> + arts.db_control
///   sde.su_iterate          ->  arts.for
///   sde.su_barrier          ->  arts.barrier
///   sde.cu_atomic           ->  arts.atomic_add
///   sde.mu_dep              ->  arts.db_control
///   sde.yield               ->  arts.yield
///==========================================================================///

#include "arts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_CONVERTSDETOARTS
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts
#include "arts/passes/Passes.h"
#include "arts/utils/LoweringContractUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/Utils.h"
#include "arts/utils/ValueAnalysis.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
    "Number of sde.su_iterate converted to arts.for"};
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

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Map SDE schedule kind to ARTS ForScheduleKind.
static ForScheduleKindAttr
convertSchedule(MLIRContext *ctx, sde::SdeScheduleKindAttr sdeSchedule) {
  if (!sdeSchedule)
    return nullptr;
  switch (sdeSchedule.getValue()) {
  case sde::SdeScheduleKind::static_:
    return ForScheduleKindAttr::get(ctx, ForScheduleKind::Static);
  case sde::SdeScheduleKind::dynamic:
    return ForScheduleKindAttr::get(ctx, ForScheduleKind::Dynamic);
  case sde::SdeScheduleKind::guided:
    return ForScheduleKindAttr::get(ctx, ForScheduleKind::Guided);
  case sde::SdeScheduleKind::auto_:
    return ForScheduleKindAttr::get(ctx, ForScheduleKind::Auto);
  case sde::SdeScheduleKind::runtime:
    return ForScheduleKindAttr::get(ctx, ForScheduleKind::Runtime);
  }
  return nullptr;
}

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

/// Map an SDE reduction strategy to the persisted ARTS contract string.
static StringAttr
convertReductionStrategy(MLIRContext *ctx,
                         sde::SdeReductionStrategyAttr strategy) {
  if (!strategy)
    return nullptr;

  namespace ReductionStrategyValue =
      AttrNames::Operation::Contract::ReductionStrategyValue;
  switch (strategy.getValue()) {
  case sde::SdeReductionStrategy::atomic:
    return StringAttr::get(ctx, ReductionStrategyValue::Atomic);
  case sde::SdeReductionStrategy::tree:
    return StringAttr::get(ctx, ReductionStrategyValue::Tree);
  case sde::SdeReductionStrategy::local_accumulate:
    return StringAttr::get(ctx, ReductionStrategyValue::LocalAccumulate);
  }
  return nullptr;
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

/// Stamp a pattern contract on an arts.for and its parent EdtOp if present.
static void stampPatternContract(Operation *artsForOp, ArtsDepPattern pattern,
                                 int64_t revision = 1) {
  stampSimple(artsForOp, pattern, revision);
  if (auto parentEdt = artsForOp->getParentOfType<EdtOp>())
    stampSimple(parentEdt.getOperation(), pattern, revision);
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

static SmallVector<int64_t, 4> toSmallVector(ArrayRef<int64_t> values) {
  return SmallVector<int64_t, 4>(values.begin(), values.end());
}

static std::optional<SmallVector<int64_t, 4>>
projectPhysicalShapeToOwnerDims(ArrayRef<int64_t> physicalShape,
                                ArrayRef<int64_t> ownerDims) {
  if (physicalShape.empty())
    return SmallVector<int64_t, 4>{};
  if (ownerDims.empty() || physicalShape.size() == ownerDims.size())
    return toSmallVector(physicalShape);

  SmallVector<int64_t, 4> result;
  result.reserve(ownerDims.size());
  for (int64_t dim : ownerDims) {
    if (dim < 0 || static_cast<size_t>(dim) >= physicalShape.size())
      return std::nullopt;
    result.push_back(physicalShape[dim]);
  }
  return result;
}

/// Stamp a recovered stencil-family contract on an arts.for and its parent
/// EdtOp.
static void stampStencilContract(Operation *artsForOp, ArtsDepPattern pattern,
                                 const StructuredNeighborhoodSummary &info,
                                 ArrayRef<int64_t> planOwnerDims = {},
                                 ArrayRef<int64_t> planBlockShape = {},
                                 int64_t revision = 1) {
  SmallVector<int64_t, 4> ownerDims = planOwnerDims.empty()
                                          ? toSmallVector(info.ownerDims)
                                          : toSmallVector(planOwnerDims);
  SmallVector<int64_t, 4> minOffsets = toSmallVector(info.minOffsets);
  SmallVector<int64_t, 4> maxOffsets = toSmallVector(info.maxOffsets);
  SmallVector<int64_t, 4> writeFootprint = toSmallVector(info.writeFootprint);
  SmallVector<int64_t, 4> blockShape = toSmallVector(planBlockShape);

  if (!planOwnerDims.empty()) {
    auto projectedMins = projectOwnerIndexedStaticValues(
        info.minOffsets, info.ownerDims, planOwnerDims);
    auto projectedMaxs = projectOwnerIndexedStaticValues(
        info.maxOffsets, info.ownerDims, planOwnerDims);
    auto projectedFootprint = projectOwnerIndexedStaticValues(
        info.writeFootprint, info.ownerDims, planOwnerDims);

    if (projectedMins && projectedMaxs && projectedFootprint) {
      minOffsets = std::move(*projectedMins);
      maxOffsets = std::move(*projectedMaxs);
      writeFootprint = std::move(*projectedFootprint);
    } else {
      ownerDims = toSmallVector(info.ownerDims);
    }
  }

  if (auto projectedBlockShape =
          projectPhysicalShapeToOwnerDims(planBlockShape, ownerDims))
    blockShape = std::move(*projectedBlockShape);
  else if (blockShape.size() != ownerDims.size())
    blockShape.clear();

  stampStencilOnOp(artsForOp, pattern, ownerDims, minOffsets, maxOffsets,
                   writeFootprint, info.spatialDims, blockShape, revision);
  if (auto parentEdt = artsForOp->getParentOfType<EdtOp>())
    stampStencilOnOp(parentEdt.getOperation(), pattern, ownerDims, minOffsets,
                     maxOffsets, writeFootprint, info.spatialDims, blockShape,
                     revision);
}

static void copyPlanAttrsToArtsForAndParent(Operation *source,
                                            Operation *artsForOp) {
  copyPlanAttrs(source, artsForOp);
  if (auto parentEdt = artsForOp->getParentOfType<EdtOp>())
    copyPlanAttrs(source, parentEdt.getOperation());
}

static void stampDistributionKind(Operation *op, EdtDistributionKind kind,
                                  int64_t version = 1) {
  if (!op)
    return;

  setEdtDistributionKind(op, kind);
  op->setAttr(
      AttrNames::Operation::DistributionVersion,
      IntegerAttr::get(IntegerType::get(op->getContext(), 32), version));
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// sde.cu_region -> arts.edt
struct CuRegionToArtsPattern : public OpRewritePattern<sde::SdeCuRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeCuRegionOp op,
                                PatternRewriter &rewriter) const override {
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
      edtType = EdtType::parallel;
      concurrency = EdtConcurrency::internode;
      break;
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

    ++numCuRegionConverted;
    rewriter.eraseOp(op);
    return success();
  }
};

/// sde.cu_task -> arts.edt <task> with db_control deps
struct CuTaskToArtsPattern : public OpRewritePattern<sde::SdeCuTaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeCuTaskOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    // Convert SDE deps to ARTS db_control ops
    SmallVector<Value> artsDeps;
    for (Value dep : op.getDeps()) {
      auto muDep = dep.getDefiningOp<sde::SdeMuDepOp>();
      if (!muDep)
        continue;

      ArtsMode mode = convertAccessMode(muDep.getMode());
      SmallVector<Value> offsets(muDep.getOffsets().begin(),
                                 muDep.getOffsets().end());
      SmallVector<Value> sizes(muDep.getSizes().begin(),
                               muDep.getSizes().end());

      // Separate pinned dims (size=1) from chunk dims
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

      rewriter.setInsertionPoint(op);
      auto dbControl =
          DbControlOp::create(rewriter, loc, mode, muDep.getSource(),
                              pinnedIndices, chunkOffsets, blockSizes);
      artsDeps.push_back(dbControl);
    }

    auto edtOp = EdtOp::create(rewriter, loc, EdtType::task,
                               EdtConcurrency::intranode, artsDeps);
    edtOp.setNoVerifyAttr(NoVerifyAttr::get(ctx));

    Block &old = op.getBody().front();
    Block &blk = edtOp.getBody().front();
    blk.getOperations().splice(blk.end(), old.getOperations());

    ++numCuTaskConverted;
    rewriter.eraseOp(op);
    return success();
  }
};

/// arts_sde.su_distribute -> inline wrapped body
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

/// sde.su_iterate -> arts.for
/// Reads the SDE classification attrs stamped by StructuredSummaries and
/// stamps the matching ARTS contract on the resulting arts.for. The loop/memref
/// IR shape.
struct SuIterateToArtsPattern : public OpRewritePattern<sde::SdeSuIterateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeSuIterateOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto schedAttr = convertSchedule(ctx, op.getScheduleAttr());
    auto enclosingDistributeOp = op->getParentOfType<sde::SdeSuDistributeOp>();
    sde::SdeDistributionKindAttr enclosingDistribution =
        enclosingDistributeOp ? enclosingDistributeOp.getKindAttr() : nullptr;
    // Also check for distribution stamped directly on the su_iterate
    // (used when su_iterate has results and can't be wrapped in su_distribute).
    if (!enclosingDistribution) {
      if (auto attr = op->getAttrOfType<sde::SdeDistributionKindAttr>(
              "distributionKind"))
        enclosingDistribution = attr;
    }

    auto artsFor = ForOp::create(
        rewriter, loc, op.getLowerBounds(), op.getUpperBounds(), op.getSteps(),
        schedAttr, op.getChunkSize(), op.getReductionAccumulators());

    copyArtsMetadataAttrs(op, artsFor);
    if (auto strategyAttr =
            convertReductionStrategy(ctx, op.getReductionStrategyAttr())) {
      artsFor->setAttr(AttrNames::Operation::Contract::ReductionStrategy,
                       strategyAttr);
      if (auto parentEdt = artsFor->getParentOfType<EdtOp>();
          parentEdt && !parentEdt->hasAttr(
                           AttrNames::Operation::Contract::ReductionStrategy))
        parentEdt->setAttr(AttrNames::Operation::Contract::ReductionStrategy,
                           strategyAttr);
    }
    if (auto reductionKindsAttr = op.getReductionKindsAttr()) {
      // Convert SDE enum attrs to plain I32 array for consumption by core
      // passes without introducing an SDE dialect dependency.
      SmallVector<Attribute> intAttrs;
      for (auto attr : reductionKindsAttr) {
        if (auto enumAttr = dyn_cast<sde::SdeReductionKindAttr>(attr))
          intAttrs.push_back(rewriter.getI32IntegerAttr(
              static_cast<int32_t>(enumAttr.getValue())));
      }
      if (!intAttrs.empty())
        artsFor->setAttr(AttrNames::Operation::Contract::ReductionKinds,
                         rewriter.getArrayAttr(intAttrs));
    }

    // Move the body
    Region &dstRegion = artsFor.getRegion();
    Region &srcRegion = op.getBody();
    if (!srcRegion.empty()) {
      if (dstRegion.empty())
        dstRegion.push_back(new Block());
      Block &dst = dstRegion.front();
      Block &src = srcRegion.front();

      // Mirror the source IV list so cloned linalg/scf bodies keep using the
      // same induction variable ordering after SDE->ARTS conversion.
      while (dst.getNumArguments() < src.getNumArguments())
        dst.addArgument(rewriter.getIndexType(), loc);

      // Map block args and clone body
      IRMapping mapper;
      for (auto [srcArg, dstArg] :
           llvm::zip(src.getArguments(), dst.getArguments()))
        mapper.map(srcArg, dstArg);

      // Look through an optional cu_region <parallel> wrapper inside the
      // su_iterate body. Clone ops from the compute block, not the wrapper.
      Block *computeBlock = sde::getSuIterateComputeBlock(op);

      OpBuilder::InsertionGuard IG(rewriter);
      rewriter.setInsertionPointToStart(&dst);
      for (Operation &srcOp : computeBlock->without_terminator())
        rewriter.clone(srcOp, mapper);
      YieldOp::create(rewriter, loc);
    }

    // Route 1: Read stamped SDE classification attrs.
    std::optional<ArtsDepPattern> selectedPattern;
    std::optional<StructuredNeighborhoodSummary> selectedStencilContract;
    int64_t selectedRevision = 1;

    if (auto classAttr = op.getStructuredClassificationAttr()) {
      selectedPattern =
          mapStructuredClassificationToArtsDepPattern(classAttr.getValue());
      selectedRevision = getPatternRevision(op.getOperation()).value_or(1);
      if (selectedPattern && isStencilFamilyDepPattern(*selectedPattern))
        selectedStencilContract = getSdeNeighborhoodSummary(op);
    }

    if (selectedPattern) {
      ARTS_DEBUG("stamping SDE-derived contract: "
                 << stringifyArtsDepPattern(*selectedPattern));
      if (selectedStencilContract &&
          isStencilFamilyDepPattern(*selectedPattern)) {
        SmallVector<int64_t, 4> planOwnerDims;
        SmallVector<int64_t, 4> planBlockShape;
        if (auto ownerDims = readI64ArrayAttr(
                op.getOperation(), AttrNames::Operation::Plan::OwnerDims))
          planOwnerDims.assign(ownerDims->begin(), ownerDims->end());
        if (auto blockShape = readI64ArrayAttr(
                op.getOperation(),
                AttrNames::Operation::Plan::PhysicalBlockShape))
          planBlockShape.assign(blockShape->begin(), blockShape->end());
        stampStencilContract(artsFor.getOperation(), *selectedPattern,
                             *selectedStencilContract, planOwnerDims,
                             planBlockShape, selectedRevision);
      } else {
        stampPatternContract(artsFor.getOperation(), *selectedPattern,
                             selectedRevision);
      }
    }

    if (auto distributionKind =
            convertDistributionKind(enclosingDistribution)) {
      bool isInternode = false;
      if (auto parentEdt = artsFor.getOperation()->getParentOfType<EdtOp>())
        isInternode = (parentEdt.getConcurrency() == EdtConcurrency::internode);
      if (!isInternode) {
        stampDistributionKind(artsFor.getOperation(), *distributionKind);
        if (auto parentEdt = artsFor.getOperation()->getParentOfType<EdtOp>())
          stampDistributionKind(parentEdt.getOperation(), *distributionKind);
      }
    }

    copyPlanAttrsToArtsForAndParent(op.getOperation(), artsFor.getOperation());
    copySdeHintAttrs(op.getOperation(), artsFor.getOperation());
    if (auto workers = getWorkers(op.getOperation())) {
      setWorkers(artsFor.getOperation(), *workers);
      if (auto parentEdt = artsFor.getOperation()->getParentOfType<EdtOp>())
        setWorkers(parentEdt.getOperation(), *workers);
    }

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
    if (!op->hasAttr(AttrNames::Operation::BarrierEliminated))
      BarrierOp::create(rewriter, op.getLoc());
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

/// sde.mu_dep (standalone) -> arts.omp_dep
/// These are mu_deps not consumed by a cu_task — recreate as OmpDepOp
/// for downstream passes (CreateDbs, etc.) to consume.
struct MuDepToArtsPattern : public OpRewritePattern<sde::SdeMuDepOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sde::SdeMuDepOp op,
                                PatternRewriter &rewriter) const override {
    // Only convert standalone mu_deps (those consumed by cu_task are
    // handled by CuTaskToArtsPattern)
    for (Operation *user : op.getDep().getUsers()) {
      if (isa<sde::SdeCuTaskOp>(user))
        return failure(); // Let CuTaskToArtsPattern handle this
    }

    ArtsMode mode = convertAccessMode(op.getMode());
    SmallVector<Value> offsets(op.getOffsets().begin(), op.getOffsets().end());
    SmallVector<Value> sizes(op.getSizes().begin(), op.getSizes().end());

    auto ompDep = OmpDepOp::create(rewriter, op.getLoc(), mode, op.getSource(),
                                   offsets, sizes);
    rewriter.replaceOp(op, ompDep.getResult());
    return success();
  }
};

/// arts_sde.mu_reduction_decl -> semantic declaration only
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

/// Trace `tensor` back through `tensor.cast` / `tensor.extract_slice` /
/// `tensor.insert` ops to the underlying memref that backs the tensor view.
/// Returns nullptr if the tensor is not backed by a memref-view chain we
/// can lower (e.g., it came from a non-tensor-path producer).
static Value traceTensorToBackingMemref(Value tensor) {
  Value cur = tensor;
  for (;;) {
    if (auto cast = cur.getDefiningOp<tensor::CastOp>()) {
      cur = cast.getSource();
      continue;
    }
    if (auto insert = cur.getDefiningOp<tensor::InsertOp>()) {
      cur = insert.getDest();
      continue;
    }
    if (auto toTensor = cur.getDefiningOp<bufferization::ToTensorOp>())
      return toTensor.getBuffer();
    if (auto muToTensor = cur.getDefiningOp<sde::SdeMuMemrefToTensorOp>())
      return muToTensor.getMemref();
    return nullptr;
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
static bool fixupRank0Indices(SmallVectorImpl<Value> &indices,
                              MemRefType mrType, OpBuilder &builder,
                              Location loc) {
  if (indices.empty() && mrType.getRank() == 1)
    indices.push_back(arts::createZeroIndex(builder, loc));
  return static_cast<int64_t>(indices.size()) == mrType.getRank();
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
      Value memref = traceTensorToBackingMemref(extract.getTensor());
      if (!memref)
        continue;
      auto mrType = dyn_cast<MemRefType>(memref.getType());
      if (!mrType)
        continue;
      SmallVector<Value> indices(extract.getIndices());
      OpBuilder builder(extract);
      if (!fixupRank0Indices(indices, mrType, builder, extract.getLoc()))
        continue;
      Value loaded =
          memref::LoadOp::create(builder, extract.getLoc(), memref, indices);
      extract.getResult().replaceAllUsesWith(loaded);
      extract.erase();
    } else if (auto insert = dyn_cast<tensor::InsertOp>(op)) {
      Value memref = traceTensorToBackingMemref(insert.getDest());
      if (!memref)
        continue;
      auto mrType = dyn_cast<MemRefType>(memref.getType());
      if (!mrType)
        continue;
      SmallVector<Value> indices(insert.getIndices());
      OpBuilder builder(insert);
      if (!fixupRank0Indices(indices, mrType, builder, insert.getLoc()))
        continue;
      memref::StoreOp::create(builder, insert.getLoc(), insert.getScalar(),
                              memref, indices);
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

      Value source = traceTensorToBackingMemref(mat.getSource());
      if (!source)
        continue;
      Value dest = mat.getDest();
      if (source != dest) {
        OpBuilder builder(mat);
        if (failed(emitMemrefCopy(builder, mat.getLoc(), source, dest)))
          continue;
      }
      mat.erase();
    }
  }

  // Dead cleanup for view / cast / slice / extract_slice ops left over.
  SmallVector<Operation *> pending;
  module.walk([&](Operation *op) {
    if (isa<bufferization::ToTensorOp, tensor::CastOp, tensor::ExtractSliceOp,
            tensor::InsertSliceOp, tensor::EmptyOp, tensor::FromElementsOp>(op))
      pending.push_back(op);
  });
  // Iterate a couple of times to propagate dead-ness up a chain.
  for (int i = 0; i < 4; ++i) {
    bool anyErased = false;
    for (Operation *&op : pending) {
      if (!op)
        continue;
      if (op->use_empty()) {
        op->erase();
        op = nullptr;
        anyErased = true;
      }
    }
    if (!anyErased)
      break;
  }
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
    patterns.add<CuRegionToArtsPattern>(context);
    patterns.add<SuDistributeToArtsPattern>(context);
    patterns.add<SuIterateToArtsPattern>(context);
    patterns.add<SuBarrierToArtsPattern>(context);
    patterns.add<CuAtomicToArtsPattern>(context);
    patterns.add<CuReduceToArtsPattern>(context);
    patterns.add<SdeYieldToArtsPattern>(context);
    patterns.add<MuDepToArtsPattern>(context);
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
