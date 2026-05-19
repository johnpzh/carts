///==========================================================================///
/// File: ArtsMaterializationUtils.h
///
/// CODIR-to-ARTS materialization helpers. This boundary is where generic CODIR
/// codelet/dependency plans become abstract ARTS DB, EDT, route, barrier, and
/// resource-query objects.
///==========================================================================///
#ifndef CARTS_DIALECT_CODIR_CONVERSION_CODIRTOARTS_ARTSMATERIALIZATIONUTILS_H
#define CARTS_DIALECT_CODIR_CONVERSION_CODIRTOARTS_ARTSMATERIALIZATIONUTILS_H

#include "../ConversionUtils.h"
#include "carts/dialect/arts/Utils/DbLayoutPlanUtils.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/RuntimeOpUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/STLExtras.h"

namespace {

static inline arts::ArtsMode convertAccessMode(codir::CodirAccessMode mode) {
  switch (mode) {
  case codir::CodirAccessMode::read:
    return arts::ArtsMode::in;
  case codir::CodirAccessMode::write:
    return arts::ArtsMode::out;
  case codir::CodirAccessMode::readwrite:
    return arts::ArtsMode::inout;
  }
  return arts::ArtsMode::inout;
}

static inline bool isReductionCodelet(codir::CodeletOp codelet) {
  auto pattern = codelet.getPatternAttr();
  return pattern && pattern.getValue() == codir::CodirPattern::reduction;
}

static inline bool
isAtomicAddAddressable(Value memref, ValueRange indices,
                       const DenseMap<Value, Value> &sourceByBlockArgument) {
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType)
    return false;
  if (memrefType.getRank() == 0)
    return indices.empty();
  if (memrefType.getRank() != 1 || indices.size() != 1)
    return false;
  return isKnownZeroIndex(indices.front(), sourceByBlockArgument);
}

static inline bool sameMemrefAccess(Value lhsMemref, ValueRange lhsIndices,
                                    Value rhsMemref, ValueRange rhsIndices) {
  return lhsMemref == rhsMemref &&
         ::mlir::carts::ValueAnalysis::areValueRangesIdentical(lhsIndices,
                                                               rhsIndices);
}

static inline unsigned lowerIntegerAddReductionsToAtomics(
    Region &region, const DenseMap<Value, Value> &sourceByBlockArgument) {
  SmallVector<memref::StoreOp, 8> stores;
  region.walk([&](memref::StoreOp store) { stores.push_back(store); });

  unsigned lowered = 0;
  for (memref::StoreOp store : stores) {
    auto add = store.getValue().getDefiningOp<arith::AddIOp>();
    if (!add || !add->hasOneUse())
      continue;

    memref::LoadOp load;
    Value increment;
    for (Value operand : add->getOperands()) {
      auto candidate = operand.getDefiningOp<memref::LoadOp>();
      if (!candidate)
        continue;
      if (!sameMemrefAccess(candidate.getMemref(), candidate.getIndices(),
                            store.getMemref(), store.getIndices()))
        continue;
      load = candidate;
      increment = add.getLhs() == operand ? add.getRhs() : add.getLhs();
      break;
    }
    if (!load || !load->hasOneUse() || !increment)
      continue;
    if (!isAtomicAddAddressable(store.getMemref(), store.getIndices(),
                                sourceByBlockArgument))
      continue;

    OpBuilder builder(store);
    arts::AtomicAddOp::create(builder, store.getLoc(), store.getMemref(),
                              increment);
    store.erase();
    if (add->use_empty())
      add.erase();
    if (load->use_empty())
      load.erase();
    ++lowered;
  }

  return lowered;
}

static inline arts::ArtsBarrierReason
convertBarrierReason(sde::SdeBarrierReason reason) {
  switch (reason) {
  case sde::SdeBarrierReason::redundant:
    return arts::ArtsBarrierReason::redundant;
  case sde::SdeBarrierReason::required_memory:
    return arts::ArtsBarrierReason::required_memory;
  case sde::SdeBarrierReason::timestep_stage_boundary:
    return arts::ArtsBarrierReason::timestep_stage_boundary;
  case sde::SdeBarrierReason::wavefront_frontier:
    return arts::ArtsBarrierReason::wavefront_frontier;
  case sde::SdeBarrierReason::unknown_required:
    return arts::ArtsBarrierReason::unknown_required;
  }
  return arts::ArtsBarrierReason::unknown_required;
}

static inline arts::ArtsBarrierReasonAttr
convertBarrierReasonAttr(MLIRContext *ctx, sde::SdeBarrierReasonAttr attr) {
  if (!attr)
    return {};
  return arts::ArtsBarrierReasonAttr::get(
      ctx, convertBarrierReason(attr.getValue()));
}

static inline arts::ArtsDepPattern convertPattern(codir::CodirPattern pattern) {
  switch (pattern) {
  case codir::CodirPattern::uniform:
    return arts::ArtsDepPattern::uniform;
  case codir::CodirPattern::stencil_tiling_nd:
    return arts::ArtsDepPattern::stencil_tiling_nd;
  case codir::CodirPattern::cross_dim_stencil_3d:
    return arts::ArtsDepPattern::cross_dim_stencil_3d;
  case codir::CodirPattern::higher_order_stencil:
    return arts::ArtsDepPattern::higher_order_stencil;
  case codir::CodirPattern::wavefront_2d:
    return arts::ArtsDepPattern::wavefront_2d;
  case codir::CodirPattern::jacobi_alternating_buffers:
    return arts::ArtsDepPattern::jacobi_alternating_buffers;
  case codir::CodirPattern::matmul:
    return arts::ArtsDepPattern::matmul;
  case codir::CodirPattern::elementwise_pipeline:
    return arts::ArtsDepPattern::elementwise_pipeline;
  case codir::CodirPattern::reduction:
    return arts::ArtsDepPattern::uniform;
  }
  return arts::ArtsDepPattern::unknown;
}

static inline arts::EdtDistributionKind
convertDistributionKind(codir::CodirDistributionKind kind) {
  switch (kind) {
  case codir::CodirDistributionKind::owner_compute:
    return arts::EdtDistributionKind::block;
  case codir::CodirDistributionKind::blocked:
    return arts::EdtDistributionKind::block;
  case codir::CodirDistributionKind::cyclic:
    return arts::EdtDistributionKind::block_cyclic;
  }
  return arts::EdtDistributionKind::block;
}

static inline arts::ArtsPlanIterationTopology
convertIterationTopology(codir::CodirIterationTopology topology) {
  switch (topology) {
  case codir::CodirIterationTopology::owner_strip:
    return arts::ArtsPlanIterationTopology::owner_strip;
  case codir::CodirIterationTopology::owner_tile:
    return arts::ArtsPlanIterationTopology::owner_tile;
  case codir::CodirIterationTopology::owner_tile_2d:
    return arts::ArtsPlanIterationTopology::owner_tile_2d;
  }
  return arts::ArtsPlanIterationTopology::owner_strip;
}

static inline arts::ArtsPlanRepetitionStructure
convertRepetitionStructure(codir::CodirRepetitionStructure structure) {
  switch (structure) {
  case codir::CodirRepetitionStructure::none:
    return arts::ArtsPlanRepetitionStructure::none;
  case codir::CodirRepetitionStructure::pair_step:
    return arts::ArtsPlanRepetitionStructure::pair_step;
  case codir::CodirRepetitionStructure::k_step:
    return arts::ArtsPlanRepetitionStructure::k_step;
  case codir::CodirRepetitionStructure::full_timestep:
    return arts::ArtsPlanRepetitionStructure::full_timestep;
  }
  return arts::ArtsPlanRepetitionStructure::none;
}

static inline arts::ArtsPlanAsyncStrategy
convertAsyncStrategy(codir::CodirAsyncStrategy strategy) {
  switch (strategy) {
  case codir::CodirAsyncStrategy::blocking:
    return arts::ArtsPlanAsyncStrategy::blocking;
  case codir::CodirAsyncStrategy::advance_stage:
  case codir::CodirAsyncStrategy::cps_chain:
    return arts::ArtsPlanAsyncStrategy::advance_edt;
  }
  return arts::ArtsPlanAsyncStrategy::blocking;
}

static inline arts::EdtDistributionPattern
getDistributionPattern(codir::CodirPattern pattern) {
  switch (pattern) {
  case codir::CodirPattern::uniform:
  case codir::CodirPattern::elementwise_pipeline:
    return arts::EdtDistributionPattern::uniform;
  case codir::CodirPattern::stencil_tiling_nd:
  case codir::CodirPattern::cross_dim_stencil_3d:
  case codir::CodirPattern::higher_order_stencil:
  case codir::CodirPattern::wavefront_2d:
  case codir::CodirPattern::jacobi_alternating_buffers:
    return arts::EdtDistributionPattern::stencil;
  case codir::CodirPattern::matmul:
    return arts::EdtDistributionPattern::matmul;
  case codir::CodirPattern::reduction:
    return arts::EdtDistributionPattern::uniform;
  }
  return arts::EdtDistributionPattern::unknown;
}

static inline void propagateCodirPlanToArts(codir::CodeletOp codelet,
                                            arts::EdtOp task) {
  if (!codelet || !task)
    return;
  MLIRContext *ctx = codelet.getContext();
  Operation *taskOp = task.getOperation();
  if (auto pattern = codelet.getPatternAttr()) {
    arts::ArtsDepPattern depPattern = convertPattern(pattern.getValue());
    if (depPattern != arts::ArtsDepPattern::unknown) {
      arts::setDepPattern(taskOp, depPattern);
      arts::setEdtDistributionPattern(
          taskOp, getDistributionPattern(pattern.getValue()));
      arts::setDistributionVersion(taskOp, 1);
      arts::setPatternRevision(taskOp, 1);
    }
  }
  if (auto kind = codelet.getDistributionKindAttr()) {
    arts::setEdtDistributionKind(taskOp,
                                 convertDistributionKind(kind.getValue()));
  }
  if (auto topology = codelet.getIterationTopologyAttr()) {
    arts::setPlanIterationTopologyAttr(
        taskOp, arts::ArtsPlanIterationTopologyAttr::get(
                    ctx, convertIterationTopology(topology.getValue())));
  }
  if (auto repetition = codelet.getRepetitionStructureAttr()) {
    arts::setPlanRepetitionStructureAttr(
        taskOp, arts::ArtsPlanRepetitionStructureAttr::get(
                    ctx, convertRepetitionStructure(repetition.getValue())));
  }
  if (auto async = codelet.getAsyncStrategyAttr()) {
    arts::setPlanAsyncStrategyAttr(
        taskOp, arts::ArtsPlanAsyncStrategyAttr::get(
                    ctx, convertAsyncStrategy(async.getValue())));
  }
  ArrayAttr tileShape = codelet.getTileShapeAttr();
  if (tileShape) {
    if (auto tileOwnerDims = codelet.getTileOwnerDimsAttr())
      arts::setPlanOwnerDimsAttr(taskOp, tileOwnerDims);
    arts::setPlanPhysicalBlockShapeAttr(taskOp, tileShape);
  } else if (auto ownerDims = codelet.getPlanOwnerDimsAttr()) {
    arts::setPlanOwnerDimsAttr(taskOp, ownerDims);
  }
  if (auto workerSlice = codelet.getLogicalWorkerSliceAttr())
    arts::setPlanLogicalWorkerSliceAttr(taskOp, workerSlice);
  if (auto haloShape = codelet.getHaloShapeAttr())
    arts::setPlanHaloShapeAttr(taskOp, haloShape);
  if (auto minOffsets = codelet.getAccessMinOffsetsAttr())
    task->setAttr(task.getStencilMinOffsetsAttrName(), minOffsets);
  if (auto maxOffsets = codelet.getAccessMaxOffsetsAttr())
    task->setAttr(task.getStencilMaxOffsetsAttrName(), maxOffsets);
  if (auto ownerDims = codelet.getPlanOwnerDimsAttr())
    task->setAttr(task.getStencilOwnerDimsAttrName(), ownerDims);
  if (auto spatialDims = codelet.getSpatialDimsAttr())
    task->setAttr(task.getStencilSpatialDimsAttrName(), spatialDims);
  if (auto writeFootprint = codelet.getWriteFootprintAttr())
    task->setAttr(task.getStencilWriteFootprintAttrName(), writeFootprint);
  if (codelet.getInPlaceSafeAttr())
    task.setInPlaceSafeAttr(UnitAttr::get(ctx));
  if (codelet.getInPlaceSharedStateAttr())
    task.setInPlaceSharedStateAttr(UnitAttr::get(ctx));
  if (auto pattern = codelet.getPatternAttr()) {
    if (pattern.getValue() == codir::CodirPattern::stencil_tiling_nd ||
        pattern.getValue() == codir::CodirPattern::cross_dim_stencil_3d ||
        pattern.getValue() == codir::CodirPattern::higher_order_stencil ||
        pattern.getValue() == codir::CodirPattern::wavefront_2d ||
        pattern.getValue() == codir::CodirPattern::jacobi_alternating_buffers) {
      if (codelet.getAccessMinOffsetsAttr() &&
          codelet.getAccessMaxOffsetsAttr())
        task->setAttr(task.getStencilSupportedBlockHaloAttrName(),
                      UnitAttr::get(ctx));
    }
  }
}

static inline FailureOr<SmallVector<Value>>
buildElementSizes(OpBuilder &builder, Location loc, MemRefType memrefType,
                  ValueRange dynamicSizes) {
  SmallVector<Value> elementSizes;
  if (memrefType.getRank() == 0) {
    if (!dynamicSizes.empty())
      return failure();
    elementSizes.push_back(createOneIndex(builder, loc));
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
        createConstantIndex(builder, loc, memrefType.getDimSize(dim)));
  }
  if (dynamicIdx != dynamicSizes.size())
    return failure();
  return elementSizes;
}

static inline MemRefType getElementMemRefType(Type elementType, unsigned rank) {
  SmallVector<int64_t> elementShape(rank, ShapedType::kDynamic);
  return MemRefType::get(elementShape, elementType);
}

static inline Value materializeInnerPayload(OpBuilder &builder, Location loc,
                                            Value sourcePtr) {
  Value zero = createZeroIndex(builder, loc);
  return arts::DbRefOp::create(builder, loc, sourcePtr,
                               SmallVector<Value>{zero});
}

static inline bool hasCodirTileOwnerSlicePlan(codir::CodeletOp op) {
  return op && op.getTileShapeAttr() && op.getTileOwnerDimsAttr();
}

static inline bool canUseCodirOwnerSliceForAlloc(codir::CodeletOp codelet,
                                                 arts::DbAllocOp alloc) {
  if (!hasCodirTileOwnerSlicePlan(codelet) || !alloc)
    return false;

  std::optional<arts::PartitionMode> mode =
      arts::getPartitionMode(alloc.getOperation());
  if (!mode || (*mode != arts::PartitionMode::block &&
                *mode != arts::PartitionMode::stencil))
    return false;

  return arts::getPlanOwnerDimsAttr(alloc.getOperation()) ==
             codelet.getTileOwnerDimsAttr() &&
         arts::getPlanPhysicalBlockShapeAttr(alloc.getOperation()) ==
             codelet.getTileShapeAttr();
}

static inline std::optional<unsigned>
getSingleCodirTileOwnerDim(codir::CodeletOp codelet) {
  if (!hasCodirTileOwnerSlicePlan(codelet))
    return std::nullopt;
  std::optional<SmallVector<int64_t, 4>> ownerDims =
      readI64ArrayAttr(codelet.getTileOwnerDimsAttr());
  if (!ownerDims || ownerDims->size() != 1 || ownerDims->front() < 0)
    return std::nullopt;
  return static_cast<unsigned>(ownerDims->front());
}

static inline std::optional<int64_t>
getSingleCodirTileOwnerBlockSize(codir::CodeletOp codelet,
                                 unsigned memrefRank) {
  std::optional<unsigned> ownerDim = getSingleCodirTileOwnerDim(codelet);
  if (!ownerDim)
    return std::nullopt;

  std::optional<SmallVector<int64_t, 4>> tileShape =
      readI64ArrayAttr(codelet.getTileShapeAttr());
  if (!tileShape || tileShape->empty())
    return std::nullopt;

  std::optional<int64_t> blockSize;
  if (tileShape->size() == memrefRank) {
    if (*ownerDim >= tileShape->size())
      return std::nullopt;
    blockSize = (*tileShape)[*ownerDim];
  } else {
    // Compact owner-slot-shaped tile metadata stores the single owner block
    // size at slot zero.
    blockSize = tileShape->front();
  }

  if (!blockSize || *blockSize <= 0)
    return std::nullopt;
  return blockSize;
}

static inline Value getCodirOwnerBaseArgument(codir::CodeletOp codelet) {
  if (!codelet || codelet.getBody().empty() || codelet.getParams().empty())
    return {};

  Block &body = codelet.getBody().front();
  unsigned depCount = codelet.getDeps().size();
  unsigned paramCount = codelet.getParams().size();
  if (body.getNumArguments() < depCount + paramCount)
    return {};
  return body.getArgument(depCount + paramCount - 1);
}

static inline bool
codirDepAccessesStayWithinSingleOwnerSlice(codir::CodeletOp codelet,
                                           unsigned depIndex) {
  std::optional<unsigned> ownerDim = getSingleCodirTileOwnerDim(codelet);
  if (!ownerDim || !codelet || codelet.getBody().empty())
    return false;

  Block &body = codelet.getBody().front();
  if (depIndex >= codelet.getDeps().size() ||
      depIndex >= body.getNumArguments())
    return false;

  Value depArg = body.getArgument(depIndex);
  auto depType = dyn_cast<MemRefType>(depArg.getType());
  if (!depType || depType.getRank() == 0 || *ownerDim >= depType.getRank())
    return false;

  Value ownerBase = getCodirOwnerBaseArgument(codelet);
  if (!ownerBase)
    return false;

  bool sawDirectRootAccess = false;
  bool rejected = false;
  body.walk([&](Operation *op) {
    if (rejected)
      return WalkResult::interrupt();

    auto access = getCodirMemoryAccessInfo(op);
    if (!access)
      return WalkResult::advance();
    if (access->memref != depArg)
      return WalkResult::advance();

    sawDirectRootAccess = true;
    if (access->indices.size() <= *ownerDim ||
        !indexSelectsOwnerSlice(access->indices[*ownerDim], ownerBase)) {
      rejected = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return sawDirectRootAccess && !rejected;
}

struct PlannedBlockLocalAccessRewrite {
  Value localMemref;
  unsigned ownerDim = 0;
  Value ownerBase;
};

static inline FailureOr<Value> materializeBlockLocalIndex(OpBuilder &builder,
                                                          Location loc,
                                                          Value index,
                                                          Value ownerBase) {
  if (!index || !ownerBase)
    return failure();
  if (::mlir::carts::ValueAnalysis::sameValue(index, ownerBase))
    return createZeroIndex(builder, loc);
  if (!indexSelectsOwnerSlice(index, ownerBase))
    return failure();
  return arith::SubIOp::create(builder, loc, index, ownerBase).getResult();
}

static inline LogicalResult rewritePlannedBlockLocalAccesses(
    arts::EdtOp task, ArrayRef<PlannedBlockLocalAccessRewrite> rewrites) {
  if (rewrites.empty())
    return success();

  auto rewriteIndices = [&](Operation *op, Value memref,
                            MutableOperandRange indices) -> WalkResult {
    for (const PlannedBlockLocalAccessRewrite &rewrite : rewrites) {
      if (memref != rewrite.localMemref)
        continue;
      if (indices.size() <= rewrite.ownerDim) {
        op->emitError("planned block-local access is missing the owner "
                      "dimension index");
        return WalkResult::interrupt();
      }

      OpBuilder builder(op);
      FailureOr<Value> localIndex = materializeBlockLocalIndex(
          builder, op->getLoc(), indices[rewrite.ownerDim].get(),
          rewrite.ownerBase);
      if (failed(localIndex)) {
        op->emitError("planned block-local access does not stay within the "
                      "owner slice");
        return WalkResult::interrupt();
      }
      indices[rewrite.ownerDim].set(*localIndex);
      return WalkResult::advance();
    }
    return WalkResult::advance();
  };

  Block &body = task.getBody().front();
  WalkResult result = body.walk([&](Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op))
      return rewriteIndices(op, load.getMemref(), load.getIndicesMutable());
    if (auto store = dyn_cast<memref::StoreOp>(op))
      return rewriteIndices(op, store.getMemref(), store.getIndicesMutable());
    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}

static inline LogicalResult
createDbBackedMemref(OpBuilder &builder, Location loc, MemRefType memrefType,
                     ValueRange dynamicSizes, Value &memref,
                     sde::SdeSuIterateOp planSource = {}) {
  FailureOr<SmallVector<Value>> elementSizes =
      buildElementSizes(builder, loc, memrefType, dynamicSizes);
  if (failed(elementSizes))
    return failure();

  SmallVector<Value> sizes{createOneIndex(builder, loc)};
  SmallVector<Value> dbElementSizes = std::move(*elementSizes);
  arts::PartitionMode partitionMode = arts::PartitionMode::coarse;
  if (planSource) {
    FailureOr<arts::DbPhysicalLayoutPlan> physicalPlan =
        arts::resolvePhysicalDbLayoutPlan(
            planSource.getPhysicalOwnerDimsAttr(),
            planSource.getPhysicalBlockShapeAttr(), dbElementSizes, builder,
            loc);
    if (failed(physicalPlan))
      return failure();
    sizes.assign(physicalPlan->outerSizes.begin(),
                 physicalPlan->outerSizes.end());
    dbElementSizes.assign(physicalPlan->innerSizes.begin(),
                          physicalPlan->innerSizes.end());
    partitionMode = physicalPlan->mode;
  }

  Value route = arts::createCurrentNodeRoute(builder, loc);
  arts::DbAllocOp dbAlloc;
  if (planSource) {
    dbAlloc = arts::DbAllocOp::create(
        builder, loc, arts::ArtsMode::inout, route, arts::DbAllocType::heap,
        arts::DbMode::write, memrefType.getElementType(), std::move(sizes),
        std::move(dbElementSizes), partitionMode);
    if (auto ownerDims = planSource.getPhysicalOwnerDimsAttr())
      arts::setPlanOwnerDimsAttr(dbAlloc.getOperation(), ownerDims);
    if (auto blockShape = planSource.getPhysicalBlockShapeAttr())
      arts::setPlanPhysicalBlockShapeAttr(dbAlloc.getOperation(), blockShape);
    if (auto workerSlice = planSource.getLogicalWorkerSliceAttr())
      arts::setPlanLogicalWorkerSliceAttr(dbAlloc.getOperation(), workerSlice);
    if (auto haloShape = planSource.getPhysicalHaloShapeAttr())
      arts::setPlanHaloShapeAttr(dbAlloc.getOperation(), haloShape);
  } else {
    Type pointerElementType =
        getElementMemRefType(memrefType.getElementType(), memrefType.getRank());
    Type pointerType =
        MemRefType::get({ShapedType::kDynamic}, pointerElementType);
    dbAlloc = arts::DbAllocOp::create(
        builder, loc, arts::ArtsMode::inout, route, arts::DbAllocType::heap,
        arts::DbMode::write, memrefType.getElementType(), pointerType,
        std::move(sizes), std::move(dbElementSizes), partitionMode);
  }

  memref = materializeInnerPayload(builder, loc, dbAlloc.getPtr());
  return success();
}

static inline LogicalResult
createDbBackedMemref(OpBuilder &builder, Location loc, MemRefType memrefType,
                     ValueRange dynamicSizes, Value &memref,
                     codir::CodeletOp planSource) {
  if (!hasCodirTileOwnerSlicePlan(planSource))
    return createDbBackedMemref(builder, loc, memrefType, dynamicSizes, memref,
                                sde::SdeSuIterateOp{});

  FailureOr<SmallVector<Value>> elementSizes =
      buildElementSizes(builder, loc, memrefType, dynamicSizes);
  if (failed(elementSizes))
    return failure();

  SmallVector<Value> dbElementSizes = std::move(*elementSizes);
  FailureOr<arts::DbPhysicalLayoutPlan> physicalPlan =
      arts::resolvePhysicalDbLayoutPlan(planSource.getTileOwnerDimsAttr(),
                                        planSource.getTileShapeAttr(),
                                        dbElementSizes, builder, loc);
  if (failed(physicalPlan))
    return failure();

  Value route = arts::createCurrentNodeRoute(builder, loc);
  auto dbAlloc = arts::DbAllocOp::create(
      builder, loc, arts::ArtsMode::inout, route, arts::DbAllocType::heap,
      arts::DbMode::write, memrefType.getElementType(),
      SmallVector<Value>(physicalPlan->outerSizes.begin(),
                         physicalPlan->outerSizes.end()),
      SmallVector<Value>(physicalPlan->innerSizes.begin(),
                         physicalPlan->innerSizes.end()),
      physicalPlan->mode);
  if (auto ownerDims = planSource.getTileOwnerDimsAttr())
    arts::setPlanOwnerDimsAttr(dbAlloc.getOperation(), ownerDims);
  if (auto blockShape = planSource.getTileShapeAttr())
    arts::setPlanPhysicalBlockShapeAttr(dbAlloc.getOperation(), blockShape);
  if (auto workerSlice = planSource.getLogicalWorkerSliceAttr())
    arts::setPlanLogicalWorkerSliceAttr(dbAlloc.getOperation(), workerSlice);
  if (auto haloShape = planSource.getHaloShapeAttr())
    arts::setPlanHaloShapeAttr(dbAlloc.getOperation(), haloShape);

  memref = materializeInnerPayload(builder, loc, dbAlloc.getPtr());
  return success();
}

static inline LogicalResult lowerMuData(sde::SdeMuDataOp op) {
  auto memrefType = dyn_cast<MemRefType>(op.getHandle().getType());
  if (!memrefType)
    return op.emitOpError()
           << "expects a memref handle before CODIR-to-ARTS materialization";
  if (memrefType.getNumDynamicDims() != 0)
    return op.emitOpError()
           << "has dynamic dimensions but carries no dynamic size operands";

  OpBuilder builder(op);
  Value replacement;
  if (failed(createDbBackedMemref(builder, op.getLoc(), memrefType,
                                  ValueRange{}, replacement)))
    return failure();

  op.getHandle().replaceAllUsesWith(replacement);
  op.erase();
  return success();
}

static inline LogicalResult lowerMuAlloc(sde::SdeMuAllocOp op) {
  auto memrefType = dyn_cast<MemRefType>(op.getMemref().getType());
  if (!memrefType)
    return op.emitOpError()
           << "expects a memref result before CODIR-to-ARTS materialization";

  OpBuilder builder(op);
  FailureOr<sde::SdeSuIterateOp> planSource = selectMuAllocWritePlan(op);
  if (failed(planSource))
    return failure();

  Value replacement;
  if (failed(createDbBackedMemref(builder, op.getLoc(), memrefType,
                                  op.getDynamicSizes(), replacement,
                                  *planSource)))
    return failure();

  op.getMemref().replaceAllUsesWith(replacement);
  op.erase();
  return success();
}

static inline arts::DbAllocOp findBackingDbAlloc(Value storage) {
  Value root = stripCodirViewOps(storage);
  Operation *def = root ? root.getDefiningOp() : nullptr;
  if (auto dbRef = dyn_cast_or_null<arts::DbRefOp>(def)) {
    Value source = dbRef.getSource();
    if (auto alloc = source.getDefiningOp<arts::DbAllocOp>())
      return alloc;
  }
  if (auto alloc = dyn_cast_or_null<arts::DbAllocOp>(def))
    return alloc;
  return nullptr;
}

static inline std::optional<unsigned>
findCodirDependencyIndexForRoot(codir::CodeletOp codelet, Value root) {
  for (auto [idx, dep] : llvm::enumerate(codelet.getDeps()))
    if (stripCodirViewOps(dep) == root)
      return static_cast<unsigned>(idx);
  return std::nullopt;
}

static inline std::optional<codir::CodirStorageViewKind>
getCodirDepStorageViewKind(codir::CodeletOp codelet, unsigned depIndex) {
  ArrayAttr views = codelet ? codelet.getDepStorageViewsAttr() : ArrayAttr{};
  if (!views || depIndex >= views.size())
    return std::nullopt;
  auto view = dyn_cast<codir::CodirStorageViewKindAttr>(views[depIndex]);
  if (!view)
    return std::nullopt;
  return view.getValue();
}

static inline std::optional<codir::CodirAccessMode>
getCodirDepAccessMode(codir::CodeletOp codelet, unsigned depIndex) {
  ArrayAttr modes = codelet ? codelet.getDepModesAttr() : ArrayAttr{};
  if (!modes || depIndex >= modes.size())
    return std::nullopt;
  auto mode = dyn_cast<codir::CodirAccessModeAttr>(modes[depIndex]);
  if (!mode)
    return std::nullopt;
  return mode.getValue();
}

static inline bool codirDepAllowsComputeBlockStorage(codir::CodeletOp codelet,
                                                     unsigned depIndex) {
  std::optional<codir::CodirStorageViewKind> view =
      getCodirDepStorageViewKind(codelet, depIndex);
  if (!view)
    return true;
  return *view == codir::CodirStorageViewKind::compute_block;
}

static inline bool codirDepRequiresComputeBlockStorage(codir::CodeletOp codelet,
                                                       unsigned depIndex) {
  std::optional<codir::CodirStorageViewKind> view =
      getCodirDepStorageViewKind(codelet, depIndex);
  return view && *view == codir::CodirStorageViewKind::compute_block;
}

static inline bool codirAccessMayWrite(codir::CodirAccessMode mode) {
  return mode == codir::CodirAccessMode::write ||
         mode == codir::CodirAccessMode::readwrite;
}

static inline bool codirAccessMayRead(codir::CodirAccessMode mode) {
  return mode == codir::CodirAccessMode::read ||
         mode == codir::CodirAccessMode::readwrite;
}

static inline Operation *
findCodirDispatchBridgeAnchor(codir::CodeletOp codelet) {
  Operation *anchor = codelet ? codelet.getOperation() : nullptr;
  for (Operation *parent = codelet ? codelet->getParentOp() : nullptr; parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (loop && containsValue(codelet.getParams(), loop.getInductionVar()))
      anchor = parent;
  }
  return anchor;
}

struct HostBridgeParticipant {
  codir::CodeletOp codelet;
  unsigned depIndex = 0;
  codir::CodirAccessMode mode = codir::CodirAccessMode::readwrite;
};

struct HostBridgeUseCollection {
  SmallVector<HostBridgeParticipant> participants;
  SmallVector<Operation *> readObservationAnchors;
};

static inline std::optional<unsigned>
getCodeletDepOperandIndex(codir::CodeletOp codelet, OpOperand &use);

static inline bool isUseInsideAnchor(Operation *anchor, Operation *owner) {
  if (!anchor || !owner)
    return false;
  if (anchor == owner)
    return true;
  for (Region &region : anchor->getRegions())
    if (region.isAncestor(owner->getParentRegion()))
      return true;
  return false;
}

static inline bool hostBridgeValueMayBeWrittenInsideAnchor(Operation *anchor,
                                                           Value value);

static inline bool hostBridgeUseMayWrite(Operation *anchor, OpOperand &use) {
  Operation *owner = use.getOwner();
  if (!anchor || !owner)
    return true;
  if (!isUseInsideAnchor(anchor, owner))
    return false;

  if (auto codelet = dyn_cast<codir::CodeletOp>(owner)) {
    std::optional<unsigned> depIndex = getCodeletDepOperandIndex(codelet, use);
    if (!depIndex)
      return true;
    std::optional<codir::CodirAccessMode> mode =
        getCodirDepAccessMode(codelet, *depIndex);
    return !mode || codirAccessMayWrite(*mode);
  }

  if (auto store = dyn_cast<memref::StoreOp>(owner))
    return store.getMemRef() == use.get();
  if (auto store = dyn_cast<polygeist::DynStoreOp>(owner))
    return store.getMemref() == use.get();
  if (auto store = dyn_cast<affine::AffineStoreOp>(owner))
    return store.getMemRef() == use.get();
  if (isa<memref::DeallocOp>(owner))
    return true;
  if (isa<memref::LoadOp, memref::DimOp>(owner))
    return false;

  if (isMemrefForwardingOp(owner)) {
    for (Value result : owner->getResults())
      if (isa<MemRefType>(result.getType()) &&
          hostBridgeValueMayBeWrittenInsideAnchor(anchor, result))
        return true;
    return false;
  }

  return true;
}

static inline bool hostBridgeValueMayBeWrittenInsideAnchor(Operation *anchor,
                                                           Value value) {
  if (!anchor || !value)
    return true;
  for (OpOperand &use : value.getUses())
    if (hostBridgeUseMayWrite(anchor, use))
      return true;
  return false;
}

static inline Operation *findHostBridgeReadObservationAnchor(OpOperand &use) {
  Operation *owner = use.getOwner();
  if (!owner)
    return nullptr;
  if (auto codelet = dyn_cast<codir::CodeletOp>(owner))
    if (Operation *anchor = findCodirDispatchBridgeAnchor(codelet))
      return anchor;
  return owner;
}

static inline void
appendUniqueHostBridgeReadObservation(Operation *anchor,
                                      SmallVectorImpl<Operation *> &anchors) {
  if (!anchor || llvm::is_contained(anchors, anchor))
    return;
  anchors.push_back(anchor);
}

static inline Operation *findHostBridgeEventUnderAnchor(Operation *anchor,
                                                        Operation *op) {
  if (!anchor || !op)
    return nullptr;
  if (anchor == op)
    return op;

  Operation *event = op;
  while (event && event->getParentOp() != anchor) {
    Operation *parent = event->getParentOp();
    if (!parent)
      return nullptr;
    event = parent;
  }
  return event;
}

static inline SmallVector<Operation *> filterHostBridgeReadSyncAnchors(
    Operation *anchor, ArrayRef<HostBridgeParticipant> participants,
    ArrayRef<Operation *> readObservationAnchors) {
  if (!anchor || readObservationAnchors.empty())
    return SmallVector<Operation *>{readObservationAnchors.begin(),
                                    readObservationAnchors.end()};

  struct Event {
    Operation *eventOp = nullptr;
    Operation *syncAnchor = nullptr;
    bool write = false;
    bool readObservation = false;
    unsigned ordinal = 0;
  };

  SmallVector<Event> events;
  unsigned ordinal = 0;
  for (const HostBridgeParticipant &participant : participants) {
    if (!codirAccessMayWrite(participant.mode))
      continue;
    Operation *dispatchAnchor =
        findCodirDispatchBridgeAnchor(participant.codelet);
    Operation *event =
        findHostBridgeEventUnderAnchor(anchor, dispatchAnchor);
    if (!event)
      return SmallVector<Operation *>{readObservationAnchors.begin(),
                                      readObservationAnchors.end()};
    events.push_back({event, nullptr, /*write=*/true,
                      /*readObservation=*/false, ordinal++});
  }

  for (Operation *syncAnchor : readObservationAnchors) {
    Operation *event = findHostBridgeEventUnderAnchor(anchor, syncAnchor);
    if (!event)
      return SmallVector<Operation *>{readObservationAnchors.begin(),
                                      readObservationAnchors.end()};
    events.push_back({event, syncAnchor, /*write=*/false,
                      /*readObservation=*/true, ordinal++});
  }

  Block *eventBlock = nullptr;
  for (const Event &event : events) {
    if (!event.eventOp || !event.eventOp->getBlock())
      return SmallVector<Operation *>{readObservationAnchors.begin(),
                                      readObservationAnchors.end()};
    if (!eventBlock) {
      eventBlock = event.eventOp->getBlock();
      continue;
    }
    if (eventBlock != event.eventOp->getBlock())
      return SmallVector<Operation *>{readObservationAnchors.begin(),
                                      readObservationAnchors.end()};
  }

  llvm::sort(events, [](const Event &lhs, const Event &rhs) {
    if (lhs.eventOp == rhs.eventOp)
      return lhs.ordinal < rhs.ordinal;
    return lhs.eventOp->isBeforeInBlock(rhs.eventOp);
  });

  SmallVector<Operation *> filtered;
  bool blockDirtyForHost = false;
  for (const Event &event : events) {
    if (event.write)
      blockDirtyForHost = true;
    if (event.readObservation && blockDirtyForHost) {
      appendUniqueHostBridgeReadObservation(event.syncAnchor, filtered);
      blockDirtyForHost = false;
    }
  }
  return filtered;
}

static inline bool
hostBridgeNeedsInitialCopyIn(Operation *anchor,
                             ArrayRef<HostBridgeParticipant> participants) {
  bool hasReadParticipant = llvm::any_of(
      participants, [](const HostBridgeParticipant &participant) {
        return codirAccessMayRead(participant.mode);
      });
  if (!hasReadParticipant)
    return false;
  if (!anchor)
    return true;

  struct Event {
    Operation *eventOp = nullptr;
    codir::CodirAccessMode mode = codir::CodirAccessMode::readwrite;
    unsigned ordinal = 0;
  };

  SmallVector<Event> events;
  unsigned ordinal = 0;
  for (const HostBridgeParticipant &participant : participants) {
    Operation *dispatchAnchor =
        findCodirDispatchBridgeAnchor(participant.codelet);
    Operation *event =
        findHostBridgeEventUnderAnchor(anchor, dispatchAnchor);
    if (!event)
      return true;
    events.push_back({event, participant.mode, ordinal++});
  }

  Block *eventBlock = nullptr;
  for (const Event &event : events) {
    if (!event.eventOp || !event.eventOp->getBlock())
      return true;
    if (!eventBlock) {
      eventBlock = event.eventOp->getBlock();
      continue;
    }
    if (eventBlock != event.eventOp->getBlock())
      return true;
  }

  llvm::sort(events, [](const Event &lhs, const Event &rhs) {
    if (lhs.eventOp == rhs.eventOp)
      return lhs.ordinal < rhs.ordinal;
    return lhs.eventOp->isBeforeInBlock(rhs.eventOp);
  });

  for (const Event &event : events) {
    if (codirAccessMayRead(event.mode))
      return true;
    if (codirAccessMayWrite(event.mode))
      return false;
  }
  return true;
}

static inline bool hasSameHostBridgePlan(codir::CodeletOp lhs,
                                         codir::CodeletOp rhs) {
  if (!lhs || !rhs)
    return false;
  return lhs.getTileOwnerDimsAttr() == rhs.getTileOwnerDimsAttr() &&
         lhs.getTileShapeAttr() == rhs.getTileShapeAttr() &&
         lhs.getLogicalWorkerSliceAttr() == rhs.getLogicalWorkerSliceAttr();
}

static inline std::optional<unsigned>
getCodeletDepOperandIndex(codir::CodeletOp codelet, OpOperand &use) {
  if (!codelet)
    return std::nullopt;
  unsigned operandIndex = use.getOperandNumber();
  if (operandIndex >= codelet.getDeps().size())
    return std::nullopt;
  return operandIndex;
}

static inline bool isCompatibleHostBridgeParticipant(codir::CodeletOp seed,
                                                     codir::CodeletOp codelet,
                                                     unsigned depIndex) {
  if (!seed || !codelet || !hasSameHostBridgePlan(seed, codelet))
    return false;
  if (!codirDepRequiresComputeBlockStorage(codelet, depIndex))
    return false;
  if (!hasCodirTileOwnerSlicePlan(codelet) ||
      !codirDepAccessesStayWithinSingleOwnerSlice(codelet, depIndex))
    return false;
  return true;
}

static inline FailureOr<HostBridgeUseCollection>
collectHostBridgeParticipants(Operation *anchor, codir::CodeletOp seed,
                              Value hostView) {
  if (!anchor || !seed || !hostView)
    return failure();

  HostBridgeUseCollection collection;
  for (OpOperand &use : hostView.getUses()) {
    Operation *owner = use.getOwner();
    if (!isUseInsideAnchor(anchor, owner))
      continue;

    auto codelet = dyn_cast<codir::CodeletOp>(owner);
    if (!codelet) {
      if (!hostBridgeUseMayWrite(anchor, use)) {
        appendUniqueHostBridgeReadObservation(
            findHostBridgeReadObservationAnchor(use),
            collection.readObservationAnchors);
        continue;
      }
      return failure();
    }

    std::optional<unsigned> depIndex = getCodeletDepOperandIndex(codelet, use);
    if (!depIndex ||
        !isCompatibleHostBridgeParticipant(seed, codelet, *depIndex)) {
      if (!hostBridgeUseMayWrite(anchor, use)) {
        appendUniqueHostBridgeReadObservation(
            findHostBridgeReadObservationAnchor(use),
            collection.readObservationAnchors);
        continue;
      }
      return failure();
    }

    std::optional<codir::CodirAccessMode> mode =
        getCodirDepAccessMode(codelet, *depIndex);
    if (!mode)
      return failure();
    collection.participants.push_back({codelet, *depIndex, *mode});
  }

  if (collection.participants.empty())
    return failure();
  return collection;
}

static inline bool canHoistHostBridgeAcrossLoop(scf::ForOp loop,
                                                codir::CodeletOp codelet,
                                                Value hostView) {
  if (!loop || !codelet || !hostView)
    return false;
  if (containsValue(codelet.getParams(), loop.getInductionVar()))
    return false;

  Region &loopRegion = loop.getRegion();
  for (OpOperand &use : hostView.getUses()) {
    Operation *owner = use.getOwner();
    if (!owner || !loopRegion.isAncestor(owner->getParentRegion()))
      continue;
    auto userCodelet = dyn_cast<codir::CodeletOp>(owner);
    std::optional<unsigned> depIndex =
        userCodelet ? getCodeletDepOperandIndex(userCodelet, use)
                    : std::nullopt;
    if (depIndex &&
        isCompatibleHostBridgeParticipant(codelet, userCodelet, *depIndex))
      continue;
    if (!hostBridgeUseMayWrite(loop.getOperation(), use))
      continue;
    return false;
  }
  return true;
}

static inline Operation *findCodirHostBridgeAnchor(codir::CodeletOp codelet,
                                                   Value hostView) {
  Operation *anchor = findCodirDispatchBridgeAnchor(codelet);
  if (!anchor)
    return nullptr;

  for (Operation *parent = anchor->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      continue;
    if (!canHoistHostBridgeAcrossLoop(loop, codelet, hostView))
      break;
    anchor = loop.getOperation();
  }
  return anchor;
}

static inline SmallVector<Value>
getBridgeLogicalElementSizes(OpBuilder &builder, Location loc, Value hostView) {
  if (arts::DbAllocOp hostAlloc = findBackingDbAlloc(hostView))
    return SmallVector<Value>(hostAlloc.getElementSizes().begin(),
                              hostAlloc.getElementSizes().end());

  SmallVector<Value> sizes;
  auto memrefType = dyn_cast<MemRefType>(hostView.getType());
  if (!memrefType)
    return sizes;
  sizes.reserve(memrefType.getRank());
  for (int64_t dim = 0, rank = memrefType.getRank(); dim < rank; ++dim) {
    if (memrefType.isDynamicDim(dim)) {
      sizes.push_back(memref::DimOp::create(builder, loc, hostView, dim));
      continue;
    }
    sizes.push_back(
        createConstantIndex(builder, loc, memrefType.getDimSize(dim)));
  }
  return sizes;
}

static inline void materializeHostBlockElementCopyNest(
    OpBuilder &builder, Location loc, Value hostView, Value blockPayload,
    ArrayRef<Value> copySizes, Value ownerOffset, unsigned ownerDim,
    bool copyIntoBlock, SmallVectorImpl<Value> &indices) {
  unsigned dim = indices.size();
  if (dim == copySizes.size()) {
    SmallVector<Value> hostIndices;
    SmallVector<Value> blockIndices;
    hostIndices.reserve(indices.size());
    blockIndices.reserve(indices.size());
    for (auto [idx, value] : llvm::enumerate(indices)) {
      blockIndices.push_back(value);
      if (idx == ownerDim) {
        hostIndices.push_back(
            arith::AddIOp::create(builder, loc, ownerOffset, value));
        continue;
      }
      hostIndices.push_back(value);
    }

    if (copyIntoBlock) {
      Value loaded =
          memref::LoadOp::create(builder, loc, hostView, hostIndices);
      memref::StoreOp::create(builder, loc, loaded, blockPayload, blockIndices);
      return;
    }
    Value loaded =
        memref::LoadOp::create(builder, loc, blockPayload, blockIndices);
    memref::StoreOp::create(builder, loc, loaded, hostView, hostIndices);
    return;
  }

  Value zero = createZeroIndex(builder, loc);
  Value one = createOneIndex(builder, loc);
  auto loop = scf::ForOp::create(builder, loc, zero, copySizes[dim], one);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  indices.push_back(loop.getInductionVar());
  materializeHostBlockElementCopyNest(builder, loc, hostView, blockPayload,
                                      copySizes, ownerOffset, ownerDim,
                                      copyIntoBlock, indices);
  indices.pop_back();
}

static inline FailureOr<Value>
materializeCoarseHostDbForBlockArgument(OpBuilder &builder, Location loc,
                                        BlockArgument blockArg) {
  Value root = blockArg;
  auto memrefType = dyn_cast<MemRefType>(root.getType());
  if (!memrefType)
    return failure();

  Block *owner = blockArg.getOwner();
  if (!owner)
    return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(owner);

  SmallVector<Value> elementSizes;
  if (memrefType.getRank() == 0) {
    elementSizes.push_back(createOneIndex(builder, loc));
  } else {
    elementSizes.reserve(memrefType.getRank());
    for (int64_t dim = 0, rank = memrefType.getRank(); dim < rank; ++dim) {
      if (memrefType.isDynamicDim(dim)) {
        elementSizes.push_back(memref::DimOp::create(builder, loc, root, dim));
        continue;
      }
      elementSizes.push_back(
          createConstantIndex(builder, loc, memrefType.getDimSize(dim)));
    }
  }

  SmallVector<Operation *> protectedOps;
  for (Value size : elementSizes)
    if (Operation *op = size.getDefiningOp())
      protectedOps.push_back(op);

  Value route = arts::createCurrentNodeRoute(builder, loc);
  auto dbAlloc = arts::DbAllocOp::create(
      builder, loc, arts::ArtsMode::inout, route, arts::DbAllocType::unknown,
      arts::DbMode::write, memrefType.getElementType(), root,
      SmallVector<Value>{createOneIndex(builder, loc)}, std::move(elementSizes),
      arts::PartitionMode::coarse);
  protectedOps.push_back(dbAlloc.getOperation());

  Value replacement = materializeInnerPayload(builder, loc, dbAlloc.getPtr());
  root.replaceUsesWithIf(replacement, [&](OpOperand &use) {
    return !llvm::is_contained(protectedOps, use.getOwner());
  });
  return replacement;
}

static inline LogicalResult
materializeHostBlockCopyLoop(OpBuilder &builder, Location loc, Value hostView,
                             arts::DbAllocOp blockAlloc,
                             codir::CodeletOp codelet, bool copyIntoBlock) {
  auto hostType = dyn_cast<MemRefType>(hostView.getType());
  if (!hostType || hostType.getRank() == 0)
    return failure();

  std::optional<unsigned> ownerDim = getSingleCodirTileOwnerDim(codelet);
  if (!ownerDim || *ownerDim >= static_cast<unsigned>(hostType.getRank()))
    return failure();
  if (blockAlloc.getSizes().size() != 1 ||
      blockAlloc.getElementSizes().size() !=
          static_cast<size_t>(hostType.getRank()))
    return failure();

  SmallVector<Value> logicalSizes =
      getBridgeLogicalElementSizes(builder, loc, hostView);
  if (logicalSizes.size() != static_cast<size_t>(hostType.getRank()))
    return failure();

  Value zero = createZeroIndex(builder, loc);
  Value one = createOneIndex(builder, loc);
  Value blockCount = blockAlloc.getSizes().front();
  Value ownerBlockSize = blockAlloc.getElementSizes()[*ownerDim];

  auto loop = scf::ForOp::create(builder, loc, zero, blockCount, one);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  Value blockIndex = loop.getInductionVar();
  Value ownerOffset =
      arith::MulIOp::create(builder, loc, blockIndex, ownerBlockSize);
  Value ownerRemaining =
      arith::SubIOp::create(builder, loc, logicalSizes[*ownerDim], ownerOffset);
  Value ownerCopySize =
      arith::MinUIOp::create(builder, loc, ownerBlockSize, ownerRemaining);

  SmallVector<Value> copySizes;
  copySizes.reserve(hostType.getRank());
  for (int64_t dim = 0, rank = hostType.getRank(); dim < rank; ++dim) {
    if (static_cast<unsigned>(dim) == *ownerDim) {
      copySizes.push_back(ownerCopySize);
    } else {
      copySizes.push_back(logicalSizes[dim]);
    }
  }

  Value blockPayload = arts::DbRefOp::create(builder, loc, blockAlloc.getPtr(),
                                             SmallVector<Value>{blockIndex});
  SmallVector<Value> indices;
  materializeHostBlockElementCopyNest(builder, loc, hostView, blockPayload,
                                      copySizes, ownerOffset, *ownerDim,
                                      copyIntoBlock, indices);
  return success();
}

static inline FailureOr<Value>
materializeHostWholeToComputeBlockBridge(codir::CodeletOp codelet,
                                         unsigned depIndex, Value hostView) {
  if (!codelet || depIndex >= codelet.getDeps().size() || !hostView)
    return failure();
  if (!codirDepRequiresComputeBlockStorage(codelet, depIndex))
    return failure();
  if (!hasCodirTileOwnerSlicePlan(codelet) ||
      !codirDepAccessesStayWithinSingleOwnerSlice(codelet, depIndex))
    return failure();

  std::optional<codir::CodirAccessMode> depMode =
      getCodirDepAccessMode(codelet, depIndex);
  if (!depMode)
    return failure();

  auto memrefType = dyn_cast<MemRefType>(hostView.getType());
  if (!memrefType || memrefType.getRank() == 0 || isCodirViewDep(hostView))
    return failure();

  Operation *anchor = findCodirHostBridgeAnchor(codelet, hostView);
  if (!anchor)
    return failure();

  SmallVector<HostBridgeParticipant> participants;
  SmallVector<Operation *> readObservationAnchors;
  FailureOr<HostBridgeUseCollection> collected =
      collectHostBridgeParticipants(anchor, codelet, hostView);
  if (succeeded(collected)) {
    participants = std::move(collected->participants);
    readObservationAnchors = std::move(collected->readObservationAnchors);
  } else {
    participants.push_back({codelet, depIndex, *depMode});
  }

  bool needsCopyIn = hostBridgeNeedsInitialCopyIn(anchor, participants);
  bool needsCopyOut = llvm::any_of(participants, [](const auto &participant) {
    return codirAccessMayWrite(participant.mode);
  });

  OpBuilder builder(anchor);
  Location loc = codelet.getLoc();
  builder.setInsertionPoint(anchor);
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> logicalSizes =
      getBridgeLogicalElementSizes(builder, loc, hostView);
  if (logicalSizes.size() != static_cast<size_t>(memrefType.getRank()))
    return failure();
  for (int64_t dim = 0, rank = memrefType.getRank(); dim < rank; ++dim) {
    if (memrefType.isDynamicDim(dim))
      dynamicSizes.push_back(logicalSizes[dim]);
  }

  Value blockView;
  if (failed(createDbBackedMemref(builder, loc, memrefType, dynamicSizes,
                                  blockView, codelet)))
    return failure();
  arts::DbAllocOp blockAlloc = findBackingDbAlloc(blockView);
  if (!blockAlloc)
    return failure();
  blockAlloc->setAttr("arts.storage_bridge",
                      builder.getStringAttr("host_whole_to_compute_block"));

  if (needsCopyIn) {
    if (failed(materializeHostBlockCopyLoop(builder, loc, hostView, blockAlloc,
                                            codelet,
                                            /*copyIntoBlock=*/true)))
      return failure();
  }

  if (needsCopyOut) {
    for (HostBridgeParticipant &participant : participants) {
      if (codirAccessMayWrite(participant.mode))
        participant.codelet.setCompletionBarrierAttr(
            UnitAttr::get(participant.codelet->getContext()));
    }
    SmallVector<Operation *> syncAnchors =
        filterHostBridgeReadSyncAnchors(anchor, participants,
                                        readObservationAnchors);
    for (Operation *observationAnchor : syncAnchors) {
      if (!observationAnchor || !isUseInsideAnchor(anchor, observationAnchor))
        continue;
      builder.setInsertionPoint(observationAnchor);
      if (failed(materializeHostBlockCopyLoop(builder, loc, hostView,
                                              blockAlloc, codelet,
                                              /*copyIntoBlock=*/false)))
        return failure();
    }
    builder.setInsertionPointAfter(anchor);
    if (failed(materializeHostBlockCopyLoop(builder, loc, hostView, blockAlloc,
                                            codelet,
                                            /*copyIntoBlock=*/false)))
      return failure();
  }

  for (HostBridgeParticipant &participant : participants)
    participant.codelet->setOperand(participant.depIndex, blockView);
  return blockView;
}

static inline LogicalResult
materializeExistingDbHostBridgeIfNeeded(codir::CodeletOp codelet,
                                        unsigned depIndex) {
  if (!codelet || depIndex >= codelet.getDeps().size())
    return failure();

  Value dep = codelet.getDeps()[depIndex];
  arts::DbAllocOp hostAlloc = findBackingDbAlloc(dep);
  if (!hostAlloc)
    return success();
  if (!codirDepRequiresComputeBlockStorage(codelet, depIndex))
    return success();
  if (!hasCodirTileOwnerSlicePlan(codelet) || isCodirViewDep(dep))
    return success();
  if (canUseCodirOwnerSliceForAlloc(codelet, hostAlloc))
    return success();

  std::optional<arts::PartitionMode> hostMode =
      arts::getPartitionMode(hostAlloc.getOperation());
  if (!hostMode || *hostMode != arts::PartitionMode::coarse)
    return success();

  FailureOr<Value> replacement =
      materializeHostWholeToComputeBlockBridge(codelet, depIndex, dep);
  return failed(replacement) ? failure() : success();
}

static inline bool
canMaterializeRawCodirDependencyWithPlan(Value root,
                                         codir::CodeletOp planSource) {
  if (!hasCodirTileOwnerSlicePlan(planSource))
    return false;
  std::optional<unsigned> depIndex =
      findCodirDependencyIndexForRoot(planSource, root);
  if (!depIndex)
    return false;
  if (!codirDepAllowsComputeBlockStorage(planSource, *depIndex))
    return false;
  return codirDepAccessesStayWithinSingleOwnerSlice(planSource, *depIndex);
}

static inline bool rawCodirDependencyNeedsHostBridge(Value root) {
  if (!root)
    return false;
  if (isa<BlockArgument>(root))
    return true;
  return hasHostMemrefAccessOutsideSchedulingUnit(root);
}

static inline LogicalResult
materializeRawCodirDependency(Value dep, codir::CodeletOp planSource,
                              unsigned depIndex) {
  if (findBackingDbAlloc(dep))
    return success();

  Value root = stripCodirViewOps(dep);
  auto memrefType = dyn_cast<MemRefType>(root.getType());
  if (!memrefType)
    return failure();

  Operation *def = root.getDefiningOp();
  OpBuilder builder(root.getContext());
  Value replacement;
  bool usePlan = canMaterializeRawCodirDependencyWithPlan(root, planSource);
  if (usePlan && codirDepRequiresComputeBlockStorage(planSource, depIndex) &&
      rawCodirDependencyNeedsHostBridge(root)) {
    if (auto blockArg = dyn_cast<BlockArgument>(root)) {
      FailureOr<Value> hostView = materializeCoarseHostDbForBlockArgument(
          builder, root.getLoc(), blockArg);
      if (failed(hostView))
        return failure();
      FailureOr<Value> bridged = materializeHostWholeToComputeBlockBridge(
          planSource, depIndex, *hostView);
      return failed(bridged) ? failure() : success();
    }

    FailureOr<Value> bridged =
        materializeHostWholeToComputeBlockBridge(planSource, depIndex, root);
    if (failed(bridged))
      return failure();
    return success();
  }
  if (!def) {
    auto blockArg = dyn_cast<BlockArgument>(root);
    if (!blockArg)
      return failure();

    Block *owner = blockArg.getOwner();
    if (!owner)
      return failure();
    builder.setInsertionPointToStart(owner);

    SmallVector<Value> elementSizes;
    SmallVector<Value> dynamicSizes;
    if (memrefType.getRank() == 0) {
      elementSizes.push_back(createOneIndex(builder, root.getLoc()));
    } else {
      elementSizes.reserve(memrefType.getRank());
      for (int64_t dim = 0, rank = memrefType.getRank(); dim < rank; ++dim) {
        if (memrefType.isDynamicDim(dim)) {
          Value size = memref::DimOp::create(builder, root.getLoc(), root, dim);
          elementSizes.push_back(size);
          dynamicSizes.push_back(size);
        } else {
          elementSizes.push_back(createConstantIndex(
              builder, root.getLoc(), memrefType.getDimSize(dim)));
        }
      }
    }

    SmallVector<Operation *> dimOps;
    for (Value size : elementSizes)
      if (Operation *op = size.getDefiningOp())
        dimOps.push_back(op);

    arts::DbAllocOp createdDbAlloc;
    if (usePlan) {
      if (failed(createDbBackedMemref(builder, root.getLoc(), memrefType,
                                      dynamicSizes, replacement, planSource)))
        return failure();
    } else {
      Value route = arts::createCurrentNodeRoute(builder, root.getLoc());
      auto dbAlloc = arts::DbAllocOp::create(
          builder, root.getLoc(), arts::ArtsMode::inout, route,
          arts::DbAllocType::unknown, arts::DbMode::write,
          memrefType.getElementType(), root,
          SmallVector<Value>{createOneIndex(builder, root.getLoc())},
          std::move(elementSizes), arts::PartitionMode::coarse);
      createdDbAlloc = dbAlloc;
      replacement =
          materializeInnerPayload(builder, root.getLoc(), dbAlloc.getPtr());
    }

    root.replaceUsesWithIf(replacement, [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      if (createdDbAlloc && owner == createdDbAlloc.getOperation())
        return false;
      return !llvm::is_contained(dimOps, owner);
    });
    return success();
  }

  builder.setInsertionPointAfter(def);
  if (auto alloc = dyn_cast<memref::AllocOp>(def)) {
    if (failed(usePlan
                   ? createDbBackedMemref(builder, alloc.getLoc(), memrefType,
                                          alloc.getDynamicSizes(), replacement,
                                          planSource)
                   : createDbBackedMemref(builder, alloc.getLoc(), memrefType,
                                          alloc.getDynamicSizes(),
                                          replacement)))
      return failure();
  } else if (auto alloca = dyn_cast<memref::AllocaOp>(def)) {
    if (failed(usePlan
                   ? createDbBackedMemref(builder, alloca.getLoc(), memrefType,
                                          alloca.getDynamicSizes(), replacement,
                                          planSource)
                   : createDbBackedMemref(builder, alloca.getLoc(), memrefType,
                                          alloca.getDynamicSizes(),
                                          replacement)))
      return failure();
  } else {
    return failure();
  }

  SmallVector<memref::DeallocOp> deallocs;
  for (Operation *user : llvm::make_early_inc_range(root.getUsers())) {
    auto dealloc = dyn_cast<memref::DeallocOp>(user);
    if (dealloc && dealloc.getMemref() == root)
      deallocs.push_back(dealloc);
  }
  for (memref::DeallocOp dealloc : deallocs)
    dealloc.erase();

  root.replaceAllUsesWith(replacement);
  if (def->use_empty())
    def->erase();
  return success();
}

struct CodirDepSlice {
  bool sliced = false;
  bool subindex = false;
  Value subindexIndex;
  SmallVector<Value> offsets;
  SmallVector<Value> sizes;
  SmallVector<OpFoldResult> mixedOffsets;
  SmallVector<OpFoldResult> mixedSizes;
  SmallVector<OpFoldResult> mixedStrides;
};

static inline CodirDepSlice getCodirDepSlice(Value dep, OpBuilder &builder,
                                             Location loc) {
  CodirDepSlice slice;
  auto subview = dep.getDefiningOp<memref::SubViewOp>();
  if (subview) {
    slice.sliced = true;
    slice.mixedOffsets = subview.getMixedOffsets();
    slice.mixedSizes = subview.getMixedSizes();
    slice.mixedStrides = subview.getMixedStrides();
    slice.offsets =
        materializeIndexFoldResults(builder, loc, slice.mixedOffsets);
    slice.sizes = materializeIndexFoldResults(builder, loc, slice.mixedSizes);
    return slice;
  }

  auto subindex = dep.getDefiningOp<polygeist::SubIndexOp>();
  if (!subindex)
    return slice;

  auto sourceType = dyn_cast<MemRefType>(subindex.getSource().getType());
  auto resultType = dyn_cast<MemRefType>(subindex.getResult().getType());
  if (!sourceType || !resultType || sourceType.getRank() == 0 ||
      resultType.getRank() + 1 != sourceType.getRank())
    return slice;

  slice.sliced = true;
  slice.subindex = true;
  slice.subindexIndex = subindex.getIndex();
  slice.mixedOffsets.push_back(subindex.getIndex());
  slice.mixedSizes.push_back(builder.getIndexAttr(1));
  slice.mixedStrides.push_back(builder.getIndexAttr(1));

  ValueRange dynamicSizes = subindex.getSizes();
  unsigned dynamicSizeIdx = 0;
  for (int64_t dim = 0, rank = resultType.getRank(); dim < rank; ++dim) {
    slice.mixedOffsets.push_back(builder.getIndexAttr(0));
    if (resultType.isDynamicDim(dim)) {
      if (dynamicSizeIdx >= dynamicSizes.size()) {
        slice.sliced = false;
        slice.mixedOffsets.clear();
        slice.mixedSizes.clear();
        slice.mixedStrides.clear();
        return slice;
      }
      slice.mixedSizes.push_back(dynamicSizes[dynamicSizeIdx++]);
    } else {
      slice.mixedSizes.push_back(
          builder.getIndexAttr(resultType.getDimSize(dim)));
    }
    slice.mixedStrides.push_back(builder.getIndexAttr(1));
  }
  slice.offsets = materializeIndexFoldResults(builder, loc, slice.mixedOffsets);
  slice.sizes = materializeIndexFoldResults(builder, loc, slice.mixedSizes);
  return slice;
}

static inline LogicalResult lowerSdeResourceQuery(sde::SdeResourceQueryOp op) {
  OpBuilder builder(op);
  switch (op.getKind()) {
  case sde::SdeResourceQueryKind::logicalWorkers: {
    auto runtimeQuery = arts::RuntimeQueryOp::create(
        builder, op.getLoc(), arts::RuntimeQueryKind::totalWorkers);
    Value asIndex = arith::IndexCastOp::create(
        builder, op.getLoc(), builder.getIndexType(), runtimeQuery.getResult());
    op.getResult().replaceAllUsesWith(asIndex);
    op.erase();
    return success();
  }
  }
  return op.emitOpError() << "unsupported SDE resource query kind";
}

static inline LogicalResult lowerSdeControlBarrier(sde::SdeSuBarrierOp op) {
  OpBuilder builder(op);
  if (!op.getBarrierEliminatedAttr()) {
    arts::BarrierOp::create(
        builder, op.getLoc(),
        convertBarrierReasonAttr(op.getContext(), op.getBarrierReasonAttr()));
  }
  op.erase();
  return success();
}

static inline LogicalResult
eraseConsumedSdeControlToken(sde::SdeControlTokenOp op) {
  if (!op.getToken().use_empty())
    return op.emitOpError()
           << "survived CODIR-to-ARTS materialization with live users; "
              "control tokens must be consumed by SDE barriers before the "
              "ARTS boundary";
  op.erase();
  return success();
}

} // namespace

#endif // CARTS_DIALECT_CODIR_CONVERSION_CODIRTOARTS_ARTSMATERIALIZATIONUTILS_H
