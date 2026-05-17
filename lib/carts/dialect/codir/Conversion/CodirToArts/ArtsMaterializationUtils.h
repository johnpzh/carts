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
#include "carts/utils/StencilAttributes.h"
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
      arts::setEdtDistributionPattern(taskOp,
                                      getDistributionPattern(pattern.getValue()));
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
    task->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::
                      FootprintMinOffsets,
                  minOffsets);
  if (auto maxOffsets = codelet.getAccessMaxOffsetsAttr())
    task->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::
                      FootprintMaxOffsets,
                  maxOffsets);
  if (auto ownerDims = codelet.getPlanOwnerDimsAttr())
    task->setAttr(
        ::mlir::carts::StencilAttrNames::Operation::Stencil::OwnerDims,
        ownerDims);
  if (auto spatialDims = codelet.getSpatialDimsAttr())
    task->setAttr(
        ::mlir::carts::StencilAttrNames::Operation::Stencil::SpatialDims,
        spatialDims);
  if (auto writeFootprint = codelet.getWriteFootprintAttr())
    task->setAttr(
        ::mlir::carts::StencilAttrNames::Operation::Stencil::WriteFootprint,
        writeFootprint);
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
        task->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::
                          SupportedBlockHalo,
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

static inline LogicalResult
materializeRawCodirDependency(Value dep, codir::CodeletOp planSource) {
  if (findBackingDbAlloc(dep))
    return success();

  Value root = stripCodirViewOps(dep);
  auto memrefType = dyn_cast<MemRefType>(root.getType());
  if (!memrefType)
    return failure();

  Operation *def = root.getDefiningOp();
  OpBuilder builder(root.getContext());
  Value replacement;
  if (!def) {
    auto blockArg = dyn_cast<BlockArgument>(root);
    if (!blockArg)
      return failure();

    Block *owner = blockArg.getOwner();
    if (!owner)
      return failure();
    builder.setInsertionPointToStart(owner);

    SmallVector<Value> elementSizes;
    if (memrefType.getRank() == 0) {
      elementSizes.push_back(createOneIndex(builder, root.getLoc()));
    } else {
      elementSizes.reserve(memrefType.getRank());
      for (int64_t dim = 0, rank = memrefType.getRank(); dim < rank; ++dim) {
        if (memrefType.isDynamicDim(dim)) {
          elementSizes.push_back(
              memref::DimOp::create(builder, root.getLoc(), root, dim));
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
    if (hasCodirTileOwnerSlicePlan(planSource)) {
      if (failed(createDbBackedMemref(builder, root.getLoc(), memrefType,
                                      ValueRange{}, replacement, planSource)))
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
    if (failed(hasCodirTileOwnerSlicePlan(planSource)
                   ? createDbBackedMemref(builder, alloc.getLoc(), memrefType,
                                          alloc.getDynamicSizes(), replacement,
                                          planSource)
                   : createDbBackedMemref(builder, alloc.getLoc(), memrefType,
                                          alloc.getDynamicSizes(),
                                          replacement)))
      return failure();
  } else if (auto alloca = dyn_cast<memref::AllocaOp>(def)) {
    if (failed(hasCodirTileOwnerSlicePlan(planSource)
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
