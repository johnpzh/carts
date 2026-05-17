///==========================================================================///
/// File: SdeToCodirMetadataUtils.h
///
/// SDE-owned planning metadata translation for the SDE -> CODIR boundary.
/// These helpers carry source scheduling intent into CODIR without creating
/// ARTS runtime/orchestration objects.
///==========================================================================///
#ifndef CARTS_DIALECT_CODIR_CONVERSION_SDETOCODIR_METADATAUTILS_H
#define CARTS_DIALECT_CODIR_CONVERSION_SDETOCODIR_METADATAUTILS_H

#include "carts/dialect/codir/IR/CodirDialect.h"
#include "carts/dialect/sde/IR/SdeDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::carts::codir::sde_to_codir {

inline CodirAccessMode convertAccessMode(sde::SdeAccessMode mode) {
  switch (mode) {
  case sde::SdeAccessMode::read:
    return CodirAccessMode::read;
  case sde::SdeAccessMode::write:
    return CodirAccessMode::write;
  case sde::SdeAccessMode::readwrite:
    return CodirAccessMode::readwrite;
  }
  return CodirAccessMode::readwrite;
}

inline CodirPattern convertPattern(sde::SdePattern pattern) {
  switch (pattern) {
  case sde::SdePattern::uniform:
    return CodirPattern::uniform;
  case sde::SdePattern::stencil_tiling_nd:
    return CodirPattern::stencil_tiling_nd;
  case sde::SdePattern::cross_dim_stencil_3d:
    return CodirPattern::cross_dim_stencil_3d;
  case sde::SdePattern::higher_order_stencil:
    return CodirPattern::higher_order_stencil;
  case sde::SdePattern::wavefront_2d:
    return CodirPattern::wavefront_2d;
  case sde::SdePattern::jacobi_alternating_buffers:
    return CodirPattern::jacobi_alternating_buffers;
  case sde::SdePattern::matmul:
    return CodirPattern::matmul;
  case sde::SdePattern::elementwise_pipeline:
    return CodirPattern::elementwise_pipeline;
  case sde::SdePattern::reduction:
    return CodirPattern::reduction;
  }
  return CodirPattern::uniform;
}

inline CodirDistributionKind
convertDistributionKind(sde::SdeDistributionKind kind) {
  switch (kind) {
  case sde::SdeDistributionKind::owner_compute:
    return CodirDistributionKind::owner_compute;
  case sde::SdeDistributionKind::blocked:
    return CodirDistributionKind::blocked;
  case sde::SdeDistributionKind::cyclic:
    return CodirDistributionKind::cyclic;
  }
  return CodirDistributionKind::owner_compute;
}

inline CodirIterationTopology
convertIterationTopology(sde::SdeIterationTopology topology) {
  switch (topology) {
  case sde::SdeIterationTopology::owner_strip:
    return CodirIterationTopology::owner_strip;
  case sde::SdeIterationTopology::owner_tile:
    return CodirIterationTopology::owner_tile;
  case sde::SdeIterationTopology::owner_tile_2d:
    return CodirIterationTopology::owner_tile_2d;
  }
  return CodirIterationTopology::owner_strip;
}

inline CodirRepetitionStructure
convertRepetitionStructure(sde::SdeRepetitionStructure structure) {
  switch (structure) {
  case sde::SdeRepetitionStructure::none:
    return CodirRepetitionStructure::none;
  case sde::SdeRepetitionStructure::pair_step:
    return CodirRepetitionStructure::pair_step;
  case sde::SdeRepetitionStructure::k_step:
    return CodirRepetitionStructure::k_step;
  case sde::SdeRepetitionStructure::full_timestep:
    return CodirRepetitionStructure::full_timestep;
  }
  return CodirRepetitionStructure::none;
}

inline CodirAsyncStrategy convertAsyncStrategy(sde::SdeAsyncStrategy strategy) {
  switch (strategy) {
  case sde::SdeAsyncStrategy::blocking:
    return CodirAsyncStrategy::blocking;
  case sde::SdeAsyncStrategy::advance_stage:
    return CodirAsyncStrategy::advance_stage;
  case sde::SdeAsyncStrategy::cps_chain:
    return CodirAsyncStrategy::cps_chain;
  }
  return CodirAsyncStrategy::blocking;
}

struct CodirCodeletMetadata {
  CodirPatternAttr pattern;
  CodirDistributionKindAttr distributionKind;
  CodirIterationTopologyAttr iterationTopology;
  CodirRepetitionStructureAttr repetitionStructure;
  CodirAsyncStrategyAttr asyncStrategy;
  ArrayAttr planOwnerDims;
  ArrayAttr tileOwnerDims;
  ArrayAttr tileShape;
  ArrayAttr logicalWorkerSlice;
  ArrayAttr haloShape;
  ArrayAttr accessMinOffsets;
  ArrayAttr accessMaxOffsets;
  ArrayAttr spatialDims;
  ArrayAttr writeFootprint;
  UnitAttr inPlaceSafe;
  UnitAttr inPlaceSharedState;
};

inline CodirCodeletMetadata getCodirMetadataFromTask(sde::SdeCuTaskOp task) {
  CodirCodeletMetadata metadata;
  if (!task)
    return metadata;
  MLIRContext *ctx = task.getContext();
  if (auto pattern = task.getPatternAttr())
    metadata.pattern =
        CodirPatternAttr::get(ctx, convertPattern(pattern.getValue()));
  return metadata;
}

inline CodirCodeletMetadata
getCodirMetadataFromSchedulingUnit(sde::SdeSuIterateOp source) {
  CodirCodeletMetadata metadata;
  if (!source)
    return metadata;
  MLIRContext *ctx = source.getContext();
  if (auto pattern = source.getPatternAttr())
    metadata.pattern =
        CodirPatternAttr::get(ctx, convertPattern(pattern.getValue()));
  if (auto kind = source.getDistributionKindAttr())
    metadata.distributionKind = CodirDistributionKindAttr::get(
        ctx, convertDistributionKind(kind.getValue()));
  else if (auto parentDistribute =
               source->getParentOfType<sde::SdeSuDistributeOp>())
    metadata.distributionKind = CodirDistributionKindAttr::get(
        ctx, convertDistributionKind(parentDistribute.getKind()));
  if (auto topology = source.getIterationTopologyAttr())
    metadata.iterationTopology = CodirIterationTopologyAttr::get(
        ctx, convertIterationTopology(topology.getValue()));
  if (auto repetition = source.getRepetitionStructureAttr())
    metadata.repetitionStructure = CodirRepetitionStructureAttr::get(
        ctx, convertRepetitionStructure(repetition.getValue()));
  if (auto async = source.getAsyncStrategyAttr())
    metadata.asyncStrategy = CodirAsyncStrategyAttr::get(
        ctx, convertAsyncStrategy(async.getValue()));
  metadata.planOwnerDims = source.getOwnerDimsAttr();
  metadata.tileOwnerDims = source.getPhysicalOwnerDimsAttr();
  metadata.tileShape = source.getPhysicalBlockShapeAttr();
  metadata.logicalWorkerSlice = source.getLogicalWorkerSliceAttr();
  metadata.haloShape = source.getPhysicalHaloShapeAttr();
  metadata.accessMinOffsets = source.getAccessMinOffsetsAttr();
  metadata.accessMaxOffsets = source.getAccessMaxOffsetsAttr();
  metadata.spatialDims = source.getSpatialDimsAttr();
  metadata.writeFootprint = source.getWriteFootprintAttr();
  if (source.getInPlaceSafeAttr())
    metadata.inPlaceSafe = UnitAttr::get(ctx);
  if (source.getInPlaceSharedStateAttr())
    metadata.inPlaceSharedState = UnitAttr::get(ctx);
  return metadata;
}

} // namespace mlir::carts::codir::sde_to_codir

#endif // CARTS_DIALECT_CODIR_CONVERSION_SDETOCODIR_METADATAUTILS_H
