#ifndef CARTS_UTILS_OPERATIONATTRIBUTES_H
#define CARTS_UTILS_OPERATIONATTRIBUTES_H

#include "arts/Dialect.h"
#include "arts/utils/StencilAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir {
namespace arts {
namespace AttrNames {

/// Module-level attributes used across ARTS passes/codegen.
namespace Module {
using namespace llvm;
constexpr StringLiteral RuntimeConfigPath = "arts.runtime_config_path";
constexpr StringLiteral RuntimeConfigData = "arts.runtime_config_data";
constexpr StringLiteral RuntimeTotalWorkers = "arts.runtime_total_workers";
constexpr StringLiteral RuntimeTotalNodes = "arts.runtime_total_nodes";
constexpr StringLiteral RuntimeStaticWorkers = "arts.runtime_static_workers";
} // namespace Module

/// Operation-level attributes used across ARTS passes
namespace Operation {
using namespace llvm;

/// Common ARTS attributes
constexpr StringLiteral Workers = "workers";
constexpr StringLiteral WorkersPerNode = "workers_per_node";
constexpr StringLiteral ArtsId = "arts.id";
constexpr StringLiteral ArtsCreateId = "arts.create_id";
constexpr StringLiteral MetadataOriginId = "arts.metadata_origin_id";
constexpr StringLiteral MetadataProvenance = "arts.metadata_provenance";
constexpr StringLiteral OutlinedFunc = "outlined_func";
constexpr StringLiteral Nowait = "nowait";
constexpr StringLiteral PreserveAccessMode = "preserve_access_mode";
constexpr StringLiteral PreserveDepEdge = "preserve_dep_edge";
constexpr StringLiteral LocalityOnly = "arts.locality_only";
constexpr StringLiteral ReadyLocalLaunch = "arts.ready_local_launch";

/// Partition-related attributes (TableGen-generated names)
constexpr StringLiteral PartitionMode = "partition_mode";
constexpr StringLiteral PartitionHint = "arts.partition_hint";
constexpr StringLiteral AccessPattern = "access_pattern";
constexpr StringLiteral Distributed = "distributed";
constexpr StringLiteral DistributionKind = "distribution_kind";
constexpr StringLiteral DistributionPattern = "distribution_pattern";
/// ODS-generated attribute name for EdtOp dep pattern
constexpr StringLiteral DepPatternAttr = "depPattern";
constexpr StringLiteral DistributionVersion = "distribution_version";
constexpr StringLiteral PatternRevision = "arts.pattern_revision";

/// Epoch without caller-active count.
constexpr StringLiteral NoStartEpoch = "arts.no_start_epoch";

/// DB storage-type inference annotations (set by DbModeTighteningPass)
constexpr StringLiteral LocalOnly = "arts.local_only";
constexpr StringLiteral ReadOnlyAfterInit = "arts.read_only_after_init";

/// Finish-EDT continuation attributes.
constexpr StringLiteral ControlDep = "arts.has_control_dep";
constexpr StringLiteral ContinuationForEpoch = "arts.continuation_for_epoch";

/// Preserves compile-time DB outer extents on rehydrated handle values when
/// outlining breaks the original DbAllocOp def-use chain.
constexpr StringLiteral DbStaticOuterShape = "arts.db_static_outer_shape";
/// Preserves the original DbAllocOp arts.id on rebuilt compile-time handle
/// scaffolding so downstream lowering/analysis can recover element sizes and
/// provenance without transporting raw handle pointers.
constexpr StringLiteral DbRootAllocId = "arts.db_root_alloc_id";

/// GUID range detection annotations (set by GUIDRangeDetection)
constexpr StringLiteral GuidRangeTripCount = "guid_range_trip_count";
constexpr StringLiteral HasGuidRangeAlloc = "has_guid_range_alloc";

/// LoweringContractOp attribute names (used in Dialect.cpp build method)
namespace Contract {
using namespace llvm;
/// Contract attribute name for dep pattern
constexpr StringLiteral DepPatternKey = "dep_pattern";
/// These are the same as Operation:: versions; kept as aliases for readability.
constexpr auto DistributionKind = Operation::DistributionKind;
constexpr auto DistributionPattern = Operation::DistributionPattern;
constexpr auto DistributionVersion = Operation::DistributionVersion;
constexpr StringLiteral OwnerDims = "owner_dims";
constexpr StringLiteral SupportedBlockHalo = "supported_block_halo";
constexpr StringLiteral StencilIndependentDims = "stencil_independent_dims";
constexpr StringLiteral PostDbRefined = "post_db_refined";
constexpr StringLiteral CriticalPathDistance = "critical_path_distance";
constexpr StringLiteral SpatialDims = "spatial_dims";
constexpr StringLiteral NarrowableDep = "narrowable_dep";
constexpr StringLiteral ContractKindKey = "contract_kind";
} // namespace Contract

/// Orchestration attributes for grouped repeated-wave lowering.
namespace Orchestration {
using namespace llvm;
constexpr StringLiteral Kind = "arts.orch.kind";
constexpr StringLiteral GroupId = "arts.orch.group_id";
constexpr StringLiteral WaveIndex = "arts.orch.wave_index";
constexpr StringLiteral WaveCount = "arts.orch.wave_count";
} // namespace Orchestration

/// Split launch state attributes.
namespace LaunchState {
using namespace llvm;
constexpr StringLiteral StateSchema = "arts.launch.state_schema";
constexpr StringLiteral DepSchema = "arts.launch.dep_schema";
constexpr StringLiteral StateSchemaVersion = "arts.launch.state_schema_version";
constexpr StringLiteral DepSchemaVersion = "arts.launch.dep_schema_version";
} // namespace LaunchState

/// Proof-driven ownership attributes.
namespace Proof {
using namespace llvm;
constexpr StringLiteral OwnerDimReachability =
    "arts.proof.owner_dim_reachability";
constexpr StringLiteral PartitionAccessMapping =
    "arts.proof.partition_access_mapping";
constexpr StringLiteral HaloLegality = "arts.proof.halo_legality";
constexpr StringLiteral DepSliceSoundness = "arts.proof.dep_slice_soundness";
constexpr StringLiteral RelaunchStateSoundness =
    "arts.proof.relaunch_state_soundness";
} // namespace Proof

/// Persistent structured region attribute.
constexpr llvm::StringLiteral PersistentRegion = "arts.persistent_region";

/// Orchestration kind value for repeated-wave groups.
constexpr llvm::StringLiteral RepeatedWaveGroup = "repeated_wave_group";

/// Block-loop strip-mining marker attribute.
constexpr llvm::StringLiteral StripMiningGenerated =
    "arts.block_loop_strip_mining.generated";

/// Marks allocas that bridge raised tensor DBs back to user-visible memrefs.
constexpr llvm::StringLiteral Preserve = "arts.preserve";

/// RT-facing loop execution hints copied onto outlined EDT functions. These
/// live on func/LLVM function ops, so they cannot be ODS accessors on CARTS
/// ops; all in-dialect producers/consumers use generated accessors instead.
namespace Rt {
using namespace llvm;
constexpr StringLiteral VectorizeWidth = "arts.rt.vectorize_width";
constexpr StringLiteral UnrollFactor = "arts.rt.unroll_factor";
constexpr StringLiteral InterleaveCount = "arts.rt.interleave_count";
} // namespace Rt

} // namespace Operation

} // namespace AttrNames

inline void copyCoreExecutionHintAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;

  UnitAttr inPlaceSafe;
  UnitAttr inPlaceSharedState;
  IntegerAttr vectorizeWidth;
  IntegerAttr unrollFactor;
  IntegerAttr interleaveCount;

  if (auto edtOp = dyn_cast<EdtOp>(source)) {
    inPlaceSafe = edtOp.getInPlaceSafeAttr();
    inPlaceSharedState = edtOp.getInPlaceSharedStateAttr();
    vectorizeWidth = edtOp.getVectorizeWidthAttr();
    unrollFactor = edtOp.getUnrollFactorAttr();
    interleaveCount = edtOp.getInterleaveCountAttr();
  } else {
    return;
  }

  if (auto edtOp = dyn_cast<EdtOp>(dest)) {
    if (inPlaceSafe)
      edtOp.setInPlaceSafeAttr(inPlaceSafe);
    if (inPlaceSharedState)
      edtOp.setInPlaceSharedStateAttr(inPlaceSharedState);
    if (vectorizeWidth)
      edtOp.setVectorizeWidthAttr(vectorizeWidth);
    if (unrollFactor)
      edtOp.setUnrollFactorAttr(unrollFactor);
    if (interleaveCount)
      edtOp.setInterleaveCountAttr(interleaveCount);
  }
}

inline void copyCoreExecutionHintAttrsToRtFunction(EdtOp source,
                                                   Operation *dest) {
  if (!source || !dest)
    return;
  if (auto attr = source.getVectorizeWidthAttr())
    dest->setAttr(AttrNames::Operation::Rt::VectorizeWidth, attr);
  if (auto attr = source.getUnrollFactorAttr())
    dest->setAttr(AttrNames::Operation::Rt::UnrollFactor, attr);
  if (auto attr = source.getInterleaveCountAttr())
    dest->setAttr(AttrNames::Operation::Rt::InterleaveCount, attr);
}

enum class MetadataProvenanceKind : uint8_t {
  Exact = 0,
  Transferred = 1,
  Recomputed = 2,
  Recovered = 3
};

inline StringRef metadataProvenanceToString(MetadataProvenanceKind kind) {
  switch (kind) {
  case MetadataProvenanceKind::Exact:
    return "exact";
  case MetadataProvenanceKind::Transferred:
    return "transferred";
  case MetadataProvenanceKind::Recomputed:
    return "recomputed";
  case MetadataProvenanceKind::Recovered:
    return "recovered";
  }
  return "exact";
}

inline std::optional<MetadataProvenanceKind>
parseMetadataProvenance(StringRef value) {
  if (value == "exact")
    return MetadataProvenanceKind::Exact;
  if (value == "transferred")
    return MetadataProvenanceKind::Transferred;
  if (value == "recomputed")
    return MetadataProvenanceKind::Recomputed;
  if (value == "recovered")
    return MetadataProvenanceKind::Recovered;
  return std::nullopt;
}

/// Check if an operation has a StringAttr with the given name whose value
/// matches the expected string. Null-safe: returns false if op is null.
inline bool hasStringAttrValue(Operation *op, StringRef attrName,
                               StringRef expected) {
  auto attr = op ? op->getAttrOfType<StringAttr>(attrName) : nullptr;
  return attr && attr.getValue() == expected;
}

inline ArrayAttr getPlanOwnerDimsAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getPlanOwnerDimsAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getPlanOwnerDimsAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getPlanOwnerDimsAttr();
  return nullptr;
}

inline ArrayAttr getPlanPhysicalBlockShapeAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getPlanPhysicalBlockShapeAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getPlanPhysicalBlockShapeAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getPlanPhysicalBlockShapeAttr();
  return nullptr;
}

inline ArrayAttr getPlanLogicalWorkerSliceAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getPlanLogicalWorkerSliceAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getPlanLogicalWorkerSliceAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getPlanLogicalWorkerSliceAttr();
  return nullptr;
}

inline ArrayAttr getPlanHaloShapeAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getPlanHaloShapeAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getPlanHaloShapeAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getPlanHaloShapeAttr();
  return nullptr;
}

inline ArtsPlanIterationTopologyAttr
getPlanIterationTopologyAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getPlanIterationTopologyAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getPlanIterationTopologyAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getPlanIterationTopologyAttr();
  return nullptr;
}

inline ArtsPlanRepetitionStructureAttr
getPlanRepetitionStructureAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getPlanRepetitionStructureAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getPlanRepetitionStructureAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getPlanRepetitionStructureAttr();
  return nullptr;
}

inline ArtsPlanAsyncStrategyAttr getPlanAsyncStrategyAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getPlanAsyncStrategyAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getPlanAsyncStrategyAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getPlanAsyncStrategyAttr();
  return nullptr;
}

inline IntegerAttr getPlanCostSchedulerOverheadAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getPlanCostSchedulerOverheadAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getPlanCostSchedulerOverheadAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getPlanCostSchedulerOverheadAttr();
  return nullptr;
}

inline IntegerAttr getPlanCostSliceWideningPressureAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getPlanCostSliceWideningPressureAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getPlanCostSliceWideningPressureAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getPlanCostSliceWideningPressureAttr();
  return nullptr;
}

inline IntegerAttr getPlanCostExpectedLocalWorkAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getPlanCostExpectedLocalWorkAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getPlanCostExpectedLocalWorkAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getPlanCostExpectedLocalWorkAttr();
  return nullptr;
}

inline IntegerAttr getPlanCostRelaunchAmortizationAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getPlanCostRelaunchAmortizationAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getPlanCostRelaunchAmortizationAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getPlanCostRelaunchAmortizationAttr();
  return nullptr;
}

inline void setPlanOwnerDimsAttr(Operation *op, ArrayAttr attr) {
  if (!op || !attr)
    return;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setPlanOwnerDimsAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setPlanOwnerDimsAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setPlanOwnerDimsAttr(attr);
}

inline void setPlanPhysicalBlockShapeAttr(Operation *op, ArrayAttr attr) {
  if (!op || !attr)
    return;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setPlanPhysicalBlockShapeAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setPlanPhysicalBlockShapeAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setPlanPhysicalBlockShapeAttr(attr);
}

inline void setPlanLogicalWorkerSliceAttr(Operation *op, ArrayAttr attr) {
  if (!op || !attr)
    return;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setPlanLogicalWorkerSliceAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setPlanLogicalWorkerSliceAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setPlanLogicalWorkerSliceAttr(attr);
}

inline void setPlanHaloShapeAttr(Operation *op, ArrayAttr attr) {
  if (!op || !attr)
    return;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setPlanHaloShapeAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setPlanHaloShapeAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setPlanHaloShapeAttr(attr);
}

inline void setPlanIterationTopologyAttr(
    Operation *op, ArtsPlanIterationTopologyAttr attr) {
  if (!op || !attr)
    return;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setPlanIterationTopologyAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setPlanIterationTopologyAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setPlanIterationTopologyAttr(attr);
}

inline void setPlanRepetitionStructureAttr(
    Operation *op, ArtsPlanRepetitionStructureAttr attr) {
  if (!op || !attr)
    return;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setPlanRepetitionStructureAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setPlanRepetitionStructureAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setPlanRepetitionStructureAttr(attr);
}

inline void setPlanAsyncStrategyAttr(Operation *op,
                                     ArtsPlanAsyncStrategyAttr attr) {
  if (!op || !attr)
    return;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setPlanAsyncStrategyAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setPlanAsyncStrategyAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setPlanAsyncStrategyAttr(attr);
}

inline void setPlanCostSchedulerOverheadAttr(Operation *op, IntegerAttr attr) {
  if (!op || !attr)
    return;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setPlanCostSchedulerOverheadAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setPlanCostSchedulerOverheadAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setPlanCostSchedulerOverheadAttr(attr);
}

inline void setPlanCostSliceWideningPressureAttr(Operation *op,
                                                 IntegerAttr attr) {
  if (!op || !attr)
    return;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setPlanCostSliceWideningPressureAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setPlanCostSliceWideningPressureAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setPlanCostSliceWideningPressureAttr(attr);
}

inline void setPlanCostExpectedLocalWorkAttr(Operation *op,
                                             IntegerAttr attr) {
  if (!op || !attr)
    return;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setPlanCostExpectedLocalWorkAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setPlanCostExpectedLocalWorkAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setPlanCostExpectedLocalWorkAttr(attr);
}

inline void setPlanCostRelaunchAmortizationAttr(Operation *op,
                                                IntegerAttr attr) {
  if (!op || !attr)
    return;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setPlanCostRelaunchAmortizationAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setPlanCostRelaunchAmortizationAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setPlanCostRelaunchAmortizationAttr(attr);
}

/// Check whether a CARTS operation carries any structured plan attr. This is a
/// generic contract-presence test; callers that need semantic families should
/// consume explicit dep/distribution contract attrs instead.
inline bool hasStructuredPlanAttrs(Operation *op) {
  return getPlanOwnerDimsAttr(op) || getPlanPhysicalBlockShapeAttr(op) ||
         getPlanLogicalWorkerSliceAttr(op) || getPlanHaloShapeAttr(op) ||
         getPlanIterationTopologyAttr(op) ||
         getPlanRepetitionStructureAttr(op) ||
         getPlanAsyncStrategyAttr(op) ||
         getPlanCostSchedulerOverheadAttr(op) ||
         getPlanCostSliceWideningPressureAttr(op) ||
         getPlanCostExpectedLocalWorkAttr(op) ||
         getPlanCostRelaunchAmortizationAttr(op);
}

inline std::optional<StringRef> getRuntimeConfigPath(ModuleOp module) {
  if (!module)
    return std::nullopt;
  if (auto attr = module->getAttrOfType<StringAttr>(
          AttrNames::Module::RuntimeConfigPath))
    return attr.getValue();
  return std::nullopt;
}

inline void setRuntimeConfigPath(ModuleOp module, StringRef path) {
  if (!module || path.empty())
    return;
  module->setAttr(AttrNames::Module::RuntimeConfigPath,
                  StringAttr::get(module.getContext(), path));
}

inline std::optional<StringRef> getRuntimeConfigData(ModuleOp module) {
  if (!module)
    return std::nullopt;
  if (auto attr = module->getAttrOfType<StringAttr>(
          AttrNames::Module::RuntimeConfigData))
    return attr.getValue();
  return std::nullopt;
}

inline void setRuntimeConfigData(ModuleOp module, StringRef data) {
  if (!module || data.empty())
    return;
  module->setAttr(AttrNames::Module::RuntimeConfigData,
                  StringAttr::get(module.getContext(), data));
}

inline std::optional<int64_t> getRuntimeTotalWorkers(ModuleOp module) {
  if (!module)
    return std::nullopt;
  if (auto attr = module->getAttrOfType<IntegerAttr>(
          AttrNames::Module::RuntimeTotalWorkers))
    return attr.getInt();
  return std::nullopt;
}

inline void setRuntimeTotalWorkers(ModuleOp module, int64_t workers) {
  if (!module || workers <= 0)
    return;
  module->setAttr(
      AttrNames::Module::RuntimeTotalWorkers,
      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workers));
}

inline std::optional<SmallVector<int64_t, 4>>
getDbStaticOuterShape(Operation *op) {
  return readI64ArrayAttr(op, AttrNames::Operation::DbStaticOuterShape);
}

inline std::optional<SmallVector<int64_t, 4>>
getDbStaticOuterShape(Value value) {
  return value ? getDbStaticOuterShape(value.getDefiningOp()) : std::nullopt;
}

inline void setDbStaticOuterShape(Operation *op, ArrayRef<int64_t> shape) {
  if (!op || shape.empty())
    return;
  op->setAttr(AttrNames::Operation::DbStaticOuterShape,
              buildI64ArrayAttr(op, shape));
}

inline std::optional<int64_t> getDbRootAllocId(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr =
          op->getAttrOfType<IntegerAttr>(AttrNames::Operation::DbRootAllocId))
    return attr.getInt();
  return std::nullopt;
}

inline std::optional<int64_t> getDbRootAllocId(Value value) {
  return value ? getDbRootAllocId(value.getDefiningOp()) : std::nullopt;
}

inline void setDbRootAllocId(Operation *op, int64_t id) {
  if (!op || id <= 0)
    return;
  auto *ctx = op->getContext();
  auto type = IntegerType::get(ctx, 64);
  op->setAttr(AttrNames::Operation::DbRootAllocId, IntegerAttr::get(type, id));
}

inline bool getRuntimeStaticWorkers(ModuleOp module) {
  if (!module)
    return false;
  if (auto attr = module->getAttrOfType<BoolAttr>(
          AttrNames::Module::RuntimeStaticWorkers))
    return attr.getValue();
  return false;
}

inline void setRuntimeStaticWorkers(ModuleOp module, bool enabled) {
  if (!module)
    return;
  module->setAttr(AttrNames::Module::RuntimeStaticWorkers,
                  BoolAttr::get(module.getContext(), enabled));
}

inline std::optional<int64_t> getRuntimeTotalNodes(ModuleOp module) {
  if (!module)
    return std::nullopt;
  if (auto attr = module->getAttrOfType<IntegerAttr>(
          AttrNames::Module::RuntimeTotalNodes))
    return attr.getInt();
  return std::nullopt;
}

inline void setRuntimeTotalNodes(ModuleOp module, int64_t nodes) {
  if (!module || nodes <= 0)
    return;
  module->setAttr(
      AttrNames::Module::RuntimeTotalNodes,
      IntegerAttr::get(IntegerType::get(module.getContext(), 64), nodes));
}

struct PartitioningHint {
  PartitionMode mode = PartitionMode::coarse;
  std::optional<int64_t> blockSize;

  static PartitioningHint block(std::optional<int64_t> size) {
    PartitioningHint h;
    h.mode = PartitionMode::block;
    h.blockSize = size;
    return h;
  }

  DictionaryAttr toAttribute(MLIRContext *ctx) const;
  static std::optional<PartitioningHint> fromAttribute(Attribute attr);
};

inline int64_t getArtsId(Operation *op) {
  if (!op)
    return 0;
  if (auto attr = op->getAttrOfType<IntegerAttr>(AttrNames::Operation::ArtsId))
    return attr.getInt();
  return 0;
}

inline void setArtsId(Operation *op, int64_t id) {
  if (!op || id <= 0)
    return;
  auto *ctx = op->getContext();
  auto type = IntegerType::get(ctx, 64);
  op->setAttr(AttrNames::Operation::ArtsId, IntegerAttr::get(type, id));
}

inline std::optional<int64_t> getArtsCreateId(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr =
          op->getAttrOfType<IntegerAttr>(AttrNames::Operation::ArtsCreateId))
    return attr.getInt();
  return std::nullopt;
}

inline void setArtsCreateId(Operation *op, int64_t id) {
  if (!op)
    return;
  if (id <= 0) {
    op->removeAttr(AttrNames::Operation::ArtsCreateId);
    return;
  }
  auto *ctx = op->getContext();
  auto type = IntegerType::get(ctx, 64);
  op->setAttr(AttrNames::Operation::ArtsCreateId, IntegerAttr::get(type, id));
}

inline std::optional<int64_t> getMetadataOriginId(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(
          AttrNames::Operation::MetadataOriginId))
    return attr.getInt();
  return std::nullopt;
}

inline void setMetadataOriginId(Operation *op, int64_t id) {
  if (!op)
    return;
  if (id <= 0) {
    op->removeAttr(AttrNames::Operation::MetadataOriginId);
    return;
  }
  auto *ctx = op->getContext();
  auto type = IntegerType::get(ctx, 64);
  op->setAttr(AttrNames::Operation::MetadataOriginId,
              IntegerAttr::get(type, id));
}

inline std::optional<MetadataProvenanceKind>
getMetadataProvenance(Operation *op) {
  if (!op)
    return std::nullopt;
  auto attr =
      op->getAttrOfType<StringAttr>(AttrNames::Operation::MetadataProvenance);
  if (!attr)
    return std::nullopt;
  return parseMetadataProvenance(attr.getValue());
}

inline void setMetadataProvenance(Operation *op, MetadataProvenanceKind kind) {
  if (!op)
    return;
  op->setAttr(
      AttrNames::Operation::MetadataProvenance,
      StringAttr::get(op->getContext(), metadataProvenanceToString(kind)));
}

inline void ensureMetadataProvenance(Operation *op,
                                     MetadataProvenanceKind kind) {
  if (!op || op->hasAttr(AttrNames::Operation::MetadataProvenance))
    return;
  setMetadataProvenance(op, kind);
}

inline std::optional<StringRef> getOutlinedFunc(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr =
          op->getAttrOfType<StringAttr>(AttrNames::Operation::OutlinedFunc))
    return attr.getValue();
  return std::nullopt;
}

inline void setOutlinedFunc(Operation *op, StringRef name) {
  if (!op)
    return;
  if (name.empty()) {
    op->removeAttr(AttrNames::Operation::OutlinedFunc);
    return;
  }
  op->setAttr(AttrNames::Operation::OutlinedFunc,
              StringAttr::get(op->getContext(), name));
}

inline bool hasNowait(Operation *op) {
  return op && op->hasAttr(AttrNames::Operation::Nowait);
}

inline void setNowait(Operation *op, bool enabled = true) {
  if (!op)
    return;
  if (enabled) {
    op->setAttr(AttrNames::Operation::Nowait, UnitAttr::get(op->getContext()));
    return;
  }
  op->removeAttr(AttrNames::Operation::Nowait);
}

inline std::optional<PartitionMode> getPartitionMode(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<PartitionModeAttr>(
          AttrNames::Operation::PartitionMode))
    return attr.getValue();
  return std::nullopt;
}

inline void setPartitionMode(Operation *op, PartitionMode mode) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::PartitionMode,
              PartitionModeAttr::get(op->getContext(), mode));
}

inline std::optional<DbAccessPattern> getDbAccessPattern(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<DbAccessPatternAttr>(
          AttrNames::Operation::AccessPattern))
    return attr.getValue();
  return std::nullopt;
}

inline void setDbAccessPattern(Operation *op, DbAccessPattern pattern) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::AccessPattern,
              DbAccessPatternAttr::get(op->getContext(), pattern));
}

inline bool hasDistributedDbAllocation(Operation *op) {
  if (!op)
    return false;
  return op->hasAttr(AttrNames::Operation::Distributed);
}

inline void setDistributedDbAllocation(Operation *op, bool enabled) {
  if (!op)
    return;
  if (enabled) {
    op->setAttr(AttrNames::Operation::Distributed,
                UnitAttr::get(op->getContext()));
    return;
  }
  op->removeAttr(AttrNames::Operation::Distributed);
}

inline std::optional<int64_t> getWorkers(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(AttrNames::Operation::Workers))
    return attr.getInt();
  return std::nullopt;
}

inline void setWorkers(Operation *op, int64_t workers) {
  if (!op)
    return;
  if (workers <= 0) {
    op->removeAttr(AttrNames::Operation::Workers);
    return;
  }
  int64_t clamped =
      std::min<int64_t>(workers, std::numeric_limits<int32_t>::max());
  op->setAttr(
      AttrNames::Operation::Workers,
      IntegerAttr::get(IntegerType::get(op->getContext(), 32), clamped));
}

inline std::optional<int64_t> getWorkersPerNode(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr =
          op->getAttrOfType<IntegerAttr>(AttrNames::Operation::WorkersPerNode))
    return attr.getInt();
  return std::nullopt;
}

inline void setWorkersPerNode(Operation *op, int64_t workersPerNode) {
  if (!op)
    return;
  if (workersPerNode <= 0) {
    op->removeAttr(AttrNames::Operation::WorkersPerNode);
    return;
  }
  op->setAttr(
      AttrNames::Operation::WorkersPerNode,
      IntegerAttr::get(IntegerType::get(op->getContext(), 64), workersPerNode));
}

inline bool isLocalityOnly(Operation *op) {
  return op && op->hasAttr(AttrNames::Operation::LocalityOnly);
}

inline void copyWorkerTopologyAttrs(Operation *from, Operation *to) {
  if (!to)
    return;
  if (!from) {
    setWorkers(to, 0);
    setWorkersPerNode(to, 0);
    return;
  }
  setWorkers(to, getWorkers(from).value_or(0));
  setWorkersPerNode(to, getWorkersPerNode(from).value_or(0));
}

inline std::optional<EdtDistributionKind>
getEdtDistributionKind(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    if (auto attr = edtOp.getDistributionKindAttr())
      return attr.getValue();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    if (auto attr = epochOp.getDistributionKindAttr())
      return attr.getValue();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    if (auto attr = dbAllocOp.getDistributionKindAttr())
      return attr.getValue();
  if (auto dbAcquireOp = dyn_cast<DbAcquireOp>(op))
    if (auto attr = dbAcquireOp.getDistributionKindAttr())
      return attr.getValue();
  if (auto attr = op->getAttrOfType<EdtDistributionKindAttr>(
          AttrNames::Operation::DistributionKind))
    return attr.getValue();
  return std::nullopt;
}

inline void setEdtDistributionKind(Operation *op, EdtDistributionKind kind) {
  if (!op)
    return;
  auto attr = EdtDistributionKindAttr::get(op->getContext(), kind);
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setDistributionKindAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setDistributionKindAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setDistributionKindAttr(attr);
  else if (auto dbAcquireOp = dyn_cast<DbAcquireOp>(op))
    dbAcquireOp.setDistributionKindAttr(attr);
  else
    op->setAttr(AttrNames::Operation::DistributionKind, attr);
}

inline std::optional<EdtDistributionPattern>
getEdtDistributionPattern(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    if (auto attr = edtOp.getDistributionPatternAttr())
      return attr.getValue();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    if (auto attr = epochOp.getDistributionPatternAttr())
      return attr.getValue();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    if (auto attr = dbAllocOp.getDistributionPatternAttr())
      return attr.getValue();
  if (auto dbAcquireOp = dyn_cast<DbAcquireOp>(op))
    if (auto attr = dbAcquireOp.getDistributionPatternAttr())
      return attr.getValue();
  if (auto attr = op->getAttrOfType<EdtDistributionPatternAttr>(
          AttrNames::Operation::DistributionPattern))
    return attr.getValue();
  return std::nullopt;
}

inline void setEdtDistributionPattern(Operation *op,
                                      EdtDistributionPattern pattern) {
  if (!op)
    return;
  auto attr = EdtDistributionPatternAttr::get(op->getContext(), pattern);
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setDistributionPatternAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setDistributionPatternAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setDistributionPatternAttr(attr);
  else if (auto dbAcquireOp = dyn_cast<DbAcquireOp>(op))
    dbAcquireOp.setDistributionPatternAttr(attr);
  else
    op->setAttr(AttrNames::Operation::DistributionPattern, attr);
}

inline std::optional<ArtsDepPattern> getDepPattern(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    if (auto attr = edtOp.getDepPatternAttr())
      return attr.getValue();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    if (auto attr = epochOp.getDepPatternAttr())
      return attr.getValue();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    if (auto attr = dbAllocOp.getDepPatternAttr())
      return attr.getValue();
  if (auto dbAcquireOp = dyn_cast<DbAcquireOp>(op))
    if (auto attr = dbAcquireOp.getDepPatternAttr())
      return attr.getValue();
  if (auto attr = op->getAttrOfType<ArtsDepPatternAttr>(
          AttrNames::Operation::DepPatternAttr))
    return attr.getValue();
  return std::nullopt;
}

inline void setDepPattern(Operation *op, ArtsDepPattern pattern) {
  if (!op)
    return;
  auto attr = ArtsDepPatternAttr::get(op->getContext(), pattern);
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setDepPatternAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setDepPatternAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setDepPatternAttr(attr);
  else if (auto dbAcquireOp = dyn_cast<DbAcquireOp>(op))
    dbAcquireOp.setDepPatternAttr(attr);
  else
    op->setAttr(AttrNames::Operation::DepPatternAttr, attr);
}

inline IntegerAttr getDistributionVersionAttr(Operation *op) {
  if (!op)
    return nullptr;
  if (auto edtOp = dyn_cast<EdtOp>(op))
    return edtOp.getDistributionVersionAttr();
  if (auto epochOp = dyn_cast<EpochOp>(op))
    return epochOp.getDistributionVersionAttr();
  if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    return dbAllocOp.getDistributionVersionAttr();
  if (auto dbAcquireOp = dyn_cast<DbAcquireOp>(op))
    return dbAcquireOp.getDistributionVersionAttr();
  return op->getAttrOfType<IntegerAttr>(AttrNames::Operation::DistributionVersion);
}

inline void setDistributionVersionAttr(Operation *op, IntegerAttr attr) {
  if (!op)
    return;
  if (!attr) {
    op->removeAttr(AttrNames::Operation::DistributionVersion);
    return;
  }
  if (auto edtOp = dyn_cast<EdtOp>(op))
    edtOp.setDistributionVersionAttr(attr);
  else if (auto epochOp = dyn_cast<EpochOp>(op))
    epochOp.setDistributionVersionAttr(attr);
  else if (auto dbAllocOp = dyn_cast<DbAllocOp>(op))
    dbAllocOp.setDistributionVersionAttr(attr);
  else if (auto dbAcquireOp = dyn_cast<DbAcquireOp>(op))
    dbAcquireOp.setDistributionVersionAttr(attr);
  else
    op->setAttr(AttrNames::Operation::DistributionVersion, attr);
}

inline void setDistributionVersion(Operation *op, int64_t version) {
  if (!op)
    return;
  if (version <= 0) {
    op->removeAttr(AttrNames::Operation::DistributionVersion);
    return;
  }
  setDistributionVersionAttr(
      op, IntegerAttr::get(IntegerType::get(op->getContext(), 32), version));
}

inline bool isStencilFamilyDepPattern(ArtsDepPattern pattern) {
  switch (pattern) {
  case ArtsDepPattern::stencil:
  case ArtsDepPattern::stencil_tiling_nd:
  case ArtsDepPattern::cross_dim_stencil_3d:
  case ArtsDepPattern::higher_order_stencil:
  case ArtsDepPattern::wavefront_2d:
  case ArtsDepPattern::jacobi_alternating_buffers:
    return true;
  case ArtsDepPattern::unknown:
  case ArtsDepPattern::uniform:
  case ArtsDepPattern::triangular:
  case ArtsDepPattern::matmul:
  case ArtsDepPattern::elementwise_pipeline:
    return false;
  }
}

inline bool isStencilHaloDepPattern(ArtsDepPattern pattern) {
  switch (pattern) {
  case ArtsDepPattern::stencil:
  case ArtsDepPattern::jacobi_alternating_buffers:
  case ArtsDepPattern::stencil_tiling_nd:
  case ArtsDepPattern::cross_dim_stencil_3d:
  case ArtsDepPattern::higher_order_stencil:
    return true;
  case ArtsDepPattern::unknown:
  case ArtsDepPattern::uniform:
  case ArtsDepPattern::triangular:
  case ArtsDepPattern::matmul:
  case ArtsDepPattern::elementwise_pipeline:
  case ArtsDepPattern::wavefront_2d:
    return false;
  }
}

inline bool isUniformFamilyDepPattern(ArtsDepPattern pattern) {
  switch (pattern) {
  case ArtsDepPattern::uniform:
  case ArtsDepPattern::elementwise_pipeline:
    return true;
  case ArtsDepPattern::unknown:
  case ArtsDepPattern::stencil:
  case ArtsDepPattern::matmul:
  case ArtsDepPattern::triangular:
  case ArtsDepPattern::wavefront_2d:
  case ArtsDepPattern::jacobi_alternating_buffers:
  case ArtsDepPattern::stencil_tiling_nd:
  case ArtsDepPattern::cross_dim_stencil_3d:
  case ArtsDepPattern::higher_order_stencil:
    return false;
  }
}

inline std::optional<EdtDistributionPattern>
getDistributionPatternForDepPattern(ArtsDepPattern pattern) {
  switch (pattern) {
  case ArtsDepPattern::unknown:
    return std::nullopt;
  case ArtsDepPattern::uniform:
  case ArtsDepPattern::elementwise_pipeline:
    return EdtDistributionPattern::uniform;
  case ArtsDepPattern::stencil:
  case ArtsDepPattern::stencil_tiling_nd:
  case ArtsDepPattern::cross_dim_stencil_3d:
  case ArtsDepPattern::higher_order_stencil:
  case ArtsDepPattern::wavefront_2d:
  case ArtsDepPattern::jacobi_alternating_buffers:
    return EdtDistributionPattern::stencil;
  case ArtsDepPattern::matmul:
    return EdtDistributionPattern::matmul;
  case ArtsDepPattern::triangular:
    return EdtDistributionPattern::triangular;
  }
}

inline std::optional<ArtsDepPattern> getEffectiveDepPattern(Operation *op) {
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (auto pattern = getDepPattern(current))
      return pattern;
  }
  return std::nullopt;
}

inline std::optional<int64_t> getPatternRevision(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr =
          op->getAttrOfType<IntegerAttr>(AttrNames::Operation::PatternRevision))
    return attr.getInt();
  return std::nullopt;
}

inline void setPatternRevision(Operation *op, int64_t revision) {
  if (!op)
    return;
  if (revision <= 0) {
    op->removeAttr(AttrNames::Operation::PatternRevision);
    return;
  }
  op->setAttr(
      AttrNames::Operation::PatternRevision,
      IntegerAttr::get(IntegerType::get(op->getContext(), 64), revision));
}

inline void copyDepPatternAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;

  if (auto pattern = getDepPattern(source))
    setDepPattern(dest, *pattern);
  else
    dest->removeAttr(AttrNames::Operation::DepPatternAttr);
}

inline void inheritDepPatternAttrs(Operation *source, Operation *dest) {
  if (!source || !dest || getDepPattern(dest))
    return;
  if (auto pattern = getDepPattern(source))
    setDepPattern(dest, *pattern);
}

/// Copy distribution_* attributes between operations.
/// This intentionally transfers only distribution contracts:
///   - distribution_kind
///   - distribution_pattern
///   - distribution_version
inline void copyDistributionAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;

  if (auto kind = getEdtDistributionKind(source))
    setEdtDistributionKind(dest, *kind);
  else
    dest->removeAttr(AttrNames::Operation::DistributionKind);

  if (auto pattern = getEdtDistributionPattern(source))
    setEdtDistributionPattern(dest, *pattern);
  else
    dest->removeAttr(AttrNames::Operation::DistributionPattern);

  if (auto version = getDistributionVersionAttr(source))
    setDistributionVersionAttr(dest, version);
  else
    dest->removeAttr(AttrNames::Operation::DistributionVersion);
}

inline void inheritDistributionAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;

  if (!getEdtDistributionKind(dest))
    if (auto kind = getEdtDistributionKind(source))
      setEdtDistributionKind(dest, *kind);

  if (!getEdtDistributionPattern(dest))
    if (auto pattern = getEdtDistributionPattern(source))
      setEdtDistributionPattern(dest, *pattern);

  if (!getDistributionVersionAttr(dest))
    if (auto version = getDistributionVersionAttr(source))
      setDistributionVersionAttr(dest, version);
}

/// Copy semantic pattern attributes between operations.
/// This is the canonical helper for structural rewrites that replace a loop,
/// EDT, or epoch with an equivalent operation and want downstream passes to
/// keep seeing the same high-level dep/distribution family.
inline void copyPatternAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;
  copyDistributionAttrs(source, dest);
  copyDepPatternAttrs(source, dest);
  if (auto revision = getPatternRevision(source))
    setPatternRevision(dest, *revision);
  else
    dest->removeAttr(AttrNames::Operation::PatternRevision);
}

inline void inheritPatternAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;
  inheritDistributionAttrs(source, dest);
  inheritDepPatternAttrs(source, dest);
  if (!getPatternRevision(dest))
    if (auto revision = getPatternRevision(source))
      setPatternRevision(dest, *revision);
}

/// Use only when the destination preserves the same loop semantics/identity as
/// the source. Structural rewrites that create a new iteration space should
/// restamp the specific attrs they still mean instead of cloning all metadata.
inline void copyArtsMetadataAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;
  if (auto id =
          source->getAttrOfType<IntegerAttr>(AttrNames::Operation::ArtsId))
    dest->setAttr(AttrNames::Operation::ArtsId, id);
  if (auto originId = source->getAttrOfType<IntegerAttr>(
          AttrNames::Operation::MetadataOriginId))
    dest->setAttr(AttrNames::Operation::MetadataOriginId, originId);
  if (auto provenance = source->getAttrOfType<StringAttr>(
          AttrNames::Operation::MetadataProvenance))
    dest->setAttr(AttrNames::Operation::MetadataProvenance, provenance);
  if (auto mode = source->getAttrOfType<PartitionModeAttr>(
          AttrNames::Operation::PartitionMode))
    dest->setAttr(AttrNames::Operation::PartitionMode, mode);
  if (auto hint = source->getAttr(AttrNames::Operation::PartitionHint))
    dest->setAttr(AttrNames::Operation::PartitionHint, hint);
}

/// Copy only the semantic contract attrs that specialized pattern detection
/// stamps before DB values exist. Structural rewrites should use this helper
/// when they want to preserve pattern meaning without also copying unrelated
/// ids or bookkeeping metadata.
inline void copySemanticContractAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;
  copyPatternAttrs(source, dest);
  copyStencilContractAttrs(source, dest);
  if (auto contractKind = source->getAttrOfType<IntegerAttr>(
          AttrNames::Operation::Contract::ContractKindKey))
    dest->setAttr(AttrNames::Operation::Contract::ContractKindKey,
                  contractKind);
  else
    dest->removeAttr(AttrNames::Operation::Contract::ContractKindKey);
  if (source->hasAttr(AttrNames::Operation::Contract::NarrowableDep))
    dest->setAttr(AttrNames::Operation::Contract::NarrowableDep,
                  UnitAttr::get(dest->getContext()));
  else
    dest->removeAttr(AttrNames::Operation::Contract::NarrowableDep);
}

/// Copy structured kernel plan attrs from source to dest.
/// Used by Core materializers to propagate SDE-authored plan attrs to concrete
/// ARTS objects.
inline void copyPlanAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;
  setPlanOwnerDimsAttr(dest, getPlanOwnerDimsAttr(source));
  setPlanPhysicalBlockShapeAttr(dest, getPlanPhysicalBlockShapeAttr(source));
  setPlanLogicalWorkerSliceAttr(dest, getPlanLogicalWorkerSliceAttr(source));
  setPlanHaloShapeAttr(dest, getPlanHaloShapeAttr(source));
  setPlanIterationTopologyAttr(dest, getPlanIterationTopologyAttr(source));
  setPlanRepetitionStructureAttr(dest, getPlanRepetitionStructureAttr(source));
  setPlanAsyncStrategyAttr(dest, getPlanAsyncStrategyAttr(source));
  setPlanCostSchedulerOverheadAttr(dest,
                                   getPlanCostSchedulerOverheadAttr(source));
  setPlanCostSliceWideningPressureAttr(
      dest, getPlanCostSliceWideningPressureAttr(source));
  setPlanCostExpectedLocalWorkAttr(dest,
                                   getPlanCostExpectedLocalWorkAttr(source));
  setPlanCostRelaunchAmortizationAttr(
      dest, getPlanCostRelaunchAmortizationAttr(source));
}

inline void inheritSemanticContractAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;
  inheritPatternAttrs(source, dest);
  inheritStencilContractAttrs(source, dest);
  if (!dest->hasAttr(AttrNames::Operation::Contract::ContractKindKey))
    if (auto contractKind = source->getAttrOfType<IntegerAttr>(
            AttrNames::Operation::Contract::ContractKindKey))
      dest->setAttr(AttrNames::Operation::Contract::ContractKindKey,
                    contractKind);
  if (!dest->hasAttr(AttrNames::Operation::Contract::NarrowableDep) &&
      source->hasAttr(AttrNames::Operation::Contract::NarrowableDep))
    dest->setAttr(AttrNames::Operation::Contract::NarrowableDep,
                  UnitAttr::get(dest->getContext()));
}

inline DictionaryAttr PartitioningHint::toAttribute(MLIRContext *ctx) const {
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(
      {StringAttr::get(ctx, "mode"),
       IntegerAttr::get(IntegerType::get(ctx, 8), static_cast<uint8_t>(mode))});
  if (blockSize)
    attrs.push_back({StringAttr::get(ctx, "blockSize"),
                     IntegerAttr::get(IntegerType::get(ctx, 64), *blockSize)});
  return DictionaryAttr::get(ctx, attrs);
}

inline std::optional<PartitioningHint>
PartitioningHint::fromAttribute(Attribute attr) {
  auto dictAttr = dyn_cast_or_null<DictionaryAttr>(attr);
  if (!dictAttr)
    return std::nullopt;
  PartitioningHint hint;
  if (auto modeAttr = dictAttr.getAs<IntegerAttr>("mode"))
    hint.mode = static_cast<PartitionMode>(modeAttr.getInt());
  if (auto chunkAttr = dictAttr.getAs<IntegerAttr>("blockSize"))
    hint.blockSize = chunkAttr.getInt();
  return hint;
}

inline std::optional<PartitioningHint> getPartitioningHint(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttr(AttrNames::Operation::PartitionHint))
    return PartitioningHint::fromAttribute(attr);
  return std::nullopt;
}

inline void setPartitioningHint(Operation *op, const PartitioningHint &hint) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::PartitionHint,
              hint.toAttribute(op->getContext()));
}

} // namespace arts
} // namespace mlir

#endif /// CARTS_UTILS_OPERATIONATTRIBUTES_H
