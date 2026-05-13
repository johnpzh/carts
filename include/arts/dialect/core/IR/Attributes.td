///==========================================================================///
/// File: Attributes.td
/// This file declares the ARTS dialect attributes.
///==========================================================================///

#ifndef ARTS_ATTRS
#define ARTS_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "Dialect.td"

///===----------------------------------------------------------------------===///
// ARTS Enum Helpers
///===----------------------------------------------------------------------===///

class Arts_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let cppNamespace = "::mlir::arts";
  let genSpecializedAttr = 0;
}

class Arts_I32EnumAttr<string mnemonic, Arts_I32Enum enumInfo>
    : EnumAttr<ArtsDialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

///===----------------------------------------------------------------------===///
// ARTS Attributes
///===----------------------------------------------------------------------===///
class Arts_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ArtsDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

// Enum info for ARTS access modes (in, out, inout, uninitialized)
def ArtsMode : Arts_I32Enum<"ArtsMode", "ARTS DataBlock access mode", [
  I32EnumAttrCase<"in", 0, "in">, I32EnumAttrCase<"out", 1, "out">,
  I32EnumAttrCase<"inout", 2, "inout">,
  I32EnumAttrCase<"uninitialized", 3, "uninitialized">
]>;

// Enum attribute for ARTS access modes
def ArtsModeAttr : Arts_I32EnumAttr<"mode", ArtsMode>;

/// Enum info for DB allocation types
def DbAllocType : Arts_I32Enum<"DbAllocType", "ARTS DB allocation type", [
  I32EnumAttrCase<"stack", 0, "stack">, I32EnumAttrCase<"heap", 1, "heap">,
  I32EnumAttrCase<"global", 2, "global">,
  I32EnumAttrCase<"unknown", 3, "unknown">
]>;

/// Enum attribute for DB allocation types
def DbAllocTypeAttr : Arts_I32EnumAttr<"alloc_type", DbAllocType>;

/// Enum info for EDT types
def EdtType : Arts_I32Enum<"EdtType", "ARTS EDT type", [
  I32EnumAttrCase<"parallel", 0, "parallel">,
  I32EnumAttrCase<"single", 1, "single">, I32EnumAttrCase<"sync", 2, "sync">,
  I32EnumAttrCase<"task", 3, "task">
]>;

/// Enum attribute for EDT types
def EdtTypeAttr : Arts_I32EnumAttr<"edt_type", EdtType>;

/// Enum info for compiler DB access modes.
/// Only `read` and `write` are actively used in the compiler.
/// Integer values intentionally match ARTS runtime dependency modes
/// `DB_MODE_RO` (1) and `DB_MODE_EW` (2) so dependency lowering can pass them
/// through directly without an extra translation layer.
def DbMode : Arts_I32Enum<"DbMode", "ARTS DB access mode", [
  I32EnumAttrCase<"read", 1, "read">, I32EnumAttrCase<"write", 2, "write">
]>;

/// Enum attribute for DB allocation modes
def DbModeAttr : Arts_I32EnumAttr<"db_mode", DbMode>;

/// Enum info for DB release types
def DbReleaseType : Arts_I32Enum<"DbReleaseType", "ARTS DB release type", [
  I32EnumAttrCase<"normal", 0, "normal">,
  I32EnumAttrCase<"dynamic_free", 1, "dynamic_free">
]>;

/// Enum attribute for DB release types
def DbReleaseTypeAttr : Arts_I32EnumAttr<"release_type", DbReleaseType>;

/// NOTE: This enum intentionally duplicates omp::ClauseScheduleKind (from the
/// OpenMP dialect). Using the OMP dialect enum directly would add a tablegen
/// dependency on the OMP dialect for ARTS attributes, which is undesirable.
/// ConvertOpenMPToArts.cpp maps between the two enums at conversion time.
def ForScheduleKind
    : Arts_I32Enum<"ForScheduleKind", "ARTS for-loop schedule kind", [
      I32EnumAttrCase<"Static", 0, "static">,
      I32EnumAttrCase<"Dynamic", 1, "dynamic">,
      I32EnumAttrCase<"Guided", 2, "guided">,
      I32EnumAttrCase<"Auto", 3, "auto">,
      I32EnumAttrCase<"Runtime", 4, "runtime">
    ]>;

def ForScheduleKindAttr : Arts_I32EnumAttr<"for_schedule", ForScheduleKind>;

def ArtsPlanIterationTopology
    : Arts_I32Enum<"ArtsPlanIterationTopology",
                   "Core ARTS physical iteration topology", [
      I32EnumAttrCase<"owner_strip", 0, "owner_strip">,
      I32EnumAttrCase<"owner_tile", 1, "owner_tile">,
      I32EnumAttrCase<"owner_tile_2d", 2, "owner_tile_2d">
    ]>;
def ArtsPlanIterationTopologyAttr
    : Arts_I32EnumAttr<"plan_iteration_topology",
                       ArtsPlanIterationTopology>;

def ArtsPlanRepetitionStructure
    : Arts_I32Enum<"ArtsPlanRepetitionStructure",
                   "Core ARTS repeated-wave structure", [
      I32EnumAttrCase<"none", 0, "none">,
      I32EnumAttrCase<"pair_step", 1, "pair_step">,
      I32EnumAttrCase<"k_step", 2, "k_step">,
      I32EnumAttrCase<"full_timestep", 3, "full_timestep">
    ]>;
def ArtsPlanRepetitionStructureAttr
    : Arts_I32EnumAttr<"plan_repetition_structure",
                       ArtsPlanRepetitionStructure>;

def ArtsPlanAsyncStrategy
    : Arts_I32Enum<"ArtsPlanAsyncStrategy",
                   "Core ARTS async-loop strategy", [
      I32EnumAttrCase<"blocking", 0, "blocking">,
      I32EnumAttrCase<"advance_edt", 1, "advance_edt">,
      I32EnumAttrCase<"cps_chain", 2, "cps_chain">
    ]>;
def ArtsPlanAsyncStrategyAttr
    : Arts_I32EnumAttr<"plan_async_strategy", ArtsPlanAsyncStrategy>;

def ArtsBarrierReason
    : Arts_I32Enum<"ArtsBarrierReason",
                   "Core ARTS barrier preservation or elimination reason", [
      I32EnumAttrCase<"redundant", 0, "redundant">,
      I32EnumAttrCase<"required_memory", 1, "required_memory">,
      I32EnumAttrCase<"timestep_stage_boundary", 2,
                      "timestep_stage_boundary">,
      I32EnumAttrCase<"wavefront_frontier", 3, "wavefront_frontier">,
      I32EnumAttrCase<"unknown_required", 4, "unknown_required">
    ]>;
def ArtsBarrierReasonAttr
    : Arts_I32EnumAttr<"barrier_reason", ArtsBarrierReason>;

/// Enum info for runtime query kinds (consolidated from 5 individual ops)
def RuntimeQueryKind
    : Arts_I32Enum<"RuntimeQueryKind", "ARTS runtime query kind", [
      I32EnumAttrCase<"totalWorkers", 0, "total_workers">,
      I32EnumAttrCase<"totalNodes", 1, "total_nodes">,
      I32EnumAttrCase<"currentWorker", 2, "current_worker">,
      I32EnumAttrCase<"currentNode", 3, "current_node">,
      I32EnumAttrCase<"parallelWorkerId", 4, "parallel_worker_id">
    ]>;

/// Enum attribute for runtime query kinds
def RuntimeQueryKindAttr : Arts_I32EnumAttr<"runtime_query_kind", RuntimeQueryKind>;

/// Enum info for EDT concurrency types
def EdtConcurrency
    : Arts_I32Enum<"EdtConcurrency", "ARTS EDT concurrency type", [
      I32EnumAttrCase<"intranode", 0, "intranode">,
      I32EnumAttrCase<"internode", 1, "internode">
    ]>;

/// Enum attribute for EDT concurrency types
def EdtConcurrencyAttr : Arts_I32EnumAttr<"concurrency", EdtConcurrency>;

/// Enum info for DB partition modes
/// Used to track how a DbAllocOp was partitioned and how DbAcquireOp accesses:
///   - coarse: Single DB for entire array (outerRank=0), full array access
///   - block: Fixed-size blocks, contiguous block by offset/size
///   - fine_grained: One DB per element/row (outerRank>0), element-wise access
///   - stencil: Halo pattern access (offset + size for neighboring elements)
def PartitionMode
    : Arts_I32Enum<"PartitionMode", "DB allocation partition mode", [
      I32EnumAttrCase<"coarse", 0, "coarse">,
      I32EnumAttrCase<"block", 1, "block">,
      I32EnumAttrCase<"fine_grained", 2, "fine_grained">,
      I32EnumAttrCase<"stencil", 3, "stencil">
    ]>;

/// Enum attribute for DB partition modes
def PartitionModeAttr : Arts_I32EnumAttr<"partition_mode", PartitionMode>;

/// DB access pattern classification (set by analysis, read by ForLowering).
def DbAccessPattern
    : Arts_I32Enum<"DbAccessPattern", "DB access pattern classification", [
      I32EnumAttrCase<"unknown", 0, "unknown">,
      I32EnumAttrCase<"uniform", 1, "uniform">,
      I32EnumAttrCase<"stencil", 2, "stencil">,
      I32EnumAttrCase<"indexed", 3, "indexed">
    ]>;
def DbAccessPatternAttr : Arts_I32EnumAttr<"access_pattern", DbAccessPattern>;

/// Distribution strategy selected for an arts.for loop by EdtDistribution pass.
def EdtDistributionKind
    : Arts_I32Enum<"EdtDistributionKind", "EDT distribution strategy kind", [
      I32EnumAttrCase<"block", 0, "block">,
      I32EnumAttrCase<"two_level", 1, "two_level">,
      I32EnumAttrCase<"block_cyclic", 2, "block_cyclic">,
      I32EnumAttrCase<"tiling_2d", 3, "tiling_2d">,
      I32EnumAttrCase<"replicated", 4, "replicated">
    ]>;
def EdtDistributionKindAttr
    : Arts_I32EnumAttr<"distribution_kind", EdtDistributionKind>;

/// Loop pattern classification used for strategy selection diagnostics.
def EdtDistributionPattern
    : Arts_I32Enum<"EdtDistributionPattern", "EDT loop pattern kind", [
      I32EnumAttrCase<"unknown", 0, "unknown">,
      I32EnumAttrCase<"uniform", 1, "uniform">,
      I32EnumAttrCase<"stencil", 2, "stencil">,
      I32EnumAttrCase<"matmul", 3, "matmul">,
      I32EnumAttrCase<"triangular", 4, "triangular">
    ]>;
def EdtDistributionPatternAttr
    : Arts_I32EnumAttr<"distribution_pattern", EdtDistributionPattern>;

/// Canonical dep/schedule family recognized by ARTS transforms.
/// Unlike distribution_pattern, this records the semantic rewrite family so
/// downstream passes do not need to rediscover it from raw memory accesses.
def ArtsDepPattern
    : Arts_I32Enum<"ArtsDepPattern",
                   "Canonical dep/schedule family", [
      I32EnumAttrCase<"unknown", 0, "unknown">,
      I32EnumAttrCase<"uniform", 1, "uniform">,
      I32EnumAttrCase<"stencil", 2, "stencil">,
      I32EnumAttrCase<"matmul", 3, "matmul">,
      I32EnumAttrCase<"triangular", 4, "triangular">,
      I32EnumAttrCase<"wavefront_2d", 5, "wavefront_2d">,
      I32EnumAttrCase<"jacobi_alternating_buffers", 6,
                      "jacobi_alternating_buffers">,
      I32EnumAttrCase<"elementwise_pipeline", 7, "elementwise_pipeline">,
      I32EnumAttrCase<"stencil_tiling_nd", 8, "stencil_tiling_nd">,
      I32EnumAttrCase<"cross_dim_stencil_3d", 9, "cross_dim_stencil_3d">,
      I32EnumAttrCase<"higher_order_stencil", 10, "higher_order_stencil">
    ]>;
def ArtsDepPatternAttr
    : Arts_I32EnumAttr<"dep_pattern", ArtsDepPattern>;

/// Integer attribute for storing number of workers for EDT operations
def WorkersAttr : AttrDef<ArtsDialect, "workers"> {
  let summary = "Number of parallel workers for EDT operations";
  let description = [{
    Specifies the number of parallel workers to use for executing an EDT.
    This attribute is set by the concurrency pass based on the abstract machine
    configuration and determines how many task instances to create.
  }];
  let parameters = (ins "int32_t" : $value);
  let mnemonic = "workers";
  let assemblyFormat = "`<` $value `>`";
}

/// Unit attribute to skip EDT verification during intermediate passes
def NoVerifyAttr : AttrDef<ArtsDialect, "NoVerify"> {
  let summary = "Skip verification for partially formed EDT";
  let description = [{
    Marks an EDT that should skip verification. This is used during
    ConvertOpenMPToArts before CreateDbPass properly sets up dependencies
    and block arguments. The attribute is removed after CreateDbPass.
  }];
  let mnemonic = "no_verify";
  let assemblyFormat = "";
}

/// Marker on DbAcquireOp to preserve authoritative access-mode semantics.
/// This only protects the acquire's in/out/inout mode from later
/// re-inference/optimization. It does not by itself imply that the dependency
/// edge must stay live.
def PreserveAccessModeAttr : AttrDef<ArtsDialect, "PreserveAccessMode"> {
  let summary = "Preserve explicit access-mode semantics on a DbAcquireOp";
  let description = [{
    Marks DbAcquireOps whose access mode should be treated as an explicit
    dependency contract rather than inferred from loads/stores. This attribute
    is only set when the compiler already knows the exact mode and later passes
    should not re-check or optimize it.
  }];
  let mnemonic = "preserve_access_mode";
  let assemblyFormat = "";
}

/// Marker on DbAcquireOp to preserve an explicit dependency edge.
/// This is only set for acquires materialized from explicit DbControlOp/task
/// dependency contracts whose EDT dependency must not be DCE'd away just
/// because the local pointer/block argument is otherwise unused.
def PreserveDepEdgeAttr : AttrDef<ArtsDialect, "PreserveDepEdge"> {
  let summary = "Preserve explicit dependency edge semantics on a DbAcquireOp";
  let description = [{
    Marks DbAcquireOps whose EDT dependency edge is semantically required even
    if the acquired pointer has no local memory uses. This is used for
    explicit task-control dependencies and is intentionally separate from
    `preserve_access_mode`, which only protects access-mode inference.
  }];
  let mnemonic = "preserve_dep_edge";
  let assemblyFormat = "";
}

/// Canonical semantic pattern payload shared by lowering contracts.
def PatternAttr : Arts_Attr<"Pattern", "pattern"> {
  let summary = "Canonical ARTS semantic pattern payload";
  let description = [{
    Bundles the semantic pattern family and distribution selection into one
    typed attribute so lowering contracts do not need to spread the same
    meaning across unrelated top-level attrs.
  }];
  let parameters = (ins
    OptionalParameter<"::mlir::arts::ArtsDepPatternAttr">:$depPattern,
    OptionalParameter<"::mlir::arts::EdtDistributionKindAttr">:$distributionKind,
    OptionalParameter<"::mlir::arts::EdtDistributionPatternAttr">:$distributionPattern,
    OptionalParameter<"::mlir::IntegerAttr">:$distributionVersion,
    OptionalParameter<"::mlir::IntegerAttr">:$revision
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

/// Canonical static lowering-contract payload shared by lowering contracts.
def ContractAttr : Arts_Attr<"Contract", "contract"> {
  let summary = "Canonical ARTS lowering-contract payload";
  let description = [{
    Bundles static lowering-contract refinements such as owner dimensions,
    stencil center/halo semantics, and post-DB analysis flags into one typed
    attribute. Dynamic quantities remain SSA operands on
    `arts.lowering_contract`.
  }];
  let parameters = (ins
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$ownerDims,
    OptionalParameter<"::mlir::IntegerAttr">:$centerOffset,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$spatialDims,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$stencilIndependentDims,
    OptionalParameter<"::mlir::BoolAttr">:$supportedBlockHalo,
    OptionalParameter<"::mlir::BoolAttr">:$narrowableDep,
    OptionalParameter<"::mlir::BoolAttr">:$postDbRefined,
    OptionalParameter<"::mlir::IntegerAttr">:$criticalPathDistance,
    OptionalParameter<"::mlir::IntegerAttr">:$contractKind
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

///===----------------------------------------------------------------------===///
// ARTS Metadata Attributes
///===----------------------------------------------------------------------===///

/// Attribute to store dimensional analysis results for flattened memrefs
def MemrefDimInfoAttr : Arts_Attr<"MemrefDimInfo", "memref_dim_info"> {
  let summary = "Recovered dimensional information for flattened memrefs";
  let description = [{
    Stores inferred multi-dimensional shape and affine access patterns
    for array allocations that were originally flattened (e.g., array[N*M]).
    This enables canonicalization of memref<N*M> to memref<NxM> with proper
    multi-dimensional indexing.

    The attribute captures:
    - inferredDims: The recovered dimensions (e.g., [128, 128] for N=128)
    - strideExprs: Stride multipliers for each dimension (e.g., [128, 1])
    - accessPattern: Affine map showing linearization pattern
  }];
  let parameters = (ins "::mlir::ArrayAttr"
                    : $inferredDims, "::mlir::ArrayAttr"
                    : $strideExprs, "::mlir::AffineMapAttr"
                    : $accessPattern);
  let mnemonic = "memref_dim_info";
  let assemblyFormat =
      "`<` $inferredDims `,` $strideExprs `,` $accessPattern `>`";
}

#endif // ARTS_ATTRS
