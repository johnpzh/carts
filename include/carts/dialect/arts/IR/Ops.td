///==========================================================================///
/// File: Ops.td
/// This file defines multiple operations for the ARTS Dialect.
///==========================================================================///

#ifndef ARTS_OPS
#define ARTS_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"

/// ARTS Dialect
include "Dialect.td"
include "Attributes.td"
include "Types.td"

///===----------------------------------------------------------------------===///
/// Custom Traits for ARTS
///===----------------------------------------------------------------------===///

class Arts_InferShapedTypeOpAdaptor<string mnemonic, list<Trait> traits = []>
    : Arts_Op<mnemonic, !listconcat(traits, [InferShapedTypeOpAdaptor, Pure])> {
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

///===----------------------------------------------------------------------===///
/// arts.undef
///===----------------------------------------------------------------------===///
def UndefOp : Arts_Op<"undef", [Pure]> {
  let summary = "More flexible undef op";
  let results = (outs AnyType : $result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Type"
                             : $type),
                            [{ $_state.addTypes(type); }]>];
}

///===----------------------------------------------------------------------===///
/// arts.alloc
///===----------------------------------------------------------------------===///
def AllocOp : Arts_Op<"alloc", []> {
  let summary = "Allocate a memref in ARTS dialect";
  let description = [{
    The `arts.alloc` operation allocates a memref with a given shape.
    It mirrors `memref.alloc` but is customized for the ARTS dialect to enable
    specific behavior during code generation and lowering.
  }];

  let arguments = (ins Variadic<Index> : $dynamicSizes);
  let results = (outs AnyMemRef : $result);
  let assemblyFormat =
      [{
    `(` $dynamicSizes `)` attr-dict `:` type($result)}];
}

///===----------------------------------------------------------------------===///
/// Arts DB Operations
///===----------------------------------------------------------------------===///
def DbDimOp : Arts_Op<"db_dim", [Pure]> {
  let summary = "Query dimension of a datablock-backed memref";
  let description =
      [{The `arts.db_dim` op queries the size of the specified dimension of a
            datablock -
        backed memref.It is preferred over `memref.dim` so DB -
        oriented canonicalization can fold it to explicit values coming
            from `arts.db_alloc` or `arts.db_acquire` slice sizes.}];
  let arguments = (ins AnyMemRef : $source, Index : $dim);
  let results = (outs Index : $result);
  let assemblyFormat =
      "$source `,` $dim attr-dict `:` type($source) `->` type($result)";
  let hasCanonicalizer = 1;
  let builders = [OpBuilder<(ins "Value" : $source, "int64_t" : $dim)>];
}

def DbAllocOp : Arts_Op<"db_alloc",
                        [AttrSizedOperandSegments, MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate an ARTS DataBlock and get its pointer and GUID.";
  let description = [{
    This operation corresponds to the `artsDbCreate` runtime function.
    It allocates a DataBlock and returns its pointer and GUID.
    - mode: enum for
    "in", "out",
        "inout" -
            ptr
        : address that will be copied to the DataBlock -
          sizes : outer sizes(number of datablocks along each dimension) -
                  elementSizes
        : logical dimensions of the datablock element payload -
          allocType : "stack",
        "heap", "global", "unknown" - tracks original allocation type
  }];
  let arguments = (ins
    ArtsModeAttr:$mode,
    I32:$route,
    DbAllocTypeAttr:$allocType,
    DbModeAttr:$dbMode,
    TypeAttr:$elementType,
    Optional<AnyMemRef>:$address,
    Variadic<Index>:$sizes,
    Variadic<Index>:$elementSizes,
    OptionalAttr<UnitAttr>:$distributed,
    OptionalAttr<PartitionModeAttr>:$partition_mode,
    OptionalAttr<DbAccessPatternAttr>:$access_pattern,
    OptionalAttr<ArtsDepPatternAttr>:$depPattern,
    OptionalAttr<EdtDistributionKindAttr>:$distribution_kind,
    OptionalAttr<EdtDistributionPatternAttr>:$distribution_pattern,
    OptionalAttr<I32Attr>:$distribution_version,
    OptionalAttr<I64Attr>:$stencil_center_offset,
    OptionalAttr<ArrayAttr>:$stencil_min_offsets,
    OptionalAttr<ArrayAttr>:$stencil_max_offsets,
    OptionalAttr<ArrayAttr>:$stencil_spatial_dims,
    OptionalAttr<ArrayAttr>:$stencil_owner_dims,
    OptionalAttr<ArrayAttr>:$stencil_block_shape,
    OptionalAttr<ArrayAttr>:$stencil_write_footprint,
    OptionalAttr<UnitAttr>:$stencil_supported_block_halo,
    OptionalAttr<ArrayAttr>:$planOwnerDims,
    OptionalAttr<ArrayAttr>:$planPhysicalBlockShape,
    OptionalAttr<ArrayAttr>:$planLogicalWorkerSlice,
    OptionalAttr<ArrayAttr>:$planHaloShape,
    OptionalAttr<ArtsPlanIterationTopologyAttr>:$planIterationTopology,
    OptionalAttr<ArtsPlanRepetitionStructureAttr>:$planRepetitionStructure,
    OptionalAttr<ArtsPlanAsyncStrategyAttr>:$planAsyncStrategy,
    OptionalAttr<I64Attr>:$planCostSchedulerOverhead,
    OptionalAttr<I64Attr>:$planCostSliceWideningPressure,
    OptionalAttr<I64Attr>:$planCostExpectedLocalWork,
    OptionalAttr<I64Attr>:$planCostRelaunchAmortization
  );
  let results = (outs MemRefOf<[I64]> : $guid, AnyMemRef : $ptr);
  let assemblyFormat = [{
    `[` $mode `,` $allocType `,` $dbMode (`,` $partition_mode^)? (`,` $access_pattern^)? `]`
    `route` `(` $route `:` type($route) `)`
    ( `(` $address^ `:` type($address) `)` )?
    `sizes` `[` $sizes `]`
    `elementType` `(` $elementType `)`
    `elementSizes` `[` $elementSizes `]`
    attr-dict `:` `(` type($guid) `,` type($ptr) `)`
  }];
  let builders = [
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$route,
                   "DbAllocType":$allocType,
                   "DbMode":$dbMode,
                   "Type":$elementType,
                   CArg<"Value", "nullptr">:$address,
                   CArg<"SmallVector<Value>", "{}">:$sizes,
                   CArg<"SmallVector<Value>", "{}">:$elementSizes)>,
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$route,
                   "DbAllocType":$allocType,
                   "DbMode":$dbMode,
                   "Type":$elementType,
                   CArg<"SmallVector<Value>", "{}">:$sizes,
                   CArg<"SmallVector<Value>", "{}">:$elementSizes)>,
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$route,
                   "DbAllocType":$allocType,
                   "DbMode":$dbMode,
                   "Type":$elementType,
                   "Type":$pointerType,
                   CArg<"SmallVector<Value>", "{}">:$sizes,
                   CArg<"SmallVector<Value>", "{}">:$elementSizes)>,
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$route,
                   "DbAllocType":$allocType,
                   "DbMode":$dbMode,
                   "Type":$elementType,
                   CArg<"Value", "nullptr">:$address,
                   CArg<"SmallVector<Value>", "{}">:$sizes,
                   CArg<"SmallVector<Value>", "{}">:$elementSizes,
                   CArg<"PartitionMode", "PartitionMode::coarse">:$partitionMode)>,
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$route,
                   "DbAllocType":$allocType,
                   "DbMode":$dbMode,
                   "Type":$elementType,
                   CArg<"SmallVector<Value>", "{}">:$sizes,
                   CArg<"SmallVector<Value>", "{}">:$elementSizes,
                   CArg<"PartitionMode", "PartitionMode::coarse">:$partitionMode)>,
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$route,
                   "DbAllocType":$allocType,
                   "DbMode":$dbMode,
                   "Type":$elementType,
                   "Type":$pointerType,
                   CArg<"SmallVector<Value>", "{}">:$sizes,
                   CArg<"SmallVector<Value>", "{}">:$elementSizes,
                   CArg<"PartitionMode", "PartitionMode::coarse">:$partitionMode)>
  ];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    MemRefType getAllocatedElementType();
  }];
}

def DbAcquireOp : Arts_Op<"db_acquire", [AttrSizedOperandSegments]> {
  let summary = "Acquires access to an ARTS DataBlock.";
  let description = [{
    This operation acquires a local view of the DataBlock for safe access within EDTs.
    It subsumes dependency creation with mode and subregion support.
    - mode: enum for "in", "out", "inout"
    - sourceGuid: GUID memref from DbAllocOp or previous DbAcquireOp (optional)
    - sourcePtr: Ptr memref from DbAllocOp or previous DbAcquireOp

    DB-space fields: indices, offsets and sizes

    Partition hints (element-space, consumed by task/DB-space lowering):
    - partition_mode: coarse, chunked, fine_grained, or stencil - describes access pattern
    - partition_indices: element indices for fine_grained access (from depend(inout: A[i][j]))
    - partition_offsets: element offsets for chunked/stencil access (from depend(in: A[off:sz]))
    - partition_sizes: element sizes for chunked/stencil access

    Multiple partition entries (for multiple depend clause entries):
    When an OpenMP task has multiple depend entries like `depend(in: A[i][j], A[i-1][j])`,
    each entry becomes a separate partition entry. The segment attributes track how many
    values belong to each entry:
    - partition_indices_segments: [2, 2] means entry 0 has 2 indices, entry 1 has 2 indices
    - partition_entry_modes: per-entry partition mode (fine_grained, chunked, etc.)

    When source_guid is absent (nullptr), it means this acquire is chaining from
    a parent EDT's acquire where only the ptr was passed through block arguments.

    ESD partial acquisition (optional, for stencil halo fetching):
    - element_offsets: Offset within the chunk (element-level, not byte-level)
    - element_sizes: Size within the chunk (element-level)
    When these are set, EdtLowering computes byte_offset/byte_size using the
    allocation's elementSizes and emits the runtime dependency-recording op
    with byte_offsets for ESD (Ephemeral Slice Dependencies) partial chunk
    acquisition.

    Returns (guid, ptr) tuple for the acquired datablock.
  }];
  let arguments = (ins
    ArtsModeAttr:$mode,
    Optional<MemRefOf<[I64]>>:$source_guid,
    AnyMemRef:$source_ptr,
    OptionalAttr<PartitionModeAttr>:$partition_mode,
    Variadic<Index>:$indices,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$partition_indices,
    Variadic<Index>:$partition_offsets,
    Variadic<Index>:$partition_sizes,
    Optional<I1>:$bounds_valid,
    Variadic<Index>:$element_offsets,
    Variadic<Index>:$element_sizes,
    OptionalAttr<PreserveAccessModeAttr>:$preserve_access_mode,
    OptionalAttr<PreserveDepEdgeAttr>:$preserve_dep_edge,
    OptionalAttr<UnitAttr>:$replicatedRead,
    OptionalAttr<ArtsDepPatternAttr>:$depPattern,
    OptionalAttr<EdtDistributionKindAttr>:$distribution_kind,
    OptionalAttr<EdtDistributionPatternAttr>:$distribution_pattern,
    OptionalAttr<I32Attr>:$distribution_version,
    OptionalAttr<I64Attr>:$stencil_center_offset,
    OptionalAttr<ArrayAttr>:$stencil_min_offsets,
    OptionalAttr<ArrayAttr>:$stencil_max_offsets,
    OptionalAttr<ArrayAttr>:$stencil_spatial_dims,
    OptionalAttr<ArrayAttr>:$stencil_owner_dims,
    OptionalAttr<ArrayAttr>:$stencil_block_shape,
    OptionalAttr<ArrayAttr>:$stencil_write_footprint,
    OptionalAttr<UnitAttr>:$stencil_supported_block_halo,
    /// Segment attributes for multiple partition entries
    OptionalAttr<DenseI32ArrayAttr>:$partition_indices_segments,
    OptionalAttr<DenseI32ArrayAttr>:$partition_offsets_segments,
    OptionalAttr<DenseI32ArrayAttr>:$partition_sizes_segments,
    /// Per-entry partition modes
    OptionalAttr<DenseI32ArrayAttr>:$partition_entry_modes
  );
  let results = (outs MemRefOf<[I64]> : $guid, AnyMemRef : $ptr);
  let assemblyFormat = [{
    `[` $mode `]`
    `(` ($source_guid^ `:` type($source_guid) `,`)? $source_ptr `:` type($source_ptr) `)`
    ( `partitioning` `(`
        $partition_mode^
        (`,` `indices` `[` $partition_indices^ `]`)?
        (`,` `offsets` `[` $partition_offsets^ `]`)?
        (`,` `sizes` `[` $partition_sizes^ `]`)?
      `)`
    )?
    `,` `indices` `[` $indices `]`
    (`,` `offsets` `[` $offsets^ `]`)?
    (`,` `sizes` `[` $sizes^ `]`)?
    ( `bounds_valid` `(` $bounds_valid^ `)` )?
    ( `element_offsets` `[` $element_offsets^ `]` )?
    ( `element_sizes` `[` $element_sizes^ `]` )?
    attr-dict `->` `(` type($guid) `,` type($ptr) `)`
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    /// Builder for CreateDbs: sets partition hints, leaves DB-space empty
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$source_guid,
                   "Value":$source_ptr,
                   CArg<"std::optional<PartitionMode>", "std::nullopt">:$partitionMode,
                   CArg<"SmallVector<Value>", "{}">:$indices,
                   CArg<"SmallVector<Value>", "{}">:$offsets,
                   CArg<"SmallVector<Value>", "{}">:$sizes,
                   CArg<"SmallVector<Value>", "{}">:$partition_indices,
                   CArg<"SmallVector<Value>", "{}">:$partition_offsets,
                   CArg<"SmallVector<Value>", "{}">:$partition_sizes,
                   CArg<"Value", "{}">:$bounds_valid,
                   CArg<"SmallVector<Value>", "{}">:$element_offsets,
                   CArg<"SmallVector<Value>", "{}">:$element_sizes)>,
    /// Builder with partition_mode and explicit ptrType
    OpBuilder<(ins "ArtsMode":$mode,
                   "Value":$source_guid,
                   "Value":$source_ptr,
                   "Type":$ptrType,
                   CArg<"std::optional<PartitionMode>", "std::nullopt">:$partitionMode,
                   CArg<"SmallVector<Value>", "{}">:$indices,
                   CArg<"SmallVector<Value>", "{}">:$offsets,
                   CArg<"SmallVector<Value>", "{}">:$sizes,
                   CArg<"SmallVector<Value>", "{}">:$partition_indices,
                   CArg<"SmallVector<Value>", "{}">:$partition_offsets,
                   CArg<"SmallVector<Value>", "{}">:$partition_sizes,
                   CArg<"Value", "{}">:$bounds_valid,
                   CArg<"SmallVector<Value>", "{}">:$element_offsets,
                   CArg<"SmallVector<Value>", "{}">:$element_sizes)>
  ];
  let extraClassDeclaration = [{
    /// Check if this acquire has element-level offset/size info (ESD partial
    /// acquisition)
    bool hasElementOffsets() { return !getElementOffsets().empty(); }

    /// Check if this acquire has a partition mode set
    bool hasPartitionMode() { return getPartitionMode().has_value(); }

    /// Get partition mode, returns coarse if not set
    PartitionMode getPartitionModeOr(PartitionMode defaultMode =
                                         PartitionMode::coarse) {
      return getPartitionMode().value_or(defaultMode);
    }

    /// Check if this is a fine-grained (element-wise) acquire
    bool isFineGrained() {
      return getPartitionModeOr() == PartitionMode::fine_grained;
    }

    /// Check if this is a block acquire
    bool isBlock() { return getPartitionModeOr() == PartitionMode::block; }

    /// Check if this is a stencil acquire
    bool isStencil() { return getPartitionModeOr() == PartitionMode::stencil; }

    /// Check if this is a coarse acquire
    bool isCoarse() { return getPartitionModeOr() == PartitionMode::coarse; }

    /// Check if this acquire has partition hints (element-space
    /// indices/offsets/sizes)
    bool hasPartitionHints() { return hasExplicitPartitionHints(); }

    /// Mark or clear the authoritative dependency-mode contract carried by
    /// this acquire.
    void setPreserveAccessMode(bool preserve = true);

    /// Mark or clear the explicit dependency-edge contract carried by
    /// this acquire.
    void setPreserveDepEdge(bool preserve = true);

    /// Mark or clear the full explicit dependency contract carried by this
    /// acquire. This preserves both the authoritative mode and the edge
    /// itself.
    void setExplicitDepContract(bool preserve = true);

    /// Attach or clear the semantic dep family for this acquire.
    void setDepPattern(ArtsDepPattern pattern);
    void clearDepPattern();

    /// Check if this acquire has explicit partition hints (including sizes)
    bool hasExplicitPartitionHints();

    /// Copy partition segment metadata from another acquire
    void copyPartitionSegmentsFrom(DbAcquireOp source);

    //===------------------------------------------------------------------===//
    // Multiple Partition Entry Support
    //===------------------------------------------------------------------===//

    /// Check if this acquire has multiple partition entries
    bool hasMultiplePartitionEntries();

    /// Get number of partition entries.
    size_t getNumPartitionEntries();

    /// Get the partition mode for a specific entry (uses partition_entry_modes
    /// if set)
    PartitionMode getPartitionEntryMode(size_t entryIdx);

    /// Get partition indices for a specific entry
    SmallVector<Value> getPartitionIndicesForEntry(size_t entryIdx);

    /// Get partition offsets for a specific entry
    SmallVector<Value> getPartitionOffsetsForEntry(size_t entryIdx);

    /// Get partition sizes for a specific entry
    SmallVector<Value> getPartitionSizesForEntry(size_t entryIdx);

    /// Set segment attributes for multi-entry partition hints.
    /// Only sets non-empty arrays, skips empty ones.
    void setPartitionSegments(
        ArrayRef<int32_t> indicesSegments, ArrayRef<int32_t> offsetsSegments,
        ArrayRef<int32_t> sizesSegments, ArrayRef<int32_t> entryModes);

    /// Get all partition entries as unified PartitionInfo structs.
    /// Returns one PartitionInfo per depend entry.
    SmallVector<PartitionInfo> getPartitionInfos();

    /// Clear all partition hints (operands and segment attributes).
    void clearPartitionHints();

    /// Check if all partition entries have fine_grained mode.
    /// Returns true if hasMultiplePartitionEntries() and all entry modes are
    /// fine_grained. Used to distinguish task dependency patterns from stencil
    /// patterns.
    bool hasAllFineGrainedEntries();
  }];
}

def LoweringContractOp : Arts_Op<"lowering_contract",
                                 [AttrSizedOperandSegments]> {
  let summary = "First-class lowering contract associated with a DB value";
  let description = [{
    `arts.lowering_contract` carries the high-level partitioning, stencil, and
    distribution contract selected by upstream pattern detection and contract
    emission passes. The contract is attached to a datablock pointer-like
    memref value and is the single downstream source of truth for specialized
    lowering behavior.

    Graph analyses may cache or refine derived facts while the pipeline
    evolves, but those facts are implementation detail. Any semantic decision
    that downstream passes depend on must be expressible in this contract
    vocabulary rather than in graph-only side channels.

    Static fields remain attributes when they are genuinely compile-time
    constants (for example `owner_dims` or the dep/distribution family).
    Potentially dynamic quantities are represented as SSA operands so the
    contract survives through structural rewrites without being recomputed or
    forced into attrs.
  }];
  let arguments = (ins
    AnyMemRef:$target,
    Variadic<Index>:$block_shape,
    Variadic<Index>:$min_offsets,
    Variadic<Index>:$max_offsets,
    Variadic<Index>:$write_footprint,
    OptionalAttr<PatternAttr>:$pattern,
    OptionalAttr<ContractAttr>:$contract
  );
  let assemblyFormat = [{
    `(` $target `:` type($target) `)`
    (`pattern` `(` $pattern^ `)` )?
    (`block_shape` `[` $block_shape^ `]` )?
    (`min_offsets` `[` $min_offsets^ `]` )?
    (`max_offsets` `[` $max_offsets^ `]` )?
    (`write_footprint` `[` $write_footprint^ `]` )?
    (`contract` `(` $contract^ `)` )?
    attr-dict
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static void build(OpBuilder &builder, OperationState &state, Value target,
                      const LoweringContractInfo &info);
    static LoweringContractOp create(OpBuilder &builder, Location location,
                                     Value target,
                                     const LoweringContractInfo &info);
    std::optional<ArtsDepPattern> getDepPattern();
    std::optional<EdtDistributionKind> getDistributionKind();
    std::optional<EdtDistributionPattern> getDistributionPattern();
    std::optional<int64_t> getDistributionVersion();
    std::optional<int64_t> getPatternRevision();
    std::optional<SmallVector<int64_t, 4>> getOwnerDims();
    std::optional<int64_t> getCenterOffset();
    std::optional<SmallVector<int64_t, 4>> getSpatialDims();
    std::optional<SmallVector<int64_t, 4>> getStencilIndependentDims();
    std::optional<bool> getSupportedBlockHalo();
    std::optional<bool> getNarrowableDep();
    std::optional<bool> getPostDbRefined();
    std::optional<int64_t> getCriticalPathDistance();
    std::optional<int64_t> getContractKind();
  }];
  let builders = [
    OpBuilder<(ins "Value":$target,
                   CArg<"std::optional<ArtsDepPattern>", "std::nullopt">:$depPattern,
                   CArg<"std::optional<EdtDistributionKind>", "std::nullopt">:$distributionKind,
                   CArg<"std::optional<EdtDistributionPattern>", "std::nullopt">:$distributionPattern,
                   CArg<"std::optional<int64_t>", "std::nullopt">:$distributionVersion,
                   CArg<"std::optional<int64_t>", "std::nullopt">:$revision,
                   CArg<"SmallVector<int64_t>", "{}">:$ownerDims,
                   CArg<"SmallVector<Value>", "{}">:$blockShape,
                   CArg<"SmallVector<Value>", "{}">:$minOffsets,
                   CArg<"SmallVector<Value>", "{}">:$maxOffsets,
                   CArg<"SmallVector<Value>", "{}">:$writeFootprint,
                   CArg<"std::optional<int64_t>", "std::nullopt">:$centerOffset,
                   CArg<"bool", "false">:$supportedBlockHalo,
                   CArg<"SmallVector<int64_t>", "{}">:$spatialDims,
                   CArg<"SmallVector<int64_t>", "{}">:$stencilIndependentDims,
                   CArg<"bool", "false">:$narrowableDep,
                   CArg<"bool", "false">:$postDbRefined,
                   CArg<"std::optional<int64_t>", "std::nullopt">:$criticalPathDistance,
                   CArg<"std::optional<int64_t>", "std::nullopt">:$contractKind)>
  ];
}

def DbRefOp : Arts_Op<"db_ref", [Pure]> {
  let summary = "References an outer datablock element.";
  let description = [{
    References an element from a datablock handle using only the outer
    dimensions. The result type must match the element type of the source
    memref.
  }];
  let arguments = (ins AnyMemRef : $source, Variadic<Index> : $indices);
  let results = (outs AnyType : $result);
  let assemblyFormat = [{
    $source `[` $indices `]` attr-dict `:` type($source) `->` type($result)
  }];
  let builders = [
    OpBuilder<(ins "Value":$source,
                   CArg<"ArrayRef<Value>", "{}">:$indices)>
  ];
  let hasVerifier = 1;
}

def DbReleaseOp : Arts_Op<"db_release", []> {
  let summary = "Releases access to an ARTS DataBlock source.";
  let description = [{
    This operation releases the acquired source, signaling completion to the runtime.
    - release_type: specifies whether to perform memory freeing for dynamic allocations
  }];
  let arguments = (ins
    AnyMemRef:$source,
    OptionalAttr<DbReleaseTypeAttr>:$release_type
  );
  let assemblyFormat = [{
    `(` $source `)` `:` type($source) attr-dict}];

  let builders = [
    OpBuilder<(ins "Value"
               : $source)>,
    OpBuilder<(ins "Value"
               : $source, "DbReleaseType"
               : $release_type)>
  ];
}

def DbFreeOp : Arts_Op<"db_free", []> {
  let summary = "Frees a DataBlock allocated by arts.db_alloc.";
  let description = [{
    This operation frees a DataBlock that was previously allocated with arts.db_alloc.
    It is used for memory management of dynamically allocated datablocks.
  }];
  let arguments = (ins AnyMemRef : $source);
  let assemblyFormat = [{
    `(` $source `)` `:` type($source) attr-dict}];
}

def DbNumElementsOp : Arts_Op<"db_num_elements", [Pure]> {
  let summary = "Compute the number of elements in a datablock.";
  let description = [{
    Computes the total number of elements in a datablock. This is used
    to calculate dependency counts for EDT creation.
  }];
  let arguments = (ins Variadic<Index> : $sizes);
  let results = (outs Index : $count);
  let assemblyFormat = "`sizes` `(` $sizes `)` attr-dict `:` `->` type($count)";
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins "DbAllocOp"
               : $allocOp)>,
    OpBuilder<(ins "DbAcquireOp"
               : $acquireOp)>
  ];
}

///===----------------------------------------------------------------------===///
/// arts.edt
///===----------------------------------------------------------------------===///
def EdtOp : Arts_Op<"edt", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"arts::YieldOp">, RecursiveMemoryEffects
]> {
  let summary = "Event-Driven Task in ARTS";
  let description = [{
    Defines an ARTS Event-Driven Task (EDT).

    An EDT encapsulates a unit of asynchronous work that operates on
    DataBlocks (DBs) through explicit memory dependencies.

    ### Region Structure

    The operation has a single-block region (`$body`) terminated by an
    implicit `arts.yield`. The body region contains the task code that
    executes when all dependencies are satisfied.

    ### Block Arguments and Dependencies

    The body block starts with one block argument per dependency. Each
    dependency block argument corresponds positionally to one entry in the
    `$dependencies` variadic operand list. Any explicit scalar params follow
    the dependency block arguments and correspond positionally to `$params`:

    ```
    arts.edt task intranode route(%r)
        (%acq0, %acq1 : memref<...>, memref<...>)
        params(%p0 : index) {
    ^bb0(%arg0: memref<...>, %arg1: memref<...>, %p0_arg: index):
      // %arg0 corresponds to %acq0, %arg1 corresponds to %acq1
      // %p0_arg corresponds to %p0
      ...
      arts.yield
    }
    ```

    At runtime, block arguments receive the acquired views of the
    corresponding datablocks. The task body must use these block arguments
    (not the outer dependency values) to access datablock memory.

    ### Invariants (enforced by verifier)

    1. The number of block arguments must equal dependencies + params.
    2. Every dependency must be defined by a `DbAcquireOp`.
    3. Every explicit param must be a scalar integer, index, or float value.
    4. MemRef-typed operands inside the body must be one of:
       - A block argument of this EDT (backed by a DbAcquire).
       - Defined within the EDT body region itself.
       - An external `memref.alloca` (sunk into the EDT later in the pipeline).
       - A `DbAcquireOp` source handle (`sourceGuid` or `sourcePtr`)
         referencing an outer-scope datablock for nested view derivation.
    5. External DbAcquire values must not be used directly inside the body;
       they must be passed as dependencies and accessed via block arguments.
    6. Nonconstant scalar captures must be explicit params for every EDT.
    7. Nested `EdtOp`s are skipped during verification (they verify
       independently).

    ### Attributes

    - `type`: EDT execution type (`single`, `sync`, `task`).
      - `single`: guaranteed to run as a single task.
      - `sync`: synchronous execution.
      - `task`: general task.
    - `concurrency`: execution scope (`intranode` or `internode`).
    - `route`: routing value (I32) for task placement.
    - `nowait` (optional): indicates the EDT does not require an epoch
      wrapper (typically used when the EDT is already inside an epoch).
  }];
  let arguments = (ins
    EdtTypeAttr:$type,
    EdtConcurrencyAttr:$concurrency,
    I32:$route,
    OptionalAttr<UnitAttr>:$nowait,
    OptionalAttr<UnitAttr>:$storageBridgeCopy,
    UnitAttr:$inPlaceSafe,
    UnitAttr:$inPlaceSharedState,
    OptionalAttr<I64Attr>:$vectorizeWidth,
    OptionalAttr<I64Attr>:$unrollFactor,
    OptionalAttr<I64Attr>:$interleaveCount,
    OptionalAttr<ArtsDepPatternAttr>:$depPattern,
    OptionalAttr<EdtDistributionKindAttr>:$distribution_kind,
    OptionalAttr<EdtDistributionPatternAttr>:$distribution_pattern,
    OptionalAttr<I32Attr>:$distribution_version,
    OptionalAttr<I64Attr>:$stencil_center_offset,
    OptionalAttr<ArrayAttr>:$stencil_min_offsets,
    OptionalAttr<ArrayAttr>:$stencil_max_offsets,
    OptionalAttr<ArrayAttr>:$stencil_spatial_dims,
    OptionalAttr<ArrayAttr>:$stencil_owner_dims,
    OptionalAttr<ArrayAttr>:$stencil_block_shape,
    OptionalAttr<ArrayAttr>:$stencil_write_footprint,
    OptionalAttr<UnitAttr>:$stencil_supported_block_halo,
    OptionalAttr<ArrayAttr>:$planOwnerDims,
    OptionalAttr<ArrayAttr>:$planPhysicalBlockShape,
    OptionalAttr<ArrayAttr>:$planLogicalWorkerSlice,
    OptionalAttr<ArrayAttr>:$planHaloShape,
    OptionalAttr<ArtsPlanIterationTopologyAttr>:$planIterationTopology,
    OptionalAttr<ArtsPlanRepetitionStructureAttr>:$planRepetitionStructure,
    OptionalAttr<ArtsPlanAsyncStrategyAttr>:$planAsyncStrategy,
    OptionalAttr<ArtsReductionStrategyAttr>:$reductionStrategy,
    UnitAttr:$partialReduction,
    OptionalAttr<ArrayAttr>:$partialReductionDims,
    OptionalAttr<ArrayAttr>:$partialReductionOwnerDims,
    OptionalAttr<ArrayAttr>:$partialReductionDepResultDimMaps,
    UnitAttr:$partialReductionSplitRequired,
    OptionalAttr<ArrayAttr>:$partialReductionSplitDims,
    OptionalAttr<I64Attr>:$partialReductionSplitFactor,
    OptionalAttr<I64Attr>:$partialReductionSplitOwnerTaskCount,
    OptionalAttr<I64Attr>:$partialReductionSplitTargetWorkerCount,
    OptionalAttr<I64Attr>:$planCostSchedulerOverhead,
    OptionalAttr<I64Attr>:$planCostSliceWideningPressure,
    OptionalAttr<I64Attr>:$planCostExpectedLocalWork,
    OptionalAttr<I64Attr>:$planCostRelaunchAmortization,
    Variadic<AnyMemRef>:$dependencies,
    Variadic<AnyType>:$params
  );
  let regions = (region SizedRegion<1> : $body);
  let assemblyFormat = [{
    $type $concurrency `route` `(` $route `)`
    (`(` $dependencies^ `)` `:` type($dependencies))?
    (`params` `(` $params^ `:` type($params) `)`)?
    attr-dict-with-keyword
    $body
  }];
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "EdtType"
               : $type, "EdtConcurrency"
               : $concurrency)>,
    OpBuilder<(ins "EdtType"
               : $type, "EdtConcurrency"
               : $concurrency, "ValueRange"
               : $dependencies)>,
    OpBuilder<(ins "EdtType"
               : $type, "EdtConcurrency"
               : $concurrency, "ValueRange"
               : $dependencies, "ValueRange"
               : $params)>,
    OpBuilder<(ins "EdtType"
               : $type, "EdtConcurrency"
               : $concurrency, "Value"
               : $route, "ValueRange"
               : $dependencies)>,
    OpBuilder<(ins "EdtType"
               : $type, "EdtConcurrency"
               : $concurrency, "Value"
               : $route, "ValueRange"
               : $dependencies, "ValueRange"
               : $params)>
  ];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    SmallVector<Value> getDependenciesAsVector();
    void setDependencies(ValueRange newDeps);
    void appendDependency(Value dep);
  }];
}

///===----------------------------------------------------------------------===///
/// arts.epoch
///===----------------------------------------------------------------------===///
def EpochOp
    : Arts_Op<"epoch", [
      SingleBlockImplicitTerminator<"arts::YieldOp">,
      RecursiveMemoryEffects
    ]> {

  let summary = "Epoch region in ARTS";
  let description = [{
    An ARTS epoch region. It can spawn multiple tasks and
    synchronize them at the end of the epoch. Returns the epoch GUID
    for use by contained EDTs.
  }];
  let arguments = (ins
    OptionalAttr<ArtsDepPatternAttr>:$depPattern,
    OptionalAttr<EdtDistributionKindAttr>:$distribution_kind,
    OptionalAttr<EdtDistributionPatternAttr>:$distribution_pattern,
    OptionalAttr<I32Attr>:$distribution_version,
    OptionalAttr<I64Attr>:$stencil_center_offset,
    OptionalAttr<ArrayAttr>:$stencil_min_offsets,
    OptionalAttr<ArrayAttr>:$stencil_max_offsets,
    OptionalAttr<ArrayAttr>:$stencil_spatial_dims,
    OptionalAttr<ArrayAttr>:$stencil_owner_dims,
    OptionalAttr<ArrayAttr>:$stencil_block_shape,
    OptionalAttr<ArrayAttr>:$stencil_write_footprint,
    OptionalAttr<UnitAttr>:$stencil_supported_block_halo,
    OptionalAttr<ArrayAttr>:$planOwnerDims,
    OptionalAttr<ArrayAttr>:$planPhysicalBlockShape,
    OptionalAttr<ArrayAttr>:$planLogicalWorkerSlice,
    OptionalAttr<ArrayAttr>:$planHaloShape,
    OptionalAttr<ArtsPlanIterationTopologyAttr>:$planIterationTopology,
    OptionalAttr<ArtsPlanRepetitionStructureAttr>:$planRepetitionStructure,
    OptionalAttr<ArtsPlanAsyncStrategyAttr>:$planAsyncStrategy,
    OptionalAttr<I64Attr>:$planCostSchedulerOverhead,
    OptionalAttr<I64Attr>:$planCostSliceWideningPressure,
    OptionalAttr<I64Attr>:$planCostExpectedLocalWork,
    OptionalAttr<I64Attr>:$planCostRelaunchAmortization
  );
  let regions = (region SizedRegion<1> : $body);
  let results = (outs I64 : $epochGuid);

  let assemblyFormat =
      [{attr-dict-with-keyword $body `:` type($epochGuid)}];
  let builders = [
    OpBuilder<(ins)>,
    OpBuilder<(ins "Type":$epochGuid)>
  ];
}

/// arts.yield
///===----------------------------------------------------------------------===///
def YieldOp : Arts_Op<"yield", [
  Pure, ReturnLike, Terminator, ParentOneOf<["EdtOp", "EpochOp"]>
]> {

  let summary = "ARTS dialect yield terminator";
  let description = [{
    The `arts.yield` operation serves as a terminator for various 
    ARTS operations, such as `arts.epoch` and `arts.edt`. It ensures proper
    control flow within the ARTS dialect by marking the end of a region.
  }];
  let assemblyFormat = [{attr-dict-with-keyword}];
}

///===----------------------------------------------------------------------===///
/// arts.barrier
///===----------------------------------------------------------------------===///
def BarrierOp : Arts_Op<"barrier", [MemoryEffects<[MemWrite]>]> {
  let summary = "Barrier synchronization in ARTS";
  let description =
      [{Barrier synchronization operation in
            ARTS.It synchronizes all previous spawned EDTs of the parent region
                .This operation has memory write effects as it affects the
                    synchronization state of the runtime system.}];
  let arguments = (ins
    OptionalAttr<ArtsBarrierReasonAttr>:$barrierReason
  );
  let assemblyFormat = "attr-dict";
}

///===----------------------------------------------------------------------===///
/// arts.atomic_add
///===----------------------------------------------------------------------===///
def AtomicAddOp : Arts_Op<"atomic_add", [MemoryEffects<[MemWrite]>]> {
  let summary = "Atomic add to a scalar memref element";
  let description = [{
    Atomically adds a value to the element at the given address. This is the
    ARTS representation for reductions when converting from OpenMP.
    Only scalar memrefs (rank-0) are supported for now.
  }];

  let arguments = (ins AnyType : $addr, AnyType : $value);
  let assemblyFormat =
      "`(` $addr `,` $value `:` type($addr) `,` type($value) `)` attr-dict";
}

///===----------------------------------------------------------------------===///
/// arts.runtime_query
///===----------------------------------------------------------------------===///
def RuntimeQueryOp : Arts_Op<"runtime_query", [Pure]> {
  let summary = "Query the ARTS runtime for topology or identity information";
  let description = [{
    The `arts.runtime_query` operation queries the ARTS runtime for information
    about the execution environment. The `kind` attribute selects what to query:

    - `total_workers`: Cluster-wide logical worker capacity.
    - `total_nodes`: Total number of nodes available.
    - `current_worker`: ID of the current worker executing this operation.
    - `current_node`: ID of the current node executing this operation.
    All runtime query results are `i32`; materializers cast to `index` when
    using a topology value as loop control.
  }];
  let arguments = (ins RuntimeQueryKindAttr:$kind);
  let results = (outs AnyType : $result);
  let hasCanonicalizer = 1;
  let assemblyFormat = "$kind attr-dict `->` type($result)";
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "RuntimeQueryKind":$kind), [{
      $_state.addTypes(IntegerType::get($_builder.getContext(), 32));
      $_state.addAttribute("kind",
          RuntimeQueryKindAttr::get($_builder.getContext(), kind));
    }]>
  ];
}

def GetEdtEpochGuidOp : Arts_Op<"get_edt_epoch_guid", [Pure]> {
  let summary = "Get the epoch GUID that the current EDT is enrolled in";
  let description = [{
    Returns the epoch_guid field from the current EDT structure. This is
    the epoch that this EDT was enrolled in at creation time, NOT the
    current epoch from the epoch stack.
  }];
  let results = (outs I64 : $epochGuid);
  let assemblyFormat = "attr-dict";
}

#endif // ARTS_OPS
