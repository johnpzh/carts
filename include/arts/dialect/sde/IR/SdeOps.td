///==========================================================================///
/// File: SdeOps.td
/// This file defines the SDE dialect operations: CU, SU, MU, and yield.
///==========================================================================///

#ifndef ARTS_SDE_OPS
#define ARTS_SDE_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "arts/dialect/sde/IR/SdeDialect.td"
include "arts/dialect/sde/IR/SdeTypes.td"

def SdeDepType
    : Type<CPred<"::llvm::isa<::mlir::arts::sde::DepType>($_self)">,
           "SDE dependency token">;
def SdeCompletionType
    : Type<CPred<"::llvm::isa<::mlir::arts::sde::CompletionType>($_self)">,
           "SDE completion token">;
def SdeTokenType
    : Type<CPred<"::llvm::isa<::mlir::arts::sde::TokenType>($_self)">,
           "SDE tensor access token">;

//===----------------------------------------------------------------------===//
// Enum helpers
//===----------------------------------------------------------------------===//

class ArtsSde_I32Enum<string name, string description,
                      list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let cppNamespace = "::mlir::arts::sde";
  let genSpecializedAttr = 0;
}

class ArtsSde_I32EnumAttr<string mnemonic, ArtsSde_I32Enum enumInfo>
    : EnumAttr<ArtsSdeDialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

def SdeCuKind
    : ArtsSde_I32Enum<"SdeCuKind", "Compute unit kind", [
      I32EnumAttrCase<"parallel", 0, "parallel">,
      I32EnumAttrCase<"single", 1, "single">,
      I32EnumAttrCase<"task", 2, "task">
    ]>;
def SdeCuKindAttr : ArtsSde_I32EnumAttr<"cu_kind", SdeCuKind>;

def SdeAccessMode
    : ArtsSde_I32Enum<"SdeAccessMode", "Memory access mode", [
      I32EnumAttrCase<"read", 0, "read">,
      I32EnumAttrCase<"write", 1, "write">,
      I32EnumAttrCase<"readwrite", 2, "readwrite">
    ]>;
def SdeAccessModeAttr : ArtsSde_I32EnumAttr<"access_mode", SdeAccessMode>;

def SdeReductionKind
    : ArtsSde_I32Enum<"SdeReductionKind", "Reduction operation kind", [
      I32EnumAttrCase<"add", 0, "add">,
      I32EnumAttrCase<"mul", 1, "mul">,
      I32EnumAttrCase<"min", 2, "min">,
      I32EnumAttrCase<"max", 3, "max">,
      I32EnumAttrCase<"land", 4, "and">,
      I32EnumAttrCase<"lor", 5, "or">,
      I32EnumAttrCase<"lxor", 6, "xor">,
      I32EnumAttrCase<"custom", 7, "custom">
    ]>;
def SdeReductionKindAttr : ArtsSde_I32EnumAttr<"reduction_kind", SdeReductionKind>;

def SdeReductionStrategy
    : ArtsSde_I32Enum<"SdeReductionStrategy",
                      "Reduction execution strategy", [
      I32EnumAttrCase<"atomic", 0, "atomic">,
      I32EnumAttrCase<"tree", 1, "tree">,
      I32EnumAttrCase<"local_accumulate", 2, "local_accumulate">
    ]>;
def SdeReductionStrategyAttr
    : ArtsSde_I32EnumAttr<"reduction_strategy", SdeReductionStrategy>;

def SdeStructuredClassification
    : ArtsSde_I32Enum<"SdeStructuredClassification",
                      "Structured loop classification", [
      I32EnumAttrCase<"elementwise", 0, "elementwise">,
      I32EnumAttrCase<"stencil", 1, "stencil">,
      I32EnumAttrCase<"matmul", 2, "matmul">,
      I32EnumAttrCase<"reduction", 3, "reduction">,
      I32EnumAttrCase<"elementwise_pipeline", 4, "elementwise_pipeline">
    ]>;
def SdeStructuredClassificationAttr
    : ArtsSde_I32EnumAttr<"structured_classification",
                          SdeStructuredClassification>;

def SdeScheduleKind
    : ArtsSde_I32Enum<"SdeScheduleKind", "Scheduling policy", [
      I32EnumAttrCase<"static_", 0, "static">,
      I32EnumAttrCase<"dynamic", 1, "dynamic">,
      I32EnumAttrCase<"guided", 2, "guided">,
      I32EnumAttrCase<"auto_", 3, "auto">,
      I32EnumAttrCase<"runtime", 4, "runtime">
    ]>;
def SdeScheduleKindAttr : ArtsSde_I32EnumAttr<"schedule", SdeScheduleKind>;

def SdeConcurrencyScope
    : ArtsSde_I32Enum<"SdeConcurrencyScope", "Concurrency scope", [
      I32EnumAttrCase<"local", 0, "local">,
      I32EnumAttrCase<"distributed", 1, "distributed">
    ]>;
def SdeConcurrencyScopeAttr
    : ArtsSde_I32EnumAttr<"concurrency_scope", SdeConcurrencyScope>;

def SdeDistributionKind
    : ArtsSde_I32Enum<"SdeDistributionKind", "Distribution advisory hint", [
      I32EnumAttrCase<"owner_compute", 0, "owner_compute">,
      I32EnumAttrCase<"blocked", 1, "blocked">,
      I32EnumAttrCase<"cyclic", 2, "cyclic">
    ]>;
def SdeDistributionKindAttr
    : ArtsSde_I32EnumAttr<"distribution_kind", SdeDistributionKind>;

//===----------------------------------------------------------------------===//
// CU (Compute Unit) Ops
//===----------------------------------------------------------------------===//

def SdeCuRegionOp
    : ArtsSde_Op<"cu_region", [
      SingleBlock,
      RecursiveMemoryEffects
    ]> {
  let summary = "Parallel or single execution region";
  let description = [{
    Captures `omp.parallel`, `omp.single`, `omp.master`. Represents a
    region of computation that runs with a given concurrency kind and scope.

    The `nowait` attribute preserves OpenMP nowait semantics. The
    `concurrency_scope` attribute indicates local vs distributed execution.

    Optional `iter_args` thread tensor SSA values through the region body.
    When present, the body block receives matching block arguments and the
    region must terminate with `sde.yield` returning updated values. When
    absent, the body may omit a terminator or yield empty.
  }];

  let arguments = (ins
    SdeCuKindAttr:$kind,
    OptionalAttr<SdeConcurrencyScopeAttr>:$concurrency_scope,
    UnitAttr:$nowait,
    Variadic<AnyType>:$iterArgs
  );

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def SdeCuTaskOp
    : ArtsSde_Op<"cu_task", [
      SingleBlock, NoTerminator,
      RecursiveMemoryEffects
    ]> {
  let summary = "Task with explicit data dependencies";
  let description = [{
    Captures `omp.task` with `depend(in/out/inout)` clauses. Dependencies
    are specified via `sde.mu_dep` results or inferred from SSA when using
    tensor operands.
  }];

  let arguments = (ins
    Variadic<SdeDepType>:$deps
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    (`deps` `(` $deps^ `:` type($deps) `)`)? $body attr-dict
  }];
}

def SdeCuReduceOp
    : ArtsSde_Op<"cu_reduce", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Reduction accumulation";
  let description = [{
    Captures reduction semantics from `omp.wsloop reduction(+: sum)`.
    PRESERVES the combiner semantics that the current pipeline loses.

    The reduction kind (add, mul, min, max, etc.) and identity value are
    explicit. For custom reductions, the combiner region carries the
    reduction body.
  }];

  let arguments = (ins
    SdeReductionKindAttr:$reduction_kind,
    AnyType:$accumulator,
    AnyType:$partial,
    AnyType:$identity
  );

  let results = (outs AnyType:$result);

  let regions = (region AnyRegion:$combiner);

  let assemblyFormat = [{
    $reduction_kind `(` $accumulator `,` $partial `:` type($accumulator) `,` type($partial) `)`
    `identity` `(` $identity `:` type($identity) `)`
    ($combiner^)? attr-dict `->` type($result)
  }];
}

def SdeCuAtomicOp
    : ArtsSde_Op<"cu_atomic", [MemoryEffects<[MemWrite]>]> {
  let summary = "Atomic memory update";
  let description = [{
    Captures `omp.atomic.update`. Currently only supports add{i,f}.
    Memref-only since atomics are inherently side-effecting operations
    on mutable memory locations.
  }];

  let arguments = (ins
    SdeReductionKindAttr:$reduction_kind,
    AnyMemRef:$addr,
    AnyType:$value
  );

  let assemblyFormat = [{
    $reduction_kind `(` $addr `,` $value `:` type($addr) `,` type($value) `)`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// SU (Scheduling Unit) Ops
//===----------------------------------------------------------------------===//

def SdeSuIterateOp
    : ArtsSde_Op<"su_iterate", [
      SingleBlock,
      RecursiveMemoryEffects,
      AttrSizedOperandSegments,
      DeclareOpInterfaceMethods<LoopLikeOpInterface,
        ["getLoopRegions", "getLoopInductionVars", "getLoopLowerBounds",
         "getLoopUpperBounds", "getLoopSteps"]>
    ]> {
  let summary = "Parallel iteration space";
  let description = [{
    Captures `omp.wsloop + omp.loop_nest`, `omp.taskloop`, `scf.parallel`.
    Represents a parallel iteration space with optional schedule, chunk size,
    nowait, and reduction metadata.

    Reduction accumulators can carry the combiner kind and identity value,
    preserving information lost by the current pipeline.

    Optional `iter_args` thread tensor SSA values through the loop body.
    When present, the body block receives matching block arguments (after
    induction variables) and the region must terminate with `sde.yield`
    returning updated values.  Results match iter_args 1:1.

    When iter_args are absent the op produces no results and the body
    terminates with an empty `sde.yield`.
  }];

  let arguments = (ins
    Variadic<Index>:$lowerBounds,
    Variadic<Index>:$upperBounds,
    Variadic<Index>:$steps,
    OptionalAttr<SdeScheduleKindAttr>:$schedule,
    Optional<Index>:$chunkSize,
    UnitAttr:$nowait,
    Variadic<AnyType>:$reductionAccumulators,
    OptionalAttr<ArrayAttr>:$reductionKinds,
    OptionalAttr<SdeReductionStrategyAttr>:$reductionStrategy,
    OptionalAttr<SdeStructuredClassificationAttr>:$structuredClassification,
    OptionalAttr<ArrayAttr>:$accessMinOffsets,
    OptionalAttr<ArrayAttr>:$accessMaxOffsets,
    OptionalAttr<ArrayAttr>:$ownerDims,
    OptionalAttr<ArrayAttr>:$spatialDims,
    OptionalAttr<ArrayAttr>:$writeFootprint
  );

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def SdeSuDistributeOp
    : ArtsSde_Op<"su_distribute", [
      SingleBlock, NoTerminator,
      RecursiveMemoryEffects
    ]> {
  let summary = "Distribution advisory wrapper for SDE work";
  let description = [{
    Wraps a scheduling-unit region with a runtime-neutral distribution hint.
    This is an SDE-level advisory carrier only; it does not encode ARTS worker,
    route, or node identifiers.
  }];

  let arguments = (ins SdeDistributionKindAttr:$kind);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $kind $body attr-dict
  }];
}

def SdeSuBarrierOp
    : ArtsSde_Op<"su_barrier", [MemoryEffects<[MemWrite]>]> {
  let summary = "Synchronization point";
  let description = [{
    Captures `omp.barrier` (implicit) and `omp.taskwait` (explicit tokens).
    When tokens are provided, waits only for those specific tasks.
  }];

  let arguments = (ins Variadic<SdeCompletionType>:$tokens);

  let assemblyFormat = [{
    (`(` $tokens^ `:` type($tokens) `)`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// MU (Memory Unit) Ops
//===----------------------------------------------------------------------===//

def SdeMuDepOp
    : ArtsSde_Op<"mu_dep", [Pure, AttrSizedOperandSegments]> {
  let summary = "Data dependency declaration (memref fallback)";
  let description = [{
    Declares a data dependency with access mode and region. Used in the
    memref fallback path. In the tensor path, `tensor.extract_slice`/
    `tensor.insert_slice` carry the same information natively.
  }];

  let arguments = (ins
    SdeAccessModeAttr:$mode,
    AnyMemRef:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes
  );

  let results = (outs SdeDepType:$dep);

  let assemblyFormat = [{
    $mode $source (`[` $offsets^ `]`)? (`size` `[` $sizes^ `]`)?
    `:` type($source) `->` type($dep) attr-dict
  }];
}

def SdeMuTokenOp
    : ArtsSde_Op<"mu_token", [Pure, AttrSizedOperandSegments]> {
  let summary = "Produce a tensor-path access token (whole or slice).";
  let description = [{
    Names a tensor (or a rectangular slice of one) for consumption by a
    `sde.cu_codelet`, with an explicit access mode. When `offsets` / `sizes`
    are omitted the token addresses the whole tensor.

    Tokens are the tensor-path analogue of `sde.mu_dep`. At the SDE to ARTS
    boundary a `mu_token` lowers 1:1 to `arts.db_acquire <mode>`.

    Two tokens with disjoint slices of the same tensor expose slice-
    parallelism that would be serialized if codelets took raw tensor
    operands; the SSA dep edge still flows through the parent tensor, but
    the analyses see the disjointness.
  }];

  let arguments = (ins
    SdeAccessModeAttr:$mode,
    AnyTensor:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes
  );

  let results = (outs SdeTokenType:$token);

  let assemblyFormat = [{
    $mode $source (`[` $offsets^ `]`)? (`size` `[` $sizes^ `]`)?
    `:` type($source) `->` type($token) attr-dict
  }];

  let hasVerifier = 1;
}

def SdeCuCodeletOp
    : ArtsSde_Op<"cu_codelet", [
      IsolatedFromAbove,
      SingleBlock,
      RecursiveMemoryEffects,
      AttrSizedOperandSegments
    ]> {
  let summary = "Dataflow codelet with tensor tokens and scalar captures.";
  let description = [{
    A compute unit whose only communication with the enclosing scope is
    through `sde.mu_token` operands, scalar capture operands, and yielded
    tensor results. The name follows OCR's "codelet": a purely asynchronous,
    pure-dataflow unit of work whose inputs and outputs are entirely explicit.

    Operand shape: token operands are `!sde.token<tensor<...>>`; scalar
    captures are ordinary SSA values that are copied into the downstream EDT
    parameter pack. The body's block arguments first list one tensor view per
    token, typed as the token's `slice_type`, followed by one block argument per
    scalar capture.

    Result shape: one SSA result per `write` or `readwrite` token, of the
    same type as the source parent tensor. The op implicitly reconstructs
    the parent tensor with the slice replaced by the yielded value
    (destination-passing style, as in linalg-on-tensors).

    `IsolatedFromAbove` guarantees the body references no SSA value from
    outside its block arguments. Dependencies are SSA-visible via the token
    operand list; scalar firstprivate-style values are SSA-visible via the
    capture operand list.
  }];

  let arguments = (ins
    Variadic<SdeTokenType>:$tokens,
    Variadic<AnyType>:$captures
  );
  let results   = (outs Variadic<AnyTensor>:$outputs);
  let regions   = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` ($tokens^ `:` type($tokens))? `)`
    (`captures` `(` $captures^ `:` type($captures) `)`)?
    (`->` `(` type($outputs)^ `)`)?
    $body attr-dict
  }];

  let hasVerifier = 1;
}

def SdeMuAllocOp : ArtsSde_Op<"mu_alloc", [Pure]> {
  let summary = "Allocate a tensor value with given shape.";
  let description = [{
    Creates a tensor SSA value of the given type with undefined contents.
    This is the SDE-level equivalent of `memref.alloca` — it declares a
    data container for the SDE dataflow graph. Supports both static and
    dynamic dimensions.

    Lowered by ConvertSdeToArts to `tensor.empty()` for the bufferization
    path, or directly to `arts.db_alloc` for distributed execution.

    Examples:
      %arr = sde.mu_alloc : tensor<100xi32>
      %mat = sde.mu_alloc(%n, %m) : tensor<?x?xf64>
  }];
  let arguments = (ins Variadic<Index>:$dynamicSizes);
  let results = (outs AnyRankedTensor:$tensor);
  let assemblyFormat = [{
    (`(` $dynamicSizes^ `)`)? attr-dict `:` type($tensor)
  }];
  let hasVerifier = 1;
}

def SdeMuMemrefToTensorOp : ArtsSde_Op<"mu_memref_to_tensor", [Pure]> {
  let summary = "Snapshot a memref into a tensor value at an SDE boundary.";
  let description = [{
    Creates a tensor SSA value from a memref at an SDE region boundary (e.g.,
    before entering a cu_region). The tensor value enters pure SSA dataflow;
    the memref is not modified. Lowered by ConvertSdeToArts to
    `bufferization.to_tensor`.
  }];
  let arguments = (ins AnyMemRef:$memref);
  let results = (outs AnyTensor:$tensor);
  let assemblyFormat = "$memref `:` type($memref) `->` type($tensor) attr-dict";
}

def SdeMuTensorToMemrefOp
    : ArtsSde_Op<"mu_tensor_to_memref", [MemoryEffects<[MemWrite]>]> {
  let summary = "Materialize a tensor value back into a memref at an SDE boundary.";
  let description = [{
    Writes a tensor SSA value back to its backing memref at an SDE region
    boundary (e.g., after a cu_region yields tensor results). This is the
    inverse of `sde.mu_memref_to_tensor`. Lowered by ConvertSdeToArts to
    `bufferization.materialize_in_destination`.
  }];
  let arguments = (ins AnyTensor:$source, Arg<AnyMemRef, "", [MemWrite]>:$dest);
  let assemblyFormat = "$source `in` `writable` $dest `:` type($source) `,` type($dest) attr-dict";
}

def SdeMuDataOp : ArtsSde_Op<"mu_data", [Pure]> {
  let summary = "Declarative SDE shared-data handle.";
  let description = [{
    Canonical anchor for data that crosses SDE tasks. Replaces the pattern
    of inheriting Polygeist `memref.alloca` for OMP-shared variables.
    Produced by `ConvertOpenMPToSde` for every variable named by an OMP
    `shared(...)` clause (or otherwise shared across tasks), consumed by
    `RaiseMemrefToTensor` as an enumeration anchor, lowered by
    `ConvertSdeToArts` to `arts.db_alloc`.
  }];

  let arguments = (ins
    OptionalAttr<AnyAttr>:$init,
    UnitAttr:$shared
  );

  let results = (outs AnyType:$handle);

  let assemblyFormat = [{
    (`shared` $shared^)? (`init` `(` $init^ `)`)?
    attr-dict `:` type($handle)
  }];
}

def SdeMuReductionDeclOp
    : ArtsSde_Op<"mu_reduction_decl", [IsolatedFromAbove, Symbol]> {
  let summary = "Module-level SDE reduction declaration";
  let description = [{
    Declares a named SDE reduction symbol with its element type, reduction
    classification, optional identity attribute, and an optional combiner
    region for custom reductions.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    SdeReductionKindAttr:$reduction_kind,
    OptionalAttr<AnyAttr>:$identity
  );

  let regions = (region AnyRegion:$combiner);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $sym_name `:` $type `kind` `(` $reduction_kind `)`
    (`identity` `(` $identity^ `)` )?
    ($combiner^)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Yield
//===----------------------------------------------------------------------===//

def SdeYieldOp
    : ArtsSde_Op<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "SDE region terminator";
  let description = [{
    Terminates all SDE regions. Can optionally return values for reductions
    and iter_args.
  }];

  let arguments = (ins Variadic<AnyType>:$values);

  let assemblyFormat = [{ ($values^ `:` type($values))? attr-dict }];
}

#endif // ARTS_SDE_OPS
