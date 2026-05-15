# Memref MU/Token Rewrite Plan

## Objective

Make SDE tiling real by rewriting memory units, compute units, and scheduling
units together at the memref level. The target path has no tensor fallback:
MU allocation allocates memrefs, MU tokens describe memref access windows, and
CODIR codelets consume token-local memref views.

## Problem Statement

Changing only CU/SU loop shape is not a tiling transformation. If the MU storage
and codelet accesses still address the original whole memref incorrectly, the
compiler either loses locality or needs late ARTS compatibility rewrites. That
is the source of much of the current complexity around `CreateDbs` and DB
indexers. The Core raw indexers have now been removed; blocked/tiled accesses
must be handled by this SDE/CODIR rewrite path.

## Current Surface

Relevant current areas:

- `PatternAnalysis`
- `Tiling`
- `DistributionPlanning`
- `MemoryUnitMaterialization`
- `MemrefNormalization`
- `RaiseMemrefToTensor`
- `RaiseToTensor`
- `LowerToMemref`
- `TensorCleanup`
- current `ConvertSdeToArts` token/codelet handling

## Target Contract

For every supported transformed region:

- SDE identifies source memref roots and structured access maps.
- SDE chooses owner dims, block shape, halo/window shape, and logical grain.
- MU storage reflects the chosen physical layout.
- MU tokens reflect each codelet's access window.
- CODIR codelet body accesses are rewritten to token-local coordinates.
- ND, strided, halo, and reduction cases are represented in element space until
  the boundary.

## Phases

### Phase 1: Pattern Facts

- Keep the shared pass named `PatternAnalysis`.
- Stamp approved `sde.pattern` facts, not loose "structured summaries".
- Record memref rank, roots, access modes, owner dims, spatial dims,
  reduction dims, affine/structured access maps, and self-read status.

Exit gate:

- lit tests show stable SDE facts for elementwise, matmul, reduction, stencil,
  and unknown patterns.

### Phase 2: Root And Token Coverage

- Materialize MU roots for function args, globals, allocas, intermediates, task
  depend roots, and reductions.
- Convert `sde.mu_dep` into canonical `sde.mu_token` before the boundary.
- Reject unresolved dependency declarations before CODIR/ARTS.

Exit gate:

- no supported `cu_task deps(...)` path reaches ARTS as raw `sde.mu_dep`.

### Phase 3: Token-Local Access Rewrite

- Rewrite memref load/store indices from source global coordinates to
  token-local coordinates.
- Support ND owner dims and halo offsets.
- Teach memref normalization or the SDE rewrite to preserve per-dimension
  access information for strided views.
- Use existing value utility helpers for equivalence and dominance checks.

Exit gate:

- ND and strided lit tests lower without DB-payload `memref.subview` and
  without raw-memref DB rediscovery.

### Phase 4: Reduction And Intermediate State

- Represent local accumulators, partial outputs, and phase intermediates as
  memref MU/token plans.
- Make `2mm`/`3mm` intermediates explicit producer/consumer phase objects.
- Keep scalar reductions and per-output reductions separate.

Exit gate:

- matrix-chain intermediates have explicit ownership and reuse windows before
  ARTS.

### Phase 5: Tensor Path Removal

- Remove `RaiseMemrefToTensor`, `RaiseToTensor`, `LowerToMemref`, and
  tensor-only cleanup after memref coverage exists.
- Remove tests that only protect the tensor fallback, unless converted into
  memref/CODIR tests.

Exit gate:

- supported samples and benchmarks compile without tensor raising/lowering.

## Verification

- SDE pattern lit tests.
- Memory-unit materialization lit tests for args, globals, allocas,
  intermediates, reductions, and task deps.
- CODIR token-local rewrite lit tests for 1D, ND, strided, and halo cases.
- Focused e2e tests for GEMM, elementwise, reductions, and stencil samples.
