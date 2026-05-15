# Memref MU/Token Rewrite Plan

Depends on [`codir-edt-isolation.md`](./codir-edt-isolation.md) for the
`codir.codelet` boundary.

## Objective

Make SDE tiling real by rewriting memory units, compute units, and scheduling
units together at the memref level. The target path has no tensor-carrier path:
MU allocation allocates memrefs, MU tokens describe memref access windows, and
CODIR codelets consume token-local memref views.

## Problem Statement

Changing only CU/SU loop shape is not a tiling transformation. If the MU storage
and codelet accesses still address the original whole memref incorrectly, the
compiler either loses locality or needs late ARTS layout recovery. That is the
source of much of the current complexity around `CreateDbs` and DB indexers.
The Core raw indexers have now been removed; blocked/tiled accesses must be
handled by this SDE/CODIR rewrite path.

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
- current `convert-sde-to-codir` and `convert-codir-to-arts` token/codelet
  handling
- current `codir-codelet-opt` cleanup after token-local rewrites

## Target Contract

For every supported transformed region:

- SDE identifies source memref roots and structured access maps.
- SDE chooses owner dims, block shape, halo/window shape, and logical grain.
- MU storage reflects the chosen physical layout.
- MU tokens reflect each codelet's access window.
- `codir.codelet` body accesses through dependency block arguments are
  rewritten to token-local coordinates.
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
  - Current slice: `convert-sde-to-codir` handles rank-1 and ND direct
    `memref.load`/`memref.store` uses of sliced `sde.mu_token` codelet deps
    for unit-step token windows. Constant offsets are rematerialized inside the
    isolated body; dynamic per-dimension offsets are added as explicit CODIR
    scalar params.
  - Current task-depend slice: direct `sde.cu_task` bodies with a sliced
    `sde.mu_dep` for a source memref, complete static rank-1/ND
    offsets/sizes, direct `memref.load`/`memref.store` uses of that source, and
    simple body-local `memref.subview`/`polygeist.subindex` views rooted in that
    source now lower to a CODIR subview dep and rewrite those indices to
    token-local coordinates. Identical duplicate-source static slices reuse
    one CODIR view dep with merged access mode. Mixed-window duplicate-source
    task deps now become distinct positional CODIR deps for exact-view proof
    cases where all source-rooted body uses are unit-stride `memref.subview`
    ops whose offsets/sizes match one of the `sde.mu_dep` bounds, row
    `polygeist.subindex` ops matching a unit first-dimension dep window with
    trailing full-row bounds, or direct root accesses whose indices exactly
    match one dep offset vector. Matched view results must be used only by
    direct loads/stores. Dynamic task slice bounds are
    supported only by the same exact-view proof, or by direct root load/store
    indices that exactly match the dynamic dep offsets and therefore rewrite to
    local zero. General dynamic task slice bounds and mixed-window direct root
    accesses with non-exact indices remain guarded because source task-depend
    slices may be synchronization tokens rather than the task body's complete
    access window.
- Support ND owner dims and halo offsets.
- Teach memref normalization or the SDE rewrite to preserve per-dimension
  access information for strided views.
- Use existing value utility helpers for equivalence and dominance checks.
- Keep dead rematerialized index/view cleanup in CODIR (`codir-codelet-opt`)
  before ARTS materialization.

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
- Remove tests that only protect the tensor-carrier path, unless converted into
  memref/CODIR tests.

Exit gate:

- supported samples and benchmarks compile without tensor raising/lowering.

## Verification

- SDE pattern lit tests.
- Memory-unit materialization lit tests for args, globals, allocas,
  intermediates, reductions, and task deps.
- CODIR token-local rewrite lit tests for 1D, ND, strided, and halo cases.
  - Current coverage: rank-1 and ND direct sliced-token local rewrite,
    including dynamic per-dimension offsets, at the CODIR boundary and rank-1
    after CODIR-to-ARTS materialization.
  - Current task coverage: sliced `sde.mu_dep` sources on `sde.cu_task` with
    static rank-1/ND slice bounds, direct source load/store rewriting, simple
    body-local subview/subindex rewriting, and identical duplicate-source slice
    reuse at the CODIR boundary and through CODIR-to-ARTS materialization, plus
    mixed-window duplicate-source positional deps for exact subview, exact row
    `polygeist.subindex`, and exact-offset direct-root accesses, dynamic
    exact-view `memref.subview` proof slices, and exact-offset dynamic
    direct-root access rewriting at the CODIR boundary and through
    CODIR-to-ARTS materialization.
- Focused e2e tests for GEMM, elementwise, reductions, and stencil samples.
