# SDE Optimization Notes

This directory is the compiler-facing home for SDE optimization planning. SDE is
runtime-agnostic: it owns OpenMP semantics, structured memref state, dependence
proofs, effect summaries, task shape, MU token slices, and target-neutral
layout/grain intent before boundary materialization.

The target stack is `sde -> codir -> arts -> arts-rt`. In that stack SDE
authors the MU/CU/SU plan, CODIR materializes isolated codelets with explicit
deps and params, `arts` binds those codelets to DB/EDT/epoch objects, and
`arts-rt` lowers the runtime ABI. The current `sde.cu_codelet` path is a
migration surface for this split, not the final SDE ownership model.

`arts` target binding and `arts-rt` runtime-call work should wait until SDE and
CODIR plans are present and traces still show launch, CPS, dependency, or
scheduling overhead.

For the current dialect split, including SDE logical-worker planning and the
target CODIR boundary, see
[`../dialect-layering-vision.md`](../dialect-layering-vision.md). SDE should
model logical worker capacity, not concrete nodes, routes, workers-per-node, or
target API queries. When symbolic SDE arithmetic needs execution capacity, use
`sde.resource_query <logical_workers>` and let the `arts` boundary bind it to
the abstract ARTS machine.

For the migration sequence, use
[`../master-plan.md`](../master-plan.md) and the focused subplans under
[`../plans/`](../plans/).
The target per-dialect docs live under
[`../dialects/sde/`](../dialects/sde/). SDE-owned analyses are listed in
[`../dialects/sde/analysis.md`](../dialects/sde/analysis.md), and SDE-owned
optimizations are listed in
[`../dialects/sde/optimizations.md`](../dialects/sde/optimizations.md).

## Current SDE Spine

The live SDE pass order is inside `openmp-to-arts`:

```text
ConvertOpenMPToSde
PatternAnalysis
LoopInterchange
Tiling
ElementwiseFusion
Vectorization
ScheduleRefinement
ChunkOpt
ReductionStrategy
DistributionPlanning
IterationSpaceDecomposition
BarrierElimination
VerifySdeCpsPlan
MemoryUnitMaterialization
ConvertSdeToArts
VerifySdeLowered
VerifyCoreObjectsOnly
DeadCodeElimination
CSE
VerifyEdtCreated
```

Use `dekk carts pipeline --json` and `tools/compile/Compile.cpp` as the source
of truth when pass order matters.

The target spine splits the boundary:

```text
ConvertOpenMPToSde
PatternAnalysis
SDE transforms
MemoryUnitMaterialization
ConvertSdeToCodir
VerifyCodir
ConvertCodirToArts
VerifySdeLowered
VerifyArtsObjectsOnly
```

That target keeps codelet isolation out of SDE and keeps DB/EDT creation out of
CODIR.

## Documents

- [`../master-plan.md`](../master-plan.md): top-level migration and performance
  plan.
- [`../dialects/sde/`](../dialects/sde/): target SDE analysis and optimization
  ownership.
- [`../codir/`](../codir/): proposed codelet dialect responsibilities,
  isolation contract, and migration checklist.
- [`physical-layout-optimization-plan.md`](physical-layout-optimization-plan.md):
  SDE-first roadmap for moving physical storage layout and access-slice
  authorship into MU/token plans, with downstream materialization left only as
  the raw-memref bridge.
- [`state-optimization-ideas.md`](state-optimization-ideas.md): memref-level
  state, MU/token, tiling, and task-shape ideas.
- [`dependency-optimization-ideas.md`](dependency-optimization-ideas.md):
  barrier removal, dependency windows, phase edges, timestep, and wavefront
  ideas.
- [`effect-optimization-ideas.md`](effect-optimization-ideas.md): memory effect,
  fusion, reduction, stencil slab, and alias precision ideas.

## Plan Contract

SDE optimization work should stamp or refine SDE-owned plan facts before the
SDE boundary:

- `pattern = #sde.pattern<...>`
- `structuredClassification = #sde.structured_classification<...>`
- `ownerDims`
- `spatialDims`
- `writeFootprint`
- `physicalOwnerDims`
- `physicalBlockShape`
- `logicalWorkerSlice`
- `physicalHaloShape`
- MU facts: `mu_data`/`mu_alloc` roots, memref `mu_token` access mode, token
  offsets, token sizes, and the planned codelet dep/param boundary
- `sde.resource_query <logical_workers>` for symbolic grain arithmetic
- `iterationTopology`
- `repetitionStructure`
- `asyncStrategy`
- `reductionStrategy`
- CPS candidate/final-stage attributes
- execution hints such as `vectorizeWidth`, `unrollFactor`, and
  `interleaveCount`

These facts are the SDE plan. `PatternAnalysis` and later SDE transforms may
consume and update them; raw SDE pattern-analysis facts must not become an
`arts` analysis API. In the target stack, `ConvertSdeToCodir` mechanically
consumes the final SDE plan and produces isolated codelet deps, params,
token-local views, and body rewrites. `ConvertCodirToArts` then turns that
explicit codelet contract into DB storage, DB acquires, EDTs, synchronization,
distribution, dependency-pattern, and resource-query contracts.

For the current migration state, `ConvertSdeToArts` still performs direct
materialization and `CreateDbs` may still bridge legacy raw-memref
`su_iterate` form using SDE physical plan attrs. There is no ARTS
dependency-marker op. Task dependency declarations that remain as `sde.mu_dep`
at the boundary are rejected; SDE must turn those facts into memref-level MU
tokens and CODIR codelets before ARTS. `arts` and `arts-rt` may materialize or
validate that lowered contract, but they must not invent partition policy late.

## Heuristic Analysis Spine

Use `PatternAnalysis` as the name for the shared SDE fact pass. It is not an
`arts` input object and it should not be renamed to vague "structured
summaries".
The pass should stamp stable SDE facts that downstream SDE passes can consume:

- memref family and rank;
- read/write roots, access modes, and self-read status;
- owner, spatial, component, batch, and reduction dimensions;
- affine or structured index maps when available;
- legal interchange, tile, fusion, memref-level vectorization, and CPS
  candidates;
- physical owner slices and MU token slices approved by SDE.

Downstream SDE passes should consume these facts in order: first classify and
prove legality, then choose task/data shape, then rewrite CU/SU/MU together.
Tiling only CU/SU loops without rewriting MU token slices and physical storage
layout is not a valid optimized plan.

## MU Materialization

`MemoryUnitMaterialization` is the current late SDE pass that rewrites eligible
shared memref allocations used by scheduling units into `sde.mu_alloc`. It runs
after CPS/barrier planning and before the dialect boundary, so downstream
materialization can lower MU storage directly to DB storage instead of making
ARTS rediscover those roots from raw memrefs.

For simple owner-slice plans, the current SDE/ARTS boundary consumes the SDE
classification and owner-slice facts directly: it creates block-partitioned
acquires and keeps task bodies on full MU payload coordinates. Direct
MU/token/codelet lowering follows the same rule: `mu_alloc` lowers to DB
storage, `mu_token` lowers to `db_acquire`, and slice offsets are folded into
the cloned memref indices instead of creating `memref.subview` on DB payloads.
More complex ND, halo, and strided cases still require the full SDE
token/CODIR rewrite before their raw-memref bridge paths can be retired.

## Tensor Path Removal

The optimization direction is memref-native. MU allocation should allocate
memrefs, MU tokens should carry memref access windows, and codelet deps should
stay at the memref/token level. Tensor raising/lowering paths are legacy
support and should be removed after the memref MU/token/CODIR path covers task
deps, reductions, codelet-local state, and all maintained benchmark cases.

Candidate removal targets include:

- `RaiseMemrefToTensor`
- `RaiseToTensor`
- `LowerToMemref`
- tensor-only codelet cleanup utilities

Do not keep a tensor path as a fallback once the memref path is complete.

## Verification Rule

Every new optimization needs:

- SDE lit coverage for positive planning and negative fallback.
- CODIR/ARTS coverage proving direct MU/token/codelet lowering for tiled paths;
  `CreateDbs` may cover only coarse raw memrefs during the transition.
- Focused large/64 benchmark evidence for the affected family.
- `dekk carts test`, plus e2e or benchmark sweeps when shared lowering,
  analysis, or runtime shape changes.
