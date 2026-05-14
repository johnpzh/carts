# SDE Optimization Notes

This directory is the compiler-facing home for SDE optimization planning. SDE is
runtime-agnostic: it owns OpenMP semantics, structured state, dependence proofs,
effect summaries, task shape, MU token slices, and target-neutral layout/grain
intent before boundary materialization. The canonical SDE boundary consumes
`sde.mu_data`, `sde.mu_token`, and `sde.cu_codelet` as storage, access-window,
and compute-unit requests. Core remains responsible for target binding and for
the remaining raw-memref compatibility bridge. RT/runtime work should wait
until SDE/Core plans are present and traces still show launch, CPS, dependency,
or scheduling overhead.

For the current dialect split, including SDE logical-worker planning after the
removal of the Core loop carrier, see
[`../dialect-layering-vision.md`](../dialect-layering-vision.md). SDE should
model logical worker capacity, not concrete nodes, routes, workers-per-node, or
target API queries. When symbolic SDE arithmetic needs execution capacity, use
`sde.resource_query <logical_workers>` and let the boundary layer bind it to
Core.

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
ConvertSdeToArts
VerifySdeLowered
DeadCodeElimination
CSE
VerifyEdtCreated
```

Use `dekk carts pipeline --json` and `tools/compile/Compile.cpp` as the source
of truth when pass order matters.

## Documents

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

SDE optimization work should stamp or refine SDE-owned plan facts before
`ConvertSdeToArts`:

- `pattern = #sde.pattern<...>`
- `structuredClassification = #sde.structured_classification<...>`
- `ownerDims`
- `spatialDims`
- `writeFootprint`
- `physicalOwnerDims`
- `physicalBlockShape`
- `logicalWorkerSlice`
- `physicalHaloShape`
- MU facts: `mu_data` roots, memref `mu_token` access mode, token offsets,
  token sizes, and codelet token/capture boundaries
- `sde.resource_query <logical_workers>` for symbolic grain arithmetic
- `iterationTopology`
- `repetitionStructure`
- `asyncStrategy`
- `reductionStrategy`
- CPS candidate/final-stage attributes
- execution hints such as `vectorizeWidth`, `unrollFactor`, and
  `interleaveCount`

These facts are the SDE plan. `PatternAnalysis` and later SDE transforms may
consume and update them; raw SDE pattern-analysis facts must not become a Core
analysis API. At the dialect boundary, `ConvertSdeToArts` mechanically consumes
the final SDE plan and produces Core storage, access, compute, synchronization,
distribution, dependency-pattern, and resource-query contracts. For canonical
MU/CU form this means direct storage/access/codelet materialization. For legacy
raw-memref `su_iterate` form, Core may still materialize raw external memref
captures using SDE physical plan attrs, but there is no Core dependency-marker
op. Task dependency declarations that remain as `sde.mu_dep` at the boundary
are rejected; SDE must turn those facts into memref-level MU tokens/codelets
before Core. Core and RT may materialize or validate that lowered contract, but
they must not invent partition policy late.

## Heuristic Analysis Spine

Use `PatternAnalysis` as the name for the shared SDE fact pass. It is not a
Core input object and it should not be renamed to vague "structured summaries".
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

## Verification Rule

Every new optimization needs:

- SDE lit coverage for positive planning and negative fallback.
- Core coverage proving direct MU/token lowering or `CreateDbs` consumption of
  SDE-authored physical layouts without creating late policy.
- Focused large/64 benchmark evidence for the affected family.
- `dekk carts test`, plus e2e or benchmark sweeps when shared lowering,
  analysis, or runtime shape changes.
