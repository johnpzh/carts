# SDE Optimization Notes

This directory is the compiler-facing home for SDE optimization planning. SDE is
runtime-agnostic: it owns OpenMP semantics, structured state, dependence proofs,
effect summaries, task shape, and target-neutral layout/grain intent before
`ConvertSdeToArts`. Core is ARTS-machine-aware and materializes those plans in
`CreateDbs`, preserving DB/EDT/epoch orchestration and binding SDE logical
resource queries to ARTS runtime queries. RT/runtime work should wait until
SDE/Core plans are present and traces still show launch, CPS, dependency, or
runtime scheduling overhead.

For the current dialect split, including SDE logical-worker planning after the
removal of the Core loop carrier, see
[`../dialect-layering-vision.md`](../dialect-layering-vision.md). SDE should
model logical worker capacity, not ARTS nodes, routes, workers-per-node, or
runtime API queries. When symbolic SDE arithmetic needs execution capacity, use
`sde.resource_query <logical_workers>` and let `ConvertSdeToArts` lower it to
Core.

## Current SDE Spine

The live SDE pass order is inside `openmp-to-arts`:

```text
ConvertOpenMPToSde
RaiseToTensor
RaiseToLinalg
PatternAnalysis
LoopInterchange
Tiling
ElementwiseFusion
ScopeSelection
ScheduleRefinement
ChunkOpt
ReductionStrategy
DistributionPlanning
IterationSpaceDecomposition
BarrierElimination
Vectorization
LowerToMemref
ConvertToCodelet
TensorCleanup
TokenModeRefine
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
  SDE-first roadmap for making `CreateDbs` materialize planned physical DB
  layouts directly.
- [`state-optimization-ideas.md`](state-optimization-ideas.md): state, tensor,
  carrier, tiling, and task-shape ideas.
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
analysis API. At the dialect boundary, `ConvertSdeToArts` mechanically lowers
the final SDE plan into Core `arts.plan.*`, dependency-pattern, distribution,
DB, EDT, barrier contracts, and `arts.runtime_query` operations for SDE logical
resource queries. Core and RT may materialize or validate that lowered
contract, but they must not invent tensor partition policy late.

## Verification Rule

Every new optimization needs:

- SDE lit coverage for positive planning and negative fallback.
- Core coverage proving `CreateDbs` consumes SDE-authored physical layouts
  without creating late policy.
- Focused large/64 benchmark evidence for the affected family.
- `dekk carts test`, plus e2e or benchmark sweeps when shared lowering,
  analysis, or runtime shape changes.
