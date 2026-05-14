# SDE Optimization Notes

This directory is the compiler-facing home for SDE optimization planning. SDE is
runtime-agnostic: it owns OpenMP semantics, structured state, dependence proofs,
effect summaries, task shape, and physical DB layout policy before
`ConvertSdeToArts`. Core is ARTS-machine-aware and materializes those plans in
`CreateDbs`, preserving DB/EDT/epoch orchestration. RT/runtime work should wait
until SDE/Core plans are present and traces still show launch, CPS, dependency,
or runtime scheduling overhead.

For the current dialect split, including SDE logical-worker planning after the
removal of the Core loop carrier, see
[`../dialect-layering-vision.md`](../dialect-layering-vision.md). SDE should
model logical worker capacity, not ARTS nodes, routes, workers-per-node, or
runtime API queries.

## Current SDE Spine

The live SDE pass order is inside `openmp-to-arts`:

```text
ConvertOpenMPToSde
RaiseToTensor
RaiseToLinalg
LoopInterchange
Tiling
StructuredSummaries
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

SDE optimization work should stamp or refine the existing `arts.plan.*`
contract before `ConvertSdeToArts`:

- `arts.plan.kernel_family`
- `arts.plan.owner_dims`
- `arts.plan.physical_block_shape`
- `arts.plan.logical_worker_slice`
- `arts.plan.halo_shape`
- `arts.plan.iteration_topology`
- `arts.plan.repetition_structure`
- `arts.plan.async_strategy`
- `arts.plan.cost.*`

The plan is valid only when the SDE pass that stamps it also owns the legality
proof. `CreateDbs` may consume the plan; Core and RT must not invent tensor
partition policy late.

## Verification Rule

Every new optimization needs:

- SDE lit coverage for positive planning and negative fallback.
- Core coverage proving `CreateDbs` consumes SDE-authored physical layouts
  without creating late policy.
- Focused large/64 benchmark evidence for the affected family.
- `dekk carts test`, plus e2e or benchmark sweeps when shared lowering,
  analysis, or runtime shape changes.
