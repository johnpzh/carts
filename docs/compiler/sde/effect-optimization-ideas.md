# SDE Effect Optimization Ideas

## Current Effect Surface

SDE effect optimization is where scheduling decisions become concrete SDE plan
facts. It connects SDE pattern analysis, memory effects, fusion, reduction
strategy, tiling, distribution, and barrier decisions before Core.

Current passes:

- `ScopeSelection` chooses or refines concurrency scope.
- `ScheduleRefinement` selects static, guided, or dynamic scheduling for narrow
  loop shapes.
- `ChunkOpt` chooses chunk size for dynamic and guided schedules.
- `ReductionStrategy` annotates tree, atomic, or local-accumulate strategy.
- `DistributionPlanning` stamps physical owner dims, physical block shape,
  halo shape, logical worker slice, iteration topology, and distribution kind.
- `BarrierElimination` removes barriers when structured effects prove no
  cross-unit conflict.

Current analysis facts:

- `StructuredMemoryEffectSummary` collects root reads, root writes, linalg DPS
  roots, and unknown effects.
- `StructuredOutputLayoutPlan` maps loop dims to physical output dims.
- `LoopIndexedOutputPlan` proves simple owner-IV-indexed external writes.
- In-place self-read detection prevents unsafe physical stencil plans.

## Optimization Ideas

### Effect-Aware Tiling

Tile sizes should come from structured effect facts and the cost model:

- enough useful work per EDT to amortize launch;
- cache footprint and read-reuse for tensor contractions;
- halo widening pressure for stencils;
- local reduction footprint for vector and norm kernels;
- DB family count and dependency-window count.

The selected tile should be the same shape SDE stamps as `physicalBlockShape`,
avoiding a mismatch between loop tiling and DB materialization.

### Block Pipeline Fusion

Extend elementwise fusion from sibling pointwise loops into block pipelines.
Legal fusion requires:

- same owner IV or same planned worker slice;
- no root/window conflict between stages;
- disjoint writes or proven producer-consumer flow inside the fused block;
- recomputed physical plan for all outputs, not inherited from the first stage.

This targets `stream` and `ml-kernels/activations`, where many simple maps over
the same input should become one larger owner task writing multiple blocked
outputs.

### Reduction-Mixed Planning

Make `ReductionStrategy` feed `DistributionPlanning` instead of staying
annotation-only. A reduction-mixed plan should include:

- local block accumulation shape;
- fan-in strategy: local accumulate, tree, or atomic;
- physical output/vector DB layout when the final result is array-backed;
- follower phase shape for normalize, affine, or verification work.

This targets `atax`, `bicg`, `batchnorm`, `layernorm`, and `pooling`.

### Stencil Slab Effects

For 3D component stencils, effect summaries should distinguish:

- spatial dims that participate in halos;
- component dims that should remain local inside the task;
- batch or element dims that can own slabs;
- read-only coefficient tensors that should stay coarse or be tiled by reuse
  proof.

The physical plan should stamp slab DBs and bounded halo windows so Core does
not acquire full component tensors for every worker.

### Alias And Subview Precision

Root-level memory effects are safe but coarse. Add optional alias-aware effect
summaries for memref views, tensor carriers, and affine subregions:

- same root, disjoint subviews should not force a barrier;
- overlapping subviews should preserve ordering;
- linalg DPS roots should carry per-output effect facts;
- unknown effects should keep the current conservative fallback.

This precision is needed before aggressive barrier sinking, block pipeline
fusion, and same-root dependency-window narrowing.

### Physical DB Plan Stamping

Broaden physical plan stamping beyond current stencil, matmul, uniform, and
minimal reduction paths:

- ML tensor outputs with NCHW/channel/block plans;
- stream-style vector outputs and fused phases;
- triangular or symmetric outputs such as correlation;
- batched tensor contractions;
- window reductions for pooling.

All physical DB creation remains write-backed first. Reader-only input tiling
must be separately proved and cost-justified.

## Pass Grouping Proposal

Keep effect decisions after SDE pattern analysis and structural tensor
transforms, and before lowering:

```text
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
ConvertSdeToArts
```

Two refinements are worth investigating:

- split fusion into a pre-plan phase and a post-plan compatibility phase; or
  make fusion recompute multi-output physical plans;
- run final distribution planning after any pass that changes the iteration
  space or scheduling-unit boundaries.

## Risks

- Effect summaries that miss aliasing can make fusion or barrier removal
  unsound.
- Over-partitioning reader-only inputs can increase DB count and dependency
  traffic without improving locality.
- Physical plan stamping must not turn unknown or in-place self-read kernels
  into blocked DB families.
- Vectorization and lowering must preserve plan attributes until
  `ConvertSdeToArts`.

## Tests

- Positive block pipeline fusion tests with multiple outputs and matching
  worker slices.
- Negative fusion tests for overlapping writes, unknown effects, and mismatched
  worker slices.
- Reduction-mixed tests for local accumulation, tree fan-in, and atomic fallback.
- 3D component stencil slab tests with halo-local reads and coarse fallback for
  unsafe component layouts.
- Alias/subview tests proving same-root disjoint windows can skip barriers while
  overlapping windows preserve them.
