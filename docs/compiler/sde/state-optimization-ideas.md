# SDE State Optimization Ideas

## Current State Surface

SDE state work is centered on `sde.su_iterate`, memref-level access analysis,
and MU/token/codelet planning before Core. The important state-bearing
attributes are structured classification, approved pattern, owner dims, spatial
dims, write footprint, physical owner dims, physical block shape, logical
worker slice, physical halo shape, iteration topology, repetition structure,
async strategy, and distribution kind.

Current state-related passes and utilities:

- `ConvertOpenMPToSde` creates the initial SDE semantic structure and explicit
  memory/control dependency surfaces.
- `PatternAnalysis` classifies loops, records stencil neighborhoods,
  recognizes wavefront and alternating-buffer families, and stamps approved
  SDE-only memref/ND pattern facts for later SDE consumers.
- `LoopInterchange` and `Tiling` consume approved SDE pattern facts before
  distribution planning.
- `DistributionPlanning` is currently the main pass that stamps physical block
  plans.
- `StructuredOpAnalysis` and `SdeAnalysisUtils` provide output-layout,
  loop-indexed output, in-place self-read, and root memory-effect facts.
- Memref-level `sde.mu_token` is the canonical access handle for planned MU
  slices. Tensor tokens remain a legacy/non-default representation for existing
  tests and compatibility, not the production structured-loop path.

## Optimization Ideas

### Multi-Output Physical Plans

The current uniform path is biased toward one write-backed output and often a
single owner dimension. Extend SDE planning to produce coherent physical plans
for multiple write roots in the same scheduling unit, including fused activation
outputs, `2mm`/`3mm` intermediates, and ML tensors.

The rule should be: each write-backed root gets a proven owner mapping and block
shape, while reader-only plans remain advisory unless SDE proves input tiling is
profitable and legal.

### Matmul And Tensor-Contraction State

Make matmul-like state explicit before Core:

- output tile ownership for `C`, `tmp`, `D`, `E`, `F`, `G`, and batched tensor
  outputs;
- tile sizes based on useful work per task, cache footprint, reduction
  locality, and abstract logical capacity;
- phase state for chained products, with only true producer-consumer edges.
  Current repeated large/64 evidence makes matrix-chain intermediates the
  priority: `gemm` is faster than OpenMP at median, while `2mm` and `3mm` still
  lack stable explicit intermediate reuse across producer and consumer
  contractions.

This prevents Core from seeing one coarse output storage object plus late
per-task slices.

### Reduction-Aware Vector State

Add a `reduction_mixed` state plan that separates local block accumulation from
fan-in strategy. The plan should carry block-owned vector output layout,
reduction strategy, and follower-phase shape for normalize, affine, or verify
stages.

This targets `atax`, `bicg`, `stream`, `batchnorm`, `layernorm`, and `pooling`.

### Component And Slab State

Teach SDE pattern analysis to separate spatial, component, batch, and element
dims for 3D component stencils. SDE should choose a slab owner dimension,
preserve component-local extents inside each task, and stamp halo-aware physical
block shapes that `CreateDbs` can materialize directly.

### Timestep State

Represent repeated time structure in SDE:

- `repetition_structure = pair_step`, `k_step`, or `full_timestep`;
- `iteration_topology = owner_strip`, `owner_tile`, or `wavefront`;
- `async_strategy` values that distinguish advanceable task stages, CPS chains,
  and deferred persistent regions.

Double-buffer Jacobi can group timesteps when buffer parity and halo radius are
proved. In-place Seidel needs wavefront or split-phase state before any physical
block plan is safe.

## Pass Grouping Proposal

Keep the high-level spine:

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
```

The main state-ordering question is whether `IterationSpaceDecomposition`
should run earlier or trigger replanning. A physical plan stamped before
boundary peeling can become stale if the iteration space changes. Either move
decomposition before final `DistributionPlanning`, or split planning into an
early advisory pass and a final physical plan pass after decomposition.

## Risks

- A stale physical plan after loop decomposition can misdescribe the physical
  storage layout.
- Fused elementwise pipelines currently risk inheriting one stage's plan when
  multiple outputs need distinct plans.
- `findLoopIndexedOutputPlan` is intentionally narrow; using it for noncanonical
  traversal can stamp unsound blocked layouts.
- In-place stencils must continue to reject unsafe owner-strip physical plans.

## Tests

- Positive SDE tests for multi-output blocked vector, matmul output tile, and
  component-stencil slab plans.
- Negative SDE tests for unknown effects, self-read in-place writes, escaped
  roots, and noncanonical owner mappings.
- Core tests proving physical storage families are created only from
  write-backed SDE plans.
- Focused benchmark checks for `gemm`, `stream`, ML kernels, and 3D stencils at
  large size and 64 threads.
