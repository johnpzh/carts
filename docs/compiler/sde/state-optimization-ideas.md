# SDE State Optimization Ideas

## Current State Surface

SDE state work is centered on `arts_sde.su_iterate` and the tensor/carrier
pipeline before Core. The important state-bearing attributes are structured
classification, dependency family, owner dims, spatial dims, write footprint,
physical owner dims, physical block shape, logical worker slice, physical halo
shape, iteration topology, repetition structure, async strategy, and
distribution kind.

Current state-related passes and utilities:

- `RaiseToTensor` and `RaiseToLinalg` create the tensor/linalg form used for
  structured analysis.
- `StructuredSummaries` classifies loops, records stencil neighborhoods,
  recognizes wavefront and alternating-buffer families, and stamps in-place
  safety facts.
- `Tiling` and `LoopInterchange` reshape loops before summaries and
  distribution planning.
- `DistributionPlanning` is currently the main pass that stamps physical block
  plans.
- `LowerToMemref`, `ConvertToCodelet`, and `TensorCleanup` must preserve valid
  physical plan attributes until `ConvertSdeToArts`.
- `StructuredOpAnalysis` and `SdeAnalysisUtils` provide output-layout,
  loop-indexed output, in-place self-read, and root memory-effect facts.

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
- cost-model tile sizes based on useful work per EDT, cache footprint, and
  reduction locality;
- phase state for chained products, with only true producer-consumer edges.

This prevents Core from seeing one coarse output DB plus late per-task slices.

### Reduction-Aware Vector State

Add a `reduction_mixed` state plan that separates local block accumulation from
fan-in strategy. The plan should carry block-owned vector output layout,
reduction strategy, and follower-phase shape for normalize, affine, or verify
stages.

This targets `atax`, `bicg`, `stream`, `batchnorm`, `layernorm`, and `pooling`.

### Component And Slab State

Teach structured summaries to separate spatial, component, batch, and element
dims for 3D component stencils. SDE should choose a slab owner dimension,
preserve component-local extents inside each task, and stamp halo-aware physical
block shapes that `CreateDbs` can materialize directly.

### Timestep State

Represent repeated time structure in SDE:

- `repetition_structure = pair_step`, `k_step`, or `full_timestep`;
- `iteration_topology = owner_strip`, `owner_tile`, or `wavefront`;
- `async_strategy = advance_edt`, `cps_chain`, or deferred
  `persistent_region`.

Double-buffer Jacobi can group timesteps when buffer parity and halo radius are
proved. In-place Seidel needs wavefront or split-phase state before any physical
block plan is safe.

## Pass Grouping Proposal

Keep the high-level spine:

```text
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
```

The main state-ordering question is whether `IterationSpaceDecomposition`
should run earlier or trigger replanning. A physical plan stamped before
boundary peeling can become stale if the iteration space changes. Either move
decomposition before final `DistributionPlanning`, or split planning into an
early advisory pass and a final physical plan pass after decomposition.

## Risks

- A stale physical plan after loop decomposition can misdescribe the DB layout.
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
- Core tests proving `CreateDbs` creates physical DB families only from
  write-backed SDE plans.
- Focused benchmark checks for `gemm`, `stream`, ML kernels, and 3D stencils at
  large size and 64 threads.
