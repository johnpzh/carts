# SDE Dependency Optimization Ideas

## Current Dependency Surface

SDE dependency optimization covers structural loop transforms, elementwise
fusion legality, structured memory effects, barriers, timestep stamps, and
dependency-window intent before `convert-sde-to-codir`.

Current pieces:

- `PatternAnalysis` stamps `pattern`, stencil offsets, owner dims,
  wavefront families, Jacobi alternating-buffer facts, and in-place safety.
- `ElementwiseFusion` fuses consecutive sibling elementwise loops only when
  iteration spaces and schedules match and write roots are disjoint.
- `SdeAnalysisUtils::collectStructuredMemoryEffects` summarizes root reads,
  root writes, linalg DPS roots, and unknown effects.
- `BarrierElimination` compares the nearest predecessor and successor
  scheduling units around a barrier. It eliminates the barrier when root-level
  read/write conflicts are absent, otherwise it preserves or classifies the
  barrier reason. For repeated timestep boundaries it also stamps CPS
  candidate groups and inserts the explicit completion-token edge that proves
  the stage boundary to Core.
- The boundary layer omits downstream synchronization only when SDE marked the
  barrier as eliminated.

The current model is safe, but mostly root-level and adjacent-pair based. That
is too coarse for benchmarks with phase chains, shared arrays with disjoint
windows, repeated timesteps, and wavefronts.

## Optimization Ideas

### Phase Graph Barrier Planning

Replace pairwise barrier reasoning with a compact phase graph over
`su_iterate` and `su_distribute` units:

- nodes are scheduling units;
- edges are RAW, WAR, or WAW conflicts;
- edge labels carry root or window footprints;
- barriers with no crossing conflict are removed;
- duplicate barriers guarding the same cut are coalesced;
- barriers are sunk to the first conflicting consumer.

This keeps synchronization decisions in SDE and lets Core preserve narrow phase
edges instead of global barriers.

### Dependency Windows

Root equality is conservative. Add optional affine footprint summaries for
structured accesses so SDE can prove non-overlapping windows on the same root.

Useful proofs:

- disjoint block writes to one vector or tensor;
- read-only halo windows contained inside a planned block plus halo;
- matmul output tile writes disjoint across workers;
- stream or activation phases with identical owner chunks and disjoint outputs.

When SDE proves a window, it should stamp the dependency-window intent in the
plan so Core lowers window-local dependencies instead of global block ids over
one coarse storage object.

### Explicit Phase Edges

Multi-phase kernels need order without global serialization. SDE should encode
producer-consumer edges for:

- `2mm`: `tmp = A * B` before `D = tmp * C + beta * D`;
- `3mm`: independent `E = A * B` and `F = C * D`, then `G = E * F`;
- `batchnorm` and `layernorm`: reductions before normalize/affine phases;
- `pooling`: local window reductions before global-average or output phases.

The edge should be a plan-level ordering fact, not a late Core heuristic.

### Taskwait And Barrier Semantics

OpenMP taskwait and barrier semantics should remain distinct in SDE. A taskwait
can be removed only when the SDE dependency proof subsumes the task ordering it
represents. Otherwise it must survive as an explicit ordering edge.

### Timestep And Wavefront Dependencies

SDE should bundle wavefront and timestep legality with the dependency plan:

- Jacobi-style double-buffer loops can use k-step grouping when buffer parity
  and halo requirements are proved.
- Seidel-style in-place loops need macro-tile wavefront tasks with diagonal
  predecessor edges.
- `fdtd-2d`-style sequential timestep loops should lower to staged per-timestep
  tasks with barriers only between dependent fields.

Persistent regions are a later option after non-persistent SDE plans are
correct and profiling shows launch/CPS overhead dominates.

## Pass Grouping Proposal

Dependency planning should run after loop/tile shape is stable enough to reason
about windows, and before `convert-sde-to-codir`:

```text
PatternAnalysis
LoopInterchange
Tiling
ElementwiseFusion
Vectorization
ReductionStrategy
DistributionPlanning
IterationSpaceDecomposition
BarrierElimination or PhaseGraphPlanning
VerifySdeCpsPlan
MemoryUnitMaterialization
ConvertSdeToCodir
ConvertCodirToArts
```

If `IterationSpaceDecomposition` changes the scheduling-unit shape, phase graph
construction should run after decomposition. If decomposition moves earlier,
`DistributionPlanning` can use the final interior/boundary shape directly.

## Risks

- Over-aggressive barrier removal can violate OpenMP ordering when unknown or
  aliased effects are under-modeled.
- Window proofs over one root need affine and subview-aware alias reasoning;
  root-level sets alone are not enough.
- Wavefront transforms can incorrectly parallelize in-place Gauss-Seidel if the
  diagonal predecessor graph is incomplete.
- Phase sinking must preserve explicit taskwait semantics unless the dependency
  proof is stronger than the taskwait.

## Tests

- Positive and negative SDE tests for barrier sink, coalesce, and eliminate.
- Same-root disjoint-window tests and overlapping-window rejection tests.
- `3mm`-style phase graph tests proving independent first phases are not
  serialized.
- Taskwait preservation tests.
- SDE-to-CODIR and CODIR-to-ARTS tests proving eliminated barriers do not emit
  downstream synchronization, and preserved or narrowed phase edges remain
  visible to ARTS.
