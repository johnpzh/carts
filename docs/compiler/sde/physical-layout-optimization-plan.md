# SDE Physical Layout Optimization Plan

## Purpose

The large/64 benchmark sweep is correctness-clean, but most benchmarks remain
slower than OpenMP because task shape and physical DB layout are still missing
or too coarse when control reaches Core. The fix is to move layout, tiling,
dependency-window, and phase-shape decisions into SDE while OpenMP semantics,
SDE pattern analysis, tensor carriers, reductions, and barriers are still
available.

Layer limits:

- SDE owns semantic recognition, legality proofs, tiling, task shape, barrier
  intent, reduction strategy, and target-neutral physical layout/grain intent.
- Core owns materializing SDE-authored plans in `CreateDbs`, preserving the
  chosen DB/EDT shape, validating contracts, lowering dependency windows, and
  binding SDE logical-resource queries to ARTS runtime queries.
- RT/runtime work waits until the SDE/Core shape is present and traces still
  show launch, CPS, dependency, or scheduling overhead.

## Shared Contract

Every optimization should produce or refine the SDE plan before
`ConvertSdeToArts`:

- `pattern = #sde.pattern<...>`
- `ownerDims` and `spatialDims`
- `physicalOwnerDims`
- `physicalBlockShape`
- `logicalWorkerSlice`
- `physicalHaloShape`
- `sde.resource_query <logical_workers>` for symbolic grain expressions
- `iterationTopology`
- `repetitionStructure`
- `asyncStrategy`
- reduction and CPS plan attributes

The plan belongs on the SDE scheduling unit that owns the proof.
`PatternAnalysis` and later SDE transforms consume the plan directly inside SDE.
`ConvertSdeToArts` lowers the final plan into Core `arts.plan.*` and dependency
contracts at the boundary; `CreateDbs` must create the chosen physical DB layout
directly from that lowered contract. Late ARTS heuristics may refine mechanics
but must not invent partition policy. SDE passes must not materialize
`arts.runtime_query` or ARTS worker ids directly.

## Optimization Tracks

### Output DB Plan Synthesis

Target `gemm`, `2mm`, `3mm`, `atax`, `bicg`, `stream`, ML kernels, and
`seissol/volume-integral`.

SDE should prove owner dims and block shapes for write-backed outputs and
intermediates, then stamp physical plans before Core. Reader-only inputs stay
coarse unless SDE proves a reuse-friendly input tile. This is the lowest-risk
track because it reuses existing physical plan consumers and addresses the
largest class of coarse-DB bottlenecks.

### Matmul And Tensor-Contraction Tiling

Target `gemm`, `2mm`, `3mm`, and batched tensor contractions.

SDE should normalize matmul-like loops to contraction-tile ownership, choose
tile sizes from tensor shape, cache/reuse facts, and abstract logical capacity,
and preserve real phase edges for chained products. Direct-memory output
ownership must stay row-strip until SDE can also prove packed A/B panel or
intermediate-tile reuse. `3mm` first-phase products should remain independent
when their roots are disjoint. Current repeated large/64 evidence has `gemm`
faster than OpenMP at median, while `2mm` and `3mm` are blocked or unstable, so
prioritize phase-local physical DBs for `tmp`, `E`, and `F` with explicit
producer/consumer reuse before adding new tile-size heuristics.

### Vector, Reduction, And Elementwise Fusion

Target `stream`, `atax`, `bicg`, `activations`, `batchnorm`, `layernorm`, and
`pooling`.

SDE should fuse compatible block pipelines, stamp blocked vector outputs, and
make reductions explicit as local-accumulate, tree, or atomic strategies. The
goal is to avoid many tiny epochs over coarse DBs and keep simple memory kernels
from being dominated by barriers and launch overhead.

### 3D Component Stencil Slabs

Target `specfem3d/*` and `sw4lite/*`.

SDE should distinguish spatial, component, and batch dimensions, then stamp
component-aware slab layouts with bounded halo windows. `CreateDbs` should
materialize write-backed slab DBs directly so 3D stencil tasks read and write
local slabs instead of coarse component tensors.

### Timestep And Wavefront Shape

Target `jacobi2d`, `seidel-2d`, KaStORS Jacobi, and disabled `fdtd-2d`.

SDE should represent repeated timesteps, k-step grouping, wavefront edges, and
safe async strategy before Core. In-place Seidel-style loops must not be
owner-strip parallelized without a wavefront or split-phase proof.

### Barrier Placement Planning

Current barrier elimination only proves adjacent disjoint scheduling units. The
next step is a phase graph over SDE scheduling units with root/window conflicts
as edges. That graph can eliminate redundant barriers, coalesce duplicates, and
sink synchronization to the first conflicting consumer while preserving OpenMP
taskwait and barrier semantics when proof is missing.

## Rollout Order

1. Output DB plan synthesis.
2. Matmul and tensor-contraction tiling.
3. Vector/reduction fusion and barrier planning.
4. 3D component stencil slab planning.
5. Timestep and wavefront planning.
6. RT/runtime follow-up only after planned SDE/Core shapes remain bottlenecked.

## Benchmark Success Criteria

- Checksum parity remains mandatory.
- No benchmark-specific constants in compiler decisions.
- `fast` means ARTS kernel time is faster than OpenMP.
- `competitive` means ARTS kernel time is within `1.25x` of OpenMP.
- `blocked` requires a named compiler/runtime limitation and next SDE/Core
  owner.
