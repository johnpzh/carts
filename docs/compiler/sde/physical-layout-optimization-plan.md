# SDE Physical Layout Optimization Plan

## Purpose

The large/64 benchmark sweep is correctness-clean, but most benchmarks remain
slower than OpenMP because task shape and physical DB layout are still missing
or too coarse when control reaches Core. The fix is to move layout, tiling,
dependency-window, and phase-shape decisions into SDE while OpenMP semantics,
structured summaries, tensor carriers, reductions, and barriers are still
available.

Layer limits:

- SDE owns semantic recognition, legality proofs, tiling, task shape, barrier
  intent, reduction strategy, and physical DB layout policy.
- Core owns materializing SDE-authored plans in `CreateDbs`, preserving the
  chosen DB/EDT shape, validating contracts, and lowering dependency windows.
- RT/runtime work waits until the SDE/Core shape is present and traces still
  show launch, CPS, dependency, or scheduling overhead.

## Shared Contract

Every optimization should produce or refine `arts.plan.*` before
`ConvertSdeToArts`:

- `arts.plan.kernel_family`
- `arts.plan.owner_dims`
- `arts.plan.physical_block_shape`
- `arts.plan.logical_worker_slice`
- `arts.plan.halo_shape`
- `arts.plan.iteration_topology`
- `arts.plan.repetition_structure`
- `arts.plan.async_strategy`
- `arts.plan.cost.*`

The plan belongs on the SDE scheduling unit that owns the proof. `CreateDbs`
must create the chosen physical DB layout directly; late ARTS heuristics may
refine mechanics but must not invent partition policy.

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

SDE should normalize matmul-like loops to output-tile ownership, choose tile
sizes from the cost model, and preserve real phase edges for chained products.
`3mm` first-phase products should remain independent when their roots are
disjoint; `2mm` intermediates should become physical DB blocks before the
consumer phase.

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
