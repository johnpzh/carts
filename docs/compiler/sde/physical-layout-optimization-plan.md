# SDE Physical Layout Optimization Plan

## Purpose

The large/64 benchmark sweep is correctness-clean, but most benchmarks remain
slower than OpenMP because task shape and physical storage layout are still
missing or too coarse when control reaches ARTS. The fix is to move layout,
tiling, dependency-window, and phase-shape decisions into SDE while OpenMP
semantics, SDE pattern analysis, memref access facts, reductions, and barriers
are still available, then materialize isolated codelets in CODIR before ARTS
DB/EDT creation.

Layer limits:

- SDE owns semantic recognition, legality proofs, tiling, task shape, barrier
  intent, reduction strategy, and target-neutral physical layout/grain intent.
- CODIR owns codelet isolation, explicit deps/params, and token-local memref
  access rewrites from the SDE MU/CU/SU plan.
- ARTS owns materializing SDE/CODIR-authored plans, preserving the chosen
  storage/task shape, validating contracts, lowering dependency windows, and
  binding SDE logical-resource queries to target-specific mechanisms.
- ARTS-RT/runtime work waits until the SDE/CODIR/ARTS shape is present and
  traces still show launch, CPS, dependency, or scheduling overhead.

## Shared Contract

Every optimization should produce or refine the SDE plan before SDE-to-CODIR
materialization:

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
- planned codelet deps, params, and token-local view requirements

The plan belongs on the SDE scheduling unit that owns the proof.
`PatternAnalysis` and later SDE transforms consume the plan directly inside SDE.
`ConvertSdeToCodir` should lower the final plan into isolated codelet deps,
params, and token-local memref views. `ConvertCodirToArts` should then create
ARTS planning and dependency contracts at the boundary; downstream storage
materialization must create the chosen physical layout directly from that
lowered contract. Late target heuristics may refine mechanics but must not
invent partition policy. SDE passes must not materialize target runtime queries
or concrete worker ids directly.

## Optimization Tracks

### Output Storage Plan Synthesis

Target `gemm`, `2mm`, `3mm`, `atax`, `bicg`, `stream`, ML kernels, and
`seissol/volume-integral`.

SDE should prove owner dims and block shapes for write-backed outputs and
intermediates, then stamp physical plans before CODIR/ARTS. Reader-only inputs
stay coarse unless SDE proves a reuse-friendly input tile. This is the
lowest-risk track because it reuses existing physical plan consumers and
addresses the largest class of coarse-storage bottlenecks.

### Matmul And Contraction Tiling

Target `gemm`, `2mm`, `3mm`, and batched memref contractions.

SDE should normalize matmul-like loops to contraction-tile ownership, choose
tile sizes from memref shape, cache/reuse facts, and abstract logical capacity,
and preserve real phase edges for chained products. Direct-memory output
ownership must stay row-strip until SDE can also prove packed A/B panel or
intermediate-tile reuse. `3mm` first-phase products should remain independent
when their roots are disjoint. The latest working-tree large/64 evidence is
correctness-clean but slow after the DB-payload subview removal, so prioritize
pipeline-shape comparison, phase-local physical storage for `tmp`, `E`, and
`F`, and explicit producer/consumer reuse before adding new tile-size
heuristics.

### Vector, Reduction, And Elementwise Fusion

Target `stream`, `atax`, `bicg`, `activations`, `batchnorm`, `layernorm`, and
`pooling`.

SDE should fuse compatible block pipelines, stamp blocked vector outputs, and
make reductions explicit as local-accumulate, tree, or atomic strategies. The
goal is to avoid many tiny phases over coarse storage and keep simple memory
kernels from being dominated by barriers and launch overhead.

### 3D Component Stencil Slabs

Target `specfem3d/*` and `sw4lite/*`.

SDE should distinguish spatial, component, and batch dimensions, then stamp
component-aware slab layouts with bounded halo windows. `CreateDbs` should
materialize write-backed slab storage directly so 3D stencil tasks read and
write local slabs instead of coarse component memrefs.

### Timestep And Wavefront Shape

Target `jacobi2d`, `seidel-2d`, KaStORS Jacobi, and disabled `fdtd-2d`.

SDE should represent repeated timesteps, k-step grouping, wavefront edges, and
safe async strategy before CODIR/ARTS. In-place Seidel-style loops must not be
owner-strip parallelized without a wavefront or split-phase proof.

### Barrier Placement Planning

Current barrier elimination only proves adjacent disjoint scheduling units. The
next step is a phase graph over SDE scheduling units with root/window conflicts
as edges. That graph can eliminate redundant barriers, coalesce duplicates, and
sink synchronization to the first conflicting consumer while preserving OpenMP
taskwait and barrier semantics when proof is missing.

## Rollout Order

1. Output storage plan synthesis.
2. Matmul and contraction tiling.
3. Vector/reduction fusion and barrier planning.
4. 3D component stencil slab planning.
5. Timestep and wavefront planning.
6. CODIR isolation and token-local access rewrite work in parallel with the
   first two tracks.
7. ARTS-RT/runtime follow-up only after planned SDE/CODIR/ARTS shapes remain
   bottlenecked.

## Benchmark Success Criteria

- Checksum parity remains mandatory.
- No benchmark-specific constants in compiler decisions.
- `fast` means generated task-runtime kernel time is faster than OpenMP.
- `competitive` means generated task-runtime kernel time is within `1.25x` of
  OpenMP.
- `blocked` requires a named compiler/runtime limitation and next SDE, CODIR,
  ARTS, ARTS-RT, or runtime owner.
