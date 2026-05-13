# SDE Physical Layout Optimization Plan

## Purpose

The benchmark sweep shows that correctness is no longer the main blocker for
the runnable large/64 suite. The main blocker is that many benchmarks still
reach Core with task shape and physical DB layout decisions either missing or
too coarse. This plan moves those decisions into SDE, where OpenMP semantics,
structured summaries, tensor carriers, reductions, and barriers are still
available.

Layer limits:

- SDE owns OpenMP semantics, structured family recognition, legality proofs,
  tiling, loop/task shape, barrier intent, and physical DB layout policy.
- Core owns materializing SDE-authored plans in `CreateDbs`, preserving the
  chosen EDT/DB shape, validating contracts, and lowering dependency windows.
- RT/runtime work is deferred until the SDE/Core shape is present and the
  remaining bottleneck is demonstrably launch, CPS, dependency, or runtime
  scheduling overhead.

Current evidence:

- Full large/64 sweep:
  `.carts/outputs/benchmarks-large-64-final/20260512_204622`.
- Focused 3-run large/64 stability sweep:
  `.carts/outputs/benchmarks-large-64-final-focused/20260512_210324`.
- Current large/64 outcome: 23 runnable benchmarks pass with no timeouts, but
  only `polybench/convolution-3d` and `polybench/correlation` are `fast`, and
  only `polybench/convolution-2d` is `competitive`.

## Shared Plan Contract

Every optimization below should produce or refine the existing `arts.plan.*`
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

The contract should be stamped on the SDE scheduling unit that owns the semantic
proof. `CreateDbs` should consume the plan and create the physical DB family
directly. Late ARTS heuristics may refine mechanics, but must not invent tensor
partition policy.

## Optimization 1: Output DB Plan Synthesis

Target benchmarks:

- `polybench/gemm`, `polybench/2mm`, `polybench/3mm`
- `polybench/atax`, `polybench/bicg`
- `stream`
- `ml-kernels/activations`, `ml-kernels/batchnorm`,
  `ml-kernels/layernorm`, `ml-kernels/pooling`
- `seissol/volume-integral`

Problem:

Several kernels are distributed as tasks over late slices of one coarse DB. The
runner confirms they are checksum-clean but slow because `CreateDbs` receives no
physical layout plan for the real outputs or intermediates.

Proposal:

1. Extend SDE distribution planning from the current uniform/stencil stamping
   into a family-specific output planner.
2. Use `findLoopIndexedOutputPlan`, structured memory effects, tensor carrier
   summaries, and linalg indexing maps to prove which loop IVs own each output
   dimension.
3. Stamp:
   - `kernel_family = uniform`, `reduction_mixed`, or a matmul/tensor family
     encoded through the plan vocabulary,
   - `owner_dims`,
   - `physical_block_shape`,
   - `logical_worker_slice`,
   - `iteration_topology = owner_strip` or `owner_tile`.
4. Keep reader-only inputs coarse unless SDE proves a reuse-friendly input tile.
   Start by physically partitioning write-backed outputs and intermediates.
5. Make `CreateDbs` materialize only write-backed SDE plans as physical block
   DBs; reader-only plans remain advisory unless the proof explicitly permits
   physical input partitioning.

Implementation surface:

- `lib/arts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`
- `lib/arts/dialect/sde/Analysis/StructuredOpAnalysis.cpp`
- `include/arts/dialect/sde/Analysis/SdeAnalysisUtils.h`
- `lib/arts/dialect/core/Transforms/db/CreateDbs.cpp`
- `lib/arts/dialect/core/Transforms/db/DbLayoutPlanUtils.cpp`

Initial tests:

- Positive SDE tests for output DB plans on uniform vector writes, matmul
  outputs, and two-phase vector outputs.
- Negative tests where a self-read in-place write, unknown effects, or escaped
  root prevents physical output planning.
- Core tests proving `CreateDbs` creates block DBs only from write-backed SDE
  plans.

## Optimization 2: Matmul And Tensor-Contraction Tiling

Target benchmarks:

- `polybench/gemm`
- `polybench/2mm`
- `polybench/3mm`
- `seissol/volume-integral`

Problem:

The matmul/tensor benchmarks are correct but still see coarse output DBs and
late per-task slices. `gemm` is `0.404x` at large/64; `2mm` and `3mm` are below
`0.20x`; `seissol/volume-integral` is below the competitive threshold.

Proposal:

1. In SDE, normalize matmul-like loops to explicit output-tile ownership:
   `C[i_tile, j_tile]` or `Out[element_tile, basis_tile]`.
2. Pick tile sizes from the SDE cost model, not benchmark constants:
   - enough flops per EDT to amortize launch,
   - row/column tile fits cache target,
   - reduction `K` remains local at first.
3. Stamp output/intermediate physical DB shapes:
   - `gemm`: block or 2-D tile `C`;
   - `2mm`: block `tmp` and `D` with an explicit phase edge;
   - `3mm`: block `E`, `F`, and `G` with two independent first phases and one
     dependent final phase;
   - `seissol`: block `fluxOut` by element batch, with read-only tensor reuse.
4. Preserve barriers only for true producer/consumer phase edges. Independent
   `3mm` first-phase products should not be serialized by a global barrier when
   their read/write roots are disjoint.
5. Emit `owner_tile` task topology when both row and column ownership are
   proven; otherwise emit `owner_strip` as a conservative first step.

Expected effect:

- DB creation matches the matmul output shape instead of producing one coarse
  output DB.
- Runtime dependencies become tile-local for outputs and phase-local for
  intermediates.
- `TIME_EDT_EXEC` should stop tracking total wall time for ostensibly parallel
  matmul phases.

## Optimization 3: Vector, Reduction, And Elementwise Phase Fusion

Target benchmarks:

- `stream`
- `polybench/atax`, `polybench/bicg`
- `ml-kernels/activations`
- `ml-kernels/batchnorm`
- `ml-kernels/layernorm`
- `ml-kernels/pooling`

Problem:

These benchmarks expose many independent or phase-ordered vector loops. Today
the compiler often creates multiple epochs over coarse DBs and only slices them
at acquire time. Barriers and phase boundaries dominate simple memory kernels.

Proposal:

1. Extend `ElementwiseFusion` from adjacent pointwise loops into block-pipeline
   fusion when loops share the same owner IV and have no root-level conflict.
2. Add a `reduction_mixed` SDE plan for vector loops that have local maps plus
   reductions:
   - local block accumulation inside each owner chunk,
   - tree or local-accumulate strategy chosen by `ReductionStrategy`,
   - physical output/vector DBs for the block-owned arrays.
3. Add phase-aware barrier planning:
   - eliminate barriers between disjoint outputs,
   - coalesce adjacent barriers that protect the same producer/consumer edge,
   - sink barriers until immediately before the first conflicting consumer.
4. For `stream`, fuse copy/scale/add/triad into a block pipeline when SDE can
   prove same-shape arrays and no cross-block dependence. Keep the verification
   reduction as a separate reduction phase.
5. For `activations`, fuse independent maps over the same input into one owner
   task that writes multiple blocked outputs.
6. For `batchnorm` and `layernorm`, make mean/variance reductions first-class
   row/channel plans, then run normalize/affine as owner-local follow-up phases.

Expected effect:

- Fewer EDTs and epochs for tiny per-element work.
- Physical vector outputs are created before Core.
- Repeated full coarse-input dependencies are replaced by block-local reads
  where SDE proves exact owner-local reachability.

## Optimization 4: 3D Component Stencil Slab Planning

Target benchmarks:

- `specfem3d/stress`
- `specfem3d/velocity`
- `sw4lite/rhs4sg-base`
- `sw4lite/vel4sg-base`

Problem:

SDE recognizes 3D stencil metadata late enough to drive task metadata, but
`CreateDbs` still materializes component tensors as coarse DBs. The final
large/64 runs are correctness-clean but roughly 19x-25x slower than OpenMP.

Proposal:

1. Teach structured summaries to distinguish spatial dims, component dims, and
   element/batch dims for component stencils.
2. Choose a slab owner dimension from proof and cost:
   - prefer element or z/slab ownership when halo is small,
   - keep component dims local inside the task,
   - avoid splitting across dims that force full-range reads.
3. Stamp:
   - `kernel_family = stencil`,
   - `owner_dims` for the proven slab dimension,
   - `physical_block_shape` for component-aware slabs,
   - `halo_shape` from min/max offsets,
   - `logical_worker_slice` including component-local extents.
4. Extend `CreateDbs` to materialize component arrays as slab/block DBs when
   the plan is write-backed and halo-legal.
5. Validate dependency windows are local to the chosen slab plus halo, not
   global component tensors.

Expected effect:

- 3D stencil tasks operate on physical z/element slabs instead of coarse arrays.
- Halo widening is explicit and bounded by SDE proof.
- SW4Lite large should remain under timeout with substantially lower kernel
  time, not merely complete.

## Optimization 5: Timestep And Wavefront Task Shape

Target benchmarks:

- `polybench/jacobi2d`
- `polybench/seidel-2d`
- `kastors-jacobi/jacobi-for`
- `kastors-jacobi/poisson-for`
- disabled `polybench/fdtd-2d`

Problem:

Timestep kernels need more than block partitioning. Jacobi-like loops pay
relaunch and barrier overhead. In-place Seidel cannot be owner-strip
parallelized without a wavefront/split-phase proof. `fdtd-2d` is blocked by
nested timestep loop lowering.

Proposal:

1. Add SDE `timestep_chain` and `wavefront` planning before Core:
   - `repetition_structure = pair_step`, `k_step`, or `full_timestep`,
   - `iteration_topology = wavefront`, `owner_strip`, or `persistent_tile`,
   - `async_strategy = cps_chain`, `advance_edt`, or `persistent_region`.
2. For double-buffer Jacobi, apply time tiling or k-step grouping when SDE
   proves buffer parity and halo radius. Run multiple timesteps per launch to
   amortize CPS and barriers.
3. For in-place Seidel, generate macro-tile wavefront tasks with explicit
   diagonal predecessor edges. Do not create physical block DBs from unsafe
   owner-strip assumptions.
4. For `fdtd-2d`, emit staged per-timestep EDTs for each OpenMP loop inside the
   sequential timestep loop, with barriers only between dependent fields.
5. Use persistent regions only after the non-persistent plan is proven and the
   cost model predicts launch overhead dominates useful work.

Expected effect:

- No unsafe parallelization of in-place dependences.
- Fewer CPS relaunches for repeated timesteps.
- `fdtd-2d` becomes runner-clean once nested `arts.for` timestep shape is
  represented as an SDE plan instead of reaching `EdtDistribution` unsupported.

## Optimization 6: Barrier Placement Planning

Target benchmarks:

- All multi-phase kernels, especially `stream`, ML kernels, `2mm`, `3mm`,
  Jacobi/Seidel, and `fdtd-2d`.

Problem:

Current SDE barrier elimination proves adjacent disjoint scheduling units and
marks barriers as eliminated. That is necessary but not enough for phase-heavy
benchmarks: barriers can often be sunk, coalesced, or replaced with narrower
phase edges.

Proposal:

1. Extend barrier elimination into a small SDE phase graph:
   - nodes are `su_iterate`/`su_distribute` scheduling units,
   - edges are root-level RAW/WAR/WAW conflicts,
   - non-conflicting barriers are eliminated,
   - duplicate barriers guarding the same edge are coalesced,
   - barriers are sunk to the earliest conflicting consumer.
2. Preserve explicit OpenMP/taskwait semantics unless the same root-level proof
   justifies removal.
3. Stamp phase edges into the plan when Core must preserve ordering without a
   global barrier.
4. Keep barrier planning before `ConvertSdeToArts`; Core should receive either
   no barrier, a narrow phase edge, or a deliberately preserved barrier.

Expected effect:

- Phase-heavy vector and ML kernels avoid global synchronization where only a
  subset of roots are dependent.
- Matmul chains preserve true intermediate order without serializing unrelated
  phases.
- Timestep kernels keep correctness while reducing redundant global barriers.

## Rollout Order

1. **Output DB Plan Synthesis**
   - Lowest risk and highest reuse.
   - Covers vector outputs, matmul outputs, ML output tensors, and several
     existing `arts.plan.*` consumers.
2. **Matmul/Tensor-Contraction Tiling**
   - Builds on output planning.
   - Targets `gemm`, `2mm`, `3mm`, and `seissol/volume-integral`.
3. **Vector/Reduction Phase Fusion And Barrier Planning**
   - Targets the largest throughput losses in `stream`, ML kernels, `atax`,
     and `bicg`.
4. **3D Component Stencil Slabs**
   - Requires stronger component/spatial proofs but targets the large 3D
     stencil slowdowns.
5. **Timestep/Wavefront Plans**
   - Highest semantic risk; implement after the plan/proof/verifier surface is
     stronger.
6. **Runtime/RT Follow-Up**
   - Only after benchmarks show that planned SDE/Core DB/task shapes are still
     bottlenecked by launch, CPS continuation, or dependency mechanics.

## Verification Gates

For each optimization:

- Add focused SDE lit tests for positive plan creation and negative fallback.
- Add Core tests showing `CreateDbs` consumes the SDE-authored physical layout
  without inventing late policy.
- Run the owning focused benchmark at `--size large --threads 64 --nodes 1`.
- Run `dekk carts test`.
- Run affected e2e cases or `dekk carts test --suite e2e` when lowering or
  shared analysis changes.
- Run the full large/64 benchmark sweep before updating performance
  classifications.

Benchmark success criteria:

- correctness parity remains mandatory;
- no benchmark-specific constants in compiler decisions;
- `fast` means ARTS kernel time is faster than OpenMP;
- `competitive` means ARTS kernel time is within `1.25x` of OpenMP;
- `blocked` requires a named compiler/runtime limitation and next SDE/Core
  owner.
