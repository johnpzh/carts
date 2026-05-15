# Large/64 Performance Plan

## Objective

Make every maintained benchmark correctness-clean and performance-credible at
`large`, 64 threads, and 1 node without benchmark-specific constants or late
ARTS policy inference.

The detailed result log remains
[`../benchmark-performance-goal.md`](../benchmark-performance-goal.md). This
subplan owns the execution strategy.

## Current Evidence

Current maintained large/64 sweep:

- Result directory:
  `.carts/outputs/benchmarks-large-64-maintained-20260515/20260515_065122`
- Command:
  `dekk carts benchmarks run --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-large-64-maintained-20260515`
- Revisions: CARTS `ce4e61e1`, carts-benchmarks `20db3fd`, ARTS `c777522`.
- Counts: 23 configured entries, 11 passed, 12 failed, 0 skipped by the
  runner.
- Geometric mean kernel speedup over reported passed results: `1.73x`.
- Classes:
  - `fast`: `ml-kernels/batchnorm`, `polybench/2mm`, `polybench/3mm`,
    `polybench/correlation`, `polybench/gemm`, `sw4lite/rhs4sg-base`.
  - `competitive`: `kastors-jacobi/jacobi-for`,
    `kastors-jacobi/poisson-for`.
  - `blocked`: `ml-kernels/activations`, `ml-kernels/layernorm`,
    `ml-kernels/pooling`, `monte-carlo/ensemble`, `polybench/atax`,
    `polybench/bicg`, `polybench/convolution-2d`,
    `polybench/convolution-3d`, `polybench/jacobi2d`,
    `polybench/seidel-2d`, `seissol/volume-integral`,
    `specfem3d/stress`, `specfem3d/velocity`, `stream`,
    `sw4lite/vel4sg-base`.

All 12 failed entries fail at compile time with the same boundary diagnostic:
`SDE-authored physical DB layout reached CreateDbs as a raw memref`. This is an
M3 coverage gap. The fix is not to make ARTS rediscover layout in `CreateDbs`;
the fix is to route those plans through SDE MU/token/CODIR materialization.
Until that coverage exists, SDE must drop unsupported physical storage attrs
before the coarse raw bridge and the plan must remain guarded.

Focused boundary follow-up:

- Result directory:
  `.carts/outputs/benchmarks-raw-layout-demotion-12failures-final-20260515/20260515_072611`
- Command:
  `dekk carts benchmarks run ml-kernels/layernorm ml-kernels/pooling monte-carlo/ensemble polybench/atax polybench/bicg polybench/convolution-2d polybench/convolution-3d polybench/jacobi2d seissol/volume-integral specfem3d/stress specfem3d/velocity sw4lite/vel4sg-base --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-raw-layout-demotion-12failures-final-20260515`
- Counts: 12 passed, 0 failed, 0 skipped.
- Geometric mean kernel speedup for the formerly failing slice: `0.92x`.
- Result: the raw-layout boundary class is fixed for these maintained entries.
  SDE now demotes unsupported physical storage attrs before the raw
  `CreateDbs` bridge, while supported MU/token/codelet plans still materialize
  directly and `CreateDbs` still rejects raw blocked layouts.
- Remaining slow entries in this slice: `polybench/jacobi2d` is blocked on
  timestep/wavefront grain (`0.222x`), while `polybench/atax` (`0.786x`) and
  `polybench/convolution-2d` (`0.791x`) are just outside the competitive gate.
  `polybench/bicg`, `ml-kernels/pooling`, and `seissol/volume-integral` are
  competitive but below the fast gate.

Latest focused working-tree matrix checkpoints:

- `gemm` verification rerun:
  `.carts/outputs/benchmark-migration-20260515/20260515_092802`
  (`dekk carts benchmarks run polybench/gemm --size large --threads 64 --arts
  --runs 1 --timeout 180 --results-dir
  .carts/outputs/benchmark-migration-20260515 --trace`): checksum-clean,
  ARTS kernel `0.481429s`, checksum `2.211832507057e+05`.
- `gemm` result directory:
  `.carts/outputs/benchmarks-gemm-large-64-crossphase-guard-20260515/20260515_064151`
- `2mm` and `3mm` result directory:
  `.carts/outputs/benchmarks-2mm-3mm-large-64-crossphase-coarse-20260515/20260515_064040`
- `gemm`: `0.408783s` ARTS, `6.215503s` OpenMP, `15.20x`
- `2mm`: `0.853778s` ARTS, `5.554953s` OpenMP, `6.51x`
- `3mm`: `0.691686s` ARTS, `4.905503s` OpenMP, `7.09x`

All focused matrix runs are checksum-clean and fast. The `gemm` task shape
regression is fixed on the production SDE -> CODIR -> ARTS codelet path: the
EDT/epoch plan preserves `physicalBlockShape = [75, 4800]` and the lowered
worker body executes 75-row owner slices. The root DB allocation may still stay
coarse when the same memref has host initialization or verification accesses
outside the selected scheduling unit; the current guard deliberately avoids
handing a block-backed first-slice view to whole-matrix host code. `2mm` and
`3mm` are also fast after the conservative cross-phase guard: block layout is
used only when every current access stays inside the selected scheduling unit,
and cross-phase intermediates fall back to coarse materialization until M3 adds
explicit phase-local MU/token plans.

This focused evidence supersedes the slow 2026-05-14 no-subview matrix
checkpoint. The matrix family stayed fast in the maintained sweep above. The
large/64 goal remains partial because the former compile failures are now
checksum-clean but several maintained entries remain slower than the
performance gate.

## Performance Rules

- No hardcoded benchmark constants in passes.
- No SDE dependency on ARTS runtime topology.
- No ARTS rediscovery of source-level tiling policy.
- No ARTS-RT optimization before DB/EDT/dependency shape is correct.
- Every claim needs checksum parity and repeated evidence when noisy.

## Workstreams

### Matrix Contractions

Targets: `gemm`, `2mm`, `3mm`.

See [`memref-mu-token-rewrite.md`](./memref-mu-token-rewrite.md) for the
underlying MU/token rewrite that owns block shape, K tile, and intermediate
reuse plumbing.

SDE responsibilities:

- classify contractions and phase chains;
- preserve reduction locality;
- plan owner dims, block shape, K tile, and phase-local intermediate reuse;
- model packed-panel or microkernel shape only when reuse is proven.

CODIR responsibilities:

- expose explicit A/B/C/intermediate deps and scalar params;
- rewrite token-local accesses for output and intermediate tiles.

ARTS responsibilities:

- materialize output/intermediate DBs and acquires directly from the plan;
- preserve task count, block shape, and phase boundaries.

ARTS-RT responsibilities:

- run generic LICM, scalar replacement, vectorization metadata, and pointer
  cleanup after codelet shape is correct.

Immediate tasks:

1. Keep `CreateDbs` guarded as coarse-only for raw memrefs; expand M3
   MU/token/CODIR materialization coverage for the currently demoted plans only
   when token-local rewrites exist. For GEMM-like arrays with host
   initialization/verification, the production fix is a whole-array host view
   over block DB roots, not re-enabling unsafe first-block substitution.
2. For noisy, borderline, or unexpectedly slow matrix results, run a focused
   `--runs 3` follow-up and compare DB layout, task count, acquire windows,
   dependency count, and runtime variance.
3. Implement real M3 phase-local contraction plans for chained intermediates,
   replacing the current conservative coarse bridge where reuse is provable.
4. Tune the passed-but-slow groups: `ml-kernels/activations`,
   `polybench/atax`, `polybench/convolution-2d`, `polybench/jacobi2d`,
   `polybench/seidel-2d`, and `stream`.

### Vector And Reductions

Targets: `stream`, `activations`, `batchnorm`, `layernorm`, `atax`, `bicg`.

Priorities:

- block vector outputs and inputs where legal;
- distinguish scalar reductions from per-output reductions;
- fuse compatible elementwise stages before task creation;
- choose local accumulate, tree, or atomic strategy in SDE.

### Stencils And Timesteps

Targets: `jacobi2d`, `seidel-2d`, `fdtd-2d`, KaStORS Jacobi.

Priorities:

- model halo/window shapes in SDE;
- represent timestep and wavefront legality before CODIR/ARTS;
- keep in-place updates out of owner-strip parallelization without proof;
- lower bounded neighbor windows to local dependency slots.

### 3D Component Slabs

Targets: `specfem3d/*`, `sw4lite/*`, `convolution-3d`.

Priorities:

- separate spatial, component, and batch dimensions;
- keep component locality when it improves reuse;
- materialize slab/halo windows directly from SDE/CODIR.

## Evidence Protocol

For every benchmark change:

1. Run the focused benchmark with trace enabled.
2. Dump the affected pipeline stages under `.carts/sessions/...`.
3. Record checksum parity.
4. Record task count, DB count, acquire count, dependency count, and epoch
   shape when available.
5. Run `--runs 3` for noisy or surprising results.
6. Update only the current summary in `benchmark-performance-goal.md`.

## Success Criteria

- `fast`: ARTS kernel is faster than OpenMP.
- `competitive`: ARTS kernel is within `1.25x` of OpenMP.
- `blocked`: a named owner and next action explain why performance is not yet
  credible.

The large/64 goal is complete only when every maintained benchmark is classified
and every blocked group has a concrete SDE, CODIR, ARTS, ARTS-RT, runtime, or
tooling owner.
