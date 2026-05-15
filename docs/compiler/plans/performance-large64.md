# Large/64 Performance Plan

## Objective

Make every maintained benchmark correctness-clean and performance-credible at
`large`, 64 threads, and 1 node without benchmark-specific constants or late
ARTS policy inference.

The detailed result log remains
[`../benchmark-performance-goal.md`](../benchmark-performance-goal.md). This
subplan owns the execution strategy.

## Current Evidence

Latest working-tree matrix checkpoint:

- Result directory:
  `.carts/outputs/benchmarks-gemm-family-large-64-no-boundary-subview-20260514/20260514_231930`
- `gemm`: `16.851228s` ARTS, `6.257675s` OpenMP, `0.371x`
- `2mm`: `30.755523s` ARTS, `5.777342s` OpenMP, `0.188x`
- `3mm`: `26.141955s` ARTS, `4.973020s` OpenMP, `0.190x`

This checkpoint is correctness-clean but slow. It fixed a lowering crash by
removing DB-payload subviews; it did not solve the performance shape.

Earlier focused runs showed faster matrix behavior, so the first action is a
shape comparison, not a new heuristic.

## Performance Rules

- No hardcoded benchmark constants in passes.
- No SDE dependency on ARTS runtime topology.
- No ARTS rediscovery of source-level tiling policy.
- No ARTS-RT optimization before DB/EDT/dependency shape is correct.
- Every claim needs checksum parity and repeated evidence when noisy.

## Workstreams

### Matrix Contractions

Targets: `gemm`, `2mm`, `3mm`.

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

1. Dump current slow `gemm`, `2mm`, and `3mm` pipeline stages.
2. Compare against earlier fast and median-fast runs.
3. Identify whether the regression is task count, dependency window, local loop
   shape, DB layout, or runtime variance.
4. Implement the smallest SDE/CODIR shape fix with focused tests.

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
