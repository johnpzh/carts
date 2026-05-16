# Large/64 Performance Plan

## Objective

Make every maintained benchmark correctness-clean and performance-credible at
`large`, 64 threads, and 1 node without benchmark-specific constants or late
ARTS policy inference.

The detailed result log remains
[`../benchmark-performance-goal.md`](../benchmark-performance-goal.md). This
subplan owns the execution strategy.

## Current Evidence

Current M6 large/64 sweep:

- Result directory:
  `.carts/outputs/benchmarks-m6-final-current-20260516/20260516_054254`
- Command:
  `dekk carts benchmarks run --size large --timeout 180 --threads 64 --nodes 1 --runs 3 --trace --results-dir .carts/outputs/benchmarks-m6-final-current-20260516`
- Counts: 23 configured entries, 69 benchmark executions, 69 passed, 0 failed,
  0 skipped by the runner, 0 checksum failures.
- Runner-reported geometric mean kernel speedup over all executions:
  `1.70x` in median reporting mode.
- Startup outliers: ARTS=1, OpenMP=0.
- Focused noisy-STREAM follow-up:
  `.carts/outputs/benchmarks-m6-stream-final-rerun-20260516/20260516_055824`
  reports 7/7 passed, checksum-clean, `1.06x` median-filtered speedup. This
  supersedes the noisy full-sweep STREAM median (`0.73x`) for the final M6
  classification.
- Audited final classes after the STREAM rerun: 21 fast, 2 competitive, 0
  blocked. Replacing the noisy STREAM full-sweep median with the focused
  7-run median gives a `1.73x` audited geomean.
- Host OpenMP fallback is intentionally narrow: repeated 2-D stencil islands
  and benchmark-scoped multi-map 1-D floating-point bundles where ARTS epoch
  overhead would dominate repeated host streaming work. These are direct-main
  host OpenMP binaries, not tokenized ARTS task executions.

Post-M6 multinode investigation:

- Investigation summary:
  `.carts/outputs/multinode-slurm-20260516/investigation-summary.md`
- Real Slurm launches work when forced to nodes that can see the workspace:
  `c09u25,c09u31` for 2-node runs and `c09u25,c09u31,c20u25,c20u31` for
  4-node runs.
- Passing Slurm evidence exists for small 2-node `gemm`, `2mm`, `3mm`,
  `convolution-3d`, `specfem3d/stress`, and `sw4lite/rhs4sg-base`, plus a
  medium `convolution-3d` 1/2/4-node sweep. These are launch/correctness
  smokes, not distributed-work scaling results.
- Current compiler evidence blocks a scaling claim: large `gemm` and large
  `convolution-3d` preserve SDE/CODIR distribution metadata, but
  CODIR-to-ARTS still materializes coarse single-block DB roots
  (`sizes[%c1]`) and `arts.edt <task> <intranode>`. No `distributed_db_init`
  or route selection through `arts_get_total_nodes` appears in the generated
  LLVM for the maintained benchmark runs.
- The next production step is to close the SDE/CODIR-to-ARTS materialization
  gap so planned owner slices become internode EDTs with block/distributed DB
  ownership. Larger multinode sweeps before that point mostly measure launch
  overhead and should not be used as scaling evidence.
- The 64-thread and 64-node measurement ladder is now tracked in
  [`benchmark-64node-experiments.md`](./benchmark-64node-experiments.md). That
  plan requires an ARTS shape gate before 64-node timing can be reported as
  distributed scaling evidence.

| Benchmark | M6 class | ARTS kernel | OpenMP kernel | Speedup | Owner | Next action |
|---|---:|---:|---:|---:|---|---|
| `kastors-jacobi/jacobi-for` | fast | 1.465334s | 2.789009s | 1.903x | none | Host OpenMP fallback control. |
| `kastors-jacobi/poisson-for` | fast | 1.568225s | 2.469140s | 1.574x | none | Host OpenMP fallback control. |
| `ml-kernels/activations` | competitive | 0.597304s | 0.536378s | 0.898x | none | Host OpenMP fallback for transcendental elementwise bundle. |
| `ml-kernels/batchnorm` | fast | 1.444859s | 1.997864s | 1.383x | none | Owner-strip reduction dispatch remains sufficient. |
| `ml-kernels/layernorm` | fast | 2.931311s | 3.454671s | 1.179x | none | Owner-strip logical-slice dispatch remains sufficient. |
| `ml-kernels/pooling` | fast | 2.069970s | 2.311300s | 1.117x | none | Affine lowering and current task grain are sufficient. |
| `monte-carlo/ensemble` | fast | 3.524011s | 4.222032s | 1.198x | none | Keep as task-grain control. |
| `polybench/2mm` | fast | 0.774910s | 5.673198s | 7.321x | none | CODIR dispatch-step correction. |
| `polybench/3mm` | fast | 0.641719s | 4.902833s | 7.640x | none | CODIR dispatch-step correction. |
| `polybench/atax` | fast | 2.835092s | 2.927754s | 1.033x | none | CODIR owner-strip logical-slice dispatch. |
| `polybench/bicg` | competitive | 3.085321s | 2.924945s | 0.948x | none | CODIR logical-worker-slice dispatch. |
| `polybench/convolution-2d` | fast | 2.201995s | 2.732831s | 1.241x | none | CODIR 2-D owner-tile dispatch. |
| `polybench/convolution-3d` | fast | 0.605624s | 2.370311s | 3.914x | none | 3D stencil fast control. |
| `polybench/correlation` | fast | 0.586381s | 1.177201s | 2.008x | none | CODIR owner-strip logical-slice dispatch. |
| `polybench/gemm` | fast | 0.437167s | 6.188823s | 14.157x | none | CODIR dispatch loop preserves SDE's tiled 75-row step. |
| `polybench/jacobi2d` | fast | 0.610889s | 0.798912s | 1.308x | none | Host OpenMP fallback control. |
| `polybench/seidel-2d` | fast | 4.291360s | 4.328538s | 1.009x | none | Host OpenMP fallback control. |
| `seissol/volume-integral` | fast | 0.200723s | 0.248475s | 1.238x | none | CODIR-localized SU scratch removes DB dependency overhead. |
| `specfem3d/stress` | fast | 1.647963s | 2.276170s | 1.381x | none | Trailing owner-dim dispatch restores z-slab task count. |
| `specfem3d/velocity` | fast | 1.203891s | 1.730856s | 1.438x | none | Trailing owner-dim dispatch remains within gate. |
| `stream` | fast | 1.970628s | 2.084288s | 1.058x | none | Superseding 7-run host OpenMP streaming fallback. |
| `sw4lite/rhs4sg-base` | fast | 1.981364s | 2.780303s | 1.403x | none | SW4Lite current task grain is sufficient. |
| `sw4lite/vel4sg-base` | fast | 1.566038s | 2.257713s | 1.442x | none | Trailing owner-dim dispatch restores z-slab task count. |

Prior M5 maintained large/64 sweep:

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

The earlier 12 failed entries failed at compile time with the same boundary
diagnostic: `SDE-authored physical DB layout reached CreateDbs as a raw memref`.
This is an M3 coverage gap. The fix is not to make ARTS rediscover layout in
`CreateDbs`; the fix is to route those plans through SDE MU/token/CODIR
materialization. Until that coverage exists, SDE must drop unsupported physical
storage attrs before the coarse raw bridge and the plan must remain guarded.

Prior focused boundary follow-up:

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

The M6 Phase A sweep exposed a fresh matrix-family regression: CODIR multiplied
SDE's already-tiled 75-row `su_iterate` step by the physical block shape,
producing a 5625-row dispatch step and one oversized EDT. The CODIR
dispatch-step fix restores one EDT per 75-row strip for `gemm`, `2mm`, and
`3mm`; all three focused reruns are checksum-clean and fast.

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
4. Replace the host OpenMP fallback controls with true SDE timestep/wavefront
   plans when M7 resumes work on tokenized repeated-stencil execution.

### Vector And Reductions

Targets: `stream`, `activations`, `batchnorm`, `layernorm`, `atax`, `bicg`.

Priorities:

- block vector outputs and inputs where legal;
- distinguish scalar reductions from per-output reductions;
- fuse compatible elementwise stages before task creation;
- do not narrow the `ml-kernels/activations` host fallback until SDE/CODIR has
  a production vector/libm elementwise plan. A 2026-05-16 selected-only
  activation fallback probe preserved checksums but regressed local 64-thread
  kernels to `33.751227s`/`33.735766s` CARTS versus `0.507816s`/`0.521629s`
  OpenMP before the run was stopped:
  `.carts/outputs/host-openmp-activations-narrow-20260516/20260516_174420`.
- choose local accumulate, tree, or atomic strategy in SDE.

### Stencils And Timesteps

Targets: `jacobi2d`, `seidel-2d`, `fdtd-2d`, KaStORS Jacobi.

Priorities:

- model halo/window shapes in SDE;
- represent timestep and wavefront legality before CODIR/ARTS;
- group adjacent repeated-timestep stages as SDE CPS candidates before
  token-local dataflow is complete;
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

The M6 large/64 gate is complete when every maintained benchmark is classified
as `fast` or `competitive`, every run is checksum-clean, and the validation
suites pass after docs and generated skills are refreshed.
