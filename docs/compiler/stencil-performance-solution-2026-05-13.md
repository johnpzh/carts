# Stencil Performance Solution - 2026-05-13

> **Terminology note (2026-05-15):** This document predates the four-layer rename. Read references to "Core" as the current `arts` dialect (source tree still under `lib/carts/dialect/arts/`) and "RT" as `arts-rt` (source tree under `lib/carts/dialect/arts-rt/`). Ownership claims should be interpreted against the four-layer split documented in [`master-plan.md`](./master-plan.md).

This note records the proposed production path for making the stencil-like
benchmark family performance-credible at `large`, 64 threads, and 1 node after
the correctness-clean sweep in
`.carts/outputs/benchmarks-large-64-after-heap-depv/20260513_174047`.

The fixes below keep the dialect boundary explicit:

- SDE owns stencil recognition, legality proofs, owner dimensions, halo/window
  contracts, task grain, barrier/timestep intent, and physical layout policy.
- CODIR/ARTS own codelet isolation plus ARTS DB/EDT/dependency/epoch
  materialization and may refine the SDE-authored plan without inventing
  OpenMP semantics.
- ARTS-RT/runtime owns low-level launch and dependency descriptor overhead only
  after the SDE/CODIR/ARTS task shape is correct.

## Evidence

The maintained large sweep is correctness-clean, but the slow stencil-like
cases are dominated by task and dependency shape rather than missing
correctness:

| Benchmark | CARTS kernel s | OpenMP kernel s | Speedup | Runtime EDT creates | Runtime DB creates |
|---|---:|---:|---:|---:|---:|
| `polybench/jacobi2d` | 9.401501 | 0.851211 | 0.09x | 129129 | 257 |
| `polybench/convolution-2d` | 12.003936 | 2.956942 | 0.25x | 14081 | 129 |
| `kastors-jacobi/jacobi-for` | 9.613102 | 2.793346 | 0.29x | 12801 | 194 |
| `sw4lite/rhs4sg-base` | 0.937903 | 2.765063 | 2.95x | 641 | 67 |
| `sw4lite/vel4sg-base` | 0.377839 | 2.334956 | 6.18x | 641 | 199 |
| `specfem3d/velocity` | 0.402153 | 1.704339 | 4.24x | 641 | 199 |

A later focused run with corrected overhead-counter profiles is recorded in
`docs/compiler/stencil-counter-analysis-2026-05-13.md`. In that run,
`convolution-2d` and `jacobi-for` are faster than their OpenMP baselines, while
`jacobi2d` remains slow at 0.43x with 129129 EDT creates and 385382 DB read
acquisitions. The plan below still targets stencil task/dependency shape first
because the high EDT/read-acquire counts remain the durable signal.

The fast controls all have one regular ready-local launch site, roughly 640
worker EDTs in the measured run, no standard `arts_add_dependence` launch path,
and write dependencies lowered as `EW`. The slow cases launch 20x to 200x more
EDTs; `jacobi2d` additionally mixes ready-local worker launches with standard
`arts_edt_create_with_epoch` plus `arts_add_dependence`.

This is the key shape problem:

- `jacobi2d`: repeated timestep pipeline creates about `1000 * 128` worker EDTs
  plus continuation/orchestration EDTs. Whole-read standard dependencies are
  still visible in the generated LLVM IR.
- `convolution-2d`: ready-local only, but `NREPS=110` creates `110 * 128`
  worker EDTs. Output dependencies are `RW` instead of `EW`, and the task shape
  is row-strip only.
- `jacobi-for`: ready-local only, but `NREPS=10` and two stencil/copy phases
  still create about 12800 worker EDTs. It has dynamic slice clues, but not the
  fast control shape.
- Fast SW4/SPECFEM controls do enough work per task, retain a regular launch
  structure, and use `EW` write dependencies.

## Root Causes

1. **SDE analyzes inner spatial loops but does not promote them into owner
   tasks for normal OpenMP source.**

   `StructuredOpAnalysis` can see nested stencil dimensions, but
   `DistributionPlanning` chooses owner loop dims only from the SDE
   `su_iterate` rank. For source shaped as `#pragma omp parallel for` over `i`
   with inner `j/k` `scf.for` loops, the physical plan remains row/slab-strip
   oriented even when the stencil is semantically 2-D or 3-D.

2. **Core can widen or drop stencil windows before lowering.**

   direct materialization can compute a coarse read-only bridge before
   combining the full loop contract, and the coarse bridge can skip contract
   projection. Rank-mismatch paths can preserve parent full ranges for read-only
   acquires instead of worker-local stencil windows.

3. **Write mode is too conservative in slow cases.**

   Slow stencils show `RW` output dependencies where the owner tile writes an
   output without reading its previous contents. Fast controls use `EW`. The
   compiler should only use `RW/inout` for true in-place/self-read or
   owner-mismatched write patterns.

4. **Repeated phases launch fresh workers too often.**

   The current grouped lowering only handles a narrow repeated-wave contract.
   `jacobi2d` still forms many per-timestep epochs/continuations, and repeated
   benchmark reps in `convolution-2d` and `jacobi-for` pay CARTS launch
   overhead much more heavily than OpenMP.

5. **RT launch overhead is now stable but still costly at high task counts.**

   Ready-local descriptor scratch is heap allocated per task and freed after
   the runtime copies it. That preserves the important contract that user
   dynamic arrays are DB dependencies, not EDT params, but it is still overhead
   when task counts exceed ten thousand.

## Production Plan

### Phase 0: Add Characterization Before Rewriting

Add lit tests and counters that make the current losses visible:

- `task_loop_stencil_read_keeps_worker_local_dep_window.mlir`
- `task_loop_nd_stencil_read_materializes_nd_block_window.mlir`
- `task_loop_stencil_inout_write_not_read_mode.mlir`
- `db_acquire_offsets_only_segments_multi_entry.mlir`
- `task_loop_repeated_stencil_group_single_epoch.mlir`

Add runtime counters for:

- ready-local creates and total ready-local dep slots,
- ready-local dep scratch pool hits/misses,
- DB acquire local hits, frontier defers, remote requests, and sliced acquires,
- time spent between EDT create, acquire completion, and EDT run.

Then collect repeated measurements:

```bash
dekk carts benchmarks run polybench/jacobi2d polybench/convolution-2d \
  kastors-jacobi/jacobi-for sw4lite/vel4sg-base \
  --size large --timeout 120 --threads 64 --nodes 1 --trace --perf --runs 3 \
  --results-dir .carts/outputs/depv-cost-stencil-vs-control
```

### Phase 1: Fix Core Dependency Shape First

These are the lowest-risk production fixes because they preserve existing SDE
contracts instead of changing stencil semantics.

1. **Resolve contracts before coarse-read bridging in direct materialization.**

   Move loop/acquire contract resolution ahead of `forceCoarseReadOnlyDep`.
   Explicit stencil, block-halo, owner-dim, or `narrowable_dep` contracts must
   keep worker-local dependency windows unless the owner-mismatch proof forces a
   full read.

2. **Scan both the parent acquire pointer and the EDT block argument.**

   `inferLoopLocalMode` and loop-IV-dependent access detection should share a
   helper that follows both aliases. This avoids classifying a stencil input or
   output from stale parent-acquire uses only.

3. **Tighten write modes to `EW/out` when legal.**

   For out-of-place stencil owner tiles, input roots should lower to `in/RO`,
   output roots to `out/EW`, and only in-place/self-read or owner-mismatched
   writes should remain `inout/RW`.

4. **Add N-D task acquire window materialization.**

   Replace the rank > 1 bail-out in task acquire slice planning with an N-D
   block-window path. Preserve owner-local dimensions from the worker slice and
   complete non-owner dimensions from the parent range. Use existing DB slice
   utilities instead of ad hoc range rewriting.

5. **Fix partition segment accounting.**

   `DbAcquireOp` multi-entry detection should use the maximum non-empty segment
   count across indices, offsets, and sizes, not only
   `partition_indices_segments`. Stencil reads with offsets/sizes-only segment
   metadata must not collapse to one logical entry.

6. **Narrow tiny-table coarse bridging.**

   Keep true coefficient tables coarse, but do not treat every small stencil
   allocation as a table. Data DBs with owner/halo contracts should still get
   worker-local windows.

Expected impact:

- `convolution-2d` and `jacobi-for` should stop paying unnecessary `RW` and
  whole-window costs.
- `jacobi2d` should lose the standard whole-read dependency path where stencil
  contracts are complete.
- These changes are testable with focused Core lit tests before running
  benchmarks.

### Phase 2: Promote SDE Owner Tiles For Normal OpenMP Stencils

This is the high-payoff semantic transformation.

For eligible out-of-place stencils shaped as:

```c
#pragma omp parallel for
for (i = ...)
  for (j = ...)
    out[i][j] = stencil(in, i, j);
```

SDE should promote the inner spatial loop dimensions into the owner-task plan:

- produce owner dimensions `[0, 1]` for 2-D and `[0, 1, 2]` for 3-D,
- choose a tile shape from launch cost, halo surface/volume, cache footprint,
  element size, and target ready-task count,
- stamp low/high halo per owner dimension, not just one symmetric max,
- record per-root roles: read-only input, write-only output, in-place
  self-read, coefficient table, or scalar,
- keep component/batch dimensions local unless the analysis proves they are
  owner dimensions,
- reject in-place Seidel-style self-read unless a wavefront plan proves
  legality.

CODIR/ARTS should then materialize exactly that plan through the
`sde-to-codir` and `codir-to-arts` boundary, using the coarse `CreateDbs`
bridge only for residual raw memref compatibility; ARTS should not rediscover
stencil geometry.

Expected impact:

- `convolution-2d` should move from row-strip tasks toward 2-D tiles with fewer
  dependency hazards and better cache locality.
- `jacobi2d` should gain tile contracts that make later timestep grouping
  possible.
- 3-D stencils should avoid regressing the already-fast SW4/SPECFEM controls by
  preserving their existing component/slab-friendly plans when those are better.

### Phase 3: Add Repeated Stencil Phase Planning

Once owner tiles and dependency windows are correct, SDE should build a phase
graph for repeated stencil loops:

- recognize A/B alternating-buffer Jacobi phases across explicit barriers,
  adjacent scheduling units, and guarded second phases,
- stamp `jacobi_alternating_buffers`, phase count, buffer parity, and
  compatible owner tile shape,
- group compatible phases into one Core lowering unit when no host work or
  reduction boundary intervenes,
- for safe cases, use temporal batching or `k_step` planning with halo widened
  by `k` and local scratch where required,
- preserve barrier semantics for in-place wavefront or mismatched tile plans.

Core should generalize the existing repeated-wave grouped lowering into a
stencil phase group lowering path. The goal is to stop launching a fresh full
worker set for every tiny phase when SDE has proven that a grouped phase plan is
legal.

Expected impact:

- `jacobi2d` is the primary target. Its current 129129 EDT creates are the
  wrong shape for an OpenMP-comparable timestep kernel.
- `jacobi-for` should benefit for copy + stencil phase pairs if the copy phase
  can be fused, coarsened, or represented as an owner-local update.

### Phase 4: Optimize RT/Runtime Launch Mechanics

Only after the Core/SDE task shape is correct:

1. Add a TLS ready-local dependency descriptor scratch pool, then lower RT
   ready-local scratch acquire/release to that pool instead of per-task
   `arts_calloc`/`arts_free`.
2. Add a local ready-create fast path that returns the created EDT pointer and
   avoids the immediate route-table lookup after insertion.
3. Coalesce duplicate read-only whole-DB acquire slots only when every depv slot
   remains addressable and release/refcount semantics stay explicit.

These changes should preserve the DB dependency contract: dynamic arrays remain
dependencies, not EDT params. They are performance cleanup for high task counts,
not a replacement for correct stencil planning.

## First Implementation Slice

Start with Phase 1 because it is narrow, testable, and protects future SDE
owner-tile work:

1. Add the Core lit tests for dependency window preservation, N-D task windows,
   partition segment counts, and write-mode tightening.
2. Move contract resolution earlier in direct materialization and gate
   `forceCoarseReadOnlyDep` on the resolved contract.
3. Tighten task-local modes by scanning both acquire pointer and EDT block
   argument uses.
4. Fix `DbAcquireOp` segment counting and verifier checks.
5. Run:

```bash
dekk carts lit lib/carts/dialect/arts/test/task_loop_scalar_control_dep_keeps_stencil_parallelism.mlir
dekk carts lit lib/carts/dialect/arts/test/task_loop_read_only_owner_mismatch_keeps_parallelism.mlir
dekk carts lit lib/carts/dialect/arts/test/task_loop_aligned_bounds_unconditional.mlir
dekk carts lit lib/carts/dialect/arts/test/task_loop_db_alignment_normalizes_nonunit_step.mlir
dekk carts lit tests/e2e/parallel_for_stencil.c
dekk carts lit tests/e2e/stencil.c
dekk carts lit tests/e2e/jacobi_for.c
dekk carts test
```

Then rerun the slow stencil subset with `--runs 3` before attempting owner-tile
promotion.

## Success Criteria

- `polybench/convolution-2d`, `polybench/jacobi2d`, and
  `kastors-jacobi/jacobi-for` remain checksum-clean.
- Slow stencil cases no longer use unnecessary `RW` output dependencies where
  `EW/out` is legal.
- Stencil read dependencies keep worker-local halo windows instead of falling
  back to parent whole ranges.
- Runtime EDT create counts move toward the fast-control shape where semantics
  allow grouping or coarsening.
- Any remaining slowdown is classified with counters as launch overhead,
  memory bandwidth, vectorization, or required dependency ordering.
