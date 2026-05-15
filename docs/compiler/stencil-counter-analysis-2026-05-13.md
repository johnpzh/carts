# Stencil Counter Analysis - 2026-05-13

> **Terminology note (2026-05-15):** This document predates the four-layer rename. Read references to "Core" as the current `arts` dialect (source tree still under `lib/arts/dialect/core/`) and "RT" as `arts-rt` (source tree under `lib/arts/dialect/rt/`). Ownership claims should be interpreted against the four-layer split documented in [`master-plan.md`](./master-plan.md).

This note records the current measurement state for the stencil-like CARTS
benchmarks after fixing ARTS counter-profile plumbing. The focus is `large`,
64 threads, and 1 local node.

## Instrumentation Fixes

The first blocker was measurement quality, not stencil lowering itself.

- The top-level `Makefile` now passes the selected `COUNTER_CONFIG_PATH` into
  ARTS CMake as an absolute path and includes the absolute path plus file hash
  in the ARTS configuration cache key.
- The benchmark counter profiles now use the ARTS counter enum names that
  generate real `ENABLE_*` macros, for example `NUM_EDT_CREATE` instead of the
  old `numEdtsCreated` alias.
- The benchmark report parser accepts both current ARTS counter names and the
  old lower-camel aliases, so old fixtures and new runs remain readable.
- Benchmark-runner profile rebuilds now prefer `dekk carts build --arts ...`
  instead of hard-coded bare `carts build ...`; this fixes the earlier
  `PermissionError: 'carts'` failure when using `--profile`.
- `dekk carts build --arts --profile <profile>` now marks the Make layer as a
  counter/metrics build, matching the profile-driven ARTS CMake state.

Verification:

```bash
dekk carts build --arts --counters 3 \
  --profile external/carts-benchmarks/configs/profiles/profile-overhead.cfg
```

The generated
`external/arts/build/libs/include/internal/arts/counter/Preamble.h` now uses
`profile-overhead.cfg` as its source and enables the required counters:

- `ENABLE_TIME_EDT_EXEC=1`
- `ENABLE_NUM_EDT_CREATE=1`
- `ENABLE_NUM_EDT_ACQUIRE=1`
- `ENABLE_NUM_EDT_FINISH=1`
- `ENABLE_NUM_DB_CREATE=1`
- `ENABLE_NUM_DB_ACQUIRE_READ=1`
- `ENABLE_NUM_DB_ACQUIRE_WRITE=1`
- `ENABLE_TIME_INIT=1`
- `ENABLE_TIME_TOTAL=1`

The benchmark `--profile` path was smoke-tested with:

```bash
dekk carts benchmarks run sw4lite/vel4sg-base \
  --size large --timeout 180 --threads 64 --nodes 1 --trace --runs 1 \
  --profile external/carts-benchmarks/configs/profiles/profile-overhead.cfg \
  --results-dir .carts/outputs/profile-rebuild-smoke-20260513
```

That run passed and rebuilt ARTS through `dekk carts build`.

## Latest Focused Run

Run directory:
`.carts/outputs/stencil-overhead-counters-20260513-fixed/20260513_184105`

Command:

```bash
dekk carts benchmarks run polybench/jacobi2d polybench/convolution-2d \
  kastors-jacobi/jacobi-for sw4lite/vel4sg-base \
  --size large --timeout 180 --threads 64 --nodes 1 --trace --runs 1 \
  --results-dir .carts/outputs/stencil-overhead-counters-20260513-fixed
```

All four benchmarks passed checksum verification.

| Benchmark | CARTS kernel s | OpenMP kernel s | Speedup | EDT creates | EDT acquires | DB creates | DB read acquires | DB write acquires |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `polybench/jacobi2d` | 1.944227 | 0.835762 | 0.43x | 129129 | 129129 | 257 | 385382 | 0 |
| `polybench/convolution-2d` | 2.088335 | 2.765178 | 1.32x | 14081 | 1 | 129 | 14080 | 0 |
| `kastors-jacobi/jacobi-for` | 2.148456 | 2.710518 | 1.26x | 12801 | 1 | 194 | 50800 | 0 |
| `sw4lite/vel4sg-base` | 0.411037 | 2.353797 | 5.73x | 641 | 1 | 199 | 4480 | 1920 |

Additional counters:

| Benchmark | `TIME_EDT_EXEC` ms | `TIME_TOTAL` ms | Memory footprint bytes | Remote sent | Remote received |
|---|---:|---:|---:|---:|---:|
| `polybench/jacobi2d` | 79811.686819 | 2213.241775 | 3808692329 | 0 | 0 |
| `polybench/convolution-2d` | 116132.390870 | 3471.889516 | 6220782703 | 0 | 0 |
| `kastors-jacobi/jacobi-for` | 116204.716756 | 3299.848977 | 5788250825 | 0 | 0 |
| `sw4lite/vel4sg-base` | 23159.702478 | 1514.694943 | 5106680266 | 0 | 0 |

## Interpretation

The focused run is materially better than the earlier large sweep for
`convolution-2d` and `jacobi-for`; both are now faster than OpenMP in this
single run. They are still not in the same task-shape class as the fast control:
they launch roughly 20x more worker EDTs than `vel4sg-base`, and `jacobi-for`
does about 8x more DB read acquisitions. The production goal should still
reduce launch/read-dependency volume so that this result is stable across
repeated runs and less sensitive to runtime overhead.

`jacobi2d` remains the primary failing stencil case. It creates 129129 EDTs,
129129 acquire-ready events, and 385382 DB read acquisitions. That is the wrong
shape for an OpenMP-comparable timestep kernel. The first real optimization
target is not vector math or remote placement; it is timestep/task grouping and
dependency-window planning.

There is no remote traffic in this local-node run. Multi-node performance should
not be optimized until the local task/dependency shape is fixed.

The runtime `NUM_DB_ACQUIRE_WRITE` counter is zero for the three slow stencil
cases in this run, while the fast control records write acquisitions. That means
the current measured local bottleneck is not runtime write-acquire volume.
Static IR checks for `RW` versus `EW` are still needed, because older lowered
IR showed conservative write modes and because mode mistakes can serialize or
widen dependencies before runtime counters make the cost obvious.

The dynamic-array contract remains unchanged: user dynamic arrays must be DB
dependencies, not EDT params. Scalar bounds, flags, worker ids, and tile
metadata can be captured as params; array payloads and dynamic array bases
belong in dependency slots.

## Production Transformations

1. **Keep instrumentation reliable.**

   The Makefile/profile/report/runner fixes above should stay in place before
   making performance claims. Any future counter profile must use ARTS enum
   names or an explicit parser alias table that maps to enum names before
   generating `Preamble.h`.

2. **Add Core regression tests before changing lowering.**

   Add focused lit tests for:

   - stencil read dependencies keeping worker-local halo windows,
   - N-D task acquire windows,
   - offsets/sizes-only partition segment counts,
   - `inout` parent acquires tightening to task-local `in` or `out` where
     legal,
   - repeated stencil groups lowering through one epoch/dispatch plan.

3. **Fix Core dependency-window preservation first.**

   In direct materialization, resolve loop/acquire contracts before coarse-read bridging.
   Explicit stencil, block-halo, owner-dim, or `narrowable_dep` contracts should
   keep worker-local dependency windows unless an owner-mismatch proof forces a
   full read. `inferLoopLocalMode` and loop-IV access checks should scan both
   the parent acquire pointer and the EDT block argument.

4. **Materialize N-D windows instead of falling back to parent ranges.**

   The task acquire slice path should support rank > 1 using the existing DB
   slice utilities. Fill owner dimensions from the worker slice and non-owner
   dimensions from the parent range; do not treat rank mismatch as a reason to
   widen read-only dependencies to whole parent ranges.

5. **Promote normal OpenMP stencils to SDE owner tiles.**

   For out-of-place stencils shaped as `parallel for i` with inner `j/k` loops,
   SDE should promote the inner spatial dimensions into the owner-task plan and
   stamp per-root read/write roles plus low/high halo windows. Core should
   materialize that plan, not rediscover stencil geometry.

6. **Plan repeated stencil phases, especially `jacobi2d`.**

   `jacobi2d` needs a phase graph for alternating buffers and timestep grouping.
   Once owner tiles and dependency windows are correct, SDE should stamp phase
   count, buffer parity, compatible owner-tile shape, and legal `k_step` or
   grouped-phase intent. Core should lower compatible phases through one
   grouped dispatch where barriers/reductions do not force separation.

7. **Optimize RT ready-local mechanics after task shape is fixed.**

   Add a TLS ready-local dependency descriptor scratch pool and a local
   ready-create fast path that avoids immediate route-table lookup after
   insertion. This is cleanup for high task counts; it should not compensate for
   incorrect SDE/CODIR/ARTS planning.

## Next Implementation Slice

The most defensible first code slice is:

1. Add the Core lit tests listed above.
2. Fix `DbAcquireOp` multi-entry segment accounting across indices, offsets,
   and sizes.
3. Move contract resolution earlier in direct materialization.
4. Add the shared use scanner for parent acquire pointer plus EDT block
   argument.
5. Re-run the slow stencil subset with `--runs 3` and the overhead profile.

Success for this slice is not only “faster than OpenMP once.” Success is:

- `polybench/jacobi2d`, `polybench/convolution-2d`, and
  `kastors-jacobi/jacobi-for` remain checksum-clean,
- `jacobi2d` drops standard whole-read/acquire-heavy behavior,
- stencil reads retain worker-local halo windows,
- task counts and DB read acquire counts move toward the fast-control shape
  when semantics allow it,
- any remaining slowdown has a counter-backed explanation.
