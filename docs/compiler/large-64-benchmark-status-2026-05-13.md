# CARTS Large-64 Benchmark Status - 2026-05-13

> **Status (2026-05-15):** Numbers below are superseded by the 2026-05-14 large/64 evidence in [`benchmark-performance-goal.md`](./benchmark-performance-goal.md) (gemm 0.371x, 2mm 0.188x, 3mm 0.190x). This document is retained as historical evidence of the 2026-05-13 sweep.

This note records the state after the ready-local dependency-buffer fix and
the large-size, 64-thread, one-node benchmark sweep.

## Implementation Summary

- ARTS ready-local EDT creation now copies initial `arts_edt_dep_t` metadata
  before the EDT is published through the route table. This avoids exposing a
  partially initialized EDT to scheduler/frontier threads.
- RT lowering no longer emits dynamic stack `alloca` arrays for ready-local
  dependency descriptors. It allocates local descriptor scratch with
  `arts_calloc`, calls `arts_edt_create_ready_local_with_epoch`, and frees the
  scratch after the runtime has copied the descriptors.
- `PtrSize` in the runtime ABI model now uses pointer size in bits, not bytes.
  This fixes generated declarations such as `arts_calloc(i64, i64)` instead of
  the broken `arts_calloc(i8, i8)`.
- The `carts-debug` skill documents LLDB discovery, empty-core checks, and
  signal-only `strace` diagnostics for stack-guard faults.

The key compiler contract remains: user dynamic arrays should not be captured
as EDT parameters. They should flow through DB dependencies. The heap-backed
descriptor buffer is only RT-local launch metadata, not user data.

## Verification Commands

```bash
dekk carts build --arts --debug 1
dekk carts build
dekk carts skills generate
dekk carts test
dekk carts benchmarks run ml-kernels/batchnorm --size large --timeout 120 \
  --threads 64 --nodes 1 --trace \
  --results-dir .carts/outputs/benchmarks-batchnorm-64-heap-depv-sizeabi
dekk carts benchmarks run --size large --timeout 120 --threads 64 --nodes 1 \
  --trace --results-dir .carts/outputs/benchmarks-large-64-after-heap-depv
```

`dekk carts test` passed: 124 pass, 1 expected failure.

## Correctness Status

The maintained large benchmark sweep is now correctness-clean at large size,
64 threads, and one node:

- Results: `.carts/outputs/benchmarks-large-64-after-heap-depv/20260513_174047`
- Pass rate: 23/23
- Geometric mean kernel speedup: 0.76x vs OpenMP

The focused batchnorm rerun after the ABI fix also passed:

- Results: `.carts/outputs/benchmarks-batchnorm-64-heap-depv-sizeabi/20260513_173739`
- Kernel: CARTS 1.261707s, OpenMP 1.995867s, 1.58x speedup

Batchnorm was slower in the full sweep than in the focused rerun, so treat its
performance as noisy until repeated median runs are collected.

## Large-64 Sweep Results

| Benchmark | CARTS kernel s | OpenMP kernel s | Speedup | Correct |
|---|---:|---:|---:|---|
| kastors-jacobi/jacobi-for | 9.613102 | 2.793346 | 0.29x | yes |
| kastors-jacobi/poisson-for | 2.050658 | 2.562649 | 1.25x | yes |
| ml-kernels/activations | 7.314199 | 0.590145 | 0.08x | yes |
| ml-kernels/batchnorm | 4.672305 | 2.050310 | 0.44x | yes |
| ml-kernels/layernorm | 16.266258 | 3.323092 | 0.20x | yes |
| ml-kernels/pooling | 1.975321 | 2.339390 | 1.18x | yes |
| monte-carlo/ensemble | 3.392211 | 4.344037 | 1.28x | yes |
| polybench/2mm | 1.496333 | 5.556951 | 3.71x | yes |
| polybench/3mm | 1.192571 | 4.923211 | 4.13x | yes |
| polybench/atax | 2.730184 | 3.207256 | 1.17x | yes |
| polybench/bicg | 22.973762 | 2.973333 | 0.13x | yes |
| polybench/convolution-2d | 12.003936 | 2.956942 | 0.25x | yes |
| polybench/convolution-3d | 2.898897 | 2.385356 | 0.82x | yes |
| polybench/correlation | 0.525734 | 1.175389 | 2.24x | yes |
| polybench/gemm | 4.421876 | 6.287265 | 1.42x | yes |
| polybench/jacobi2d | 9.401501 | 0.851211 | 0.09x | yes |
| polybench/seidel-2d | 6.255567 | 4.513833 | 0.72x | yes |
| seissol/volume-integral | 2.556975 | 0.290078 | 0.11x | yes |
| specfem3d/stress | 3.129007 | 2.304185 | 0.74x | yes |
| specfem3d/velocity | 0.402153 | 1.704339 | 4.24x | yes |
| stream | 3.284995 | 2.144133 | 0.65x | yes |
| sw4lite/rhs4sg-base | 0.937903 | 2.765063 | 2.95x | yes |
| sw4lite/vel4sg-base | 0.377839 | 2.334956 | 6.18x | yes |

## Priority Fix Ideas

1. **ML elementwise kernels: activations, layernorm, batchnorm**

   These are correctness-clean but slow. They likely suffer from excessive EDT
   launch/dependency overhead and poor vector-shaped work aggregation. Next
   steps:

   - Audit SDE distribution plans for these kernels and confirm that dynamic
     arrays are DB dependencies, not params.
   - Prefer coarser launch granularity for pure elementwise/reduction bodies
     where the per-task work is too small.
   - Check whether vectorization hints survive through RT lowering for hot
     inner loops.
   - Run repeated focused medians for batchnorm to separate real performance
     from run-to-run launch noise.

2. **Stencil and wavefront cases: jacobi-for, jacobi2d, convolution-2d**

   These remain the largest stencil outliers. Next steps:

   - Inspect block layout choices and halo dependency shape. Whole-DB bridging
     or over-wide dependencies will serialize otherwise parallel work.
   - Verify that owner-slice and boundary plans stay in SDE/arts (lib/arts/dialect/core/) and are not
     recovered late in RT.
   - Compare task counts and DB acquire counts against OpenMP loop partitioning.
   - Tighten the cost model so small halo work does not create too many EDTs.

3. **BiCG and SeisSol volume-integral**

   These are correctness-clean but far behind OpenMP. Next steps:

   - Determine whether the lowering is preserving the intended reduction
     structure or forcing serialized DB updates.
   - Check if temporary reductions can use local accumulation or tree reduction
     instead of high-contention DB writes.
   - Inspect whether SDE pattern/effect facts identify the true read/write
     footprint.

4. **STREAM**

   STREAM is close enough that startup/cleanup and memory placement matter.
   Next steps:

   - Keep using release ARTS for performance claims.
   - Compare interleaved DB allocation against first-touch and owner-local
     allocation.
   - Measure copy/scale/add/triad separately across repeated runs.

5. **Utility and pass structure**

   The current direction is right, but the compiler still has oversized files
   and utility leakage:

   - Keep type predicates, constant checks, DB alias tracing, range checks, and
     stencil/layout helpers in utility modules, not inside pass files.
   - Keep direct materialization helpers split by orchestration, body cloning,
     reduction planning, and loop-bound utilities.
   - Continue splitting `CreateDbs.cpp` by layout-plan resolution, acquire
     grouping, host/global view materialization, and DB alias helpers.

## Debugging Notes

LLDB was not available in this environment. The useful alternate diagnostic was:

```bash
strace -ff -tt -e trace=signal -o .carts/outputs/<topic>/trace \
  env artsConfig=/abs/path/to/arts.cfg /abs/path/to/executable
```

The batchnorm crash showed `SEGV_ACCERR` near a worker stack guard. Generated
LLVM then showed dynamic dependency-descriptor `alloca` operations inside the
launch loop. Moving only the RT-local descriptor scratch to heap fixed the
stack exhaustion while keeping user dynamic arrays in the dependency layer.
