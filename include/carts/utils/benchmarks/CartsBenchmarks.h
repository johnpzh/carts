///===----------------------------------------------------------------------===///
/// File: CartsBenchmarks.h
///
/// C benchmark utilities for CARTS benchmarks.
/// Provides standardized checksum reporting and kernel timing.
///
///
/// Usage:
///   #include "carts/utils/benchmarks/CartsBenchmarks.h"
///
///   int main() {
///     CARTS_KERNEL_TIMER_START("kernel");
///     kernel_function(...);
///     CARTS_KERNEL_TIMER_STOP("kernel");
///
///     double checksum = 0.0; for (int i = 0; i < N; i++)
///     checksum += data[i];
///     CARTS_BENCH_CHECKSUM(checksum);
///     return 0;
///   }
///
/// Output format:
///   kernel.name: 0.1234s
///   checksum: 1.234567e+00
///
/// Note: Link with libcartsbenchmarks (-lcartsbenchmarks) for the
/// implementation.
///===----------------------------------------------------------------------===///

#ifndef CARTS_BENCHMARKS_H
#define CARTS_BENCHMARKS_H

#include <stdint.h>
#include <stdio.h>

#define CARTS_BENCHMARK_STRING_ARG(name) ((uintptr_t)(const void *)(name))

/// Single source of truth for the symbol name CARTS inserts to mark a binary
/// as the host-OpenMP fallback. Compile.cpp creates the call; CartsBenchmarks
/// defines it; tools/scripts/compile.py greps the LLVM IR for it.
#define CARTS_BENCHMARKS_HOST_OPENMP_MARKER_NAME                               \
  "carts_benchmarks_mark_host_openmp"

#ifdef __cplusplus
extern "C" {
#endif

///===----------------------------------------------------------------------===///
/// Benchmark Start/Stop (call to set up fair timing)
///===----------------------------------------------------------------------===///

/// Pre-warm OMP thread pool.
void carts_benchmarks_warmup_omp(void);

/// Mark a CARTS-generated binary as a host OpenMP fallback so benchmark setup
/// uses OpenMP warmup even if ARTS symbols are linked.
void carts_benchmarks_mark_host_openmp(void);

/// Benchmark start hook for fair timing comparisons.
/// - OpenMP reference: warms up the OMP thread pool and may emit `init.omp`.
/// - ARTS binary: may emit `init.arts` (process start -> entering mainBody).
void carts_benchmarks_start(void);

/// For OMP: calls extern warmup function to initialize thread pool before
/// timing. For ARTS: no-op (ARTS initializes at startup).
#define CARTS_BENCHMARKS_START() carts_benchmarks_start()

#define CARTS_BENCHMARKS_STOP() ((void)0)

///===----------------------------------------------------------------------===///
/// End-to-End Timing (high-precision wall clock via std::chrono::steady_clock)
///
/// Use for fair ARTS vs OMP comparison. Measures wall clock time including
/// DAG construction (ARTS) and parallel region overhead (OMP).
///
/// Usage:
///   CARTS_E2E_TIMER_START("kernel_name");
///   kernel(...);
///   CARTS_E2E_TIMER_STOP();
///
/// Output format:
///   e2e.kernel_name: 0.123456789s
///===----------------------------------------------------------------------===///

/// Start E2E timer (stores name and start time in library-side state).
void carts_e2e_timer_start(uintptr_t name);

/// Stop E2E timer and print elapsed time.
void carts_e2e_timer_stop(void);

#define CARTS_E2E_TIMER_START(name)                                            \
  carts_e2e_timer_start(CARTS_BENCHMARK_STRING_ARG(name))
#define CARTS_E2E_TIMER_STOP() carts_e2e_timer_stop()

///===----------------------------------------------------------------------===///
/// Phase Timing
///
/// Benchmarks still emit explicit startup / verification / cleanup sections,
/// and the harness parses those prefixes.
///===----------------------------------------------------------------------===///

/// Start a named benchmark phase timer.
void carts_phase_timer_start(uintptr_t phase, uintptr_t name);

/// Stop a named benchmark phase timer and print elapsed time.
void carts_phase_timer_stop(uintptr_t phase);

#define CARTS_STARTUP_TIMER_START(name)                                        \
  carts_phase_timer_start(CARTS_BENCHMARK_STRING_ARG("startup"),               \
                          CARTS_BENCHMARK_STRING_ARG(name))
#define CARTS_STARTUP_TIMER_STOP()                                             \
  carts_phase_timer_stop(CARTS_BENCHMARK_STRING_ARG("startup"))
#define CARTS_VERIFICATION_TIMER_START(name)                                   \
  carts_phase_timer_start(CARTS_BENCHMARK_STRING_ARG("verification"),          \
                          CARTS_BENCHMARK_STRING_ARG(name))
#define CARTS_VERIFICATION_TIMER_STOP()                                        \
  carts_phase_timer_stop(CARTS_BENCHMARK_STRING_ARG("verification"))
#define CARTS_CLEANUP_TIMER_START(name)                                        \
  carts_phase_timer_start(CARTS_BENCHMARK_STRING_ARG("cleanup"),               \
                          CARTS_BENCHMARK_STRING_ARG(name))
#define CARTS_CLEANUP_TIMER_STOP()                                             \
  carts_phase_timer_stop(CARTS_BENCHMARK_STRING_ARG("cleanup"))

///===----------------------------------------------------------------------===///
/// Program-Level Timing
///===----------------------------------------------------------------------===///

/// Get current wall-clock time in seconds.
double carts_bench_get_time(void);

/// Start program-level timer.
void carts_bench_timer_start(void);

/// Return elapsed seconds since carts_bench_timer_start().
double carts_bench_timer_stop(void);

/// Print elapsed program time.
void carts_bench_timer_print(void);

#define CARTS_BENCH_TIMER_START() carts_bench_timer_start()
#define CARTS_BENCH_TIMER_STOP() carts_bench_timer_stop()
#define CARTS_BENCH_TIMER_PRINT() carts_bench_timer_print()

///===----------------------------------------------------------------------===///
/// Kernel Timing (for fine-grained measurement)
///===----------------------------------------------------------------------===///

/// Start timing a kernel (resets accumulator).
void carts_kernel_timer_start(uintptr_t name);

/// Stop kernel timer and print elapsed time.
void carts_kernel_timer_stop(uintptr_t name);

/// Accumulate kernel time (for iterative methods — call each iteration).
void carts_kernel_timer_accum(uintptr_t name);

/// Print accumulated kernel time (call after all iterations).
void carts_kernel_timer_print(uintptr_t name);

#define CARTS_KERNEL_TIMER_START(name)                                         \
  carts_kernel_timer_start(CARTS_BENCHMARK_STRING_ARG(name))
#define CARTS_KERNEL_TIMER_STOP(name)                                          \
  carts_kernel_timer_stop(CARTS_BENCHMARK_STRING_ARG(name))
#define CARTS_KERNEL_TIMER_ACCUM(name)                                         \
  carts_kernel_timer_accum(CARTS_BENCHMARK_STRING_ARG(name))
#define CARTS_KERNEL_TIMER_PRINT(name)                                         \
  carts_kernel_timer_print(CARTS_BENCHMARK_STRING_ARG(name))

///===----------------------------------------------------------------------===///
/// Parallel Region & Task Timing (thread-safe, per-worker)
///
/// These macros measure time spent in parallel regions and tasks separately.
/// Used to analyze whether delayed MLIR optimizations (CSE, DCE) due to DB
/// partitioning affect sequential kernel performance.
///
/// Usage:
///   #pragma omp parallel
///   {
///       CARTS_PARALLEL_TIMER_START("gemm");
///       #pragma omp task
///       {
///           CARTS_TASK_TIMER_START("gemm:kernel");
///           kernel_computation(...);
///           CARTS_TASK_TIMER_STOP("gemm:kernel");
///       }
///       #pragma omp taskwait
///       CARTS_PARALLEL_TIMER_STOP("gemm");
///   }
///
/// Output format:
///   parallel.gemm[worker=0]: 0.001234s
///   task.gemm:kernel[worker=0]: 0.001100s
///
/// Enable with: -DCARTS_ENABLE_TIMING or #define CARTS_ENABLE_TIMING
/// Disabled by default to avoid overhead in production runs.
///===----------------------------------------------------------------------===///

#ifdef CARTS_ENABLE_TIMING

/// Start timing a parallel region.
void carts_parallel_timer_start(uintptr_t name);

/// Stop parallel region timer and print elapsed time with worker ID.
void carts_parallel_timer_stop(uintptr_t name);

/// Start timing a task.
void carts_task_timer_start(uintptr_t name);

/// Stop task timer and print elapsed time with worker ID.
void carts_task_timer_stop(uintptr_t name);

#define CARTS_PARALLEL_TIMER_START(name)                                       \
  carts_parallel_timer_start(CARTS_BENCHMARK_STRING_ARG(name))
#define CARTS_PARALLEL_TIMER_STOP(name)                                        \
  carts_parallel_timer_stop(CARTS_BENCHMARK_STRING_ARG(name))
#define CARTS_TASK_TIMER_START(name)                                           \
  carts_task_timer_start(CARTS_BENCHMARK_STRING_ARG(name))
#define CARTS_TASK_TIMER_STOP(name)                                            \
  carts_task_timer_stop(CARTS_BENCHMARK_STRING_ARG(name))

#else /* !CARTS_ENABLE_TIMING */

#define CARTS_PARALLEL_TIMER_START(name) ((void)0)
#define CARTS_PARALLEL_TIMER_STOP(name) ((void)0)
#define CARTS_TASK_TIMER_START(name) ((void)0)
#define CARTS_TASK_TIMER_STOP(name) ((void)0)

#endif /* CARTS_ENABLE_TIMING */

///===----------------------------------------------------------------------===///
/// Checksum/Result Reporting
///===----------------------------------------------------------------------===///

/// Print checksum as double (most common case)
/// Output: "checksum: <value>"
void carts_bench_checksum_d(double value);

/// Print checksum as float
/// Output: "checksum: <value>"
void carts_bench_checksum_f(float value);

/// Print checksum as integer
/// Output: "checksum: <value>"
void carts_bench_checksum_i(int value);

/// Print checksum as long
/// Output: "checksum: <value>"
void carts_bench_checksum_l(long value);

/// Print RMS error (for verification benchmarks)
/// Output: "RMS error: <value>"
void carts_bench_rms_error(double error);

/// Print a named result value
/// Output: "<name>: <value>"
void carts_bench_result_named(uintptr_t name, double value);

/// Print sum (alternative to checksum for some benchmarks)
/// Output: "sum: <value>"
void carts_bench_sum(double value);

/// Print total (for aggregation benchmarks)
/// Output: "total: <value>"
void carts_bench_total(double value);

/// Verify result against expected value with tolerance
/// Returns 0 if passed, 1 if failed
/// Output: "Verification: PASS (checksum: <value>)" or
///         "Verification: FAIL (checksum: <value>, expected: <value>, diff:
///         <value>)"
int carts_bench_verify_d(double actual, double expected, double tolerance);

///===----------------------------------------------------------------------===///
/// Convenience Macros
///===----------------------------------------------------------------------===///

/// Generic checksum macro - automatically selects correct function
#if __STDC_VERSION__ >= 201112L
#define CARTS_BENCH_CHECKSUM(val)                                              \
  _Generic((val),                                                              \
      double: carts_bench_checksum_d,                                          \
      float: carts_bench_checksum_f,                                           \
      int: carts_bench_checksum_i,                                             \
      long: carts_bench_checksum_l,                                            \
      default: carts_bench_checksum_d)(val)
#else
#define CARTS_BENCH_CHECKSUM(val) carts_bench_checksum_d((double)(val))
#endif

/// RMS error reporting
#define CARTS_BENCH_RMS_ERROR(error) carts_bench_rms_error(error)

/// Named result reporting
#define CARTS_BENCH_RESULT(name, value)                                        \
  carts_bench_result_named(CARTS_BENCHMARK_STRING_ARG(name), value)

/// Sum reporting
#define CARTS_BENCH_SUM(value) carts_bench_sum(value)

/// Total reporting
#define CARTS_BENCH_TOTAL(value) carts_bench_total(value)

/// Verification with tolerance (returns 0 on pass, 1 on fail)
#define CARTS_BENCH_VERIFY(actual, expected, tol)                              \
  carts_bench_verify_d((double)(actual), (double)(expected), (double)(tol))

#ifdef __cplusplus
}
#endif

#endif // CARTS_BENCHMARKS_H
