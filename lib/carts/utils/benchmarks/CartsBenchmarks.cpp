///===----------------------------------------------------------------------===///
/// File: CartsBenchmarks.cpp
///
/// Implementation of CARTS benchmark utilities.
/// This file MUST be compiled with -O0 to prevent the empty parallel region
/// from being optimized away.
///===----------------------------------------------------------------------===///

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

///===----------------------------------------------------------------------===///
/// High-Precision Wall Clock Timing
///===----------------------------------------------------------------------===///

double carts_bench_get_time(void);

static const char *carts_e2e_timer_name = nullptr;
static uint64_t carts_e2e_timer_start_ns = 0;
static double carts_program_timer_start_sec = 0.0;
static double carts_kernel_timer_start_sec = 0.0;
static double carts_kernel_timer_accum_sec = 0.0;
static thread_local double carts_parallel_timer_start_sec = 0.0;
static thread_local double carts_task_timer_start_sec = 0.0;

struct PhaseTimerState {
  const char *name;
  double start_sec;
};

static PhaseTimerState carts_startup_timer = {nullptr, 0.0};
static PhaseTimerState carts_verification_timer = {nullptr, 0.0};
static PhaseTimerState carts_cleanup_timer = {nullptr, 0.0};

static const char *carts_benchmark_string(uintptr_t value) {
  return reinterpret_cast<const char *>(value);
}

static PhaseTimerState *carts_get_phase_timer(const char *phase) {
  if (!phase)
    return nullptr;
  if (std::strcmp(phase, "startup") == 0)
    return &carts_startup_timer;
  if (std::strcmp(phase, "verification") == 0)
    return &carts_verification_timer;
  if (std::strcmp(phase, "cleanup") == 0)
    return &carts_cleanup_timer;
  return nullptr;
}

static int carts_benchmarks_worker_id(void) {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

/// Get current time in nanoseconds using std::chrono::steady_clock
uint64_t carts_e2e_timer_get_time_ns(void) {
  auto now = std::chrono::steady_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      now.time_since_epoch());
  return static_cast<uint64_t>(ns.count());
}

double carts_bench_get_time(void) {
  return (double)carts_e2e_timer_get_time_ns() * 1e-9;
}

__attribute__((noinline)) void carts_e2e_timer_start(uintptr_t name) {
  carts_e2e_timer_name = carts_benchmark_string(name);
  carts_e2e_timer_start_ns = carts_e2e_timer_get_time_ns();
}

__attribute__((noinline)) void carts_e2e_timer_stop(void) {
  uint64_t start_ns = carts_e2e_timer_start_ns;
  uint64_t end_ns = carts_e2e_timer_get_time_ns();
  const char *name = carts_e2e_timer_name ? carts_e2e_timer_name : "unnamed";
  double elapsed_sec = start_ns ? (double)(end_ns - start_ns) * 1e-9 : 0.0;
  printf("e2e.%s: %.9fs\n", name, elapsed_sec);
  fflush(stdout);
}

__attribute__((noinline)) void carts_phase_timer_start(uintptr_t phase,
                                                       uintptr_t name) {
  PhaseTimerState *timer = carts_get_phase_timer(carts_benchmark_string(phase));
  if (!timer)
    return;
  timer->name = carts_benchmark_string(name);
  timer->start_sec = carts_bench_get_time();
}

__attribute__((noinline)) void carts_phase_timer_stop(uintptr_t phase) {
  const char *phaseNameArg = carts_benchmark_string(phase);
  PhaseTimerState *timer = carts_get_phase_timer(phaseNameArg);
  if (!timer)
    return;

  const char *phaseName = phaseNameArg ? phaseNameArg : "phase";
  const char *name = timer->name ? timer->name : "unnamed";
  double elapsed_sec =
      timer->start_sec == 0.0 ? 0.0 : carts_bench_get_time() - timer->start_sec;
  printf("%s.%s: %.6fs\n", phaseName, name, elapsed_sec);
  fflush(stdout);
}

__attribute__((noinline)) void carts_bench_timer_start(void) {
  carts_program_timer_start_sec = carts_bench_get_time();
}

__attribute__((noinline)) double carts_bench_timer_stop(void) {
  if (carts_program_timer_start_sec == 0.0)
    return 0.0;
  return carts_bench_get_time() - carts_program_timer_start_sec;
}

__attribute__((noinline)) void carts_bench_timer_print(void) {
  printf("time: %.6f\n", carts_bench_timer_stop());
  fflush(stdout);
}

__attribute__((noinline)) void carts_kernel_timer_start(uintptr_t name) {
  (void)name;
  carts_kernel_timer_accum_sec = 0.0;
  carts_kernel_timer_start_sec = carts_bench_get_time();
}

__attribute__((noinline)) void carts_kernel_timer_stop(uintptr_t name) {
  const char *kernelNameArg = carts_benchmark_string(name);
  const char *kernelName = kernelNameArg ? kernelNameArg : "unnamed";
  double elapsed_sec =
      carts_kernel_timer_start_sec == 0.0
          ? 0.0
          : carts_bench_get_time() - carts_kernel_timer_start_sec;
  printf("kernel.%s: %.6fs\n", kernelName, elapsed_sec);
  fflush(stdout);
}

__attribute__((noinline)) void carts_kernel_timer_accum(uintptr_t name) {
  (void)name;
  if (carts_kernel_timer_start_sec == 0.0)
    return;
  double now_sec = carts_bench_get_time();
  carts_kernel_timer_accum_sec += now_sec - carts_kernel_timer_start_sec;
  carts_kernel_timer_start_sec = now_sec;
}

__attribute__((noinline)) void carts_kernel_timer_print(uintptr_t name) {
  const char *kernelNameArg = carts_benchmark_string(name);
  const char *kernelName = kernelNameArg ? kernelNameArg : "unnamed";
  printf("kernel.%s: %.6fs\n", kernelName, carts_kernel_timer_accum_sec);
  fflush(stdout);
}

__attribute__((noinline)) void carts_parallel_timer_start(uintptr_t name) {
  (void)name;
  carts_parallel_timer_start_sec = carts_bench_get_time();
}

__attribute__((noinline)) void carts_parallel_timer_stop(uintptr_t name) {
  const char *regionNameArg = carts_benchmark_string(name);
  const char *regionName = regionNameArg ? regionNameArg : "unnamed";
  double elapsed_sec =
      carts_parallel_timer_start_sec == 0.0
          ? 0.0
          : carts_bench_get_time() - carts_parallel_timer_start_sec;
  printf("parallel.%s[worker=%d]: %.6fs\n", regionName,
         carts_benchmarks_worker_id(), elapsed_sec);
  fflush(stdout);
}

__attribute__((noinline)) void carts_task_timer_start(uintptr_t name) {
  (void)name;
  carts_task_timer_start_sec = carts_bench_get_time();
}

__attribute__((noinline)) void carts_task_timer_stop(uintptr_t name) {
  const char *taskNameArg = carts_benchmark_string(name);
  const char *taskName = taskNameArg ? taskNameArg : "unnamed";
  double elapsed_sec =
      carts_task_timer_start_sec == 0.0
          ? 0.0
          : carts_bench_get_time() - carts_task_timer_start_sec;
  printf("task.%s[worker=%d]: %.6fs\n", taskName, carts_benchmarks_worker_id(),
         elapsed_sec);
  fflush(stdout);
}

///===----------------------------------------------------------------------===///
/// OMP Thread Pool Warmup
///===----------------------------------------------------------------------===///

/// Pre-warm OMP thread pool with empty parallel region.
/// Compiled with -O0 to prevent the optimizer from removing this.
#ifdef _OPENMP
__attribute__((noinline)) void carts_benchmarks_warmup_omp(void) {
#pragma omp parallel
  {
    /// Intentionally empty - forces thread pool creation
  }
}
#else
__attribute__((noinline)) void carts_benchmarks_warmup_omp(void) {
  /// No-op for non-OMP builds
}
#endif

///===----------------------------------------------------------------------===///
/// Reporting Controls
///===----------------------------------------------------------------------===///

static int carts_benchmarks_should_report_init(void) {
  const char *v = std::getenv("CARTS_BENCHMARKS_REPORT_INIT");
  if (!v || !*v)
    return 0;
  return !(v[0] == '0' && v[1] == '\0');
}

static int carts_benchmarks_host_openmp = 0;

__attribute__((noinline)) void carts_benchmarks_mark_host_openmp(void) {
  carts_benchmarks_host_openmp = 1;
}

///===----------------------------------------------------------------------===///
/// ARTS Init Time (process start -> entering mainBody)
///===----------------------------------------------------------------------===///

static uint64_t carts_benchmarks_process_start_ns = 0;
static int carts_benchmarks_reported_arts_init = 0;

__attribute__((constructor)) static void carts_benchmarks_ctor(void) {
  carts_benchmarks_process_start_ns = carts_e2e_timer_get_time_ns();
}

static int carts_benchmarks_is_arts_binary(void) {
  if (carts_benchmarks_host_openmp)
    return 0;
  /// OpenMP reference binaries do not link against ARTS, but CARTS-generated
  /// binaries do. Use symbol presence as a robust discriminator even when
  /// `artsConfig` env var is not set (e.g., using ./arts.cfg in CWD).
  return dlsym(RTLD_DEFAULT, "arts_rt") != nullptr;
}

__attribute__((noinline)) static void carts_benchmarks_report_arts_init(void) {
  if (!carts_benchmarks_should_report_init())
    return;
  if (__sync_lock_test_and_set(&carts_benchmarks_reported_arts_init, 1))
    return;

  uint64_t start_ns = carts_benchmarks_process_start_ns;
  if (!start_ns)
    start_ns = carts_e2e_timer_get_time_ns();
  uint64_t now_ns = carts_e2e_timer_get_time_ns();
  double elapsed_sec = (double)(now_ns - start_ns) * 1e-9;
  printf("init.arts: %.9fs\n", elapsed_sec);
  fflush(stdout);
}

///===----------------------------------------------------------------------===///
/// Benchmark Start Hook
///===----------------------------------------------------------------------===///

__attribute__((noinline)) void carts_benchmarks_start(void) {
  if (carts_benchmarks_is_arts_binary()) {
    carts_benchmarks_report_arts_init();
    return;
  }

  /// OpenMP reference path. Safe as a no-op when OpenMP is not enabled.
#ifdef _OPENMP
  uint64_t t0 = carts_e2e_timer_get_time_ns();
  carts_benchmarks_warmup_omp();
  uint64_t t1 = carts_e2e_timer_get_time_ns();
  if (carts_benchmarks_should_report_init()) {
    double elapsed_sec = (double)(t1 - t0) * 1e-9;
    printf("init.omp: %.9fs\n", elapsed_sec);
    fflush(stdout);
  }
#else
  (void)0;
#endif
}

///===----------------------------------------------------------------------===///
/// Checksum/Result Reporting
///===----------------------------------------------------------------------===///

void carts_bench_checksum_d(double value) {
  printf("checksum: %.12e\n", value);
}
void carts_bench_checksum_f(float value) { printf("checksum: %.6e\n", value); }
void carts_bench_checksum_i(int value) { printf("checksum: %d\n", value); }
void carts_bench_checksum_l(long value) { printf("checksum: %ld\n", value); }
void carts_bench_rms_error(double error) {
  printf("RMS error: %.12e\n", error);
}
void carts_bench_result_named(uintptr_t name, double value) {
  const char *resultName = carts_benchmark_string(name);
  printf("%s: %.12e\n", resultName ? resultName : "result", value);
}
void carts_bench_sum(double value) { printf("sum: %.12e\n", value); }
void carts_bench_total(double value) { printf("total: %.12e\n", value); }
int carts_bench_verify_d(double actual, double expected, double tolerance) {
  double diff;
  if (expected == 0.0)
    diff = fabs(actual);
  else
    diff = fabs((actual - expected) / expected);

  int passed = (diff < tolerance);
  if (passed) {
    printf("Verification: PASS (checksum: %.12e)\n", actual);
  } else {
    printf(
        "Verification: FAIL (checksum: %.12e, expected: %.12e, diff: %.2e)\n",
        actual, expected, diff);
  }

  return passed ? 0 : 1;
}

} /// extern "C"
