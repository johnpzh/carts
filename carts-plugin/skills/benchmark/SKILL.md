---
name: carts-benchmark
description: Use when the user asks about benchmarks, performance testing, regression testing, speedup measurements, benchmark listing, or benchmark comparison.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [list | run <name> | build | clean]
---

# CARTS Benchmarks

Run and manage CARTS benchmarks. NEVER use `./carts-benchmarks` or benchmark scripts directly.

## Usage

Run `dekk carts benchmarks --help` for the latest options. Common:
- `dekk carts benchmarks list` — list all available benchmarks
- `dekk carts benchmarks run polybench/gemm --size small` — run a specific benchmark
- `dekk carts benchmarks build --suite polybench` — build a suite
- `dekk carts benchmarks perf-gate RESULTS --policy POLICY` — regression gate check

## Performance Regression Testing

1. Build first: `dekk carts build`
2. List benchmarks: `dekk carts benchmarks list`
3. Run benchmarks of interest
4. Compare against known baselines (check memory notes)

### Known Baselines
- Use `docs/compiler/benchmark-performance-goal.md` as the current benchmark
  memory. Recent repeated large/64 focused evidence has `polybench/gemm` faster
  than OpenMP at median but noisy, with `polybench/2mm` and `polybench/3mm`
  blocked on matrix-chain/intermediate reuse and stability.
- **specfem3d**: 126x slower (triple-indirected arrays, known root cause)

### Known Issues
- **STREAM**: Nested `omp.wsloop` inside serial loop creates 1 serial EDT
- **jacobi2d**: Benchmark runner regex bug matches commented config values

## Instructions

When the user asks about benchmarks:
1. If no specific request, run `dekk carts benchmarks list` first
2. Run the requested benchmark(s)
3. Report results with speedup/slowdown metrics
4. Flag results that differ from known baselines as potential regressions
