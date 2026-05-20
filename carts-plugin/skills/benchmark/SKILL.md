---
name: carts-benchmark
description: Use when the user asks to list, build, run, or compare CARTS benchmarks. Use carts-benchmark-triage for failing, timing out, or suspicious benchmark results.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [list | run <name> | build | clean]
---

# CARTS Benchmarks

Run and manage CARTS benchmarks. NEVER use `./carts-benchmarks` or benchmark scripts directly.

## Usage

Run `dekk carts benchmarks --help` for the latest options. Common:
- `dekk carts benchmarks list` — list all available benchmarks
- `dekk carts benchmarks run polybench/gemm --size small` — run a specific benchmark; single-node configs use TCP
- `dekk carts benchmarks run polybench/gemm --size small --nodes 2` — run a multinode benchmark with RDMA/RoCE transport by default
- `dekk carts benchmarks run polybench/gemm --size small --nodes 2 --no-rdma` — run a multinode TCP fallback experiment
- `dekk carts benchmarks build --suite polybench` — build a suite
- `dekk carts benchmarks perf-gate RESULTS --policy POLICY` — regression gate check

## Performance Regression Testing

1. Build first: `dekk carts build`
2. List benchmarks: `dekk carts benchmarks list`
3. Run benchmarks of interest
4. Compare against known baselines (check memory notes)

### Known Baselines
- Use live benchmark output and `.carts/sessions/...` investigation archives as
  benchmark memory; experiment notes should not live in `docs/compiler`.
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
