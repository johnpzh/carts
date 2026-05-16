---
name: carts-benchmark-triage
description: Use when a benchmark fails, times out, produces wrong checksums, shows suspicious speedups, or needs pass-by-pass/runtime diagnosis.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [<benchmark-path>]
parameters:
  - name: benchmark_suite
    type: str
    gather: "Which benchmark suite/name is failing? (e.g., polybench/2mm, nas/cg)"
  - name: baseline_results
    type: str
    gather: "Path to baseline results or 'none' to establish fresh baseline"
---

# CARTS Benchmark Triage

Use bundled helpers when they fit:
- `scripts/rerun-benchmark.sh` — rerun `dekk carts triage-benchmark` with explicit size/threads/stages
- `scripts/locate-run-artifacts.sh` — locate logs, flags, and configs under a benchmark results tree
- `scripts/find-benchmark-codepaths.sh` — grep benchmark-driver and compiler codepaths

Read these before classifying the regression:
- `references/failure-classification.md`
- `../debug/references/failure-signatures.md`
- `../debug/references/command-patterns.md`

## Primary Command

```bash
dekk carts triage-benchmark <suite/name> --size small --threads 2
```

Optional narrowing to specific stages:
```bash
dekk carts triage-benchmark <suite/name> --stages sde-planning,codir-to-arts,post-db-refinement
```

Only use stage names from `dekk carts pipeline --json`.

## Triage Order

1. `dekk carts doctor` + `dekk carts build` to confirm environment health
2. Rerun the failing benchmark with same size and thread count
3. Check runner artifacts: `arts.log`, `omp.log`, `.arts_cflags`, `.omp_cflags`, `arts.cfg`
4. Classify the failure:
   - OpenMP baseline also broken (test with `dekk carts clang`)
   - ARTS-only regression from a pipeline stage
   - Benchmark-side UB or invalid verification
   - Runtime/distributed lowering issue
5. Read stage dumps in order: `sde-planning` -> `sde-to-codir` -> `codir-to-arts` -> `edt-transforms` -> `create-dbs` -> `db-opt` -> `post-db-refinement` -> `late-concurrency-cleanup` -> `epochs` -> `pre-lowering` -> `arts-rt-to-llvm`

If the regression is multi-node specific or depends on `--distributed-db`, switch to `carts-distributed-triage`.

## OpenMP Baseline

Always check the reference path before assuming a CARTS transform bug:
```bash
dekk carts cgeist source.c -O3 -S --emit-llvm -fopenmp ... -o bench-omp.ll
dekk carts clang bench-omp.ll ... -o bench_omp
```

## Key Docs

- `docs/compiler/pipeline.md`, `docs/heuristics/partitioning.md`, `docs/heuristics/distribution.md`
- `tools/compile/Compile.cpp`
- `lib/carts/dialect/codir/Conversion/SdeToCodir/SdeToCodir.cpp`
- `lib/carts/dialect/codir/Conversion/CodirToArts/CodirToArts.cpp`
- `lib/carts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`
- `lib/carts/dialect/sde/Transforms/effect/scheduling/ReductionStrategy.cpp`
- `lib/carts/dialect/arts/Transforms/db/DbTransformsPass.cpp`
- `external/carts-benchmarks/common/carts.mk`

## Hand-off

- Wrong output / checksum mismatch -> `carts-miscompile-triage`
- Runtime hang / crash -> `carts-runtime-triage`
- Multi-node specific -> `carts-distributed-triage`
- Need pipeline bisection -> `/stage-diff`
- Need minimal test case -> `carts-reproducer`
- Partitioning looks wrong -> `/heuristic-explain`

## Validation

Always rerun before closing: `dekk carts test`, the affected benchmark, and the relevant sweep. For scheduling/lowering/ownership changes, also rerun a larger case + 64-thread case.
