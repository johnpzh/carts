---
name: carts-distributed-triage
description: Use when a failure only appears with `--distributed-db`, multiple nodes, distributed host outlining, or uneven remote work distribution.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [<input-file | benchmark-path>]
---

# CARTS Distributed Triage

Goal: determine whether a multi-node failure comes from ownership marking, lowering, runtime routing, or benchmark/runtime configuration.

Use bundled helpers when they fit:
- `scripts/run-multinode-benchmark.sh` — rerun a benchmark with logs retained
- `scripts/inspect-distributed-ir.sh` — capture the key distributed pipeline stages
- `scripts/find-distributed-sites.sh` — grep the main ownership/routing codepaths
- `scripts/summarize-distributed-artifacts.sh` — locate logs and per-node counter JSONs

Read these before patching anything:
- `references/distributed-checklist.md`
- `references/distributed-codepaths.md`
- `../debug/references/failure-signatures.md`
- `../debug/references/codepath-map.md`
- `../debug/references/command-patterns.md`

## Triage Order

1. Confirm that single-node still works.
2. Reproduce with explicit node/thread/config inputs.
3. Check whether `--distributed-db` is actually active in the failing path.
4. Inspect IR around:
   - `edt-distribution`
   - `db-partitioning`
   - `post-db-refinement`
   - `pre-lowering`
5. Check ownership constraints:
   - `distributed` marker present on eligible `DbAllocOp`
   - host loop outlining happened when required
   - routed work and owner hints agree
6. Inspect runtime artifacts:
   - `arts.log`, `omp.log`
   - `cluster.json`, `n0.json`, `n1.json`, ...
7. If the bug reduces to wrong output rather than multi-node structure, hand off to `carts-miscompile-triage`.

## Common Commands

```bash
# Generate stage dumps with distributed ownership enabled
dekk carts compile input.mlir --distributed-db --pipeline=db-partitioning

# Multi-node benchmark run
dekk carts benchmarks run polybench/2mm \
  --size small \
  --nodes 2 \
  --threads 4 \
  --arts-config docker/arts-docker-2node.cfg \
  --debug 2
```

## Key Files

- `docs/heuristics/distribution.md`
- `lib/arts/dialect/core/Transforms/DbDistributedOwnership.cpp`
- `lib/arts/dialect/core/Transforms/DistributedHostLoopOutlining.cpp`
- `lib/arts/dialect/core/Conversion/ArtsToLLVM/ConvertArtsToLLVM.cpp`
- `lib/arts/dialect/core/Transforms/ForLowering.cpp`
- `lib/arts/codegen/Codegen.cpp`

## Validation

Rerun the distributed workload after every change, and compare against the single-node path before closing.
