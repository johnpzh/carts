---
name: carts-stage-diff
description: Use when debugging miscompiles, verifying pass correctness, comparing MLIR between pipeline stages, or finding where semantics diverge.
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob, Agent
argument-hint: [<file> <stage1> <stage2> | bisect <file> | all <file>]
parameters:
  - name: input_file
    type: str
    gather: "Path to the MLIR or C input file to compile"
  - name: action
    type: str
    gather: "Action: 'diff <stage1> <stage2>', 'bisect' (find first bad stage), or 'all' (dump all stages)"
---

# CARTS Pipeline Stage Comparison

## Purpose

Compare IR between pipeline stages to understand transformations, find the
first stage where semantics diverge, or bisect a miscompilation.

## Quick Commands

### Dump all stages
```bash
dekk carts compile <file> --all-pipelines -o /tmp/stages/
ls /tmp/stages/
```

### Compare two stages
```bash
dekk carts compile <file> --pipeline=<stage1> > /tmp/before.mlir
dekk carts compile <file> --pipeline=<stage2> > /tmp/after.mlir
diff /tmp/before.mlir /tmp/after.mlir | head -100
```

### Bisect: find first divergent stage
```bash
# Dump all stages, then compare adjacent pairs
dekk carts compile <file> --all-pipelines -o /tmp/stages/
# Check each stage for the symptom (e.g., missing op, wrong attribute)
for f in /tmp/stages/*.mlir; do
  echo "=== $(basename $f) ==="
  grep -c "symptom_pattern" "$f" || true
done
```

## 16 Core Pipeline Stages (Canonical Order)

Canonical source: `tools/compile/Compile.cpp` (`getStageRegistry()` + pass arrays).
Two additional epilogue stages (`complete-mlir`, `emit-llvm`) run conditionally
on `--O3` / `--emit-llvm`.

| # | Stage | Primary Question |
|---|-------|-----------------|
| 1 | raise-memref-dimensionality | Are memrefs normalized? |
| 2 | initial-cleanup | Is dead code removed? |
| 3 | openmp-to-arts | OMP → SDE → ARTS conversion correct? (contains the full SDE lifecycle — RaiseToLinalg, LoopInterchange, DistributionPlanning, ConvertSdeToArts, etc.) |
| 4 | edt-transforms | Is EDT structure optimized? |
| 5 | create-dbs | Are DataBlock allocations created? |
| 6 | db-opt | Are DB access modes correct? |
| 7 | edt-opt | Are EDTs fused/optimized? |
| 8 | concurrency | Is the concurrency graph correct? |
| 9 | edt-distribution | Is distribution strategy assigned? |
| 10 | post-distribution-cleanup | Are loops lowered correctly? |
| 11 | db-partitioning | Are DBs partitioned correctly? |
| 12 | post-db-refinement | Are contracts validated? |
| 13 | late-concurrency-cleanup | Is hoisting/sinking correct? |
| 14 | epochs | Are epochs created correctly? |
| 15 | pre-lowering | Are EDTs/DBs/epochs lowered to RT calls? |
| 16 | arts-to-llvm | Is final LLVM IR correct? |

## Bisection Strategy by Symptom

| Symptom | Start Checking At |
|---------|------------------|
| Wrong array values | db-partitioning (11), then post-db-refinement (12) |
| Missing parallelism | concurrency (8), then edt-distribution (9) |
| Deadlock/hang | epochs (14), then pre-lowering (15) |
| Wrong loop bounds | edt-distribution (9), then post-distribution-cleanup (10) |
| Missing DB | create-dbs (5), then db-opt (6) |
| Pattern/semantic issue | openmp-to-arts (3) — inspect SDE sub-passes via `--arts-debug=<pass>` |
| LLVM crash | arts-to-llvm (16), then pre-lowering (15) |

## What to Look for in Diffs

### Structural changes (ops added/removed)
```bash
# Count ops by type at each stage
grep -o 'arts\.[a-z_]*' /tmp/stages/*.mlir | sort | uniq -c | sort -rn
```

### Attribute changes
```bash
# Track specific attributes across stages
grep 'arts.dep_pattern\|arts.partition_mode\|arts.distribution_kind' /tmp/stages/*.mlir
```

### Loop structure changes
```bash
# Count loops at each stage
for f in /tmp/stages/*.mlir; do
  echo "$(basename $f): $(grep -c 'arts.for\|scf.for' $f) loops"
done
```

## Instructions

When the user asks to compare stages or bisect:

1. Determine input file and action (diff, bisect, or all)
2. Dump relevant stages using `dekk carts compile --pipeline=<stage>`
3. For bisection: use the symptom table to pick starting stages
4. Compare adjacent stages, narrowing to the first divergent one
5. Once found, report: which stage, what changed, which pass is responsible
6. Suggest debug channel: `--arts-debug=<snake_case_pass_name>`
