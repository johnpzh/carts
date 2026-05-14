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
Additional epilogue stages (`post-o3-opt`, `llvm-ir-emission`) run
conditionally when requested.

| # | Stage | Primary Question |
|---|-------|-----------------|
| 1 | raise-memref-dimensionality | Are memrefs normalized? |
| 2 | initial-cleanup | Is dead code removed? |
| 3 | openmp-to-arts | OMP → SDE → ARTS conversion correct? (contains the full SDE lifecycle — RaiseToLinalg, LoopInterchange, DistributionPlanning, ConvertSdeToArts, etc.) |
| 4 | edt-transforms | Is EDT structure optimized? |
| 5 | create-dbs | Are DataBlock allocations created? |
| 6 | db-opt | Are DB access modes correct? |
| 7 | post-db-refinement | Are contracts validated and DB/EDT refinements correct? |
| 8 | late-concurrency-cleanup | Is hoisting/sinking correct? |
| 9 | epochs | Are epochs created correctly? |
| 10 | pre-lowering | Are EDTs/DBs/epochs lowered to RT calls? |
| 11 | arts-to-llvm | Is final LLVM IR correct? |

## Bisection Strategy by Symptom

| Symptom | Start Checking At |
|---------|------------------|
| Wrong array values | create-dbs (5), db-opt (6), then post-db-refinement (7) |
| Missing parallelism | openmp-to-arts (3), then edt-transforms (4) |
| Deadlock/hang | epochs (9), then pre-lowering (10) |
| Wrong loop bounds | openmp-to-arts (3), then late-concurrency-cleanup (8) |
| Missing DB | create-dbs (5), then db-opt (6) |
| Pattern/semantic issue | openmp-to-arts (3) — inspect SDE sub-passes via `--arts-debug=<pass>` |
| LLVM crash | arts-to-llvm (11), then pre-lowering (10) |

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
  echo "$(basename $f): $(grep -c 'scf.for' $f) implementation loops"
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
