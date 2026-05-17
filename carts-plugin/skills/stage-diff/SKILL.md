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
dekk carts compile <file> --all-pipelines -o .carts/outputs/stages/<topic>/
ls .carts/outputs/stages/<topic>/
```

### Compare two stages
```bash
dekk carts compile <file> --pipeline=<stage1> > .carts/outputs/stages/<topic>-before.mlir
dekk carts compile <file> --pipeline=<stage2> > .carts/outputs/stages/<topic>-after.mlir
diff .carts/outputs/stages/<topic>-before.mlir .carts/outputs/stages/<topic>-after.mlir | head -100
```

### Bisect: find first divergent stage
```bash
# Dump all stages, then compare adjacent pairs
dekk carts compile <file> --all-pipelines -o .carts/outputs/stages/<topic>/
# Check each stage for the symptom (e.g., missing op, wrong attribute)
for f in .carts/outputs/stages/<topic>/*.mlir; do
  echo "=== $(basename $f) ==="
  grep -c "symptom_pattern" "$f" || true
done
```

## 13 Canonical Pipeline Stages

Canonical source: `tools/compile/Compile.cpp` (`getStageRegistry()` + pass arrays).
Additional epilogue stages (`post-o3-opt`, `llvm-ir-emission`) run
conditionally when requested.

| # | Stage | Primary Question |
|---|-------|-----------------|
| 1 | sde-input-normalization | Are Polygeist shapes and OpenMP deps normalized for SDE? |
| 2 | initial-cleanup | Is dead code removed? |
| 3 | sde-planning | Did OMP become the right SDE plan? |
| 4 | sde-to-codir | Are SDE codelets isolated with explicit CODIR deps/params? |
| 5 | codir-to-arts | Did CODIR materialize the intended ARTS DB/acquire/EDT objects? |
| 6 | edt-transforms | Is EDT structure optimized? |
| 7 | create-dbs | Are coarse raw DataBlock allocations created only where allowed? |
| 8 | db-opt | Are DB access modes correct? |
| 9 | post-db-refinement | Are contracts validated and DB/EDT refinements correct? |
| 10 | late-concurrency-cleanup | Is hoisting/sinking correct? |
| 11 | epochs | Are epochs created correctly? |
| 12 | pre-lowering | Are EDTs/DBs/epochs lowered to RT calls? |
| 13 | arts-rt-to-llvm | Is final LLVM IR correct? |

## Bisection Strategy by Symptom

| Symptom | Start Checking At |
|---------|------------------|
| Wrong array values | codir-to-arts (5), create-dbs (7), db-opt (8), then post-db-refinement (9) |
| Missing parallelism | sde-planning (3), sde-to-codir (4), codir-to-arts (5), then edt-transforms (6) |
| Deadlock/hang | epochs (11), then pre-lowering (12) |
| Wrong loop bounds | sde-planning (3), then late-concurrency-cleanup (10) |
| Missing DB | codir-to-arts (5), create-dbs (7), then db-opt (8) |
| Pattern/semantic issue | sde-planning (3) — inspect SDE sub-passes via `--arts-debug=<pass>` |
| LLVM crash | arts-rt-to-llvm (13), then pre-lowering (12) |

## What to Look for in Diffs

### Structural changes (ops added/removed)
```bash
# Count ops by type at each stage
grep -o 'arts\.[a-z_]*' .carts/outputs/stages/<topic>/*.mlir | sort | uniq -c | sort -rn
```

### Attribute changes
```bash
# Track specific attributes across stages
grep 'arts.dep_pattern\|arts.partition_mode\|arts.distribution_kind' .carts/outputs/stages/<topic>/*.mlir
```

### Loop structure changes
```bash
# Count loops at each stage
for f in .carts/outputs/stages/<topic>/*.mlir; do
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
