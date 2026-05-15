---
name: carts-heuristic-explain
description: Use when a benchmark has unexpected partitioning, wrong distribution mode, heuristic drift, or performance anomalies.
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob, Agent
argument-hint: [partition <file> | distribution <file> | thresholds]
parameters:
  - name: input_file
    type: str
    gather: "Path to MLIR or C input file to analyze"
  - name: focus
    type: str
    gather: "What to explain: 'partition' (H1 DB partitioning), 'distribution' (H2 EDT strategy), or 'both'"
---

# CARTS Heuristic Decision Explainer

## Purpose

Trace and explain why the compiler chose a specific partitioning mode or
distribution strategy. The compiler has 22 named heuristic rules and 21
hardcoded thresholds — this skill makes those decisions transparent.

## Quick Diagnostic Commands

```bash
# Dump IR at partitioning stage to see partition modes
dekk carts compile <file> --pipeline=post-db-refinement 2>/dev/null | grep 'partition_mode'

# Dump IR after SDE planning/materialization to see distribution strategy
dekk carts compile <file> --pipeline=sde-planning 2>/dev/null | grep 'distribution_kind'

# Enable debug output for partitioning decisions
dekk carts compile <file> --pipeline=post-db-refinement --arts-debug=db_transforms 2>&1

# Enable debug output for DB mode/refinement decisions
dekk carts compile <file> --pipeline=db-opt --arts-debug=db_mode_tightening 2>&1

# Full diagnostic JSON
dekk carts compile <file> --diagnose --diagnose-output /tmp/diag.json 2>/dev/null
```

## H1 Partition Heuristic Chain

The compiler evaluates these rules **in order** — first match wins:

| Rule | Condition | Result | Typical Trigger |
|------|-----------|--------|-----------------|
| H1.C0 | Tiny read-only stencil coefficient | COARSE | Small constant arrays |
| H1.C1 | Pointer-of-pointer type | COARSE | `memref<memref<T>>` |
| H1.C2 | Single-node + all read-only | COARSE | Read-only arrays on 1 node |
| H1.C3 | Explicit coarse contract | COARSE | Consumer override |
| H1.C4 | No block/element capability | COARSE | No partition dims found |
| H1.B1 | Indirect reads + block writes | BLOCK | Mixed access patterns |
| H1.B2 | Uniform direct access | BLOCK | Regular array operations |
| H1.B3 | Double-buffer stencil (Jacobi) | BLOCK | Alternating buffers |
| H1.B4 | Indexed access + block capable | BLOCK | Index-based patterns |
| H1.S1 | Element-wise stencil | STENCIL | Fine-grained stencil |
| H1.S2 | Block-capable stencil | STENCIL | Block stencil with halo |
| H1.E1 | Element-wise capable | FINE | Per-element partitioning |
| H1.F | Fallback | COARSE | Safety default |

## H2 Distribution Strategy Selection

| Pattern | Machine | Result |
|---------|---------|--------|
| Matmul + internode | numNodes > 1 | Tiling2D |
| Matmul + intranode | numNodes == 1 | Block |
| Any + internode | numNodes > 1 | TwoLevel |
| Triangular | any | BlockCyclic |
| Stencil/Uniform/Unknown | any | Block |

## Hardcoded Thresholds

| Threshold | Value | Location | Purpose |
|-----------|-------|----------|---------|
| kMaxOuterDBs | 1024 | PartitioningHeuristics.h:67 | Max partitions before coarse materialization |
| kMaxDepsPerEDT | 8 | PartitioningHeuristics.h:68 | Per-EDT acquire threshold |
| kMinInnerBytes | 64 | PartitioningHeuristics.h:69 | Minimum inner-rank byte size |

## Key Source Files

```
include/arts/dialect/core/Analysis/heuristics/PartitioningHeuristics.h  — H1 data structures
lib/arts/dialect/core/Analysis/heuristics/PartitioningHeuristics.cpp     — H1 evaluation chain
include/arts/dialect/core/Analysis/heuristics/DistributionHeuristics.h   — H2 data structures
lib/arts/dialect/core/Analysis/heuristics/DistributionHeuristics.cpp     — H2 strategy selection
lib/arts/dialect/core/Transforms/db/DbTransformsPass.cpp    — Core DB refinement controller
lib/carts/dialect/codir/Conversion/SdeToCodir/SdeToCodir.cpp — SDE-to-CODIR materialization
lib/carts/dialect/codir/Conversion/CodirToArts/CodirToArts.cpp — CODIR-to-ARTS materialization
```

## Instructions

When the user asks to explain a heuristic decision:

1. Identify the focus: partitioning (H1) or distribution (H2)
2. Compile with the relevant `--arts-debug` channel to capture decisions
3. Dump IR at the decision stage (`sde-planning`, `codir-to-arts`, `db-opt`, or `post-db-refinement`)
4. Parse debug output for which heuristic rule fired
5. Explain: what the rule checks, why it matched, what alternatives exist
6. If the decision seems wrong, suggest: which input properties to change,
   or which heuristic rule to investigate in the source code
7. Cross-reference with `docs/heuristics/partitioning.md` and `distribution.md`
