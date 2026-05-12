# Live Pipeline Reference

Source of truth: `tools/compile/Compile.cpp`, especially `getStageRegistry()`
and the pass arrays. Validate with:

```bash
dekk carts pipeline --json
```

Use the JSON manifest when exact pass labels, dependency edges, or bracketed
pass names matter.

## Core Stage Order

1. `raise-memref-dimensionality`
2. `initial-cleanup`
3. `openmp-to-arts`
4. `edt-transforms`
5. `create-dbs`
6. `db-opt`
7. `edt-opt`
8. `concurrency`
9. `edt-distribution`
10. `post-distribution-cleanup`
11. `db-partitioning`
12. `post-db-refinement`
13. `late-concurrency-cleanup`
14. `epochs`
15. `pre-lowering`
16. `arts-to-llvm`

`--pipeline` also accepts the sentinel `complete`. `--start-from` accepts core
stages only.

## Conditional Epilogues

- `post-o3-opt` runs when `--O3` is active.
- `llvm-ir-emission` runs when LLVM IR emission is requested.

Do not use stale epilogue names such as `complete-mlir` or `emit-llvm` unless
you are specifically fixing outdated docs.

## High-Value Drift Checks

- `raise-memref-dimensionality` includes `ScalarForwarding`.
- `openmp-to-arts` owns the complete SDE lifecycle and currently contains
  tensor raising, iteration-space decomposition, codelet conversion, token mode
  refinement, and `ConvertSdeToArts`.
- `post-distribution-cleanup` starts with `VerifyEdtBodyPreserved`.
- `pre-lowering` uses `EdtParallelMonolithicLowering`.
