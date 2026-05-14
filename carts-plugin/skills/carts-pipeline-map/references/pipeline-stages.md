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
7. `post-db-refinement`
8. `late-concurrency-cleanup`
9. `epochs`
10. `pre-lowering`
11. `arts-to-llvm`

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
- `openmp-to-arts` owns SDE planning, direct SDE-to-Core materialization,
  `VerifySdeLowered`, and `VerifyCoreObjectsOnly`.
- `post-db-refinement` runs DB/EDT refinements and contract validation after
  DB mode tightening.
- `pre-lowering` lowers Core DB/EDT/epoch objects to RT-shaped operations and
  verifies the result with `VerifyPreLowered`.
