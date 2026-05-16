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
3. `sde-planning`
4. `sde-to-codir`
5. `codir-to-arts`
6. `edt-transforms`
7. `create-dbs`
8. `db-opt`
9. `post-db-refinement`
10. `late-concurrency-cleanup`
11. `epochs`
12. `pre-lowering`
13. `arts-rt-to-llvm`

`--pipeline` also accepts the sentinel `complete`. `--start-from` accepts core
stages only.

## Conditional Epilogues

- `post-o3-opt` runs when `--O3` is active.
- `llvm-ir-emission` runs when LLVM IR emission is requested.

Do not use stale epilogue names such as `complete-mlir` or `emit-llvm` unless
you are specifically fixing outdated docs.

## High-Value Drift Checks

- `raise-memref-dimensionality` includes `ScalarForwarding`.
- `sde-planning` owns OpenMP-to-SDE conversion, pattern/distribution/reduction
  planning, iteration-space decomposition, and MU materialization intent.
- `sde-to-codir` owns codelet isolation, explicit deps/params, token-local
  memref views, and `VerifyCodir`.
- `codir-to-arts` owns CODIR-to-ARTS DB/acquire/EDT materialization and
  rejects any surviving SDE operation.
- `post-db-refinement` runs DB/EDT refinements and contract validation after
  DB mode tightening.
- `pre-lowering` lowers Core DB/EDT/epoch objects to RT-shaped operations and
  verifies the result with `VerifyPreLowered`.
