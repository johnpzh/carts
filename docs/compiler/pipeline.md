# CARTS Compiler Pipeline

This document mirrors the pipeline order defined in `tools/compile/Compile.cpp`.
If this file disagrees with the compiler source or `dekk carts pipeline --json`,
the live compiler wins.

For per-dialect documentation (analysis, optimizations, READMEs), see
[`dialects/sde/`](./dialects/sde/), [`dialects/codir/`](./dialects/codir/),
[`dialects/arts/`](./dialects/arts/), and
[`dialects/arts-rt/`](./dialects/arts-rt/). For the target
`sde -> codir -> arts -> arts-rt` split, see
[`dialect-layering.md`](./dialect-layering.md).

Planning notes and experiment records live under `.carts/sessions/...`; this
file stays limited to the live compiler pipeline.

## CLI Introspection

- `dekk carts pipeline`: show pipeline order and pass counts.
- `dekk carts pipeline --json`: print the machine-readable manifest.
- `dekk carts compile <file> --all-pipelines`: dump stage and pass outputs for
  a focused input.
- `carts-compile <file> --pass-pipeline='builtin.module(...)'`: run a focused
  MLIR pass pipeline for registration and boundary smoke tests. This bypasses
  the staged CARTS pipeline and should not be used as normal project entrypoint
  work; use `dekk carts ...` commands for regular compilation and testing.

The JSON manifest includes both executable pipeline steps and dialect grouping
metadata. `dialect_groups.current` names the groups implemented by the live
stages. The codelet path now runs through `sde -> codir -> arts`: SDE performs
planning, CODIR isolates codelets and token-local views, and ARTS materializes
DB/EDT objects. These group records are descriptive; only `pipeline`,
`start_from`, and `pipeline_sequence` list canonical stage tokens.

## Pipeline Order

Driver stages:

1. `sde-input-normalization`
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

Conditional epilogues:

- `post-o3-opt`: runs when `--O3` is active.
- `llvm-ir-emission`: runs when LLVM IR emission is requested.

## Stage Summaries

### `sde-input-normalization`

```text
LowerAffine(func)
CSE
SdeInputInliner
PolygeistCanonicalize
ScalarForwarding
PolygeistCanonicalize
SdeMemrefNormalization
SdeHandleDeps
SdeDeadStateCleanup
CSE
```

### `initial-cleanup`

```text
LowerAffine(func)
CSE(func)
PolygeistCanonicalizeFor(func)
```

### `sde-planning`

This stage covers OpenMP-to-SDE conversion and SDE planning only. It
intentionally stops before codelet materialization; `sde-to-codir` owns the
codelet boundary and `codir-to-arts` owns ARTS object materialization.

```text
ConvertOpenMPToSde
PatternAnalysis
LoopInterchange
Tiling
ElementwiseFusion
Vectorization
ScheduleRefinement
ChunkOpt
ReductionStrategy
DistributionPlanning
IterationSpaceDecomposition
BarrierElimination
VerifySdeCpsPlan
MemoryUnitMaterialization
```

### `sde-to-codir`

```text
ConvertSdeToCodir
CodirCodeletOpt
VerifyCodir
```

### `codir-to-arts`

This stage materializes CODIR codelets as ARTS objects and then verifies that
no SDE operations survived the boundary.

```text
ConvertCodirToArts
VerifySdeLowered
VerifyArtsObjectsOnly
ArtsDeadCodeElimination
CSE(arts.edt)
VerifyEdtCreated
```

### `edt-transforms`

```text
EdtStructuralOpt(runAnalysis=false)
ArtsDeadCodeElimination
SymbolDCE
CSE(arts.edt)
EdtPtrRematerialization
```

### `create-dbs`

```text
CreateDbs
PolygeistCanonicalize
CSE(arts.edt)
SymbolDCE
Mem2Reg
PolygeistCanonicalize
```

### `db-opt`

```text
DbModeTightening
PolygeistCanonicalize
CSE(arts.edt)
Mem2Reg
```

### `post-db-refinement`

```text
DbModeTightening
DbDistributedOwnership (conditional)
EdtTransforms
DbTransforms
ContractValidation
DbScratchElimination
PolygeistCanonicalize
CSE(arts.edt)
```

### `late-concurrency-cleanup`

```text
BlockLoopStripMining(func)
Hoisting
PolygeistCanonicalize
CSE(arts.edt)
EdtAllocaSinking
ArtsDeadCodeElimination
Mem2Reg
```

### `epochs`

```text
PolygeistCanonicalize
CreateEpochs
VerifyEpochCreated
EpochOpt[scheduling] (conditional)
PolygeistCanonicalize
```

### `pre-lowering`

This stage is implemented by ARTS-RT lowering code even though it still consumes
the abstract ARTS object graph.

```text
EdtAllocaSinking
PolygeistCanonicalize
CSE(arts.edt)
DbLowering
PolygeistCanonicalize
CSE(arts.edt)
EdtLowering
PolygeistCanonicalize
CSE
VerifyEdtLowered
LICM
DataPtrHoisting
PolygeistCanonicalize
CSE
ScalarReplacement
PolygeistCanonicalize
CSE
EpochLowering
PolygeistCanonicalize
CSE
VerifyEpochLowered
VerifyPreLowered
```

### `arts-rt-to-llvm`

This is the ARTS-RT-owned runtime ABI and LLVM-facing lowering stage. The
canonical manifest token is `arts-rt-to-llvm`; no legacy stage alias is
accepted.

```text
LowerAffine(func)
ConvertArtsRtToLLVM
LoweringContractCleanup
GuidRangeCallOpt
RuntimeCallOpt
DataPtrHoisting
PolygeistCanonicalize
CSE
Mem2Reg
PolygeistCanonicalize
ControlFlowSink
PolygeistCanonicalize
VerifyDbLowered
VerifyLowered
```

## Stage Dependencies

- `initial-cleanup` depends on `sde-input-normalization`.
- `sde-planning` depends on `initial-cleanup`.
- `sde-to-codir` depends on `sde-planning`.
- `codir-to-arts` depends on `sde-to-codir`.
- `edt-transforms` depends on `codir-to-arts`.
- `create-dbs` depends on `codir-to-arts`.
- `db-opt` depends on `create-dbs`.
- `post-db-refinement` depends on `create-dbs`.
- `late-concurrency-cleanup` depends on `post-db-refinement`.
- `epochs` depends on `post-db-refinement`.
- `pre-lowering` depends on `epochs` and `late-concurrency-cleanup`.
- `arts-rt-to-llvm` depends on `pre-lowering`.

## Ownership Notes

- SDE inside `sde-planning` owns semantic decomposition, `PatternAnalysis`,
  state planning, dependency/effect proofs, and physical DB layout policy.
- CODIR is the active isolated-codelet layer. It owns explicit deps, params,
  token-local views, and codelet capture verification before ARTS EDT creation.
- `CreateDbs` is now only a coarse raw-memref bridge. It rejects blocked/tiled
  raw memrefs because SDE/CODIR must perform MU/token storage and access
  rewrites before ARTS.
- `arts` owns DB/EDT/epoch orchestration and analysis-backed refinement
  (source: `lib/carts/dialect/arts/`).
- `arts-rt` lowering belongs in `pre-lowering` and `arts-rt-to-llvm`, after the
  compiler has already chosen the DB and task shape (source:
  `lib/carts/dialect/arts-rt/`).
