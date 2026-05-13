# CARTS Compiler Pipeline

This document mirrors the pipeline order defined in `tools/compile/Compile.cpp`.
If this file disagrees with the compiler source or `dekk carts pipeline --json`,
the live compiler wins.

For layer-specific optimization planning, see [`sde/`](./sde/),
[`core/`](./core/), and [`rt/`](./rt/). For the target split that removes Core
`arts.for` and keeps SDE/Core/RT responsibilities separate, see
[`dialect-layering-vision.md`](./dialect-layering-vision.md).

## CLI Introspection

- `dekk carts pipeline`: show pipeline order and pass counts.
- `dekk carts pipeline --json`: print the machine-readable manifest.
- `dekk carts compile <file> --all-pipelines`: dump stage and pass outputs for
  a focused input.

## Pipeline Order

Core stages:

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
11. `post-db-refinement`
12. `late-concurrency-cleanup`
13. `epochs`
14. `pre-lowering`
15. `arts-to-llvm`

`--pipeline` also accepts the sentinel `complete`. `--start-from` accepts core
stages only.

Conditional epilogues:

- `post-o3-opt`: runs when `--O3` is active.
- `llvm-ir-emission`: runs when LLVM IR emission is requested.

## Stage Summaries

### `raise-memref-dimensionality`

```text
LowerAffine(func)
CSE
ArtsInliner
PolygeistCanonicalize
ScalarForwarding
PolygeistCanonicalize
RaiseMemRefDimensionality
HandleDeps
DeadCodeElimination
CSE
```

### `initial-cleanup`

```text
LowerAffine(func)
CSE(func)
PolygeistCanonicalizeFor(func)
```

### `openmp-to-arts`

The complete SDE lifecycle lives inside this stage.

```text
ConvertOpenMPToSde
RaiseToTensor
RaiseToLinalg
LoopInterchange
Tiling
StructuredSummaries
ElementwiseFusion
ScopeSelection
ScheduleRefinement
ChunkOpt
ReductionStrategy
DistributionPlanning
IterationSpaceDecomposition
BarrierElimination
Vectorization
LowerToMemref
ConvertToCodelet
TensorCleanup
TokenModeRefine
ConvertSdeToArts
VerifySdeLowered
DeadCodeElimination
CSE
VerifyEdtCreated
```

### `edt-transforms`

```text
EdtStructuralOpt(runAnalysis=false)
EdtICM
DeadCodeElimination
SymbolDCE
CSE
EdtPtrRematerialization
```

### `create-dbs`

```text
DistributedHostLoopOutlining (conditional)
CreateDbs
CSE (bridge cleanup, conditional)
PolygeistCanonicalize
CSE
SymbolDCE
Mem2Reg
PolygeistCanonicalize
```

### `db-opt`

```text
DbModeTightening
PolygeistCanonicalize
CSE
Mem2Reg
```

### `edt-opt`

```text
PolygeistCanonicalize
EdtStructuralOpt(runAnalysis=true)
LoopFusion
CSE
```

### `concurrency`

```text
PolygeistCanonicalize
Concurrency
ForOpt
PolygeistCanonicalize
```

### `edt-distribution`

```text
EdtDistribution
EdtOrchestrationOpt
ForLowering
VerifyForLowered
```

### `post-distribution-cleanup`

```text
EdtStructuralOpt(runAnalysis=false)
DeadCodeElimination
PolygeistCanonicalize
CSE
EdtStructuralOpt(runAnalysis=false)
EpochOpt
PolygeistCanonicalize
CSE
```

### `post-db-refinement`

```text
DbModeTightening
EdtTransforms
DbTransforms
ContractValidation
DbScratchElimination
PolygeistCanonicalize
CSE
```

### `late-concurrency-cleanup`

```text
BlockLoopStripMining(func)
Hoisting
PolygeistCanonicalize
CSE
EdtAllocaSinking
DeadCodeElimination
Mem2Reg
```

### `epochs`

```text
PolygeistCanonicalize
CreateEpochs
EpochOpt (conditional)
PolygeistCanonicalize
```

### `pre-lowering`

```text
EdtAllocaSinking
ParallelEdtLowering
EpochOpt (conditional)
PolygeistCanonicalize
CSE
DbLowering
PolygeistCanonicalize
CSE
EdtLowering
PolygeistCanonicalize
CSE
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
VerifyPreLowered
```

### `arts-to-llvm`

```text
ConvertArtsToLLVM
LoweringContractCleanup
GuidRangCallOpt
RuntimeCallOpt
DataPtrHoisting
PolygeistCanonicalize
CSE
Mem2Reg
PolygeistCanonicalize
ControlFlowSink
PolygeistCanonicalize
VerifyLowered
```

## Stage Dependencies

- `initial-cleanup` depends on `raise-memref-dimensionality`.
- `openmp-to-arts` depends on `initial-cleanup`.
- `edt-transforms` depends on `openmp-to-arts`.
- `create-dbs` depends on `openmp-to-arts`.
- `db-opt`, `edt-opt`, and `concurrency` depend on `create-dbs`.
- `edt-distribution` depends on `concurrency`.
- `post-distribution-cleanup` and `post-db-refinement` depend on
  `edt-distribution`.
- `late-concurrency-cleanup` depends on `post-db-refinement`.
- `pre-lowering` depends on `epochs` and `late-concurrency-cleanup`.
- `arts-to-llvm` depends on `pre-lowering`.

## Ownership Notes

- SDE inside `openmp-to-arts` owns semantic decomposition, structured summaries,
  state planning, dependency/effect proofs, and physical DB layout policy.
- `CreateDbs` materializes SDE-authored DB layouts. It should not invent tensor
  partition policy that was visible to SDE.
- Core owns DB/EDT/epoch orchestration and analysis-backed refinement.
- RT-facing lowering belongs in `pre-lowering` and `arts-to-llvm`, after the
  compiler has already chosen the DB and task shape.
