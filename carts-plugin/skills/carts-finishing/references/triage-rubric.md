# Error Triage Rubric

Apply this rubric whenever a sample or benchmark fails. The principle: **errors usually surface several stages downstream of where the regression was introduced.** Fixing in the surface layer creates silent miscompilation.

## How to use this file

1. Run the failing test, capture the error output.
2. Identify the surface stage (usually printed in the MLIR error or in the runtime stack trace).
3. Match against the **decision tree** to identify likely originating stages.
4. Walk the **verification barrier ladder** to bound the regression window: the bug lives between the last green barrier and the first failing barrier.
5. Use the **stage-to-file map** to locate the file to inspect.
6. Apply the fix in the originating layer, NOT the surface layer.
7. Cross-check against **anti-patterns** to confirm you are not repeating a documented mistake.

## Triage decision tree

When an error surfaces at stage N, check the listed prior stages first.

| Surface | Symptom text | Likely originating stages | First thing to check |
|---|---|---|---|
| Stage 3 (openmp-to-arts) | `SDE operation … survived past SDE-to-ARTS conversion` | `ConvertOpenMPToSde`, `RaiseToLinalg`, `RaiseToTensor`, `ConvertSdeToArts` | Did `ConvertSdeToArts` erase all transient linalg/tensor carriers? See `SdeToArtsPatterns.cpp` lines 520–610. |
| Stage 3 | `cannot trace memref operand to its underlying allocation` | `RaiseMemrefToTensor`, `CreateDbs`, `RaiseMemRefDimensionality` | Memref-of-memref reaching CreateDbs from a heap allocation. |
| Stage 5 (CreateDbs) | `un-normalizable nested memref pattern (element type is memref)` | `RaiseMemRefDimensionality`, `ConvertOpenMPToSde` | Polygeist producing `memref<?xmemref<?xT>>` from `int *A = malloc(N)`. Fix upstream. |
| Stage 8 (DbPartitioning) | Stack overflow / infinite recursion in `copyArtsMetadataAttrs` | DbPartitioning metadata copy, rewriter loops | Add depth guard at `PartitioningHeuristics.cpp` near line 706 (Phase 2 work). |
| Stage 8 (DbPartitioning) | Wrong answer; `fine_grained` silently downgraded to `coarse`, indices dropped | DbPartitioning memref-of-memref handling | Single-element wrapper DB instead of N-element data. The fix is upstream in CreateDbs (Phase 3), NOT in the partition heuristic. |
| `post-db-refinement` | Stencil halo bounds wrong | SDE access-window plan, DB refinement contract rewrite | Confirm SDE stamped the right halo/window, then check whether DB refinement rewrote it. |
| `pre-lowering` / `arts-to-llvm` | `arts.db_alloc` / `arts.edt` / `arts.epoch` survived to LLVM | DbLowering, EdtLowering, EpochLowering | A lowering was skipped. Look for a gating attribute (e.g., `NoOpDbRewriter`, `IsExplicitLeafContract`) that excluded this op. |
| `pre-lowering` | `arts.db_acquire` references GUID that does not exist | DB refinement (GUID lost), SDE-to-Core materialization (dep routing), EpochLowering (CPS carry corruption) | Trace the GUID source. Did an intermediate pass erase it? |
| Stage 14 (EpochLowering) | CPS chain: wrong iteration counter or outer epoch GUID | `EpochOpt` CPS-8 carry re-analysis, `EpochLowering` propagation | Check `CPSParamPerm` and `CPSIterCounterParamIdx` post-EpochOpt. |
| Stage 14 | `CPS advance: rebuilt continuation pack with N schema holes` | `EpochOpt` carry analysis, intermediate EDT fusion, `EdtLowering` pack ordering | Carry arity changed between EpochOpt and EpochLowering. Zero-filled slots carry garbage. |
| `post-db-refinement` (distributed) | Stencil halo not applied, internode acquire uses full range | SDE scope/window mismatch, DbDistributedOwnership attr missing | Is `distributed` attr on the DB? Did SDE mark a distributed scope before Core materialization? |
| `post-db-refinement` (distributed) | Task hangs on remote data acquire | Dependency-window narrowing, DB refinement scope-aware bounds | A dep window may have narrowed to local-only; distributed task cannot reach halo. |
| Stage 16 (ArtsToRt) | `arts_rt.edt_create` argument count mismatch | `EdtLowering` pack construction, prior DB/EDT rewrites invalidating pack operand | `EdtParamPackOp` was rewritten; `EdtCreateOp` references old pack. |
| Post-LLVM | Wrong value in loop or task parameter (silent miscompute) | `EpochLowering` CPS carry slot corruption, `EdtLowering` pack slot reordering | Run `dekk carts compile --all-pipelines` and inspect `pipelines/4_rt/` for the pack structure. |

## Verification barrier ladder

Each verification pass is a freeze point. If barrier X passes but barrier Y fails, the bug is in stages between them. Run `dekk carts compile <file> --all-pipelines -o /tmp/stages/` then inspect each stage dump.

| Barrier | File | Asserts | Failure severity |
|---|---|---|---|
| `VerifySdeLowered` | `lib/arts/dialect/sde/Verify/VerifySdeLowered.cpp` | No `arts_sde.*` ops survive stage 3. No transient linalg/tensor carriers survive the SDE/Core boundary. | Fatal |
| `VerifyCoreObjectsOnly` | `lib/arts/dialect/core/Transforms/verify/VerifyCoreObjectsOnly.cpp` | No Core loop carrier or semantic parallel EDT survives stage 3. Core contains runtime-shaped EDT/DB/epoch objects plus implementation `scf.for`. | Fatal |
| `VerifyEdtCreated` | `lib/arts/dialect/core/Transforms/verify/VerifyEdtCreated.cpp` | At least one `arts.edt` exists post-OpenMP conversion. | Warning |
| `VerifyDbLowered` | `lib/arts/dialect/core/Transforms/verify/VerifyDbLowered.cpp` | No `arts.db_alloc` / `db_acquire` / `db_release` survive pre-lowering. | Fatal |
| `VerifyEpochLowered` | `lib/arts/dialect/core/Transforms/verify/VerifyEpochLowered.cpp` | No `arts.epoch` survives pre-lowering. All become `arts_rt.create_epoch` + `wait_on_epoch`. | Fatal |
| `VerifyEdtLowered` | `lib/arts/dialect/core/Transforms/verify/VerifyEdtLowered.cpp` | No `arts.edt` survives pre-lowering. All lower to `arts_rt.edt_create` + pack. | Fatal |
| `VerifyPreLowered` | `lib/arts/dialect/core/Transforms/verify/VerifyPreLowered.cpp` | IR ready for codegen. Param packs, dep routing, CPS attrs all consistent. | Fatal |
| `VerifyLowered` | `lib/arts/dialect/rt/Transforms/VerifyLowered.cpp` | No `arts.*` or `arts_rt.*` ops survive post-LLVM lowering. | Fatal |

## Stage-to-file map

Use to locate where to grep when triaging.

- Stages 1–3 (SDE): `lib/arts/dialect/sde/Transforms/` + `sde/Conversion/OmpToSde/`
- Stages 4–7 (EDT cleanup, CreateDbs): `lib/arts/dialect/core/Transforms/edt/`, `core/Transforms/db/CreateDbs.cpp`, `core/Transforms/RaiseMemRefDimensionality.cpp`
- Stage 8 (DbPartitioning): `lib/arts/dialect/core/Transforms/db/DbPartitioning.cpp`, `core/Analysis/heuristics/PartitioningHeuristics.cpp`
- Stages 9–10 (DB opt, post-distribution cleanup): `lib/arts/dialect/core/Transforms/db/`, `core/Transforms/edt/`
- SDE/Core materialization: `lib/arts/dialect/core/Conversion/SdeToArts/SdeToArtsPatterns.cpp`
- SDE distribution/reduction planning: `lib/arts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`, `lib/arts/dialect/sde/Transforms/effect/scheduling/ReductionStrategy.cpp`
- Stage 13 (ConvertArtsToLLVM): `lib/arts/dialect/core/Conversion/ArtsToLLVM/`
- Stage 14 (EpochLowering): `lib/arts/dialect/rt/Conversion/ArtsToRt/EpochLowering.cpp`
- Stage 15 (EdtLowering): `lib/arts/dialect/rt/Conversion/ArtsToRt/EdtLowering.cpp`
- Stage 16 (RtToLLVM): `lib/arts/dialect/rt/Conversion/RtToLLVM/RtToLLVMPatterns.cpp`
- Multinode-conditional passes: `DbDistributedOwnership`, `DbDistributedEligibility`, SDE distribution planning, Core DB refinement

## Anti-patterns from prior fixes

These are real cases where fixes landed in the wrong layer and caused regressions. Cross-check before applying any fix.

### Anti-pattern 1 — fixing partition mode instead of memref shape

17 samples failed on `memref<?xmemref<?xi32>>` (heap-allocated arrays). The error surfaced in `DbPartitioning::downgrade_to_coarse`. The temptation: patch the partition heuristic.

**The actual fix is upstream in shape normalization** (`CreateDbs` or `RaiseMemRefDimensionality`). Patching the partition heuristic would silently mask the wrapper-of-pointer shape and produce coarse-grained partitions for what should be fine-grained, dropping `partition_indices[%i]` and serializing tasks.

**Lesson:** when the error surface is in a heuristic but the input shape is wrong, the fix belongs in shape normalization, not in the heuristic gating logic.

### Anti-pattern 2 — fixing CPS attr at relaunch instead of at carry construction

`EpochOpt` CPS-8 carry re-analysis changed carry arity between EpochOpt and EpochLowering. A fix landed in `rebuildCpsPackToTargetSchema` (EpochLowering) by zero-filling missing slots. But `EpochOpt` never updated `CPSDepRouting` to match.

**Result:** outer epoch received wrong dep slot count; downstream dep forwarding used stale indices. Deadlock or race.

**Lesson:** layer accountability. EpochOpt owns carry construction + arity. EpochLowering only consumes contracts. If EpochOpt changes the contract post-CPS-8, it must update all downstream attributes BEFORE passing to EpochLowering. Fixing at relaunch time is too late.

### Anti-pattern 3 — gating a pass on IR structure instead of contract validity

`EdtLowering` inserted a pack-rewrite gate `NoOpDbRewriter` to skip lowering for allocations with explicit distributed leaf contracts. The gate checked a custom attr (`IsExplicitLeafContract`) stamped in stage 8 without proof.

**Result:** when an intermediate pass invalidated the contract by rewriting the allocation, the gate did not re-validate; EDT lowering emitted code that silently ignored the invalid contract.

**Lesson:** contracts must travel with proof. If a pass stamps a contract, downstream passes either consume it immediately or store a proof token. A gate that skips re-validation is brittle; later passes must re-check.

## When to delegate

The triage rubric tells you the originating layer. Once you know it, delegate the actual code work to the right specialist:

| Symptom class | Delegate to |
|---|---|
| Wrong output / silent miscompilation | `carts-miscompile-triage` |
| Failure only with `--distributed-db` or multinode | `carts-distributed-triage` |
| Behavior depends on pass order, stale graphs, metadata inconsistency | `carts-analysis-triage` |
| Crash, segfault, or generic compile error | `carts-debug` |
| Need to compare IR between two stages | `carts-stage-diff` |
| Runtime hang, wrong worker placement, ARTS scheduler issue | `carts-runtime-triage` |
