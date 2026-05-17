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
| `codir-to-arts` | `SDE operation ... survived past SDE lowering` | `ConvertOpenMPToSde`, `sde-planning`, `ConvertSdeToCodir` | Which SDE op did not become a CODIR codelet or explicit CODIR boundary object? |
| `sde-input-normalization` / `create-dbs` | `cannot trace memref operand to its underlying allocation` | `SdeMemrefNormalization`, `SdeHandleDeps`, `CreateDbs` | Memref-of-memref reaching CreateDbs from a heap allocation. |
| `create-dbs` | `un-normalizable nested memref pattern (element type is memref)` | `SdeMemrefNormalization`, `ConvertOpenMPToSde` | Polygeist producing `memref<?xmemref<?xT>>` from `int *A = malloc(N)`. Fix upstream. |
| `create-dbs` / `post-db-refinement` | Metadata-copy recursion or DB-mode churn | Live `copyArtsMetadataAttrs` call sites, `DbAnalysis`, DB refinement rewrites | Fix the producer or rewrite that invalidates the contract. Do not recreate the retired monolithic partitioning layer. |
| `create-dbs` / `post-db-refinement` | Wrong answer; a raw memref bridge silently becomes one coarse wrapper DB, indices dropped | SDE/CODIR shape normalization or direct CODIR-to-ARTS materialization | Single-element wrapper DB instead of N-element data. The fix is upstream in shape/token-local materialization; `CreateDbs` is only a guarded coarse raw bridge. |
| `post-db-refinement` | Stencil halo bounds wrong | SDE access-window plan, DB refinement contract rewrite | Confirm SDE stamped the right halo/window, then check whether DB refinement rewrote it. |
| `pre-lowering` / `arts-rt-to-llvm` | `arts.db_alloc` / `arts.edt` / `arts.epoch` survived to LLVM | DbLowering, EdtLowering, EpochLowering | A lowering was skipped. Check the owning lowering and any contract-validity gate that excluded this op. |
| `pre-lowering` | `arts.db_acquire` references GUID that does not exist | DB refinement (GUID lost), CODIR-to-ARTS materialization for codelet deps, EpochLowering (CPS carry corruption) | Trace the GUID source. Did an intermediate pass erase it? |
| `pre-lowering` | CPS chain: wrong iteration counter or outer epoch GUID | `EpochOpt` CPS-8 carry re-analysis, `EpochLowering` propagation | Check `CPSParamPerm` and `CPSIterCounterParamIdx` post-EpochOpt. |
| `pre-lowering` | `CPS advance: rebuilt continuation pack with N schema holes` | `EpochOpt` carry analysis, intermediate EDT fusion, `EdtLowering` pack ordering | Carry arity changed between EpochOpt and EpochLowering. Zero-filled slots carry garbage. |
| `post-db-refinement` (distributed) | Stencil halo not applied, internode acquire uses full range | SDE window contract mismatch, ARTS distributed ownership attr missing | Is `distributed` attr on the DB? Did ARTS select distributed ownership after consuming the SDE/CODIR window contract? |
| `post-db-refinement` (distributed) | Task hangs on remote data acquire | Dependency-window narrowing, DB refinement scope-aware bounds | A dep window may have narrowed to local-only; distributed task cannot reach halo. |
| `pre-lowering` | `arts_rt.edt_create` argument count mismatch | `EdtLowering` pack construction, prior DB/EDT rewrites invalidating pack operand | `EdtParamPackOp` was rewritten; `EdtCreateOp` references old pack. |
| Post-LLVM | Wrong value in loop or task parameter (silent miscompute) | `EpochLowering` CPS carry slot corruption, `EdtLowering` pack slot reordering | Run `dekk carts compile --all-pipelines` and inspect `pipelines/4_rt/` for the pack structure. |

## Verification barrier ladder

Each verification pass is a freeze point. If barrier X passes but barrier Y fails, the bug is in stages between them. Run `dekk carts compile <file> --all-pipelines -o /tmp/stages/` then inspect each stage dump.

| Barrier | File | Asserts | Failure severity |
|---|---|---|---|
| `VerifySdeLowered` | `lib/carts/dialect/sde/Verify/VerifySdeLowered.cpp` | No `sde.*` ops survive `codir-to-arts`. No transient linalg/tensor carriers survive the SDE-to-CODIR / CODIR-to-ARTS boundary. | Fatal |
| `VerifyArtsObjectsOnly` | `lib/carts/dialect/arts/Transforms/verify/VerifyArtsObjectsOnly.cpp` | No source semantic carrier survives `codir-to-arts`. ARTS contains runtime-shaped EDT/DB/epoch objects plus implementation `scf.for`. | Fatal |
| `VerifyEdtCreated` | `lib/carts/dialect/arts/Transforms/verify/VerifyEdtCreated.cpp` | At least one `arts.edt` exists post-OpenMP conversion. | Warning |
| `VerifyDbLowered` | `lib/carts/dialect/arts-rt/Transforms/VerifyDbLowered.cpp` | No `arts.db_alloc` / `db_acquire` / `db_release` survive pre-lowering. | Fatal |
| `VerifyEpochLowered` | `lib/carts/dialect/arts-rt/Transforms/VerifyEpochLowered.cpp` | No `arts.epoch` survives pre-lowering. All become `arts_rt.create_epoch` + `wait_on_epoch`. | Fatal |
| `VerifyEdtLowered` | `lib/carts/dialect/arts-rt/Transforms/VerifyEdtLowered.cpp` | No `arts.edt` survives pre-lowering. All lower to `arts_rt.edt_create` + pack. | Fatal |
| `VerifyPreLowered` | `lib/carts/dialect/arts-rt/Transforms/VerifyPreLowered.cpp` | IR ready for codegen. Param packs, dep routing, CPS attrs all consistent. | Fatal |
| `VerifyLowered` | `lib/carts/dialect/arts-rt/Transforms/VerifyLowered.cpp` | No `arts.*` or `arts_rt.*` ops survive post-LLVM lowering. | Fatal |

## Stage-to-file map

Use to locate where to grep when triaging.

- `sde-input-normalization` / `sde-planning`: `lib/carts/dialect/sde/Transforms/` + `lib/carts/dialect/sde/Conversion/OmpToSde/`
- `sde-to-codir`: `lib/carts/dialect/codir/Conversion/SdeToCodir/SdeToCodir.cpp`
- `codir-to-arts`: `lib/carts/dialect/codir/Conversion/CodirToArts/CodirToArts.cpp`
- `create-dbs`: `lib/carts/dialect/arts/Transforms/db/CreateDbs.cpp`
- `db-opt` / `post-db-refinement`: `lib/carts/dialect/arts/Transforms/db/`, `lib/carts/dialect/arts/Analysis/db/`, `lib/carts/dialect/arts/Analysis/heuristics/DbHeuristics.cpp`
- CODIR-to-ARTS materialization: `lib/carts/dialect/codir/Conversion/CodirToArts/CodirToArts.cpp`
- SDE distribution/reduction planning: `lib/carts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`, `lib/carts/dialect/sde/Transforms/effect/scheduling/ReductionStrategy.cpp`
- ARTS-RT ABI conversion: `pre-lowering` implementation in `lib/carts/dialect/arts-rt/Conversion/ArtsToRt/` and LLVM lowering in `lib/carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/`
- ARTS-RT LLVM cleanup: `lib/carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/ArtsRtOpToLLVMPatterns.cpp`
- Multinode-conditional passes: `DbDistributedOwnership`, `DbDistributedEligibility`, SDE distribution planning, ARTS DB refinement

## Anti-patterns from prior fixes

These are real cases where fixes landed in the wrong layer and caused regressions. Cross-check before applying any fix.

### Anti-pattern 1 — fixing partition mode instead of memref shape

17 historical samples failed on `memref<?xmemref<?xi32>>` (heap-allocated arrays). The error surfaced downstream in DB refinement. The temptation was to patch partition-mode selection.

**The actual fix is upstream in SDE shape normalization** (`SdeMemrefNormalization` or `ConvertOpenMPToSde`) or in direct CODIR-to-ARTS token-local materialization. Patching DB mode selection would silently mask the wrapper-of-pointer shape and produce coarse-grained partitions for what should be fine-grained, dropping partition indices and serializing tasks.

**Lesson:** when the error surface is in a heuristic but the input shape is wrong, the fix belongs in shape normalization, not in the heuristic gating logic.

### Anti-pattern 2 — fixing CPS attr at relaunch instead of at carry construction

`EpochOpt` CPS-8 carry re-analysis changed carry arity between EpochOpt and EpochLowering. A fix landed in `rebuildCpsPackToTargetSchema` (EpochLowering) by zero-filling missing slots. But `EpochOpt` never updated `CPSDepRouting` to match.

**Result:** outer epoch received wrong dep slot count; downstream dep forwarding used stale indices. Deadlock or race.

**Lesson:** layer accountability. EpochOpt owns carry construction + arity. EpochLowering only consumes contracts. If EpochOpt changes the contract post-CPS-8, it must update all downstream attributes BEFORE passing to EpochLowering. Fixing at relaunch time is too late.

### Anti-pattern 3 — gating a pass on IR structure instead of contract validity

`EdtLowering` inserted a pack-rewrite gate to skip lowering for allocations with explicit distributed leaf contracts. The gate checked a custom attr (`IsExplicitLeafContract`) stamped before `post-db-refinement` without proof.

**Result:** when an intermediate pass invalidated the contract by rewriting the allocation, the gate did not re-validate; EDT lowering emitted code that silently ignored the invalid contract.

**Lesson:** contracts must travel with proof. If a pass stamps a contract, downstream passes either consume it immediately or store a proof token. A gate that skips re-validation is brittle; later passes must re-check.

### Anti-pattern 4 — rediscovering DB dependencies in ARTS

ARTS no longer has a dependency-marker operation. If a failure involves
unmaterialized `sde.mu_dep`, missing DB acquires, or a raw memref body that was
not localized, do not add new ARTS rediscovery logic. The originating issue is
that SDE did not produce canonical `mu_data`/`mu_token`/`cu_codelet` form before
the SDE-to-CODIR / CODIR-to-ARTS boundary.

**Lesson:** user dependency slices are SDE MU facts. ARTS may bind them to DBs,
but it should not infer them by rescanning task bodies.

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
