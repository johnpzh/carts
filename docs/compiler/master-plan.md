# CARTS Compiler Master Plan

This is the planning entry point for the SDE/CODIR/ARTS restructuring and the
large/64 performance recovery work. It coordinates the subplans under
[`plans/`](./plans/) and points to the live pipeline when implementation facts
change.

The live compiler still wins over documentation:

```bash
dekk carts pipeline --json
```

## North Star

The target stack is:

```text
Polygeist -> sde -> codir -> arts -> arts-rt -> LLVM
```

The stack has four non-negotiable contracts:

- SDE proves source semantics and authors the MU/CU/SU plan.
- CODIR materializes isolated codelets with complete deps, params, and
  token-local memref views.
- ARTS binds those codelets to abstract DB/EDT/epoch objects and ARTS-machine
  topology.
- ARTS-RT lowers the already-chosen object graph to runtime ABI calls.

The architecture must remove late rediscovery. CARTS should not infer DB roots,
owner dimensions, dependency windows, codelet captures, or tiling policy after
SDE/CODIR have enough information to say those things directly.

## Current Checkpoint

The current implementation still uses the transitional path:

```text
SDE planning -> ConvertSdeToArts direct MU/token lowering
             -> CreateDbs coarse raw bridge only
             -> ARTS lowering
```

Important current facts:

- `sde.cu_codelet` is a migration surface, not the final codelet layer.
- The Core raw DB indexers have been removed. `CreateDbs` may only materialize
  coarse whole-storage raw memrefs; a blocked/tiled raw memref reaching it is
  a boundary error because SDE/CODIR must own the token-local access rewrite.
- The tensor raising/lowering path is legacy. The target is memref-native MU
  tokens and codelet deps.
- The latest working-tree large/64 matrix run after removing DB-payload
  `memref.subview` creation is correctness-clean but slow:
  `gemm 0.371x`, `2mm 0.188x`, `3mm 0.190x`.
- Earlier focused runs showed that the matrix family can be much faster, so the
  next performance investigation should compare stage shape rather than add
  benchmark-specific constants.

## Subplan Map

| Subplan | Purpose | Primary Exit Gate |
|---|---|---|
| [`folder-reorganization.md`](./plans/folder-reorganization.md) | Establish the physical `carts/dialect/{sde,codir,arts,arts-rt}` source layout and staged move order. | Target folders exist and each physical move has focused build/test gates. |
| [`subdialect-analysis-optimization.md`](./plans/subdialect-analysis-optimization.md) | Give each subdialect local analysis and optimization ownership. | No dialect relies on hidden analysis objects from another dialect. |
| [`utility-ownership.md`](./plans/utility-ownership.md) | Give each dialect its own utility surface and prevent pass-local utility sprawl. | New helpers land in the narrowest owning Utils/Support area or carry a pass-local justification. |
| [`dialect-stack-migration.md`](./plans/dialect-stack-migration.md) | Split the conceptual and driver stack into SDE, CODIR, ARTS, and ARTS-RT. | The driver can expose target boundary stages without changing semantics. |
| [`codir-edt-isolation.md`](./plans/codir-edt-isolation.md) | Make codelet deps/params explicit and guarantee EDT isolation from above. | `EdtLowering` rejects implicit captures and lowers explicit dep/param ABI mechanically. |
| [`memref-mu-token-rewrite.md`](./plans/memref-mu-token-rewrite.md) | Make tiling real by rewriting MU, CU, and SU together at memref level. | ND/strided owner slices lower through token-local codelets without tensor fallback. |
| [`arts-materialization-cleanup.md`](./plans/arts-materialization-cleanup.md) | Convert CODIR directly to ARTS and remove legacy DB rediscovery paths. | Supported benchmarks no longer need raw-memref `CreateDbs` materialization. |
| [`performance-large64.md`](./plans/performance-large64.md) | Recover and stabilize benchmark performance at large/64. | Maintained benchmarks are correctness-clean and classified as fast, competitive, or blocked with owner. |
| [`verification-release.md`](./plans/verification-release.md) | Keep every migration phase testable and reversible by evidence. | Each milestone has focused lit, pipeline, e2e, and benchmark evidence. |

## Dependency Graph

```text
Folder reorganization
        |
        v
Subdialect analysis/optimization ownership
        |
        v
Dialect utility ownership
        |
        v
Dialect stack migration
        |
        v
CODIR + EDT isolation  <---- Memref MU/token rewrite
        |                         |
        +-----------+-------------+
                    v
       ARTS materialization cleanup
                    |
                    v
        Large/64 performance recovery

Verification and release gates run across every stage.
```

The graph allows local implementation in parallel, but not semantic shortcuts:

- Performance work may continue on the current transitional path only when it
  produces facts that will survive the CODIR split.
- ARTS cleanup may shrink compatibility code only after equivalent SDE/CODIR
  coverage exists.
- ARTS-RT work is allowed only after SDE/CODIR/ARTS object shape is correct and
  traces show runtime ABI overhead.

## Milestones

### Master Checklist

Use this as the top-level work queue. Each item points to the owning subplan
for the detailed checklist and exit gate.

- [ ] M0: checkpoint the current tree and record baseline evidence.
- [ ] M1: make the target folder/dialect skeleton visible.
- [ ] M2: create CODIR and enforce isolated deps/params.
- [ ] M3: make memref MU/CU/SU tiling rewrite real.
- [ ] M4: remove legacy ARTS DB rediscovery.
- [ ] M5: recover large/64 performance across maintained benchmarks.
- [ ] Cross-cutting: enforce dialect-local Utils ownership before every pass
  edit.
- [ ] Cross-cutting: update docs and generated skills whenever boundaries,
  commands, or ownership rules change.

### M0: Checkpoint And Baseline

Goal: make the current state easy to restart from.

Tasks:

- Keep the no-subview correctness fix and current benchmark evidence recorded.
- Create focused sample and pipeline dumps for one GEMM-family case under
  `.carts/sessions/...`.
- Confirm `dekk carts build` and focused lit tests still pass before deeper
  migration.

Exit gate:

- `git diff --check`
- focused SDE/ARTS lit tests for current changed files
- one current matrix-family pipeline dump bundle

### M1: Stack Boundary Skeleton

Goal: make the target split visible without moving behavior prematurely.

Tasks:

- Add CODIR dialect skeleton or an equivalent staged namespace wrapper.
- Add `Utils/` folders for each dialect next to `IR/`, `Analysis/`,
  `Transforms/`, `Conversion/`, and `Verify/`.
- Start moving files into the `carts/dialect/...` target layout only for the
  dialect slice being implemented.
- Add driver stage names or internal pass groups for SDE-to-CODIR and
  CODIR-to-ARTS.
- Keep current `ConvertSdeToArts` behavior behind compatibility staging until
  direct CODIR lowering exists.

Exit gate:

- pipeline manifest documents current and target stage groups clearly;
- no benchmark or e2e behavior changes from skeleton-only work.

### M2: CODIR Isolation And EDT ABI

Goal: make implicit captures impossible before ARTS-RT lowering.

Tasks:

- Move codelet boundary ownership out of SDE into CODIR.
- Introduce CODIR verification for deps, params, yielded values, and no
  implicit above captures.
- Create ARTS EDTs with complete dep/param lists.
- Simplify `EdtLowering` around explicit dep/param ABI.

Exit gate:

- negative tests fail on implicit codelet/EDT capture;
- positive tests lower memory deps and scalar params without fallback capture
  recovery.

### M3: Memref MU/CU/SU Rewrite

Goal: make tiling a real data-layout transformation.

Tasks:

- Keep `PatternAnalysis` as the SDE fact pass.
- Put reusable SDE memref/access helpers in SDE `Utils/`, not inside the
  transformation pass that first needs them.
- Materialize MU roots and MU tokens for function args, globals, allocas,
  task deps, reductions, and intermediates.
- Rewrite codelet loads/stores to token-local memref coordinates for ND,
  strided, and halo windows.
- Remove tensor raising/lowering once memref coverage is complete.

Exit gate:

- selected ND/strided tests lower through memref tokens;
- no `RaiseMemrefToTensor`, `RaiseToTensor`, `LowerToMemref`, or tensor-only
  cleanup is needed for supported cases.

### M4: Direct ARTS Materialization

Goal: remove late DB rediscovery.

Tasks:

- Lower CODIR deps directly to `arts.db_acquire`.
- Lower MU storage directly to `arts.db_alloc`.
- Lower CODIR codelets directly to `arts.edt`.
- Keep `CreateDbs` limited to coarse raw-memref materialization, then remove it
  when direct MU/token coverage is complete.

Exit gate:

- no `arts.db_control` operation or equivalent marker;
- no supported benchmark relies on raw-memref DB root discovery;
- blocked/tiled raw memrefs reaching `CreateDbs` fail with unsupported
  diagnostics instead of being reindexed in ARTS.

### M5: Large/64 Performance Recovery

Goal: make every maintained benchmark correctness-clean and
performance-credible at large/64.

Tasks:

- Reproduce current slow matrix-family results and compare stage shape against
  prior fast runs.
- Implement SDE contraction phase plans for `gemm`, `2mm`, and `3mm` without
  hardcoded benchmark constants.
- Add generic CODIR/ARTS-RT cleanup only after codelet shape is correct.
- Sweep all maintained benchmarks and record current classifications.

Exit gate:

- all runnable maintained benchmarks compile and pass checksum;
- each benchmark is classified as fast, competitive, or blocked with an owning
  layer and next action.

## Immediate Next Session

Start with these tasks:

1. Use [`utility-ownership.md`](./plans/utility-ownership.md) and the
   `carts-check-utils` skill before adding or moving helper code.
2. Build and run focused lit for the current code changes.
3. Create a small GEMM sample/session dump with all relevant pipeline stages
   under `.carts/sessions/...`.
4. Compare the no-subview slow shape against the earlier fast and median-fast
   matrix runs.
5. Pick the first implementation slice from the checklist below.

## Implementation Task Board

### A. Planning And Documentation

- [ ] Update `docs/compiler/dialects/*` whenever a dialect responsibility
  changes.
- [ ] Keep `docs/compiler/pipeline.md` synchronized with
  `tools/compile/Compile.cpp`.
- [ ] Keep `carts-plugin/project.md` and generated agent resources aligned
  with this master plan.
- [ ] Refresh `dekk carts skills generate` after docs or skill changes.

### B. Folder And Utility Ownership

- [ ] Add or verify `Utils/` folders for SDE, CODIR, ARTS, and ARTS-RT.
- [ ] Move reusable helpers to the earliest owning dialect utility surface.
- [ ] Keep pass-local static helpers only when they are genuinely local to one
  pass and not a dialect invariant.
- [ ] Delete duplicated local helpers after extraction.
- [ ] Update include paths and CMake in small, buildable slices.

### C. CODIR Boundary

- [ ] Define CODIR IR for isolated codelets, deps, params, and token-local
  memref views.
- [ ] Move codelet-specific SDE verification into CODIR verification.
- [ ] Add negative tests for implicit captures.
- [ ] Add positive tests for explicit memory deps and scalar params.

### D. Memref MU/CU/SU Rewrite

- [ ] Extend PatternAnalysis for ND, strided, contraction, reduction, and halo
  facts at memref level.
- [ ] Materialize MU storage for args, globals, allocas, task deps,
  reductions, and intermediates.
- [ ] Rewrite codelet loads/stores to token-local coordinates before ARTS.
- [ ] Remove tensor fallback only after equivalent memref tests pass.

### E. ARTS And ARTS-RT Cleanup

- [ ] Lower CODIR deps directly to `arts.db_acquire`.
- [ ] Lower MU storage directly to `arts.db_alloc`.
- [ ] Lower CODIR codelets directly to `arts.edt`.
- [ ] Remove coarse `CreateDbs` after direct coverage exists.
- [ ] Keep runtime ABI and pointer packing cleanup in ARTS-RT.

### F. Performance Evidence

- [ ] Re-run large/64 matrix-family benchmarks after each materialization
  slice.
- [ ] Compare task count, DB count, acquire modes, dep windows, and runtime
  counters against fast controls.
- [ ] Sweep maintained benchmarks before declaring the migration stable.
- [ ] Record blocked benchmarks with owning layer and next action.

## Rules

- Do not add hardcoded benchmark constants to passes.
- Do not add SDE references to ARTS runtime topology.
- Do not make ARTS infer source-level owner dims or dependency-window legality.
- Do not pass hidden C++ analysis objects across dialect boundaries; materialize
  required facts in IR.
- Do not keep tensor fallback once memref MU/token/CODIR coverage exists.
- Do not remove legacy code until the replacement path has focused tests and at
  least one affected e2e or benchmark proof.
- Put generated artifacts under `.carts/outputs/...` or
  `.carts/sessions/...`; do not commit them unless promoted to real tests.
