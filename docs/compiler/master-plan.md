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

Concurrency follows the same ownership rule. SDE and CODIR may reason about
generic logical workers and codelet-local work shape, but they must not observe
node count, workers per node, local-vs-multinode placement, routes, or runtime
replication policy. The first dialect allowed to bind those logical workers to
single-node or multinode ARTS topology is ARTS; ARTS-RT receives that decision
as lowering-ready runtime shape.

The architecture must remove late rediscovery. CARTS should not infer DB roots,
owner dimensions, dependency windows, codelet captures, or tiling policy after
SDE/CODIR have enough information to say those things directly.

## Current Checkpoint

_Snapshot as of 2026-05-16. Refresh whenever a milestone closes or the live
pipeline changes._

**Source layout:** Migration to the unified `carts/` tree is complete.
`include/arts/` and `lib/arts/` no longer exist. Headers and sources live at:

```text
include/carts/{Dialect.h, dialect/{sde,codir,arts,arts-rt}/, passes/, utils/}
lib/carts/    {                dialect/{sde,codir,arts,arts-rt}/, passes/, utils/}
```

`utils/` contains CARTS-shared utilities; each dialect's `Utils/` contains its
private utilities (see [`utility-ownership.md`](./plans/utility-ownership.md)).

**C++ namespaces:** unified under `mlir::carts`. `mlir::carts::sde`,
`mlir::carts::arts`, `mlir::carts::arts_rt`, `mlir::carts::codir`. The legacy
`mlir::arts` (which used to mean both the project umbrella AND the ARTS
dialect) is gone.

**CMake targets:** `MLIRCartsArts`, `MLIRCartsArtsRt`, `MLIRCartsSde`,
`MLIRCartsCodir{,Conversion,Transforms,Utils}` for dialect IR; per-dialect
pass libraries `MLIRCartsSdeTransforms`, `MLIRCartsArtsTransforms`,
`MLIRCartsArtsRtTransforms`; thin umbrella `MLIRCartsTransforms` aggregates
the three.

Build + 170/170 lit + 27/27 e2e green.

The current implementation uses the production codelet path:

```text
SDE planning -> CODIR codelet isolation -> CODIR-to-ARTS DB/EDT materialization
             -> CreateDbs coarse raw bridge for ordinary memrefs only
             -> ARTS lowering
```

The live core stage tokens (from `tools/compile/Compile.cpp`) are:

```text
sde-input-normalization, initial-cleanup, sde-planning, sde-to-codir,
codir-to-arts, edt-transforms, create-dbs, db-opt, post-db-refinement,
late-concurrency-cleanup, epochs, pre-lowering, arts-rt-to-llvm
```

Plus the `post-o3-opt` and `llvm-ir-emission` epilogues. The `sde-to-codir`
and `codir-to-arts` stages run by default. The pipeline manifest marks the
SDE-to-CODIR, CODIR, and
CODIR-to-ARTS bridge groups as current; they are no longer planned-only
metadata.

Important current facts:

- `sde.cu_codelet` is a source-side planning surface, not the ARTS lowering
  boundary. Codelet-shaped work must route through CODIR. Any SDE operation
  that survives past `sde-to-codir` is a boundary failure, not an ARTS EDT
  producer.
- The CODIR-to-ARTS verifier in the live staged pipeline is
  `VerifyArtsObjectsOnly`; old verifier entry points have been removed.
- `arts.edt` no longer has a verifier-bypass attribute. Unsupported task
  shapes must fail at the owning boundary; tests cannot keep invalid ARTS IR
  alive by marking it unchecked.
- CODIR is now active in the default pipeline:
  - Registered dialect under `include/carts/dialect/codir/IR` and
    `lib/carts/dialect/codir/IR`, with parser/verifier coverage for an
    isolated `codir.codelet` boundary.
  - `convert-sde-to-codir`, `codir-codelet-opt`, `verify-codir`, and
    `convert-codir-to-arts` are default stage passes.
  - `verify-codir` enforces: dep operands are memrefs with one access-mode
    entry per dep; params are scalar values; body block args mirror deps then
    params; yielded values are scalar; and codelet bodies do not implicitly
    capture values from above.
  - `convert-sde-to-codir` materializes `sde.cu_codelet` into `codir.codelet`,
    including whole-storage MU tokens and sliced token-local views represented
    as CODIR dep memrefs.
  - For the first M3 token-local rewrite slice, `convert-sde-to-codir`
    rewrites direct `memref.load`/`memref.store` indices on rank-1 and ND
    sliced `sde.mu_token` deps from captured global coordinates to token-local
    coordinates. Constant and dynamic token offsets are covered; dynamic
    offsets are carried as explicit CODIR scalar params so the body remains
    isolated.
  - The task-depend M3 slice now handles `sde.cu_task` bodies with a sliced
    `sde.mu_dep` source, complete static rank-1/ND slice bounds, direct
    `memref.load`/`memref.store` uses of that source, and body-local
    `memref.subview`/`polygeist.subindex` views rooted in that source when the
    view result is used only by direct loads/stores. Duplicate deps for the
    same source reuse a single CODIR view when their static slices are
    identical and merge access modes at that dep slot. Different slices from
    the same source now become distinct positional CODIR deps only for exact
    body-view proof cases: every source-rooted body use must be a unit-stride
    `memref.subview` matching one of the `sde.mu_dep` windows, a row
    `polygeist.subindex` matching a unit first-dimension dep window with
    trailing full-row bounds, or a direct root access whose indices exactly
    match one dep offset vector. Matched view results must be used only by
    direct loads/stores, and matched root accesses rewrite to local zero. A
    narrow dynamic slice case is covered when the
    task body proves the window either by using only source-rooted
    `memref.subview` ops whose offsets/sizes exactly match the `sde.mu_dep`
    bounds and whose strides are unit, or by using direct root load/store
    indices that exactly match the dynamic `sde.mu_dep` offsets. General
    dynamic task slice bounds and mixed-slice direct root accesses with
    non-exact indices remain guarded because source task-depend slices can be
    synchronization tokens rather than complete access windows; enabling them
    safely requires a precise token-local access-window proof.
  - `convert-codir-to-arts` lowers CODIR codelets to `arts.db_acquire` plus
    `arts.edt`.
  - `codir-codelet-opt` performs CODIR-owned cleanup of isolated codelet bodies
    after token-local rewrites and before ARTS materialization, so dead
    rematerialized constants/views do not rely on later ARTS cleanup.
  - CODIR-owned plan attributes live in `CodirAttrs.td` and on
    `codir.codelet`; CODIR-to-ARTS translates them mechanically to ARTS
    attributes, including distribution kind, repetition/async strategy,
    owner/logical slice metadata, stencil min/max offsets, stencil
    owner/spatial dims, write footprint, and supported block-halo markers.
  - Reusable CODIR-only ABI predicates live in
    `carts/dialect/codir/Utils/CodeletABIUtils`. SDE-dependent
    task-dependency slice proof helpers live under
    `lib/carts/dialect/codir/Conversion/SdeToCodir/TaskDepSliceUtils.*`
    instead of `codir/Utils`, so the CODIR dialect utility library no longer
    depends on SDE.
- The Core raw DB indexers have been removed. `CreateDbs` may only materialize
  coarse whole-storage raw memrefs; a blocked/tiled raw memref reaching it is
  a boundary error because SDE/CODIR must own the token-local access rewrite.
- `verify-arts-objects-only` now also guards the distributed boundary: an
  `internode` ARTS task may not depend on a coarse single-block aggregate user
  DB. If that happens, the fix belongs in SDE/CODIR MU/token/codelet
  materialization, not in late ARTS `CreateDbs` recovery.
- CODIR may carry generic scheduling/distribution intent before storage is
  ready, but CODIR-to-ARTS only turns that intent into `internode` EDT routing
  when generic tile owner/shape storage metadata is present (or the codelet has
  no DB dependencies). Planned-but-unmaterialized user DBs stay local instead
  of creating remote coarse DB traffic.
- The tensor raising/lowering passes (`RaiseMemrefToTensor`, `RaiseToTensor`,
  `LowerToMemref`) are no longer in any live pass list, but the source still
  lives under `lib/carts/dialect/sde/Transforms/state/raising/`. M3
  source-level removal is still pending. The M3 target is memref-native MU
  tokens and codelet deps. (Note: `SdeInputNormalization` in the
  `sde-input-normalization` stage is a separate pass that is part of the
  live pipeline and is not slated for removal.)
- The ARTS lowering target ops are `arts.db_alloc`, `arts.db_acquire`, and
  `arts.edt`. The removed `arts.db_control` op does not exist in the current
  dialect; lowering-contract markers are tracked via `arts.lowering_contract`
  (`LoweringContractOp`).
- ARTS-RT owns the runtime ABI lowering sources. `DbLowering`,
  `EdtLowering`, `EpochLowering`, and `ConvertArtsRtToLLVM` are built from
  `lib/carts/dialect/arts-rt/Conversion/`; the staged pipeline token is
  `arts-rt-to-llvm` and no legacy stage alias is accepted.
- The latest maintained large/64 sweep (2026-05-16) configured 23 benchmark
  entries and passed all 69 median-run executions checksum-clean:
  `.carts/outputs/benchmarks-m6-final-current-20260516/20260516_054254`.
  Runner-reported geometric mean kernel speedup is `1.70x`. STREAM was noisy
  in that full sweep, so the final table uses the superseding 7-run focused
  rerun:
  `.carts/outputs/benchmarks-m6-stream-final-rerun-20260516/20260516_055824`.
  Current audited classes are 21 fast, 2 competitive, and 0 blocked. Host
  OpenMP fallback is intentionally limited to repeated 2-D stencils and
  benchmark-scoped multi-map 1-D floating-point bundles where ARTS epoch
  overhead would dominate repeated host streaming work; `CreateDbs` remains
  guarded against raw blocked layouts and is not allowed to rediscover
  SDE/CODIR layout. See
  [`benchmark-performance-goal.md`](./benchmark-performance-goal.md) and
  [`performance-large64.md`](./plans/performance-large64.md) for result
  directories and the current per-benchmark table.

## Subplan Map

| Subplan | Purpose | Primary Exit Gate |
|---|---|---|
| [`folder-reorganization.md`](./plans/folder-reorganization.md) | Establish the physical `carts/dialect/{sde,codir,arts,arts-rt}` source layout and staged move order. | **Done** as of 2026-05-15: all moves landed; build + 163/163 lit tests green. |
| [`subdialect-analysis-optimization.md`](./plans/subdialect-analysis-optimization.md) | Give each subdialect local analysis and optimization ownership. | No dialect relies on hidden analysis objects from another dialect. |
| [`utility-ownership.md`](./plans/utility-ownership.md) | Give each dialect its own utility surface and prevent pass-local utility sprawl. | **Done for the cross-dialect blob** as of 2026-05-15. Pass-local SDE/ARTS-RT extractions remain as a watch item. New helpers land in the narrowest owning dialect `Utils/` (or `carts/utils` for shared needs) or carry a pass-local justification. |
| [`dialect-stack-migration.md`](./plans/dialect-stack-migration.md) | Split the conceptual and driver stack into SDE, CODIR, ARTS, and ARTS-RT. | The driver runs SDE -> CODIR -> ARTS; any surviving SDE op fails the boundary. |
| [`codir-edt-isolation.md`](./plans/codir-edt-isolation.md) | Make codelet deps/params explicit, guarantee CODIR isolation from above, then lower mechanically to `arts.edt`. | `verify-codir` rejects implicit captures and `EdtLowering` consumes the explicit dep/param ABI without recovery code. |
| [`memref-mu-token-rewrite.md`](./plans/memref-mu-token-rewrite.md) | Make tiling real by rewriting MU, CU, and SU together at memref level. | ND/strided owner slices lower through token-local codelets without tensor-carrier paths. |
| [`arts-materialization-cleanup.md`](./plans/arts-materialization-cleanup.md) | Convert CODIR directly to ARTS and remove DB rediscovery paths. | Supported benchmarks no longer need raw-memref `CreateDbs` materialization. |
| [`performance-large64.md`](./plans/performance-large64.md) | Recover and stabilize benchmark performance at large/64. | Maintained benchmarks are correctness-clean and classified as fast or competitive. |
| [`verification-release.md`](./plans/verification-release.md) | Keep every migration phase testable and reversible by evidence. | Each milestone has focused lit, pipeline, e2e, and benchmark evidence. |

For the M6 post-restructuring validation+optimization loop see the
[M6](#m6-post-restructuring-validation--optimization-loop) milestone section
below; it consumes evidence from `performance-large64.md` and routes each
performance gap through the four-layer ownership model defined in
[`benchmark-performance-goal.md`](./benchmark-performance-goal.md).

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
CODIR + EDT isolation
        |
        v
Memref MU/CU/SU rewrite       (consumes codir.codelet body)
        |
        v
Post-restructuring validation+optimization loop (M6)   (consumes M5 evidence,
                                                        attributes perf gaps
                                                        to owning dialect)
        |
        v
ARTS materialization cleanup
        |
        v
Large/64 performance recovery (may run in parallel on the
                               transitional path; see rule below)

Verification and release gates run across every stage.
```

The graph allows local implementation in parallel, but not semantic shortcuts:

- Performance work must preserve the production SDE -> CODIR -> ARTS codelet
  path. `CreateDbs` remains only as the coarse raw-memref bridge for ordinary
  memrefs that have not yet been promoted into MU/token storage.
- ARTS cleanup may shrink `CreateDbs` only after equivalent SDE/CODIR coverage
  exists.
- ARTS-RT work is allowed only after SDE/CODIR/ARTS object shape is correct and
  traces show runtime ABI overhead.

## Milestones

### Milestone Index

Top-level work queue. Each `M` corresponds to a Milestone section below. The
Implementation Task Board (sections A–H) cross-cuts these by component and
each Task Board bullet is tagged with the milestones it advances.

Status legend: `[x]` complete; `[~]` partial; `[ ]` not started.

- [x] M0: baseline recorded (see [Current Checkpoint](#current-checkpoint));
  refresh after each milestone closes.
- [x] M1: target folder/dialect skeleton in place. The full physical layout
  migration is complete. `include/arts/` and `lib/arts/` are gone; everything
  lives under `include/carts/{dialect/{sde,codir,arts,arts-rt}, passes, utils,
  Dialect.h}` and `lib/carts/` mirrors it. Subdir `core/` was renamed to
  `arts/`; `rt/` to `arts-rt/`. CMake targets renamed to `MLIRCartsArts`,
  `MLIRCartsArtsRt`, `MLIRCartsTransforms` (umbrella) plus per-dialect
  transform libraries `MLIRCartsSdeTransforms`, `MLIRCartsArtsTransforms`,
  `MLIRCartsArtsRtTransforms` (split from the former monolith). Utility
  hierarchy is reclassified: shared utils stay in `carts/utils/`,
  dialect-specific utils live under each dialect's `Utils/`. C++ namespaces
  are unified: `mlir::carts::{sde,arts,arts_rt,codir}` (no more ambiguous
  `mlir::arts` for both project umbrella and ARTS dialect). See
  [`folder-reorganization.md`](./plans/folder-reorganization.md) and
  [`utility-ownership.md`](./plans/utility-ownership.md) for details.
- [x] M2: CODIR isolation enforced and codelet ownership migrated onto the
  `sde-planning -> sde-to-codir -> codir-to-arts` path; direct
  SDE-to-ARTS codelet lowering is gone.
- [ ] M3: memref MU/CU/SU tiling rewrite real. Current completed slices:
  sliced `sde.mu_token` rank-1/ND local rewrites, static rank-1/ND
  `sde.mu_dep` task local rewrites including body-local
  `memref.subview`/`polygeist.subindex` views, identical duplicate-source
  static task-dep slice reuse, exact body-local dynamic subview task-dep
  rewrites, mixed-window duplicate-source task deps when the body uses exact
  matching subviews or exact row `polygeist.subindex` views, exact-offset
  mixed-window direct root access rewrites, exact-offset dynamic direct root
  access rewrites, and dynamic sliced-dep scalar plumbing for CODIR/EDT
  rematerialization. General dynamic task slice bounds, mixed-window direct
  root accesses with non-exact indices,
  halo/strided access maps, OpenMP reduction materialization before CODIR,
  opaque full-memref calls inside token-local codelets, and tensor-carrier
  source removal remain open.
- [x] M4: EDTs are created from CODIR codelets with explicit ARTS EDT scalar
  params. Scalar capture recovery and CODIR-origin ABI marker attrs have been
  removed; `arts.edt` now rejects nonconstant scalar above-captures
  universally. The `no_verify` verifier-bypass attribute has been deleted from
  ARTS. Remaining ARTS cleanup is raw-memref DB materialization removal once
  MU/token coverage allows.
- [~] M5: large/64 performance recovered across maintained benchmarks.
  The focused matrix-family task-shape regression is fixed and verified; root
  DB strip materialization remains M3 work when whole-array host accesses share
  the same memref.
  The 12 raw-layout boundary failures from the maintained sweep now pass with
  checksum parity after SDE demotes unsupported physical storage attrs before
  the raw `CreateDbs` bridge. The current M6 evidence has no maintained
  benchmark below the large/64 performance gate; remaining M5-adjacent work is
  M3/M7 production coverage that replaces narrow host OpenMP fallbacks with
  tokenized SDE/CODIR/ARTS plans.
- [x] M6: post-restructuring validation + optimization loop. Drives the
  remaining performance work to convergence using the four-layer ownership
  model now that the M1 path/namespace/library refactor is in place. Re-run
  samples + benchmarks, triage failures to the owning dialect (SDE / CODIR /
  ARTS / ARTS-RT), apply the fix in that dialect, and iterate until every
  maintained benchmark is fast or competitive with zero blocked entries.
  Cross-references M5 perf evidence and
  [`benchmark-performance-goal.md`](./benchmark-performance-goal.md) for the
  per-dialect optimization decision tree.
- Cross-cutting:
  - [ ] Verification gates ([`verification-release.md`](./plans/verification-release.md))
    track every milestone with focused lit, pipeline, e2e, and benchmark
    evidence.
  - [ ] Subdialect analysis/optimization ownership
    ([`subdialect-analysis-optimization.md`](./plans/subdialect-analysis-optimization.md))
    advances together with M1–M2.
  - [ ] Documentation and generated skills are refreshed whenever boundaries,
    commands, or ownership rules change.

### M0: Checkpoint And Baseline

Goal: make the current state easy to restart from.

Tasks:

- Tag the baseline commit and capture pipeline + benchmark dumps under
  `.carts/sessions/<YYYYMMDD-HHMMSS>-baseline/`.
- Create focused sample and pipeline dumps for one GEMM-family case under
  `.carts/sessions/...`.
- Confirm `dekk carts build` and focused lit tests still pass before deeper
  migration.
- Refresh after each milestone closes.

Exit gate:

- `git diff --check`
- focused SDE/ARTS lit tests for current changed files
- one current matrix-family pipeline dump bundle

### M1: Stack Boundary Skeleton

Goal: make the target split visible without moving behavior prematurely.

Tasks:

- [x] CODIR dialect skeleton registered under
  `include/carts/dialect/codir/` and `lib/carts/dialect/codir/`.
- [x] Driver stage tokens `sde-planning`, `sde-to-codir`, and `codir-to-arts`
  are executable default stages and exposed by
  `--print-pipeline-manifest-json`.
- [x] Add `Utils/` folders for each dialect next to `IR/`, `Analysis/`,
  `Transforms/`, `Conversion/`, and `Verify/`. All four are now populated
  with real headers (not README-only).
- [x] Move all source into the `carts/dialect/...` target layout. The legacy
  `include/arts/` and `lib/arts/` umbrellas are gone. `core/` → `arts/`,
  `rt/` → `arts-rt/`. CARTS-shared utils sit at `carts/utils/`;
  dialect-specific utils sit under each dialect's `Utils/`.
- [x] Wire executable stage descriptors that run `convert-sde-to-codir`,
  `verify-codir`, and `convert-codir-to-arts` inside the default pipeline.
- [x] Remove direct SDE-to-ARTS lowering from the live compiler pipeline. Keep
  only the `CreateDbs` coarse raw bridge until MU/token coverage replaces it.

Exit gate:

- pipeline manifest documents current and target stage groups clearly;
- no benchmark or e2e behavior changes from skeleton-only work.

### M2: CODIR Isolation

Goal: make implicit captures impossible at the CODIR boundary, before any
lowering can rediscover them.

Tasks:

- [x] Move codelet boundary ownership out of SDE into CODIR (so production
  codelets run `convert-sde-to-codir` and then `convert-codir-to-arts`).
- [x] Extend CODIR verification to detect implicit above-captures in addition
  to the existing operand-shape checks (deps, params, yielded values).
  Implemented as a body-walk in `VerifyCodirPass` using `Region::isAncestor`;
  emits an `InFlightDiagnostic` naming the offending body op, the captured
  operand index, and the defining op name. Defense-in-depth on top of the
  `codir.codelet` op's existing `IsolatedFromAbove` trait verifier.
- [x] Negative tests for implicit captures
  (`codir-verify-implicit-capture-{memref,scalar}.mlir`).
- [x] Positive test for explicit memory deps and scalar params
  (`codir-verify-explicit-capture-ok.mlir`).

(EDT materialization with complete dep/param lists and `EdtLowering`
simplification belong to M4; see [`codir-edt-isolation.md`](./plans/codir-edt-isolation.md)
for the joint work order.)

Exit gate:

- negative tests fail on implicit codelet capture;
- positive tests lower memory deps and scalar params without capture recovery.

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

Goal: remove late DB rediscovery and create EDTs with complete dep/param
lists directly from CODIR.

Tasks:

- [x] Lower CODIR deps directly to `arts.db_acquire`.
- [x] Lower MU storage directly to `arts.db_alloc`.
- [x] Lower CODIR codelets directly to `arts.edt`, with deps and params already
  materialized at the CODIR boundary and no capture recovery.
- [~] Simplify `EdtLowering` around the explicit dep/param ABI. `arts.edt`
  now has an explicit `params(...)` operand segment for every supported EDT;
  lowering consumes those block args directly and rejects nonconstant scalar
  captures. The verifier also rejects nonconstant non-scalar above-captures,
  except for external `memref.alloca` stack scratch that is sunk/cloned into
  the EDT body, and rejects direct use of an explicit EDT param operand inside
  the body. EDT bodies must use the corresponding param block argument.
- [ ] Keep `CreateDbs` limited to coarse raw-memref materialization, then remove
  it when direct MU/token coverage is complete.

Exit gate:

- no `arts.lowering_contract` op or equivalent late-rediscovery marker
  survives into ARTS-RT;
- no supported benchmark relies on raw-memref DB root discovery;
- blocked/tiled raw memrefs reaching `CreateDbs` fail with unsupported
  diagnostics instead of being reindexed in ARTS.

### M5: Large/64 Performance Recovery

Goal: make every maintained benchmark correctness-clean and
performance-credible at large/64.

**Investigation status (2026-05-15):** The matrix slowdown was not a codelet
codegen regression. The codelet bodies in slow vs fast 10x builds are nearly
identical (same loop nest, same `fmul`+`fadd`). What diverges is the
**physical DB shape produced by `MemoryUnitMaterialization` and consumed at the
CODIR-to-ARTS boundary**: the slow build emitted

```
arts.db_alloc[<inout>, <heap>, <write>, <coarse>] sizes[%c1] elementSizes[%c4800, %c4800]
```

— a single coarse 4800×4800 DB — even though the SDE plan correctly carries
`physicalBlockShape = [75, 4800]` and `physicalOwnerDims = [0]`. The fast 10x
build emits a 75×4800 row-strip DB and a parametric per-block EDT (5 params,
dynamic dep count). With one big coarse block the runtime cannot fan out:
the entire matmul is serialized into a single 64-thread codelet, which
explains the ~30x gap. Likely culprit: commit `d5f01952` (`Materialize coarse
SDE memory units before Core lowering`). Session dump:
`.carts/sessions/20260515-032018-matrix-shape-baseline/`.

**Resolution status (2026-05-15):** CODIR-to-ARTS now selects DB layout from
the SDE-authored MU plan when all current accesses are compatible with the
selected scheduling unit. Whole-array host initialization/verification uses
still force a coarse root DB until M3 provides safe whole-array views over
block storage; the production codelet path still
preserves the owner-slice task shape. Cross-phase contraction intermediates
fall back to coarse materialization unless a real phase-local token plan proves
the selected block is valid for every access. Latest focused evidence:

- `gemm`: `.carts/outputs/benchmarks-gemm-large-64-crossphase-guard-20260515/20260515_064151`,
  `0.408783s` ARTS vs `6.215503s` OpenMP, `15.20x`.
- `2mm` and `3mm`: `.carts/outputs/benchmarks-2mm-3mm-large-64-crossphase-coarse-20260515/20260515_064040`,
  `2mm 0.853778s` vs `5.554953s`, `6.51x`; `3mm 0.691686s` vs
  `4.905503s`, `7.09x`.
- Fresh `gemm` ARTS-only verification:
  `.carts/outputs/benchmark-migration-20260515/20260515_092802`,
  kernel `0.481429s`, checksum `2.211832507057e+05`.

**Maintained sweep status (2026-05-16):**
`.carts/outputs/benchmarks-m6-final-current-20260516/20260516_054254`
configured 23 benchmark entries and reported 69 passed executions, 0 failed,
0 skipped, 0 checksum failures, and runner-reported geometric mean kernel speedup
`1.70x`.
The focused STREAM rerun
`.carts/outputs/benchmarks-m6-stream-final-rerun-20260516/20260516_055824`
reported 7/7 passed, checksum-clean, and `1.06x` median-filtered speedup,
superseding the noisy full-sweep STREAM median. Current audited classes are
21 fast, 2 competitive, and 0 blocked.

Tasks:

- [x] Reproduce the 2026-05-14 slow matrix-family results and compare stage
  shape against prior fast runs.
- [x] Stage-diff the pre-CODIR direct codelet output between current HEAD and
  the commit before `d5f01952` on the same `gemm.mlir`; the regression was
  traced to coarse MU materialization collapsing block shapes before the
  current CODIR-to-ARTS production path.
- [x] Move planned write-owner MU layout selection into CODIR-to-ARTS
  lowering for eligible all-access-in-unit cases; keep whole-matrix host access
  cases coarse until M3 can expose safe whole-array views over block DB roots.
- [x] Add conservative cross-phase access guards so row-strip/block DB layout
  is used only when the SDE plan is valid for every current access; otherwise
  keep the temporary bridge coarse until M3 phase-local MU/token plans exist.
- [x] Re-run focused large/64 `gemm`, `2mm`, and `3mm` after the
  materialization fix and record checksum-clean fast results.
- [x] Demote unsupported SDE physical storage attrs before the raw `CreateDbs`
  bridge while preserving semantic stencil/distribution metadata, and keep a
  Core negative test so raw blocked layouts still fail at `CreateDbs`.
- [ ] Implement SDE contraction phase plans for `gemm`, `2mm`, and `3mm`
  without hardcoded benchmark constants.
- [ ] Add generic CODIR/ARTS-RT cleanup only after codelet shape is correct.
- [x] Sweep all maintained benchmarks and record current classifications in
  [`performance-large64.md`](./plans/performance-large64.md).

Exit gate:

- all runnable maintained benchmarks compile and pass checksum;
- each benchmark is classified as fast or competitive, with no blocked entries
  remaining for M6.

### M6: Post-Restructuring Validation + Optimization Loop

Goal: prove the M1 restructuring (path rename, namespace migration, util
hierarchy, library split) preserved end-to-end behavior, then drive every
maintained sample and benchmark to a known-good state by attributing each
remaining gap to its owning dialect (SDE / CODIR / ARTS / ARTS-RT) and
fixing it there.

This is a structured loop. The validation in **Phase A** is a one-shot
sanity sweep that runs everything once. The fix-up in **Phases B-E** walks
the inventory **one benchmark at a time** so each landing diff has a single
attribution, a focused lit regression, and a benchmark rerun proving the
gain. It cross-cuts M3 (memref/MU/CU/SU coverage) and M5 (large/64 perf
gates) by giving those workstreams an explicit owning-dialect tag and a
ranked work queue.

**Inventory (all 27 samples and all 23 benchmarks must reach exit gate).**

Samples (`dekk carts examples list` — runs via
`dekk carts test --suite e2e`):

```
concurrent, convolution, deps, dotproduct, gemm, matrix, matrixmul,
mixed_access, mixed_orientation, parallel, smith-waterman, stencil, task,
array/chunks, array/deps,
cholesky/dynamic, cholesky/static,
jacobi/deps, jacobi/for,
parallel_for/<entries>,
reduction/<entries>,
task_dep/<entries>,
worksharing/<entries>
```

Benchmarks (`dekk carts benchmarks list`):

```
stream,
kastors-jacobi/jacobi-for, kastors-jacobi/poisson-for,
ml-kernels/activations, ml-kernels/batchnorm, ml-kernels/layernorm,
ml-kernels/pooling,
monte-carlo/ensemble,
polybench/2mm, polybench/3mm, polybench/atax, polybench/bicg,
polybench/convolution-2d, polybench/convolution-3d, polybench/correlation,
polybench/gemm, polybench/jacobi2d, polybench/seidel-2d,
seissol/volume-integral,
specfem3d/stress, specfem3d/velocity,
sw4lite/rhs4sg-base, sw4lite/vel4sg-base
```

The exit gate requires every entry on both lists to be classified `fast` or
`competitive`. No entry is allowed to remain blocked or unattributed at
milestone close.

**Phase A — Validate (one-shot sanity sweep).**

Run the full inventory once against HEAD to establish a post-restructuring
baseline. Do not start any optimization work until Phase A is recorded.

1. Lit suite: `dekk carts test` (current close evidence is 170/170).
2. Compile-and-run samples: `dekk carts test --suite e2e` (must stay 27/27).
3. Maintained benchmark sweep, all 23 entries at once:
   `dekk carts benchmarks run --size large --threads 64 --nodes 1 --trace
   --results-dir .carts/outputs/benchmarks-m6-<YYYYMMDD>/`. (`benchmarks run`
   with no entry-list defaults to the maintained sweep.)
4. Distributed smoke test: the smallest benchmark that exercises
   `--distributed-db` plus one multinode example, to confirm the namespace
   migration didn't break the ownership-marking path.
5. Refresh the classification table in
   [`performance-large64.md`](./plans/performance-large64.md) with the new
   sweep date, result directory, and per-entry status.

**Phase B-E — Address one benchmark (or sample) at a time.**

After Phase A, build a ranked work queue from the classification table.
Order: highest-leverage first (a fix in a shared SDE/CODIR/ARTS pass that
covers a benchmark family beats a one-off tweak). Then loop, picking ONE
entry per iteration:

1. **B (triage)**: capture per-stage IR for that one entry via
   `dekk carts compile <input> --all-pipelines --diagnose
   --diagnose-output ...`. If the entry FAILS, bisect to the first
   divergent stage. If the entry PASSES but is slow, stage-diff against
   the closest fast control.

2. **C (attribute)**: map the divergent stage to the owning dialect using
   the table below, and state the divergent fact in one sentence (e.g. "DB
   shape `1x[4800x4800]` instead of `64x[75x4800]`").

   Stage → dialect mapping:
   - `sde-input-normalization`, `initial-cleanup`, `sde-planning` → SDE
   - `sde-to-codir` → SDE/CODIR boundary
   - `codir-to-arts` → CODIR/ARTS boundary
   - `edt-transforms`, `create-dbs`, `db-opt`, `post-db-refinement`,
     `late-concurrency-cleanup`, `epochs` → ARTS
   - `pre-lowering`, `arts-rt-to-llvm`, `post-o3-opt`, `llvm-ir-emission` →
     ARTS-RT

   Per-dialect ownership for performance decisions:
   - **SDE** owns partition shape (`physicalBlockShape`,
     `physicalOwnerDims`), tiling, reductions (atomic vs tree vs
     local-accumulate), halo / window shape, scheduling intent (auto /
     runtime / static / dynamic chunk), timestep / wavefront grain, and
     MU/CU/SU plan completeness.
   - **CODIR** owns codelet isolation quality, dep-slice extraction
     (whole-token vs sliced), token-local rewrite coverage, and
     scalar-param rematerialization decisions.
   - **ARTS** owns DB granularity (coarse vs row-strip vs block), EDT count
     per epoch, epoch shape (continuation vs barrier), dependency-slot
     mechanics, amortization eligibility, and distribution placement.
   - **ARTS-RT** owns runtime-call overhead, packing / unpacking,
     CPS lowering, scalar replacement,
     GUID-range opt, alias-scope generation, and vectorization hints.

3. **D (fix)**: land the change in the owning dialect's pass or `Utils/`.
   Use `check-utils` before adding any helper. Cover the new behavior with
   a focused lit at the earliest stable IR stage. Re-run that one
   benchmark with `dekk carts benchmarks run <name> --size large --threads
   64 --trace ...` to prove the speedup. Run `dekk carts test` to confirm
   no other lit regressed.

4. **E (record + advance)**: update that entry's row in
   `performance-large64.md` (new class + speedup + result dir + commit
   hash). Then move to the next entry in the queue. Do not batch multiple
   entries into one diff; one entry at a time keeps attribution honest and
   lets the rule "the fix lives in the owning dialect" stay enforceable.

When a single fix moves multiple benchmarks in the same family
(e.g. an SDE timestep/wavefront improvement helps `polybench/jacobi2d`,
`polybench/seidel-2d`, and `polybench/atax` together), still re-run each
affected benchmark independently and update each row in the classification
table.

**Existing performance evidence to consume rather than re-run blindly.**

- M5 maintained sweep:
  `.carts/outputs/benchmarks-large-64-maintained-20260515/20260515_065122`
  (23 entries, 11 passed, 12 failed).
- M5 raw-layout-demotion follow-up (those 12 now pass checksum-clean):
  `.carts/outputs/benchmarks-raw-layout-demotion-12failures-final-20260515/20260515_072611`.
- Matrix-family fast results: see M5 section above.
- Per-benchmark classification table:
  [`performance-large64.md`](./plans/performance-large64.md).

The remaining work queue is now post-M6 quality work:

1. Replace host OpenMP fallback controls with tokenized SDE timestep/wavefront
   plans for `polybench/jacobi2d`, `polybench/seidel-2d`, and KaStORS Jacobi
   when M7 resumes repeated-stencil execution.
2. Replace the host OpenMP activation fallback with a production SDE/CODIR plan
   only when elementwise bundle splitting or vector math support can beat the
   fallback without losing checksum parity.
3. Add ARTS-RT/runtime-call cleanup only after upstream task shape is already
   correct; runtime ABI work is not allowed to mask upstream policy gaps.

Exit gate:

- `dekk carts test` 170/170 + `dekk carts test --suite e2e` 27/27 after the
  loop converges.
- Every entry in the inventory (27 samples + 23 benchmarks) is `fast`,
  `competitive`, with no blocked or unattributed entries.
- The classification table in `performance-large64.md` is dated within one
  week of the milestone close.
- `benchmark-performance-goal.md` references the latest sweep result
  directory.

## Immediate Next Session

M1 (folder/utility hierarchy) and M2 (CODIR isolation) are closed. The full
top-level path migration landed on 2026-05-15: `include/arts/` and `lib/arts/`
no longer exist; the source tree lives under `include/carts/` and `lib/carts/`
with subdialects `sde`, `arts` (was `core`), `arts-rt` (was `rt`), and `codir`.
Utilities are reclassified by ownership. CMake targets are renamed to the
`MLIRCarts*` prefix and the umbrella pass library is split into per-dialect
transform libraries. C++ namespaces are unified under
`mlir::carts::{sde,arts,arts_rt,codir}`. The live route remains
`sde-planning -> sde-to-codir -> codir-to-arts`; any direct SDE-to-ARTS/codelet
wording in docs, skills, or tests is stale unless it is in a removal note.

M6 final evidence is recorded for the current tree:
`dekk carts test` passed 170/170, `dekk carts test --suite e2e` passed 27/27,
and the latest full 23-entry large/64 sweep is under
`.carts/outputs/benchmarks-m6-final-current-20260516/20260516_054254`.
That sweep is 23/23 checksum-clean with runner-reported geometric mean kernel
speedup `1.70x`. STREAM was noisy in the full sweep, so the final M6
classification uses the superseding focused rerun
`.carts/outputs/benchmarks-m6-stream-final-rerun-20260516/20260516_055824`,
which passed 7/7 checksum-clean with `1.06x` median-filtered speedup.
The earlier distributed DB smoke is under
`.carts/outputs/benchmarks-m6-distributed-db-smoke-20260515/20260515_191546`,
and the local two-node example smoke compiled and ran under
`.carts/outputs/multinode-example-smoke-20260515/`.
The post-M6 Slurm investigation is under
`.carts/outputs/multinode-slurm-20260516/investigation-summary.md`: real
2-node and 4-node Slurm launches work on workspace-visible nodes, but current
maintained benchmark lowering still emits coarse single-block DB roots and
`arts.edt <task> <intranode>` even when SDE/CODIR distribution metadata is
present. Treat those runs as multinode launch/correctness smokes only, not
distributed scaling evidence, until CODIR-to-ARTS materializes planned owner
slices as internode EDTs with block/distributed DB ownership.
Phase B-E follow-ups moved owner-strip vector/reduction and 3-D component-slab
benchmarks to fast or competitive by fixing CODIR dispatch-step construction,
adding CODIR 2-D owner-tile dispatch, localizing SeisSol SU scratch, and
mapping one-dimensional owner-strip SUs to trailing physical owner dimensions
for SpecFEM/SW4Lite. The same round fixed the ARTS-RT affine-to-LLVM cleanup
issue for ML kernels.

M6 is closed. Continue post-M6 production coverage one slice at a time through
the ranked queue (Section H); each landing still needs owning-dialect
attribution and focused evidence. The highest-leverage post-M6 slices are:

1. **SDE repeated-timestep planning.** Use `polybench/jacobi2d` as the lead
   benchmark. Preserve full-grid owner-tile dispatch, but carry
   repetition/CPS structure so epochs are not rebuilt inside the 1000-step
   loop. Current post-M6 progress: `BarrierElimination` now groups adjacent
   same-loop repeated-timestep stages into one CPS candidate chain instead of
   disconnected pairs, with explicit completion-token barriers between every
   candidate stage. The next step is token-local dataflow/carry materialization
   before any candidate can become a final `cps_chain`.
2. **SDE wavefront/frontier planning.** Use `polybench/seidel-2d` as the
   in-place update case. Prove or reject parallel wavefront legality and choose
   legal grain.
3. **M3 PatternAnalysis extension.** Extend `PatternAnalysis` for
   ND/strided/contraction facts at memref level
   ([`memref-mu-token-rewrite.md`](./plans/memref-mu-token-rewrite.md)).
4. **M3 materialization coverage.** Expand MU/token/CODIR coverage for the
  currently demoted stencil/reduction/nested-memref plans when token-local
  rewrites are available. The raw bridge must stay coarse and guarded until it
  is deleted. Highest-priority correctness gaps are explicit OpenMP reduction
  materialization before CODIR, opaque full-memref call handling inside
  token-local codelets, and host whole-view initialization/verification over
  block DB storage so distributed kernels do not fall back to coarse roots.

Use [`utility-ownership.md`](./plans/utility-ownership.md) and the
`check-utils` skill before adding or moving helper code in any of the above
slices.

## Implementation Task Board

Each bullet is tagged with the milestones it advances. Use the Task Board to
pick the next concrete slice; use the Milestones to confirm exit gates.

### A. Planning And Documentation

- [ ] (cross-cutting) Update `docs/compiler/dialects/*` whenever a dialect
  responsibility changes.
- [ ] (cross-cutting) Keep `docs/compiler/pipeline.md` synchronized with
  `tools/compile/Compile.cpp`.
- [ ] (cross-cutting) Keep `carts-plugin/project.md` and generated agent
  resources aligned with this master plan.
- [ ] (cross-cutting) Refresh `dekk carts skills generate` after docs or
  skill changes.
- [x] (cross-cutting) Use the live CODIR-to-ARTS verifier surface
  `VerifyArtsObjectsOnly` in `Compile.cpp`, docs, and generated skills; remove
  old verifier entry points.
- [x] (cross-cutting) Remove ARTS EDT verifier bypass support (`no_verify`) and
  stale invalid tests that depended on it. Current evidence: `dekk carts build`
  and `dekk carts test` passed on 2026-05-15 after the removal.

### B. Folder And Utility Ownership

- [x] (M1) Add or verify `Utils/` folders for SDE, CODIR, ARTS, and ARTS-RT.
  All four dialects now have populated `Utils/` directories with real headers
  and CMake wiring (no longer README-only).
- [x] (M1, M2) Move reusable helpers from passes into the earliest owning
  dialect's `Utils/` surface (`check-utils` skill). The cross-dialect blob
  formerly in `include/carts/utils/` is reclassified: ARTS-only headers
  (DbUtils, EdtUtils, LoweringContractUtils, PartitionPredicates,
  BlockedAccessUtils, MetadataEnums, ARTSCostModel) live in
  `carts/dialect/arts/Utils/`; ARTS-RT ID materialization (`IdRegistry`) lives
  in `carts/dialect/arts-rt/Utils/`; cross-layer helpers
  (`LocationMetadata`, `LoopInvarianceUtils`, `RuntimeConfig`) live in
  `carts/utils/`;
  SDE-only SDECostModel moved to `carts/dialect/sde/Utils/`. CODIR ABI
  predicates already lived in `carts/dialect/codir/Utils/`. Remaining:
  pass-local SDE memref/access helpers and ARTS-RT pointer/packing helpers
  stay pass-local until a second consumer appears.
- [ ] (cross-cutting) Keep pass-local static helpers only when they are
  genuinely local to one pass and not a dialect invariant.
- [x] (M1, M2) Delete duplicated local helpers after extraction. The former
  mixed CODIR boundary implementation has been split into
  `Conversion/SdeToCodir/SdeToCodir.cpp`,
  `Conversion/CodirToArts/CodirToArts.cpp`, `Transforms/VerifyCodir.cpp`, and
  `Transforms/CodirCodeletOpt.cpp`. Cross-dialect dedupe pass on 2026-05-15
  removed several duplicate utilities (ConversionUtils.h shadow index helpers,
  the byte-identical PolygeistToSde `materializeDependView`).
- [x] (M1) Update include paths and CMake in small, buildable slices. The
  full top-level path rename (`include/arts/` → `include/carts/`, `core/` →
  `arts/`, `rt/` → `arts-rt/`) and per-dialect Utils/ wiring landed; lit/test
  paths and `tools/scripts/test.py` follow. Library targets renamed to
  `MLIRCartsArts`, `MLIRCartsArtsRt`, `MLIRCartsTransforms` (with matching
  PassIncGen targets).

### C. CODIR Boundary

- [x] (M1) CODIR dialect IR for isolated `codir.codelet` with deps, params,
  and yields registered.
- [x] (M1) Wire `convert-sde-to-codir`, `codir-codelet-opt`, `verify-codir`,
  `convert-codir-to-arts` into executable default stage descriptors.
- [x] (M2) Move codelet-specific SDE verification into CODIR verification.
- [x] (M2) Extend `verify-codir` to reject implicit above-captures.
- [x] (M2) Negative tests for implicit captures.
- [x] (M2) Positive tests for explicit memory deps and scalar params.
- [x] (M3) CODIR-local cleanup removes dead token-local rematerialization inside
  isolated `codir.codelet` bodies before `verify-codir` and ARTS lowering.

### D. Memref MU/CU/SU Rewrite

- [ ] (M3) Extend `PatternAnalysis` for ND, strided, contraction, reduction,
  and halo facts at memref level.
- [ ] (M3) Materialize MU storage for args, globals, allocas, task deps,
  reductions, and intermediates.
- [~] (M3) Rewrite `codir.codelet` body loads/stores to token-local
  coordinates before ARTS. Direct `sde.cu_codelet` sliced-token loads/stores
  are covered for rank-1 and ND unit-step slices, including constant and
  dynamic per-dimension offsets. Direct `sde.cu_task` bodies with a sliced
  `sde.mu_dep` source, complete static unit-step offsets/sizes, direct
  load/store uses, and simple body-local `memref.subview`/`polygeist.subindex`
  views now materialize a CODIR subview dep and rewrite body indices to
  token-local coordinates; identical duplicate-source static slices reuse that
  CODIR view with merged access mode. Mixed-window duplicate-source task deps
  now materialize distinct positional CODIR deps when all source-rooted task
  body uses are exact matching unit-stride `memref.subview` access proofs,
  exact row `polygeist.subindex` proofs, or exact-offset direct root accesses.
  Mixed-window direct root accesses are covered only when each load/store index
  exactly matches one of the dep offsets and therefore rewrites to local zero.
  Dynamic task-dep bounds are covered only by the exact-view proof shape or by
  direct root loads/stores whose indices exactly match the dynamic dep offsets.
  Remaining: general dynamic task slice bounds, mixed-window direct root
  accesses with non-exact indices, halo, strided/windowed views with non-unit
  access maps, and broader `sde.mu_dep` canonicalization.
- [ ] (M3) Remove `RaiseMemrefToTensor`, `RaiseToTensor`, `LowerToMemref`
  source files after equivalent memref tests pass.

### E. ARTS And ARTS-RT Cleanup

- [x] (M4) Lower CODIR deps directly to `arts.db_acquire`.
- [x] (M4) Lower MU storage directly to `arts.db_alloc`.
- [~] (M4) Lower CODIR codelets directly to `arts.edt` and simplify
  `EdtLowering` around the explicit dep/param ABI. CODIR creates EDTs with
  explicit dep and `params(...)` operands on the production path; lowering and
  verification consume explicit param block args and reject nonconstant
  above-captures for explicit-param EDTs, including non-scalar captures except
  external `memref.alloca` stack scratch, and direct body uses of the EDT param
  operand itself. Legacy scalar capture recovery and the CODIR explicit ABI
  marker are removed; all supported EDT params are explicit `params(...)`
  operands. Remaining cleanup is ARTS-RT/raw-bridge removal.
- [x] (M4) Use `VerifyArtsObjectsOnly` in the live staged pipeline after
  `VerifySdeLowered`; obsolete verifier pass names are removed.
- [ ] (M4) Remove coarse `CreateDbs` after direct coverage exists.
- [ ] (M4) Keep runtime ABI and pointer packing cleanup in ARTS-RT.

### F. Performance Evidence

- [x] (M5) Re-run large/64 matrix-family benchmarks after the current
  materialization slice (`dekk carts benchmarks run gemm,2mm,3mm`).
- [~] (M5) Compare task count, DB count, acquire modes, dep windows, and
  runtime counters against fast controls. The DB layout and acquire guards have
  been compared for the matrix-family fix; full runtime-counter comparison
  remains part of the maintained sweep.
- [x] (M5) Sweep all maintained benchmarks (`dekk carts benchmarks list`,
  then run each) and refresh the classification table in
  [`performance-large64.md`](./plans/performance-large64.md) before declaring
  the migration stable.
- [x] (M5) Record blocked benchmarks with owning layer and next action in
  the same classification table.

### G. Verification Gates

- [~] (cross-cutting) Each milestone closes only when the gate matrix in
  [`verification-release.md`](./plans/verification-release.md) is green for
  the affected slice. Current migration evidence includes `dekk carts build`,
  `dekk carts test` (170/170), `dekk carts test --suite e2e` (27/27),
  `dekk carts skills generate`, `git diff --check`, focused M3/M4 lit
  coverage, and the 23/23 checksum-clean large/64 benchmark evidence.
- [~] (cross-cutting) Add focused lit, pipeline-dump, e2e, and benchmark
  evidence per slice; do not promote `.carts/` artifacts unless they become
  intentional fixtures. Current benchmark evidence covers the matrix-family
  task-shape fix, raw-layout boundary follow-up, CODIR owner-tile/trailing-dim
  dispatch fixes, and the latest 23/23 checksum-clean large/64 sweep. Only
  the SDE timestep/wavefront performance blockers remain open.

### H. Post-Restructuring Validation Loop

Tracks the M6 validation+optimization cycle. Phase A is a one-shot
sanity sweep; Phases B-E walk the inventory one entry at a time so each
landing diff has a single owning-dialect attribution.

**Phase A — one-shot post-restructuring sweep:**

- [x] (M6 Phase A) Re-run `dekk carts test` against HEAD. Current evidence:
  170/170 lit after adding the final SDE host-fallback regression.
- [x] (M6 Phase A) Re-run `dekk carts test --suite e2e` against HEAD;
  confirm 27/27 e2e parity covering all samples in `dekk carts examples
  list`.
- [x] (M6 Phase A) Re-run the full maintained benchmark sweep at once with
  `dekk carts benchmarks run --size large --threads 64 --nodes 1 --trace
  --results-dir .carts/outputs/benchmarks-m6-<YYYYMMDD>/` covering all 23
  entries. Current evidence:
  `.carts/outputs/benchmarks-m6-final-current-20260516/20260516_054254`
  plus superseding focused STREAM rerun
  `.carts/outputs/benchmarks-m6-stream-final-rerun-20260516/20260516_055824`.
- [x] (M6 Phase A) Run one distributed smoke test (`--distributed-db`) and
  one multinode example to verify the namespace migration didn't break the
  ownership-marking path. Current evidence:
  `.carts/outputs/benchmarks-m6-distributed-db-smoke-20260515/20260515_191546`
  and `.carts/outputs/multinode-example-smoke-20260515/`.
- [x] (M6 Phase A) Refresh the classification table in
  [`performance-large64.md`](./plans/performance-large64.md) with the new
  sweep date, result directory, and per-entry status.

**Phases B-E — one entry per iteration (do not batch):**

- [x] (M6 B-E) For each iteration, pick ONE entry from the ranked queue:
  capture `--all-pipelines` stage dumps under `.carts/sessions/<topic>/`,
  bisect to the first divergent stage (Phase B), attribute it to SDE /
  CODIR / ARTS / ARTS-RT per the decision tree in
  [`benchmark-performance-goal.md`](./benchmark-performance-goal.md)
  (Phase C), land the fix in the owning dialect with a focused lit
  regression and a benchmark rerun proving the speedup (Phase D), then
  update that entry's row in `performance-large64.md` (Phase E).
- [x] (M6 E) When a single dialect-level fix moves multiple benchmarks in
  the same family, still re-run each affected entry independently and
  update each row in the classification table.
- [x] (M6 E) Repeat the iteration until every entry in the inventory (27
  samples + 23 benchmarks) is `fast` or `competitive` with zero blocked and
  zero unattributed entries.

## Rules

- Do not add hardcoded benchmark constants to passes.
- Do not add SDE references to ARTS runtime topology.
- Do not make ARTS infer source-level owner dims or dependency-window legality.
- Do not pass hidden C++ analysis objects across dialect boundaries; materialize
  required facts in IR.
- Do not keep tensor-carrier paths once memref MU/token/CODIR coverage exists.
- Remove obsolete code once the replacement path has focused tests and at least
  one affected e2e or benchmark proof.
- Enforce dialect-local `Utils/` ownership before every pass edit; use the
  `check-utils` skill before adding any helper.
- Put generated artifacts under `.carts/outputs/...` or
  `.carts/sessions/...`; do not commit them unless promoted to real tests.
