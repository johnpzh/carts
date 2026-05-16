# Dialect Utility Ownership Plan

## Objective

Prevent helper sprawl while the compiler is split into SDE, CODIR, ARTS, and
ARTS-RT. Every dialect should have a local `Utils/` surface for reusable facts,
predicates, builders, and small IR queries that belong to that dialect. Pass
files should contain only helpers that are genuinely local to one transform.

This plan is cross-cutting and must be checked before implementation work in
any dialect subplan.

## Target Layout

Each dialect owns:

```text
IR/
Analysis/
Transforms/
Conversion/
Verify/
Utils/
```

Use the narrowest correct home:

| Utility kind | Owner |
|---|---|
| Source semantics, memref roots, access maps, SDE patterns, MU/CU/SU plan helpers | `carts/dialect/sde/Utils` |
| Codelet capture checks, dep/param ABI helpers, token-local view helpers | `carts/dialect/codir/Utils` |
| DB/EDT/epoch object helpers, dependency slots, ARTS placement/resource helpers | `carts/dialect/arts/Utils` |
| Runtime ABI packing, depv layout, runtime-call/pointer helpers | `carts/dialect/arts-rt/Utils` |
| Cross-dialect CARTS compiler helpers with no dialect semantics | `carts/support` |
| Compatibility-only helpers for the current tree | Current `include/carts/utils` or nearest existing support file until moved |

## Pass-Local Helper Rule

A helper may stay `static` in a pass file only when all are true:

- [ ] It is used by exactly one pass implementation.
- [ ] It does not express a dialect invariant.
- [ ] It does not duplicate an existing helper by name or behavior.
- [ ] It is not needed by a verifier, analysis, conversion, or sibling pass.
- [ ] It is small enough that extracting it would make the code harder to read.

If any item is false, move it to the owning dialect `Utils/`, an owning
analysis API, or a pass-area support file.

## Required Pre-Flight

Before adding a helper to a pass:

- [ ] Run the `check-utils` skill or perform the equivalent search.
- [ ] Search by behavior, not only by name.
- [ ] Check existing shared utilities under `include/carts/utils` and
  `lib/carts/utils`.
- [ ] Check current support files such as `*Support.cpp` and `*Support.h`.
- [ ] Check the target dialect utility location if it already exists.
- [ ] Decide the owning dialect and write that down in the patch notes or
  review summary.

## Extraction Checklist

When moving a helper out of a pass:

- [ ] Name it after the dialect concept, not the first pass that needed it.
- [ ] Put declarations in the owning dialect `Utils/` header.
- [ ] Put implementation in the owning dialect `Utils/` source.
- [ ] Add or update focused tests at the earliest stable IR stage.
- [ ] Remove the pass-local duplicate.
- [ ] Update includes without broad churn.
- [ ] Run `dekk carts build` and the focused lit test for the owning pass.

## Attribute And String Rule

- [ ] Do not hardcode project attribute names in pass helpers.
- [ ] Use existing `AttrNames::*` constants and operation attribute helpers.
- [ ] Add new attribute constants before using them in passes.
- [ ] Keep textual dialect names aligned with the owning dialect:
  `sde.*`, `codir.*`, `arts.*`, or `arts_rt.*`. Folder names use the hyphenated
  form (`arts-rt`); MLIR textual dialect names use the underscore form
  (`arts_rt`).
- [ ] Attribute name constants belong in the owning dialect's
  `Utils/AttrNames.h`.

## Migration Phases

### Phase 1: Skill And Docs

Status: complete for the current guidance surface.

- [x] Harden `check-utils` so it triggers before pass-local helper work.
- [x] Link this plan from the master plan and plan index.
- [x] Add `Utils/` to the target folder plan.
- [x] Regenerate agent resources with `dekk carts skills generate`.

Exit gate:

- `check-utils` describes the pass-local helper rule and dialect utility
  decision tree.

### Phase 2: Skeleton

Status: complete. The include/lib skeletons exist for SDE, CODIR, ARTS, and
ARTS-RT; they are README-only and intentionally not wired into CMake yet.

- [x] Add `Utils/README.md` skeletons under each target dialect in
  `include/carts/dialect`.
- [x] Add `Utils/README.md` skeletons under each target dialect in
  `lib/carts/dialect`.
- [x] Do not wire empty utility folders into CMake until a real helper moves.

Exit gate:

- Target folders show where utilities belong without changing build behavior.

### Phase 3: First Real Extractions — Done for cross-dialect blob

The previously-monolithic `include/carts/utils/` has been reclassified by
ownership:

- [x] CODIR ABI predicates live in `include/carts/dialect/codir/Utils/`
  (`CodeletABIUtils`). SDE-dependent task-dependency slice proof helpers stay
  scoped to `lib/carts/dialect/codir/Conversion/SdeToCodir/TaskDepSliceUtils.*`
  because they are conversion logic, not CODIR dialect utilities.
- [x] ARTS-only utilities moved to `include/carts/dialect/arts/Utils/`:
  DbUtils, EdtUtils, LoweringContractUtils, PartitionPredicates,
  BlockedAccessUtils, MetadataEnums, ARTSCostModel (was utils/costs/).
  Legacy metadata attr-name JSON plumbing has been removed.
- [x] ARTS-RT-only utility moved to `include/carts/dialect/arts-rt/Utils/`:
  IdRegistry, which is only used by DB/EDT runtime-lowering ID materialization.
- [x] Cross-layer utilities kept in `include/carts/utils/`: LocationMetadata,
  LoopInvarianceUtils, RuntimeConfig.
- [x] SDE-only utility moved to `include/carts/dialect/sde/Utils/`:
  SDECostModel.
- [ ] Pass-local SDE memref/access helpers used by PatternAnalysis, tiling,
  and MU materialization remain to be extracted to SDE Utils when a second
  caller appears. Today they sit inside their pass file or under
  `Analysis/AffineAccessUtils.h`.
- [ ] Pass-local ARTS-RT pointer/packing helpers in lowering passes remain
  pass-local until a second consumer appears.

Exit gate:

- No newly added helper is duplicated across pass files.

### Phase 4: Cleanup — Done for the major blob

- [x] Stale `include/carts/utils/{costs,machine}/` subdirs removed; ARTSCostModel
  lives in ARTS Utils, while RuntimeConfig now lives in CARTS-shared utils
  because ARTS analysis, ARTS-RT lowering, and the compile driver all consume it.
- [x] CARTS-shared `include/carts/utils/` is now scoped: Debug, LoopUtils,
  OperationAttributes, PassInstrumentation, RemovalUtils, StencilAttributes,
  Utils, ValueAnalysis, LocationMetadata, LoopInvarianceUtils, RuntimeConfig,
  plus benchmarks/ and testing/. Every entry is verifiably used across 2+
  subdialects or is project-wide infrastructure.
- [ ] Watch for new pass-local duplicates as compiler work continues.

Exit gate:

- A repo search for duplicated helper names or behavior has no unresolved
  production duplicates.

## Verification

- `git diff --check`
- `dekk carts skills generate` after skill/doc changes
- `dekk carts skills status`
- `dekk carts build` after real source moves
- focused lit tests for the pass or dialect that consumed the utility
- 2026-05-15 CODIR extraction evidence: `dekk carts build`, focused sliced
  task-dependency CODIR/SDE lit tests, `dekk carts test` (163/163), and
  `dekk carts test --suite e2e` (27/27) passed after
  the first CODIR utility slice was wired.
