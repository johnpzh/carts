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
| Compatibility-only helpers for the current tree | Current `include/arts/utils` or nearest existing support file until moved |

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

- [ ] Run the `carts-check-utils` skill or perform the equivalent search.
- [ ] Search by behavior, not only by name.
- [ ] Check existing shared utilities under `include/arts/utils` and
  `lib/arts/utils`.
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
  `sde.*`, `codir.*`, `arts.*`, or `arts_rt.*`.

## Migration Phases

### Phase 1: Skill And Docs

- [ ] Harden `carts-check-utils` so it triggers before pass-local helper work.
- [ ] Link this plan from the master plan and plan index.
- [ ] Add `Utils/` to the target folder plan.
- [ ] Regenerate agent resources with `dekk carts skills generate`.

Exit gate:

- `carts-check-utils` describes the pass-local helper rule and dialect utility
  decision tree.

### Phase 2: Skeleton

- [ ] Add `Utils/README.md` skeletons under each target dialect in
  `include/carts/dialect`.
- [ ] Add `Utils/README.md` skeletons under each target dialect in
  `lib/carts/dialect`.
- [ ] Do not wire empty utility folders into CMake until a real helper moves.

Exit gate:

- Target folders show where utilities belong without changing build behavior.

### Phase 3: First Real Extractions

- [ ] Extract SDE memref/access helpers used by PatternAnalysis, tiling, and MU
  materialization.
- [ ] Extract CODIR codelet capture and token-local view helpers when CODIR is
  introduced.
- [ ] Extract ARTS DB/EDT/epoch helpers currently repeated in transforms.
- [ ] Extract ARTS-RT pointer/packing helpers from lowering passes.

Exit gate:

- No newly added helper is duplicated across pass files.

### Phase 4: Cleanup

- [ ] Delete dead compatibility helpers after direct dialect utilities exist.
- [ ] Retire stale `include/arts/utils` entries that now have dialect owners.
- [ ] Keep only truly cross-dialect helpers in common CARTS support.

Exit gate:

- A repo search for duplicated helper names or behavior has no unresolved
  production duplicates.

## Verification

- `git diff --check`
- `dekk carts skills generate` after skill/doc changes
- `dekk carts skills status`
- `dekk carts build` after real source moves
- focused lit tests for the pass or dialect that consumed the utility
