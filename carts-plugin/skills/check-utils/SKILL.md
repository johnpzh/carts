---
name: carts-check-utils
description: Use before adding helper or utility functions to CARTS pass files, when writing static helpers, when deciding loop/value/SDE/CODIR/ARTS/ARTS-RT utility placement, or when reviewing duplicated helpers and utility sprawl.
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Agent
argument-hint: <function-name-or-description>
parameters:
  - name: function_name
    type: str
    gather: "Name or short description of the helper you want to add (for example: 'is loop IV', 'fold constant index', 'hoist loop invariant op')"
---

# CARTS Utility Ownership Check

## Purpose

Run this skill before adding any new static helper, utility file, or `*Utils`
surface. It prevents pass-local helper sprawl by finding existing helpers,
choosing one canonical owner, and documenting why the helper belongs there.

## Hard Rule

Do not create a new utility file until all of these are true:

- no existing helper covers the behavior;
- the helper is needed by more than one pass or expresses a dialect invariant;
- the proposed file has one clear semantic category, not a grab bag;
- the owning dialect or shared layer is explicit in the patch summary;
- the declaration, implementation, users, and focused verification land
  together.

One semantic category gets one canonical home. Do not create `LoopIVUtils`,
`IndexUtils`, `ValueHelpers`, or pass-local copies when `LoopUtils`,
ARTS `LoopInvarianceUtils`, shared `ValueAnalysis`, or `Utils` already own the
category.

## Pass-Local Helper Rule

A helper may stay `static` in a pass file only when every item is true:

- it is used by exactly one pass implementation;
- it does not express a dialect invariant;
- it does not duplicate an existing helper by name or behavior;
- it is not needed by a verifier, analysis, conversion, or sibling pass;
- extracting it would make the code harder to read.

If any item is false, move it to the owning dialect `Utils/`, an owning
analysis API, a current shared `include/carts/utils` helper, or a pass-area
support file.

Judge ownership by path and semantic layer first. A helper under
`include/carts/utils` is shared CARTS infrastructure even when most current
callers are ARTS-heavy.

## Placement Matrix

Use the narrowest correct home:

| Helper kind | Canonical home |
|-------------|----------------|
| Dialect-neutral value constants, folding, casts, memref views, and same-value checks | `include/carts/utils/ValueAnalysis.h` |
| ARTS DB/EDT/runtime-query value folding and provenance | `include/carts/dialect/arts/Utils/ValueAnalysisUtils.h` |
| ARTS-RT depv/DB pointer value folding and provenance | `include/carts/dialect/arts-rt/Utils/RtDbUtils.h` |
| Index builders and generic pure-op predicates | `include/carts/utils/Utils.h` |
| Loop shape, loop IVs, nearest enclosing loops, trip counts, loop depth | `include/carts/utils/LoopUtils.h` |
| ARTS/ARTS-RT loop invariance, hoist legality, dominance for hoisting | `include/carts/dialect/arts/Utils/LoopInvarianceUtils.h` |
| Deferred op removal | `include/carts/utils/RemovalUtils.h` |
| Runtime config, source-location IDs, DB/EDT runtime metadata | `include/carts/dialect/arts/Utils` |
| Compiler instrumentation or driver-only helpers | existing compile-driver owner or current project-wide helper |
| SDE source semantics, memref roots/access maps, PatternAnalysis facts, MU/CU/SU planning | `include/carts/dialect/sde/Analysis` or `include/carts/dialect/sde/Utils` |
| CODIR codelet isolation, dep/param ABI, token-local views | `include/carts/dialect/codir/Utils` or a boundary-specific conversion helper |
| ARTS DB/EDT/epoch objects, dependency slots, placement, distributed ownership | `include/carts/dialect/arts/Utils` or `include/carts/dialect/arts/Analysis` |
| ARTS-RT runtime ABI packing, depv layout, runtime calls, pointer lowering | `include/carts/dialect/arts-rt/Utils` |
| Shared compiler helper with no dialect semantics | current `include/carts/utils`; create `carts/support` only as a planned repo-wide migration |

Loop IV helpers are not a new category. Generic IV recognition belongs in
`LoopUtils`. SDE-specific structured-access IV interpretation belongs in SDE
Analysis/Utils. ARTS graph loop state belongs in ARTS Analysis; do not move it
into generic loop helpers.

## Shared Utility Catalog

Read `references/utility-catalog.md` when the helper touches values, loops,
DB/EDT mechanics, runtime ABI, or known duplicate categories. It is the quick
search map for the current utility surface.

## Search Strategy

Search by behavior, not only by name:

1. Exact name across `include/carts` and `lib/carts`.
2. Semantic keywords:
   - zero/one/constant/fold/same value/provenance;
   - loop IV/trip count/nearest loop/innermost/depth;
   - invariant/hoist/dominance/div/rem safety;
   - memref/access/load/store/root/slice;
   - DB/acquire/alloc/partition/ownership;
   - EDT/epoch/dependency slot/placement;
   - runtime ABI/deps/packing/pointer/call.
3. Existing shared utilities:
   - `include/carts/utils` and `lib/carts/utils`;
   - `include/carts/dialect/*/Utils` and `lib/carts/dialect/*/Utils`;
   - owning `Analysis/` APIs.
4. Pass-area support files such as `*Support.cpp`, `*Internal.h`, and
   boundary-specific conversion helpers.
5. Duplicate behavior in pass files, including lambdas and anonymous namespace
   helpers.

## Required Answer

Every utility placement decision must state exactly one of:

- `Use existing helper at <path> because <reason>.`
- `Extract to <path> because <owning semantic category>.`
- `Keep pass-local because all pass-local rule items are true.`

If extraction is chosen, the patch must include the declaration, definition,
call-site update, old helper removal, and focused verification. If a helper uses
CARTS IR attributes, add or use the owning TableGen attribute/accessor instead
of hardcoded strings. Use `AttrNames::` only for remaining shared or
transitional metadata.

## Existing Helpers To Reuse

Do not re-add these local copies:

| Need | Existing helper |
|------|-----------------|
| zero/one/constant checks | `ValueAnalysis::{isZeroConstant,isOneConstant,tryFoldConstantIndex}` |
| structurally one-like value | `ValueAnalysis::isOneLikeValue` |
| same-value or range equivalence | `ValueAnalysis::{sameValue,areValuesEquivalent,areValueRangesEquivalent}` |
| create zero/one/arbitrary index | `createZeroIndex`, `createOneIndex`, `createConstantIndex` |
| loop IV, trip count, nearest loop, loop depth | `LoopUtils.h` |
| loop-invariant hoisting target | ARTS `LoopInvarianceUtils.h` |
| side-effect-free arithmetic-like op | `isSideEffectFreeArithmeticLikeOp` in `Utils.h` |
| SDE Polygeist dependency index clamping or OMP-region predicates | `PolygeistToSdeUtils.h` |
| undef-like op detection | `isUndefLikeOp` in ARTS `RuntimeOpUtils.h` |
| DB provenance/access info | `DbUtils.h` |
| EDT environment/captures | `EdtUtils.h` |
| lowering contracts | `LoweringContractUtils.h` |
| runtime IDs/calls/DB ABI | ARTS-RT `Utils/` |

## Attribute Strings

Never hardcode project attribute strings in new helpers. Use
generated ODS accessors such as `op.getStencilMinOffsetsAttrName()` or the
owning dialect's attribute helpers for CARTS IR attrs. Use
`AttrNames::Operation::*` only for remaining shared or transitional metadata.
Add an ODS attr/accessor first when an attr is part of CARTS IR.
