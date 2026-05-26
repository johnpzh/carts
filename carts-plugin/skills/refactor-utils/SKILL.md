---
name: carts-refactor-utils
description: Use when consolidating duplicate CARTS helpers or extracting reusable pass-local helpers after placement is decided.
---

# CARTS Utility Refactoring

Use after [[carts-check-utils]] has chosen the canonical owner. Use
[[carts-simplify]] first when the broader question is patch complexity.

## Hard Rule

- Do not extract helpers that are intentionally pass-specific.
- Promote multi-consumer `lib/` Utils headers to `include/`; see [[carts-include-tier]].
- Do not create a new `*Utils` file when an existing semantic owner fits.
- Do not add hardcoded CARTS IR attribute strings; see [[carts-attr-consolidation]].
- Keep unrelated refactors out of the patch.

## Procedure

1. Record the [[carts-check-utils]] decision: use existing, extract, or keep local.
2. Read `references/known-duplicates.md` only when the task touches the backlog.
3. Choose the narrowest correct home: pass-local, dialect `Utils/`, shared
   `include/carts/utils`, or owning `Analysis/`.
4. If a `lib/` Utils header is included by more than one `.cpp`, promote it as
   part of the same extraction patch.
5. Add or move declarations and implementations.
6. Update every call site with `rg`.
7. Remove old static helpers and duplicate wrappers.
8. Run `dekk carts format`, then the smallest meaningful build/test command.

## Required Answer

State the source helper, canonical owner, removed duplicates, include-tier
decision, and verification command.
