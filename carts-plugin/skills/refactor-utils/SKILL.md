---
name: carts-refactor-utils
description: Use when auditing code quality, consolidating duplicate helpers, extracting reusable pass-local helpers, or fixing hardcoded utility placement.
---

# CARTS Utility Refactoring

Use this skill for focused utility extraction after `carts-check-utils` has
chosen the canonical owner. Use `carts-simplify` first when the broader
question is whether a patch is too complex.

Read `references/known-duplicates.md` only when the task needs the historical
duplicate-helper backlog.

## Workflow

1. Run or consult `carts-check-utils` and record one placement decision:
   use existing helper, extract to a named owner, or keep pass-local.
2. Choose the narrowest correct home from that decision:
   - pass-local for logic used by one transform only and not a dialect fact;
   - dialect utility when it expresses a dialect invariant;
   - `include/carts/utils/` and `lib/carts/utils/` for broadly shared behavior;
   - analysis APIs when the logic belongs to DB/EDT/loop/cache/metadata state.
3. Add or move the declaration and implementation.
4. Update every call site with `rg`.
5. Remove the old static helper.
6. Run `dekk carts format`, then the smallest meaningful build/test command.

## Guardrails

- Do not extract helpers that are intentionally pass-specific.
- Do not create a new `*Utils` file when an existing semantic owner such as
  `LoopUtils`, ARTS `LoopInvarianceUtils`, shared `ValueAnalysis`, `DbUtils`,
  or `EdtUtils` already fits.
- Do not add new hardcoded project attribute strings; use centralized constants.
- Do not use a later cleanup pass to make malformed IR correct.
- Keep unrelated refactors out of the patch.
