---
name: carts-review
description: Use before committing CARTS changes, during PR review, after substantial compiler/runtime edits, or when checking conventions, missing tests, fixture refreshes, or regression risk.
---

# CARTS Review

Review findings first. Focus on behavioral bugs, missing tests, convention
violations, stale assumptions, and verification gaps.

Read `conventions.md` for CARTS-specific rules.

## Checklist

1. Is this a production fix for the root cause, not a band-aid over a symptom?
2. Does the change belong to the dialect/stage it edits, with the dialect's
   function and limits understood?
3. Are attributes centralized and analysis APIs respected?
4. Is there a focused regression test at the owning stage?
5. Did fixture updates reflect intended IR changes, not hidden bugs?
6. Was verification run in the current workspace?
7. For runtime/distributed paths, was local behavior proven before multinode?
8. Was `carts-simplify` run before review/commit, including patch size,
   duplicated helpers, utility placement, stale debug output, existing API
   reuse, and `.carts/` artifact discipline?

When no issues are found, say so and state any remaining test gaps.
