---
name: carts-session-start
description: Use at the start of any CARTS compiler session before reconnaissance begins.
---

# CARTS Session Start

This meta-skill routes the first edit through the load-bearing project skills.

## Hard Rule

- Before compiler source edits, read the matching skill docs first.
- Helper work starts with [[carts-find-utils]] or [[carts-check-utils]].
- Dialect boundary work starts with [[carts-dialect-map]].
- Attribute work starts with [[carts-attr-consolidation]].
- Before commit, run [[carts-simplify]] then [[carts-review]].

## Procedure

1. Restate the user task in one sentence.
2. Classify touched surface: helper, dialect boundary, attribute, include tier,
   pass development, debugging, tests, runtime, benchmark, or skill maintenance.
3. Read the matching skills:
   - helpers: [[carts-find-utils]], [[carts-check-utils]]
   - dialect boundaries: [[carts-dialect-map]], [[carts-pipeline-map]]
   - attributes: [[carts-attr-consolidation]]
   - include placement: [[carts-include-tier]]
   - duplicate helper cleanup: [[carts-refactor-utils]]
4. For duplicate-helper work, also read
   `refactor-utils/references/known-duplicates.md`.
5. Emit one line naming the skills that apply before reconnaissance begins.
6. Keep generated diagnostics and scratch state under `.carts/outputs` or
   `.carts/sessions`.

## Required Answer

`This task uses: <skill-list>. Boundary/placement risk: <one sentence>.`
