---
name: carts-find-utils
description: Use when looking for an existing CARTS helper before writing one.
---

# CARTS Find Utilities

This is read-only discovery. If no helper exists, invoke [[carts-check-utils]]
before adding one.

## Hard Rule

- Search by behavior, not just the desired function name.
- Do not edit files while running this skill.
- Check public utilities, dialect utilities, analyses, and pass-local helpers.
- Treat similar behavior with a different name as a possible reusable helper.
- If no helper exists, do not choose placement here; use [[carts-check-utils]].

## Procedure

1. Search the exact requested name:
   ```bash
   rg '<name>' include/carts lib/carts
   ```
2. Search behavior keywords across `include/carts` and `lib/carts`.
3. Inspect likely owners: `include/carts/utils`,
   `include/carts/dialect/*/Utils`, `lib/carts/dialect/*/Utils`, and
   `include/carts/dialect/*/Analysis`.
4. Search anonymous namespaces and lambdas in pass files for duplicate behavior.
5. If attributes are involved, route to [[carts-attr-consolidation]].
6. If a candidate is found, read its declaration and at least one caller.

## Required Answer

Return exactly one:

- `Found at <path>: <helper> because <behavior match>.`
- `Not found - invoke [[carts-check-utils]] to place a new helper.`
