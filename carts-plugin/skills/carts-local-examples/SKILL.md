---
name: carts-local-examples
description: Use when listing, running, fixing, sweeping, or validating CARTS local sample programs, examples runner behavior, or e2e compile-and-run tests.
---

# CARTS Local Examples

Read `references/examples-local.md` before sweeping samples. `examples run` is
not read-only because it cleans and rewrites ignored sample artifacts.

## Workflow

1. List examples:
   ```bash
   dekk carts examples list
   ```
2. Prefer a focused e2e test for a known sample:
   ```bash
   dekk carts test --suite e2e
   dekk carts lit tests/e2e/<case>.c
   ```
3. Use the examples runner when you need the user-facing workflow:
   ```bash
   dekk carts examples run <name> --trace
   ```
4. For failures, classify with `carts-debug`, then reduce with
   `carts-reproducer`.

Do not rely on stale `tests/examples/` paths.
