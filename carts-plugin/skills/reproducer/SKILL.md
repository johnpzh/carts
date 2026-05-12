---
name: carts-reproducer
description: Use when a large failing program, benchmark, or stage dump needs to become a minimal C, MLIR, or lit reproducer.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [<input-file | benchmark-path>]
parameters:
  - name: failing_test
    type: str
    gather: "Path to the failing test file or benchmark input"
  - name: regression_commit
    type: str
    gather: "Git commit hash where regression was introduced, or 'unknown'"
---

# CARTS Reproducer Reduction

Goal: keep the failure, remove everything else.

Use bundled helpers when they fit:
- `scripts/snapshot-stage.sh` — capture a specific stage dump to a file
- `scripts/find-related-tests.sh` — search existing regressions before inventing a new one
- `scripts/scaffold-contract-test.sh` — create a lit test skeleton for a stage-boundary contract
- `scripts/list-stage-boundaries.sh` — print the canonical stage list from the CLI

Read these while shrinking a case:
- `references/reduction-checklist.md`
- `../debug/references/stage-ownership.md`
- `../debug/references/command-patterns.md`

## Choose the Right Target Form

- C/C++ reproducer: use when the frontend or OpenMP lowering matters
- MLIR reproducer: use when the failing stage is already known
- `lib/arts/dialect/{sde,core,rt}/test/*.mlir`: use for compiler IR contracts
- `samples/*`: use for end-to-end runtime behavior

## Reduction Order

1. Freeze the oracle.
   - checksum, FileCheck line, verifier error, crash, timeout, or stage-specific IR invariant
2. Freeze the failing boundary.
   - identify the first bad pipeline stage with `dekk carts compile --pipeline=<stage>`
   - if needed, dump all stages with `--all-pipelines`
3. Remove unrelated structure aggressively.
   - extra loops
   - unrelated memrefs
   - independent tasks / acquires
   - helper functions that do not affect the failure
4. Preserve semantic markers.
   - dep patterns
   - `distribution_*`
   - partition modes / full-range behavior
   - metadata and contract attributes
5. End in a checked-in regression test whenever possible.

## Lit Test Pattern

```mlir
// RUN: %carts-compile %s --pipeline <stage> | %FileCheck %s
```

Use `%carts-compile`, `%S`, and `FileCheck`. Keep one stage boundary per test unless the bug inherently spans multiple stages.

## Hand-off

- wrong-code bug -> pair with `carts-miscompile-triage`
- runtime-only failure -> pair with `carts-runtime-triage`
- stale-analysis / ordering bug -> pair with `carts-analysis-triage`

## Validation

A reproducer is only done when it still fails before the fix and passes after the fix.
