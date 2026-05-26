---
name: carts-debug
description: Use when starting an unclassified CARTS debug session for crashes, wrong results, hangs, logs, counters, or pipeline artifacts.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [<input-file>]
parameters:
  - name: bug_report
    type: str
    gather: "Describe the bug, expected behavior, and error messages"
  - name: affected_files
    type: str
    gather: "Affected files or tests, or 'unknown'"
---

# CARTS Debug

This is the canonical entry for an unclassified CARTS failure. Once classified,
switch to the narrower triage skill instead of duplicating that workflow here.

## Hard Rule

- Classify the failure before changing code or refreshing fixtures.
- Use `dekk carts ...` commands and keep artifacts under `.carts/outputs`.
- Preserve crash artifacts and exact command lines.
- Use the earliest stable pipeline stage that reproduces the symptom.
- Route specialized cases to sibling skills.

## Procedure

1. Record the observed symptom, expected behavior, input, command, and build
   mode.
2. Classify the primary failure:
   - wrong output or semantic drift -> [[carts-miscompile-triage]]
   - runtime crash, hang, route, epoch, or counter issue -> [[carts-runtime-triage]]
   - multinode, ownership, or distributed DB issue -> [[carts-distributed-triage]]
   - stale graph, invalidation, or pass-ordering issue -> [[carts-analysis-triage]]
   - benchmark failure or regression -> [[carts-benchmark-triage]]
   - heuristic decision explanation -> [[carts-heuristic-explain]]
   - benchmark-to-test shrinking -> [[carts-reproducer]]
   - stage-to-stage IR comparison -> [[carts-stage-diff]]
3. For compile-time failures, inspect stages with `dekk carts pipeline --json`
   and `dekk carts compile <file> --pipeline=<stage> -O3`.
4. For MLIR inputs, dump stages with:
   ```bash
   dekk carts compile <file.mlir> --all-pipelines -o .carts/outputs/stages/<topic>/
   ```
5. For C/C++ inputs, compile once to get the Polygeist MLIR, then inspect the
   relevant CARTS stages.
6. For crashes, save core/logs/backtraces under `.carts/outputs/<topic>/` and
   verify the core is nonempty before relying on it.
7. For runtime progress, rebuild ARTS only as needed with `dekk carts build
   --arts --debug <level>` or `--counters <level>`, then restore release mode.
8. Before fixing source, invoke [[carts-session-start]] for the classified
   compiler surface.

## Shared References

Use these when the classification is unclear:

- `references/failure-signatures.md`
- `references/codepath-map.md`
- `references/command-patterns.md`
- `references/stage-ownership.md`

## Required Answer

State the classification, reproducing command, first bad stage or runtime
boundary, artifact path, and the narrower skill used next.
