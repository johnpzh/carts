---
name: carts-cli
description: Use when asking about CARTS commands, environment setup, wrappers, compile flags, pipeline inspection, examples, benchmarks, generated skills, or how to run project lifecycle tasks.
---

# CARTS CLI

Use the project command model first. Read `references/command-model.md` when a
task depends on entrypoints, wrappers, generated skills, or command drift.

## Default Rule

Use `dekk carts ...` unless you have confirmed the project-local `carts`
wrapper is available. Do not use raw build tools or
`python tools/carts_cli.py` for normal project work.

## Quick Commands

```bash
dekk carts doctor
dekk carts build
dekk carts compile <file> -O3
dekk carts pipeline --json
dekk carts test
dekk carts lit <test.mlir>
dekk carts examples list
dekk carts examples run <name>
dekk carts benchmarks list
dekk carts skills generate
```

## Workflow

1. Run `dekk carts <command> --help` before using unfamiliar flags.
2. Prefer live command output over stale docs.
3. For pipeline facts, exact pass labels, and dependencies, check
   `dekk carts pipeline --json`.
4. Report the exact command run and whether it passed, failed, or was not run.
