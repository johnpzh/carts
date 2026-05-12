---
name: carts-pipeline-map
description: Use when inspecting CARTS pipeline stages, pass order, stage tokens, start-from/pipeline ranges, epilogues, stale pipeline docs, or ownership of a transformation stage.
---

# CARTS Pipeline Map

Read `references/pipeline-stages.md` before relying on stage names or pass
order. The compiler source and live manifest beat docs.

## Workflow

1. Query the live pipeline, including pass labels and dependency edges.
   ```bash
   dekk carts pipeline --json
   ```
2. Use `--pipeline=<stage>` to stop after a stage.
3. Use `--start-from=<stage>` only with core stages, and ensure the start stage
   is not after the stop stage.
4. Use `dekk carts compile <file> -O3 --all-pipelines -o .carts/outputs/stages/<case>`
   when stage output, pass dumps, or dialect-boundary artifacts are needed.
5. When updating docs or skills, compare against `tools/compile/Compile.cpp`.

## Handoffs

- Semantic wrong output: `carts-miscompile-triage`.
- Runtime crash or hang: `carts-runtime-triage`.
- Operation lifecycle: `carts-dialect-map`.
- New or moved pass: `carts-pass-dev`.
