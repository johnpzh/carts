---
name: carts-dialect-map
description: Use when locating CARTS dialect code, tracing SDE/CODIR/ARTS/ARTS-RT op lifecycle, choosing where a compiler change belongs, or checking dialect boundary invariants.
---

# CARTS Dialect Map

Read `references/dialect-map.md` when deciding where an operation, pass,
analysis, or test belongs.

## Boundary Rule

- SDE: OpenMP semantic decomposition and runtime-agnostic scheduling/state.
- CODIR: isolated codelet bodies with explicit deps/params and codelet-local
  verification.
- ARTS: DB/EDT/epoch orchestration and compiler analyses.
- ARTS-RT: runtime-call-shaped IR before LLVM lowering.

Production fixes require understanding the function and limits of the dialect
being changed. Do not patch a symptom in a downstream dialect when the semantic
owner is upstream.

## Trace Workflow

1. Find the op definition in the dialect-specific TableGen file.
2. Find creation sites with `rg "OpName|op mnemonic" include lib tools docs`.
3. Check where it is transformed, verified, and erased/lowered.
4. Confirm the stage using `carts-pipeline-map`.
5. State the dialect contract and what that dialect must not own.
6. Add tests in the owning dialect test directory.

Do not put new semantics in ARTS-RT just because a runtime call is nearby;
ARTS-RT is for lowering-ready runtime shape.
