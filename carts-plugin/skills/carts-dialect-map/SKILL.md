---
name: carts-dialect-map
description: Use when locating CARTS dialect code, tracing op lifecycle, choosing owners, or checking boundary invariants.
---

# CARTS Dialect Map

Read `references/dialect-map.md` when deciding where an operation, pass,
analysis, or test belongs.

## Hard Rule

- SDE owns OpenMP semantics and runtime-agnostic scheduling/state.
- CODIR owns isolated codelets with explicit deps/params and local verification.
- ARTS owns DB/EDT/epoch orchestration, analyses, placement, and ownership.
- ARTS-RT owns lowering-ready runtime ABI shape before LLVM.
- Do not patch downstream symptoms when the semantic owner is upstream.

## Procedure

1. Find the op definition in the dialect-specific TableGen file.
2. Find creation sites with `rg "OpName|op mnemonic" include lib tools docs`.
3. Check where it is transformed, verified, and erased or lowered.
4. Confirm stage ownership with [[carts-pipeline-map]] and the live compiler.
5. State the dialect contract and what that dialect must not own.
6. Add tests in the owning dialect test directory.
7. For duplicated attribute enums or convert-only boundaries, invoke
   [[carts-attr-consolidation]].

## Shared Attribute Boundary

A `convert<Name>` switch that maps identical enum cases across dialects is a
boundary smell, not a durable API. Hoist identical enums instead of preserving
rename-only conversion code.

## Required Answer

State the owning dialect, lifecycle path, boundary rule, first affected pipeline
stage, and test location.
