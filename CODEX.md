# CARTS Agent Guide

CARTS is an MLIR-based compiler that transforms C/C++ programs with OpenMP
pragmas into task-based parallel executables targeting the ARTS asynchronous
runtime.

This file is intentionally short. The canonical shared agent policy is
[`AGENTS.md`](./AGENTS.md); the live compiler manifest in
[`tools/compile/Compile.cpp`](./tools/compile/Compile.cpp) wins when docs drift.

## Command Policy

- Use `dekk carts ...` for normal project work.
- Use bare `carts ...` only after confirming the project-local wrapper exists
  and is on `PATH`.
- Do not use raw `make`, `ninja`, `cmake`, or `python tools/carts_cli.py` for
  normal project work; Dekk sets up the required environment.

## Essential Commands

```bash
dekk carts doctor
dekk carts build
dekk carts build --arts
dekk carts compile <file> -O3
dekk carts pipeline --json
dekk carts test
dekk carts test --suite e2e
dekk carts lit <file>
dekk carts examples list
dekk carts examples run <name>
dekk carts benchmarks list
dekk carts skills generate
```

## Current Compiler Shape

The project is organized around four dialect layers:

- `sde`: source semantics, OpenMP/source planning, PatternAnalysis, MU/CU/SU
  planning, memref tiling/access windows, reductions, and target-neutral
  scheduling intent.
- `codir`: isolated codelets, explicit deps/params, token-local memref views,
  and codelet-local verification.
- `arts`: abstract DB, EDT, epoch, dependency-slot, placement, and distributed
  ownership objects.
- `arts_rt`: runtime ABI, packing, pointer lowering, runtime calls, and
  LLVM-facing cleanup.

Physical layout:

```text
include/carts/dialect/{sde,codir,arts,arts-rt}/
lib/carts/dialect/{sde,codir,arts,arts-rt}/
lib/carts/dialect/*/test/
tools/compile/Compile.cpp
```

## Engineering Rules

- Passes, ops, attributes, and enums that are part of the IR surface must be
  declared in TableGen. C++ implements behavior behind generated surfaces.
- Do not hardcode project attribute names. Use generated ODS accessors such as
  `op.getStencilMinOffsetsAttrName()` or the owning dialect utility API.
- SDE and CODIR must stay runtime-neutral. ARTS owns abstract orchestration and
  multinode placement. ARTS-RT owns runtime ABI and LLVM-facing lowering.
- Use `carts-check-utils` before adding helper code. Shared helpers belong in
  the narrowest owning dialect `Utils/` area or in a real shared utility.
- Generated diagnostics, dumps, benchmark outputs, and scratch state belong
  under `.carts/outputs/...` or `.carts/sessions/...`, not committed fixtures.

## Verification

Match verification to the change:

- Skill/doc/config changes: `dekk carts skills generate` plus discovery checks.
- Single pass change: focused `dekk carts lit <test>`, then the owning dialect
  suite.
- Pipeline or shared analysis change: `dekk carts test` and affected e2e cases.
- Runtime or distributed change: rebuild ARTS as needed, run local first, then
  the smallest multinode or distributed benchmark that exercises the path.
