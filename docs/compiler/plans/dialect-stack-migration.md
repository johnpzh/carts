# Dialect Stack Migration Plan

## Objective

Move CARTS from the current transitional implementation toward:

```text
Polygeist -> sde -> codir -> arts -> arts-rt -> LLVM
```

This plan owns the driver shape, naming, dialect boundaries, and namespace
migration. It does not own codelet ABI details, token-local memref rewrites, or
benchmark-specific performance policy; those are covered by sibling subplans.
Physical source-tree layout is owned by
[`folder-reorganization.md`](./folder-reorganization.md).

## Current Surface

Current live driver stages are defined in `tools/compile/Compile.cpp` and shown
by `dekk carts pipeline --json`.

Important current names:

- `openmp-to-arts` contains SDE planning and `ConvertSdeToArts`.
- `create-dbs` runs the raw-memref compatibility DB bridge.
- `pre-lowering` contains DB, EDT, and epoch lowering.
- Source paths still use `lib/arts/dialect/core`, `lib/arts/dialect/sde`, and
  `lib/arts/dialect/rt`.
- Target source paths are staged under `include/carts/dialect/...` and
  `lib/carts/dialect/...`. CODIR is the first build-visible target dialect
  skeleton: `codir.codelet` and `codir.yield` parse and verify in the
  `carts/dialect/codir` tree. The placeholder `convert-sde-to-codir`,
  `verify-codir`, and `convert-codir-to-arts` passes are registered for focused
  pass-pipeline tests. `verify-codir` owns the first CODIR-only operand-shape
  checks: deps are memory dependencies with one access-mode entry per dep,
  params are scalar values, block args mirror deps then params, and yielded
  values are scalar. Lowering behavior still routes through the current direct
  SDE-to-ARTS compatibility conversion, but `convert-sde-to-codir` has focused
  `sde.cu_codelet` smoke conversions for whole-storage MU tokens and sliced
  token-local views represented as CODIR dep memrefs.

## Target Contract

- SDE is a CARTS-owned semantic planning dialect.
- CODIR is a CARTS-owned codelet dialect.
- `arts` is the abstract ARTS-machine dialect.
- `arts-rt` is the runtime ABI bridge.
- Current `core/` and `rt/` source paths may remain during migration, but docs
  and new code should use the target conceptual names.

## Phases

### Phase 1: Staged Naming

- Keep existing build paths stable.
- Update docs, generated agent resources, and comments to use SDE/CODIR/ARTS
  and ARTS-RT terminology.
- Avoid broad namespace renames until dialect boundaries are mechanically
  stable.

Exit gate:

- docs and generated resources point at the master plan;
- `dekk carts pipeline --json` is documented as current implementation, not the
  target stack.

### Phase 2: CODIR Skeleton

- Add minimal CODIR dialect registration.
- Add placeholder pass declarations for SDE-to-CODIR, CODIR verification, and
  CODIR-to-ARTS.
- Keep behavior identical by routing unsupported paths through the current
  direct conversion until real CODIR lowering exists.

Exit gate:

- build passes;
- pipeline can include or report the staged CODIR boundary without changing
  existing tests.

### Phase 3: Boundary Split

- Move transitional `sde.cu_codelet` lowering through CODIR.
- Split `ConvertSdeToArts` into:
  - SDE-to-CODIR plan materialization;
  - CODIR-to-ARTS object materialization.
- Rename verification entry points conceptually:
  - `VerifyCodir`;
  - `VerifyArtsObjectsOnly`.

Exit gate:

- focused codelet tests pass through the split;
- existing SDE-to-ARTS tests are either migrated or explicitly marked as
  compatibility tests.

### Phase 4: Source Namespace Cleanup

- Rename namespaces and directory ownership only after behavior is stable.
- Keep changes mechanical and separately reviewable.
- Preserve lit test paths or update test discovery in the same patch.

Exit gate:

- no stale docs or generated resources refer to old conceptual ownership except
  when describing current source-tree compatibility.

## Risks

- Renaming too early can hide behavior regressions in build churn.
- Keeping old names too long can confuse ownership. The compromise is explicit:
  target names in docs and new design, old source paths until migration is
  mechanically safe.

## Verification

- `dekk carts build`
- `dekk carts pipeline --json`
- focused dialect registration tests
- focused SDE/CODIR/ARTS conversion lit tests
- `dekk carts skills generate` after command or architecture docs change
