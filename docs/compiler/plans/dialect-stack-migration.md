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

- `sde-planning` is the canonical stage name for OpenMP-to-SDE conversion and
  SDE planning.
- `sde-to-codir` materializes SDE codelet plans as isolated CODIR codelets.
- `codir-to-arts` materializes CODIR codelets as ARTS DB/EDT objects and
  rejects any SDE operation that survived `sde-to-codir`.
- `create-dbs` runs remaining raw-memref DB materialization. Core raw DB
  indexers have been removed; only coarse whole-storage materialization
  remains while direct CODIR coverage is completed.
- `pre-lowering` contains DB, EDT, and epoch lowering.
- Source paths still use `lib/arts/dialect/core`, `lib/arts/dialect/sde`, and
  `lib/arts/dialect/rt`.
- Target source paths are staged under `include/carts/dialect/...` and
  `lib/carts/dialect/...`. CODIR is the first build-visible target dialect
  skeleton: `codir.codelet` and `codir.yield` parse and verify in the
  `carts/dialect/codir` tree. The `convert-sde-to-codir`,
  `codir-codelet-opt`, `verify-codir`, and `convert-codir-to-arts` passes are
  in the default staged pipeline.
  `verify-codir` owns CODIR-only operand-shape checks: deps are memory
  dependencies with one access-mode entry per dep, params are scalar values,
  block args mirror deps then params, and yielded values are scalar. Production
  codelet lowering no longer routes directly from SDE to ARTS: `sde.cu_codelet`
  is consumed by CODIR before ARTS DB acquire and EDT creation.

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

Status: complete.

- Minimal CODIR dialect registration is present.
- SDE-to-CODIR, CODIR verification, and CODIR-to-ARTS passes are registered.
- Focused CODIR parsing, verification, and boundary smoke tests exist.

Exit gate:

- build passes;
- pipeline can include or report the staged CODIR boundary without changing
  existing tests.

### Phase 3: Boundary Split

Status: production codelet path complete; residual cleanup active.

- The default codelet route is now `sde-planning`, `sde-to-codir`, then
  `codir-to-arts`.
- `convert-sde-to-codir` consumes `sde.cu_codelet` and materializes explicit
  CODIR dependencies, scalar params, and token-local memref views.
- `codir-codelet-opt` performs CODIR-owned cleanup of isolated codelet bodies
  after token-local rematerialization and before `verify-codir`.
- `convert-codir-to-arts` lowers CODIR deps and codelet bodies to ARTS
  `db_acquire` and `edt` objects.
- Direct SDE-to-ARTS lowering is no longer part of the live compiler pipeline.
  Codelet-shaped work must pass through `convert-sde-to-codir` and
  `convert-codir-to-arts`; leftovers are boundary errors.
- `VerifyCodir` and `VerifyArtsObjectsOnly` are wired around the boundary.

Exit gate:

- focused codelet tests pass through the split;
- codelet tests cover SDE-to-CODIR and CODIR-to-ARTS; historical SDE-to-ARTS
  tests are removed or rewritten as strict boundary negative tests.

### Phase 4: Source Namespace Cleanup

- Rename namespaces and directory ownership only after behavior is stable.
- Keep changes mechanical and separately reviewable.
- Preserve lit test paths or update test discovery in the same patch.

Exit gate:

- no stale docs or generated resources refer to old conceptual ownership.

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
