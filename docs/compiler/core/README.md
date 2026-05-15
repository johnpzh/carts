# ARTS Dialect Optimization Notes

This directory is the compiler-facing home for the `arts` dialect optimization
plan. The source tree still uses `core/` for this area, but the conceptual layer
is the abstract ARTS-machine dialect after SDE/CODIR and before `arts-rt`
lowering. It materializes remaining SDE-authored plans into concrete DB, EDT,
dependency, and epoch structure, then refines that structure with ARTS
analyses. When SDE/CODIR have already produced canonical MU/token/codelet form,
the boundary may create DBs/acquires/EDTs directly and ARTS should preserve
that shape.

For the target `sde -> codir -> arts -> arts-rt` split, see
[`../dialect-layering-vision.md`](../dialect-layering-vision.md).
The target per-dialect docs live under
[`../dialects/arts/`](../dialects/arts/). ARTS-owned analyses are listed in
[`../dialects/arts/analysis.md`](../dialects/arts/analysis.md), and
ARTS-owned optimizations are listed in
[`../dialects/arts/optimizations.md`](../dialects/arts/optimizations.md).

## Boundary

ARTS owns:

- `CreateDbs` as the coarse raw-memref compatibility bridge for cases that have
  not yet become MU tokens.
- DB, EDT, epoch, and distributed-orchestration analyses over ARTS objects.
- DB mode tightening, DB/EDT transforms, dependency-window lowering, and
  contract validation.
- Direct materialization of EDT/DB/Epoch objects from the SDE/CODIR work plan.
- Epoch creation, CPS/continuation structure, and ARTS-side epoch cleanup.

ARTS must not own:

- OpenMP semantic decomposition.
- Memref/linalg legality proofs.
- Physical data partition policy that SDE could prove before CODIR/ARTS
  materialization.
- Runtime-call ABI details that belong in `arts-rt`.

`CreateDbs` is not a data-layout optimizer. Its remaining allowed work is
mechanical coarse raw-memref bridging: allocate one whole-storage DB, acquire
that DB, redirect direct raw load/store operands through `db_ref[0]`, and
preserve the dependency contract. Blocked/tiled raw memrefs must not be
reindexed in ARTS. If a performance fix requires owner dims, tile sizes, loop
interchange, dependency-window legality, or token-local coordinates, the
missing pass belongs in SDE/CODIR.

## Pipeline Spine

This layer begins at `codir-to-arts`, where CODIR codelets become ARTS DB/EDT
objects. SDE operations must not survive into this layer. The primary ARTS
stages are:

```text
edt-transforms
create-dbs
db-opt
post-db-refinement
late-concurrency-cleanup
epochs
pre-lowering
```

`pre-lowering` is a boundary stage: ARTS DB lowering runs there, followed by
`arts-rt` shaped EDT and epoch lowering.

## Optimization Focus

- Prefer direct MU/token lowering. Where raw memrefs remain, ensure
  `CreateDbs` stays coarse-only and does not consume SDE block/tile plans.
- Preserve planned DB/EDT shape through distribution, orchestration, and
  lowering.
- Keep dependency windows local to the planned block/window geometry.
- Use ARTS analyses as the facade for DB/EDT/epoch refinements; do not
  duplicate SDE/CODIR dependence proofs in local heuristics.
- Classify ARTS bottlenecks as DB materialization, EDT shape, dependency
  window, epoch structure, distributed ownership, or contract-validation gaps.

## Verification

ARTS changes currently need focused tests under `lib/arts/dialect/core/test/`,
plus the owning benchmark or pipeline dump when they affect large/64
performance.
