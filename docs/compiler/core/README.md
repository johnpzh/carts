# ARTS Core Optimization Notes

This directory is the compiler-facing home for ARTS Core optimization planning.
Core is the ARTS-machine-aware layer after SDE and before RT-shaped lowering.
It materializes remaining SDE-authored plans into concrete DB, EDT, dependency,
and epoch structure, then refines that structure with Core analyses. When SDE
has already produced canonical MU/token/codelet form, the SDE-to-Core boundary
may create DBs/acquires/EDTs directly and Core should preserve that shape.

For the dialect split that removed the Core semantic loop/parallel scheduler
carrier and related scaffolding, see
[`../dialect-layering-vision.md`](../dialect-layering-vision.md).

## Boundary

Core owns:

- `CreateDbs` as the raw-memref compatibility bridge for SDE-authored physical
  DB layouts and dependency slices that have not yet become MU tokens.
- DB, EDT, epoch, and distributed-orchestration analyses over runtime-shaped
  Core objects.
- DB mode tightening, DB/EDT transforms, dependency-window lowering, and
  contract validation.
- Direct materialization of EDT/DB/Epoch objects from the SDE work plan.
- Epoch creation, CPS/continuation structure, and Core-side epoch cleanup.

Core must not own:

- OpenMP semantic decomposition.
- Memref/linalg legality proofs.
- Physical data partition policy that SDE could prove before
  `ConvertSdeToArts`.
- Runtime-call ABI details that belong in RT.

`CreateDbs` is not a data-layout optimizer. Its allowed work is mechanical: consume
an SDE-authored layout/slice, allocate the Core DB, turn element-space slices
into DB-space acquires, rewrite raw memref accesses to the acquired DB view,
and preserve the dependency contract. If a performance fix requires choosing
owner dims, tile sizes, loop interchange, or whether a dependency window is
legal, the missing pass belongs in SDE.

## Pipeline Spine

Core begins when `ConvertSdeToArts` crosses out of the SDE stage. The primary
Core stages are:

```text
edt-transforms
create-dbs
db-opt
post-db-refinement
late-concurrency-cleanup
epochs
pre-lowering
```

`pre-lowering` is a boundary stage: Core DB lowering runs there, followed by
RT-shaped EDT and epoch lowering.

## Optimization Focus

- Prefer direct MU/token lowering. Where raw memrefs remain, ensure
  `CreateDbs` consumes `arts.plan.*` and explicit dependency slices without
  inventing late data partition policy.
- Preserve planned DB/EDT shape through distribution, orchestration, and
  lowering.
- Keep dependency windows local to the planned block/window geometry.
- Use Core analyses as the facade for DB/EDT/epoch refinements; do not duplicate
  SDE dependence proofs in local heuristics.
- Classify Core bottlenecks as DB materialization, EDT shape, dependency window,
  epoch structure, distributed ownership, or contract-validation gaps.

## Verification

Core changes need focused tests under `lib/arts/dialect/core/test/`, plus the
owning benchmark or pipeline dump when they affect large/64 performance.
