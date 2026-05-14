# ARTS Core Optimization Notes

This directory is the compiler-facing home for ARTS Core optimization planning.
Core is the ARTS-machine-aware layer after SDE and before RT-shaped lowering.
It materializes SDE-authored plans into concrete DB, EDT, dependency, and epoch
structure, then refines that structure with Core analyses.

For the dialect split that removed the Core loop carrier, semantic
`arts.edt<parallel>`, `ForLowering`, and related scheduler scaffolding, see
[`../dialect-layering-vision.md`](../dialect-layering-vision.md).

## Boundary

Core owns:

- `CreateDbs` and direct materialization of SDE-authored physical DB layouts.
- DB, EDT, epoch, and distributed-orchestration analyses over runtime-shaped
  Core objects.
- DB mode tightening, DB/EDT transforms, dependency-window lowering, and
  contract validation.
- Direct materialization of EDT/DB/Epoch objects from the SDE work plan.
- Epoch creation, CPS/continuation structure, and Core-side epoch cleanup.

Core must not own:

- OpenMP semantic decomposition.
- Tensor/linalg legality proofs.
- Physical tensor partition policy that SDE could prove before
  `ConvertSdeToArts`.
- Runtime-call ABI details that belong in RT.

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

- Ensure `CreateDbs` consumes `arts.plan.*` without inventing late tensor
  partition policy.
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
