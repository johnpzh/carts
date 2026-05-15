# Stage Focus Guide

Use this as a starting point for pipeline bisection.

| Symptom | First Stages to Inspect |
|---|---|
| OpenMP structure missing or wrong task boundaries | `sde-planning` (SDE sub-passes), `sde-to-codir`, `codir-to-arts`, `edt-transforms` |
| Wrong loop shape or reordered access pattern | `sde-planning` (inspect SDE sub-passes such as `LoopInterchange`, `DistributionPlanning`, etc. via `--arts-debug`) |
| DB count, acquire mode, or DB shape wrong | `codir-to-arts`, `create-dbs`, `db-opt` |
| Work chunks or distribution attrs look wrong | `sde-planning`, `codir-to-arts`, `post-db-refinement` |
| Full-range/coarse/block decision wrong | `post-db-refinement` |
| Runtime call structure wrong but high-level IR is fine | `pre-lowering`, `arts-to-llvm` |

If you do not know where to begin, dump:

1. `sde-planning`
2. `sde-to-codir`
3. `codir-to-arts`
4. `edt-transforms`
5. `create-dbs`
6. `db-opt`
7. `post-db-refinement`
8. `pre-lowering`
