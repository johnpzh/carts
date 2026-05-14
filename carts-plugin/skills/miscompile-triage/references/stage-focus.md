# Stage Focus Guide

Use this as a starting point for pipeline bisection.

| Symptom | First Stages to Inspect |
|---|---|
| OpenMP structure missing or wrong task boundaries | `openmp-to-arts` (SDE sub-passes), `edt-transforms` |
| Wrong loop shape or reordered access pattern | `openmp-to-arts` (inspect SDE sub-passes — `SdeLoopInterchange`, `SdeScopeSelection`, etc. — via `--arts-debug`) |
| DB count, acquire mode, or DB shape wrong | `create-dbs`, `db-opt` |
| Work chunks or distribution attrs look wrong | `openmp-to-arts`, `post-db-refinement` |
| Full-range/coarse/block decision wrong | `post-db-refinement` |
| Runtime call structure wrong but high-level IR is fine | `pre-lowering`, `arts-to-llvm` |

If you do not know where to begin, dump:

1. `openmp-to-arts`
2. `edt-transforms`
3. `create-dbs`
4. `db-opt`
5. `post-db-refinement`
6. `pre-lowering`
