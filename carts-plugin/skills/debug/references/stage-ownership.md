# Stage Ownership

Use this when deciding where to start a bisection.

| Stage | Primary Question |
|---|---|
| `openmp-to-arts` | Did OpenMP structure become the right SDE plan and Core EDT/DB/epoch shape? Did SDE pattern discovery (`DistributionPlanning`, `LoopInterchange`, ...) stamp the right contracts before `ConvertSdeToArts`? |
| `edt-transforms` | Did EDT structure/invariant motion change dependencies or task boundaries? |
| `create-dbs` | Did the right allocations become DBs with plausible initial partition hints? |
| `db-opt` | Did load/store reality produce the correct in/out/inout modes? |
| `post-db-refinement` | Did validation or cleanup destroy a valid contract? |
| `pre-lowering` | Is the runtime-facing IR still semantically correct before LLVM conversion? |
| `arts-to-llvm` | Did final lowering or route emission break an otherwise-correct plan? |

Fast default for unknown compiler failures:

1. `create-dbs`
2. `db-opt`
3. `post-db-refinement`
4. `pre-lowering`
