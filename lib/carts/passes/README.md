# CARTS Compiler Passes

Pass implementations organized by category:

| Directory | Purpose |
|-----------|---------|
| `transforms/` | Lowering and conversion passes (OMP-to-ARTS, EDT/DB/epoch lowering, ARTS-to-LLVM) |
| `opt/` | Optimization passes (DB, EDT, epoch, codegen, loop, kernel, general) |
| `verify/` | Pipeline verification barriers (assert IR invariants between stages) |

See `CLAUDE.md` at the project root for pipeline ordering and pass conventions.
