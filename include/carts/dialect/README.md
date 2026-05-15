# CARTS Dialect Source Root

This is the target source root for CARTS-owned dialect layers.

Target subdialects:

- `sde`: semantic decomposition, MU/CU/SU planning, and source legality.
- `codir`: isolated codelets, explicit deps/params, and token-local views.
- `arts`: abstract ARTS DB/EDT/epoch object dialect.
- `arts-rt`: runtime ABI bridge before LLVM lowering.

Each subdialect has its own `IR/`, `Analysis/`, `Transforms/`, `Conversion/`,
and `Verify/` include area. Analysis APIs should be dialect-local unless a fact
has been materialized into an explicit IR contract.

The current implementation still compiles from `include/arts/dialect/...`.
Files should move here in staged slices following
`docs/compiler/plans/folder-reorganization.md`.
