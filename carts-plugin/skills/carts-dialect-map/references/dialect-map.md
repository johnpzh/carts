# CARTS Dialect Map

## SDE Dialect: `sde`

Paths:

- `include/carts/dialect/sde/IR/`
- `lib/carts/dialect/sde/`
- tests: `lib/carts/dialect/sde/test/`

Purpose: runtime-agnostic semantic decomposition for OpenMP regions, memref
state normalization, reductions, scheduling plans, and distribution decisions.

Limits: SDE should not encode ARTS runtime call shape or LLVM-facing ABI
details. It should preserve high-level semantics so later dialects do not need
to rediscover OpenMP intent.

Important transform areas:

- `state/` - representation changes, PatternAnalysis, MU materialization,
  scalar forwarding.
- `dep/` - structural and dependency transforms.
- `effect/` - scheduling, distribution, fusion/vectorization decisions.
- `Verify/` - SDE boundary contracts.

## CODIR Dialect: `codir`

Paths:

- `include/carts/dialect/codir/IR/`
- `lib/carts/dialect/codir/`
- tests: `lib/carts/dialect/codir/test/`

Purpose: isolated codelet bodies with explicit dependencies, params, and
codelet-local verification. Sits between SDE (which owns OpenMP semantics)
and ARTS (which owns abstract orchestration).

Limits: CODIR should not encode runtime ABI or planning heuristics. SDE
hands codelets here for isolation; ARTS picks them up for orchestration.

Important areas:

- `IR/` - CodirDialect, CodirOps (codelet, dep slice).
- `Conversion/SdeToCodir/`, `Conversion/CodirToArts/`.
- `Transforms/` - CodirCodeletOpt, VerifyCodir.
- `Utils/` - CodeletABIUtils.

## ARTS Dialect: `arts`

Paths:

- `include/carts/dialect/arts/IR/`
- `lib/carts/dialect/arts/`
- tests: `lib/carts/dialect/arts/test/`

Purpose: high-level ARTS orchestration IR: EDTs, DBs, epochs, implementation
`scf.for` loops inside tasks/dispatch, barriers, atomics, runtime queries, and
lowering contracts.

Limits: ARTS should not become a runtime ABI shim or a place to patch
frontend semantic loss. It should own orchestration invariants and
analysis-backed decisions.

Important areas:

- `Analysis/db`, `Analysis/edt`, `Analysis/loop`, `Analysis/heuristics`.
- `Transforms/db`, `Transforms/edt`, `Transforms/epoch`, `Transforms/verify`.
- `Utils/` - DbUtils, EdtUtils, LoweringContractUtils,
  PartitionPredicates, BlockedAccessUtils, ARTSCostModel.

Use `AnalysisManager` accessors for DB/EDT/loop analyses. Do not reach into
graphs directly from passes.

## ARTS-RT Dialect: `arts_rt`

Paths:

- `include/carts/dialect/arts-rt/`
- `lib/carts/dialect/arts-rt/`
- tests: `lib/carts/dialect/arts-rt/test/`

Purpose: flat runtime-facing bridge before LLVM. Runtime ops model epoch
creation/wait, EDT launch, param pack/unpack, dependency records, DB acquire/GEP,
state pack/unpack, dep bind/forward, and call-friendly values.

Limits: ARTS-RT should not introduce new high-level semantics, scheduling
policy, or analysis decisions. If a fix requires understanding OpenMP intent,
DB/EDT graph state, or partition ownership, it likely belongs in SDE, CODIR,
or ARTS.

Important areas:

- `Conversion/ArtsToRt` - lower chosen ARTS objects into runtime-shaped IR.
- `Conversion/ArtsRtToLLVM` - lower runtime-shaped IR into LLVM dialect
  runtime calls.
- `Transforms/` - LLVM-facing cleanup, runtime-call optimization, data pointer
  hoisting, alias scopes, loop hints, and lowered-form verification.

## TableGen Boundaries

Edit dialect-specific files under `include/carts/dialect/*`. Each dialect
owns its own `IR/*.td` (Dialect, Ops, Attributes, Types) and `Transforms/Passes.td`.
