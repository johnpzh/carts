# CARTS Dialect Map

## SDE Dialect: `arts_sde`

Paths:

- `include/arts/dialect/sde/IR/`
- `lib/arts/dialect/sde/`
- tests: `lib/arts/dialect/sde/test/`

Purpose: runtime-agnostic semantic decomposition for OpenMP regions, state,
tensor/memref transitions, reductions, scheduling plans, codelets, and
distribution decisions.

Limits: SDE should not encode ARTS runtime call shape or LLVM-facing ABI
details. It should preserve high-level semantics so later dialects do not need
to rediscover OpenMP intent.

Important transform areas:

- `state/` - representation changes, raising/lowering, token mode.
- `dep/` - structural and dependency transforms.
- `effect/` - scheduling, distribution, fusion/vectorization decisions.
- `Verify/` - SDE boundary contracts.

## Core ARTS Dialect: `arts`

Paths:

- `include/arts/dialect/core/IR/`
- `lib/arts/dialect/core/`
- tests: `lib/arts/dialect/core/test/`

Purpose: high-level ARTS orchestration IR: EDTs, DBs, epochs, `arts.for`,
barriers, atomics, runtime queries, and lowering contracts.

Limits: Core should not become a runtime ABI shim or a place to patch frontend
semantic loss. It should own orchestration invariants and analysis-backed
decisions.

Important areas:

- `Analysis/db`, `Analysis/edt`, `Analysis/loop`, `Analysis/heuristics`.
- `Transforms/db`, `Transforms/edt`, `Transforms/epoch`, `Transforms/loop`.
- `Conversion/SdeToArts`, `Conversion/ArtsToRt`, `Conversion/ArtsToLLVM`.

Use `AnalysisManager` accessors for DB/EDT/loop analyses. Do not reach into
graphs directly from passes.

## Runtime Dialect: `arts_rt`

Paths:

- `include/arts/dialect/rt/`
- `lib/arts/dialect/rt/`
- tests: `lib/arts/dialect/rt/test/`

Purpose: flat runtime-facing bridge before LLVM. Runtime ops model epoch
creation/wait, EDT launch, param pack/unpack, dependency records, DB acquire/GEP,
state pack/unpack, dep bind/forward, and call-friendly values.

Limits: RT should not introduce new high-level semantics, scheduling policy, or
analysis decisions. If a fix requires understanding OpenMP intent, DB/EDT graph
state, or partition ownership, it likely belongs in SDE or Core.

## TableGen Boundaries

Edit dialect-specific files under `include/arts/dialect/*`. Old top-level
`include/arts/Ops.td`, `Dialect.td`, `Attributes.td`, and `Types.td` are legacy
forwarding files, not the place for new dialect definitions.
