# Dialect Charter

The CARTS compiler has three dialects. Each owns a clear slice of compiler responsibility. Pass placement is decided against this charter; if a placement decision feels arbitrary, the charter is wrong (update it) or the pass is wrong (move it).

## Source of truth

The authoritative documents are:

- `docs/architecture/sde-dialect.md` — SDE charter
- `docs/architecture/arts-rt-dialect.md` — RT charter
- `docs/architecture/op-classification.md` — which op belongs where
- `docs/architecture/pass-placement.md` — placement rules
- `docs/architecture/architecture-reaudit-2026-04-11.md` — most recent audit

This file is a fast-lookup distillation. If it disagrees with the docs above, the docs win — update this file to match.

## The three charters

### SDE (`arts_sde`) — Semantic planning

**Owns:** semantic decisions about what to do, irrespective of how the runtime realizes them.

- Loop classification: elementwise / stencil / matmul / reduction / wavefront / Jacobi / mixed
- Reduction strategy: atomic / tree / local_accumulate
- Scope: local vs distributed
- Schedule: static / dynamic / guided
- Chunk size for dynamic/guided
- Tile geometry, halo geometry (when stamped as a contract)
- Vectorization hints

**IR level:** memref + scf + transient `linalg.generic` / `tensor` carriers. Carriers are created during analysis and erased before SDE→ARTS conversion. No carriers escape the SDE window.

**Forbidden:**
- Touching any `arts.*` op
- Including any header from `core/Analysis` or `core/Transforms`
- Baking ARTS-specific runtime semantics into decisions (decisions should be expressible as portable contracts)

### Core (`arts.*`) — ARTS structural realization

**Owns:** structural realization of SDE contracts in ARTS-runtime-shaped IR.

- DBs: alloc, partition, distributed-ownership marking, mode tightening, scratch elimination
- EDTs: structural opt, ICM, distribution contract realization, orchestration
- Implementation loops: local `scf.for` control flow inside concrete Core objects
- Epochs: creation, CPS scheduling, optimization
- Contract attributes: encode SDE decisions for downstream consumption

**IR level:** ARTS structural — regions, DBs, partitions, contracts. No tensor/linalg ops post-stage-3.

**Forbidden:**
- Re-deriving classifications SDE already stamped (Invariant 5)
- Emitting `arts_rt.*` ops before stage 16/17
- Performing semantic analysis from scratch (consume SDE contracts instead)

### RT (`arts_rt.*`) — Runtime API mapping

**Owns:** thin 1:1 mapping to ARTS C runtime calls + final low-level passes immediately before LLVM.

- 14 ops mapping directly to runtime: `arts_rt.edt_create`, `record_dep`, `dep_gep`, `state_pack`, `create_epoch`, `wait_on_epoch`, etc.
- 4 passes: `DataPtrHoisting`, `GuidRangCallOpt`, `RuntimeCallOpt`, `VerifyLowered`
- Pure code-gen intermediates with no semantic meaning of their own

**IR level:** LLVM-near, flat. No regions, no contracts.

**Forbidden:**
- Any analysis or contract reading
- Optimization passes that need to understand DB/EDT semantics (those belong in core)

## Five hard invariants

These are placement rules. If you are tempted to break one, update the charter first.

1. **If a pass *decides*, it lives in SDE — or consumes an SDE contract from core.** No core pass should make a fresh classification decision.

2. **Cross-dialect op creation only at stage boundaries.** SDE→ARTS at stage 3 (ConvertSdeToArts), core→RT at stage 16/17 (EdtLowering / EpochLowering). No pass creates ops outside its dialect, except those boundary conversion passes.

3. **RT has zero semantic deps on core/SDE.** RT passes only touch `arts_rt.*` ops. If an RT pass needs to inspect core semantics, the pass belongs in core.

4. **SDE does not include any header from `core/Analysis/` or `core/Transforms/`.** Mechanical check; grep for it.

5. **Cost-model-driven decisions belong in the *decision-owner*, not the *realizer*.** SDE stamps a contract (e.g., tile geometry); core consumes it. If both compute the same thing, one is wrong.

## Currently-known violations (2026-04-11 audit)

These do not block correctness but should be cleaned up in Phase 9. Cite when delegating fixes; a regression here is not a new bug, it is technical debt.

| # | Violation | Files | Fix |
|---|---|---|---|
| 1 | Wavefront family detection in core, should be SDE (Invariant 5) | `lib/arts/dialect/core/Analysis/heuristics/StructuredKernelPlanAnalysis.cpp`, `core/Transforms/dep/Seidel2DWavefrontPattern.cpp` | Move family + tile geometry into `SdeStructuredSummaries` or new `SdeWavefrontAnalysis`; refactor core realizers to consume the contract. |
| 2 | KernelTransforms re-detects elementwise/stencil/matmul (Invariant 5) | `lib/arts/dialect/core/Transforms/kernel/KernelTransforms.cpp` | Delete `ElementwisePipelinePattern` (redundant); refactor `StencilTilingNDPattern` and `MatmulReductionPattern` to consume SDE contracts. |
| 3 | `RaiseMemRefDimensionality` is a Polygeist→ARTS conversion, lives in `core/Transforms/` (Invariant 1) | `lib/arts/dialect/core/Transforms/RaiseMemRefDimensionality.cpp` | Move to `core/Conversion/PolygeistToArts/`. |
| 4 | DepTransforms creates `EpochOp`/`EdtOp` from re-detected wavefront/Jacobi patterns (Invariants 1 & 5) | `lib/arts/dialect/core/Transforms/dep/DepTransforms.cpp`, `core/Transforms/dep/Seidel2DWavefrontPattern.cpp` | Enhance `SdeStructuredSummaries`; refactor `DepTransforms` to consume hints. |
| 5 | Doc disagreement: `op-classification.md` says `arts.lowering_contract` and `arts.omp_dep` should migrate to SDE; `pass-placement.md` says the live pipeline keeps planning inside `openmp-to-arts`. | `docs/architecture/op-classification.md` vs `docs/architecture/pass-placement.md` | `pass-placement.md` is current; the migration is aspirational and tracked under violation #1. |

## Open questions (Phase 0 / task #1)

The user must decide these before further restructuring. Answers go to `docs/architecture/charter-decisions.md` (created on first answer).

1. **DestinationStyleOpInterface on SDE ops.** Make CU/SU ops implement `ins`/`outs` for tensor composition, or keep transient `linalg.generic` carriers? Recommendation: keep transient until benchmarks are green, then upgrade. (Effort if upgrading: 8–10h.)

2. **Scope of `SdeStructuredSummaries`.** Does it own ALL semantic planning (incl. wavefront/Jacobi), or split into a separate `SdeWavefrontAnalysis`? Recommendation: own all; otherwise core keeps re-deriving and Invariant 5 stays broken.

3. **Plug `ARTSCostModel` into all heuristics.** ~20 hardcoded thresholds remain in `PartitioningHeuristics`, `DistributionHeuristics`, `DbHeuristics`. Effort 12–16h. Recommendation: defer until phase 8 is green; cost model only matters once structural plumbing is correct.

4. **Backend-neutral SDE narrative.** Docs aspire to multi-backend (Legion / StarPU / GPU) but namespace, schedule kinds, and conversion target are all ARTS-tied. Keep aspiration or reframe as ARTS-optimized? Recommendation: reframe; pretending otherwise misleads contributors.

5. **LoopReordering migration to SDE.** Blocked on (1). Prioritize the DSI + `SdeLoopInterchange` stack, or defer? Recommendation: defer.

## Quick lookup: "where does X belong?"

| If X is… | It belongs in… |
|---|---|
| A loop classification (stencil, matmul, reduction, etc.) | SDE (`SdeStructuredSummaries`) |
| A reduction strategy choice (atomic / tree / accumulate) | SDE (`ReductionStrategy`) |
| A scope choice (local vs distributed) | SDE (`ScopeSelection`, `DistributionPlanning`) |
| A schedule choice (static / dynamic / guided) | SDE (`ScheduleRefinement`) |
| Tile / chunk / halo geometry **as a contract** | SDE (decision); core (realization) |
| DB allocation / acquire / release / partitioning | core |
| EDT structural rewrite, fusion, distribution | core |
| Epoch creation, CPS scheduling | core |
| `arts_rt.edt_create` argument lowering | RT |
| LLVM-near final cleanup (DataPtrHoisting, GuidRangCallOpt) | RT |
| Polygeist→MLIR frontend bridge | core/Conversion (NOT core/Transforms) |
| `linalg.*`, `tensor.*` ops | SDE only, transient |
