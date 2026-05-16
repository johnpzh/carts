# Dialect Charter

The CARTS compiler is migrating around four dialect layers: SDE, CODIR, ARTS,
and ARTS-RT. Each owns a clear slice of compiler responsibility. Pass
placement is decided against this charter; if a placement decision feels
arbitrary, the charter is wrong (update it) or the pass is wrong (move it).

## Source of truth

The authoritative documents are:

- `docs/architecture/sde-dialect.md` — SDE charter
- `docs/architecture/arts-rt-dialect.md` — RT charter
- `docs/architecture/op-classification.md` — which op belongs where
- `docs/architecture/pass-placement.md` — placement rules
- `docs/architecture/architecture-reaudit-2026-04-11.md` — most recent audit

This file is a fast-lookup distillation. If it disagrees with the docs above, the docs win — update this file to match.

## The four charters

### SDE (`sde`) — Semantic planning

**Owns:** semantic decisions about what to do, irrespective of how the runtime realizes them.

- Loop classification: elementwise / stencil / matmul / reduction / wavefront / Jacobi / mixed
- Reduction strategy: atomic / tree / local_accumulate
- Schedule: static / dynamic / guided
- Chunk size for dynamic/guided
- Tile geometry, halo geometry (when stamped as a contract)
- Vectorization hints

**IR level:** memref + scf + transient `linalg.generic` / `tensor` carriers. Carriers are created during analysis and erased before CODIR/ARTS materialization. No carriers escape the SDE window.

**Forbidden:**
- Touching any `arts.*` op
- Including any header from `core/Analysis` or `core/Transforms`
- Baking ARTS-specific runtime semantics into decisions (decisions should be expressible as portable contracts)

### CODIR (`codir.*`) — Codelet isolation

**Owns:** isolated codelet ABI between SDE planning and ARTS object creation.

- Codelet boundaries with explicit deps and scalar params
- Token-local memref views
- Codelet-local verification and body isolation
- Mechanical handoff to ARTS `db_acquire` and `edt`

**IR level:** isolated codelet regions. No implicit captures from above.

**Forbidden:**
- Choosing source-level tiling or scheduling policy
- Choosing ARTS runtime topology or depv layout
- Recovering deps/params by scanning outer SSA uses

### ARTS (`arts.*`) — Abstract runtime object realization

**Owns:** structural realization of SDE/CODIR contracts as abstract ARTS
objects.

- DBs: alloc, partition, distributed-ownership marking, mode tightening, scratch elimination
- EDTs: structural opt, ICM, distribution contract realization, orchestration
- Scope: local vs distributed, using abstract-machine analysis
- Implementation loops: local `scf.for` control flow inside concrete Core objects
- Epochs: creation, CPS scheduling, optimization
- Contract attributes: encode SDE decisions for downstream consumption

**IR level:** ARTS structural: regions, DBs, partitions, contracts. No
tensor/linalg carriers survive the SDE-to-CODIR / CODIR-to-ARTS boundary.

**Forbidden:**
- Re-deriving classifications SDE already stamped (Invariant 5)
- Emitting `arts_rt.*` ops before stage 16/17
- Performing semantic analysis from scratch (consume SDE contracts instead)

### ARTS-RT (`arts_rt.*`) — Runtime API mapping

**Owns:** thin 1:1 mapping to ARTS C runtime calls + final low-level passes immediately before LLVM.

- 14 ops mapping directly to runtime: `arts_rt.edt_create`, `record_dep`, `dep_gep`, `state_pack`, `create_epoch`, `wait_on_epoch`, etc.
- 4 passes: `DataPtrHoisting`, `GuidRangCallOpt`, `RuntimeCallOpt`, `VerifyLowered`
- Pure code-gen intermediates with no semantic meaning of their own

**IR level:** LLVM-near, flat. No regions, no contracts.

**Forbidden:**
- Any analysis or contract reading
- Optimization passes that need to understand DB/EDT semantics (those belong in ARTS)

## Five hard invariants

These are placement rules. If you are tempted to break one, update the charter first.

1. **If a pass *decides*, it lives in SDE — or consumes an SDE contract from ARTS.** No ARTS pass should make a fresh classification decision.

2. **Cross-dialect op creation only at stage boundaries.** SDE planning feeds CODIR at `sde-to-codir`; CODIR creates ARTS objects at `codir-to-arts`; any SDE op left after CODIR conversion fails verification. ARTS lowers to ARTS-RT in pre-lowering (`EdtLowering` / `EpochLowering`). No pass creates ops outside its dialect, except those boundary conversion passes.

3. **ARTS-RT has zero semantic deps on ARTS/SDE.** ARTS-RT passes only touch `arts_rt.*` ops. If an ARTS-RT pass needs to inspect ARTS semantics, the pass belongs in ARTS.

4. **SDE does not include any header from `core/Analysis/` or `core/Transforms/`.** Mechanical check; grep for it.

5. **Cost-model-driven decisions belong in the *decision-owner*, not the *realizer*.** SDE stamps a contract (e.g., tile geometry); core consumes it. If both compute the same thing, one is wrong.

## Currently-known violations (2026-04-11 audit)

These do not block correctness but should be cleaned up in Phase 9. Cite when delegating fixes; a regression here is not a new bug, it is technical debt.

| # | Violation | Files | Fix |
|---|---|---|---|
| 1 | Wavefront family detection in core, should be SDE (Invariant 5) | `lib/carts/dialect/arts/Analysis/heuristics/StructuredKernelPlanAnalysis.cpp`, `core/Transforms/dep/Seidel2DWavefrontPattern.cpp` | Move family + tile geometry into `PatternAnalysis` or a later SDE wavefront-planning pass; refactor core realizers to consume the contract. |
| 2 | KernelTransforms re-detects elementwise/stencil/matmul (Invariant 5) | `lib/carts/dialect/arts/Transforms/kernel/KernelTransforms.cpp` | Delete `ElementwisePipelinePattern` (redundant); refactor `StencilTilingNDPattern` and `MatmulReductionPattern` to consume SDE contracts. |
| 3 | DepTransforms creates `EpochOp`/`EdtOp` from re-detected wavefront/Jacobi patterns (Invariants 1 & 5) | `lib/carts/dialect/arts/Transforms/dep/DepTransforms.cpp`, `core/Transforms/dep/Seidel2DWavefrontPattern.cpp` | Enhance `PatternAnalysis` and SDE dependency planning; refactor `DepTransforms` to consume lowered SDE contracts. |
| 4 | Doc disagreement: `op-classification.md` says `arts.lowering_contract` and `arts.omp_dep` should migrate to SDE; `pass-placement.md` says the live pipeline keeps planning inside `sde-planning`. | `docs/architecture/op-classification.md` vs `docs/architecture/pass-placement.md` | `pass-placement.md` is current; the migration is aspirational and tracked under violation #1. |

## Open questions (Phase 0 / task #1)

The user must decide these before further restructuring. Answers go to `docs/architecture/charter-decisions.md` (created on first answer).

1. **DestinationStyleOpInterface on SDE ops.** Make CU/SU ops implement `ins`/`outs` for tensor composition, or keep transient `linalg.generic` carriers? Recommendation: keep transient until benchmarks are green, then upgrade. (Effort if upgrading: 8–10h.)

2. **Scope of `PatternAnalysis`.** Does it own ALL semantic pattern approval (incl. wavefront/Jacobi), or split later execution planning into a separate SDE wavefront pass? Recommendation: keep pattern approval centralized; otherwise core keeps re-deriving and Invariant 5 stays broken.

3. **Plug `ARTSCostModel` into all heuristics.** ~20 hardcoded thresholds remain in `PartitioningHeuristics`, `DistributionHeuristics`, `DbHeuristics`. Effort 12–16h. Recommendation: defer until phase 8 is green; cost model only matters once structural plumbing is correct.

4. **Backend-neutral SDE narrative.** Docs aspire to multi-backend (Legion / StarPU / GPU) but namespace, schedule kinds, and conversion target are all ARTS-tied. Keep aspiration or reframe as ARTS-optimized? Recommendation: reframe; pretending otherwise misleads contributors.

5. **LoopReordering migration to SDE.** Blocked on (1). Prioritize the DSI + `SdeLoopInterchange` stack, or defer? Recommendation: defer.

## Quick lookup: "where does X belong?"

| If X is… | It belongs in… |
|---|---|
| A loop classification (stencil, matmul, reduction, etc.) | SDE (`PatternAnalysis`) |
| A reduction strategy choice (atomic / tree / accumulate) | SDE (`ReductionStrategy`) |
| A scope choice (local vs distributed) | Core, using abstract-machine analysis |
| A schedule choice (static / dynamic / guided) | SDE (`ScheduleRefinement`) |
| Tile / chunk / halo geometry **as a contract** | SDE (decision); core (realization) |
| DB allocation / acquire / release / partitioning | core |
| EDT structural rewrite, fusion, distribution | core |
| Epoch creation | core |
| CPS legality, candidate grouping, and dataflow/token planning | SDE |
| CPS continuation materialization | core/RT, consuming SDE plans |
| `arts_rt.edt_create` argument lowering | RT |
| LLVM-near final cleanup (DataPtrHoisting, GuidRangCallOpt) | RT |
| Polygeist→MLIR frontend bridge | core/Conversion (NOT core/Transforms) |
| `linalg.*`, `tensor.*` ops | SDE only, transient |
