# Dialect Charter

The CARTS compiler is migrating around four dialect layers: SDE, CODIR, ARTS,
and ARTS-RT. Each owns a clear slice of compiler responsibility. Pass
placement is decided against this charter; if a placement decision feels
arbitrary, the charter is wrong (update it) or the pass is wrong (move it).

## Source of truth

The authoritative documents are:

- `docs/compiler/dialect-layering-vision.md` — layer contract
- `docs/compiler/pipeline.md` — live stage order and barriers
- `docs/compiler/dialects/*/{README,analysis,optimizations}.md` — per-dialect
  responsibility
- `.carts/sessions/...` — active planning notes and investigation state

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
- Including any header from retired ARTS-era path aliases
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
- Implementation loops: local `scf.for` control flow inside concrete ARTS objects
- Epochs: creation, CPS scheduling, optimization
- Contract attributes: encode SDE decisions for downstream consumption

**IR level:** ARTS structural: regions, DBs, partitions, contracts. No
tensor/linalg carriers survive the SDE-to-CODIR / CODIR-to-ARTS boundary.

**Forbidden:**
- Re-deriving classifications SDE already stamped (Invariant 5)
- Emitting `arts_rt.*` ops before `pre-lowering`
- Performing semantic analysis from scratch (consume SDE contracts instead)

### ARTS-RT (`arts_rt.*`) — Runtime API mapping

**Owns:** thin 1:1 mapping to ARTS C runtime calls + final low-level passes immediately before LLVM.

- 14 ops mapping directly to runtime: `arts_rt.edt_create`, `record_dep`, `dep_gep`, `state_pack`, `create_epoch`, `wait_on_epoch`, etc.
- 4 passes: `DataPtrHoisting`, `GuidRangeCallOpt`, `RuntimeCallOpt`, `VerifyLowered`
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

4. **SDE does not include any header from retired ARTS-era path aliases.** Mechanical check; grep for old path aliases.

5. **Cost-model-driven decisions belong in the *decision-owner*, not the *realizer*.** SDE stamps a contract (e.g., tile geometry); ARTS consumes it. If both compute the same thing, one is wrong.

## Currently-known violations (2026-04-11 audit)

These do not block correctness but should be cleaned up in Phase 9. Cite when delegating fixes; a regression here is not a new bug, it is technical debt.

| # | Violation | Files | Fix |
|---|---|---|---|
| 1 | Wavefront family detection in ARTS, should be SDE (Invariant 5) | `lib/carts/dialect/sde/Transforms/state/PatternAnalysis.cpp`, `lib/carts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`, `lib/carts/dialect/arts/Analysis/db/DbAnalysis.cpp`, `lib/carts/dialect/arts/Transforms/db/DbTransformsPass.cpp` | Move family + tile geometry into `PatternAnalysis` or a later SDE wavefront-planning pass; refactor ARTS realizers to consume the contract. |
| 2 | ARTS re-detects elementwise/stencil/matmul facts that SDE already proved (Invariant 5) | `lib/carts/dialect/sde/Transforms/state/PatternAnalysis.cpp`, `lib/carts/dialect/arts/Analysis/db/DbAnalysis.cpp`, `lib/carts/dialect/arts/Transforms/db/DbTransformsPass.cpp` | Refactor ARTS refinement to consume SDE/CODIR contracts instead of reclassifying source semantics. |
| 3 | ARTS creates epoch/EDT structure from re-detected wavefront/Jacobi patterns (Invariants 1 & 5) | `lib/carts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`, `lib/carts/dialect/arts/Analysis/db/DbAnalysis.cpp`, `lib/carts/dialect/arts/Transforms/db/DbTransformsPass.cpp` | Enhance `PatternAnalysis` and SDE dependency planning; refactor ARTS materialization/refinement to consume lowered SDE contracts. |
| 4 | Historical docs disagreed about `arts.lowering_contract` ownership. | Archived planning notes under `.carts/sessions/...` | The live contract is in `docs/compiler/dialect-layering-vision.md`: ARTS may carry abstract lowering contracts, but SDE/CODIR must materialize source facts before ARTS. |

## Open questions (Phase 0 / task #1)

The user must decide these before further restructuring. Answers go to a
session note under `.carts/sessions/<topic>/charter-decisions.md`.

1. **DestinationStyleOpInterface on SDE ops.** Make CU/SU ops implement `ins`/`outs` for tensor composition, or keep transient `linalg.generic` carriers? Recommendation: keep transient until benchmarks are green, then upgrade. (Effort if upgrading: 8–10h.)

2. **Scope of `PatternAnalysis`.** Does it own ALL semantic pattern approval (incl. wavefront/Jacobi), or split later execution planning into a separate SDE wavefront pass? Recommendation: keep pattern approval centralized; otherwise ARTS keeps re-deriving and Invariant 5 stays broken.

3. **Plug `ARTSCostModel` into live decision owners.** Do not resurrect retired monolithic partition/distribution heuristic passes; remaining thresholds belong in `DbHeuristics`, SDE planning, or ARTS ownership/refinement according to the contract they decide. Effort 12–16h. Recommendation: defer until phase 8 is green; cost model only matters once structural plumbing is correct.

4. **Backend-neutral SDE narrative.** Docs aspire to multi-backend (Legion / StarPU / GPU) but namespace, schedule kinds, and conversion target are all ARTS-tied. Keep aspiration or reframe as ARTS-optimized? Recommendation: reframe; pretending otherwise misleads contributors.

5. **LoopReordering migration to SDE.** Blocked on (1). Prioritize the DSI + `SdeLoopInterchange` stack, or defer? Recommendation: defer.

## Quick lookup: "where does X belong?"

| If X is… | It belongs in… |
|---|---|
| A loop classification (stencil, matmul, reduction, etc.) | SDE (`PatternAnalysis`) |
| A reduction strategy choice (atomic / tree / accumulate) | SDE (`ReductionStrategy`) |
| A scope choice (local vs distributed) | ARTS, using abstract-machine analysis |
| A schedule choice (static / dynamic / guided) | SDE (`ScheduleRefinement`) |
| Tile / chunk / halo geometry **as a contract** | SDE (decision); ARTS (realization) |
| DB allocation / acquire / release / partitioning | ARTS |
| EDT structural rewrite, fusion, distribution | ARTS |
| Epoch creation | ARTS |
| CPS legality, candidate grouping, and dataflow/token planning | SDE |
| CPS continuation materialization | ARTS/ARTS-RT, consuming SDE plans |
| `arts_rt.edt_create` argument lowering | ARTS-RT |
| LLVM-near final cleanup (DataPtrHoisting, GuidRangeCallOpt) | ARTS-RT |
| Polygeist→MLIR frontend bridge | frontend conversion, before SDE planning |
| `linalg.*`, `tensor.*` ops | SDE only, transient |
