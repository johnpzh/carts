# CARTS Distribution Framework

## Purpose

This document is the technical reference for CARTS H2 distribution:
- runtime capability constraints from ARTS
- algorithm compatibility and tradeoffs
- pattern classification ownership (analysis API)
- selected lowering architecture and pipeline integration

H2 decides *work distribution* and task loop structure.
H1 partitioning still decides DB layout/rewrite details.

Related guide:
- `docs/heuristics/partitioning.md`
- `docs/compiler/pipeline.md`

Transitional note:

- This document still describes the current ARTS-side distribution machinery.
- Architecturally, semantic distribution family, wavefront classification, and
  cost-model-driven wavefront tile policy belong to SDE.
- Any ARTS-side classification or attr stamping described below should be read
  as current implementation debt, fallback behavior, or contract
  materialization after the CODIR-to-ARTS boundary.

## 0. Motivation: Bridging AMT, OpenMP, and MPI

CARTS bridges three major parallel programming models, combining the strengths
of each while avoiding their individual limitations:

| Model | Data Distribution | Computation Distribution | Programmer Burden |
|-------|------------------|-------------------------|-------------------|
| **OpenMP** | Shared memory (single node) | Implicit (pragmas) | Low |
| **MPI** | Explicit per-node allocation | Explicit message passing | High |
| **AMT (ARTS)** | Was: single-node allocation | Task-based + work-stealing | Medium |
| **CARTS** | Compiler-managed distributed | Compiler-managed task distribution | Low (OpenMP-like source) |

**The problem.** OpenMP-style source code expresses parallelism through simple
loop annotations, but the memory model is restricted to a single shared address
space. MPI achieves multi-node memory scaling through explicit per-node
allocation and message passing, but at the cost of significant programmer effort.
ARTS (the underlying AMT runtime) provides dynamic task scheduling, work-stealing,
and task migration across nodes, but its programming model still requires manual
data placement decisions.

**What CARTS achieves.** The compiler takes OpenMP-annotated C source and
automatically:

1. **Distributes memory like MPI** — each node allocates only its portion of
   data (via `DbDistributedOwnershipPass` and round-robin route selection in
   `ConvertArtsRtToLLVM`). Memory capacity scales with node count.
2. **Distributes computation like AMT** — tasks are routed to nodes based on
   data ownership, with work-stealing for dynamic load balancing. No manual
   message passing required.
3. **Preserves OpenMP-like source** — the programmer writes standard parallel
   loops. Distribution strategy, data placement, and communication are all
   compiler-managed decisions.

This positions CARTS as a practical bridge: the ease of OpenMP, the scalability
of MPI, and the runtime flexibility of AMT — all derived from the same source
annotation level.

## 1. ARTS Runtime Capability Summary

### 1.1 Memory and ownership model

| Capability | Status | Implication |
|---|---|---|
| DB owner node fixed by GUID | Yes | owner-computes is natural for writes |
| Single writer per generation | Yes | no concurrent multi-writer update of one DB |
| Read sharing | Yes | multiple readers can acquire concurrently |
| Partial read dependency (`artsRecordDepAt`) | Yes | stencil/halo exchanges are efficient |
| Partial point-to-point put (`artsPutInDb`) | Yes | targeted transfer works |
| Partial write-back of modified DB | No (disabled) | write release remains full DB cost |

### 1.2 EDT execution model

| Capability | Status | Implication |
|---|---|---|
| Route-based EDT placement | Yes | static first placement is controllable |
| EDT migration | Yes | runtime can relocate work |
| Work stealing (intra + inter) | Yes | dynamic balance exists but static mapping still matters |
| Active messages (`artsActiveMessageWithDb`) | Yes | compute-to-data patterns are feasible |
| Dynamic route table update | Yes | signals follow migrated EDTs |

### 1.3 Practical constraints for distribution

- Keep write ownership simple (prefer owner-computes).
- Avoid expensive remote write-back when possible.
- Use read-only fan-out for shared inputs when possible.
- Expect no native high-level collectives; compose from EDT/event primitives.

## 2. Algorithm Compatibility (Research)

### 2.1 Compatibility matrix

| Algorithm | ARTS fit | Current status in CARTS |
|---|---|---|
| Block | Strong | Implemented baseline |
| Two-level block (internode) | Strong | Implemented default internode path |
| Block-cyclic | Good | Implemented task-loop lowering path |
| 2D block tiling (matmul-oriented) | Good when input/intermediate reuse is also planned | Strategy exists, but direct-memory 2D output ownership is rejected for current one-node GEMM-family paths unless SDE also proves matching panel/intermediate reuse |
| Cannon-style shifts | Feasible but complex | Not implemented (future) |
| SUMMA-style broadcast panels | Feasible via events/active messages | Not implemented (future) |
| 2.5D replication | Possible but high complexity | Runtime can create/read cached duplicates through DB frontiers; compiler-directed eager replication is not implemented |
| Stencil halo | Strong | SDE stamps halo/window facts; current raw-memref fallback materializes block slices without a dedicated halo rewriter |

### 2.2 Why CARTS currently prefers 2D tiling over Cannon/SUMMA

For current compiler/runtime integration:
- 2D tiling keeps ownership decisions explicit and local to task/DB hints.
- It avoids multi-phase shift orchestration required by Cannon.
- It avoids introducing new broadcast protocol machinery needed by SUMMA.

Cannon and SUMMA remain viable future paths once collective-like orchestration is first-class in lowering/runtime APIs.

## 3. Strategy Selection Policy (H2)

Current selection policy is implemented in SDE by
`DistributionPlanning` (`lib/carts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`).
ARTS consumes and materializes the selected distribution intent; it should not
recover source-level distribution policy after the CODIR-to-ARTS boundary.

Selection order matters:

1. Pattern override:
   - `matmul` + internode (`TwoLevel` strategy) -> `tiling_2d`
   - `matmul` + intranode (`Flat` strategy) -> `block`
2. Otherwise by strategy:
   - internode (`TwoLevel`) -> `two_level`
   - block-cyclic strategy -> `block_cyclic`
   - tiling strategy -> `tiling_2d`
   - flat strategy -> pattern fallback below
3. Pattern fallback (flat/default):
   - `triangular` -> `block_cyclic`
   - `stencil`/`uniform`/`unknown` -> `block`

Operational note:
- `tiling_2d` for internode matmul is selected by policy, but it is not always
  the fastest choice for every dataset size and machine topology.
- For small/medium problems where communication dominates, `two_level` may
  still outperform `tiling_2d`; treat this as a performance-tuning axis, not a
  correctness requirement.

## 4. Current ARTS-Side Analysis Inputs (Transitional)

Current ARTS-side pattern inputs are centralized in analysis APIs, not in
lowering passes. After the SDE boundary, these analyses should consume or
validate SDE contracts rather than redefine semantic pattern family.

### 4.1 SDE-backed source of truth

- SDE `PatternAnalysis` classifies work families, access windows, reductions,
  and distribution intent while source semantics are still visible.
- `DistributionPlanning` consumes those SDE facts and stamps the contract that
  Core materialization uses.
- Core DB analysis validates concrete DB/EDT/epoch shape and should not recover
  semantic loop families from implementation loops.

### 4.2 EDT-facing view

- `EdtAnalysis` consumes DB-derived loop summaries for EDT-level reporting.
- Passes do not need to rescan memops directly to classify loop patterns.

### 4.3 Access-pattern unification

- Shared utility: `AccessPatternAnalysis` (`include/carts/dialect/arts/Analysis/AccessPatternAnalysis.h`, `lib/carts/dialect/arts/Analysis/AccessPatternAnalysis.cpp`)
- DB graph nodes and DB analysis both use the same bounds logic.

## 5. Pipeline Architecture

Distribution is planned inside `sde-planning`. `sde-to-codir` isolates the
planned codelets, and `codir-to-arts` realizes the MU/CU/SU plan as ARTS DB/EDT
objects. The remaining `create-dbs` stage is a compatibility bridge for raw
memref work that has not yet become canonical MU token/codelet form.

- `sde-planning`: SDE pattern/distribution/reduction planning.
- `sde-to-codir`: SDE codelet plans become explicit CODIR deps, params, and
  token-local views.
- `codir-to-arts`: CODIR deps and codelets become ARTS DB/acquire/EDT objects,
  followed by residual non-codelet SDE compatibility lowering and verification.
- `create-dbs`: transitional raw-memref bridge. It consumes SDE-authored layout
  attrs and dependency slices, then creates DB objects and localizes raw memref
  uses. It must not choose distribution policy.
- `db-opt`: tightens DB access modes from real uses.
- `post-db-refinement`: validates and refines DB/EDT contracts.
- `late-concurrency-cleanup`: strip-mining, hoisting, and late cleanup

Key files:
- `tools/compile/Compile.cpp`
- `lib/carts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`
- `lib/carts/dialect/codir/Conversion/CodirToArts/CodirToArts.cpp`
- `lib/carts/dialect/arts/Transforms/db/DbTransformsPass.cpp`

Useful stop points:

```bash
dekk carts compile input.mlir --pipeline=sde-planning
dekk carts compile input.mlir --pipeline=codir-to-arts
dekk carts compile input.mlir --pipeline=post-db-refinement
dekk carts compile input.mlir --pipeline=pre-lowering
```

## 6. Distributed DB Ownership

Goal:
- distribute eligible host-level DB allocations across nodes so memory scales
  with node count.

Current implementation:
- `DbAllocOp` supports a `distributed` marker attribute.
- Pass: `DbDistributedOwnershipPass`
  (`lib/carts/dialect/arts/Transforms/db/DbDistributedOwnership.cpp`).
- Pipeline placement: Core DB refinement after SDE distribution planning
  (gated by `--distributed-db` in `carts-compile`).
- `--distributed-db` relies on SDE-authored work units and Core DB ownership
  marking; late Core loop-carrier producers are not part of the contract.
- Lowering support: `ConvertArtsRtToLLVM` uses round-robin route selection for
  marked multi-DB allocations:
  - route = `linearIndex % artsGetTotalNodes()`
  - unmarked allocations keep the existing route behavior.
- Boundary guard: after CODIR-to-ARTS materialization, an `internode` task must
  not depend on a coarse single-block aggregate user DB. This keeps distributed
  execution from silently scaling one large DB through remote task traffic.
  SDE/CODIR must materialize the block DB layout before ARTS binds routes.
- CODIR-to-ARTS only binds DB-dependent codelets to `internode` placement when
  the codelet carries generic tile owner/shape storage metadata. Distribution
  intent without a materialized storage plan remains local until the SDE/CODIR
  MU/token path can create shaped DBs.

Current eligibility policy is intentionally conservative:
- allocation is host-level (outside `arts.edt`)
- not `DbAllocType::global`
- has multiple DB blocks
- has supported matrix/tensor allocation shape (`elementSizes` rank >= 2)
- handle uses are restricted to DB dependency flow; `db_ref` / `db_gep` and
  memref/cast forwarding are allowed only when their result users remain in
  the allowed flow
- acquire users must be EDT-backed
- allocation is consumed by internode EDTs
- read-only/read-only-after-init distributed DBs may request runtime duplicate
  preference on read dependency slots; this is a runtime cache/duplicate hint,
  not an eager compiler scatter/replicate ABI
- stencil-style read-only internode uses are only eligible when ARTS marks the
  DB as replicated and ARTS-RT can rely on runtime duplicate-frontier behavior

Distributed init split is implemented:
- `Codegen.cpp` generates a `distributed_db_init` callback that runs on ALL
  nodes inside `initPerNode` and reserves a deterministic GUID sequence.
- `Codegen.cpp` also generates `distributed_db_init_worker` in
  `initPerWorker`; only the primary local worker performs DB creation, and only
  for GUIDs whose rank matches the local node (`artsGuidGetRank(guid) == node`).
- `ConvertArtsRtToLLVM.cpp` lowers marked multi-DB allocations with
  `route = linearIndex % artsGetTotalNodes()`. Runtime DB creation remains
  local-only via `artsDbCreateWithGuid(AndArtsId)` semantics.
- ARTS runtime replication is demand-driven: read-only dependency slots can
  carry `ARTS_DEP_FLAG_PREFER_DUPLICATE`, and the runtime tracks duplicate
  ranks through DB frontiers and invalidates/updates them on later writes. The
  compiler does not yet emit an explicit "replicate this initialized aggregate
  to all nodes now" operation.

Future work:
- distributed free policy beyond current `artsShutdown()` behavior
- writer-aware ownership (align DB route with writer EDT node)
- broader eligibility (stencil arrays, global initializers with distributed init EDTs)
- post-allocation distribution for host-initialized data:
  - scatter writable owner blocks from a temporary root DB when initialization is
    naturally single-node
  - add an explicit compiler-facing publish/replicate operation for read-only
    or read-mostly DBs after initialization, using the existing runtime
    duplicate/frontier machinery intentionally instead of relying only on
    demand-driven reads

## 7. IR Contract

SDE distribution planning and direct Core materialization may stamp concrete
Core object attrs:

- `distribution_kind` (`#arts.distribution_kind<...>`)
- `distribution_pattern` (`#arts.distribution_pattern<...>`)
- `distribution_version = 1`

These attributes represent SDE-forwarded contracts and Core machine-binding
facts. They are not a fallback semantic classifier.

## 8. Loop Transform Compatibility (R8)

Question: do loop normalization/reordering transforms harm multi-node distribution?

Current answer: **no for existing 1D outer-loop distribution path**.

Reason:
- The semantic work inside `sde-planning` mostly targets inner serial
  `scf.for` structure.
- `sde-to-codir`, `codir-to-arts`, and later ARTS stages consume the
  SDE-authored materialization contract and preserve concrete DB/EDT/epoch
  distribution facts.

Pass-level summary (current behavior):
- Loop normalization (triangular-to-rectangular forms): compatible
- Loop reordering/interchange: compatible for current outer-loop distribution
- Matmul inner tiling/reduction reshaping: compatible with current H2 selection
- Loop fusion with matched bounds: compatible if resulting top-level loop semantics remain equivalent

Future caveat:
- If distribution expands to full 2D outer-loop ownership transforms, some inner-loop tiling/reordering choices may need explicit coordination metadata.

## 9. Lowering Architecture

The production lowering path materializes SDE plan data through CODIR codelets
and then ARTS DB/EDT/epoch objects. Strategy-specific helpers must consume
explicit plan data rather than a late ARTS semantic loop carrier.

### 9.1 Acquire rewriting helpers

- DB acquire-window planning uses SDE-authored access windows and Core DB
  refinement validation.
- Stencil and block windows should be explicit contract facts before Core
  lowering consumes them.

### 9.2 Task loop lowering helpers

- Dispatch and task-local `scf.for` loops are implementation control flow
  generated by the materializer.
- Reduction materialization consumes SDE reduction accumulators, kinds, and
  strategy; Core does not rediscover missing reduction semantics.

### 9.3 Partitioning integration for 2D owner hints

Core DB refinement consumes tiling-2D owner hints on writable (`inout`) task acquires to force N-D block ownership where valid.

This coupling is what keeps data ownership and routed work aligned for
`matmul` loops when H2 selects `tiling_2d`.

## 10. Heuristics Placement

Distribution selection now lives in SDE:
- `DistributionPlanning` reads SDE pattern/effect facts and the SDE cost model.
- It stamps `sde.su_distribute` or distribution attributes on eligible
  `sde.su_iterate` operations.
- CODIR carries the explicit codelet contract.
- ARTS DB/EDT/epoch passes consume concrete distribution attrs and ownership
  metadata; they should validate and refine the object graph, not reselect
  source-level policy.

Wavefront note:

- Wavefront tile shape, frontier budget, and per-task work thresholds are
  semantic planning choices and should migrate behind the SDE contract
  boundary. ARTS-side heuristics should eventually only consume them.

## 11. Validation Checklist

```bash
dekk carts build

# Missing config must fail fast
dekk carts compile input.mlir --pipeline=codir-to-arts

# Inspect distribution attributes
dekk carts compile gemm.mlir -O3 --arts-config arts.cfg --pipeline=codir-to-arts

# Multi-node counters
# (example harness depends on your local benchmark setup)
```

Expected checks:
- SDE planning attrs are present before CODIR-to-ARTS materialization when
  applicable.
- Core `distribution_kind/pattern/version` attrs are present on concrete
  DB/EDT/epoch objects after `codir-to-arts` when the plan needs them.
- checksums unchanged for benchmark kernels
- node/thread counters show non-zero work on remote nodes for distributed runs

## 12. Compiler-System Design Principles (Survey Notes)

Practical principles adopted from systems like Halide/Legion/Chapel/HPF/GSPMD/Charm++:

- Separate policy from lowering: distribution plan first, IR rewrite second.
- Keep strategy catalog extensible through enums + specialized lowerers.
- Keep correctness independent from mapper choice; mapper changes should be performance policy.
- Prefer analysis-backed decisions over pass-local ad-hoc classification.

In CARTS, this maps to:
- SDE-owned pattern/distribution planning first, with Core materialization
  consuming that plan directly
- specialized task/acquire materialization components for execution
- DB/EDT analysis APIs as single pattern source of truth
