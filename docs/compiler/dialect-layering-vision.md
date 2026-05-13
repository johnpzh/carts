# CARTS Dialect Layering Vision

This document is the target architecture for the CARTS compiler dialect
layers. It is intentionally more specific than the current implementation. If
this document disagrees with the live pipeline, `tools/compile/Compile.cpp` and
`dekk carts pipeline --json` describe the implementation that exists today.

## One-Line Rule

SDE describes source semantics and logical work. Core ARTS describes abstract
ARTS objects: EDTs, DBs, and epochs. RT describes runtime API calls.

Everything else is either a temporary compatibility bridge or should move to
the layer that owns the decision.

## Target Dialect Inventory

### SDE

SDE is the high-level compiler dialect. It owns OpenMP semantics, structured
program analysis, tensor/linalg facts, scheduling-unit intent, dependency
families, and task-body isolation.

SDE may contain:

- `cu_region`: source compute regions such as `parallel`, `single`, and
  `task`.
- `su_iterate`: worksharing/taskloop iteration spaces.
- `su_barrier`: source synchronization and SDE barrier decisions.
- `mu_data`, `mu_dep`, `mu_token`: high-level data and dependency handles.
- `cu_codelet`: isolated task bodies with explicit tokens and captures.
- Future SDE work-plan ops or attrs that describe logical workers, access
  windows, reductions, orchestration groups, and fallback reasons.

SDE must not contain ARTS-machine concepts:

- no node count;
- no workers-per-node;
- no route;
- no current node or current worker;
- no ARTS runtime API query;
- no runtime depv or DB pointer layout;
- no decision that depends on ARTS local-vs-internode placement.

SDE can use a machine-neutral logical capacity. That should be modeled as an
SDE-level value or plan field such as `logical_worker_count` or
`requested_workers`, not as `arts.runtime_query<total_workers>`. SDE can say
"this work unit is planned for N logical lanes"; Core decides whether those
lanes become local workers, nodes, or a two-level ARTS mapping.

### Core ARTS

Core ARTS is the abstract ARTS-machine layer. Its dialect should mirror the
runtime objects without exposing runtime ABI calls.

Core may contain:

- `arts.edt`: a concrete EDT object, not a source-level parallel region.
- `arts.db_*`: DB allocation, acquire, release, ref, mode, layout, and access
  windows.
- `arts.epoch_*`: abstract epoch creation, grouping, waits, continuation, and
  CPS structure.
- typed contract metadata on EDTs, DBs, and epochs while those contracts are
  still being materialized or validated.
- `arts.runtime_query` only as a Core/RT bridge for runtime topology values
  selected by Core, never as an SDE semantic operation.

Core must not contain:

- `arts.for`;
- a semantic `arts.edt<parallel>` wrapper;
- OpenMP worksharing semantics;
- stencil/matmul/reduction classification from loop-body rediscovery;
- loop fusion policy;
- SDE-style distribution planning;
- hardcoded string contracts inside passes.

Core may bind logical worker lanes to the ARTS abstract machine: total local
workers, total nodes, workers per node, routes, DB placement, and EDT placement.
That binding happens after SDE has produced a logical work plan.

### RT

RT is the runtime ABI bridge.

RT may contain:

- EDT create calls;
- dependency record calls;
- state and parameter packing;
- depv addressing;
- DB pointer/GUID GEPs;
- epoch runtime calls;
- low-level cleanup needed before LLVM lowering.

RT must not choose task grain, stencil layout, loop distribution, DB layout, or
epoch topology from semantic facts. Those decisions should already be fixed by
SDE and Core.

## Current Mismatch

The current compiler still uses Core `arts.for` as a compatibility bridge:

1. SDE `su_iterate` is converted to Core `arts.for`.
2. Core passes inspect and annotate `arts.for`.
3. `ForLowering` lowers `arts.for` into task EDTs, DB acquires, and epochs.
4. `VerifyForLowered` checks that no `arts.for` reaches pre-lowering.

That works as a transition mechanism, but it violates the target layering.
`arts.for` duplicates SDE's `su_iterate` and keeps semantic loop decisions in
Core.

## Passes That Must Move, Shrink, Or Disappear

The following list is the migration target, not an immediate delete list.

| Current pass or surface | Current role | Target action |
| --- | --- | --- |
| `ForLowering` | Lowers Core `arts.for` by cloning loop bodies, mapping IVs, computing worker chunks, rewriting DB acquires, detecting owner mismatches, handling halos, reductions, routes, task EDTs, and epochs. | Remove as a Core pass. Split its work: SDE owns work plan, codelet body, captures, logical worker shape, reductions, and access windows. Core materializes EDT/DB/Epoch objects from that plan without a Core loop carrier. |
| `ForOpt` | Coarsens `arts.for` and writes partition hints from worker counts and loop analysis. | Move to SDE scheduling/chunk/work-granularity planning. SDE chooses logical chunking; Core maps it to ARTS workers/nodes. |
| `EdtDistribution` | Classifies `arts.for` access patterns and stamps Core distribution attrs consumed by `ForLowering`. | Semantic classification moves to SDE `DistributionPlanning` and structured summaries. Core keeps only machine binding and validation of the SDE plan. |
| `LoopFusion` | Fuses adjacent `arts.for` ops in parallel EDTs after Core conversion. | Move to SDE scheduling-unit fusion. Existing `ElementwiseFusion` is the seed; general fusion should use SDE effects and structured facts. |
| `EdtOrchestrationOpt` | Recognizes repeated timestep/wave groups on Core EDTs/`arts.for` and stamps orchestration contracts. | Recognition moves to SDE, where repeated work families and barriers are visible. Core only emits the requested epoch/CPS structure. |
| `ParallelEdtLowering` | Lowers semantic `arts.edt<parallel>` wrappers to worker loops/task graphs and replaces `parallel_worker_id`. | Remove after Core no longer has semantic parallel EDTs. Core should receive concrete EDT graph materialization requests, not a parallel-region wrapper. |
| `DistributedHostLoopOutlining` | Late Core pass that creates `arts.for` from host `scf.for` for distributed producer loops. | Move before/into SDE or replace with a direct SDE work-unit producer. It must not create new Core `arts.for`. |
| `VerifyForLowered` | Verifies temporary `arts.for` was removed. | Replace with `VerifyCoreObjectsOnly`: no `arts.for`, no semantic parallel EDT, no SDE scheduler op after the SDE/Core boundary. |
| `DbPatternMatchers` and loop-facing parts of `DbAnalysis` | Recover matmul, triangular, stencil, mapped dims, cross-element self-read, and access summaries from Core `arts.for` bodies. | Move semantic matching to SDE structured analysis. Core DB analysis should operate on DB graph facts and SDE-authored contracts. |
| `LoopAnalysis`, `LoopNode`, `LoopUtils`, `EdtUtils`, `OperationAttributes` `arts::ForOp` support | Utility support for Core loop carrier and top-level `arts.for` discovery. | Remove `arts::ForOp` support once the new plan is in place. Keep ordinary `scf.for` utilities for local task-body loops and low-level cleanup. |

Passes that should remain, but with narrower responsibilities:

- `CreateDbs`: stays Core, but consumes SDE-authored DB layout/access plans
  instead of finding layout intent on `arts.for`.
- `Concurrency`: should become Core machine binding. It maps logical workers to
  ARTS local/internode execution and routes. It should not classify loop
  families or decide semantic work shape.
- `DbModeTightening`, `DbTransforms`, `EdtStructuralOpt`, `EdtTransforms`,
  `CreateEpochs`, `EpochOpt`, `DbLowering`, `EdtLowering`, and
  `EpochLowering`: stay if they operate on concrete DB/EDT/Epoch structure.
  Any semantic loop-policy code inside them should move to SDE.
- RT conversion and runtime-call optimization passes stay RT.

## Replacement Architecture

### SDE Work Plan

SDE needs one coherent work-plan contract on `su_iterate` or a first-class SDE
plan op. The plan should include:

- work family: elementwise, stencil, matmul, reduction, wavefront, Jacobi, or
  fallback;
- logical worker count or requested logical lanes;
- iteration domain: rank, bounds, steps, owner dims, spatial dims, and local
  task-loop shape;
- schedule: static/dynamic/guided/runtime source intent and selected logical
  chunking;
- access plan: per-root read/write mode, offsets, sizes, min/max halo offsets,
  write footprint, owner dims, disjointness proof, and in-place/self-read
  status;
- physical data plan: block shape, halo shape, DB layout request, and root data
  value that owns the plan;
- reduction plan: accumulator, reduction kind, identity, combiner, partial
  storage request, and final result exposure;
- orchestration plan: barrier status, timestep/wave group, repetition
  structure, async strategy, and CPS/continuation preference;
- codelet boundary: tokens, scalar captures, local values, yielded results, and
  fallback reason if no codelet can be formed.

The capture rule must be explicit:

- scalar firstprivate-style values can become EDT parameters;
- dynamic arrays, memrefs, tensors, and mutable shared state must become
  dependencies;
- values that can be constructed locally inside a codelet should be constructed
  locally rather than captured.

### Logical Worker Operations

SDE should have a high-level way to refer to the number of logical lanes
available to a work unit. This is not an ARTS runtime query. It is a symbolic
or constant planning value.

Possible SDE design:

- `sde.logical_worker_count` returns an index value for planning.
- `su_iterate` accepts an optional logical worker count operand or attr.
- `su_distribute` or a replacement work-plan op records how the logical lanes
  map to iteration dimensions.

Core then lowers that logical worker model to one of:

- a constant worker count from the active ARTS config;
- `arts.runtime_query<total_workers>` for local execution;
- `arts.runtime_query<total_nodes>` plus workers-per-node for internode
  execution;
- a two-level node/local-worker mapping;
- route and placement attrs on concrete EDTs.

This keeps SDE portable and keeps ARTS machine binding in Core.

### Core Materialization

`ConvertSdeToArts` should stop translating `su_iterate` into `arts.for`.
Instead it should either:

- directly materialize Core EDT/DB/Epoch structure from SDE work plans; or
- translate SDE work plans into a temporary Core materialization contract that
  is immediately consumed before normal Core optimization begins.

The end state after the SDE/Core boundary should be:

- concrete EDTs;
- concrete DB alloc/acquire/release/ref operations;
- concrete epoch structure or explicit requests for Core epoch creation;
- typed contracts on those objects while validation/materialization continues;
- no Core loop carrier for semantic parallel work.

## Migration Phases

### Phase 0: Inventory And Verifiers

Goal: make the current mismatch measurable.

Tasks:

- Generate stage dumps for representative maintained benchmarks.
- List every `arts.for` producer and consumer.
- List every Core pass that reads `ForOp` attrs, IVs, bodies, or parent EDTs.
- Add a temporary verifier mode that reports why `arts.for` still exists after
  SDE conversion.
- Define the future `VerifyCoreObjectsOnly` checks.

Exit criteria:

- There is a tracked table of `arts.for` dependencies.
- Every dependency is labeled as "move to SDE", "keep in Core but change
  input", or "delete".

### Phase 1: Define The SDE Work Plan

Goal: make SDE's output complete enough that Core does not inspect loop bodies
for semantics.

Tasks:

- Define the work-plan schema.
- Define logical worker count semantics.
- Add per-root access-window and DB-layout contracts.
- Add reduction-plan and orchestration-plan fields.
- Add fallback reasons for unsupported work units.
- Decide whether `su_distribute` remains a wrapper or becomes direct plan data
  on `su_iterate`.

Exit criteria:

- SDE tests prove plans for elementwise, stencil, matmul, reduction, and
  fallback cases.
- The plan contains enough information to build DB acquires and task bodies
  without Core body classification.

### Phase 2: Move Semantic Core Decisions To SDE

Goal: remove semantic policy from Core while keeping current lowering alive.

Tasks:

- Move `ForOpt` policy into SDE schedule/chunk planning.
- Move `EdtDistribution` classification into SDE `DistributionPlanning`.
- Expand `ElementwiseFusion` into the SDE replacement for Core `LoopFusion`.
- Move repeated timestep/wave recognition from `EdtOrchestrationOpt` to SDE.
- Move matmul/triangular/stencil pattern matchers from Core DB analysis to SDE
  structured analysis.
- Reorder SDE passes if needed so `IterationSpaceDecomposition` runs before
  final distribution planning for stencil interiors.

Exit criteria:

- Core can consume SDE-authored contracts for the same cases it previously
  rediscovered from `arts.for`.
- Core semantic matchers are unused or only remain as assertions during
  migration.

### Phase 3: Make Codelets The Task Boundary

Goal: replace loop-body cloning from `ForLowering` with SDE-authored isolated
task bodies.

Tasks:

- Extend `ConvertToCodelet` beyond `cu_region <single>` to parallel
  `su_iterate` bodies.
- Make token/capture planning explicit and test scalar-vs-dynamic-array rules.
- Represent local task-loop indices and logical worker lanes as codelet
  parameters or generated local values.
- Ensure linalg/tensor lowering does not erase information needed by the plan.

Exit criteria:

- A planned `su_iterate` can produce isolated codelets and tokens without
  Core cloning the original loop body.

### Phase 4: Replace `ForLowering`

Goal: eliminate the Core `arts.for` lowering path.

Tasks:

- Introduce a new materializer that consumes SDE work plans and emits Core
  EDT/DB/Epoch structure.
- Move DB acquire-window generation to the materializer using SDE access plans.
- Move reduction materialization to consume SDE reduction plans.
- Move orchestration materialization to consume SDE timestep/wave plans.
- Bind logical workers to ARTS local/internode topology in Core.
- Keep the current `ForLowering` path behind a compatibility flag until parity
  is demonstrated.

Exit criteria:

- Selected benchmarks compile through the direct SDE-plan path with no
  `arts.for`.
- The compatibility path and direct path produce equivalent DB/EDT/Epoch shape
  for covered cases.

### Phase 5: Remove Core Scheduler Scaffolding

Goal: make Core contain only EDTs, DBs, and epochs.

Tasks:

- Stop creating `arts.for` in `ConvertSdeToArts`.
- Stop creating `arts.for` in distributed host-loop outlining.
- Remove `ForLowering`.
- Remove `ForOpt`.
- Remove Core `LoopFusion`.
- Remove semantic parts of `EdtDistribution`.
- Remove semantic parts of `EdtOrchestrationOpt`.
- Remove `ParallelEdtLowering` once semantic parallel EDTs are gone.
- Replace `VerifyForLowered` with `VerifyCoreObjectsOnly`.
- Delete `ForOp` from Core IR after all tests and benchmarks use the direct
  path.

Exit criteria:

- No Core pass depends on `arts::ForOp`.
- No Core dialect op represents source-level parallel work.
- Core tests assert the post-SDE IR contains only EDT/DB/Epoch objects plus
  typed contracts.

### Phase 6: Benchmark And Cleanup

Goal: preserve correctness and performance while deleting compatibility code.

Tasks:

- Run maintained CARTS benchmarks at large size, 64 threads, and one node.
- Compare against OpenMP correctness and performance.
- Add distributed smoke tests for any path that maps logical workers to nodes.
- Remove compatibility tests that only exercise deleted `arts.for` behavior.
- Update skills and docs after the pipeline manifest changes.

Exit criteria:

- Benchmarks are correctness-clean and performance-credible.
- The pipeline no longer contains `ForOpt`, `EdtDistribution` semantic
  planning, `ForLowering`, `ParallelEdtLowering`, or `VerifyForLowered`.

## Investigation Checklist Before Implementation

Before changing code, answer these questions with stage dumps and focused lit
tests:

- Which maintained benchmarks still need `arts.for` for correctness today?
- Which facts does `ForLowering` infer that are not present on SDE
  `su_iterate`?
- Which facts does `EdtDistribution` infer that SDE does not already stamp?
- Which reductions need combiner/identity data that SDE does not preserve?
- Which stencil cases need rank-N owner dims, halo shapes, and boundary
  decomposition before distribution planning?
- Which paths create late `arts.for` after SDE, especially distributed host
  producer loops?
- Which Core analyses use `ForOp` only for mechanics and which use it for
  semantic proof?
- Where does SDE currently consume ARTS machine facts through the cost model or
  scope selection?
- What is the neutral replacement for each such machine fact: logical worker
  count, locality preference, communication cost, or fallback?
- What should Core do when the logical worker count is symbolic and no config
  file is present?
- What verifier proves dynamic arrays are dependencies and only scalar captures
  become params?
- What verifier proves no hardcoded string contracts are introduced?

## Stencil-Specific Rule

Stencil handling should not live in `ForLowering`.

SDE should identify stencil family, owner dims, spatial dims, halo, write
footprint, boundary/interior decomposition, repeated timestep structure, and
logical worker tiling while structured IR is still visible.

Core should only materialize that plan into DB layout, dependency windows,
EDTs, and epochs. If Core needs stencil-specific conditionals to recover the
shape, the SDE plan is incomplete.

RT should only lower the resulting DB/EDT/Epoch graph to runtime calls.

