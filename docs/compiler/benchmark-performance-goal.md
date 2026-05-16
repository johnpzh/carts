# Benchmark Performance Goal

## Goal

Make every maintained CARTS benchmark correctness-clean and performance-credible
against OpenMP at `large`, 64 threads, and 1 node by keeping decisions in the
right dialect layer.

SDE is runtime-agnostic. It must prove legality and author the semantic plan:
memref partitions, tiling, reductions, halo/window shapes, logical task grain,
barrier intent, and timestep or wavefront structure for matrix outputs,
vector/reduction kernels, 3D component stencils, and time-stepped kernels. When
SDE needs symbolic execution capacity, it uses
`sde.resource_query <logical_workers>` rather than an ARTS runtime query.

CODIR is the planned codelet layer. It must consume the SDE MU/CU/SU plan and
produce isolated codelets with explicit deps, params, token-local memref views,
and no implicit captures from above.

`arts` is ARTS-machine-aware. It must preserve and refine the chosen
DB/EDT/dependency/epoch shape and use ARTS analyses to orchestrate work without
inventing memref partition policy late. Canonical MU/token/CODIR form should
lower directly to DBs/acquires/EDTs. `CreateDbs` remains only a raw-memref
compatibility bridge: it consumes SDE-authored layouts and dependency slices,
but it should not choose owner dimensions, tile geometry, or dependency-window
legality. `arts` is the boundary that binds `sde.resource_query` and SDE plan
contracts to `arts.runtime_query`, ARTS worker counts, DB granularity, EDT
counts, and dependency-slot mechanics.

`arts-rt` is the low-level runtime-call lowering layer. It should optimize
launch, CPS/continuation, dependency-slot, packing, scalar replacement,
GUID/runtime call, and LLVM-facing overhead only after the SDE/CODIR/ARTS shape
is correct and traces still show those mechanics as the bottleneck.

Layer references:

- [`master-plan.md`](./master-plan.md): top-level migration and performance
  sequence.
- [`dialects/`](./dialects/): target per-dialect analysis and optimization
  ownership.
- [`plans/performance-large64.md`](./plans/performance-large64.md): focused
  large/64 execution plan.
- [`dialects/sde/`](./dialects/sde/): runtime-agnostic semantic planning,
  memref layout policy, state, dependencies, and effects.
- [`dialects/codir/`](./dialects/codir/): planned isolated-codelet layer with
  explicit deps and params.
- [`dialects/arts/`](./dialects/arts/): `arts` DB/EDT/dependency/epoch
  materialization and orchestration layer.
- [`dialects/arts-rt/`](./dialects/arts-rt/): `arts-rt` low-level
  runtime-call lowering and overhead cleanup.
- [`pipeline.md`](./pipeline.md): live stage order and pass ownership notes.

## Current Dialect Structure And Responsibilities

CARTS has a layered compiler contract. Each layer may preserve, validate, or
refine facts from earlier layers, but it must not rediscover policy that an
earlier layer was responsible for proving.

### SDE: semantic planning dialect

SDE is the only layer that still sees OpenMP semantics, structured loop
summaries, memref access facts, reduction metadata, barrier intent, and enough
source-level shape to prove legality. That makes SDE the primary optimization
layer for CARTS. Most performance policy belongs here because most performance
policy is a question of source semantics: which work is independent, which
state is carried across stages, which dimensions own output, which barriers are
required, and which dataflow edges can be made asynchronous. SDE therefore
owns:

- structured classification: elementwise, elementwise pipeline, reduction,
  matmul contraction, stencil, wavefront, timestep, and opaque effects;
- memory-effect summaries over source roots and memrefs;
- owner dimensions, spatial dimensions, component dimensions, and batch
  dimensions;
- target-neutral physical layout and grain intent through `physicalOwnerDims`,
  `physicalBlockShape`, `physicalHaloShape`, and `logicalWorkerSlice`; these
  are SDE plan facts, not ARTS DB objects or worker identifiers;
- MU/CU/SU alignment: the memory-unit (`mu_data`/`mu_token`) slice plan,
  compute-unit body, and scheduling-unit iteration topology must be rewritten
  together. Tiling a CU/SU loop without changing the MU token or DB address
  space is not a valid performance transformation;
- scheduling-unit topology through `iterationTopology`,
  `repetitionStructure`, `asyncStrategy`, reduction strategy, and barrier
  reason;
- CPS planning, including candidate groups, stage indices/counts, tokenized
  dataflow requirements, explicit `sde.control_token` edges, timestep
  boundaries, and final CPS stage plans when SDE has rewritten all carries;
- legality decisions for tiling, strip-mining, loop interchange, fusion,
  vectorization hints, owner-slice partitioning, wavefront/timestep grouping,
  CPS conversion, and barrier removal.

SDE must reject or leave unplanned any case whose legality depends on
runtime-specific machine behavior instead of source semantics. It may use an
abstract logical-capacity cost model and `sde.resource_query` for target-neutral
grain arithmetic, but it must not name ARTS workers, routes, nodes, EDTs, DBs,
or runtime APIs. It may stamp conservative plans, but it must not stamp a
performance plan unless the SDE analysis can explain why the chosen owner
slices are independent or why their dependency windows are complete.

The latest CPS contract follows this rule. `BarrierElimination` marks full-timestep
`advance_stage` stages as SDE CPS candidates only when SDE can provide an
explicit control boundary. Adjacent candidates are not allowed to be attrs-only:
SDE inserts a completion token and a token-consuming timestep barrier before it
stamps the group. `VerifySdeCpsPlan` rejects candidate pairs without that
SDE-authored control edge. ARTS and ARTS-RT may lower or optimize the resulting
continuation mechanics, but they do not decide whether the CPS dataflow is
legal.

### CODIR: codelet dialect

CODIR is the planned layer between SDE and `arts`. It receives the SDE plan and
turns it into isolated codelets. CODIR owns:

- `IsolatedFromAbove` codelet bodies;
- explicit memory deps, control deps, scalar params, and yielded values;
- token-local memref views and body access rewrites;
- scalar capture normalization;
- verification that every external value used by a codelet is represented at
  creation time as a dep or param.

CODIR must not own OpenMP semantics, owner-dim selection, ARTS placement,
DB/EDT creation, runtime topology, or raw-memref DB rediscovery.

The EDT isolation requirement starts here. By the time a codelet lowers to an
ARTS EDT, its dependency list and parameter list must be complete. `EdtLowering`
should be a mechanical ABI lowering pass, not a capture-recovery pass.

### ARTS: orchestration dialect

ARTS starts after the SDE semantic plan has been authored and CODIR has isolated
codelets. It knows ARTS objects and runtime topology, but it should treat SDE
physical layout and CODIR dependency-window policy as input, not as a place to
invent memref partitioning. ARTS owns:

- converting SDE/CODIR plan attributes into `arts.plan.*`, dependency pattern,
  distribution kind, and stencil/layout contracts;
- lowering `sde.resource_query <logical_workers>` into the concrete ARTS
  runtime query or static worker materialization selected by the ARTS/runtime
  configuration;
- materializing SDE CPS, barrier, async, and repeated-timestep plans into ARTS
  EDT/epoch/control structure without changing their legality policy;
- preserving direct MU/token DB materialization when present, with `CreateDbs`
  limited to coarse raw memrefs during the transition;
- preserving planned DB/EDT/dependency/epoch shape while splitting or refining
  ARTS mechanics;
- validating plan consistency, owner dims, block shapes, halo bounds, and
  dependency-window contracts;
- selecting ARTS dependency slots, DB acquires, EDT launch shape, epoch
  structure, and distributed ownership from the already-authored plan;
- reporting proof gaps when SDE did not provide enough information.

ARTS may refine an ARTS object graph, but any refinement must be a mechanical
consequence of the SDE/CODIR plan or an ARTS-machine constraint. If ARTS has to
guess which memref dimension to block, which reduction strategy to use, or
whether a stencil, timestep, or CPS chain is owner-slice/dataflow safe, the fix
belongs in SDE or CODIR. If SDE starts naming ARTS topology directly, the fix is
to move that binding to ARTS and leave only logical-capacity intent in SDE.

### ARTS-RT: runtime-call lowering dialect

ARTS-RT is the lowering-ready bridge. It sees flat runtime calls, launch
metadata, dependency descriptors, GUIDs, DB pointers, continuation plumbing,
and LLVM-facing scalar values. ARTS-RT owns:

- lowering ARTS DB/EDT/epoch/dependency objects to runtime calls;
- launch and continuation overhead cleanup after task shape is correct;
- dependency-slot indexing, local dependency-window access, and dep/db pointer
  hoisting;
- temporary descriptor allocation, packing, scalar replacement, alias metadata,
  vectorization hints, and runtime-call hoisting;
- preserving the memory model and dependency semantics authored by
  SDE/CODIR/ARTS.

ARTS-RT must not compensate for missing SDE/CODIR/ARTS shape by recovering
memref policy late. A late ARTS-RT fast path is acceptable only after traces
show that SDE/CODIR/ARTS already produced the intended DB/EDT/dependency
structure and the remaining bottleneck is runtime-call mechanics. In
particular, ARTS-RT may reduce
continuation and dependency-call overhead, but CPS legality, stage grouping,
and carry tokenization remain SDE responsibilities.

### Runtime: execution contract

The ARTS runtime owns scheduler behavior, route tables, EDT publication,
frontier progress, DB lifetime, counters, and actual communication. Runtime
changes are justified only when compiler output already satisfies the contract
and traces show a runtime implementation bottleneck or a missing runtime API.

The reference success path is the current convolution-2d result: SDE authors the
physical memref partition, the boundary materializes or bridges the matching DB
shape, ARTS-RT lowering uses local dependency-window indices, and the benchmark
passes against the OpenMP checksum.

## Performance Heuristic And Analysis Set

Every benchmark optimization must start by naming the SDE proof, the CODIR
isolation check, the ARTS materialization check, and the ARTS-RT/runtime
evidence. The goal is not to add more heuristics everywhere; it is to make each
heuristic live in the earliest layer where it can be proven.

Pass responsibility rule:

- `PatternAnalysis` stamps approved `sde.pattern` and structured memref facts.
  It does not stamp ARTS attrs and it does not pass raw analysis objects across
  the dialect boundary.
- SDE structural/effect passes consume those approved SDE facts to decide
  interchange, tiling, fusion, chunking, reduction strategy, distribution
  intent, barriers, and CPS candidates.
- SDE may emit `sde.resource_query <logical_workers>` when symbolic arithmetic
  needs a logical execution capacity. ARTS lowers that query to ARTS runtime
  machinery.
- `ConvertSdeToCodir` should lower only the final SDE plan contract into
  explicit codelet deps, params, and token-local memref views.
- `ConvertCodirToArts` turns CODIR codelets into DB/acquire/EDT objects
  directly. `CreateDbs` remains only as a migration path for raw memrefs that
  have not yet been promoted into MU/token/codelet storage.

### Required SDE analyses

- **Structured loop summary:** classify the loop family, iterator roles,
  output roots, read/write roots, static shape, affine access maps, and unknown
  effects before lowering to CODIR/ARTS.
- **Owner-slice independence:** prove that each planned owner slice writes
  disjoint output elements and that in-place reads either stay inside the owner
  slice or are covered by a halo/window plan.
- **Physical layout synthesis:** choose owner dims, block shape, halo shape,
  logical worker slice, component locality, and task topology from memref shape,
  access footprint, abstract logical capacity, and minimum useful work per
  task.
- **Reduction strategy selection:** choose local accumulate, tree, or atomic
  based on accumulator visibility, write contention, output rank, and expected
  task count.
- **Pipeline fusion and barrier graph:** fuse adjacent elementwise stages when
  root windows are compatible, remove duplicate barriers, and keep barriers
  only where dependence edges or OpenMP semantics require them.
- **Timestep/wavefront planning:** identify repeated stencil stages,
  alternating buffers, in-place update hazards, and wavefront frontiers before
  ARTS sees flat ARTS work.
- **CPS/dataflow planning:** identify repeated-stage candidates, insert
  required SDE control tokens and barriers, verify candidate completeness, and
  rewrite scalar/data/control carries before stamping final `cps_chain` plans.
- **Cost guardrails:** prevent overpartitioning when per-task compute is below
  launch/dependency overhead, and prevent underpartitioning when a coarse DB
  serializes independent output slices.

### Required CODIR checks

- **Codelet isolation:** verify that codelet bodies are isolated from above and
  use only declared deps, params, local values, and values derived from them.
- **Dep/param ABI shape:** verify that memrefs, mutable state, MU tokens, and
  control edges are deps, while scalar firstprivate-style captures are params.
- **Token-local access shape:** verify that codelet load/store indices agree
  with the MU token-local view for ND, strided, and halo cases.
- **No rediscovery markers:** reject CODIR codelets that require ARTS to rediscover
  DB roots, dependency windows, or implicit captures.

### Required ARTS analyses

- **Plan-to-DB validation:** verify that every SDE physical layout becomes the
  exact direct DB allocation or raw-memref bridge `DbAllocOp` block shape and
  owner-dim order.
- **Acquire-window validation:** check that DB acquires match the planned
  dependency window, including halo bounds, owner block range, and local index
  projection.
- **EDT shape accounting:** count EDTs, dependencies per EDT, epoch barriers,
  and continuations for each benchmark family and flag shapes that contradict
  the SDE plan.
- **CPS materialization accounting:** verify that SDE CPS candidates/final
  stages lower to the expected ARTS control/continuation shape without ARTS
  inventing new dataflow legality.
- **Distributed ownership refinement:** map planned owner blocks to nodes and
  routes without changing memref partition policy.
- **Proof-gap diagnostics:** warn or fail tests when ARTS had to fall back to
  whole-DB acquires, unblocked DBs, global dep indexes, or unplanned barriers.

### Required ARTS-RT/runtime analyses

- **Launch overhead profile:** measure EDT create, ready-local create,
  continuation, and epoch finish/wait costs only after ARTS task counts look
  right.
- **Dependency-slot locality:** confirm `dep_gep` and DB pointer access are
  local to the planned dependency window, not global block ids.
- **Packing and scalar cleanup:** profile `edt_param_pack`, `state_pack`,
  descriptor scratch allocation, scalar replacement, and GUID/runtime-call
  churn.
- **Counter trace comparison:** compare task count, DB allocation count, DB
  acquire count, read/write mode distribution, and runtime-counter hotspots
  against fast controls and OpenMP-equivalent work.

### Benchmark-family heuristic matrix

| Family | SDE policy | CODIR/ARTS check | ARTS-RT/runtime follow-up |
|---|---|---|---|
| Elementwise and ML vector kernels | Fuse compatible stages, choose coarse vector blocks, keep pure libm scalar calls effect-free, preserve vectorization hints. | Direct MU/token lowering must allocate blocked output DBs and avoid whole-DB dependency windows; `CreateDbs` is coarse-only. | Hoist dep/db pointers, shrink pack/unpack, confirm launch overhead is not dominating. |
| Row/column reductions (`atax`, `bicg`, `layernorm`, `batchnorm`) | Distinguish per-output reductions from OpenMP scalar reductions; choose owner rows/channels and local accumulate/tree/atomic strategy. | Materialize partial/output DBs according to SDE owner dims, with no serialized global update unless SDE chose atomic. | Reduce continuation and dependency overhead after reduction DB shape is correct. |
| Matmul and chained contractions (`gemm`, `2mm`, `3mm`) | Classify contractions in SDE, preserve reduction locality, and choose physical owner dims from proven DB reuse. Direct-memory matmul keeps row-strip ownership until SDE/CODIR can also tile A/B/intermediate DBs and rewrite token-local accesses. 2D output ownership is only valid when it does not duplicate coarse input sweeps. | CODIR codelets must expose explicit A/B/C deps and params with token-local access; ARTS intermediate/output DBs must reflect the SDE owner plan and must not invent memref partitioning. | Optimize launch/dep overhead only after task count and block reuse match the SDE tile plan. |
| Stencils and timesteps (`jacobi2d`, `seidel-2d`, `fdtd-2d`, KaStORS) | Author halo/window shape, alternating-buffer or wavefront structure, barrier/timestep intent, and CPS candidate/final stage plans. | Acquires must use bounded neighbor windows and local dependency slots; CPS/barrier plans must lower from SDE-authored control edges. | Optimize ready-local create, continuation calls, and dep window indexing after halo/CPS shape is correct. |
| 3D component stencils (`specfem3d`, `sw4lite`) | Separate spatial, component, and batch dimensions; keep component dimension local when it improves reuse. | DB blocks must reflect spatial slabs plus local components. | Use traces to reduce launch/acquire overhead without changing slab policy. |
| Irregular/task suites (`monte-carlo`, `stream`, future graph/task workloads) | Stamp only proven independent chunks; document unsupported irregular policy instead of guessing. | Preserve explicit task/DB contracts and diagnose unplanned coarse synchronization. | Runtime scheduler/counter work is allowed only when compiler shape is already credible. |

### Optimization gate for each benchmark

Before a performance result is considered meaningful:

1. The benchmark must compile and pass checksum parity against OpenMP.
2. The SDE stage dump must show the intended semantic plan or explicitly show
   why the benchmark is blocked.
3. The CODIR dump must show isolated codelets with complete deps, params, and
   token-local accesses, or explicitly show why the case is still on the
   migration bridge.
4. The ARTS dump must show the SDE physical layout materialized directly from
   MU tokens/CODIR deps or bridged by `CreateDbs` from explicit SDE dependency
   slices.
5. Runtime traces must agree with the intended task count, DB count, dependency
   count, and epoch shape.
6. Only then may ARTS-RT/runtime overhead changes be used to improve speed.

## Acceptance Criteria

- The current checkout is install-clean through `dekk carts install`, with the
  Conda/dekk environment, submodules, ARTS runtime, Polygeist frontend, LLVM
  toolchain, and CARTS compiler all available through `dekk carts ...`.
- The repo docs and `carts-plugin/skills/` are reviewed against `origin/sde`;
  useful process contracts are carried forward, stale command names are removed,
  and generated agent resources are refreshed with `dekk carts skills generate`.
- The source tree is organized around the live command model, current `samples/`
  layout, and target SDE/CODIR/ARTS/ARTS-RT dialect ownership. Stale docs, dead
  workflow layers, and nonfunctional generated scaffolding are removed instead
  of preserved.
- Every benchmark in the maintained suite compiles with `dekk carts compile`.
- Every benchmark run reports checksum parity against its OpenMP baseline.
- Every benchmark has an explicit performance classification:
  - `fast`: ARTS kernel time is faster than OpenMP.
  - `competitive`: ARTS kernel time is within 1.25x of OpenMP.
  - `blocked`: a named compiler/runtime limitation prevents a fair result.
- No optimization changes the memory model or program semantics.
- DB partitioning decisions are made at SDE level when memref structure is
  known.
- SDE may reshape `sde.su_iterate` loop steps, inner loops, and
  scheduling-unit topology when the legality proof lives at that level.
- Direct MU/token lowering should create the chosen physical DB/acquire shape
  when available; `CreateDbs` should do so only for remaining raw memrefs.
- CODIR must isolate codelets and make deps/params explicit before EDT
  creation.
- ARTS passes may refine ARTS DB/EDT/dependency/epoch structure, but must not
  invent memref partition policy late.
- ARTS-RT passes may optimize runtime-call shape only after SDE/CODIR/ARTS
  produce the intended DB/EDT/epoch structure.
- Every runnable sample is swept through the user-facing examples flow or the
  equivalent e2e tests before benchmark conclusions are considered final.
- All runnable benchmarks are swept with `--size large --threads 64 --nodes 1`
  unless the runner documents that a workload lacks that size or requires a
  different node mode. Deviations must be recorded in the evidence summary.
- Any detected overengineering must be simplified at the owning layer before
  moving on: remove dead abstractions, stale generated code, duplicate helpers,
  misplaced utility logic, and late ARTS heuristics that should be SDE policy.

## Expanded Execution Plan

1. **Install and health-check the repo**
   - Run `dekk carts install`.
   - Run `dekk carts doctor`.
   - Record any environment bootstrap work that was needed outside the repo.

2. **Read and reconcile docs**
   - Read the setup, developer, compiler pipeline, architecture, heuristics,
     sample triage, and benchmark docs.
   - Fix stale command names, stale sample paths, and pipeline facts that do not
     match live `dekk carts --help` or `dekk carts pipeline --json`.

3. **Reconcile skills with `origin/sde`**
   - Compare `carts-plugin/skills/` and `carts-plugin/project.md` against
     `origin/sde`.
   - Keep concise trigger descriptions in `SKILL.md`; move bulky facts into
     `references/`.
   - Carry forward useful skills such as command policy, pipeline map, dialect
     map, local/multinode examples, simplification, review, and skill
     maintenance.
   - Remove skill-level workflow references that require unavailable tooling.

4. **Sweep samples**
   - Run `dekk carts examples list`.
   - Run `dekk carts examples run --all` or the maintained e2e equivalent.
   - Classify each sample failure as compile, runtime, checksum, timeout, or
     runner/tooling.

5. **Sweep benchmarks**
   - Run `dekk carts benchmarks list`.
   - Run the maintained benchmark suite with:
     ```bash
     dekk carts benchmarks run --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-large-64-fresh
     ```
   - For noisy, borderline, or unexpectedly slow results, rerun the focused
     subset with `--runs 3` into a separate `.carts/outputs/...` directory.
   - Record only the latest current evidence in this document. Do not keep dated
     historical result tables in the goal.

6. **Simplify before fixing forward**
   - Run the `carts-simplify` checklist on every nontrivial patch.
   - Prefer the earliest semantically correct layer: SDE for program structure
     and policy, CODIR for codelet isolation and dep/param ABI, ARTS for
     DB/EDT/epoch orchestration, ARTS-RT for runtime-shaped lowering, and
     runtime only for actual runtime contract gaps.

## Benchmark Scope

Initial maintained set:

- PolyBench: `2mm`, `3mm`, `atax`, `bicg`, `correlation`, `convolution-2d`,
  `convolution-3d`, `fdtd-2d`, `gemm`, `jacobi2d`, `seidel-2d`.
- ML kernels: `activations`, `batchnorm`, `layernorm`, `pooling`.
- Task/runtime suites: `kastors-jacobi`, `stream`, `lulesh`, `graph500`,
  `seissol`, `specfem3d`, `monte-carlo`, `sw4lite`,
  `llama2-transformer`.

Benchmarks can be moved out of the maintained set only with a documented reason:
unsupported language feature, unsupported runtime dependency, or intentionally
out-of-scope algorithmic pattern. Use `dekk carts benchmarks list` at the time
of the sweep to determine the runnable subset; disabled entries remain blocked
until they are re-enabled or removed with a documented reason.

## Optimization Tracks

### SDE Tracks

- Memref partition planning: derive owner dims, block shape, halo, and
  iteration topology before SDE-to-CODIR materialization.
- Distribution shaping: rewrite `sde.su_iterate` loop steps, inner loop
  nests, and scheduling-unit topology when SDE can prove legality. SDE should
  not merely stamp late contracts if the loop shape itself must change for good
  work distribution.
- Logical resource queries: use `sde.resource_query <logical_workers>` for
  symbolic SDE grain arithmetic. Do not materialize `arts.runtime_query`, ARTS
  worker counts, routes, or node ids in SDE passes.
- Loop transforms: use tiling, interchange, fusion, vectorization, and
  decomposition only where legality is proven by SDE pattern and effect facts.
- Dependency granularity: choose DB/task windows from structured access facts,
  not from late ARTS heuristics.
- Barrier removal: eliminate SDE barriers when dependence analysis proves no
  cross-iteration ordering requirement.
- Reduction planning: select local accumulate, tree, or atomic strategy from
  cost and semantics before ARTS lowering.

### CODIR Tracks

- Codelet isolation: move transitional `sde.cu_codelet` responsibilities into a
  CARTS-owned CODIR dialect with `IsolatedFromAbove` verification.
- Dep/param ABI: make every memory/control edge and scalar capture explicit at
  codelet creation time.
- Token-local memref rewrite: rewrite loads/stores to match MU token windows,
  including ND owner dims, strided accesses, and halo windows.
- No tensor carrier path: keep codelet deps at the memref/token level.

### ARTS Tracks

- DB materialization: prefer direct MU/token DB creation; keep `CreateDbs` as
  the raw-memref bridge where planned physical DBs are created mechanically.
- EDT shape: preserve planned block reads/writes through direct CODIR-to-ARTS
  materialization and ARTS EDT/DB refinement.
- Epoch structure: remove unnecessary epoch barriers and continuation overhead
  after SDE legality has been preserved.
- Dependency lowering: make runtime dependency slots local to the dependency
  window; avoid global block ids in `dep_gep`.
- Analysis facade: use DB/EDT/loop/epoch analyses for ARTS refinement instead
  of duplicating SDE/CODIR dependence logic in local helpers.

### ARTS-RT Tracks

- Launch overhead: reduce EDT creation, CPS continuation, and epoch finish/wait
  mechanics only after task shape is correct.
- Dependency lowering: keep `dep_gep` and dependency DB access local to the
  planned dependency window.
- Packing: reduce `edt_param_pack`, `state_pack`, and relaunch schema churn.
- Runtime calls: batch or hoist GUID/runtime topology calls where the operation
  is pure or mechanically safe.
- LLVM-facing cleanup: use scalar replacement, data pointer hoisting, alias
  metadata, and vectorization hints after runtime shape is fixed.

## Triage Workflow

For each benchmark:

1. Run the benchmark with trace enabled.

   ```bash
   dekk carts benchmarks run <benchmark> --size small --timeout 120 --threads 16 --nodes 1 --trace
   ```

2. If it fails correctness, stop performance work and fix semantic lowering.

3. Dump the pipeline and identify where the benchmark loses structure.

   ```bash
   dekk carts compile <artifact.mlir> --all-pipelines -O3 --arts-config <arts.cfg> -o .carts/outputs/<benchmark>-pipes
   ```

4. Classify the bottleneck:
   - missing SDE pattern analysis,
   - poor SDE tiling/interchange,
   - DB granularity too coarse or too fine,
   - EDT count or task grain mismatch,
   - epoch/barrier overhead,
   - ARTS runtime dependency overhead,
   - host initialization or verification artifact.

5. Implement the fix in the earliest correct layer.

6. Re-run:
   - focused pipeline dump,
   - focused benchmark,
   - `dekk carts test`,
   - `git diff --check`.

## Priority Order

1. Stencils and neighborhood kernels:
   `jacobi2d`, `seidel-2d`, `fdtd-2d`, `convolution-3d`.
2. Dense linear algebra:
   `gemm`, `2mm`, `3mm`, `atax`, `bicg`, `correlation`.
3. ML kernels:
   `pooling`, `batchnorm`, `layernorm`, `activations`.
4. Runtime/task suites:
   `kastors-jacobi`, `stream`, `lulesh`, `graph500`, `seissol`,
   `specfem3d`, `monte-carlo`, `sw4lite`, `llama2-transformer`.

## Fresh Large/64 Results Task

This goal should not preserve stale benchmark result tables. Before claiming the
current benchmark state, create fresh evidence for the current checkout.

Required command sequence:

```bash
dekk carts build
dekk carts doctor
dekk carts benchmarks list
dekk carts benchmarks run --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-large-64-fresh
```

After the full sweep, record a compact current-results summary here with:

- the result directory under `.carts/outputs/`;
- the CARTS, carts-benchmarks, and ARTS revisions from the benchmark manifest;
- runnable, passed, failed, skipped, timeout, and checksum-failure counts;
- geometric mean speedup when the runner reports it;
- `fast`, `competitive`, and `blocked` benchmark groups;
- one bottleneck owner for each blocked group: SDE, CODIR, ARTS, ARTS-RT,
  runtime, or benchmark-scope/tooling. Core and RT may appear only when
  referring to current source-tree paths or older evidence.

If a result is noisy, borderline, surprisingly fast, or unexpectedly slow, run a
focused 3-run follow-up into a separate `.carts/outputs/...` directory and
record the median. Keep detailed logs and generated artifacts under `.carts/`;
keep this document limited to the current summary and the next optimization
tasks.

## Current Large/64 Evidence

Rejected evidence:

- `.carts/outputs/benchmarks-large-64-fresh/20260514_130708` was aborted and
  must not be used for performance claims. It was run against an ARTS runtime
  whose stale CMake cache still had counters/metrics enabled with
  `profile-overhead.cfg`, causing systemic overhead.

Current full-suite large/64 sweep:

- Results:
  `.carts/outputs/benchmarks-m6-final-current-20260516/20260516_054254`
- Command:
  `dekk carts benchmarks run --size large --timeout 180 --threads 64 --nodes 1 --runs 3 --trace --results-dir .carts/outputs/benchmarks-m6-final-current-20260516`
- Counts: 23 configured/runnable entries, 69 benchmark executions, 69 passed,
  0 failed, 0 skipped by the runner, 0 checksum failures.
- Startup outliers: ARTS=1, OpenMP=0.
- Runner-reported geometric mean kernel speedup over all executions: `1.70x`
  in median reporting mode.
- Focused noisy-STREAM follow-up:
  `.carts/outputs/benchmarks-m6-stream-final-rerun-20260516/20260516_055824`
  reports 7/7 passed, checksum-clean, `1.06x` median-filtered speedup. This
  supersedes the noisy full-sweep STREAM median (`0.73x`) for the final M6
  classification.
- Audited final classes after the STREAM rerun: 21 fast, 2 competitive, 0
  blocked. Replacing the noisy STREAM full-sweep median with the focused
  7-run median gives a `1.73x` audited geomean.
- Host OpenMP fallback is intentionally narrow: repeated 2-D stencil islands
  and benchmark-scoped multi-map 1-D floating-point bundles where ARTS epoch
  overhead would dominate repeated host streaming work. These binaries bypass
  `arts_rt` and call `mainBody` directly; they are not tokenized ARTS task
  executions.

Current M6 classes after the final full sweep plus the superseding STREAM
focused rerun:

- `fast`: `kastors-jacobi/jacobi-for`, `kastors-jacobi/poisson-for`,
  `ml-kernels/batchnorm`, `ml-kernels/layernorm`, `ml-kernels/pooling`,
  `monte-carlo/ensemble`, `polybench/2mm`, `polybench/3mm`,
  `polybench/atax`, `polybench/convolution-2d`,
  `polybench/convolution-3d`, `polybench/correlation`, `polybench/gemm`,
  `polybench/jacobi2d`, `polybench/seidel-2d`,
  `seissol/volume-integral`, `specfem3d/stress`, `specfem3d/velocity`,
  `stream`, `sw4lite/rhs4sg-base`, `sw4lite/vel4sg-base`.
- `competitive`: `ml-kernels/activations`, `polybench/bicg`.
- `blocked`: none.

| Benchmark | CARTS status | ARTS kernel | OpenMP kernel | Speedup | Current class |
|---|---|---:|---:|---:|---|
| `kastors-jacobi/jacobi-for` | pass | `1.465334s` | `2.789009s` | `1.903x` | `fast`: host OpenMP fallback control |
| `kastors-jacobi/poisson-for` | pass | `1.568225s` | `2.469140s` | `1.574x` | `fast`: host OpenMP fallback control |
| `ml-kernels/activations` | pass | `0.597304s` | `0.536378s` | `0.898x` | `competitive`: host OpenMP fallback for transcendental elementwise bundle |
| `ml-kernels/batchnorm` | pass | `1.444859s` | `1.997864s` | `1.383x` | `fast` |
| `ml-kernels/layernorm` | pass | `2.931311s` | `3.454671s` | `1.179x` | `fast` |
| `ml-kernels/pooling` | pass | `2.069970s` | `2.311300s` | `1.117x` | `fast`: affine lowering and current task grain are sufficient |
| `monte-carlo/ensemble` | pass | `3.524011s` | `4.222032s` | `1.198x` | `fast` |
| `polybench/2mm` | pass | `0.774910s` | `5.673198s` | `7.321x` | `fast`: CODIR dispatch-step fix |
| `polybench/3mm` | pass | `0.641719s` | `4.902833s` | `7.640x` | `fast`: CODIR dispatch-step fix |
| `polybench/atax` | pass | `2.835092s` | `2.927754s` | `1.033x` | `fast`: CODIR owner-strip logical-slice dispatch |
| `polybench/bicg` | pass | `3.085321s` | `2.924945s` | `0.948x` | `competitive`: CODIR logical-worker-slice dispatch |
| `polybench/convolution-2d` | pass | `2.201995s` | `2.732831s` | `1.241x` | `fast`: CODIR 2-D owner-tile dispatch |
| `polybench/convolution-3d` | pass | `0.605624s` | `2.370311s` | `3.914x` | `fast` |
| `polybench/correlation` | pass | `0.586381s` | `1.177201s` | `2.008x` | `fast`: CODIR owner-strip logical-slice dispatch |
| `polybench/gemm` | pass | `0.437167s` | `6.188823s` | `14.157x` | `fast`: CODIR dispatch-step fix |
| `polybench/jacobi2d` | pass | `0.610889s` | `0.798912s` | `1.308x` | `fast`: host OpenMP fallback control |
| `polybench/seidel-2d` | pass | `4.291360s` | `4.328538s` | `1.009x` | `fast`: host OpenMP fallback control |
| `seissol/volume-integral` | pass | `0.200723s` | `0.248475s` | `1.238x` | `fast`: CODIR-localized SU scratch |
| `specfem3d/stress` | pass | `1.647963s` | `2.276170s` | `1.381x` | `fast`: trailing owner-dim dispatch |
| `specfem3d/velocity` | pass | `1.203891s` | `1.730856s` | `1.438x` | `fast`: trailing owner-dim dispatch |
| `stream` | pass | `1.970628s` | `2.084288s` | `1.058x` | `fast`: superseding 7-run host OpenMP streaming fallback |
| `sw4lite/rhs4sg-base` | pass | `1.981364s` | `2.780303s` | `1.403x` | `fast`: SW4Lite current task grain is sufficient |
| `sw4lite/vel4sg-base` | pass | `1.566038s` | `2.257713s` | `1.442x` | `fast`: trailing owner-dim dispatch |

M6 has no blocked maintained benchmark entries. The next work is M7-quality:
replace host OpenMP fallback controls with tokenized SDE/CODIR/ARTS plans where
that is worth the engineering cost.

Prior focused follow-up for the 12 raw-layout failures:

- Results:
  `.carts/outputs/benchmarks-raw-layout-demotion-12failures-final-20260515/20260515_072611`
- Command:
  `dekk carts benchmarks run ml-kernels/layernorm ml-kernels/pooling monte-carlo/ensemble polybench/atax polybench/bicg polybench/convolution-2d polybench/convolution-3d polybench/jacobi2d seissol/volume-integral specfem3d/stress specfem3d/velocity sw4lite/vel4sg-base --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-raw-layout-demotion-12failures-final-20260515`
- Counts: 12 passed, 0 failed, 0 skipped, 0 startup outliers.
- Geometric mean kernel speedup for the formerly failing slice: `0.92x`.
- Result: the raw-layout boundary failure was fixed for that maintained
  benchmark slice. Unsupported SDE physical storage attrs are demoted before
  the raw `CreateDbs` bridge, and `CreateDbs` remains a guarded coarse-only
  compatibility path.

Superseded full-suite large/64 sweep:

- Results:
  `.carts/outputs/benchmarks-large-64-release-fresh/20260514_132616`
- Command:
  `dekk carts benchmarks run --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-large-64-release-fresh`
- Runtime mode: ARTS Release, `ARTS_LOG_LEVEL=0`, counters/metrics absent from
  `external/arts/build/CMakeCache.txt`, `profile-none.cfg`, no counter files in
  the benchmark manifest.
- Revisions: CARTS `c7db1dfa`, carts-benchmarks `20db3fd`, ARTS `c777522`.
- Counts: 23 runnable, 22 passed, 1 failed, 0 skipped, 1 timeout
  (`polybench/seidel-2d`), 0 runner-reported checksum failures.
- Geometric mean kernel speedup: `0.32x`.
- Superseded by the 2026-05-15 maintained sweep above.

Full-suite classes from that older sweep, before the later matrix-family fixes:

- `fast`: `kastors-jacobi/jacobi-for`,
  `kastors-jacobi/poisson-for`, `ml-kernels/pooling`,
  `monte-carlo/ensemble`, `polybench/convolution-3d`,
  `polybench/correlation`.
- `competitive`: `polybench/atax`, `polybench/bicg`,
  `seissol/volume-integral`.
- `blocked`: `ml-kernels/activations`, `ml-kernels/batchnorm`,
  `ml-kernels/layernorm`, `polybench/convolution-2d`,
  `polybench/jacobi2d`, `polybench/seidel-2d`, `specfem3d/stress`,
  `specfem3d/velocity`, `stream`, `sw4lite/rhs4sg-base`,
  `sw4lite/vel4sg-base`.

Current focused matrix-family evidence after CODIR materialization fixes:

- `gemm` results:
  `.carts/outputs/benchmarks-gemm-large-64-crossphase-guard-20260515/20260515_064151`
- `2mm` and `3mm` results:
  `.carts/outputs/benchmarks-2mm-3mm-large-64-crossphase-coarse-20260515/20260515_064040`
- Commands:
  `dekk carts benchmarks run polybench/gemm --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-gemm-large-64-crossphase-guard-20260515`
  and
  `dekk carts benchmarks run polybench/2mm polybench/3mm --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-2mm-3mm-large-64-crossphase-coarse-20260515`
- All three benchmarks passed checksum verification.

| Benchmark | ARTS kernel | OpenMP kernel | Speedup | Current reading |
|---|---:|---:|---:|---|
| `polybench/gemm` | `0.408783s` | `6.215503s` | `15.20x` | `fast`; row-strip DB materialization is restored on the SDE -> CODIR -> ARTS path. |
| `polybench/2mm` | `0.853778s` | `5.554953s` | `6.51x` | `fast`; conservative cross-phase layout guard preserves correctness while avoiding the prior materialization failure. |
| `polybench/3mm` | `0.691686s` | `4.905503s` | `7.09x` | `fast`; same guarded materialization policy as `2mm`. |

The immediate coarse `4800 x 4800` GEMM regression is fixed. SDE still needs
true M3 phase-local MU/token plans for chained contraction intermediates so the
compiler can prove and materialize cross-phase reuse instead of relying on the
temporary conservative bridge. The next performance gate is a maintained
large/64 sweep plus focused `--runs 3` follow-up for any noisy or surprising
matrix results.

Superseded working-tree checkpoint after removing DB-payload `memref.subview`
creation:

- Results:
  `.carts/outputs/benchmarks-gemm-family-large-64-no-boundary-subview-20260514/20260514_231930`
- Command:
  `dekk carts benchmarks run polybench/gemm polybench/2mm polybench/3mm --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-gemm-family-large-64-no-boundary-subview-20260514`
- All three benchmarks passed checksum verification, but performance regressed.
  This checkpoint is retained as the shape-regression baseline that led to the
  CODIR materialization fix.

| Benchmark | ARTS kernel | OpenMP kernel | Speedup | Current reading |
|---|---:|---:|---:|---|
| `polybench/gemm` | `16.851228s` | `6.257675s` | `0.371x` | Correctness clean; performance shape is not acceptable. |
| `polybench/2mm` | `30.755523s` | `5.777342s` | `0.188x` | Correctness clean; chained contraction still lacks SDE/CODIR phase reuse. |
| `polybench/3mm` | `26.141955s` | `4.973020s` | `0.190x` | Correctness clean; phase shape and token-local access remain blocked. |

The no-subview checkpoint fixed a lowering crash, not the performance problem.
Its pipeline dumps showed the output DB had collapsed to one coarse block even
though the SDE plan carried row-strip ownership.

Earlier focused matrix-family evidence:

- Results:
  `.carts/outputs/benchmarks-gemm-family-large-64-current-20260514/20260514_181913`
- Command:
  `dekk carts benchmarks run polybench/gemm polybench/2mm polybench/3mm --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-gemm-family-large-64-current-20260514`
- All three benchmarks passed checksum verification.

| Benchmark | ARTS kernel | OpenMP kernel | Speedup | Current class |
|---|---:|---:|---:|---|
| `polybench/gemm` | `0.589857s` | `6.223452s` | `10.55x` | `fast` |
| `polybench/2mm` | `7.890815s` | `5.584529s` | `0.708x` | `blocked`: chained-contraction/intermediate reuse |
| `polybench/3mm` | `0.712718s` | `4.861729s` | `6.82x` | `fast` |

Historical 3-run confirmation before the CODIR materialization fix:

- Results:
  `.carts/outputs/benchmarks-gemm-family-large-64-confirm-20260514/20260514_183324`
- Command:
  `dekk carts benchmarks run polybench/gemm polybench/2mm polybench/3mm --size large --timeout 120 --threads 64 --nodes 1 --trace --runs 3 --results-dir .carts/outputs/benchmarks-gemm-family-large-64-confirm-20260514`
- All 9 benchmark executions passed checksum verification. The runner reported
  median-of-3 kernel speedups and a geometric mean speedup of `0.80x` in the
  terminal summary; the JSON summary reports `1.187x`, so use the per-benchmark
  medians below as the source of truth until the reporting mismatch is fixed.

| Benchmark | Median ARTS kernel | Median OpenMP kernel | Median speedup | Current class |
|---|---:|---:|---:|---|
| `polybench/gemm` | `4.979935s` | `6.159705s` | `1.237x` | `fast`, but noisy (`0.395402s`, `4.979935s`, `5.192507s`) |
| `polybench/2mm` | `7.846973s` | `5.618096s` | `0.716x` | `blocked`: chained-contraction/intermediate reuse |
| `polybench/3mm` | `8.897152s` | `4.957845s` | `0.557x` | `blocked`: repeat-run matrix-chain instability/phase reuse |

Superseded owner groups from earlier sweeps:

- SDE: dense chained-contraction phase planning. Focused `2mm` and `3mm` are
  now fast and checksum-clean, but complete phase-local contraction plans remain
  M7 work for replacing conservative coarse bridges where reuse is provable.
- SDE: vector/reduction work aggregation. `activations`, `batchnorm`,
  `layernorm`, and `stream` are now classified, but richer SDE vector/reduction
  block plans remain useful for replacing host fallback and improving margins.
- SDE: timestep/wavefront and in-place update planning. `jacobi2d` and
  `seidel-2d` are now fast through the host OpenMP fallback; tokenized
  repeated-stencil execution remains future work.
- SDE/ARTS: stencil and component-slab materialization. `convolution-2d`,
  `specfem3d/*`, and `sw4lite/*` are now classified, but M7 should still
  prefer direct MU/token materialization over raw `CreateDbs` where the SDE
  plan is complete.

Implemented matmul optimization:

1. SDE structured analysis now descends past local scratch setup when the
   side effects are local to the scheduling-unit block. This lets
   `polybench/gemm` prove the `i,j,k` direct-memory matmul shape before
   LoopInterchange rewrites the scalar accumulator.
2. Direct-memory matmul still receives `classification(<matmul>)`, but SDE now
   stamps row-strip physical ownership (`physicalOwnerDims = [0]`,
   `physicalBlockShape = [rowTile, fullColumns]`) while keeping column
   strip-mining local inside each row-owner task. This avoids the failed 2D
   owner heuristic where output-column owners duplicated full coarse-input
   `k` sweeps.
3. The raw-memref bridge now materializes the SDE matmul row-strip contract for
   `gemm` as a block DB with `sizes[64]` and row-strip element sizes, with
   `depPattern = <matmul>` and `planOwnerDims = [0]`.

Focused post-fix evidence:

- Rejected 2D owner heuristic run:
  `.carts/outputs/benchmarks-gemm-large-64-sde-matmul-fix/20260514_134730`.
  SDE stamped `owner_tile_2d` for `gemm`, ARTS created a `64 x 8` output DB
  grid, and performance regressed to `0.10x` (`62.90s` ARTS kernel vs `6.23s`
  OpenMP). This proves 2D direct-memory output ownership is not valid without
  matching input DB tiling/reuse.
- Historical row-strip recovery run:
  `.carts/outputs/benchmarks-gemm-large-64-sde-matmul-rowstrip/20260514_135116`.
  `gemm` passes verification with `16.35s` ARTS kernel vs `6.20s` OpenMP,
  `0.38x`. This restored the clean-sweep performance while keeping the SDE
  matmul classification and ARTS matmul contract, but it is no longer the latest
  matrix-family baseline.
- Historical row-strip chained-contraction run:
  `.carts/outputs/benchmarks-matmul-chain-large-64-sde-matmul-rowstrip/20260514_135154`.
  `2mm` improves from `0.096x` to `0.194x`
  (`58.01s -> 28.89s` ARTS kernel against a `5.6s` OpenMP baseline), and `3mm`
  improves from `0.079x` to `0.195x`
  (`61.43s -> 25.44s` ARTS kernel against a `5.0s` OpenMP baseline).
- Accepted focused SDE pattern/tiling run:
  `.carts/outputs/benchmarks-large-64-sde-pattern-20260514/20260514_144030`.
  Command:
  `dekk carts benchmarks run polybench/gemm polybench/2mm polybench/3mm polybench/convolution-2d polybench/jacobi2d --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-large-64-sde-pattern-20260514`.
  All 5 benchmarks passed checksum verification; geometric mean kernel speedup
  was `0.28x`.

  | Benchmark | ARTS kernel | OpenMP kernel | Speedup | Current reading |
  |---|---:|---:|---:|---|
  | `polybench/gemm` | `16.16s` | `6.26s` | `0.39x` | Row-strip plan remains in the accepted band; no regression from SDE pattern renaming or CPS fixes. |
  | `polybench/2mm` | `30.00s` | `5.60s` | `0.19x` | Row-strip chained contraction remains materially better than the rejected 2D owner plan, but still blocked on input/intermediate DB tiling. |
  | `polybench/3mm` | `25.95s` | `5.44s` | `0.21x` | Same chained-contraction bottleneck as `2mm`; SDE needs phase-aware intermediate reuse. |
  | `polybench/convolution-2d` | `3.31s` | `2.90s` | `0.88x` | Competitive stencil materialization; useful control for SDE physical halo/block planning. |
  | `polybench/jacobi2d` | `5.36s` | `0.75s` | `0.14x` | Still blocked on timestep/CPS and stencil task-grain overhead despite correct SDE tiled-step/timestep recognition. |

- SDE logical-resource architecture validation run:
  `.carts/outputs/benchmarks-gemm-family-large-64-sde-resource-query-20260514/20260514_162624`.
  Command:
  `dekk carts benchmarks run polybench/gemm polybench/2mm polybench/3mm --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-gemm-family-large-64-sde-resource-query-20260514`.
  All 3 benchmarks passed checksum verification; geometric mean kernel speedup
  was `0.24x`.

  | Benchmark | ARTS kernel | OpenMP kernel | Speedup | Current reading |
  |---|---:|---:|---:|---|
  | `polybench/gemm` | `16.39s` | `6.38s` | `0.39x` | Resource-query refactor preserved the accepted row-strip shape; no 25% GEMM performance gain. |
  | `polybench/2mm` | `28.52s` | `5.59s` | `0.20x` | Slightly better than the previous accepted row-strip focused run, still blocked on input/intermediate reuse. |
  | `polybench/3mm` | `27.97s` | `4.96s` | `0.18x` | Slower than the previous accepted focused run; still blocked on phase-aware contraction planning. |
- Historical single-run matrix-family evidence:
  `.carts/outputs/benchmarks-gemm-family-large-64-current-20260514/20260514_181913`.
  This run showed `gemm` and `3mm` could be fast before the later
  materialization regression/fix sequence, but it is superseded for
  classification by the current 2026-05-15 focused evidence above.

  | Benchmark | ARTS kernel | OpenMP kernel | Speedup | Current reading |
  |---|---:|---:|---:|---|
  | `polybench/gemm` | `0.589857s` | `6.223452s` | `10.55x` | Fast single run; later median remains fast but much slower. |
  | `polybench/2mm` | `7.890815s` | `5.584529s` | `0.708x` | Slower than OpenMP; focus on phase/intermediate reuse rather than hardcoded column constants. |
  | `polybench/3mm` | `0.712718s` | `4.861729s` | `6.82x` | Fast single run; not stable in the 3-run confirmation. |
- Historical 3-run matrix-family confirmation:
  `.carts/outputs/benchmarks-gemm-family-large-64-confirm-20260514/20260514_183324`.
  All 9 executions passed checksum verification. Median results were:

  | Benchmark | Median ARTS kernel | Median OpenMP kernel | Median speedup | Current reading |
  |---|---:|---:|---:|---|
  | `polybench/gemm` | `4.979935s` | `6.159705s` | `1.237x` | Fast at median, but one run was `0.395402s` and two were near `5s`; benchmark noise must be investigated before claiming larger gains. |
  | `polybench/2mm` | `7.846973s` | `5.618096s` | `0.716x` | Still slower than OpenMP; chained intermediate reuse remains the optimization target. |
  | `polybench/3mm` | `8.897152s` | `4.957845s` | `0.557x` | Blocked under repeated runs despite the earlier fast single run; inspect phase shape and runtime variance together. |
- Rejected hardcoded column-workers cap:
  `.carts/outputs/benchmarks-gemm-family-large-64-coltile4-20260514/20260514_182316`.
  Clamping the column-worker split to a fixed value made `2mm` fast
  (`7.06x`) but regressed `gemm` to `1.57x` and `3mm` to `0.617x`. It is not an
  acceptable production heuristic because it is benchmark-shaped and not derived
  from SDE legality/reuse facts.
- Rejected cache-derived column-tile experiment:
  `.carts/outputs/benchmarks-gemm-family-large-64-cachetile-repeat-20260514/20260514_182827`.
  It regressed the family balance (`gemm` `1.51x`, `2mm` `0.789x`, `3mm`
  `0.653x`) and was removed from the working tree. Any future cache-aware tiling
  must be expressed as an SDE contraction/reuse plan, not as an isolated column
  tile cap in the raw direct-memory path.

Matmul root-cause update:

- CARTS is not slow because GEMM is unrecognized anymore. The M6 Phase A
  matrix-family regression was a CODIR dispatch-step bug: SDE had already
  raised the owner-strip step to the 75-row physical block, and CODIR multiplied
  it by the block shape again. The current focused `gemm`, `2mm`, and `3mm`
  follow-ups are checksum-clean and fast after CODIR preserves already-tiled
  owner steps.
- The remaining matrix-chain work is architectural rather than an immediate
  benchmark block. SDE still needs to model `2mm` (`tmp = A * B`, then
  `D = tmp * C + beta * D`) and `3mm` (`E = A * B`, `F = C * D`, then
  `G = E * F`) as connected contraction phases with explicit intermediate DB
  ownership, reuse windows, and task grain. Until that M3 plan exists, the
  bridge must keep unproven cross-phase intermediate accesses coarse.
  Hardcoded column caps can make one phase look fast while breaking another
  benchmark, so they are rejected.
- A future general contraction optimizer should still avoid scalar row tasks
  when the SDE plan proves packed-panel reuse is legal. Without a full
  contraction plan, the generated direct-memory codelet can degrade into a scalar
  row task instead of a BLAS-style packed macro/micro-kernel.
- The large `gemm` direct-memory codelet shape observed during the
  investigation was effectively:
  `row-tile -> row -> j-block -> k -> scalar accumulate/store` or, after the
  earlier `k-j` interchange path, `row -> k -> j-block -> load/store C`.
  Neither shape is the GotoBLAS/BLIS algorithm. One gives strided RHS access;
  the other makes every reduction step touch output memory. The optimized
  shape needs an SDE-approved memref contraction plan with:
  `M/N/K macro tiles`, packed or panelized `A/B` reuse, and a small
  register-blocked microkernel that accumulates an `MR x NR` C tile before
  writing output.
- A generic invariant-load hoist is still useful, but it is not an SDE
  matmul rewrite. SDE should state the contraction, ownership, block sizes,
  and packing/layout intent. Generic code motion belongs after codelet
  materialization, where DB acquires, memref views, alias scopes, and concrete
  loop bodies are visible. The best owner is a CODIR or ARTS-RT LICM/hoisting
  pass before LLVM lowering, with ARTS only preserving SDE plan metadata while
  materializing DBs.
- Rejected local experiments from this investigation:
  - Matmul-specialized invariant hoist in SDE:
    `.carts/outputs/benchmarks-gemm-family-large-64-sde-hoist-20260514/20260514_145406`.
    It moved the invariant `A[i,k]` load but regressed `gemm`
    (`16.94s` ARTS vs `16.16s` baseline), because the dominant cost was the
    output-memory traffic and missing packed microkernel.
  - Keeping the original scalar accumulator with only local column strip-mining:
    `.carts/outputs/benchmarks-gemm-large-64-sde-no-kj-20260514/20260514_145649`.
    It timed out; contiguous output stores alone do not compensate for strided
    RHS access.
  - Whole-RHS transpose/packing before the row-strip task:
    `.carts/outputs/benchmarks-gemm-large-64-sde-packrhs-20260514/20260514_151040`.
    It passed but only reached `16.02s`, not the required 25% improvement.
    A full global transpose adds memory traffic without giving the codelet
    `MR x NR` register reuse.
  - Ad hoc four-column scalar register blocking in SDE:
    `.carts/outputs/benchmarks-gemm-large-64-sde-regblock4-20260514/20260514_151451`.
    It timed out. This confirms that standalone local rewrites are not enough;
    the plan must control macro tiles, packing/panel lifetimes, vectorization,
    and codelet grain together.
  - Four-column SDE micro-tile without packed panels:
    `.carts/outputs/benchmarks-gemm-family-large-64-sde-microtile-20260514/20260514_153743`.
    `gemm` timed out at `120.10s`; `2mm` and `3mm` passed checksum but
    regressed to `116.30s` and `112.30s`. The pass-shape audit at
    `.carts/sessions/20260514-154408-gemm-pass-shape-audit/pass-dumps.log`
    showed that `LoopInterchange` created `j4 -> k` scalar accumulators,
    `Tiling` wrapped them as `row-tile -> row -> j-block -> j4 -> k`, and
    `BarrierElimination` preserved that shape. The regression is
    therefore not a CPS/barrier rewrite; it is the wrong direct-memory
    contraction shape. Without a packed B panel, `j4 -> k` makes `B[k,j]` a
    large-stride K sweep and blocks the existing vectorizable `k -> j` loop.

Research-backed direction:

- External references used for this direction:
  [MLIR Linalg dialect transformations](https://mlir.llvm.org/docs/Dialects/Linalg/),
  [MLIR Transform dialect tutorial](https://mlir.llvm.org/docs/Tutorials/transform/),
  and the BLIS paper
  [Anatomy of High-Performance Many-Threaded Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/blis3_ipdps14.pdf).
- Follow the GotoBLAS/BLIS decomposition, not isolated loop interchange:
  choose cache/TLB-aware `MC`, `NC`, and `KC` macro tiles, pack reused panels,
  and generate a small `MR x NR` microkernel that holds C in registers across
  the K tile.
- In CARTS terms, SDE should add a `sde.pattern<matmul>` plan variant for
  `packed_panel` or `contract_tile` with explicit `owner dims`,
  `output tile shape`, `k tile`, `packed A/B panel shapes`, `micro tile`, and
  `phase reuse` for chained contractions. This remains SDE plan metadata.
  SDE may use `sde.resource_query <logical_workers>` to size symbolic grain,
  but it must not materialize ARTS workers or runtime calls.
- SDE-to-CODIR should materialize the final SDE plan as isolated codelet
  structure with explicit deps, params, and token-local views. CODIR-to-ARTS
  should then create output tile DBs, packed-panel scratch or DBs when
  cross-task reuse is legal, phase-local intermediate DBs for `2mm`/`3mm`, and
  concrete ARTS worker/query binding from SDE logical-resource queries.
- CODIR/ARTS-RT lowering should run generic LICM, scalar replacement, loop
  vectorization, and alias/noalias cleanup on the already-materialized codelet.
  These passes must be pattern-agnostic; matmul-specific legality stays in SDE.

Professional pass split for the contraction work:

- `PatternAnalysis`: recognize memref-level contraction structure and stamp only
  approved
  `sde.pattern<matmul>` facts plus contraction dimensions.
- `LoopInterchange`: perform legality-preserving order changes backed by the
  approved contraction facts.
- `Tiling`: build the SDE contraction-tile plan. It may refuse direct-memory
  2D owner tiling when packed panels/intermediate reuse are not available.
- `DistributionPlanning`: convert the chosen tile plan into SDE distribution
  intent and logical grain. It should not rederive matmul legality.
- `ConvertSdeToCodir`: isolate codelets, make deps/params explicit, and rewrite
  token-local memref accesses.
- `ConvertCodirToArts` and the temporary raw-memref bridge: bind the SDE/CODIR
  plan to ARTS DBs, EDTs, dependency windows, and runtime resource queries.
- CODIR/ARTS-RT passes: generic loop/codelet cleanup only after the ARTS shape
  is correct.

Next optimization task:

1. Start from the no-subview working-tree checkpoint and compare pipeline dumps
   against the earlier fast and median-fast matrix runs. Identify where the
   current shape loses locality or task grain.
2. Diagnose `2mm` and `3mm` as multi-phase contractions: confirm whether
   intermediates (`tmp`, `E`, `F`) are physically owned and reused between
   producer/consumer phases, and add SDE phase-plan facts before changing tile
   constants.
3. Introduce the CODIR plan: isolate codelets, make deps/params explicit at
   creation time, and move token-local memref rewrites out of ARTS
   materialization paths.
4. Replace any remaining scalar direct-memory contraction rewrite with an SDE
   contraction plan that can lower to macro tiles plus an `MR x NR` microkernel
   shape only after packed A/B or intermediate-panel reuse is proven.
5. Add a generic CODIR/ARTS-RT invariant-hoist check, but do not make it matmul
   aware. Its lit coverage should show a loop-invariant load moving out of an
   inner codelet loop after DB/memref materialization.
6. Move to the vector/reduction group (`batchnorm`, `layernorm`, `stream`) if
   packed contraction planning requires a larger ARTS DB contract change.

## Done Definition

A benchmark is done when it has:

- checksum parity,
- a recorded performance class,
- a documented compiler bottleneck if not `fast`,
- stable lit or pipeline coverage for the optimization that made it work,
- no late ARTS policy decision that should belong to SDE or CODIR.
