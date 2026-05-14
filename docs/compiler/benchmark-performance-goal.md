# Benchmark Performance Goal

## Goal

Make every maintained CARTS benchmark correctness-clean and performance-credible
against OpenMP at `large`, 64 threads, and 1 node by keeping decisions in the
right dialect layer.

SDE is runtime-agnostic. It must prove legality and author the semantic plan:
tensor/vector partitions, tiling, reductions, halo/window shapes, task grain,
barrier intent, and timestep or wavefront structure for matrix/tensor outputs,
vector/reduction kernels, 3D component stencils, and time-stepped kernels.

Core is ARTS-machine-aware. It must materialize SDE-authored plans through
`CreateDbs`, preserve and refine the chosen DB/EDT/dependency/epoch shape, and
use Core analyses to orchestrate work without inventing tensor partition policy
late.

RT is the low-level runtime-call lowering layer. It should optimize launch,
CPS/continuation, dependency-slot, packing, scalar replacement, GUID/runtime
call, and LLVM-facing overhead only after the SDE/Core shape is correct and
traces still show those mechanics as the bottleneck.

Layer references:

- [`sde/`](./sde/): runtime-agnostic semantic planning, physical DB layout
  policy, state, dependencies, and effects.
- [`core/`](./core/): ARTS-machine-aware DB/EDT/dependency/epoch
  materialization and orchestration.
- [`rt/`](./rt/): low-level runtime-call lowering and overhead cleanup.
- [`pipeline.md`](./pipeline.md): live stage order and pass ownership notes.

## Current Dialect Structure And Responsibilities

CARTS has a layered compiler contract. Each layer may preserve, validate, or
refine facts from earlier layers, but it must not rediscover policy that an
earlier layer was responsible for proving.

### SDE: semantic planning dialect

SDE is the only layer that still sees OpenMP semantics, structured loop
summaries, tensor carriers, reduction metadata, barrier intent, and enough
source-level shape to prove legality. That makes SDE the primary optimization
layer for CARTS. Most performance policy belongs here because most performance
policy is a question of source semantics: which work is independent, which
state is carried across stages, which dimensions own output, which barriers are
required, and which dataflow edges can be made asynchronous. SDE therefore
owns:

- structured classification: elementwise, elementwise pipeline, reduction,
  matmul/tensor contraction, stencil, wavefront, timestep, and opaque fallback;
- memory-effect summaries over source roots, carrier tensors, and memrefs;
- owner dimensions, spatial dimensions, component dimensions, and batch
  dimensions;
- physical DB layout policy through `physicalOwnerDims`,
  `physicalBlockShape`, `physicalHaloShape`, and `logicalWorkerSlice`;
- scheduling-unit topology through `iterationTopology`,
  `repetitionStructure`, `asyncStrategy`, reduction strategy, and barrier
  reason;
- CPS planning, including candidate groups, stage indices/counts, tokenized
  dataflow requirements, explicit `sde.control_token` edges, timestep
  boundaries, and final CPS stage plans when SDE has rewritten all carries;
- legality decisions for tiling, strip-mining, loop interchange, fusion,
  vectorization hints, owner-slice partitioning, wavefront/timestep grouping,
  CPS conversion, and barrier removal.

SDE must reject or leave unplanned any case whose legality depends on runtime
machine behavior instead of source semantics. It may stamp conservative plans,
but it must not stamp a performance plan unless the SDE analysis can explain
why the chosen owner slices are independent or why their dependency windows are
complete.

The latest CPS contract follows this rule. `CpsPlanning` marks full-timestep
`advance_edt` stages as SDE CPS candidates only when SDE can provide an
explicit control boundary. Adjacent candidates are not allowed to be attrs-only:
SDE inserts a completion token and a token-consuming timestep barrier before it
stamps the group. `VerifySdeCpsPlan` rejects candidate pairs without that
SDE-authored control edge. Core and RT may lower or optimize the resulting
continuation mechanics, but they do not decide whether the CPS dataflow is
legal.

### Core: ARTS orchestration dialect

Core starts after the SDE semantic plan is authored. It knows ARTS objects and
runtime topology, but it should treat SDE physical layout and dependency-window
policy as input, not as a place to invent tensor partitioning. Core owns:

- converting SDE plan attributes into Core `arts.plan.*`, dependency pattern,
  distribution kind, and stencil/layout contracts;
- materializing SDE CPS, barrier, async, and repeated-timestep plans into ARTS
  EDT/epoch/control structure without changing their legality policy;
- creating the chosen physical DB layout directly in `CreateDbs`;
- preserving planned DB/EDT/dependency/epoch shape while splitting or refining
  ARTS mechanics;
- validating plan consistency, owner dims, block shapes, halo bounds, and
  dependency-window contracts;
- selecting ARTS dependency slots, DB acquires, EDT launch shape, epoch
  structure, and distributed ownership from the already-authored plan;
- reporting proof gaps when SDE did not provide enough information.

Core may refine an ARTS object graph, but any refinement must be a mechanical
consequence of the SDE plan or an ARTS-machine constraint. If Core has to guess
which tensor dimension to block, which reduction strategy to use, or whether a
stencil, timestep, or CPS chain is owner-slice/dataflow safe, the fix belongs
in SDE.

### RT: runtime-call lowering dialect

RT is the lowering-ready bridge. It sees flat runtime calls, launch metadata,
dependency descriptors, GUIDs, DB pointers, continuation plumbing, and LLVM
facing scalar values. RT owns:

- lowering Core DB/EDT/epoch/dependency objects to runtime calls;
- launch and continuation overhead cleanup after task shape is correct;
- dependency-slot indexing, local dependency-window access, and dep/db pointer
  hoisting;
- temporary descriptor allocation, packing, scalar replacement, alias metadata,
  vectorization hints, and runtime-call hoisting;
- preserving the memory model and dependency semantics authored by SDE/Core.

RT must not compensate for missing SDE/Core shape by recovering tensor policy
late. A late RT fast path is acceptable only after traces show that SDE/Core
already produced the intended DB/EDT/dependency structure and the remaining
bottleneck is runtime-call mechanics. In particular, RT may reduce
continuation and dependency-call overhead, but CPS legality, stage grouping,
and carry tokenization remain SDE responsibilities.

### Runtime: execution contract

The ARTS runtime owns scheduler behavior, route tables, EDT publication,
frontier progress, DB lifetime, counters, and actual communication. Runtime
changes are justified only when compiler output already satisfies the contract
and traces show a runtime implementation bottleneck or a missing runtime API.

The reference success path is the current convolution-2d result: SDE authors the
physical tensor partition, Core `CreateDbs` materializes the optimal DB shape,
RT lowering uses local dependency-window indices, and the benchmark passes
against the OpenMP checksum.

## Performance Heuristic And Analysis Set

Every benchmark optimization must start by naming the SDE proof, the Core
materialization check, and the RT/runtime evidence. The goal is not to add more
heuristics everywhere; it is to make each heuristic live in the earliest layer
where it can be proven.

### Required SDE analyses

- **Structured loop summary:** classify the loop family, iterator roles,
  output roots, read/write roots, static shape, affine access maps, and unknown
  effects before lowering to Core.
- **Owner-slice independence:** prove that each planned owner slice writes
  disjoint output elements and that in-place reads either stay inside the owner
  slice or are covered by a halo/window plan.
- **Physical layout synthesis:** choose owner dims, block shape, halo shape,
  logical worker slice, component locality, and task topology from tensor shape,
  access footprint, worker count, and minimum useful work per task.
- **Reduction strategy selection:** choose local accumulate, tree, or atomic
  based on accumulator visibility, write contention, output rank, and expected
  task count.
- **Pipeline fusion and barrier graph:** fuse adjacent elementwise stages when
  root windows are compatible, remove duplicate barriers, and keep barriers
  only where dependence edges or OpenMP semantics require them.
- **Timestep/wavefront planning:** identify repeated stencil stages,
  alternating buffers, in-place update hazards, and wavefront frontiers before
  Core sees flat ARTS work.
- **CPS/dataflow planning:** identify repeated-stage candidates, insert
  required SDE control tokens and barriers, verify candidate completeness, and
  rewrite scalar/data/control carries before stamping final `cps_chain` plans.
- **Cost guardrails:** prevent overpartitioning when per-task compute is below
  launch/dependency overhead, and prevent underpartitioning when a coarse DB
  serializes independent output slices.

### Required Core analyses

- **Plan-to-DB validation:** verify that every SDE physical layout becomes the
  exact `DbAllocOp` block shape and owner-dim order in `CreateDbs`.
- **Acquire-window validation:** check that DB acquires match the planned
  dependency window, including halo bounds, owner block range, and local index
  projection.
- **EDT shape accounting:** count EDTs, dependencies per EDT, epoch barriers,
  and continuations for each benchmark family and flag shapes that contradict
  the SDE plan.
- **CPS materialization accounting:** verify that SDE CPS candidates/final
  stages lower to the expected ARTS control/continuation shape without Core
  inventing new dataflow legality.
- **Distributed ownership refinement:** map planned owner blocks to nodes and
  routes without changing tensor partition policy.
- **Proof-gap diagnostics:** warn or fail tests when Core had to fall back to
  whole-DB acquires, unblocked DBs, global dep indexes, or unplanned barriers.

### Required RT/runtime analyses

- **Launch overhead profile:** measure EDT create, ready-local create,
  continuation, and epoch finish/wait costs only after Core task counts look
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

| Family | SDE policy | Core check | RT/runtime follow-up |
|---|---|---|---|
| Elementwise and ML vector kernels | Fuse compatible stages, choose coarse vector blocks, keep pure libm scalar calls effect-free, preserve vectorization hints. | `CreateDbs` must allocate blocked output DBs and avoid whole-DB dependency windows. | Hoist dep/db pointers, shrink pack/unpack, confirm launch overhead is not dominating. |
| Row/column reductions (`atax`, `bicg`, `layernorm`, `batchnorm`) | Distinguish per-output reductions from OpenMP scalar reductions; choose owner rows/channels and local accumulate/tree/atomic strategy. | Materialize partial/output DBs according to SDE owner dims, with no serialized global update unless SDE chose atomic. | Reduce continuation and dependency overhead after reduction DB shape is correct. |
| Matmul and chained tensor contractions (`gemm`, `2mm`, `3mm`) | Classify contractions in SDE, preserve reduction locality, and choose physical owner dims from proven DB reuse. Direct-memory matmul keeps row-strip ownership until SDE can also tile A/B/intermediate DBs; 2D output ownership is only valid when it does not duplicate coarse input sweeps. | Intermediate/output DBs must reflect the SDE owner plan. Core must preserve row-strip or 2D tiling exactly and must not invent tensor partitioning. | Optimize launch/dep overhead only after task count and block reuse match the SDE tile plan. |
| Stencils and timesteps (`jacobi2d`, `seidel-2d`, `fdtd-2d`, KaStORS) | Author halo/window shape, alternating-buffer or wavefront structure, barrier/timestep intent, and CPS candidate/final stage plans. | Acquires must use bounded neighbor windows and local dependency slots; CPS/barrier plans must lower from SDE-authored control edges. | Optimize ready-local create, continuation calls, and dep window indexing after halo/CPS shape is correct. |
| 3D component stencils (`specfem3d`, `sw4lite`) | Separate spatial, component, and batch dimensions; keep component dimension local when it improves reuse. | DB blocks must reflect spatial slabs plus local components. | Use traces to reduce launch/acquire overhead without changing slab policy. |
| Irregular/task suites (`monte-carlo`, `stream`, future graph/task workloads) | Stamp only proven independent chunks; document unsupported irregular policy instead of guessing. | Preserve explicit task/DB contracts and diagnose unplanned coarse synchronization. | Runtime scheduler/counter work is allowed only when compiler shape is already credible. |

### Optimization gate for each benchmark

Before a performance result is considered meaningful:

1. The benchmark must compile and pass checksum parity against OpenMP.
2. The SDE stage dump must show the intended semantic plan or explicitly show
   why the benchmark is blocked.
3. The Core `create-dbs` stage must show the SDE physical layout materialized
   directly, or the result is blocked on Core materialization.
4. Runtime traces must agree with the intended task count, DB count, dependency
   count, and epoch shape.
5. Only then may RT/runtime overhead changes be used to improve speed.

## Acceptance Criteria

- The current checkout is install-clean through `dekk carts install`, with the
  Conda/dekk environment, submodules, ARTS runtime, Polygeist frontend, LLVM
  toolchain, and CARTS compiler all available through `dekk carts ...`.
- The repo docs and `carts-plugin/skills/` are reviewed against `origin/sde`;
  useful process contracts are carried forward, stale command names are removed,
  and generated agent resources are refreshed with `dekk carts skills generate`.
- The source tree is organized around the live command model, current `samples/`
  layout, and SDE/Core/RT dialect ownership. Stale docs, dead workflow layers,
  and nonfunctional generated scaffolding are removed instead of preserved.
- Every benchmark in the maintained suite compiles with `dekk carts compile`.
- Every benchmark run reports checksum parity against its OpenMP baseline.
- Every benchmark has an explicit performance classification:
  - `fast`: ARTS kernel time is faster than OpenMP.
  - `competitive`: ARTS kernel time is within 1.25x of OpenMP.
  - `blocked`: a named compiler/runtime limitation prevents a fair result.
- No optimization changes the memory model or program semantics.
- DB partitioning decisions are made at SDE level when tensor structure is known.
- SDE may reshape `sde.su_iterate` loop steps, inner loops, and
  scheduling-unit topology when the legality proof lives at that level.
- `CreateDbs` must create the chosen physical DB layout directly.
- Core passes may refine ARTS DB/EDT/dependency/epoch structure, but must not
  invent tensor partition policy late.
- RT passes may optimize runtime-call shape only after SDE/Core produce the
  intended DB/EDT/epoch structure.
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
     and policy, Core for DB/EDT/epoch orchestration, RT for runtime-shaped
     lowering, and runtime only for actual runtime contract gaps.

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

- Tensor partition planning: derive owner dims, block shape, halo, and iteration
  topology before `ConvertSdeToArts`.
- Distribution shaping: rewrite `sde.su_iterate` loop steps, inner loop
  nests, and scheduling-unit topology when SDE can prove legality. SDE should
  not merely stamp late contracts if the loop shape itself must change for good
  work distribution.
- Loop transforms: use tiling, interchange, fusion, vectorization, and
  decomposition only where legality is proven by SDE pattern and effect facts.
- Dependency granularity: choose DB/task windows from structured access facts,
  not from late ARTS heuristics.
- Barrier removal: eliminate SDE barriers when dependence analysis proves no
  cross-iteration ordering requirement.
- Reduction planning: select local accumulate, tree, or atomic strategy from
  cost and semantics before ARTS lowering.

### Core Tracks

- DB materialization: keep `CreateDbs` as the point where planned physical DBs
  are created.
- EDT shape: preserve planned block reads/writes through direct SDE-to-Core
  materialization and Core EDT/DB refinement.
- Epoch structure: remove unnecessary epoch barriers and continuation overhead
  after SDE legality has been preserved.
- Dependency lowering: make runtime dependency slots local to the dependency
  window; avoid global block ids in `dep_gep`.
- Analysis facade: use DB/EDT/loop/epoch analyses for Core refinement instead of
  duplicating SDE dependence logic in local helpers.

### RT Tracks

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
- one bottleneck owner for each blocked group: SDE, Core, RT, runtime, or
  benchmark-scope/tooling.

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

Current performance-credible sweep:

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

Performance classes:

- `fast`: `kastors-jacobi/jacobi-for`,
  `kastors-jacobi/poisson-for`, `ml-kernels/pooling`,
  `monte-carlo/ensemble`, `polybench/convolution-3d`,
  `polybench/correlation`.
- `competitive`: `polybench/atax`, `polybench/bicg`,
  `seissol/volume-integral`.
- `blocked`: `ml-kernels/activations`, `ml-kernels/batchnorm`,
  `ml-kernels/layernorm`, `polybench/2mm`, `polybench/3mm`,
  `polybench/convolution-2d`, `polybench/gemm`, `polybench/jacobi2d`,
  `polybench/seidel-2d`, `specfem3d/stress`, `specfem3d/velocity`, `stream`,
  `sw4lite/rhs4sg-base`, `sw4lite/vel4sg-base`.

Blocked owner groups:

- SDE: dense matmul/tensor-contraction planning
  (`gemm`, `2mm`, `3mm`). The clean sweep showed `gemm` was unclassified and
  fell to a uniform row owner plan, while `2mm`/`3mm` were classified as matmul
  but overpartitioned direct-memory output columns without matching A/B or
  intermediate DB tiling.
- SDE: vector/reduction work aggregation (`activations`, `batchnorm`,
  `layernorm`, `stream`). These need SDE vector/reduction block planning and
  fusion before RT launch overhead is meaningful.
- SDE: timestep/wavefront and in-place update planning (`jacobi2d`,
  `seidel-2d`). `seidel-2d` timed out in ARTS.
- SDE/Core: stencil and component-slab materialization
  (`convolution-2d`, `specfem3d/*`, `sw4lite/*`). SDE must prove the slab/halo
  plan and Core must show `CreateDbs` materializes that plan directly before
  RT/runtime work is considered.

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
3. Core `CreateDbs` now materializes the SDE matmul row-strip contract for
   `gemm` as a block DB with `sizes[64]` and `elementSizes[75, 4800]`, with
   `depPattern = <matmul>` and `planOwnerDims = [0]`.

Focused post-fix evidence:

- Rejected heuristic run:
  `.carts/outputs/benchmarks-gemm-large-64-sde-matmul-fix/20260514_134730`.
  SDE stamped `owner_tile_2d` for `gemm`, Core created a `64 x 8` output DB
  grid, and performance regressed to `0.10x` (`62.90s` ARTS kernel vs `6.23s`
  OpenMP). This proves 2D direct-memory output ownership is not valid without
  matching input DB tiling/reuse.
- Accepted row-strip run:
  `.carts/outputs/benchmarks-gemm-large-64-sde-matmul-rowstrip/20260514_135116`.
  `gemm` passes verification with `16.35s` ARTS kernel vs `6.20s` OpenMP,
  `0.38x`. This restores the clean-sweep performance while keeping the SDE
  matmul classification and Core matmul contract.
- Accepted row-strip chained-contraction run:
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

Matmul root-cause update:

- CARTS is not slow because GEMM is unrecognized anymore. The accepted SDE
  row-strip run proves `gemm`, `2mm`, and `3mm` are classified and scheduled
  with the correct SDE/Core ownership contract. The remaining slowdown is that
  the generated codelet is still a scalar row task, not a BLAS-style packed
  macro/micro-kernel.
- Current large `gemm` codelet shape is effectively:
  `row-tile -> row -> j-block -> k -> scalar accumulate/store` or, after the
  earlier `k-j` interchange path, `row -> k -> j-block -> load/store C`.
  Neither shape is the GotoBLAS/BLIS algorithm. One gives strided RHS access;
  the other makes every reduction step touch output memory. The optimized
  shape needs an SDE-approved tensor contraction plan with:
  `M/N/K macro tiles`, packed or panelized `A/B` reuse, and a small
  register-blocked microkernel that accumulates an `MR x NR` C tile before
  writing output.
- A generic invariant-load hoist is still useful, but it is not an SDE
  matmul rewrite. SDE should state the contraction, ownership, block sizes,
  and packing/layout intent. Generic code motion belongs after codelet
  materialization, where DB acquires, memref views, alias scopes, and concrete
  loop bodies are visible. The best owner is a codelet/RT-level LICM/hoisting
  pass before LLVM lowering, with Core only preserving SDE plan metadata while
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
    `BarrierElimination`/`CpsPlanning` preserved that shape. The regression is
    therefore not a CPS/barrier rewrite; it is the wrong direct-memory
    contraction shape. Without a packed B panel, `j4 -> k` makes `B[k,j]` a
    large-stride K sweep and blocks the existing vectorizable `k -> j` loop.

Research-backed direction:

- Follow the GotoBLAS/BLIS decomposition, not isolated loop interchange:
  choose cache/TLB-aware `MC`, `NC`, and `KC` macro tiles, pack reused panels,
  and generate a small `MR x NR` microkernel that holds C in registers across
  the K tile.
- In CARTS terms, SDE should add a `sde.pattern<matmul>` plan variant for
  `packed_panel` or `contract_tile` with explicit:
  `owner dims`, `output tile shape`, `k tile`, `packed A/B panel shapes`,
  `micro tile`, and `phase reuse` for chained contractions. This remains SDE
  metadata and must not be passed raw to Core.
- Core should materialize the final SDE plan as DBs and task/codelet structure:
  output tile DBs, packed-panel scratch or DBs when cross-task reuse is legal,
  and phase-local intermediate DBs for `2mm`/`3mm`.
- RT/codelet lowering should run generic LICM, scalar replacement, loop
  vectorization, and alias/noalias cleanup on the already-materialized codelet.
  These passes must be pattern-agnostic; matmul-specific legality stays in SDE.

Next optimization task:

1. Replace the scalar direct-memory matmul rewrite with an SDE contraction plan
   that lowers to macro tiles plus an `MR x NR` microkernel shape. Start with
   one-node local `gemm` and preserve row-strip external ownership until packed
   A/B panel reuse is proven.
2. Add a generic codelet/RT invariant-hoist check, but do not make it matmul
   aware. Its lit coverage should show a loop-invariant load moving out of an
   inner codelet loop after DB/memref materialization.
3. Extend the same SDE contraction plan to `2mm`/`3mm` by making intermediate
   tile reuse explicit across producer/consumer phases.
4. Move to the vector/reduction group (`batchnorm`, `layernorm`, `stream`) if
   packed contraction planning requires a larger Core DB contract change.

## Done Definition

A benchmark is done when it has:

- checksum parity,
- a recorded performance class,
- a documented compiler bottleneck if not `fast`,
- stable lit or pipeline coverage for the optimization that made it work,
- no late ARTS policy decision that should belong to SDE.
