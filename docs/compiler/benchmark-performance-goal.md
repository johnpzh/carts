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

The reference success path is the current convolution-2d result: SDE authors the
physical tensor partition, Core `CreateDbs` materializes the optimal DB shape,
RT lowering uses local dependency-window indices, and the benchmark passes
against the OpenMP checksum.

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
- SDE may reshape `arts_sde.su_iterate` loop steps, inner loops, and
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
- Distribution shaping: rewrite `arts_sde.su_iterate` loop steps, inner loop
  nests, and scheduling-unit topology when SDE can prove legality. SDE should
  not merely stamp late contracts if the loop shape itself must change for good
  work distribution.
- Loop transforms: use tiling, interchange, fusion, vectorization, and
  decomposition only where legality is proven by SDE summaries.
- Dependency granularity: choose DB/task windows from structured access facts,
  not from late ARTS heuristics.
- Barrier removal: eliminate SDE barriers when dependence analysis proves no
  cross-iteration ordering requirement.
- Reduction planning: select local accumulate, tree, or atomic strategy from
  cost and semantics before ARTS lowering.

### Core Tracks

- DB materialization: keep `CreateDbs` as the point where planned physical DBs
  are created.
- EDT shape: preserve planned block reads/writes through `ForLowering` and EDT
  distribution.
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
   - missing SDE structured summary,
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

## Done Definition

A benchmark is done when it has:

- checksum parity,
- a recorded performance class,
- a documented compiler bottleneck if not `fast`,
- stable lit or pipeline coverage for the optimization that made it work,
- no late ARTS policy decision that should belong to SDE.
