# Benchmark Performance Goal

## Goal

Make every maintained CARTS benchmark correctness-clean and performance-credible
under the SDE -> ARTS pipeline, with DB granularity, task granularity, and epoch
structure chosen deliberately instead of emerging from late incidental rewrites.

The reference success path is the current convolution-2d result: SDE authors the
physical tensor partition, `CreateDbs` materializes the optimal DB shape, RT
lowering uses local dependency-window indices, and the benchmark passes against
the OpenMP checksum.

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
- ARTS passes may refine runtime structure, but must not invent tensor
  partition policy late.
- Every runnable sample is swept through the user-facing examples flow or the
  equivalent e2e tests before benchmark conclusions are considered final.
- All runnable benchmarks are swept with `--size large --threads 64 --nodes 1`
  unless the runner documents that a workload lacks that size or requires a
  different node mode. Deviations must be recorded in the evidence table.
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
   - Run each listed benchmark with:
     ```bash
     dekk carts benchmarks run <benchmark> --size large --timeout 120 --threads 64 --nodes 1 --trace
     ```
   - For failures, use the triage workflow in this document and update the
     evidence table with exact commands and bottlenecks.

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
out-of-scope algorithmic pattern.

As of 2026-05-12, `dekk carts benchmarks list` exposes 23 runnable entries,
including `sw4lite/*`, and omits disabled KaStORS task, Graph500, LULESH,
Llama2, and `polybench/fdtd-2d` entries. Disabled entries remain tracked below
as `blocked` until they are re-enabled or removed with a documented reason.

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

### ARTS Tracks

- DB materialization: keep `CreateDbs` as the point where planned physical DBs
  are created.
- EDT shape: preserve planned block reads/writes through `ForLowering` and EDT
  distribution.
- Epoch structure: remove unnecessary epoch barriers and continuation overhead
  after SDE legality has been preserved.
- Dependency lowering: make runtime dependency slots local to the dependency
  window; avoid global block ids in `dep_gep`.
- Runtime model: inspect `external/arts` before changing dependency behavior,
  especially byte slices, writable deps, and DB ownership.

## Triage Workflow

For each benchmark:

1. Run the benchmark with trace enabled.

   ```bash
   dekk carts benchmarks run <benchmark> --size small --timeout 120 --threads 16 --nodes 1 --trace
   ```

2. If it fails correctness, stop performance work and fix semantic lowering.

3. Dump the pipeline and identify where the benchmark loses structure.

   ```bash
   dekk carts compile <artifact.mlir> --all-pipelines -O3 --arts-config <arts.cfg> -o /tmp/carts-<benchmark>-pipes
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

## Execution Evidence: 2026-05-12

This section is the project-local record for the 2026-05-12 benchmark goal.

### Install, Docs, And Skills

- `dekk` is installed globally at `/home/raherrer/.local/bin/dekk`; the repo
  flow uses plain `dekk carts ...` commands without per-command PATH prefixes.
- `dekk carts install` completed successfully after making the global `dekk`,
  `conda`, and `mamba` shims discoverable from `~/.local/bin`.
- `dekk carts doctor` passes; the only missing tool is optional `lit`.
- Docs were updated away from stale `dekk carts agents`,
  `tests/examples`, and `tests/contracts` references toward live
  `dekk carts skills`, `samples/`, `tests/e2e/`, and
  `lib/arts/dialect/*/test/` usage.
- `carts-plugin/skills/` was reconciled against `origin/sde`; useful SDE
  skills were carried forward (`carts-simplify`,
  `carts-skill-maintenance`, `carts-pipeline-map`, `carts-dialect-map`,
  `carts-local-examples`, `carts-multinode-examples`, and
  `carts-agentic-development`), stale workflow frontmatter was removed, and
  `dekk carts skills generate` produced 29 Claude-valid skills.
- Overengineered/nonfunctional agent workflow scaffolding was removed:
  `carts-plugin/workflows/` and `carts-plugin/bin/carts-env-check` depended on
  undeclared tooling and were already absent from `origin/sde`.

### Compiler Fixes Landed During The Sweep

- Fixed ARTS-to-LLVM lowering for `arts.db_acquire` with absent
  `source_guid`. The op explicitly permits this shape when chaining from a
  parent EDT dependency, but lowering dereferenced the missing value. The fix
  resolves the paired GUID from the source pointer's defining DB/dep acquire,
  delays `dep_db_acquire` lowering while GUID-less acquires still need that
  pairing, and emits diagnostics instead of forwarding a null replacement.
- Hardened optional `source_guid` handling in epoch lowering.
- Replaced the over-broad `ValueAnalysis::stripMemrefViewOps` implementation
  with explicit stripping of supported memref view ops. The old helper could
  cross a `polygeist.pointer2memref` boundary into a raw pointer and assert in
  MLIR's casting utilities.
- Strengthened SDE barrier elimination to use a full root-level read/write
  conflict relation. Barriers are removable only when adjacent scheduling units
  have no RAW/WAR/WAW conflict; disjoint write-only successors can now proceed,
  while same-root write-after-write loops stay ordered.
- Tightened ARTS Core work-distribution alignment so physical block/stencil DB
  spans are hard chunk-alignment requirements and coarse-derived spans are only
  soft fallbacks. This prevents a late coarse dependency from shrinking worker
  chunks below an SDE-authored physical DB block.
- Kept `DbAnalysis`/`DbGraph` as the Core analysis facade and derived DB/EDT
  cache because active passes use them for dependency construction, DB mode
  tightening, distributed eligibility, and layout refinement. Simplification was
  applied inside that boundary instead: `DbAliasAnalysis` now keeps only
  root/slice/partition overlap facts, removed unused wrapper APIs and the stale
  value-pair cache, and dropped speculative metadata-based no-alias decisions.
- Removed duplicated or misplaced analysis helpers: SDE carrier tensor tracing
  now lives in an SDE analysis utility, SDE distribution planning uses the
  shared value-dependence helper for index expressions, Core memory access
  classification uses `DbUtils::getMemoryAccessInfo`, and plan-family vocabulary
  moved from Core Analysis to a dialect-neutral plan-contract utility.
- Trimmed cost-model surface area to active SDE uses. The cost model remains in
  place for current topology, schedule, tiling, and reduction decisions, but
  unused planned hooks such as allocation cost, L1/cache-line size, and reduction
  split factor were removed rather than kept as speculative API.
- Removed inferred `nowait` stamping from SDE barrier elimination because it was
  not consumed by SDE-to-ARTS conversion. Explicit OpenMP `nowait` is still
  preserved; inferred synchronization removal is now represented only by the
  `barrier_eliminated` marker that actually controls `arts.barrier` lowering.
- Removed dormant analysis scaffolding instead of carrying unused abstractions:
  `InformationCache`, stale metadata `DependenceAnalyzer`, documentary-only
  `AnalysisDependencies`, unused `StringAnalysis` bulk accessors, and unused
  EDT capture/invariance/reachability APIs. Live entities remain grouped by
  owner: SDE keeps structured semantic analysis, Core keeps DB/EDT/loop graph
  analysis, and RT stays lowering-facing.
- Consolidated SDE root-level memory effect collection into
  `SdeAnalysisUtils`. Barrier legality and in-place summary stamping now share
  the same SDE-owned read/write root summary, including scalar memref accesses
  and carrier-authoritative `linalg.generic` DPS operands.
- Preserved epoch-owned task EDTs that represent planned runtime work
  distribution. `EdtStructuralOpt` no longer inlines dependency-free task EDTs
  while they are inside an `arts.epoch`, because that turns distributed worker
  dispatch back into serial epoch-body work.
- Made the Core EDT capture contract explicit for allocation-rooted DB views.
  `EdtUtils::traceCapturedDbHandle` is now the shared contract used by EDT
  lowering and the verifier: external `db_ref`/`db_gep`/view chains rooted in a
  `DbAllocOp` or heap memref are packable state, while `DbAcquireOp` roots
  still must be explicit EDT dependencies. This lets Seidel share a coarse
  in-place DB handle without creating one coarse runtime dependency that
  serializes all worker tasks.
- Moved uniform physical-output planning into the SDE-owned structured-analysis
  utility surface. SDE now proves owner-IV indexed external writes, stamps
  `arts.plan.*` on the `su_iterate`, and wraps eligible units in
  `arts_sde.su_distribute <blocked>` before crossing into Core. Core
  `CreateDbs` now materializes only write-backed SDE plans as physical block
  DBs, instead of accepting reader-only plan sources.
- Kept in-place neighbor stencils conservative. SDE does not stamp physical
  block DB plans or distributed owner-compute plans for self-read stencil loops
  unless the transformation proves a safe wavefront/halo plan. Core's
  coarse-in-place shared-state path remains the deliberate local mitigation for
  Gauss-Seidel-style loops.
- Fixed RT CPS relaunch packing for split launch-state schemas. Direct relaunch
  now rebuilds the compact scratch GUID table from the forwarded dependency
  slot, uses the dependency-schema timing count to locate that slot, and packs
  `scratch, data-guid-0, data-guid-1, ...` in the target continuation ABI order.
  This keeps dependency forwarding and paramv reconstruction aligned after
  Core preserves DB-partition handle arrays.
- Added focused regression coverage:
  `lib/arts/dialect/core/test/edt_captures_external_db_ref_view.mlir`.
  The test verifies that an external allocation-rooted `db_ref` captured by a
  dependency-free task is lowered through paramv/rematerialization with
  `depCount(0)`.
- Added focused regression coverage:
  `lib/arts/dialect/rt/test/state_dep_schema_cps_relaunch_param_ordering.mlir`.
  The test verifies the split launch-state relaunch ABI for sibling CPS chains
  that rebuild multiple GUID tables from one compact dependency.

### Sample Sweep

Command:

```bash
dekk carts examples run --all
```

Result:

- 26 examples discovered.
- 26 passed, 0 failed, 0 skipped.
- The previously failing `convolution`, `matrixmul`, `stencil`, and
  `jacobi/for` example builds are now compile-clean and runtime-clean.

Focused compiler evidence:

```bash
dekk carts compile samples/convolution/convolution.cpp -O3 --arts-config samples/arts.cfg -o .carts/outputs/examples/convolution_arts
dekk carts compile samples/matrixmul/matrixmul.c -O3 --arts-config samples/arts.cfg -o .carts/outputs/examples/matrixmul_arts
dekk carts compile samples/stencil/stencil.c -O3 --arts-config samples/arts.cfg -o .carts/outputs/examples/stencil_arts
dekk carts compile samples/jacobi/for/jacobi-for.c -O3 --arts-config samples/arts.cfg -o .carts/outputs/examples/jacobi_for_arts
```

All four focused compiles completed successfully after the lowering fixes.

### Large 64-Thread Benchmark Sweep

Command:

```bash
dekk carts benchmarks run stream kastors-jacobi/jacobi-for kastors-jacobi/poisson-for ml-kernels/activations ml-kernels/batchnorm ml-kernels/layernorm ml-kernels/pooling monte-carlo/ensemble polybench/2mm polybench/3mm polybench/atax polybench/bicg polybench/convolution-2d polybench/convolution-3d polybench/correlation polybench/gemm polybench/jacobi2d polybench/seidel-2d seissol/volume-integral specfem3d/stress specfem3d/velocity sw4lite/rhs4sg-base sw4lite/vel4sg-base --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmarks-large-64
```

Result directory:
`.carts/outputs/benchmarks-large-64/20260512_150448`

Summary:

- 23 runnable benchmark entries discovered from `dekk carts benchmarks list`.
- 21 passed, 2 failed, 0 skipped at `--timeout 120`.
- Geometric mean speedup: `0.10x` kernel basis.
- All 21 passing benchmarks have checksum parity within runner tolerance.
- The 2 failures are ARTS-side timeouts, not compile failures:
  `monte-carlo/ensemble` and `polybench/seidel-2d`.

Timeout follow-up:

```bash
dekk carts benchmarks run monte-carlo/ensemble polybench/seidel-2d --size large --timeout 300 --threads 64 --nodes 1 --trace --arts --no-clean --results-dir .carts/outputs/benchmarks-large-64-rerun
```

Result directory:
`.carts/outputs/benchmarks-large-64-rerun/20260512_152603`

- `monte-carlo/ensemble`: ARTS completed with checksum
  `8.160014376227e+03`; kernel `240.231073s`, e2e `240.231278s`.
- `polybench/seidel-2d`: ARTS completed with checksum
  `3.072480116667e+07`; kernel `271.207509s`, e2e `271.544694s`.

Classification: both 120s failures are correctness-clean but performance-blocked
large/64 local runs. They should stay in the maintained set, but a 120s timeout
is too low for the current ARTS task plan at `large` size and 64 local threads.
The bottleneck is performance/task planning, not a new compiler crash.

Targeted follow-up for `monte-carlo/ensemble` and `polybench/seidel-2d`:

```bash
dekk carts benchmarks run monte-carlo/ensemble polybench/seidel-2d --size small --timeout 120 --threads 16 --nodes 1 --trace --results-dir .carts/outputs/benchmark-targets/after-core-align-small
dekk carts benchmarks run monte-carlo/ensemble polybench/seidel-2d --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmark-targets/after-core-align-large
```

Result directories:

- `.carts/outputs/benchmark-targets/after-core-align-small/20260512_164909`
- `.carts/outputs/benchmark-targets/after-core-align-large/20260512_165024`

Follow-up summary:

- `monte-carlo/ensemble` now completes at `large`, 64 threads, 1 node under the
  120s timeout with checksum `8.160014376227e+03`; ARTS kernel `7.240216s`,
  OpenMP kernel `4.224490s`.
- On `small`, 16 threads, 1 node, `monte-carlo/ensemble` improved from the
  earlier serialized ARTS result (`13-15s`) to ARTS kernel `1.043442s` versus
  OpenMP kernel `0.896755s`, with checksum `5.000238688912e+02`.
- `polybench/seidel-2d` remains checksum-clean on `small` but times out at
  `large`, 64 threads, 1 node under 120s. This is the expected consequence of
  preserving the in-place Gauss-Seidel loop-carried dependences by serializing
  the current plan. A real speedup still requires an SDE wavefront/timestep
  transformation rather than unsafe owner-strip parallelization.

Synchronization/work-distribution follow-up:

```bash
dekk carts benchmarks run monte-carlo/ensemble polybench/seidel-2d --size small --timeout 120 --threads 16 --nodes 1 --trace --results-dir .carts/outputs/benchmark-targets/after-sync-workdist-small
dekk carts benchmarks run monte-carlo/ensemble --size small --timeout 120 --threads 16 --nodes 1 --trace --results-dir .carts/outputs/benchmark-targets/after-sync-workdist-small-rerun
dekk carts benchmarks run monte-carlo/ensemble polybench/seidel-2d --size large --timeout 120 --threads 64 --nodes 1 --trace --results-dir .carts/outputs/benchmark-targets/after-sync-workdist-large
```

- Stable small rerun for `monte-carlo/ensemble`:
  `.carts/outputs/benchmark-targets/after-sync-workdist-small-rerun/20260512_171121`,
  checksum `5.000238688912e+02`, ARTS kernel `0.968261s`, OpenMP kernel
  `0.897146s`.
- Combined small run:
  `.carts/outputs/benchmark-targets/after-sync-workdist-small/20260512_170947`.
  `polybench/seidel-2d` remained checksum-clean with ARTS kernel `0.022111s`
  and OpenMP kernel `0.004480s`; the `monte-carlo/ensemble` ARTS kernel in
  that combined run was a `10.209911s` compute outlier, so the isolated rerun is
  the better small Monte Carlo datapoint.
- Large follow-up:
  `.carts/outputs/benchmark-targets/after-sync-workdist-large/20260512_171141`.
  `monte-carlo/ensemble` remained checksum-clean but ran slower in that
  environment sample (ARTS kernel `23.933877s`, OpenMP kernel `4.518429s`);
  the generated `monte_carlo_ensemble-arts.ll` is byte-identical to the prior
  `after-core-align-large` artifact, so this is not attributed to a compiler IR
  change. `polybench/seidel-2d` still timed out at `large`, with OpenMP kernel
  `4.644897s`.

Scalar replacement and epoch task-preservation follow-up:

```bash
dekk carts benchmarks run monte-carlo/ensemble --size small --timeout 120 --threads 16 --nodes 1 --trace --results-dir .carts/outputs/benchmark-targets/after-scalar-r0-small
dekk carts compile external/carts-benchmarks/polybench/seidel-2d/seidel-2d.c -O3 --all-pipelines --arts-config tests/inputs/arts_8t.cfg -o .carts/outputs/seidel-epoch-task-preserve-pipes
dekk carts benchmarks run polybench/seidel-2d --size small --timeout 120 --threads 16 --nodes 1 --trace --results-dir .carts/outputs/benchmark-targets/after-epoch-task-preserve-small
dekk carts benchmarks run monte-carlo/ensemble --size small --timeout 120 --threads 16 --nodes 1 --trace --results-dir .carts/outputs/benchmark-targets/after-epoch-task-preserve-small-rerun
```

- `monte-carlo/ensemble` after rank-0 RT scalar replacement:
  `.carts/outputs/benchmark-targets/after-scalar-r0-small/20260512_184208`,
  checksum `5.000238688912e+02`, ARTS kernel `1.021232s`, OpenMP kernel
  `0.889278s`. The optimized pipeline has no rank-0 scratch alloca/load/store
  in the outlined EDT; the reduction value is carried through `scf.for`
  iter_args.
- `polybench/seidel-2d` after preserving epoch worker tasks:
  `.carts/outputs/benchmark-targets/after-epoch-task-preserve-small/20260512_185326`,
  checksum `8.358450000000e+04`, ARTS kernel `0.003137s`, OpenMP kernel
  `0.004097s`, speedup `1.31x`.
- `monte-carlo/ensemble` isolated rerun after the Seidel Core fix:
  `.carts/outputs/benchmark-targets/after-epoch-task-preserve-small-rerun/20260512_185503`,
  checksum `5.000238688912e+02`, ARTS kernel `0.979991s`, OpenMP kernel
  `0.890159s`, speedup `0.91x`. A combined run produced one `9.019168s`
  ARTS outlier, but the current Monte Carlo pipeline is byte-identical to the
  earlier fast `monte-scalar-r0-pipes` pipeline, so the outlier is not
  attributed to compiler IR.

Final post-CPS-schema verification:

```bash
dekk carts test
dekk carts test --suite e2e
dekk carts benchmarks run monte-carlo/ensemble polybench/seidel-2d --size small --timeout 120 --threads 16 --nodes 1 --trace --results-dir .carts/outputs/benchmark-targets/final-small-16-post-cps-fix
dekk carts benchmarks run monte-carlo/ensemble --size large --timeout 120 --threads 64 --nodes 1 --trace --no-clean --runs 3 --results-dir .carts/outputs/benchmark-targets/final-monte-large-64-rerun-post-cps-fix
dekk carts benchmarks run polybench/seidel-2d --size large --timeout 120 --threads 64 --nodes 1 --trace --no-clean --runs 3 --results-dir .carts/outputs/benchmark-targets/final-seidel-large-64-rerun-post-cps-fix
```

- `dekk carts test`: 101 passed, 1 expected failure.
- `dekk carts test --suite e2e`: 20 passed, 7 expected failures.
- Final small target run:
  `.carts/outputs/benchmark-targets/final-small-16-post-cps-fix/20260512_200127`.
  `monte-carlo/ensemble`: ARTS kernel `0.963053s`, OpenMP kernel
  `0.900937s`, checksum `5.000238688912e+02`. `polybench/seidel-2d`: ARTS
  kernel `0.003437s`, OpenMP kernel `0.003819s`, checksum
  `8.358450000000e+04`.
- Final large Monte Carlo rerun:
  `.carts/outputs/benchmark-targets/final-monte-large-64-rerun-post-cps-fix/20260512_200546`.
  Three checksum-clean runs: ARTS kernels `7.118741s`, `7.195582s`,
  `7.239744s`; OpenMP kernels `4.272332s`, `4.204038s`, `4.343264s`;
  checksum `8.160014376227e+03`.
- Final large Seidel rerun:
  `.carts/outputs/benchmark-targets/final-seidel-large-64-rerun-post-cps-fix/20260512_200652`.
  Three checksum-clean runs: ARTS kernels `8.486309s`, `8.546592s`,
  `8.474634s`; OpenMP kernels `4.356733s`, `4.289029s`, `4.403990s`;
  checksum `3.072480116667e+07`.
- Pipeline evidence for the large Monte Carlo artifact:
  `.carts/outputs/pipes-monte-post-cps-fix-from-artifact`. SDE emits
  `arts_sde.su_distribute <blocked>` with `arts.plan.physical_block_shape =
  [255]`, and Core/RT materialize `arts.db_alloc <block>` with `sizes[%c64]`
  and `elementSizes[%c255]`.
- Post-manifest sanity: `dekk carts build`, `dekk carts pipeline --json`,
  `dekk carts test`, `dekk carts test --suite e2e`, and `git diff --check`
  completed successfully after recording that SDE may reshape `su_iterate`
  topology as part of distribution planning. The rebuilt pipeline manifest
  reports the live SDE order: `RaiseToTensor`, `RaiseToLinalg`,
  `LoopInterchange`, `Tiling`, `StructuredSummaries`, `ElementwiseFusion`,
  `ScopeSelection`, `ScheduleRefinement`, `ChunkOpt`, `ReductionStrategy`,
  `DistributionPlanning`, then `IterationSpaceDecomposition`.

## Current Classifications

| Benchmark | Date | Correctness | Class | Evidence | Bottleneck |
| --- | --- | --- | --- | --- | --- |
| `polybench/jacobi2d` | 2026-05-12 | Checksum parity on `small`, 16 threads, 1 node: ARTS and OpenMP both `3.196312170126e+01`. | `blocked` | `dekk carts benchmarks run polybench/jacobi2d --size small --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `0.004076s`, OpenMP kernel `0.000194s`, speedup `0.0476x`. Trace counters reported 867 EDT creates/acquires/finishes. | Task grain and CPS continuation overhead dominate the small stencil kernel. SDE/ARTS still emits fine-grained owner-strip phase tasks for a kernel whose OpenMP baseline is sub-millisecond; a fair result needs coarser task/timestep planning or continuation overhead reduction. |
| `polybench/seidel-2d` | 2026-05-12 | Checksum parity on `small`, 16 threads, 1 node: ARTS and OpenMP both `8.358450000000e+04`. Checksum parity on `large`, 64 threads, 1 node: ARTS and OpenMP both `3.072480116667e+07`. | `fast` on small, `blocked` on large | Final small run `.carts/outputs/benchmark-targets/final-small-16-post-cps-fix/20260512_200127`: ARTS kernel `0.003437s`, OpenMP kernel `0.003819s`. Final large rerun `.carts/outputs/benchmark-targets/final-seidel-large-64-rerun-post-cps-fix/20260512_200652`: ARTS kernels `8.486309s`, `8.546592s`, `8.474634s`; OpenMP kernels `4.356733s`, `4.289029s`, `4.403990s`. Pipeline dump `.carts/outputs/seidel-epoch-task-preserve-pipes` shows worker `arts.edt` surviving late cleanup and reaching RT as `arts_rt.edt_create ... depCount(%c0_i32)` with the coarse DB handle packed through paramv instead of recorded as one serializing DB dependency. | The original small-target serialization blocker is fixed in Core: preserve epoch worker EDTs and capture coarse in-place allocation-rooted DB state as paramv, not as a runtime dependency. Large remains slower than OpenMP because the benchmark is an in-place Gauss-Seidel stencil; SDE correctly avoids unsafe physical block partitioning/owner-strip parallelization. A fair large speedup needs an SDE wavefront or split-phase transformation, not late ARTS policy. |
| `polybench/fdtd-2d` | 2026-05-12 | Not runner-clean: the benchmark is present in `external/carts-benchmarks/polybench/fdtd-2d` but hidden by its checked-in `.disabled` marker, so `dekk carts benchmarks list` omits it and a direct runner request reports it as unknown. A direct `dekk carts compile` with the benchmark include flags fails before LLVM lowering. | `blocked` | `CARTS_COMPILE_WORKDIR=/tmp/carts-fdtd2d-small-compile dekk carts compile external/carts-benchmarks/polybench/fdtd-2d/fdtd-2d.c -O3 --arts-config external/carts-benchmarks/configs/local.cfg -o /tmp/fdtd2d_arts_small -- -lm -ldl -Iexternal/carts-benchmarks/polybench/fdtd-2d -Iexternal/carts-benchmarks/polybench/utilities -DSMALL_DATASET` fails with four `arts.for survived past edt-distribution stage` diagnostics at `fdtd-2d.c:64`, `:67`, `:71`, and `:75`. | The kernel has sequential time-stepping around several OpenMP loops. SDE lowers this to nested `arts.for` operations inside a parallel EDT under an enclosing `scf.for`, but `EdtDistribution` only distributes the top-level `arts.for` shape. A fair result needs nested `arts.for`/timestep support, or an explicit staged plan that emits legal per-timestep EDTs with the correct barriers and DB windows. |
| `polybench/convolution-2d` | 2026-05-12 | Checksum parity on `small` and `large`, 16 threads, 1 node. Large checksums: ARTS and OpenMP both `9.597900223886e+03`. | `competitive` | `dekk carts benchmarks run polybench/convolution-2d --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `5.962005s`, OpenMP kernel `4.895270s`, speedup `0.8211x`, with 17 EDT creates/finishes, 1 EDT acquire, and 32 DB creates. Pipeline dump shows SDE-authored stencil metadata preserved through lowering: owner dims `[0]`, physical block shape `[1200, 19200]`, 16 task blocks, and `CreateDbs` materializing 16 physical blocks for each tensor. Small is correctness-clean but overhead-dominated: ARTS kernel `0.000838s`, OpenMP kernel `0.000123s`, speedup `0.1468x`. | No correctness blocker. The large case is within the 1.25x competitive threshold and follows the intended SDE-owned physical DB path. The remaining gap is stencil kernel overhead and OpenMP codegen advantage rather than a late ARTS partition-policy failure. |
| `polybench/convolution-3d` | 2026-05-12 | Checksum parity on `small`, `medium`, and `large`, 16 threads, 1 node. Large checksums: ARTS and OpenMP both `5.101040000000e+05`. | `fast` | `dekk carts benchmarks run polybench/convolution-3d --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `2.183816s`, OpenMP kernel `2.909669s`, speedup `1.3324x`, with 17 EDT creates/finishes, 1 EDT acquire, and 32 DB creates. Pipeline dump shows SDE-authored 3D stencil metadata preserved through lowering: owner dims `[0]`, physical block shape `[32, 512, 512]`, 16 task blocks, and local stencil dependency windows. Small and medium are correctness-clean but overhead-dominated: small speedup `0.1628x`, medium speedup `0.4382x`. A 1-thread small control is competitive at `0.95x`. | No correctness blocker. The large benchmark is performance-credible with the current owner-strip stencil plan. The small and medium cases are too short for the ARTS task/epoch overhead model, so future work should make the benchmark policy size-aware or add a coarser/serial small-case path rather than changing the memory semantics. |
| `polybench/gemm` | 2026-05-12 | Checksum parity on `small`, `medium`, and `large`; also parity on the intended large 64-thread workload. Large 64-thread checksums: ARTS and OpenMP both `2.211832507057e+05`. | `blocked` | `dekk carts benchmarks run polybench/gemm --size large --timeout 120 --threads 64 --nodes 1 --trace`: ARTS kernel `20.369832s`, OpenMP kernel `6.044097s`, speedup `0.2967x`, with 65 EDT creates/finishes, 1 EDT acquire, and 3 DB creates. The 16-thread large run is only marginally fast: ARTS kernel `20.486701s`, OpenMP kernel `24.296057s`, speedup `1.1859x`; small and medium are slower (`0.1818x`, `0.5763x`). Pipeline dump for the 64-thread run shows `CreateDbs` materializing A, B, and C as single coarse DBs (`sizes[%c1]`, `elementSizes[%c4800, %c4800]`), then slicing C only in task-local `db_acquire` with `stencil_block_shape = [75, 4800]`. | GEMM still bypasses the intended SDE physical-layout authority path: after the SDE stages the kernel is plain OpenMP, so `CreateDbs` has no `arts.plan.physical_block_shape` to materialize. The runtime receives 64 write tasks against one coarse C DB, and the counters advance like one active EDT at a time (`TIME_EDT_EXEC` and `TIME_TOTAL` both about `20.48s`). A fair GEMM result needs SDE matmul planning to choose a physical row-blocked C layout at DB creation time, with A/B read policy chosen deliberately, instead of late ARTS-side block slices over a single inout DB. |
| `polybench/2mm` | 2026-05-12 | Checksum parity on `small` and `medium`, 16 threads, 1 node. Medium checksums: ARTS and OpenMP both `1.386033714137e+21`. | `blocked` | `dekk carts benchmarks run polybench/2mm --size medium --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `0.603484s`, OpenMP kernel `0.163873s`, speedup `0.2715x`, with 33 EDT creates/finishes, 1 EDT acquire, and 5 DB creates. Small is also slow: ARTS kernel `0.001717s`, OpenMP kernel `0.000240s`, speedup `0.1398x`. Pipeline dump shows the two matmul phases still start from plain OpenMP after SDE cleanup and `CreateDbs` materializes `tmp`, A, B, C, and D as single coarse DBs before adding block-sliced matmul acquires. | Same matmul DB-layout blocker as GEMM, with an additional phase-chain issue: `2mm` has two dependent matmul epochs (`tmp = A*B`, then `D = tmp*C + beta*D`). The output/intermediate DBs are not physically row-blocked at DB creation time, so runtime dependencies are expressed as late slices of coarse inout DBs. A fair result needs SDE to plan both matmul outputs, preserve the phase ordering, and materialize row-blocked `tmp`/`D` DBs directly in `CreateDbs`. |
| `polybench/3mm` | 2026-05-12 | Checksum parity on `small` and `medium`, 16 threads, 1 node. Medium checksums: ARTS and OpenMP both `1.496860391763e+22`. | `blocked` | `dekk carts benchmarks run polybench/3mm --size medium --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `0.913273s`, OpenMP kernel `0.242538s`, speedup `0.2656x`, with 49 EDT creates/finishes, 1 EDT acquire, and 7 DB creates. Small is also slow: ARTS kernel `0.002029s`, OpenMP kernel `0.000321s`, speedup `0.1582x`. | Same matmul DB-layout blocker as GEMM/`2mm`, repeated across three dependent matrix products (`E = A*B`, `F = C*D`, `G = E*F`). The task count scales with the three matmul epochs, but DB creation still does not physically partition the live-out/intermediate matrices, so the runtime sees late block slices of coarse inout DBs instead of independent row-block DBs. |
| `polybench/atax` | 2026-05-12 | Checksum parity on `small` and `medium`, 16 threads, 1 node. Medium checksums: ARTS and OpenMP both `7.147974114095e+20`. | `blocked` | `dekk carts benchmarks run polybench/atax --size medium --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `0.124907s`, OpenMP kernel `0.005197s`, speedup `0.0416x`, with 33 EDT creates/finishes, 1 EDT acquire, and 3 DB creates. Pipeline dump shows two uniform block epochs, but A, y, and tmp are created as single coarse DBs; tmp/y are only block-sliced at acquire time (`blockSize = 250`). | Matrix-vector phase planning is still late and coarse. `tmp = A*x` and `y = A^T*tmp` are legal as two ordered phases, but the live vector outputs are not physically block-partitioned in `CreateDbs`; the runtime sees inout/out slices of single coarse DBs, and `TIME_EDT_EXEC` tracks `TIME_TOTAL` instead of accumulated parallel work. A fair result needs SDE vector/reduction planning for physical tmp/y row-block DBs and explicit phase ordering. |
| `polybench/bicg` | 2026-05-12 | Checksum parity on `small` and `medium`, 16 threads, 1 node. Medium checksums: ARTS and OpenMP both `2.680322908719e+14`. | `blocked` | `dekk carts benchmarks run polybench/bicg --size medium --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `0.118775s`, OpenMP kernel `0.005172s`, speedup `0.0435x`, with 17 EDT creates/finishes, 1 EDT acquire, and 5 DB creates. Small is also slow: ARTS kernel `0.000531s`, OpenMP kernel `0.000101s`, speedup `0.1902x`. | Same vector-output planning blocker as `atax`. The two products (`q = A*p`, `s = A^T*r`) are checksum-clean, but the output vectors are not authored as physical block DBs before ARTS lowering, leaving the runtime to schedule late slices of coarse DBs. The fix belongs in SDE distribution/reduction planning, not in a late runtime heuristic. |
| `polybench/correlation` | 2026-05-12 | Checksum parity on `small` and `medium`, 16 threads, 1 node. Medium checksums: ARTS and OpenMP both `1.024000000000e+03`. | `blocked` | `dekk carts benchmarks run polybench/correlation --size medium --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `0.147005s`, OpenMP kernel `0.045511s`, speedup `0.3096x`, with 65 EDT creates/finishes, 1 EDT acquire, and 4 DB creates. Pipeline dump shows four block epochs: mean, stddev, normalization, and triangular correlation. All four live arrays are coarse DBs, while the final phase uses `distribution_pattern = triangular` with block-sliced `corr`/data/vector acquires. | The triangular correlation update is correctness-clean, but physical planning is still incomplete. Mean/stddev/data/corr DBs are not materialized in the layout the phases actually need, and the final symmetric write (`corr[i][j]` plus `corr[j][i]`) needs a first-class triangular/symmetric output plan rather than block slices over a single coarse corr DB. |
| `ml-kernels/pooling` | 2026-05-12 | Checksum parity on `small` and `large`, 16 threads, 1 node. Large checksums: ARTS `4.435586245537e+03`, OpenMP `4.435586240768e+03`. | `blocked` | `dekk carts benchmarks run ml-kernels/pooling --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `10.972611s`, OpenMP kernel `7.173012s`, speedup `0.6537x`, with 2881 EDT creates/finishes, 1 EDT acquire, and 4 DB creates. Pipeline dump shows input, max-pool output, average-pool output, and global-average output all created as single coarse DBs (`sizes[%c1]`); block slices are added later at `db_acquire` with `blockSize = 32`. | Pooling has three independent phases and local window reductions, but SDE does not author a physical NCHW/channel-block layout for the outputs before `CreateDbs`. The runtime therefore schedules many block tasks against coarse DBs and repeatedly reads the full input DB. A fair result needs first-class pooling/window-reduction planning with physical output blocks and deliberate input reuse policy. |
| `ml-kernels/batchnorm` | 2026-05-12 | Checksum parity on `small` and `large`, 16 threads, 1 node. Large checksums: ARTS `4.434159001168e+02`, OpenMP `4.437362334230e+02`, within the benchmark tolerance. | `blocked` | `dekk carts benchmarks run ml-kernels/batchnorm --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `13.468606s`, OpenMP kernel `3.913579s`, speedup `0.2906x`, with 3201 EDT creates/finishes, 782 EDT acquires, and 4 DB creates. Pipeline dump shows input/output/mean/variance as coarse DBs; the five batchnorm phases are later distributed with uniform block acquires (`blockSize = 32`) and multiple coarse reads of output/mean/variance. | Batch normalization needs channel-wise reduction planning across batch and spatial dimensions. Today the reductions for mean and variance, plus normalize/scale/bias phases, are emitted as separate uniform epochs over coarse DBs. A fair result needs SDE to plan channel-blocked mean/variance/output DBs, preserve phase locality, and avoid repeated coarse full-tensor dependencies. |
| `ml-kernels/layernorm` | 2026-05-12 | Checksum parity on `small` and `large`, 16 threads, 1 node. Large checksums: ARTS `7.094589257021e+03`, OpenMP `7.094585575579e+03`. | `blocked` | `dekk carts benchmarks run ml-kernels/layernorm --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `39.150835s`, OpenMP kernel `7.361034s`, speedup `0.1880x`, with 2341 EDT creates/finishes, 391 EDT acquires, and 3 DB creates. Pipeline dump shows `x`, `gamma`, and `beta` as coarse DBs, plus a CPS chain around the repeated outer loop and late block acquires over `x` (`blockSize = 4096`). | Layer normalization has per-row mean/variance reductions followed by a row-wise affine update. SDE currently treats the outer row loop as uniform tasking without a row-local reduction plan, so hidden-dimension work remains serial inside each task and each repetition pays CPS/epoch overhead over coarse DBs. A fair result needs row-block physical layout plus reduction strategy selection for mean/variance before ARTS lowering. |
| `ml-kernels/activations` | 2026-05-12 | Checksum parity on `small` and `large`, 16 threads, 1 node. Large checksums: ARTS and OpenMP both `8.547615348671e+07`. | `blocked` | `dekk carts benchmarks run ml-kernels/activations --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `14.985731s`, OpenMP kernel `1.975894s`, speedup `0.1319x`, with 5601 EDT creates/finishes, 1 EDT acquire, and 7 DB creates. Pipeline dump shows seven large activation output arrays as coarse DBs (`sizes[%c1]`, `elementSizes[%c16777216]`); each elementwise activation is later distributed into block tasks with `blockSize = 1048576`. | Elementwise activation planning is still late and phase-split. ReLU/leaky/ReLU6/GELU/fast GELU/sigmoid/tanh are independent maps over the same input, but SDE does not fuse compatible maps or materialize blocked output DBs in `CreateDbs`; softmax remains a small serial side path. A fair result needs elementwise map fusion/tiling and physical vector-output DB planning before ARTS lowering. |
| `stream` | 2026-05-12 | Checksum parity on `small` and `large`, 16 threads, 1 node. Large checksums: ARTS and OpenMP both `4.625216674438e+18`. | `blocked` | `dekk carts benchmarks run stream --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `72.046809s`, OpenMP kernel `5.774106s`, speedup `0.0801x`, with 657 EDT creates/finishes, 1 EDT acquire, and 4 DB creates. Pipeline dump shows vector arrays created as coarse DBs (`sizes[%c1]`, `elementSizes[%c700000000]`), with block slicing introduced later at acquire time. | STREAM is correctness-clean but not performance-credible. SDE does not plan a physical vector DB layout or fuse the copy/scale/add/triad/check phases, so the runtime repeatedly schedules memory-bandwidth epochs over coarse DBs and pays reduction/check overhead instead of exposing the intended streaming access pattern directly in `CreateDbs`. |
| `kastors-jacobi/jacobi-for` | 2026-05-12 | Checksum parity on `small`, 16 threads, 1 node: ARTS and OpenMP both `3.934498277405e-01`. The 1-thread control also compiles and has checksum parity. | `blocked` | `dekk carts benchmarks run kastors-jacobi/jacobi-for --size small --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `0.002837s`, OpenMP kernel `0.000177s`, speedup `0.06x` (result `20260512_090600`). `--threads 1`: ARTS kernel `0.006340s`, OpenMP kernel `0.001154s`, speedup `0.18x` (result `20260512_091450`). | Correctness is restored for both the parallel and single-worker controls. The remaining blocker is not VLA/row-array semantics but task-grain and CPS overhead on a sub-millisecond stencil baseline. The compiler fixes were: preserve the Jacobi sidecar dependency ordering, lower residual outlined `arts.db_ref` operations before their acquire source is rewritten, and use dynamic Polygeist loads/stores for multi-dynamic nested memrefs. |
| `kastors-jacobi/poisson-for` | 2026-05-12 | Checksum parity on `small`, 16 threads, 1 node: ARTS and OpenMP both `1.038656999420e-01`. The 1-thread control also compiles and has checksum parity. | `blocked` | `dekk carts benchmarks run kastors-jacobi/poisson-for --size small --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `0.002689s`, OpenMP kernel `0.000163s`, speedup `0.06x` (result `20260512_090550`). `--threads 1`: ARTS kernel `0.006126s`, OpenMP kernel `0.001152s`, speedup `0.19x` (result `20260512_091436`). | Correctness is restored by preserving the read-only forcing term through every CPS timestep: CPS relaunch now reconstructs missing structured GUID-table dependencies from the scratch dependency, and residual outlined `arts.db_ref` lowering keeps the 1-thread control compilable. The benchmark remains performance-blocked by CPS relaunch/task overhead on the small problem size; a fair result needs coarser timestep/task planning or continuation overhead reduction. |
| `kastors-jacobi/jacobi-task-dep` | 2026-05-12 | Not runner-clean: hidden by a checked-in `.disabled` marker and absent from `dekk carts benchmarks list`, so there is no checksum result. | `blocked` | The `.disabled` marker records an LLVM CodeExtractor crash for `omp.task depend` with loop-varying values inside `omp.parallel`: `OpenMPIRBuilder::finalize()` to `CodeExtractor::emitCallAndSwitchStatement` SIGSEGV. | Task-depend OpenMP frontend lowering is blocked before the CARTS SDE/ARTS performance pipeline can make a fair classification. Re-enable only after the frontend can produce stable MLIR for loop-varying task dependencies, or document removal from the maintained set. |
| `kastors-jacobi/poisson-task` | 2026-05-12 | Not runner-clean: hidden by a checked-in `.disabled` marker and absent from `dekk carts benchmarks list`, so there is no checksum result. | `blocked` | The `.disabled` marker records the same LLVM CodeExtractor crash class as `jacobi-task-dep` for `omp.task depend` with loop-varying values inside `omp.parallel`. | Same task-depend frontend blocker as `jacobi-task-dep`. This benchmark cannot be performance-triaged until the frontend produces stable IR for OpenMP tasks with depend clauses, or until the benchmark is removed with a documented unsupported-feature reason. |
| `monte-carlo/ensemble` | 2026-05-12 | Checksum parity on `small`, 16 threads, 1 node: ARTS and OpenMP both `5.000238688912e+02`. Checksum parity on `large`, 64 threads, 1 node: ARTS and OpenMP both `8.160014376227e+03`. | `competitive` on small, `blocked` on large | Final small run `.carts/outputs/benchmark-targets/final-small-16-post-cps-fix/20260512_200127`: ARTS kernel `0.963053s`, OpenMP kernel `0.900937s`. Final large rerun `.carts/outputs/benchmark-targets/final-monte-large-64-rerun-post-cps-fix/20260512_200546`: ARTS kernels `7.118741s`, `7.195582s`, `7.239744s`; OpenMP kernels `4.272332s`, `4.204038s`, `4.343264s`. Pipeline dump `.carts/outputs/pipes-monte-post-cps-fix-from-artifact` shows SDE `su_distribute <blocked>` with `physical_block_shape = [255]` and Core/RT `db_alloc <block>` with `sizes[%c64]`, `elementSizes[%c255]`. | The original timeout/serialization blocker is fixed: SDE authors the uniform physical output DB layout, `CreateDbs` materializes block DBs, Core preserves DB-aligned worker chunks, and RT scalar replacement keeps the rank-0 reduction in SSA loop carries instead of a scratch memref. The benchmark remains slower than the large 64-thread OpenMP baseline because local ARTS task/runtime overhead and scalar `sin` codegen still dominate, not because coarse DBs serialize the tasks. |
| `seissol/volume-integral` | 2026-05-12 | Checksum parity on `small` and `large`, 16 threads, 1 node. Large checksums: ARTS `1.494466070086e+00`, OpenMP `1.494466044009e+00`, within tolerance. | `blocked` | `dekk carts benchmarks run seissol/volume-integral --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `13.210589s`, OpenMP kernel `0.670275s`, speedup `0.0507x`, with 181 EDT creates/finishes, 31 EDT acquires, and 4 DB creates. Pipeline dump shows `dofs`, `gradMatrix`, `fluxMatrix`, and `fluxOut` created as coarse DBs, then output block acquires added later. | The volume integral is a batched DG tensor contraction, but SDE does not author a batched-matmul/tensor-layout plan before `CreateDbs`. Full read tensors remain coarse, the output is sliced late, and CPS/epoch overhead dominates. A fair result needs SDE-level tensor contraction planning with physical output blocks and deliberate read reuse. |
| `specfem3d/stress` | 2026-05-12 | Checksum parity on `small` and `large`, 16 threads, 1 node. Large checksums: ARTS and OpenMP both `3.523365000000e-03`. | `blocked` | `dekk carts benchmarks run specfem3d/stress --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `82.284152s`, OpenMP kernel `2.837861s`, speedup `0.0345x`, with 161 EDT creates/finishes, 1 EDT acquire, and 11 DB creates. Pipeline dump shows late 3D stencil distribution metadata (`stencil_block_shape = [288,288,24]`, owner dim `[2]`) but coarse physical DB creation. | SDE recognizes a 3D stencil schedule late enough to drive task metadata, but `CreateDbs` still materializes component arrays as coarse DBs. The runtime then reads/writes large tensor components through coarse dependencies instead of physical z-block DBs. A fair result needs 3D stencil/component layout planning to reach DB creation. |
| `specfem3d/velocity` | 2026-05-12 | Checksum parity on `small` and `large`, 16 threads, 1 node. Large checksums: ARTS and OpenMP both `-3.719727196731e-06`. | `blocked` | `dekk carts benchmarks run specfem3d/velocity --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `68.838514s`, OpenMP kernel `2.369293s`, speedup `0.0344x`, with 161 EDT creates/finishes, 1 EDT acquire, and 10 DB creates. | Same 3D stencil/component DB-layout blocker as `specfem3d/stress`. The kernel is checksum-clean, but physical partitioning is not authored at SDE level, so the ARTS pipeline schedules late block tasks over coarse displacement/velocity/stress component DBs. |
| `sw4lite/rhs4sg-base` | 2026-05-12 | ARTS now compiles and has checksum parity on `small`, 16 threads, 1 node: ARTS and OpenMP both `3.778052848577e+01`. The `large` run builds but ARTS times out before producing a checksum; OpenMP large checksum is `1.527648351669e+03`. | `blocked` | `dekk carts benchmarks run sw4lite/rhs4sg-base --size small --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `0.000225s`, OpenMP kernel `0.000044s`, speedup `0.1956x`, with 7 EDT creates/finishes, 1 EDT acquire, and 4 DB creates. `dekk carts benchmarks run sw4lite/rhs4sg-base --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS starts in `0.802608s` but times out after `120.101647s`; OpenMP kernel `5.193617s`. Pipeline dump after the fix is complete and shows coarse DB creation for the 3D component tensors, with stencil distribution metadata added later. | The immediate loop-interchange crash is removed by skipping result-bearing `scf.for` rewrites. The remaining blocker is the same physical-layout/runtime issue as the other 3D stencil kernels: SDE leaves the result-bearing component stencil as loop-carried tensor state, `CreateDbs` materializes coarse 3D/4D DBs (`sizes[%c1]`), and late stencil block acquires over coarse DBs lead to an unscalable large run that times out before verification. |
| `sw4lite/vel4sg-base` | 2026-05-12 | Checksum parity on `small` and `large`, 16 threads, 1 node. Large checksums: ARTS and OpenMP both `1.166969475506e-04`. | `blocked` | `dekk carts benchmarks run sw4lite/vel4sg-base --size large --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `89.166873s`, OpenMP kernel `3.441350s`, speedup `0.0386x`, with 161 EDT creates/finishes, 1 EDT acquire, and 10 DB creates. Pipeline dump shows late 3D stencil metadata (`stencil_block_shape = [320,320,28]`, owner dim `[2]`) but coarse physical DB creation. | Same physical-layout blocker as the Specfem 3D stencil kernels. The velocity stencil is checksum-clean, but SDE does not push the component/z-block layout into `CreateDbs`, so ARTS tasking depends on late block acquires over coarse stress, velocity, and density DBs. |
| `graph500/graph-gen` | 2026-05-12 | Not runner-clean: hidden by a checked-in `.disabled` marker and absent from `dekk carts benchmarks list`, so there is no checksum result. | `blocked` | `dekk carts benchmarks list` reports no Graph500 runnable entry; `external/carts-benchmarks/graph500/graph-gen/.disabled` is empty, so the disable reason is not documented. | The maintained-scope status is unresolved. This needs either a supported Graph500 workload with checksum/performance evidence or a documented removal reason such as unsupported graph runtime behavior or intentionally out-of-scope algorithmic pattern. |
| `lulesh` | 2026-05-12 | Not runner-clean: hidden by a checked-in `.disabled` marker and absent from `dekk carts benchmarks list`, so there is no checksum result. | `blocked` | `dekk carts benchmarks list` reports no LULESH runnable entry; `external/carts-benchmarks/lulesh/.disabled` is empty, so the disable reason is not documented. | The benchmark cannot satisfy the maintained-suite acceptance criteria without a documented blocker. Re-enable and triage it with trace evidence, or record why LULESH is unsupported or out of scope. |
| `llama2-transformer` | 2026-05-12 | Not runner-clean: hidden by a checked-in `.disabled` marker and absent from `dekk carts benchmarks list`, so there is no checksum result. | `blocked` | `dekk carts benchmarks list` reports no Llama2 runnable entry; `external/carts-benchmarks/llama2-transformer/.disabled` is empty, so the disable reason is not documented. | The benchmark cannot be classified as maintained and done until it has a runnable checksum target. It needs re-enable triage with trace evidence or a documented unsupported dependency/language/model-size reason for removal. |

## Done Definition

A benchmark is done when it has:

- checksum parity,
- a recorded performance class,
- a documented compiler bottleneck if not `fast`,
- stable lit or pipeline coverage for the optimization that made it work,
- no late ARTS policy decision that should belong to SDE.
