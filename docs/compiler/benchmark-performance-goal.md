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

- Every benchmark in the maintained suite compiles with `dekk carts compile`.
- Every benchmark run reports checksum parity against its OpenMP baseline.
- Every benchmark has an explicit performance classification:
  - `fast`: ARTS kernel time is faster than OpenMP.
  - `competitive`: ARTS kernel time is within 1.25x of OpenMP.
  - `blocked`: a named compiler/runtime limitation prevents a fair result.
- No optimization changes the memory model or program semantics.
- DB partitioning decisions are made at SDE level when tensor structure is known.
- `CreateDbs` must create the chosen physical DB layout directly.
- ARTS passes may refine runtime structure, but must not invent tensor
  partition policy late.

## Benchmark Scope

Initial maintained set:

- PolyBench: `2mm`, `3mm`, `atax`, `bicg`, `correlation`, `convolution-2d`,
  `convolution-3d`, `fdtd-2d`, `gemm`, `jacobi2d`, `seidel-2d`.
- ML kernels: `activations`, `batchnorm`, `layernorm`, `pooling`.
- Task/runtime suites: `kastors-jacobi`, `stream`, `lulesh`, `graph500`,
  `seissol`, `specfem3d`, `monte-carlo`, `llama2-transformer`.

Benchmarks can be moved out of the maintained set only with a documented reason:
unsupported language feature, unsupported runtime dependency, or intentionally
out-of-scope algorithmic pattern.

## Optimization Tracks

### SDE Tracks

- Tensor partition planning: derive owner dims, block shape, halo, and iteration
  topology before `ConvertSdeToArts`.
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
   `specfem3d`, `monte-carlo`, `llama2-transformer`.

## Done Definition

A benchmark is done when it has:

- checksum parity,
- a recorded performance class,
- a documented compiler bottleneck if not `fast`,
- stable lit or pipeline coverage for the optimization that made it work,
- no late ARTS policy decision that should belong to SDE.

