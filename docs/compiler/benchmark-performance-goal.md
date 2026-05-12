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

## Current Classifications

| Benchmark | Date | Correctness | Class | Evidence | Bottleneck |
| --- | --- | --- | --- | --- | --- |
| `polybench/jacobi2d` | 2026-05-12 | Checksum parity on `small`, 16 threads, 1 node: ARTS and OpenMP both `3.196312170126e+01`. | `blocked` | `dekk carts benchmarks run polybench/jacobi2d --size small --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `0.004076s`, OpenMP kernel `0.000194s`, speedup `0.0476x`. Trace counters reported 867 EDT creates/acquires/finishes. | Task grain and CPS continuation overhead dominate the small stencil kernel. SDE/ARTS still emits fine-grained owner-strip phase tasks for a kernel whose OpenMP baseline is sub-millisecond; a fair result needs coarser task/timestep planning or continuation overhead reduction. |
| `polybench/seidel-2d` | 2026-05-12 | Checksum parity on `small`, 16 threads, 1 node: ARTS and OpenMP both `8.358450000000e+04`. | `blocked` | `dekk carts benchmarks run polybench/seidel-2d --size small --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `0.021959s`, OpenMP kernel `0.001475s`, speedup `0.0672x`, with 21 EDT creates/acquires/finishes and 16 DB creates. Pipeline dump shows SDE now detects the in-place neighbor read and stamps `workers = #arts.workers<1>` on the stencil `arts.for` and parent EDT. | Correctness is restored by serializing the in-place Gauss-Seidel update instead of emitting unsafe owner-strip stencil tasks. The benchmark remains performance-blocked until SDE has a first-class wavefront/timestep plan that preserves the row/column loop-carried dependencies while exposing legal parallel frontiers. |
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
| `monte-carlo/ensemble` | 2026-05-12 | Checksum parity on `small`, 16 threads, 1 node: ARTS and OpenMP both `5.000238688912e+02`. | `blocked` | `dekk carts benchmarks run monte-carlo/ensemble --size small --timeout 120 --threads 16 --nodes 1 --trace`: ARTS kernel `14.833847s`, OpenMP kernel `0.898652s`, speedup `0.0606x`, with 17 EDT creates/finishes and 1 DB create. Pipeline dump shows one coarse output DB for 1000 samples and dynamic scheduling around per-sample allocation/state work. | The benchmark is correctness-clean but the current single-node task plan is the wrong grain for Monte Carlo. SDE does not model dynamic sample batching, per-sample local allocation/state reuse, or the distributed-memory scaling intent, so each EDT serializes many samples with allocation overhead while OpenMP wins locally. A fair result needs explicit Monte Carlo task-grain and state-placement planning. |
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
