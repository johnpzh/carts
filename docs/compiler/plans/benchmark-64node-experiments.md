# 64-Thread And 64-Node Benchmark Plan

## Objective

Measure CARTS in two regimes without weakening the SDE/CODIR/ARTS/ARTS-RT
contracts:

- 64-thread single-node performance against OpenMP at `large` problem sizes.
- 64-node multinode behavior at `extralarge` and manually selected weak-scaling
  sizes, but only after the generated ARTS shape proves real distributed work.

The plan must not turn launch success into a scaling claim. A 64-node result is
publishable scaling evidence only when the benchmark lowers to internode EDTs,
non-coarse DB ownership, and runtime route selection.

## Current State

The M6 local large/64 gate is already closed for the maintained 23-entry suite:

```bash
dekk carts benchmarks run --size large --timeout 180 --threads 64 --nodes 1 \
  --runs 3 --trace \
  --results-dir .carts/outputs/benchmarks-m6-final-current-20260516
```

Evidence:

- `.carts/outputs/benchmarks-m6-final-current-20260516/20260516_054254`
- 23 configured entries, 69 executions, 69 passed, 0 checksum failures.
- STREAM was re-run separately at 7 runs in
  `.carts/outputs/benchmarks-m6-stream-final-rerun-20260516/20260516_055824`.
- Final M6 classification: 21 fast, 2 competitive, 0 blocked.

The post-M6 Slurm investigation proves launch and checksum correctness on real
nodes, not distributed scaling. Known passing smoke evidence exists for 2-node
small `gemm`, `2mm`, `3mm`, `convolution-3d`, `specfem3d/stress`, and
`sw4lite/rhs4sg-base`, plus a medium `convolution-3d` 1/2/4-node sweep. The
same investigation found that large `gemm` and large `convolution-3d` still
lower to coarse single-block roots (`sizes[%c1]`) and
`arts.edt <task> <intranode>`, with no `distributed_db_init` and no route
selection through `arts_get_total_nodes`.

Until CODIR-to-ARTS materializes planned owner slices as internode EDTs backed
by block/distributed DB ownership, multinode benchmark runs are smoke tests.

## Measurement Contract

Every 64-node candidate must pass these gates before timing is used as scaling
evidence:

1. SDE carries source-level owner/window/distribution intent without naming
   ARTS topology.
2. CODIR codelets are isolated and expose complete deps, params, and token-local
   views.
3. ARTS materializes the planned DB/EDT/dependency graph directly. The ARTS IR
   must not contain a coarse single-block user DB root feeding an `internode`
   EDT.
4. The ARTS stage contains `internode` EDTs for distributed work, explicit route
   values, and DB ownership/distribution metadata.
5. Generated LLVM or runtime traces show runtime topology use, including route
   selection through the total node count where the benchmark requires it.
6. Slurm execution passes checksum against the stored OpenMP reference.
7. Counter/profile evidence agrees with the intended task count, DB count,
   acquire count, remote-byte movement, and owner-update behavior.

If any item fails, record the run as launch/correctness or compiler-shape
triage, not speedup.

## Runner Semantics To Respect

`dekk carts benchmarks run` supports the needed surface:

- `--threads`: ARTS thread count locally; Slurm `--cpus-per-task` in batch mode.
- `--nodes`: single value, comma list, colon range, and Slurm-style hyphen range
  in batch mode.
- `--slurm`, `--partition`, `--time-limit`, `--exclude-nodes`, `--max-jobs`.
- `--experiment` and repeatable `--step`.
- `--profile`, `--perf`, `--perf-interval`, `--compile-args`, and `--cflags`.

Important caveat: in Slurm mode, ARTS also sees `SLURM_CPUS_PER_TASK`. On
multinode runs, the runtime defaults missing or zero `sender_threads` and
`receiver_threads` to one each, then subtracts sender/receiver threads from the
Slurm CPU total. Therefore `--threads 64` is a 64-CPU-per-rank allocation with
network threads included, not always 64 worker threads. Use that convention for
continuity with the current experiment JSONs, but report it honestly. Before
final publication, either:

- keep the published claim as "64 CPUs per ARTS rank, including network
  threads"; or
- extend the runner so Slurm CPU allocation, ARTS worker threads, and OpenMP
  reference threads are separate fields.

## Network Configuration

Single-node timing should use no network threads:

```ini
launcher=local
node_count=1
worker_threads=64
sender_threads=0
receiver_threads=0
port_count=1
pin=1
route_table_size=16
```

Real multinode timing should use Slurm and explicit network threads:

```ini
launcher=slurm
protocol=tcp
sender_threads=1
receiver_threads=1
port_count=1
default_ports=34739
pin=1
worker_init_deque_size=2048
route_table_size=16
```

`route_table_size` is a log2 setting in ARTS. The default `16` means 65536 route
table entries, which is already above a 64-node experiment. Do not set it to
`128` or `256` as if it were an entry count.

Use `net_interface=<interface>` only after a two-node smoke proves that the
interface exists on every selected node and ARTS remaps host addresses to that
subnet. Avoid known degraded network partitions for production timing. If a
node cannot see the workspace, exclude it with `--exclude-nodes` or force a
known-good allocation through the scheduler wrapper used for that campaign.

The template config for Slurm runs is
[`samples/arts_slurm_multinode.cfg`](../../../samples/arts_slurm_multinode.cfg).

## Problem Sizes

Single-node:

- Use `large` for the full 23-entry suite.
- Run `threads=1,2,4,8,16,32,64` for scaling shape.
- Run the final 64-thread point with at least 3 runs; use 5 or 7 runs for noisy
  entries.

Multinode strong scaling:

- Use fixed `extralarge` size across `1,2,4,8,16,32,64` nodes.
- Include a 1-node Slurm anchor for the same binary/config family when possible,
  even though the local large/64 suite remains the main OpenMP comparison.
- Treat flattening at high node counts as a data point only after counters show
  enough task and DB parallelism to occupy the machine.

Multinode weak scaling:

- The runner rejects `--weak-scaling` with `--slurm`, so weak scaling must be
  encoded with explicit `--cflags` or a new runner extension.
- Keep time/repetition counts fixed unless the benchmark is too short to time.
- Scale dimensions by work complexity:
  - 2D spatial work: dimension multiplier `sqrt(target/base)`.
  - 3D/cubic work: dimension multiplier `cbrt(target/base)`.
  - Linear work: element-count multiplier `target/base`.
- Start from the existing `extralarge` size as a 32-node-era baseline, then
  grow only after memory and timeout checks pass.

Known useful `extralarge` anchors:

| Benchmark | Current `extralarge` shape |
|---|---|
| `polybench/gemm` | `NI=NJ=NK=7680` |
| `polybench/2mm` | `NI=NJ=NK=NL=5760` |
| `polybench/3mm` | `NI=NJ=5760`, `NK=NL=NM=4096` |
| `polybench/convolution-3d` | `NI=1920`, `NJ=512`, `NK=512`, `NREPS=180` |
| `specfem3d/stress` | `NX=288`, `NY=288`, `NZ=1920`, `NREPS=10` |
| `specfem3d/velocity` | `NX=288`, `NY=288`, `NZ=1920`, `NREPS=10` |
| `sw4lite/rhs4sg-base` | `NX=320`, `NY=320`, `NZ=2304`, `NREPS=10` |
| `sw4lite/vel4sg-base` | `NX=320`, `NY=320`, `NZ=1920`, `NREPS=10` |
| `seissol/volume-integral` | `N_ELEMENTS=9216000`, `NREPS=10` |
| `stream` | `STREAM_ARRAY_SIZE=2800000000`, `NTIMES=10` |
| `monte-carlo/ensemble` | `NUM_SAMPLES=61440`, `STATE_DIM=1024` |
| `ml-kernels/layernorm` | `BATCH=245760`, `HIDDEN=8192`, `NREPS=130` |

## Benchmark Set

Single-node full suite:

- Run all 23 maintained benchmarks at `large`.

64-node pilot set:

- `polybench/gemm`
- `polybench/2mm`
- `polybench/3mm`
- `polybench/convolution-3d`
- `specfem3d/stress`
- `specfem3d/velocity`
- `sw4lite/rhs4sg-base`
- `sw4lite/vel4sg-base`
- `seissol/volume-integral`
- `ml-kernels/layernorm`
- `monte-carlo/ensemble`
- `stream`

Expansion set after the pilot is credible:

- Remaining PolyBench 2D/vector kernels: `atax`, `bicg`, `correlation`,
  `convolution-2d`, `jacobi2d`, `seidel-2d`.
- Remaining ML kernels: `activations`, `batchnorm`, `pooling`.
- KaStORS stencil controls: `jacobi-for`, `poisson-for`.

Host OpenMP fallback benchmarks may remain useful single-node controls, but do
not use them as evidence for distributed ARTS task scaling unless the relevant
path is tokenized and lowers through ARTS.

## Experiment Ladder

### Tier 0 - Local Reconfirmation

Use this after compiler changes that may affect benchmark shape:

```bash
dekk carts benchmarks run --size large --timeout 180 --threads 64 --nodes 1 \
  --runs 3 --trace \
  --results-dir .carts/outputs/benchmarks-m7-local64-<YYYYMMDD>
```

For the full single-node scaling curve:

```bash
dekk carts benchmarks run --experiment all-benchmarks-full-large-extralarge \
  --results-dir .carts/outputs/benchmarks-m7-full-<YYYYMMDD>
```

Only consume the single-node steps from that experiment until the multinode
shape gate below passes.

### Tier 1 - Slurm Launch Preflight

Check scheduler visibility and workspace access before any large campaign:

```bash
sinfo -h -o '%P|%D|%t|%N'
squeue -h -u "$USER" -o '%i|%T|%D|%R|%j'
```

Then run a tiny Slurm smoke on known-good nodes or an explicitly selected
partition:

```bash
dekk carts benchmarks run polybench/gemm --slurm --partition <partition> \
  --arts-config samples/arts_slurm_multinode.cfg \
  --size small --threads 64 --nodes 2 --runs 1 --timeout 180 \
  --time-limit 00:05:00 --compile-args "--distributed-db" --max-jobs 1 \
  --results-dir .carts/outputs/benchmarks-m7-slurm-smoke-<YYYYMMDD>
```

This proves launch and checksum only.

### Tier 2 - Compiler Shape Gate

For each candidate benchmark, dump stages before timing node scaling. The dump
mechanism can be `dekk carts compile --all-pipelines --diagnose` or the
benchmark triage scripts, but the artifact must live under
`.carts/sessions/<YYYYMMDD-HHMMSS>-64node-shape/`.

Required checks:

```bash
rg -n "arts\\.edt .*<internode>|<internode>" .carts/sessions/.../
rg -n "arts\\.db_alloc" .carts/sessions/.../
rg -n "sizes\\[%c1\\]" .carts/sessions/.../
rg -n "distributed_db_init|arts_get_total_nodes|runtime_total_nodes" \
  .carts/sessions/.../
```

Passing shape for distributed timing:

- `internode` EDTs are present for distributed work.
- User DB roots are block/distributed, not one coarse aggregate root.
- Route expressions use the planned owner distribution and total-node query.
- `verify-arts-objects-only` does not reject a coarse DB feeding internode work.

Failing shape means the next task is CODIR-to-ARTS materialization, not a bigger
benchmark run.

### Tier 3 - 64-Node Pilot

Run a narrow pilot first, one job active at a time:

```bash
dekk carts benchmarks run \
  polybench/gemm polybench/convolution-3d specfem3d/stress \
  sw4lite/rhs4sg-base seissol/volume-integral ml-kernels/layernorm \
  --slurm --partition <partition> \
  --arts-config samples/arts_slurm_multinode.cfg \
  --size extralarge --threads 64 --nodes 1,2,4,8,16,32,64 \
  --runs 3 --timeout 900 --time-limit 00:20:00 --max-jobs 1 \
  --compile-args "--distributed-db" --trace \
  --results-dir .carts/outputs/benchmarks-m7-64node-pilot-<YYYYMMDD>
```

If queue policy allows it after the pilot, increase `--max-jobs` cautiously.
Do not submit the full 23-entry, 64-node, multi-run sweep without a cap.

### Tier 4 - Diagnostics

Use production timing with counters off or minimal timing counters. Use heavy
profiles only on selected node counts:

```bash
dekk carts benchmarks run polybench/gemm polybench/convolution-3d \
  --slurm --partition <partition> \
  --arts-config samples/arts_slurm_multinode.cfg \
  --size extralarge --threads 64 --nodes 8,32,64 \
  --runs 1 --timeout 900 --time-limit 00:20:00 --max-jobs 1 \
  --compile-args "--distributed-db" --profile profile-overhead.cfg --perf \
  --perf-interval 1.0 --trace \
  --results-dir .carts/outputs/benchmarks-m7-64node-overhead-<YYYYMMDD>
```

Required diagnostic fields:

- kernel and end-to-end time;
- checksum/reference source;
- ARTS task count, DB count, acquire count, epoch shape;
- `BYTES_REMOTE_SENT` and `BYTES_REMOTE_RECEIVED`;
- owner updates saved/performed;
- per-node perf summaries when enabled;
- generated ARTS and LLVM stage paths.

## Exit Criteria

The 64-thread single-node measurement is current when:

- `dekk carts test` passes.
- `dekk carts benchmarks run --size large --threads 64 --nodes 1 --runs 3`
  passes checksum-clean for all maintained entries.
- `benchmark-performance-goal.md` points to the latest result directory.

The 64-node measurement is credible when:

- the compiler shape gate passes for every reported benchmark;
- Slurm runs pass checksum-clean through 64 nodes;
- timings have at least 3 runs for production claims;
- diagnostics explain remote bytes, owner updates, DB count, EDT count, and
  launch/runtime overhead;
- the report states whether `--threads 64` means 64 total CPUs per rank or
  exactly 64 ARTS workers per rank.

Until then, report multinode results as launch/correctness smoke or
distributed-materialization triage.
