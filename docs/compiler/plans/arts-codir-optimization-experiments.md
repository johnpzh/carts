# ARTS/CODIR Optimization And Scaling Experiments

## Purpose

This plan turns the current single-node and multinode benchmark work into an
experiment matrix that demonstrates the function of each CARTS dialect:

- SDE proves source-level legality and emits runtime-neutral worker intent.
- CODIR performs codelet-level analysis and optimization on isolated codelet
  graphs without knowing machine topology.
- ARTS binds generic codelet plans to concrete runtime topology, DB ownership,
  EDT placement, dependency slots, and epochs.
- ARTS-RT removes ABI, packing, GUID, pointer, and LLVM-facing runtime overhead
  after the ARTS object graph is correct.

The goal is not just to pass benchmarks. Each experiment must explain which
compiler layer made the performance decision and which runtime or machine
factor limited scaling.

## Current Evidence

Single-node large/64 is performance-credible:

- Latest documented M6 large/64 sweep:
  `.carts/outputs/benchmarks-m6-final-current-20260516/20260516_054254`.
- Audited result: 23 maintained benchmark entries passed; final audited
  geomean is `1.73x` after the focused STREAM rerun.
- Matrix kernels show the best current upside:
  `gemm` `14.16x`, `2mm` `7.32x`, `3mm` `7.64x`,
  `convolution-3d` `3.91x`.
- Small-workload controls show expected overhead sensitivity:
  `.carts/outputs/benchmarks/full-small-20260514-postcommit/20260514_072458`
  has all 23 passing but only `0.50x` geomean, because many kernels are too
  small to amortize EDT, epoch, and runtime startup overhead.

Multinode launch is correct but not yet performance evidence:

- Slurm workspace-visible nodes:
  `c09u25,c09u31` for 2-node and `c09u25,c09u31,c20u25,c20u31`
  for 4-node runs.
- Passing checksum smokes exist for 2-node small
  `polybench/2mm`, `polybench/3mm`, `polybench/convolution-3d`,
  `specfem3d/stress`, and `sw4lite/rhs4sg-base`.
- Medium `convolution-3d` 1/2/4-node smokes passed, but kernel time worsened:
  `0.001665s`, `0.005952s`, `0.004946s`. This is launch/correctness evidence,
  not distributed-work scaling.
- The compiler gap is known: large benchmark IR had planned SDE/CODIR metadata
  but coarse DB roots. The ARTS launch-policy boundary now emits internode EDTs
  from generic codelet worker plans, but DB materialization/ownership still has
  to align with those internode EDTs before scaling claims are valid.

## Research Direction For CODIR

The relevant codelet literature points to dataflow software pipelining:

- Dataflow Software Pipelining for the Codelet Model extends codelet execution
  to exploit pipelined parallelism across loop iterations using FIFO buffers
  across producer/consumer dependencies.
- Codelet Pipe turns that idea into an explicit communication channel for
  producer-consumer codelets and emphasizes single-owner FIFO buffers as the
  enabling abstraction.

CARTS should map this to the existing dialect split:

- SDE detects repeated stage structure, timestep legality, pipeline-carried
  values, reductions, halos, and barrier/control-token legality.
- CODIR builds and optimizes the codelet graph: producer/consumer stage edges,
  codelet fusion/fission, scalar forwarding, token-local views, explicit
  stream/control dependencies, and codelet ABI shape.
- ARTS decides whether those CODIR stream/control dependencies become local
  queues, DB-backed channels, epoch continuations, internode routes, replicated
  DBs, or remote acquire/release operations.
- ARTS-RT optimizes channel packing, GUID reservation, runtime-call hoisting,
  pointer rematerialization, and final vectorization metadata.

References:

- `Codelet Pipe: Realization of Dataflow Software Pipelining for Extended
  Codelet Model`, ICPP-W 2023, DOI `10.1145/3605731.3605885`.
- `Implementation of Dataflow Software Pipelining for Codelet Model`,
  ICPE 2023, DOI `10.1145/3578244.3583734`.
- `Position Paper: Extending Codelet Model for Dataflow Software Pipelining
  using Software-Hardware Co-Design`, COMPSAC 2019.

## Optimization Backlog By Dialect

### SDE

Keep these analyses and transforms source-semantic and runtime-neutral:

- Pattern coverage: matrix contractions, elementwise pipelines, reductions,
  2-D/3-D stencils, repeated timesteps, SW4/SPECFEM trailing owner dims, and
  Monte Carlo embarrassingly parallel kernels.
- Cost model: expose target-neutral logical capacity only through
  `sde.resource_query <logical_workers>`.
- Pipeline planning: convert adjacent repeated-stage candidates into explicit
  SDE control/dataflow tokens only when legality is proven.
- Boundary decomposition: split stencils into interior and boundary regions so
  ARTS can keep interior work coarse and boundary/halo work explicit.
- Reduction planning: choose tree, local-accumulate, or atomic strategies from
  source reductions, not runtime topology.

### CODIR

CODIR should be an active codelet optimizer, not just a carrier:

- Codelet graph analysis:
  build producer/consumer summaries from explicit deps, params, yielded values,
  and token-local memref views.
- Codelet fusion:
  fuse adjacent isolated codelets when their deps are compatible, they share
  owner slices, and fusion reduces launch/dependency overhead without
  destroying locality.
- Codelet fission:
  split overly coarse codelets when SDE provided a generic worker slice but the
  isolated body still hides independent inner work.
- Codelet pipelining:
  represent generic stream/control edges between stages without saying whether
  the runtime will use local queues, DBs, epochs, or internode messages.
- ABI compression:
  deduplicate scalar params, canonicalize capture order, and eliminate dead
  deps before ARTS assigns dependency slots.
- Token-local view canonicalization:
  make all body accesses use slice-local views so ARTS does not need to recover
  locality from raw memref arithmetic.

### ARTS

ARTS is where topology and runtime object policy become legal:

- Launch policy:
  materialize intranode vs internode EDT concurrency from module runtime config
  and CODIR generic worker plans.
- Route policy:
  map generic chunk ordinals to node routes, initially `chunk % total_nodes`,
  then richer owner-dim and data-affinity routes.
- DB ownership:
  align DB block roots with internode EDT owner slices; reject or diagnose
  coarse roots that feed internode work.
- Replicated read-mostly DBs:
  implement replicated lowering for read-only stencil/input DBs so every node
  can keep local input halos without remote acquire storms.
- Dependency-slot compression:
  group identical read-only deps, collapse redundant control deps, and
  specialize slot layout by codelet family.
- Epoch scheduling:
  choose continuation/amortization policies from EDT graph critical-path and
  fanout facts.
- Network-aware grain:
  ensure per-node work is large enough to amortize sender/receiver threads,
  route-table pressure, GUID reservation, and cross-node dependency traffic.

### ARTS-RT

ARTS-RT optimizes only after the ARTS graph is correct:

- GUID-range reservation for batched EDT/DB creation.
- Runtime-query hoisting and deduplication.
- Data pointer hoisting and scalar replacement inside lowered EDT bodies.
- DB/channel packing for repeated stream edges.
- Alias scopes and loop vectorization metadata for local EDT compute loops.

## Experiment Matrix

Use three experiment tiers.

### Tier 0: compiler-shape gates

These are cheap and must pass before runtime sweeps:

- For each workload, dump at `sde-planning`, `sde-to-codir`, `codir-to-arts`,
  `create-dbs`, `post-db-refinement`, and `pre-lowering`.
- Required invariants:
  - SDE has no ARTS topology queries or local/distributed region scope.
  - CODIR has no `arts.runtime_query`, no route, no `<internode>`.
  - ARTS is the first stage with `arts.runtime_query`, route, and
    `<internode>`.
  - Multinode candidates have multiple DB blocks before
    `db-distributed-ownership` can claim distributed ownership.
  - ARTS-RT sees no high-level DB/EDT/epoch objects after lowering.

### Tier 1: single-node performance controls

Run large/64 with repeat counts:

```bash
dekk carts benchmarks run --size large --timeout 180 --threads 64 --nodes 1 \
  --runs 5 --trace --results-dir .carts/outputs/experiments/single-node-large64
```

Controls:

- Large all-maintained: full regression and geomean.
- Small all-maintained: overhead sensitivity control.
- Matrix family: `polybench/gemm`, `polybench/2mm`, `polybench/3mm`.
- Stencil family: `polybench/convolution-2d`, `polybench/convolution-3d`,
  `polybench/jacobi2d`, `polybench/seidel-2d`.
- Domain kernels: `specfem3d/stress`, `specfem3d/velocity`,
  `sw4lite/rhs4sg-base`, `sw4lite/vel4sg-base`, `seissol/volume-integral`.
- ML/reduction family: `ml-kernels/activations`, `ml-kernels/batchnorm`,
  `ml-kernels/layernorm`, `ml-kernels/pooling`, `polybench/atax`,
  `polybench/bicg`.

Acceptance:

- All checksums pass.
- Full large/64 geomean remains at or above the M6 audited result.
- Any benchmark below `1.0x` gets an owning layer and root-cause note.

### Tier 2: multinode shape and scaling

Run only after Tier 0 proves internode EDTs and non-coarse DB roots.

Node sets:

- 1 node: local baseline.
- 2 nodes: `c09u25,c09u31`.
- 4 nodes: `c09u25,c09u31,c20u25,c20u31`.
- 8/16/32/64 nodes: only after the 4-node run shows positive scaling and the
  workspace-visible node list is verified.

Thread/network sweeps:

- Worker threads per node: `28`, `56`, `64`.
- Sender/receiver threads: `0/0`, `1/1`, `2/2`.
- Route-table sizes: `16`, `64`, `256`.
- Pinning: `pin=1` for normal runs, `pin=0` only for local multi-process
  simulation.

Work-size sweeps:

- Strong scaling: fixed large/extralarge problem, nodes `1,2,4`.
- Weak scaling: scale dominant output dimension or timestep count with node
  count; use `--weak-scaling` only after verifying each benchmark's size macro
  actually scales the intended work.
- Saturation sweep: increase problem size until kernel time is at least
  `1s` on 1 node for communication-sensitive workloads.

Candidate commands:

```bash
PATH="$PWD/.carts/outputs/multinode-slurm-20260516/bin:$PATH" \
CARTS_SRUN_NODELIST=c09u25,c09u31 \
dekk carts benchmarks run polybench/gemm polybench/2mm polybench/3mm \
  --size large --threads 56 --nodes 1,2 --arts --runs 3 --trace \
  --launcher slurm --timeout 240 \
  --results-dir .carts/outputs/experiments/multinode-matrix
```

```bash
PATH="$PWD/.carts/outputs/multinode-slurm-20260516/bin:$PATH" \
CARTS_SRUN_NODELIST=c09u25,c09u31,c20u25,c20u31 \
dekk carts benchmarks run polybench/convolution-3d specfem3d/stress \
  sw4lite/rhs4sg-base --size large --threads 56 --nodes 1,2,4 \
  --arts --runs 3 --trace --launcher slurm --timeout 240 \
  --results-dir .carts/outputs/experiments/multinode-stencil-domain
```

Acceptance:

- Checksums pass on every run.
- IR proves internode EDTs and multi-block/distributed DB roots.
- Per-node work is reported:
  total logical chunks, chunks per node, DB blocks per node, average EDT body
  work, and observed kernel/e2e/startup time.
- Scaling is reported as both kernel speedup and e2e speedup. Kernel-only
  speedup without DB/route evidence is not accepted.

## First Implementation Milestones

1. ARTS launch policy boundary:
   committed in `Materialize ARTS launch policy from generic codelets`.
2. Remove SDE machine-scope leakage:
   committed in `Remove SDE concurrency scope leakage` and
   `Update SDE tests for topology-neutral regions`.
3. Add ARTS shape diagnostics:
   print or dump per benchmark the count of intranode/internode EDTs,
   route-bearing EDTs, DB block counts, distributed/replicated DBs, and
   eligibility rejection reasons.
4. Fix DB materialization for planned owner slices:
   make host initialization/verification coexist with block DB roots through
   explicit whole-array host views or host-side copy-in/copy-out, not unsafe
   first-block substitution.
5. Add CODIR graph summaries:
   record codelet count, dep count, scalar param count, candidate fusion pairs,
   pipeline-stage groups, and token-local view coverage.
6. Re-run Tier 0, then Tier 1, then the 1/2/4-node Tier 2 matrix.

