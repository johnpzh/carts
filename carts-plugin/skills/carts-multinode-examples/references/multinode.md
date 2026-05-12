# CARTS Multinode And Distributed Runs

## Concepts

Multi-node behavior depends on both runtime configuration and compiler
distributed lowering. Passing `--arts-config` at compile time matters because
the config is embedded into generated behavior.

Important config keys include worker threads, launcher, node count, and node
list. `samples/arts_multinode.cfg` is the sample-level multinode starting
point.

## Compiler Areas

- Distributed host loop outlining.
- EDT distribution.
- DB partitioning.
- Distributed DB ownership.
- Post-DB refinement.
- Pre-lowering and runtime route materialization.

## Discipline

1. Prove single-node first.
2. Use the same input with a multinode config.
3. Check that distributed flags/config are actually active.
4. Inspect stage IR around `edt-distribution`, `db-partitioning`,
   `post-db-refinement`, and `pre-lowering`.
5. Collect per-node logs/counters when the runtime fails.

Benchmarks support node sweeps via `dekk carts benchmarks run ... --nodes`.
