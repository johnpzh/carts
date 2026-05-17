---
name: carts-multinode-examples
description: Use when the user asks how to compile or run CARTS examples on multiple nodes, including ARTS configs, launchers, SSH, or Slurm. Use carts-distributed-triage for failures.
---

# CARTS Multinode Examples

Read `references/multinode.md` before debugging distributed behavior.

## Workflow

1. Run or verify the equivalent single-node case first.
2. Compile with the intended ARTS config:
   ```bash
   dekk carts compile <file> -O3 --arts-config samples/arts_multinode.cfg -o .carts/outputs/multinode/<case>_arts
   ```
3. Inspect distributed stages:
   ```bash
   dekk carts compile <file> -O3 --arts-config <cfg> --pipeline=sde-planning
   dekk carts compile <file> -O3 --arts-config <cfg> --pipeline=codir-to-arts
   dekk carts compile <file> -O3 --arts-config <cfg> --pipeline=post-db-refinement
   dekk carts compile <file> -O3 --arts-config <cfg> --pipeline=pre-lowering
   ```
4. If runtime fails, collect node logs/counters and compare against local.
5. For benchmark-scale validation, use benchmark node sweeps.

Do not treat a distributed failure as a generic runtime bug until ownership and
routing have been checked.
