# Debug Command Patterns

Reuse these before improvising.

## Pipeline Introspection

```bash
dekk carts pipeline
dekk carts pipeline --json
```

## Stage Dump

```bash
dekk carts compile input.mlir --pipeline=post-db-refinement
dekk carts compile input.mlir --all-pipelines -o outdir/
```

## Diagnose Bundle

```bash
dekk carts compile input.mlir --diagnose --diagnose-output diagnose.json
```

## ARTS Debug Build

```bash
dekk carts build --arts --debug 3
dekk carts build --arts --counters 2
```

## Benchmark Triage

```bash
dekk carts triage-benchmark suite/name --size small --threads 2
```

## Multi-Node Benchmark

```bash
dekk carts benchmarks run polybench/2mm \
  --size small \
  --nodes 2 \
  --threads 4 \
  --arts-config docker/arts-docker-2node.cfg \
  --debug 2
```
