# CARTS Examples: Analysis Guide

For pipeline steps, debug commands, and troubleshooting, see:

- `AGENTS.md`
- `docs/compiler/pipeline.md`

## Running an Example

```bash
# Full compilation
dekk carts compile tests/examples/matrixmul/matrixmul.cpp -O3

# Inspect DB refinement
dekk carts compile <file>.mlir --pipeline=post-db-refinement

# Debug a specific pass
dekk carts compile <file>.mlir --pipeline=post-db-refinement --arts-debug=db_transforms 2>&1
```

For multi-node experiments, pass `--arts-config` at compile time — the config is embedded in the binary.
