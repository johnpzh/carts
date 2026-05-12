# CARTS Local Examples

Current examples live in `samples/`. Stale docs may still mention
`tests/examples/`.

## Discovery And Runner

```bash
dekk carts examples list
dekk carts examples run <name>
dekk carts examples run --all
dekk carts test --suite e2e
```

The examples runner:

- discovers sample leaf directories under `samples/`;
- compiles from inside the sample directory;
- uses `samples/arts.cfg` unless a sample-local or explicit config overrides it;
- expects `<source-stem>_arts` in the sample directory;
- treats `[CARTS] ... FAIL` as failure;
- mutates ignored sample artifacts while cleaning/running.

For read-only validation or less workspace churn, prefer focused e2e lit tests.

## Recorded Failure Buckets

The April 2026 sweeps recorded compile failures around SDE captured allocas,
dominance/SSA escape from `cu_region`, and DB ref type mismatch; runtime wrong
results around reductions; and SIGSEGV for mixed access. Recheck live behavior
before assuming the baseline is unchanged.
