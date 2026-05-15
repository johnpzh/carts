# Verification And Release Plan

## Objective

Keep the restructuring testable at every step. Migration should proceed through
small slices with focused lit tests, pipeline dumps, e2e coverage, and benchmark
evidence matched to risk.

## Baseline Commands

Use the project entrypoint:

```bash
dekk carts doctor
dekk carts build
dekk carts pipeline --json
dekk carts test
```

Use focused commands during development:

```bash
dekk carts lit <file>
dekk carts examples run <name>
dekk carts benchmarks run <benchmark> --size large --timeout 120 --threads 64 --nodes 1 --trace
```

## Artifact Policy

- Put generated diagnostics, stage dumps, benchmark outputs, and logs under
  `.carts/outputs/...`.
- Put multi-step investigation state under
  `.carts/sessions/<YYYYMMDD-HHMMSS>-<topic>/...`.
- Do not commit `.carts/` artifacts unless a reduced case is promoted into a
  real test fixture.

## Test Matrix

### Docs And Skills

Run when docs, command policy, or generated agent resources change:

```bash
dekk carts skills generate
git diff --check
```

### Single Pass Or Dialect Change

Run:

- focused lit test for the changed pass;
- owning dialect lit suite if the pass changes shared behavior;
- `dekk carts build` when TableGen, dialect registration, or headers change.

### Conversion Boundary Change

Run:

- SDE-to-CODIR or SDE-to-ARTS focused lit tests;
- CODIR-to-ARTS focused lit tests once CODIR exists;
- ARTS/ARTS-RT lowering tests for emitted DB/EDT/epoch shape;
- one e2e sample that uses memory deps and scalar params.

### Shared Analysis Or Pipeline Change

Run:

```bash
dekk carts test
dekk carts test --suite e2e
```

Add focused benchmarks for affected families.

### Runtime Or Distributed Change

Run local first, then the smallest distributed benchmark that exercises the
path. Rebuild ARTS through the project entrypoint when runtime code changes:

```bash
dekk carts build --arts
```

## Release Gates

### Gate A: Architecture Skeleton

- `dekk carts build`
- `dekk carts pipeline --json`
- CODIR dialect registration lit tests if CODIR skeleton exists

### Gate B: Codelet Isolation

- negative capture tests;
- positive dep/param tests;
- focused EDT lowering tests;
- e2e sample with memory dep plus scalar param.

### Gate C: Memref Rewrite

- SDE token-local rewrite lit tests for 1D, ND, strided, and halo windows;
- no tensor fallback for covered cases;
- focused sample compile/run.

### Gate D: ARTS Cleanup

- direct DB/acquire/EDT materialization tests;
- negative tests for unsupported raw memref fallback;
- pipeline dumps proving supported cases bypass the coarse raw `CreateDbs`
  bridge.

### Gate E: Performance

- focused large/64 benchmark for each changed family;
- repeated run for noisy or surprising results;
- updated current evidence in `benchmark-performance-goal.md`.

## Final Done Definition

The restructuring is complete when:

- supported paths use `sde -> codir -> arts -> arts-rt`;
- EDTs are isolated from above and lower from explicit deps/params;
- tensor fallback is gone for supported cases;
- raw-memref `CreateDbs` is removed or limited to a coarse bridge plus
  unsupported diagnostics for non-coarse inputs, and Core DB indexer
  compatibility paths stay removed;
- maintained benchmarks are correctness-clean and classified;
- no late ARTS policy decision belongs in SDE or CODIR.
