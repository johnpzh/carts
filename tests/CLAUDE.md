# tests/ — Test Suites

## Directory Layout

```
cli/               CLI flag validation tests
verify/            Verification barrier pass tests
inputs/            Shared test input files (configs, snapshots)
```

Pass-level lit tests are co-located with source code:
```
lib/carts/dialect/sde/test/   Semantic (SDE) dialect tests (stages 1-3)
lib/carts/dialect/arts/test/  ARTS core dialect tests (stages 4-15)
  partitioning/              DB partitioning contract tests (modes/ + safety/)
lib/carts/dialect/arts-rt/test/    ARTS-RT lowering and LLVM-facing tests
```

Demo programs live at the project root:
```
samples/           End-to-end C/C++ demo programs (moved from tests/examples/)
```

## Which directory for new tests?

| Pipeline stage                          | Directory                          |
| --------------------------------------- | ---------------------------------- |
| 1 sde-input-normalization           | `lib/carts/dialect/sde/test/`       |
| 2 initial-cleanup                       | `lib/carts/dialect/sde/test/`       |
| 3 sde-planning                          | `lib/carts/dialect/sde/test/`       |
| 4 sde-to-codir                          | `lib/carts/dialect/codir/test/`    |
| 5 codir-to-arts                         | `lib/carts/dialect/codir/test/`    |
| 6 edt-transforms                        | `lib/carts/dialect/arts/test/`      |
| 7 create-dbs                            | `lib/carts/dialect/arts/test/`      |
| 8 db-opt                                | `lib/carts/dialect/arts/test/`      |
| 9 post-db-refinement                    | `lib/carts/dialect/arts/test/`      |
| 10 late-concurrency-cleanup             | `lib/carts/dialect/arts/test/`      |
| 11 epochs                               | `lib/carts/dialect/arts/test/`      |
| 12 pre-lowering                         | `lib/carts/dialect/arts/test/`      |
| 13 arts-rt-to-llvm                      | `lib/carts/dialect/arts-rt/test/`        |
| Verification passes (VerifyLowered etc) | `tests/verify/`                    |
| CLI flag/option validation              | `tests/cli/`                       |
