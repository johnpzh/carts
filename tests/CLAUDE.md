# tests/ — Test Suites

## Directory Layout

```
cli/               CLI flag validation tests
verify/            Verification barrier pass tests
inputs/            Shared test input files (configs, snapshots)
```

Pass-level lit tests are co-located with source code:
```
lib/arts/dialect/sde/test/   Semantic (SDE) dialect tests (stages 1-3)
lib/arts/dialect/core/test/  ARTS core dialect tests (stages 4-15)
  partitioning/              DB partitioning contract tests (modes/ + safety/)
lib/arts/dialect/rt/test/    Runtime dialect tests (stage 16 + epilogues)
```

Demo programs live at the project root:
```
samples/           End-to-end C/C++ demo programs (moved from tests/examples/)
```

## Which directory for new tests?

| Pipeline stage                          | Directory                          |
| --------------------------------------- | ---------------------------------- |
| 1 raise-memref-dimensionality           | `lib/arts/dialect/sde/test/`       |
| 2 initial-cleanup                       | `lib/arts/dialect/sde/test/`       |
| 3 sde-planning                          | `lib/arts/dialect/sde/test/`       |
| 4 sde-to-codir                          | `lib/carts/dialect/codir/test/`    |
| 5 codir-to-arts                         | `lib/carts/dialect/codir/test/`    |
| 6 edt-transforms                        | `lib/arts/dialect/core/test/`      |
| 7 create-dbs                            | `lib/arts/dialect/core/test/`      |
| 8 db-opt                                | `lib/arts/dialect/core/test/`      |
| 9 post-db-refinement                    | `lib/arts/dialect/core/test/`      |
| 10 late-concurrency-cleanup             | `lib/arts/dialect/core/test/`      |
| 11 epochs                               | `lib/arts/dialect/core/test/`      |
| 12 pre-lowering                         | `lib/arts/dialect/core/test/`      |
| 13 arts-to-llvm                         | `lib/arts/dialect/rt/test/`        |
| Verification passes (VerifyLowered etc) | `tests/verify/`                    |
| CLI flag/option validation              | `tests/cli/`                       |
