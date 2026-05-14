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
| 3 openmp-to-arts                        | `lib/arts/dialect/sde/test/`       |
| 4 edt-transforms                        | `lib/arts/dialect/core/test/`      |
| 5 create-dbs                            | `lib/arts/dialect/core/test/`      |
| 6 db-opt                                | `lib/arts/dialect/core/test/`      |
| 7 post-db-refinement                    | `lib/arts/dialect/core/test/`      |
| 8 late-concurrency-cleanup              | `lib/arts/dialect/core/test/`      |
| 9 epochs                                | `lib/arts/dialect/core/test/`      |
| 10 pre-lowering                         | `lib/arts/dialect/core/test/`      |
| 11 arts-to-llvm                         | `lib/arts/dialect/rt/test/`        |
| Verification passes (VerifyLowered etc) | `tests/verify/`                    |
| CLI flag/option validation              | `tests/cli/`                       |
