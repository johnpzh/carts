# carts

Language: C++, MLIR

CARTS is an MLIR-based compiler that transforms C/C++ programs with OpenMP
pragmas into task-based parallel executables targeting the ARTS asynchronous
runtime.

## Commands

- **build**: `dekk carts build` -- Build the CARTS compiler
- **test**: `dekk carts test` -- Run all contract tests
- **compile**: `dekk carts compile <file> -O3` -- Compile C/C++ to ARTS executable
- **pipeline**: `dekk carts pipeline --json` -- List compilation pipeline stages (16 core + 2 epilogue)
- **doctor**: `dekk carts doctor` -- Validate toolchain and environment
- **format**: `dekk carts format` -- Format C/C++/TableGen source files
- **lit**: `dekk carts lit <file>` -- Run single lit regression test
- **benchmarks**: `dekk carts benchmarks run --size large` -- Run benchmarks
- **triage-benchmark**: `dekk carts triage-benchmark` -- Benchmark triage workflow

## Build

```bash
dekk carts build
dekk carts build --arts       # Rebuild ARTS runtime
dekk carts build --clean      # Full clean rebuild
dekk carts build --polygeist  # Rebuild Polygeist frontend
```

## Test

```bash
dekk carts test
dekk carts lit lib/arts/dialect/core/test/my_test.mlir
```
