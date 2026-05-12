# Getting Started

This guide takes you from a fresh clone to your first compiled CARTS program
in about 15 minutes (plus build time).

## Prerequisites

Install the following system tools before you begin:

| Tool | Version | Check |
|------|---------|-------|
| conda | any | `conda --version` |
| git | any | `git --version` |
| cmake | >= 3.20 | `cmake --version` |
| ninja | any | `ninja --version` |
| make | any | `make --version` |
| python | >= 3.11 | `python --version` |

Optional:

| Tool | Purpose |
|------|---------|
| clang-format | Source formatting (`carts format`) |
| lit | Test runner (`carts test`) -- built automatically if missing |

## 1. Clone

```bash
git clone <repo-url>
cd carts
```

You do **not** need `--recurse-submodules` -- the install step handles
submodules automatically with shallow clones to save bandwidth.

## 2. Install

The canonical install path is dekk-first. dekk creates the project-local conda
environment, activates the `.dekk.toml` configuration, initializes submodules,
builds the toolchain, and installs the `carts` wrapper. The Conda environment
also provisions the bootstrap `clang` and `clang++` pair automatically; they
are not separate system prerequisites for the dekk-first flow.

```bash
python -m pip install --upgrade dekk
dekk carts install
```

### What install does

1. **Creates or syncs the project environment** -- ensures `.dekk/env` matches
   `environment.yml`, including the bootstrap `clang`/`clang++` toolchain used
   for the pinned LLVM/MLIR build
2. **Initializes git submodules** -- ARTS, Polygeist, LLVM, carts-benchmarks
3. **Builds the full toolchain** in order:
   - LLVM + MLIR + OpenMP (longest step, ~30-60 min)
   - Polygeist (C-to-MLIR frontend)
   - ARTS runtime
   - CARTS compiler
`dekk carts install --wrap` additionally generates a project-local wrapper at
`./.install/carts`. Add `./.install` to your `PATH` yourself if you want to use
bare `carts ...` commands outside the dekk runner.

The installed binaries go into `.install/{llvm,polygeist,arts,carts}/`.

## 3. Use CARTS

After installation, the guaranteed interface is the dekk project runner:

```bash
dekk carts doctor    # System and toolchain diagnostics
dekk carts test      # Run the full test suite
```

If you generated the optional wrapper with `dekk carts install --wrap`, you can
also run `./.install/carts ...` directly, or add `./.install` to `PATH`.

### If something fails

```bash
# Check what's missing
dekk carts doctor

# Rebuild a specific component
dekk carts build --llvm      # Rebuild LLVM
dekk carts build --arts      # Rebuild ARTS runtime
dekk carts build --polygeist # Rebuild Polygeist
dekk carts build             # Rebuild CARTS compiler only
dekk carts build --clean     # Clean rebuild of CARTS
```

If you are updating an older checkout that previously built against LLVM 18,
clean the LLVM and Polygeist build trees before rebuilding. The current
Polygeist submodule is `llvmorg-23-init-8425-g23d8651de3f8`, and stale
pre-LLVM-23 artifacts can produce mismatched TableGen/CMake failures.

## 4. Compile your first program

CARTS compiles OpenMP-annotated C/C++ into ARTS task-based executables:

```bash
dekk carts compile samples/dotproduct/dotproduct.c -O3 -o dotproduct
./dotproduct
```

The `compile` command automatically routes through the full pipeline:
C source -> Polygeist (cgeist) -> MLIR passes -> LLVM IR -> executable.

### Compile flags

```bash
dekk carts compile file.c -O3 -o output    # Optimized build
dekk carts compile file.c -g -o output     # Debug build
dekk carts compile file.c --emit-llvm      # Stop at LLVM IR
dekk carts compile file.c --diagnose       # Emit compiler diagnostics
dekk carts compile file.c --pipeline=stage # Run up to a specific pipeline stage
```

Run `dekk carts compile --help` for the full list.

## 5. Explore examples

CARTS ships with 17 example programs covering common parallel patterns:

| Example | Pattern |
|---------|---------|
| `dotproduct/` | Parallel reduction |
| `matrixmul/` | Block task dependencies |
| `stencil/` | Neighbor (point) dependencies |
| `jacobi/` | Iterative stencil with convergence |
| `cholesky/` | Complex multi-block factorization |
| `parallel_for/` | Parallel loop variants |
| `smith-waterman/` | Dynamic programming wavefront |
| `convolution/` | Sliding window computation |

Run all examples:

```bash
dekk carts examples run --all
```

List available examples:

```bash
dekk carts examples list
```

All examples live in `samples/` and serve as reference patterns for
writing your own CARTS programs.

## Project structure

```
carts/
  include/arts/        C++ headers (dialect, passes, utilities)
  lib/arts/            Core MLIR dialect, analysis, and passes
  tools/               CLI entry point (carts_cli.py) and scripts
  tools/compile/       C++ compilation driver (carts-compile)
  lib/arts/dialect/*/test/ MLIR regression tests (FileCheck)
  tests/e2e/          End-to-end C/C++ compile-and-run tests
  samples/            Example C/C++ programs
  external/            Vendored deps (ARTS, Polygeist, LLVM)
  docs/                Architecture and pipeline documentation
  docker/              Multi-node Docker cluster setup
  .dekk.toml           Environment and dependency configuration
```

## Docker setup (multi-node)

For multi-node benchmarking with SLURM:

```bash
dekk carts docker build     # Build the Docker image
dekk carts docker start     # Start a 6-node simulated cluster
dekk carts docker exec      # Open a shell in the cluster
dekk carts docker stop      # Tear down
```

See [docker/README.md](../docker/README.md) for details.

## All CLI commands

Run `dekk carts --help` for the full command list. Key commands:

| Command | Description |
|---------|-------------|
| `dekk carts install` | Full project setup: environment, submodules, and build |
| `dekk carts install --wrap` | Also generate `./.install/carts` |
| `dekk carts doctor` | System and toolchain diagnostics |
| `dekk carts build` | Build components (`--arts`, `--llvm`, `--polygeist`) |
| `dekk carts compile` | Compile C/C++/MLIR through the pipeline |
| `dekk carts test` | Run the test suite |
| `dekk carts examples` | List and run example programs |
| `dekk carts benchmarks` | Performance benchmarking tools |
| `dekk carts pipeline` | Inspect compiler pipeline stages |
| `dekk carts clean` | Clean generated files |
| `dekk carts update` | Update git submodules |
| `dekk carts docker` | Multi-node Docker operations |
| `dekk carts format` | Format source files |
| `dekk carts skills` | Generate and inspect agent skill resources |
| `dekk carts worktree` | Create and manage project worktrees |

## Next steps

- [Developer Guide](developer-guide.md) -- Supported OpenMP constructs and
  memory layout requirements
- [Compiler Pipeline](compiler/pipeline.md) -- 16-stage compilation pipeline,
  pass ordering, and stage ownership
- [Contributing](contributing.md) -- Coding style, testing, and commit
  guidelines
