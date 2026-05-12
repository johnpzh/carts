# CARTS - Compiler for Asynchronous Runtime Systems

CARTS is an MLIR-based compiler that transforms C and C++ programs annotated
with OpenMP pragmas into task-based parallel executables targeting the ARTS
asynchronous runtime. It lowers OpenMP constructs through a multi-stage pipeline
of analysis, optimization, and code-generation passes, producing executables
that exploit fine-grained dataflow parallelism on shared-memory and distributed
systems.

## Quick Start
```bash
git clone <repo-url> && cd carts
python -m pip install --upgrade dekk
dekk carts install
```

`dekk carts install` creates or synchronizes the project-local Conda
environment from `environment.yml`, including the bootstrap `clang`/`clang++`
toolchain used to build the pinned LLVM/MLIR toolchain.

Use the project through dekk's project runner:

```bash
dekk carts doctor                                           # Verify the toolchain
dekk carts compile samples/dotproduct/dotproduct.c -O3 -o dotproduct
dekk carts test                                             # Run the test suite
```

If you want a bare `carts ...` command, run `dekk carts install --wrap` and add
`./.install` to your `PATH`. The wrapper is project-local; dekk does not edit
your shell config automatically.

See **[Getting Started](docs/getting-started.md)** for the full walkthrough.

## Project Structure

| Directory | Purpose |
|---|---|
| `include/arts/` | Public C++ headers (dialect, passes, utilities) |
| `lib/arts/` | Core MLIR dialect, analysis, and pass implementations |
| `tools/` | CLI entry point (`carts_cli.py`) and compilation driver |
| `tools/compile/` | C++ compilation driver (`carts-compile`) |
| `tools/scripts/` | Python modules backing each CLI subcommand |
| `tests/e2e/` | End-to-end compile+run lit tests (drives `samples/`) |
| `samples/` | Demo programs (C/C++ with OpenMP) |
| `external/` | Vendored dependencies (ARTS, Polygeist, LLVM) |
| `docs/` | Architecture, pipeline, and developer documentation |
| `docker/` | Container workflows for multi-node execution |

## CLI Commands

| Command | Description |
|---|---|
| `dekk carts install` | Create the project environment, fetch submodules, and build the toolchain |
| `dekk carts install --wrap` | Also generate a project-local `./.install/carts` wrapper |
| `dekk carts doctor` | Validate the toolchain and environment |
| `dekk carts build` | Build CARTS (`--clean` for fresh, `--arts` for runtime) |
| `dekk carts compile <file> [flags]` | Run the full compilation pipeline |
| `dekk carts pipeline` | Display or query compiler pipeline stages |
| `dekk carts test` | Run the complete test suite |
| `dekk carts lit <test.mlir>` | Run a single lit regression test |
| `dekk carts format` | Format tracked C/C++/TableGen files (`--check` to verify) |
| `dekk carts benchmarks <cmd>` | Build and run benchmarks (list, run, build, clean) |
| `dekk carts clean` | Remove generated files (`--all` for full clean) |
| `dekk carts update` | Update git submodules |
| `dekk carts docker <cmd>` | Docker operations (build, start, stop, exec) |
| `dekk carts examples <cmd>` | List or run bundled examples |
| `dekk carts skills <cmd>` | Generate and inspect agent skill resources |
| `dekk carts worktree <cmd>` | Manage dekk-aware CARTS worktrees |

## Documentation

- [Getting Started](docs/getting-started.md) -- setup, first compilation, examples
- [Developer Guide](docs/developer-guide.md) -- supported constructs and memory layouts
- [Compiler Pipeline](docs/compiler/pipeline.md) -- pass ordering and stage ownership
- [Contributing](docs/contributing.md) -- coding style, testing, and commit guidelines
