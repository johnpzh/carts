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
toolchain used to build the pinned LLVM/MLIR toolchain. It also regenerates
the repo-managed agent resources, including MCP adapters, from `carts-plugin/`.

Use the project through dekk's project runner:

```bash
dekk carts doctor                                           # Verify the toolchain
dekk carts compile samples/dotproduct/dotproduct.c -O3 -o dotproduct
dekk carts test                                             # Run the test suite
```

If you want a bare `carts ...` command, run `dekk carts install --wrap` and add
the active install root to your `PATH`. The wrapper is project-local; dekk does
not edit your shell config automatically.

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

## Managed Artifact Layout

Dekk resolves the active artifact root from `CARTS_HOME`, then the untracked
`carts.config`, then the checkout root. By default generated build and install
outputs live under that one root:

```text
<CARTS_HOME>/build/{carts,arts,polygeist,llvm-project}
<CARTS_HOME>/.install/{carts,arts,polygeist,llvm}
```

For machines where build and install trees need different filesystems, set
explicit roots in the untracked local config:

```toml
[carts]
home = ".carts"
build = ".carts/build"
install = ".carts/install"
```

The equivalent environment overrides are `CARTS_BUILD_ROOT` and
`CARTS_INSTALL_ROOT`. Avoid using `CARTS_BUILD_DIR` or `CARTS_INSTALL_DIR` for
root overrides; those names are Makefile subproject paths.

Sources remain in the checkout (`external/arts`, `external/Polygeist`, etc.).
Build scripts and benchmark jobs should use the Dekk-resolved paths instead of
probing subproject-local build or install directories.

## CLI Commands

| Command | Description |
|---|---|
| `dekk carts install` | Create the project environment, sync agent/MCP resources, fetch submodules, and build the toolchain |
| `dekk carts install --wrap` | Also generate a wrapper under the active install root |
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

## ARTS Transport

`dekk carts build --arts` builds the ARTS runtime with RDMA/RoCE RSockets
enabled by default, matching the production multinode transport. Use
`dekk carts build --arts --no-rdma` when working on a non-RDMA developer system
or when running an intentional TCP fallback experiment. Benchmark runs use TCP
for single-node configs and use the requested `--rdma/--no-rdma` transport only
for multinode configs. Likewise, benchmark `--distributed-db` compile args are
multinode-only and are stripped from single-node benchmark builds.
