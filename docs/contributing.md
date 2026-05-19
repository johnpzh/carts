# Contributing to CARTS

## Project Structure

- `lib/carts/` - Core MLIR dialect implementation
- `include/carts/` - Public headers
- `tools/` - User-facing scripts and utilities
  - `scripts/` - Python CLI subcommands/helpers used through `dekk carts`
  - `compile/` - C++ compilation driver (`carts-compile`)
- `tests/` - Test suites
  - `contracts/` - Pipeline regression tests (MLIR + FileCheck)
  - `examples/` - End-to-end C/C++ tests
- `docker/` - Container workflows
- `external/` - Dependencies (ARTS, Polygeist, LLVM)

Compiler layering and ownership rules are documented in:
- `docs/compiler/pipeline.md`

## Build & Test Commands

```bash
# Setup and build
dekk carts install                      # Install prerequisites and build the toolchain
dekk carts install --wrap              # Also generate the optional project-local wrapper
dekk carts build                       # Build CARTS
dekk carts build --clean               # Clean build
dekk carts build --arts --debug 0      # ARTS errors only
dekk carts build --arts --debug 1      # ARTS warnings
dekk carts build --arts --debug 2      # ARTS info
dekk carts build --arts --debug 3      # Build ARTS with full debug logging

# Testing
dekk carts compile simple.cpp -o simple # Full compilation pipeline
dekk carts test                        # Run all tests
dekk carts lit lib/carts/dialect/arts/test/<file>.mlir # Run focused lit regressions
dekk carts lit --suite contracts       # Run the maintained contracts suite

# Formatting
dekk carts format                      # Format tracked C/C++/TableGen files
dekk carts format --check              # Check formatting without edits

# Docker
dekk carts docker build                # Build Docker image + workspace volume
```

## Environment Management

CARTS uses [dekk](https://pypi.org/project/dekk/) for environment management. The environment configuration is defined in `.dekk.toml` at the repository root.

### Setup

```bash
python -m pip install --upgrade dekk                      # once, globally
dekk carts install                                        # setup + build
```

dekk auto-detects the project, creates or syncs the conda environment from
`environment.yml`, activates `.dekk.toml` paths/env vars, installs the
bootstrap `clang`/`clang++` toolchain for the pinned LLVM/MLIR build, and
builds the rest of the stack. Add `--wrap` to generate the optional wrapper
under the active install root. Dekk does not modify your shell `PATH` as part of
`dekk carts install`.

Agent skill configuration is managed through dekk:

```bash
dekk carts skills status
dekk carts skills generate
dekk carts skills list
```

### Key Commands

```bash
dekk carts doctor         # Validate that all required tools and dependencies are present
dekk carts env            # Show current environment variables and paths
```

### How It Works

- `.dekk.toml` declares the required toolchain versions, environment variables, and paths.
- When you run `dekk carts <command>`, dekk reads `.dekk.toml`, activates the environment, and executes the requested project command.
- `dekk carts doctor` runs a suite of checks (cmake version, compiler availability, LLVM install, etc.) and reports any issues.

Run `dekk carts doctor` after cloning the repository and whenever you update dependencies to ensure your environment is correctly configured.

## Coding Style

Follow LLVM conventions:

- Two-space indentation
- Braces on same line
- `CamelCase` for types/ops (e.g., `DbDimOp`)
- `camelCase` for variables
- Run `dekk carts format` before committing
- Keep `// RUN:` and `FileCheck` directives aligned

## Testing Guidelines

1. **MLIR regression tests** → `lib/carts/dialect/<dialect>/test/`
   - Use the nearby lit style and FileCheck patterns
   - Cover new ops/passes with positive and negative cases

2. **End-to-end tests** → `tests/e2e/` and `tests/e2e_multinode/`
   - Use focused cases for compile-and-run behavior
   - Keep sample source programs in `samples/`

3. **Always run** `dekk carts lit <changed-test>` for focused compiler changes, and `dekk carts test` before submitting
4. **For distributed changes**, run the smallest local case first, then the
   relevant `tests/e2e_multinode/` or benchmark workflow.

## Commit Guidelines

- Imperative, single-sentence subjects (e.g., `Update Dockerfile to install Node.js`)
- Split functional and tooling changes
- Reference issues/tasks when available
- Include build/test commands in PR description
- Attach logs/screenshots for UI or benchmark changes

## Environment Notes

- Build and install trees are out-of-source under the active artifact root:
  `<CARTS_HOME>/build/{carts,arts,polygeist,llvm-project}` and
  `<CARTS_HOME>/.install/{carts,arts,polygeist,llvm}`
- Untracked `carts.config` may override build and install roots separately with
  `[carts] build = "..."` and `[carts] install = "..."`
- Environment overrides for split roots are `CARTS_BUILD_ROOT` and
  `CARTS_INSTALL_ROOT`; `CARTS_BUILD_DIR` and `CARTS_INSTALL_DIR` are reserved
  for Makefile subproject paths
- Never edit generated files
- Export `CARTS_VERBOSE=1` to debug wrapper invocations
- Docker scripts mount shared workspace volume for clean builds
- Environment configuration lives in `.dekk.toml` at the repo root
- Run `dekk carts doctor` to diagnose environment issues
