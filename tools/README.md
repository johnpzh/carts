# CARTS Tools

This directory contains the CARTS compiler CLI and supporting implementation
modules.

## Command Model

Use one command model only:

- `dekk carts ...` for project lifecycle operations and the guaranteed
  worktree-safe interface
- `carts ...` only if you explicitly generated the project-local wrapper with
  `dekk carts install --wrap` and exposed `./.install` on your `PATH`

Do not use checked-in wrappers or direct `python tools/carts_cli.py` invocations
as the public interface. The repo-local `tools/carts` file exists only as a
compatibility shim for internal benchmark/tooling paths that still expect that
entrypoint.

## Directory Layout

```text
tools/
  carts_cli.py     Main CARTS CLI entry point
  compile/         C++ compilation driver (`carts-compile`)
  scripts/         Python modules backing CLI subcommands
```

## Common Commands

```bash
dekk carts install
dekk carts skills generate

dekk carts build
dekk carts compile simple.c -O3 -o simple_arts
dekk carts test
dekk carts lit lib/arts/dialect/core/test/<case>.mlir
dekk carts benchmarks list
```

## Notes

- dekk creates the project-local conda environment from `environment.yml`
- `dekk carts install --wrap` writes the wrapper to `./.install/carts`
- `carts-compile` lives under `.install/carts/bin/`
