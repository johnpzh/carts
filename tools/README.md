# CARTS Tools

This directory contains the CARTS compiler CLI and supporting implementation
modules.

## Command Model

Use one command model only:

- `dekk carts ...` for project lifecycle operations and the guaranteed
  worktree-safe interface
- `carts ...` only if you explicitly generated the project-local wrapper with
  `dekk carts install --wrap` and exposed the active install root on your
  `PATH`

Do not use direct `python tools/carts_cli.py` invocations as the public
interface.

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
dekk carts lit lib/carts/dialect/arts/test/<case>.mlir
dekk carts benchmarks list
```

## Notes

- dekk creates the project-local conda environment from `environment.yml`
- Build/install artifacts use `CARTS_HOME` first, then local untracked
  `carts.config`, then the checkout root
- `dekk carts install --wrap` writes the wrapper under the active `.install`
  root
- `carts-compile` lives under the active `.install/carts/bin/`
