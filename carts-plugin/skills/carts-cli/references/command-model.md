# CARTS Command Model

Use this when a task depends on how commands should be run.

## Entrypoints

- Guaranteed: `dekk carts ...`
- Optional wrapper: `carts ...` only after confirming `which carts` or
  the wrapper under the active CARTS install root is usable.
- Raw implementation path: avoid `python tools/carts_cli.py ...` in normal
  work. It bypasses the project runner assumptions agents should preserve.

## Local Artifact Root

Generated build/install artifacts resolve in this order:

1. `CARTS_HOME` environment variable
2. local untracked `carts.config` file (`[carts] home = "..."`
   is the preferred shape; `[carts] build = "..."` and
   `[carts] install = "..."` may split roots)
3. checkout root

Installed tools and libraries live under `<home>/.install/{carts,arts,polygeist,llvm}`.
Build trees live under `<home>/build/{carts,arts,polygeist,llvm-project}`.
When explicit build/install roots are configured, those roots replace
`<home>/build` and `<home>/.install` respectively.
The matching environment overrides are `CARTS_BUILD_ROOT` and
`CARTS_INSTALL_ROOT`; `CARTS_BUILD_DIR` and `CARTS_INSTALL_DIR` are Makefile
subproject variables.
Keep `.dekk.toml` and agent resources portable; put machine-local paths only in
environment variables or `carts.config`.

`tools/dekk-shims` is the only tracked PATH entry for CARTS-managed LLVM tools.
Those shims resolve the active install root dynamically, so Dekk can detect
`llvm-lit`, `FileCheck`, and `clang-format` without tracked machine-local paths.

## Required Preflight

For unfamiliar commands, run the command's `--help` before inventing flags:

```bash
dekk carts <command> --help
```

For compiler state, prefer live introspection:

```bash
dekk carts doctor
dekk carts pipeline --json
```

Use the JSON pipeline manifest when auditing pass order or doc drift.

## Never Do This For Normal Project Work

- `make`
- `ninja`
- direct `cmake`
- direct edits under `external/` without recognizing submodule boundaries
- raw command paths when a `dekk carts` command exists

## Generated Skill Resources

`.dekk.toml` uses `carts-plugin` as `[agents].source`; `dekk carts skills ...`
must use that configured source for generation, listing, status, and viewing.
