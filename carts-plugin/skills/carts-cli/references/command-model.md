# CARTS Command Model

Use this when a task depends on how commands should be run.

## Entrypoints

- Guaranteed: `dekk carts ...`
- Optional wrapper: `carts ...` only after confirming `which carts` or
  `./.install/carts`/`./.install/bin/carts` is usable.
- Raw implementation path: avoid `python tools/carts_cli.py ...` in normal
  work. It bypasses the project runner assumptions agents should preserve.

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
