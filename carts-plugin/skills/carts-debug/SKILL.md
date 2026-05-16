---
name: carts-debug
description: Use when debugging CARTS crashes, core files, runtime segfaults, benchmark failures, or when choosing between LLDB, symbolizer, logs, counters, and pipeline artifacts.
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob
argument-hint: [<binary | core | benchmark-results-dir>]
parameters:
  - name: failure
    type: str
    gather: "Describe the CARTS crash, core file, benchmark failure, or runtime symptom."
  - name: artifact_path
    type: str
    gather: "Path to the executable, core, run directory, or results directory."
---

# CARTS Debug

Use this skill for crash-level debugging before changing compiler or runtime
code. CARTS has multiple tool roots; do not assume the outer shell PATH tells
the whole story.

## Debugger Discovery

Check for LLDB in the project-managed toolchains first:

```bash
command -v lldb || true
find .dekk .install external/Polygeist/llvm-project/build -maxdepth 5 \
  -type f \( -name lldb -o -name lldb-server \) -print
```

Validate a candidate before using it:

```bash
/path/to/lldb --version
```

Only the LLDB driver or `lldb-server` counts as LLDB availability. Helper
scripts such as `lldb-python` or `lldb-dotest` do not.

If a usable LLDB is found, use it. If it is not found, say so explicitly and
fall back to available symbol tools:

```bash
find .install/llvm/bin -maxdepth 1 -type f \
  \( -name llvm-symbolizer -o -name llvm-addr2line -o -name llvm-objdump \) -print
```

Do not claim LLDB is available just because the source tree contains
`external/Polygeist/llvm-project/lldb`.

## Running Under LLDB

For an ARTS executable:

```bash
lldb --batch \
  -o 'settings set target.env-vars artsConfig=/abs/path/to/arts.cfg' \
  -o run \
  -o 'thread backtrace all' \
  -o 'register read' \
  -- /abs/path/to/executable
```

For a core file:

```bash
lldb --batch \
  -o 'thread backtrace all' \
  -o 'image lookup --address <pc-if-known>' \
  --core /abs/path/to/core \
  /abs/path/to/executable
```

Before relying on a core, check that it is a real core and not an empty file:

```bash
file /abs/path/to/core
readelf -n /abs/path/to/core | sed -n '1,120p'
```

## CARTS Crash Triage

1. Preserve artifacts under `.carts/outputs/...`.
2. Confirm the failing command from `results.json`, `arts.log`, and
   `run_config.json`.
3. Reproduce directly with the same ARTS config:

```bash
env artsConfig=/abs/path/to/arts.cfg /abs/path/to/executable
```

4. If the failure is thread-sensitive, run a quick thread sweep:

```bash
dekk carts benchmarks run <suite/name> --size large --timeout 120 \
  --threads 1,8,64 --nodes 1 --trace \
  --results-dir .carts/outputs/<topic>-thread-triage
```

5. If the crash only appears at high thread count, inspect ARTS/ARTS-RT scheduling,
   dependency release, DB lifetime, and task resource assumptions before
   changing SDE semantic planning.
6. If the crash appears at one thread, inspect lowering shape, pointer
   materialization, and runtime ABI first.

## Fallback Without LLDB

If LLDB is unavailable:

- use `arts.log`, `results.json`, and generated `*.mlir` / `*-arts.ll`
- use `.install/llvm/bin/llvm-symbolizer` or
  `.install/llvm/bin/llvm-addr2line` when an address is available
- if core files are empty, check `ulimit -c` and `/proc/sys/kernel/core_pattern`
- if no usable core or debugger exists, use signal-only `strace` to capture the
  faulting thread and address:

```bash
strace -ff -tt -e trace=signal -o .carts/outputs/<topic>/trace \
  env artsConfig=/abs/path/to/arts.cfg /abs/path/to/executable
```

- a `SEGV_ACCERR` address near a worker thread stack guard usually means stack
  exhaustion; inspect generated LLVM for dynamic `alloca` inside loops
- rerun with ARTS debug/counter builds when runtime progress matters:

```bash
dekk carts build --arts --debug 3
dekk carts build --arts --counters 1
```

Rebuild release ARTS before making performance claims.

## Validation

After a fix:

```bash
dekk carts build
dekk carts test
dekk carts benchmarks run <suite/name> --size large --timeout 120 \
  --threads 64 --nodes 1 --trace \
  --results-dir .carts/outputs/<topic>-confirm
```
