---
name: carts-debug
description: Use when the user asks to debug, diagnose, inspect, troubleshoot, profile, or analyze CARTS behavior and the failure is not yet classified, including crashes, core files, runtime segfaults, benchmark failures, logs, counters, and pipeline artifacts.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [<input-file>]
parameters:
  - name: bug_report
    type: str
    gather: "Describe the bug — what happens, what should happen, any error messages"
  - name: affected_files
    type: str
    gather: "Which files or tests are affected? Comma-separated paths or 'unknown'"
---

# CARTS Debug

Two layers: **compile-time** (compiler diagnostics) and **runtime** (ARTS debug builds, crash artifacts, and counters).

Shared resources live here for the other debugging skills too:
- `references/failure-signatures.md`
- `references/codepath-map.md`
- `references/command-patterns.md`
- `references/stage-ownership.md`
- `scripts/list-pipeline-stages.sh`
- `scripts/dump-stage.sh`
- `scripts/find-debug-channels.sh`
- `scripts/find-pass-codepaths.sh`
- `scripts/collect-debug-bundle.sh`

If the symptom is already clear, prefer the narrower skills:
- wrong output / checksum mismatch / semantic drift -> `carts-miscompile-triage`
- hang / crash / deadlock / route / epoch issue -> `carts-runtime-triage`
- multi-node / ownership / route / distributed-db issue -> `carts-distributed-triage`
- stale graph / invalidation / pass-ordering bug -> `carts-analysis-triage`
- large benchmark needs shrinking into a test -> `carts-reproducer`
- benchmark-specific failure or regression -> `carts-benchmark-triage`

## Compile-Time Diagnostics

```bash
# Stop at a pipeline stage (C or MLIR input)
dekk carts compile <file> --pipeline=<stage> -O3

# Dump ALL stages (MLIR input only)
dekk carts compile <file.mlir> --all-pipelines -o .carts/outputs/stages/<topic>/

# Diagnostic JSON (partitioning decisions, EDT/DB info)
dekk carts compile <file> --diagnose --diagnose-output diag.json -O3

# Debug channels (requires debug build)
dekk carts compile <file.mlir> --arts-debug db_partitioning,loop_fusion
```

**C input** compiles through Polygeist first, saving `<name>.mlir`. For full pipeline inspection, take that `.mlir` and use `--all-pipelines`.

**Debug channel convention**: `snake_case(PassFilename)` — for example
`DbModeTightening.cpp` maps to `db_mode_tightening`.

Use `dekk carts pipeline --json` for the canonical stage list.

## Crash Debugging

Check for LLDB in project-managed toolchains before assuming it is available:

```bash
command -v lldb || true
find .dekk .install external/Polygeist/llvm-project/build -maxdepth 5 \
  -type f \( -name lldb -o -name lldb-server \) -print
```

Validate a candidate with `<path>/lldb --version`. Only the LLDB driver or
`lldb-server` counts; helper scripts such as `lldb-python` do not.

If LLDB is unavailable, fall back to available symbol tools:

```bash
find .install/llvm/bin -maxdepth 1 -type f \
  \( -name llvm-symbolizer -o -name llvm-addr2line -o -name llvm-objdump \) -print
```

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
file /abs/path/to/core
readelf -n /abs/path/to/core | sed -n '1,120p'
lldb --batch \
  -o 'thread backtrace all' \
  -o 'image lookup --address <pc-if-known>' \
  --core /abs/path/to/core \
  /abs/path/to/executable
```

Preserve crash artifacts under `.carts/outputs/<topic>/`. If no usable core or
debugger exists, use signal-only `strace`:

```bash
strace -ff -tt -e trace=signal -o .carts/outputs/<topic>/trace \
  env artsConfig=/abs/path/to/arts.cfg /abs/path/to/executable
```

An empty core is not evidence. Check `ulimit -c` and
`/proc/sys/kernel/core_pattern` before relying on it. A `SEGV_ACCERR` address
near a worker thread stack guard often means stack exhaustion; inspect generated
LLVM for dynamic `alloca` inside loops.

## Runtime Debugging (ARTS)

```bash
dekk carts build --arts --debug 0    # Release, errors only
dekk carts build --arts --debug 1    # + warnings
dekk carts build --arts --debug 2    # + info messages
dekk carts build --arts --debug 3    # Debug build (-O0, sanitizers), full logs
```

Levels 0-2 stay Release. Level 3 is a Debug build — never benchmark with it. Logs go to stderr. Rebuild in release with `dekk carts build --arts` when done.

## Counter Profiles

```bash
dekk carts build --arts --counters 0  # all OFF (baseline)
dekk carts build --arts --counters 1  # timing only (default)
dekk carts build --arts --counters 2  # workload characterization
dekk carts build --arts --counters 3  # full overhead analysis
```

Counters are compile-time configured — rebuild ARTS to change. Output: JSON in `./counters/`.

## Quick Reference

| Symptom | Action |
|---------|--------|
| Program slow | `--diagnose` for partitioning, `--counters 2` for runtime |
| Pass crashes | `--all-pipelines` to find breaking stage |
| Wrong codegen | `--pipeline=<stage>` to inspect IR |
| Runtime hang | `dekk carts build --arts --debug 3`, run, check stderr |
| Benchmark regression | `--counters 1`, compare timing JSON |

## Shared References

Read the shared references before inventing a new workflow:
- `references/failure-signatures.md` — symptom-to-owner split
- `references/codepath-map.md` — high-value files by bug class
- `references/command-patterns.md` — command templates worth reusing
- `references/stage-ownership.md` — stage-to-ownership map for bisection

## Instructions

1. Identify the layer: compile-time or runtime
2. For compile-time: use `--diagnose`, `--pipeline`, `--arts-debug`, or `--all-pipelines`
3. For C source pipeline inspection: compile C first to get `.mlir`, then `--all-pipelines`
4. For crashes: preserve artifacts, check debugger/core availability, and capture a backtrace or symbolized address
5. For runtime progress: rebuild ARTS with `--debug 3` or `--counters N`, run the program
6. Remind user to rebuild ARTS in release mode when done debugging
