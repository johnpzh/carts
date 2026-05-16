---
name: carts-runtime-triage
description: Use when compilation succeeds but the generated ARTS executable hangs, deadlocks, crashes, stalls, or reports anomalous runtime counters.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [<binary | benchmark-path>]
parameters:
  - name: bug_description
    type: str
    gather: "Describe the runtime failure — hang, crash, deadlock, etc."
  - name: trace_file
    type: str
    gather: "Path to the binary, benchmark, or ARTS trace file"
---

# CARTS Runtime Triage

Goal: classify the runtime failure before changing compiler code.

Use bundled helpers when they fit:
- `scripts/build-arts-debug.sh` — rebuild ARTS at a chosen debug level
- `scripts/build-arts-counters.sh` — rebuild ARTS with a chosen counters level
- `scripts/locate-runtime-artifacts.sh` — find logs and counter JSONs under a run directory
- `scripts/find-runtime-codepaths.sh` — grep the main runtime/lowering codepaths

Read these before changing code:
- `references/runtime-checklist.md`
- `../debug/references/failure-signatures.md`
- `../debug/references/codepath-map.md`
- `../debug/references/command-patterns.md`

## Triage Order

1. Confirm the environment and rebuild the relevant bits.
   - `dekk carts doctor`
   - `dekk carts build`
   - `dekk carts build --arts` if the runtime may be stale
2. Classify the symptom.
   - crash / assertion / sanitizer
   - hang / deadlock / no progress
   - slow execution / regression
   - distributed-only failure
3. Switch ARTS to a debug build for runtime inspection.

```bash
dekk carts build --arts --debug 3
```

Level 3 is for debugging only. Rebuild release ARTS before performance conclusions.

4. For performance or progress issues, collect counters.

```bash
dekk carts build --arts --counters 1
dekk carts build --arts --counters 2
dekk carts build --arts --counters 3
```

5. Check the runtime structure that should exist.
   - epochs: `arts.create_epoch`, `arts.wait_on_epoch`
   - task completion / continuation edges
   - distributed DB ownership, routes, and owner-local init behavior
6. If the failure smells like compiler-lowered structure rather than runtime behavior, hand off to `carts-miscompile-triage`.
7. If the failure is specific to multi-node ownership or distributed routing, hand off to `carts-distributed-triage`.

## Distributed-Specific Checks

- Verify whether the DB is marked `distributed`
- Check route selection and owner assumptions in `lib/carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/ConvertArtsRtToLLVM.cpp`
- Check distributed init / worker-local creation in `lib/carts/codegen/Codegen.cpp`
- Compare single-node vs distributed runs before assuming a general runtime bug

## Key Docs

- `docs/heuristics/distribution.md`
- `docs/heuristics/partitioning.md`
- `AGENTS.md` section: Distributed Runtime Debug
- `lib/carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/ConvertArtsRtToLLVM.cpp`
- `lib/carts/dialect/codir/Conversion/SdeToCodir/SdeToCodir.cpp`
- `lib/carts/dialect/codir/Conversion/CodirToArts/CodirToArts.cpp`
- `lib/carts/dialect/arts-rt/Conversion/ArtsToRt/EdtLowering.cpp`
- `lib/carts/dialect/arts-rt/Conversion/ArtsToRt/EpochLowering.cpp`
- `lib/carts/dialect/arts/Analysis/edt/`

## Validation

Always rerun the failing executable after changing build mode or instrumentation, and again after reverting to release mode.
