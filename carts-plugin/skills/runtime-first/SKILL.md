---
name: carts-runtime-first
description: Use after runtime triage shows the compiler must match an ARTS runtime contract for EDTs, DBs, epochs, dependencies, or distributed execution.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [<benchmark-name | bug-description>]
---

# Runtime-First Development

Goal: never guess what the compiler should generate. Instead, understand the
runtime contract, write the correct code by hand, prove it works, then teach the
compiler to emit that exact pattern.

## Methodology

### Phase 1 — Understand the Runtime

1. Read the ARTS runtime API headers and examples.
   - `external/arts/include/arts.h` — core API
   - `external/arts/include/artsEdtFunctions.h` — EDT creation/signaling
   - `external/arts/include/artsDbFunctions.h` — DB alloc/acquire/put/destroy
   - `external/arts/examples/cpu/` — working examples (especially `cps_chain.c`)
2. Identify the runtime contract for the feature in question:
   - How are DBs passed between EDTs? (answer: via depv, never raw pointers in paramv)
   - How are EDTs chained? (answer: satisfaction of deps triggers execution)
   - How do epochs work? (answer: artsInitializeAndStartEpoch, edts register, epoch completes when all finish)
   - How does CPS work? (answer: continuation EDT created with deps, signaled when predecessor finishes)
3. Document the runtime contract in a reference file.

### Phase 2 — Write Correct Code by Hand

1. Write a minimal C program using ONLY the ARTS runtime API that implements
   the same computation as the failing benchmark.
2. The hand-written code must:
   - Use `artsEdtCreate` / `artsEdtCreateWithGuid` for EDT creation
   - Pass DBs through `artsSignalEdtValue` or `artsEdtDep_t` arrays — NEVER as raw pointers
   - Use `artsDbCreate` / `artsDbCreateWithGuid` for DB allocation
   - Use `artsDbGet` to acquire DB data inside EDTs
   - Use epochs correctly (`artsInitializeAndStartEpoch` / `artsWaitOnEpoch`)
   - Work in both single-node and distributed modes
3. Place the hand-written code under `.carts/outputs/runtime-first/` or promote
   it to `samples/` only after it becomes a maintained example.

### Phase 3 — Verify It Runs

1. Compile the hand-written code:
   ```bash
   dekk carts compile .carts/outputs/runtime-first/<file>.c -O3 --arts-only
   ```
   Or compile directly against the ARTS runtime:
   ```bash
   gcc -I external/arts/include -L <arts-lib-path> -larts <file>.c -o <file>_manual
   ```
2. Run with the standard ARTS config:
   ```bash
   ARTS_CONFIG=samples/arts.cfg ./<file>_manual
   ```
3. Verify: no crashes, correct output, clean shutdown.

### Phase 4 — Compare with Compiler Output

1. Compile the same benchmark with the CARTS compiler:
   ```bash
   dekk carts compile <benchmark>.c -O3 --all-pipelines -o .carts/outputs/runtime-first/<topic>-stages/
   ```
2. Compare the compiler's `pre-lowering` and `arts-rt-to-llvm` stages with the
   hand-written code:
   - EDT creation: same number of EDTs? Same dep counts?
   - DB transport: DBs flowing through depv or paramv?
   - Epoch structure: same nesting?
   - Param packing: what goes in paramv vs depv?
3. Identify EXACTLY where the compiler diverges from the correct pattern.

### Phase 5 — Fix the Compiler

1. The fix should make the compiler generate code that matches the hand-written
   pattern — not "something that works differently."
2. Modify the relevant lowering pass(es) to emit the correct runtime calls.
3. Re-compile the benchmark and verify the output matches the hand-written code
   at the LLVM IR level.
4. Run the full test suite: `dekk carts test`
5. Run the benchmark: `dekk carts benchmarks run --size large --timeout 300 --threads 16`

## ARTS Execution Model — Key Rules

These are NON-NEGOTIABLE constraints from the ARTS distributed runtime:

1. **DBs flow through depv, not paramv.** The runtime resolves GUIDs to local
   data pointers. Raw pointers are only valid on the local node.
2. **GUIDs are globally valid.** Pass GUIDs (as i64) through paramv when you
   need to reference a DB without a dependency edge.
3. **EDTs receive data through `artsEdtDep_t *depv`.** Each dep slot contains
   a `{guid, ptr, mode}` triple. The runtime fills `ptr` before EDT execution.
4. **`artsSignalEdtValue(edtGuid, slot, dbGuid, mode)` adds a dep.** This is
   how you connect DBs to EDTs at runtime.
5. **Epochs are hierarchical.** Child epochs must complete before parent epochs.
6. **CPS chains use continuation EDTs.** The last EDT in a chain signals the
   continuation with updated state (GUIDs in depv, scalars in paramv).

## Files to Read First

| File | What It Tells You |
|------|-------------------|
| `external/arts/include/arts.h` | Full ARTS API |
| `external/arts/include/artsEdtFunctions.h` | EDT creation, signaling |
| `external/arts/include/artsDbFunctions.h` | DB lifecycle |
| `external/arts/examples/cpu/cps_chain.c` | CPS chain pattern |
| `external/arts/examples/cpu/stencil1d.c` | Stencil with halo exchange |
| `lib/carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/ConvertArtsRtToLLVM.cpp` | How compiler emits runtime calls |
| `lib/carts/dialect/arts-rt/Conversion/ArtsToRt/EdtLowering.cpp` | EDT outlining and param packing |
| `lib/carts/dialect/arts-rt/Conversion/ArtsToRt/EpochLowering.cpp` | Epoch/CPS lowering |

## Anti-Patterns (What NOT to Do)

- Do NOT pass memref pointers through paramv — they are local addresses
- Do NOT clone DbAllocOp inside outlined EDT functions — creates duplicate DBs
- Do NOT skip dependency edges for "convenience" — the runtime needs them for scheduling
- Do NOT assume single-node execution — always design for distributed
- Do NOT guess the runtime contract — read the examples and headers first
