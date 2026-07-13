# CARTS mental model — entry points, compile flow, IR dumping

Orientation notes for someone new to the CARTS codebase. Verified against the
live compiler on 2026-07-12. When these notes and the compiler disagree, the
compiler wins — re-check with `dekk carts pipeline --json` and
`tools/compile/Compile.cpp`.

CARTS = MLIR-based compiler: C/C++ + OpenMP pragmas → task-based executable on
the ARTS async runtime. It lowers through four project dialects:
**SDE → CODIR → ARTS → ARTS-RT**, then to LLVM.

## Two layers (keep these separate)

| Layer | What it is | Entry point |
|-------|-----------|-------------|
| **CLI orchestrator** (Python) | `dekk carts compile` — picks which tools run in what order, routes `.c`/`.mlir`/`.ll`, discovers `arts.cfg`, links the executable | `compile_cmd` — `tools/scripts/compile.py:613` |
| **The compiler** (C++/MLIR) | `carts-compile` — the actual MLIR pass pipeline (SDE→CODIR→ARTS→ARTS-RT→LLVM) | `int main` — `tools/compile/Compile.cpp:1712` |

`dekk carts compile` is just a driver. The real compiler is the `carts-compile`
binary. (C frontend `cgeist`/Polygeist and linker `clang` are separate binaries,
not part of `carts-compile`.)

## 1. Where is `main`?

`tools/compile/Compile.cpp:1712` — the canonical pipeline (CLAUDE.md says so).
Short and linear:

- `:1714` `registerPassManagerCLOptions()` — wires in standard MLIR debug flags
  (this is why `--mlir-print-ir-*` work; see §3E).
- `:1748` `registerDialects()` — registers `sde`, `codir`, `arts`, `arts_rt` + upstream.
- `:1760` parse input `.mlir` module.
- `:1814` `buildPassManager(...)` — assembles the stage pipeline, honoring
  `--start-from` / `--pipeline`.
- `:1842`–`:1877` — translate to LLVM IR (`--emit-llvm`) or print final MLIR.

Stages are declared as `enum class StageId` at `Compile.cpp:166` and assembled by
`getStageRegistry()` (each stage = a `StageDescriptor` at `Compile.cpp:199`
bundling a list of passes).

Python CLI entry is `tools/carts_cli.py`.

## 2. What `dekk carts compile <file>` does under the hood

Driver routes on extension (`compile_cmd`, `compile.py:666`):

- `.c` / `.cpp` → full 3-step pipeline
- `.mlir` → step 2 only
- `.ll` → step 3 only (link)

For a `.c` file, the three steps are `_compile_c_pipeline` (`compile.py:1034`):

```
  file.c
    │  Step 1: cgeist   (Polygeist frontend: C + OpenMP → MLIR)
    ▼
  file.mlir             ← polygeist/memref/scf/openmp dialects
    │  Step 2: carts-compile --O3 --emit-llvm    ← THE COMPILER
    │           runs the 13-stage SDE→CODIR→ARTS→ARTS-RT pipeline,
    │           then translates to LLVM IR
    ▼
  file-arts.ll          ← LLVM IR
    │  Step 3: clang    (link with the ARTS runtime)
    ▼
  file_arts             ← executable
```

Both intermediates (`file.mlir`, `file-arts.ll`) are left in the cwd after a
normal compile — free access to frontend MLIR and final LLVM IR.

Step 2 is the heart. The 13 stages, grouped by dialect layer
(from `dekk carts pipeline --json`):

| Dialect layer | Stages |
|---------------|--------|
| **frontend normalization** | `sde-input-normalization`, `initial-cleanup` |
| **SDE** (OpenMP semantics; tiling/vectorization/distribution *planning*) | `sde-planning` |
| **SDE → CODIR** (isolate codelets) | `sde-to-codir` |
| **CODIR → ARTS** (materialize DB/EDT objects; no SDE op may survive) | `codir-to-arts` |
| **ARTS object refinement** (abstract DB/EDT/epoch orchestration) | `edt-transforms`, `create-dbs`, `db-opt`, `post-db-refinement`, `late-concurrency-cleanup`, `epochs` |
| **ARTS-RT lowering** (runtime ABI, then LLVM) | `pre-lowering`, `arts-rt-to-llvm` |
| **conditional epilogues** | `post-o3-opt` (with `-O3`), `llvm-ir-emission` (with `--emit-llvm`) |

Each stage = a bundle of MLIR passes. `dekk carts pipeline --json` lists the
exact passes and dependency edges per stage.

## 3. Printing IR at different dialect levels

Rough level guide: `sde-planning` → SDE, `sde-to-codir` → CODIR,
`codir-to-arts`…`epochs` → ARTS, `pre-lowering` → ARTS-RT,
`arts-rt-to-llvm` → LLVM dialect.

**A. Dump every stage to a directory** (best overview; pipeline-map skill's
recommended command):
```bash
dekk carts compile file.c -O3 --all-pipelines -o .carts/outputs/stages/mycase
```
One `.mlir` per stage (`NN_stage.mlir`, grouped in phase dirs). Diff adjacent
stages to watch `sde.*` → `codir.*` → `arts.*` → `arts_rt.*`.

**B. Stop after a specific stage** (see one dialect level):
```bash
dekk carts compile file.c    --pipeline=sde-planning  -o out.mlir   # SDE-level
dekk carts compile file.mlir --pipeline=codir-to-arts -o out.mlir   # ARTS objects
```
Stage tokens: the table above, or `dekk carts pipeline --json`.

**C. Run a single stage / range** (MLIR input only; feed it a stage dump from A):
```bash
dekk carts compile stageNN.mlir --start-from=create-dbs --pipeline=db-opt -o out.mlir
```

**D. Emit final LLVM IR:**
```bash
dekk carts compile file.mlir --emit-llvm -o out.ll
```

**E. Fine-grained per-pass IR dumping.** `main` calls
`registerPassManagerCLOptions()`, so standard MLIR flags exist on `carts-compile`,
and the driver forwards long `--*` flags to it (`compile.py:431`). Combine with a
stage stop to keep output small:
```bash
dekk carts compile file.mlir --pipeline=sde-planning --mlir-print-ir-after-all
dekk carts compile file.mlir --mlir-print-ir-before=Tiling --mlir-print-ir-after=Tiling
```

**Inspect the pipeline itself** (no compile):
```bash
dekk carts pipeline --json     # every stage → passes → dependency edges
```

## Good first exploration

Run **A** on `samples/dotproduct/dotproduct.c`, then open the numbered stage
files in order — watch the OpenMP loop become SDE planning, then isolated CODIR
codelets, then ARTS DBs/EDTs, then runtime calls.

## Where to look

- Pipeline definition / `main`: `tools/compile/Compile.cpp` (`StageId` `:166`,
  `getStageRegistry`, `main` `:1712`).
- CLI driver / step wiring: `tools/scripts/compile.py`
  (`compile_cmd` `:613`, `_compile_c_pipeline` `:1034`).
- Dialect sources: `lib/carts/dialect/{sde,codir,arts,arts-rt}/`,
  headers `include/carts/dialect/*`, tests co-located under each `*/test/`.
- Live facts: `dekk carts pipeline --json`, `dekk carts compile --help`.
