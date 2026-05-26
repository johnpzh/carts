# carts

Language: C++, MLIR

CARTS is an MLIR-based compiler that transforms C/C++ programs with OpenMP
pragmas into task-based parallel executables targeting the ARTS asynchronous
runtime.

## Command Policy

- Use `dekk carts ...` as the guaranteed project entrypoint.
- Use bare `carts ...` only after confirming the project-local wrapper exists
  and is on `PATH`.
- Do not use raw `make`, `ninja`, `cmake`, or `python tools/carts_cli.py` for
  normal project work; Dekk sets up the required environment.
- Generated build/install artifacts live under `CARTS_HOME` when it is set.
  Otherwise CARTS reads the local untracked `carts.config` file, then falls
  back to the checkout root. Keep tracked config portable.
- Dekk-visible managed tools are exposed through portable shims under
  `tools/dekk-shims`; the shims resolve the active CARTS install root at
  runtime.

## Essential Commands

- `dekk carts doctor` - validate the toolchain and environment.
- `dekk carts build` - build the CARTS compiler.
- `dekk carts build --arts` - rebuild ARTS runtime with RDMA/RoCE enabled for
  production multinode transport.
- `dekk carts build --arts --no-rdma` - rebuild ARTS runtime for TCP fallback.
- `dekk carts compile <file> -O3` - compile C/C++ to an ARTS executable.
- `dekk carts pipeline --json` - inspect pipeline stage tokens.
- `dekk carts test` - run default pass tests.
- `dekk carts test --suite e2e` - run compile-and-run sample tests.
- `dekk carts lit <file>` - run one lit regression.
- `dekk carts examples list` - list sample programs.
- `dekk carts examples run <name>` - compile and run one sample.
- `dekk carts benchmarks list` - list benchmark workloads.
- `dekk carts skills generate` - regenerate agent-facing project resources.

## Compiler Shape

CARTS is migrating toward four project dialect layers:

- SDE (`sde`) - source semantics, PatternAnalysis, MU/CU/SU planning, memref
  tiling/access windows, reductions, and target-neutral scheduling intent.
- CODIR (`codir`) - isolated codelets, explicit deps/params, token-local
  memref views, and codelet-local verification.
- ARTS (`arts`) - abstract DB, EDT, epoch, dependency-slot, placement, and
  distributed ownership objects.
- ARTS-RT (`arts_rt`) - runtime ABI, packing, pointer lowering, and
  LLVM-facing cleanup.

The canonical pipeline is defined in `tools/compile/Compile.cpp`. When docs or
skills disagree with the compiler, the live compiler manifest wins.

## Engineering Standard

- Prefer production fixes over band-aids. Understand the root cause, the
  owning dialect, and the runtime/compiler contract before patching symptoms.
- Before changing compiler IR, state the function and limits of the affected
  dialect layer. SDE owns OpenMP semantics and scheduling intent; CODIR owns
  codelet isolation; ARTS owns abstract orchestration and analyses; ARTS-RT
  owns lowering-ready runtime shape.
- Do not hide correctness behind later cleanup passes, incidental pass order,
  fixture churn, or duplicated local helpers.
- Passes, operations, attributes, and dialect-owned IR metadata must be
  declared through the owning TableGen/ODS files first. C++ code should consume
  generated declarations/accessors instead of adding manual pass/attribute
  surfaces.
- Before adding a helper to a pass, use `carts-check-utils`. Reusable helpers
  belong in the narrowest owning dialect `Utils/` area or an owning analysis
  API; keep helpers pass-local only when they are genuinely one-pass logic.

**Session-start skill gate (blocking).** Before editing any file in
`lib/carts/`, `include/carts/`, or `tools/compile/`, invoke the Skill tool with
`check-utils` (helper placement), `carts-dialect-map` (boundary ownership), and
`carts-attr-consolidation` (when touching `.td` attribute enums or any
attribute string). Before any commit, invoke `carts-simplify` then
`carts-review`. Reconnaissance-first / skill-invocation-never is the failure
mode this gate prevents.

## Development Artifacts

- Put generated diagnostics, stage dumps, benchmark triage bundles, generated
  executables, logs, and one-off command outputs under `.carts/outputs/...`.
- Put multi-step investigation scratch state under
  `.carts/sessions/<YYYYMMDD-HHMMSS>-<topic>/...`.
- Do not commit `.carts/` artifacts unless promoting a reduced case into an
  intentional test fixture.

## Verification

Match verification to the change:

- Skill/doc/config changes: `dekk carts skills generate` plus discovery checks.
- Single pass change: focused `dekk carts lit <test>`, then the owning dialect
  suite.
- Pipeline or shared analysis change: `dekk carts test` and affected e2e cases.
- Runtime or distributed change: rebuild ARTS as needed, run local first, then
  the smallest multinode or distributed benchmark that exercises the path.

<!-- BEGIN SKILLS INVENTORY -->
## Available Skills

### Session lifecycle

| Skill | Description | Path |
| --- | --- | --- |
| `carts-session-start` | Use at the start of any CARTS compiler session before reconnaissance begins. | `carts-plugin/skills/carts-session-start/SKILL.md` |
| `carts-commit` | Use when staging CARTS changes, choosing a commit message scope, committing, pushing, or handing a patch to review. | `carts-plugin/skills/carts-commit/SKILL.md` |
| `carts-review` | Use before committing CARTS changes, during PR review, after substantial compiler/runtime edits, or when checking conventions, missing tests, fixture refreshes, or regression risk. | `carts-plugin/skills/carts-review/SKILL.md` |
| `carts-simplify` | Use as the final simplification gate before committing or finishing CARTS work: reduce patch complexity, remove accidental changes, confirm utility-placement decisions already made by carts-check-utils, and enforce .carts artifact discipline. | `carts-plugin/skills/carts-simplify/SKILL.md` |
| `carts-finishing` | Use when advancing, continuing, or finishing CARTS work; choosing the next fix; asking where a fix belongs; applying regression guards; or referencing the carts-finishing plan. | `carts-plugin/skills/carts-finishing/SKILL.md` |

### Discovery + placement

| Skill | Description | Path |
| --- | --- | --- |
| `carts-find-utils` | Use when looking for an existing CARTS helper before writing one. | `carts-plugin/skills/carts-find-utils/SKILL.md` |
| `check-utils` | Use before adding a CARTS helper, utility file, static helper, or utility-like attribute/string accessor. | `carts-plugin/skills/check-utils/SKILL.md` |
| `refactor-utils` | Use when consolidating duplicate CARTS helpers or extracting reusable pass-local helpers after placement is decided. | `carts-plugin/skills/refactor-utils/SKILL.md` |
| `carts-include-tier` | Use when lib headers gain multiple consumers, adding lib *Utils.h, or finding include *Internal.h. | `carts-plugin/skills/carts-include-tier/SKILL.md` |
| `carts-attr-consolidation` | Use when adding or changing CARTS TableGen attrs, enum attrs, raw dialect attr strings, convert* enum switches, or AttrNames. | `carts-plugin/skills/carts-attr-consolidation/SKILL.md` |
| `carts-dialect-map` | Use when locating CARTS dialect code, tracing op lifecycle, choosing owners, or checking boundary invariants. | `carts-plugin/skills/carts-dialect-map/SKILL.md` |
| `carts-pipeline-map` | Use when inspecting CARTS pipeline stages, pass order, stage tokens, start-from/pipeline ranges, epilogues, stale pipeline docs, or ownership of a transformation stage. | `carts-plugin/skills/carts-pipeline-map/SKILL.md` |

### Building + running

| Skill | Description | Path |
| --- | --- | --- |
| `build` | Use when the user asks to build, compile the project, rebuild CARTS/ARTS/LLVM/Polygeist, or fix build errors. | `carts-plugin/skills/build/SKILL.md` |
| `carts-cli` | Use when asking about CARTS commands, environment setup, wrappers, compile flags, pipeline inspection, examples, benchmarks, generated skills, or how to run project lifecycle tasks. | `carts-plugin/skills/carts-cli/SKILL.md` |
| `test` | Use when the user asks to test, run tests, validate, check, verify changes, run lit, or run a focused CARTS suite. | `carts-plugin/skills/test/SKILL.md` |
| `create-test` | Use when adding a new lit test, creating regression tests, writing multi-stage boundary tests, or choosing CARTS test placement and RUN lines. | `carts-plugin/skills/create-test/SKILL.md` |
| `contract-refresh` | Use when contract tests fail due to legitimate IR changes, after refactoring passes, or when git status shows stale Output/ fixtures. | `carts-plugin/skills/contract-refresh/SKILL.md` |
| `carts-local-examples` | Use when listing, running, fixing, sweeping, or validating CARTS local sample programs, examples runner behavior, or e2e compile-and-run tests. | `carts-plugin/skills/carts-local-examples/SKILL.md` |
| `carts-multinode-examples` | Use when the user asks how to compile or run CARTS examples on multiple nodes, including ARTS configs, launchers, SSH, or Slurm. Use carts-distributed-triage for failures. | `carts-plugin/skills/carts-multinode-examples/SKILL.md` |
| `benchmark` | Use when the user asks to list, build, run, or compare CARTS benchmarks. Use carts-benchmark-triage for failing, timing out, or suspicious benchmark results. | `carts-plugin/skills/benchmark/SKILL.md` |

### Triage

| Skill | Description | Path |
| --- | --- | --- |
| `debug` | Use when starting an unclassified CARTS debug session for crashes, wrong results, hangs, logs, counters, or pipeline artifacts. | `carts-plugin/skills/debug/SKILL.md` |
| `analysis-triage` | Use when behavior depends on pass order, graphs look stale, metadata is inconsistent, or a pass likely needs AnalysisDependencies or narrower invalidation. | `carts-plugin/skills/analysis-triage/SKILL.md` |
| `miscompile-triage` | Use when a program compiles but produces wrong output, checksum mismatches, phase-equivalence failures, or suspicious partitioning/distribution decisions. | `carts-plugin/skills/miscompile-triage/SKILL.md` |
| `runtime-triage` | Use when compilation succeeds but the generated ARTS executable hangs, deadlocks, crashes, stalls, or reports anomalous runtime counters. | `carts-plugin/skills/runtime-triage/SKILL.md` |
| `distributed-triage` | Use when a failure only appears with `--distributed-db`, multiple nodes, SDE/CODIR/ARTS distributed work materialization, or uneven remote work distribution. | `carts-plugin/skills/distributed-triage/SKILL.md` |
| `benchmark-triage` | Use when a benchmark fails, times out, produces wrong checksums, shows suspicious speedups, or needs pass-by-pass/runtime diagnosis. | `carts-plugin/skills/benchmark-triage/SKILL.md` |
| `heuristic-explain` | Use when a benchmark or test has unexpected partitioning, wrong distribution mode, or heuristic drift in ARTS DB/EDT placement decisions. | `carts-plugin/skills/heuristic-explain/SKILL.md` |
| `reproducer` | Use when a large failing program, benchmark, or stage dump needs to become a minimal C, MLIR, or lit reproducer. | `carts-plugin/skills/reproducer/SKILL.md` |
| `stage-diff` | Use when debugging miscompiles, verifying pass correctness, comparing MLIR between pipeline stages, or finding where semantics diverge. | `carts-plugin/skills/stage-diff/SKILL.md` |
| `dialect-trace` | Use when debugging lowering paths, understanding operation placement across SDE/CODIR/ARTS/ARTS-RT, or verifying dialect boundary invariants. | `carts-plugin/skills/dialect-trace/SKILL.md` |
| `runtime-first` | Use after runtime triage shows the compiler must match an ARTS runtime contract for EDTs, DBs, epochs, dependencies, or distributed execution. | `carts-plugin/skills/runtime-first/SKILL.md` |

### Authoring + maintenance

| Skill | Description | Path |
| --- | --- | --- |
| `pass-dev` | Use when creating a new pass, modifying an existing pass, understanding pass architecture, or working on compiler transforms. | `carts-plugin/skills/pass-dev/SKILL.md` |
| `carts-agentic-development` | Use when planning or executing CARTS work with multiple agents, independent tasks, implementation plans, code review checkpoints, or staged compiler/runtime investigations. | `carts-plugin/skills/carts-agentic-development/SKILL.md` |
| `carts-skill-maintenance` | Use when creating, regenerating, validating, or hardening CARTS project skills and agent resources. | `carts-plugin/skills/carts-skill-maintenance/SKILL.md` |

<!-- END SKILLS INVENTORY -->
