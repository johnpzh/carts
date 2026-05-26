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
