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

## Essential Commands

- `dekk carts doctor` - validate the toolchain and environment.
- `dekk carts build` - build the CARTS compiler.
- `dekk carts build --arts` - rebuild ARTS runtime.
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

CARTS has three project dialect layers:

- SDE (`arts_sde`) - semantic decomposition, tensor/state scheduling, codelets.
- Core ARTS (`arts`) - EDT, DB, epoch, loop, partitioning, and analysis IR.
- Runtime (`arts_rt`) - flat runtime-facing bridge before LLVM lowering.

The canonical pipeline is defined in `tools/compile/Compile.cpp`. When docs or
skills disagree with the compiler, the live compiler manifest wins.

## Engineering Standard

- Prefer production fixes over band-aids. Understand the root cause, the
  owning dialect, and the runtime/compiler contract before patching symptoms.
- Before changing compiler IR, state the function and limits of the affected
  dialect layer. SDE owns OpenMP semantics and scheduling intent; Core owns
  ARTS orchestration and analyses; RT owns lowering-ready runtime shape.
- Do not hide correctness behind later cleanup passes, incidental pass order,
  fixture churn, or duplicated local helpers.

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
