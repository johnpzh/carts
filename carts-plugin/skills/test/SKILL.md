---
name: carts-test
description: Use when the user asks to test, run tests, validate, check, verify changes, run lit, or run a focused CARTS suite.
user-invocable: true
allowed-tools: Bash, Read, Grep, Glob
argument-hint: [<test-file.mlir> | --suite contracts | --suite benchmarks | -v]
---

# CARTS Test

Run the CARTS test suite. Run `dekk carts test --help` and `dekk carts lit --help` for latest options.

## Common Commands

```bash
dekk carts test                                     # Run contract tests
dekk carts test --suite all -v                      # All tests, verbose
dekk carts test --suite benchmarks                  # Benchmark harness pytest suite
dekk carts lit lib/carts/dialect/arts/test/my_test.mlir  # Single test
dekk carts lit -v lib/carts/dialect/sde/test/           # Directory, verbose
dekk carts lit -- --filter=pattern lib/carts/dialect/    # Filter by pattern
```

## Test Organization

- `lib/carts/dialect/sde/test/` — SDE dialect lit tests (stages 1-2)
- `lib/carts/dialect/arts/test/` — ARTS dialect lit tests
- `lib/carts/dialect/arts-rt/test/` — ARTS-RT dialect lit tests
- `tests/cli/` — CLI flag tests
- `tests/verify/` — Cross-dialect verifier tests
- `samples/` — Integration tests (end-to-end compile + run)
- `external/carts-benchmarks/tests/` — Benchmark harness pytest tests, run through `dekk carts test --suite benchmarks`

## Writing New Tests

```mlir
// RUN: %carts-compile %s --O3 --arts-config %S/../examples/arts.cfg --pipeline=<stage> | %FileCheck %s
// CHECK: expected_output
module { ... }
```

Key patterns:
- Use `%carts-compile` substitution (not raw binary paths)
- Use `%S` for test source directory (access sibling files like `arts.cfg`)
- Use `%FileCheck` for output verification
- Test one pipeline stage per test file for isolation
- Cover both positive (CHECK) and negative (CHECK-NOT) assertions

## Debugging Test Failures

```bash
# Run with verbose output to see FileCheck diff
dekk carts lit -v lib/carts/dialect/arts/test/failing_test.mlir

# Run the command manually to inspect full output
dekk carts compile lib/carts/dialect/arts/test/failing_test.mlir --pipeline=<stage>
```

## Instructions

When the user asks to test:
1. Run `dekk carts test $ARGUMENTS` or `dekk carts lit $ARGUMENTS`
2. Report pass/fail counts and failure details
3. For failures, read the failing test and analyze the issue
4. If the test needs updating after a pass change, update the CHECK lines
