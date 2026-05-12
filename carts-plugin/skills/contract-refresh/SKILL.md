---
name: carts-contract-refresh
description: Use when contract tests fail due to legitimate IR changes, after refactoring passes, or when git status shows stale Output/ fixtures.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [<test-pattern> | --all | --changed]
---

# CARTS Contract Fixture Refresh

## Purpose

After any IR-affecting pass change, contract test fixtures in co-located
`test/Output/` directories become stale. This skill automates the regeneration,
validation, and commit of updated fixture snapshots.

**This is the #1 most repetitive task** — ~5.5% of all commits are fixture
refreshes, often touching 10-30 `.mlir` files in a single batch.

## When to Use

- Contract tests fail after a legitimate pass change (not a bug)
- `git status` shows many modified files in `*/test/Output/`
- After adding/modifying/reordering passes in the pipeline
- After `dekk carts test` reveals stale FileCheck expectations

## Workflow

### Phase 1: Identify Stale Fixtures

```bash
# See which fixture files changed
git diff --name-only -- 'lib/arts/dialect/*/test/Output/' 'tests/*/Output/'

# Run full test suite to see failures
dekk carts test 2>&1 | grep -E "FAIL:|XFAIL:" | head -30

# Run specific subset
dekk carts lit lib/arts/dialect/<dialect>/test/
```

### Phase 2: Regenerate Fixtures

For tests using `CARTS_COMPILE_WORKDIR`:
```bash
# Single test regeneration
dekk carts lit lib/arts/dialect/<dialect>/test/<test>.mlir

# The workdir output is automatically captured in Output/<test>.mlir.tmp.compile/
```

For tests with inline CHECK patterns (no Output/ directory):
```bash
# Dump the actual output and compare
dekk carts compile lib/arts/dialect/<dialect>/test/<test>.mlir --pipeline=<stage> > /tmp/actual.mlir
# Update CHECK lines in the test file to match actual output
```

### Phase 3: Validate

```bash
# Re-run affected tests to confirm they pass
dekk carts lit lib/arts/dialect/<dialect>/test/

# Run full suite to catch cascading failures
dekk carts test
```

### Phase 4: Review and Commit

```bash
# Review changes are legitimate (not hiding bugs)
git diff -- 'lib/arts/dialect/*/test/Output/' 'tests/*/Output/' | head -200

# Stage only fixture files
git add 'lib/arts/dialect/*/test/Output/' 'tests/*/Output/'
```

## Key Directories

```
lib/arts/dialect/core/test/Output/                  # Core dialect fixtures
lib/arts/dialect/rt/test/Output/                    # RT dialect fixtures
lib/arts/dialect/sde/test/Output/                   # SDE dialect fixtures
lib/arts/dialect/core/test/partitioning/safety/Output/ # Partition safety fixtures
tests/verify/Output/                                # Verifier fixtures
tests/cli/Output/                                   # CLI fixtures
```

## Config Files (for test regeneration)

| Config | When to Use |
|--------|-------------|
| `tests/inputs/arts_1t.cfg` | Single-worker isolation |
| `tests/inputs/arts_2t.cfg` | Minimal parallelism |
| `tests/inputs/arts_64t.cfg` | Large-scale validation |
| `samples/arts.cfg` | Integration tests |

## Common Patterns

### Batch refresh after a pass change
```bash
dekk carts test 2>&1 | tee /tmp/test-results.txt
grep "FAIL:" /tmp/test-results.txt | wc -l  # Count failures
dekk carts test  # Regenerate all (fixtures auto-update on rerun)
git diff --stat -- 'lib/arts/dialect/*/test/Output/' 'tests/*/Output/'  # Review scope
```

### Single-stage fixture refresh
```bash
# Only refresh fixtures for a specific pipeline stage
dekk carts lit lib/arts/dialect/ -- --filter="<stage-name>"
```

## Instructions

When the user asks to refresh contract fixtures:

1. Run `dekk carts test` to identify all failures
2. Classify failures: legitimate IR change vs actual bug
3. For legitimate changes, re-run tests to regenerate fixtures
4. Validate by running the full suite again
5. Show `git diff --stat` for review
6. Stage and describe changes for commit (do NOT auto-commit)
