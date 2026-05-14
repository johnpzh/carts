---
name: carts-create-test
description: Use when adding a new lit test, creating regression tests, writing multi-stage boundary tests, or choosing CARTS test placement and RUN lines.
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Write, Edit, Agent
argument-hint: [contract <stage> | integration <name> | boundary <stage1> <stage2>]
parameters:
  - name: test_type
    type: str
    gather: "Type of test: 'contract' (single stage), 'boundary' (multi-stage), or 'integration' (end-to-end C)"
  - name: pipeline_stage
    type: str
    gather: "Primary pipeline stage to test (e.g., post-db-refinement, pre-lowering)"
---

# CARTS Test Scaffolding

## Test Types and Directory Placement

| Stage Range | Directory | Dialect |
|-------------|-----------|---------|
| 1-2 (metadata, raise) | `lib/arts/dialect/sde/test/` | SDE |
| 3-16 (core transforms) | `lib/arts/dialect/core/test/` | Core |
| 13-14 (partitioning) | `lib/arts/dialect/core/test/partitioning/safety/` | Core (safety) |
| 17-18 (lowering) | `lib/arts/dialect/rt/test/` | RT |
| Verifiers | `tests/verify/` | Cross-dialect |
| CLI flags | `tests/cli/` | Infrastructure |
| End-to-end | `samples/` | Integration |

## Contract Test Template (Single Stage)

```mlir
// RUN: %carts-compile %s --O3 --arts-config %S/../inputs/arts_1t.cfg \
// RUN:   --pipeline <STAGE> | %FileCheck %s

// Brief description of what this test verifies

module {
  func.func @test_function(...) {
    // Minimal IR that triggers the behavior under test
  }
}

// CHECK-LABEL: func @test_function
// CHECK: expected_operation
// CHECK-NOT: unwanted_operation
```

## Multi-Stage Boundary Template

```mlir
// RUN: %carts-compile %s --O3 --arts-config %S/../inputs/arts_1t.cfg \
// RUN:   --pipeline <STAGE1> | %FileCheck %s --check-prefix=BEFORE
// RUN: %carts-compile %s --O3 --arts-config %S/../inputs/arts_1t.cfg \
// RUN:   --pipeline <STAGE2> | %FileCheck %s --check-prefix=AFTER

module { ... }

// BEFORE-LABEL: func @test_function
// BEFORE: attribute_before_transform

// AFTER-LABEL: func @test_function
// AFTER-NOT: attribute_before_transform
// AFTER: expected_lowered_form
```

## Config Selection Guide

| Scenario | Config File | Workers |
|----------|-------------|---------|
| Isolation / correctness | `arts_1t.cfg` | 1 |
| Basic parallelism | `arts_2t.cfg` | 2 |
| Scale / performance | `arts_64t.cfg` | 64 |
| Multi-node | `arts.cfg` (examples) | varies |

## Naming Convention

```
<benchmark>_<property>_<expected_behavior>.mlir

Examples:
  matmul_reduction_inputs_preserve_full_range.mlir
  seidel_wavefront_keeps_frontier_parallelism.mlir
  rhs4sg_base_block_aligned_worker_slices.mlir
```

## Test Writing Rules

1. **One stage per test** — test the specific transformation, not the whole pipeline
2. **Minimal IR** — smallest module that triggers the behavior
3. **Positive and negative checks** — use CHECK and CHECK-NOT
4. **Label anchors** — always start with `CHECK-LABEL: func @name`
5. **Preserve semantic markers** — keep `arts.dep_pattern`, `arts.partition_mode`,
   `arts.distribution_kind` attributes in test IR
6. **Use shared inputs** when possible — check `tests/inputs/` first

## Instructions

When the user asks to create a test:

1. Determine test type (contract vs boundary vs integration)
2. Identify the pipeline stage(s) to test
3. Choose the correct directory based on the stage/dialect table
4. Select the appropriate config file
5. Create the test file with proper RUN lines and CHECK patterns
6. Run `dekk carts lit <test-file>` to validate
7. If the test needs fixture output, ensure `Output/` dir is created
