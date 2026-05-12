---
name: carts-pass-dev
description: Use when creating a new pass, modifying an existing pass, understanding pass architecture, or working on compiler transforms.
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Write, Edit, Agent
argument-hint: [<pass-name> | new <pass-name>]
parameters:
  - name: pass_name
    type: str
    gather: "Name of the pass to create or modify (e.g., DbFoo, EdtBar)"
  - name: pass_description
    type: str
    gather: "What should the pass do? Describe the transformation or optimization"
  - name: target_dialect
    type: str
    gather: "Which dialect does this pass operate on? (arts, affine, scf, memref)"
---

# CARTS Pass Development

## Architecture Conventions (MANDATORY)

**Analysis interface** — always use analysis classes, NEVER access graphs directly:
```cpp
AM->getDbAnalysis().getOrCreateGraph(func)     // DB analysis
AM->getEdtAnalysis().getOrCreateEdtGraph(func) // EDT analysis
AM->getEdtAnalysis().getEdtNode(edt)           // Node lookup
AM->getDbAnalysis().getDbAcquireNode(acquire)  // Node lookup
```

**Attribute names** — NEVER hardcode strings. Use centralized names from:
- `include/arts/utils/OperationAttributes.h` (`AttrNames::Operation`)
- `include/arts/utils/StencilAttributes.h` (`AttrNames::Operation::Stencil`)

**Naming** — DB passes: `Db` prefix. EDT passes: `Edt` prefix. LLVM style: 2-space indent, CamelCase types, camelCase variables.

**Utility reuse** — before adding ANY new static helper function, run
`/check-utils <function-name>` to verify it doesn't already exist in
`include/arts/utils/` or `include/arts/dialect/core/Analysis/`. The codebase has 250+
shared utilities across 13 files. See `/refactor-utils` for the full catalog.

## Key Source Locations

- Core transforms: `lib/arts/dialect/core/Transforms/`
- ArtsToRt conversion: `lib/arts/dialect/core/Conversion/ArtsToRt/`
- SDE transforms: `lib/arts/dialect/sde/Transforms/`
- LLVM conversion: `lib/arts/dialect/core/Conversion/ArtsToLLVM/`
- Analysis: `lib/arts/dialect/core/Analysis/`
- Shared transforms: `lib/arts/dialect/core/Transforms/` (db/, dep/, edt/, loop/, kernel/)
- Pipeline setup: `tools/compile/Compile.cpp`
- Pass declarations: `include/arts/passes/Passes.h` and `Passes.td`

## Creating a New Pass

1. Create source in the appropriate dialect directory (`lib/arts/dialect/{core,rt,sde}/Transforms/`)
2. Add declaration in `include/arts/passes/Passes.h`
3. Register in pipeline at appropriate stage in `tools/compile/Compile.cpp`
4. Add lit test in the co-located `test/` directory (`lib/arts/dialect/{core,rt,sde}/test/`)
5. `dekk carts format` then `dekk carts test --suite contracts`

## Thread Safety

All passes must be thread-safe — no global/static mutable state. Use function-scoped graph access.

## Instructions

When the user asks to develop a pass:
1. Find similar existing passes for reference
2. Guide through the creation steps above
3. Ensure conventions are followed
4. Create test, build, and verify
