---
name: carts-check-utils
description: Use before adding helper or utility functions to pass files, when writing static helpers, when deciding Utils/Support placement, or when reviewing PRs for duplicated functions or pass-local utility sprawl.
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Agent
argument-hint: <function-name-or-description>
parameters:
  - name: function_name
    type: str
    gather: "Name or short description of the function you want to add (e.g., 'isOneLikeValue', 'check if value is constant zero', 'hoist loop invariant op')"
---

# CARTS Utility Ownership Check

## Purpose

**MANDATORY pre-flight check** before adding any new static helper or utility
function to a pass file. This skill searches the shared utility surface,
selects the narrowest owning dialect utility home, and prevents duplicating
existing functionality.

## The Problem

The CARTS codebase has 250+ shared utility functions across 13 utility files.
Historical audit found:
- **Exact duplicate functions** across pass files (e.g., `isOneLikeValue` in
  2 locations, `hasWorkAfterInParentBlock` in 2, `isUndefLikeOp` in 3)
- **60+ static helpers** in pass files that belong in shared utilities
- **180+ static functions** in pass files, many reimplementing existing utils

## Pass-Local Helper Rule

A helper may stay `static` in a pass file only when all are true:

- It is used by exactly one pass implementation.
- It does not express a dialect invariant.
- It does not duplicate an existing helper by name or behavior.
- It is not needed by a verifier, analysis, conversion, or sibling pass.
- Extracting it would make the code harder to read.

If any item is false, move it to the owning dialect `Utils/`, an owning
analysis API, or a pass-area support file. Do not create a new "Utils" helper
inside a pass just because it is convenient during the first implementation.

## Target Dialect Utility Homes

New utility surfaces should follow the target layout:

| Helper kind | Preferred home |
|-------------|----------------|
| SDE source semantics, memref access maps, SDE patterns, MU/CU/SU planning | `include/carts/dialect/sde/Utils` + `lib/carts/dialect/sde/Utils` |
| CODIR codelet isolation, dep/param ABI, token-local views | `include/carts/dialect/codir/Utils` + `lib/carts/dialect/codir/Utils` |
| ARTS DB/EDT/epoch object mechanics, dependency slots, placement/resource helpers | `include/carts/dialect/arts/Utils` + `lib/carts/dialect/arts/Utils` |
| ARTS-RT runtime ABI packing, depv layout, runtime-call/pointer helpers | `include/carts/dialect/arts-rt/Utils` + `lib/carts/dialect/arts-rt/Utils` |
| Cross-dialect compiler helpers with no dialect semantics | `include/carts/support` + `lib/carts/support` |
| Current-tree compatibility helpers | nearest existing `include/carts/utils`, `lib/carts/utils`, or pass-area `*Support` file |

If a target `Utils/` folder exists only as a skeleton, do not wire it into
CMake for an empty helper. Add the real utility in the same patch that first
uses it, with focused tests for the owning pass.

## Shared Utility Locations (Search These FIRST)

### Core Utilities (`include/carts/utils/` + `lib/carts/utils/`)

| File | Category | Key Functions |
|------|----------|---------------|
| `Utils.h` | General | `createZeroIndex`, `createOneIndex`, `createConstantIndex`, `dominatesOrInAncestor`, `replaceUses`, `isSideEffectFreeArithmeticLikeOp`, `combineAccessModes`, `isArtsRuntimeQuery`, `getElementTypeByteSize` |
| `DbUtils.h` | DataBlock | `traceToDbAlloc`, `getUnderlyingDb`, `getSizesFromDb`, `getAccessedMemref`, `getMemoryAccessInfo`, `getMemoryAccessIndices`, `isWriterMode`, `collectReachableMemoryOps` |
| `EdtUtils.h` | EDT | `EdtEnvManager`, `isInsideEpoch`, `getSingleTopLevelFor`, `classifyEdtArgAccesses`, `collectEdtPackedValues`, `spliceBodyBeforeTerminator`, `fuseConsecutivePairs` |
| `LoopUtils.h` | Loops | `isWorkerLoop`, `isInnermostLoop`, `isLoopInductionVar`, `haveCompatibleBounds`, `findNearestLoop`, `getStaticTripCount`, `getLoopDepth`, `containsLoop` |
| `LoopInvarianceUtils.h` | Invariance | `isLoopInvariant`, `isSafeToHoistDivRem`, `isDefinedOutside`, `allOperandsDefinedOutside`, `allOperandsDominate` |
| `BlockedAccessUtils.h` | Block Patterns | `matchLoopInvariantAddend`, `extractLocalFromBlockBase`, `isKnownNonNegative`, `isAlignedToBlock`, `loopWindowFitsSingleBlock` |
| `RemovalUtils.h` | Op Removal | `markForRemoval`, `removeAllMarked`, `replaceWithUndef` |
| `LoweringContractUtils.h` | Contracts | `getLoweringContract`, `resolveEffectiveContract`, `upsertLoweringContract`, `mergeLoweringContractInfo` |
| `PartitionPredicates.h` | Partition | `usesBlockLayout`, `supportsHaloExtension`, `usesElementLayout` |
| `PatternSemantics.h` | Patterns | `isStencilFamily`, `isUniformFamily`, `requiresHaloExchange`, `inferDistributionPattern` |
| `OperationAttributes.h` | Attr Names | `AttrNames::Operation::*` — ALL attribute string constants |
| `StencilAttributes.h` | Stencil Attrs | `AttrNames::Operation::Stencil::*` + getter/setter helpers |

### Analysis Utilities (`include/carts/dialect/arts/Analysis/`)

| File | Category | Key Functions |
|------|----------|---------------|
| `ValueAnalysis.h` | Value | `isZeroConstant`, `isOneConstant`, `tryFoldConstantIndex`, `isDerivedFromPtr`, `getUnderlyingValue` |
| `LoopNode.h` | Loop Tree | Loop nesting analysis, IV extraction |
| `DbPatternMatchers.h` | DB Patterns | Stencil/matmul/symmetric pattern detection |

### Pass Support Files (public namespaces)

| File | Category |
|------|----------|
| `BlockLoopStripMiningSupport.cpp` | Strip-mining predicates |
| `DbPartitioningSupport.cpp` | Partition planning predicates |
| `DataPtrHoistingSupport.cpp` | Pointer hoisting patterns |
| `EpochOptSupport.cpp` | CPS chain predicates |

### Target Skeletons

Also search target locations when working in the new layout:

- `include/carts/dialect/*/Utils`
- `lib/carts/dialect/*/Utils`
- `include/carts/support`
- `lib/carts/support`

## Search Strategy

When checking for an existing function, search by **behavior** not just name:

1. **Exact name match**: `Grep` for the function name across `include/carts/` and `lib/carts/utils/`
2. **Semantic match**: Search for keywords describing what the function does:
   - "is zero" / "isZero" / "zero constant" for zero-checking
   - "hoist" / "invariant" / "loop invariant" for hoisting helpers
   - "memref" / "access" / "load" / "store" for memory access helpers
   - "trace" / "underlying" / "alloc" for provenance tracking
3. **Pattern match**: Check if the function's logic is a special case of an existing utility
4. **Support file check**: Look in `*Support.cpp` files for the relevant pass area

## Known Duplicates (DO NOT RE-ADD)

These functions already exist in shared locations — NEVER add local copies:

| Function | Canonical Location |
|----------|--------------------|
| `isOneLikeValue` / `isOneLike` | `ValueAnalysis::isOneConstant` + pattern in `DbAnalysis::hasSingleSize` |
| `hasWorkAfterInParentBlock` | Duplicated 2x — needs extraction to `Utils.h` |
| `isPureOp` | Duplicated 2x — use `isSideEffectFreeArithmeticLikeOp` from `Utils.h` |
| `sortStoresInProgramOrder` | Duplicated 2x — needs extraction to `Utils.h` |
| `findHoistTarget` | Duplicated 2x — needs consolidation in `LoopInvarianceUtils.h` |
| `getMemoryAccessInfo` | `DbUtils::getMemoryAccessInfo` is canonical |
| `createZeroIndex` / `createOneIndex` | `Utils.h` — NEVER redefine |
| `isLoopInvariant` | `LoopInvarianceUtils.h` — NEVER redefine |
| `getStaticTripCount` | `LoopUtils.h` — NEVER redefine |
| `isUndefLikeOp` | Duplicated 3x (`EdtLowering.cpp`, `EdtUtils.cpp`, `EpochOptCpsChain.cpp`) — extract to `Utils.h` |

## Hardcoded String Rules

**NEVER** hardcode attribute name strings. Always use:
- `AttrNames::Operation::*` from `OperationAttributes.h`
- `AttrNames::Operation::Stencil::*` from `StencilAttributes.h`
- For new attributes, ADD them to these files first

Known violations to fix:
- `BlockLoopStripMiningSupport.cpp:18` — `"arts.block_loop_strip_mining.generated"`
- `RaiseToLinalg.cpp:240-295` — hardcoded `"parallel"`, `"reduction"`, etc.
- `EdtLowering.cpp:123` — hardcoded op names `"llvm.mlir.undef"` etc.

## Instructions

When asked to check if a utility function already exists:

1. **Parse the request** — understand what the function does, not just its name
2. **Search shared utilities** — run parallel Grep searches across:
   - `include/carts/dialect/*/Utils` and `lib/carts/dialect/*/Utils`
   - `include/carts/support` and `lib/carts/support` if present
   - `include/carts/utils/` (headers)
   - `lib/carts/utils/` (implementations)
   - `include/carts/dialect/arts/Analysis/` (analysis headers)
   - `lib/carts/dialect/arts/Analysis/` (analysis implementations)
3. **Search pass support files** — check `*Support.cpp` and `*Support.h` files
4. **Search for duplicates** — look for the function name in all of `lib/carts/`
5. **Report findings**:
   - If found: show the canonical location, signature, and how to use it
   - If similar: show the closest match and explain the difference
   - If not found: confirm whether it is truly pass-local. If not, suggest the
     best location:
     - Generic value/type helpers → common CARTS support or current `Utils.h`
     - DB-specific → `DbUtils.h`
     - EDT-specific → `EdtUtils.h`
     - Loop-specific → `LoopUtils.h`
     - Loop invariance → `LoopInvarianceUtils.h`
     - Block access patterns → `BlockedAccessUtils.h`
     - Pattern classification → `PatternSemantics.h`
     - SDE-only semantic/access planning → SDE `Utils/`
     - CODIR codelet/token-local ABI → CODIR `Utils/`
     - ARTS DB/EDT/epoch mechanics → ARTS `Utils/`
     - ARTS-RT runtime ABI mechanics → ARTS-RT `Utils/`
     - Pass-specific but shared within pass area → `*Support.cpp`
     - Truly pass-specific → keep as static in the pass file
6. **Require a placement decision** — state one of:
   - "Use existing helper at `<path>`."
   - "Extract to `<dialect>/Utils` because `<reason>`."
   - "Keep pass-local because all pass-local rule items are true."
7. **Check attribute strings** — if the function uses attribute names, verify it
   uses `AttrNames::` constants, not hardcoded strings
