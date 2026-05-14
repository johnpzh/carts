---
name: carts-check-utils
description: Use before adding helper or utility functions to pass files, when writing static helpers, or when reviewing PRs for duplicated functions.
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Agent
argument-hint: <function-name-or-description>
parameters:
  - name: function_name
    type: str
    gather: "Name or short description of the function you want to add (e.g., 'isOneLikeValue', 'check if value is constant zero', 'hoist loop invariant op')"
---

# CARTS Utility Duplication Check

## Purpose

**MANDATORY pre-flight check** before adding any new static helper or utility
function to a pass file. This skill searches the entire shared utility surface
to prevent duplicating existing functionality.

## The Problem

The CARTS codebase has 250+ shared utility functions across 13 utility files.
Historical audit found:
- **Exact duplicate functions** across pass files (e.g., `isOneLikeValue` in
  2 locations, `hasWorkAfterInParentBlock` in 2, `isUndefLikeOp` in 3)
- **60+ static helpers** in pass files that belong in shared utilities
- **180+ static functions** in pass files, many reimplementing existing utils

## Shared Utility Locations (Search These FIRST)

### Core Utilities (`include/arts/utils/` + `lib/arts/utils/`)

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

### Analysis Utilities (`include/arts/dialect/core/Analysis/`)

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

## Search Strategy

When checking for an existing function, search by **behavior** not just name:

1. **Exact name match**: `Grep` for the function name across `include/arts/` and `lib/arts/utils/`
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
   - `include/arts/utils/` (headers)
   - `lib/arts/utils/` (implementations)
   - `include/arts/dialect/core/Analysis/` (analysis headers)
   - `lib/arts/dialect/core/Analysis/` (analysis implementations)
3. **Search pass support files** — check `*Support.cpp` and `*Support.h` files
4. **Search for duplicates** — look for the function name in all of `lib/arts/`
5. **Report findings**:
   - If found: show the canonical location, signature, and how to use it
   - If similar: show the closest match and explain the difference
   - If not found: confirm it's safe to add, and suggest the best location:
     - Generic value/type helpers → `Utils.h`
     - DB-specific → `DbUtils.h`
     - EDT-specific → `EdtUtils.h`
     - Loop-specific → `LoopUtils.h`
     - Loop invariance → `LoopInvarianceUtils.h`
     - Block access patterns → `BlockedAccessUtils.h`
     - Pattern classification → `PatternSemantics.h`
     - Pass-specific but shared within pass area → `*Support.cpp`
     - Truly pass-specific → keep as static in the pass file
6. **Check attribute strings** — if the function uses attribute names, verify it
   uses `AttrNames::` constants, not hardcoded strings
