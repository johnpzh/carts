# CARTS Shared Utility Catalog

Quick-reference for searching existing utilities before adding new ones.

## Search Cheatsheet

| If you need to... | Check first | File |
|-------------------|-------------|------|
| Check if value is zero | `ValueAnalysis::isZeroConstant()` | `ValueAnalysis.h` |
| Check if value is one | `ValueAnalysis::isOneConstant()` | `ValueAnalysis.h` |
| Check if value is constant | `ValueAnalysis::tryFoldConstantIndex()` | `ValueAnalysis.h` |
| Create zero/one index | `createZeroIndex()`, `createOneIndex()` | `Utils.h` |
| Create arbitrary index constant | `createConstantIndex()` | `Utils.h` |
| Trace value to DB alloc | `DbUtils::traceToDbAlloc()` | `DbUtils.h` |
| Get underlying allocation | `DbUtils::getUnderlyingDb()` | `DbUtils.h` |
| Extract memory access info | `DbUtils::getMemoryAccessInfo()` | `DbUtils.h` |
| Get memref from load/store | `DbUtils::getAccessedMemref()` | `DbUtils.h` |
| Get indices from load/store | `DbUtils::getMemoryAccessIndices()` | `DbUtils.h` |
| Check if op is pure arithmetic | `isSideEffectFreeArithmeticLikeOp()` | `Utils.h` |
| Check loop invariance | `isLoopInvariant()` | `LoopInvarianceUtils.h` |
| Safe to hoist div/rem? | `isSafeToHoistDivRem()` | `LoopInvarianceUtils.h` |
| All operands defined outside? | `allOperandsDefinedOutside()` | `LoopInvarianceUtils.h` |
| All operands dominate point? | `allOperandsDominate()` | `LoopInvarianceUtils.h` |
| Check dominance w/ ancestors | `dominatesOrInAncestor()` | `Utils.h` |
| Get static trip count | `getStaticTripCount()` | `LoopUtils.h` |
| Is innermost loop? | `isInnermostLoop()` | `LoopUtils.h` |
| Is worker loop? | `isWorkerLoop()` | `LoopUtils.h` |
| Is loop IV? | `isLoopInductionVar()` | `LoopUtils.h` |
| Get loop depth | `getLoopDepth()` | `LoopUtils.h` |
| Find nearest enclosing loop | `findNearestLoop()` | `LoopUtils.h` |
| Is inside epoch? | `isInsideEpoch()` | `EdtUtils.h` |
| Get single top-level for | `getSingleTopLevelFor()` | `EdtUtils.h` |
| Classify EDT captures | `EdtEnvManager` class | `EdtUtils.h` |
| Combine access modes | `combineAccessModes()` | `Utils.h` |
| Is writer mode? | `DbUtils::isWriterMode()` | `DbUtils.h` |
| Get element byte size | `getElementTypeByteSize()` | `Utils.h` |
| Check block layout | `usesBlockLayout()` | `PartitionPredicates.h` |
| Supports halo? | `supportsHaloExtension()` | `PartitionPredicates.h` |
| Is stencil family? | `isStencilFamilyDepPattern()` | `StencilAttributes.h` |
| Requires halo exchange? | `supportsHaloExtension()` | `PartitionPredicates.h` |
| Get lowering contract | `getLoweringContract()` | `LoweringContractUtils.h` |
| Resolve effective contract | `resolveEffectiveContract()` | `LoweringContractUtils.h` |
| Mark op for deferred removal | `RemovalUtils::markForRemoval()` | `RemovalUtils.h` |
| Replace uses (domination-aware) | `replaceUses()` | `Utils.h` |
| Transfer attributes | `transferAttributes()` | `Utils.h` |
| Is inside OMP region? | `isInsideOmpRegion()` | `Utils.h` |
| Extract block size from hint | `extractBlockSizeFromHint()` | `DbUtils.h` |
| Is aligned to block? | `isAlignedToBlock()` | `BlockedAccessUtils.h` |
| Known non-negative? | `isKnownNonNegative()` | `BlockedAccessUtils.h` |
| Match loop-invariant addend | `matchLoopInvariantAddend()` | `BlockedAccessUtils.h` |
| Clamp dep indices | `clampDepIndices()` | `Utils.h` |
| Simplify IR (CSE) | `simplifyIR()` | `Utils.h` |

## Attribute Constants

**ALWAYS use these — NEVER hardcode strings:**

```cpp
// Operation attributes
AttrNames::Operation::ArtsId           // "arts.id"
AttrNames::Operation::Workers          // "arts.workers"
AttrNames::Operation::DepPattern       // "arts.dep_pattern"
AttrNames::Operation::PartitionMode    // "arts.partition_mode"
// ... see OperationAttributes.h for full list

// Stencil attributes
AttrNames::Operation::Stencil::StencilCenterOffset  // "arts.stencil.center_offset"
AttrNames::Operation::Stencil::ElementStride         // "arts.stencil.element_stride"
AttrNames::Operation::Stencil::FootprintMinOffsets   // "arts.stencil.footprint_min"
AttrNames::Operation::Stencil::FootprintMaxOffsets   // "arts.stencil.footprint_max"
// ... see StencilAttributes.h for full list
```

## File Paths

```
include/carts/utils/Utils.h
include/carts/utils/LoopUtils.h
include/carts/utils/LoopInvarianceUtils.h
include/carts/utils/RemovalUtils.h
include/carts/utils/LocationMetadata.h
include/carts/utils/RuntimeConfig.h
include/carts/utils/OperationAttributes.h
include/carts/utils/StencilAttributes.h
include/carts/utils/ValueAnalysis.h
include/carts/dialect/arts/Utils/DbUtils.h
include/carts/dialect/arts/Utils/EdtUtils.h
include/carts/dialect/arts/Utils/BlockedAccessUtils.h
include/carts/dialect/arts/Utils/LoweringContractUtils.h
include/carts/dialect/arts/Utils/PartitionPredicates.h
include/carts/dialect/arts-rt/Utils/IdRegistry.h
```
