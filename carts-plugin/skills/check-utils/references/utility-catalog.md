# CARTS Shared Utility Catalog

Quick-reference for searching existing utilities before adding new helpers.
Use this with `carts-check-utils`; the live compiler still wins when docs
disagree.

## Canonical Categories

| Category | Canonical owner | Examples |
|----------|-----------------|----------|
| Dialect-neutral value constants/folding | `include/carts/utils/ValueAnalysis.h` | `isZeroConstant`, `isOneConstant`, `tryFoldConstantIndex`, `sameValue`, `dependsOn`, `stripMemrefViewOps` |
| ARTS DB/EDT/runtime-query value folding and provenance | `include/carts/dialect/arts/Utils/ValueAnalysisUtils.h` | `tryFoldConstantIndex`, `getUnderlyingValue`, `isDerivedFromPtr` |
| ARTS-RT depv/DB pointer value folding and provenance | `include/carts/dialect/arts-rt/Utils/RtDbUtils.h` | `RtDbUtils::tryFoldConstantIndex`, `RtDbUtils::getUnderlyingValue`, `RtDbUtils::isDerivedFromPtr` |
| Index builders/general IR helpers | `include/carts/utils/Utils.h` | `createConstantIndex`, `createZeroIndex`, `createOneIndex`, `dominatesOrInAncestor`, `replaceInRegion` |
| Loop shape and IV helpers | `include/carts/utils/LoopUtils.h` | `isLoopInductionVar`, `getStaticTripCount`, `findNearestLoop`, `getLoopDepth` |
| ARTS/ARTS-RT loop invariance and hoisting | `include/carts/dialect/arts/Utils/LoopInvarianceUtils.h` | `isLoopInvariant`, `findHoistTarget`, `allOperandsDominate`, `isSafeDivRemToHoist` |
| Deferred removal | `include/carts/utils/RemovalUtils.h` | `markForRemoval`, `removeAllMarked`, `replaceWithUndef` |
| SDE access/planning facts | `include/carts/dialect/sde/Analysis` or `include/carts/dialect/sde/Utils` | `AffineAccessUtils`, `StructuredOpAnalysis`, `SDECostModel` |
| CODIR codelet ABI | `include/carts/dialect/codir/Utils` | `CodeletABIUtils` |
| CODIR boundary proof logic | boundary conversion helper | `SdeToCodir/TaskDepSliceUtils.*` |
| ARTS DB mechanics | `include/carts/dialect/arts/Utils/DbUtils.h` | `traceToDbAlloc`, `getUnderlyingDb`, `getMemoryAccessInfo`, `isWriterMode` |
| ARTS EDT mechanics | `include/carts/dialect/arts/Utils/EdtUtils.h` | `EdtEnvManager`, `isInsideEpoch`, `classifyEdtArgAccesses` |
| ARTS partition/block predicates and runtime topology | `include/carts/dialect/arts/Utils` | `BlockedAccessUtils`, `PartitionPredicates`, `LoweringContractUtils`, `RuntimeConfig` |
| ARTS graph/ownership analysis | `include/carts/dialect/arts/Analysis` | `DbGraph`, `EdtGraph`, `LoopNode`, `DbDistributedEligibility` |
| ARTS-RT runtime ABI | `include/carts/dialect/arts-rt/Utils` | `IdRegistry`, `RuntimeCallUtils`, `RtDbUtils` |

Do not create new files such as `LoopIVUtils`, `DbHelperUtils`, or
`ValueHelpers` unless the category is genuinely new and the patch explains why
the existing owner is wrong.

Broad type/value predicates belong under the shared value or IR helper owner,
not loop-specific utilities.

## Search Cheatsheet

| If you need to... | Check first | File |
|-------------------|-------------|------|
| Check if value is zero | `ValueAnalysis::isZeroConstant()` | `ValueAnalysis.h` |
| Check if value is one | `ValueAnalysis::isOneConstant()` | `ValueAnalysis.h` |
| Check if value is one-like after canonicalization patterns | `ValueAnalysis::isOneLikeValue()` | `ValueAnalysis.h` |
| Check if value is constant | `ValueAnalysis::tryFoldConstantIndex()` | `ValueAnalysis.h` |
| Strip casts or view-like memref wrappers | `ValueAnalysis::{stripNumericCasts,stripMemrefViewOps}` | `ValueAnalysis.h` |
| Compare values or value ranges | `ValueAnalysis::{sameValue,areValuesEquivalent,areValueRangesEquivalent}` | `ValueAnalysis.h` |
| Create zero/one index | `createZeroIndex()`, `createOneIndex()` | `Utils.h` |
| Create arbitrary index constant | `createConstantIndex()` | `Utils.h` |
| Check dominance with ancestor regions | `dominatesOrInAncestor()` | `Utils.h` |
| Replace uses or values in a region | `replaceInRegion()` | `Utils.h` |
| Detect trailing work in a block | `hasWorkAfterInParentBlock()` | `Utils.h` |
| Detect undef-like ops | `isUndefLikeOp()` | `Utils.h` |
| Check if op is side-effect-free arithmetic-like | `isSideEffectFreeArithmeticLikeOp()` | `Utils.h` |
| Check ARTS/ARTS-RT loop invariance | `isLoopInvariant()` | `dialect/arts/Utils/LoopInvarianceUtils.h` |
| Find ARTS/ARTS-RT loop hoist target | `findHoistTarget()` | `dialect/arts/Utils/LoopInvarianceUtils.h` |
| Check div/rem hoist safety | `isSafeDivRemToHoist()` | `dialect/arts/Utils/LoopInvarianceUtils.h` |
| All operands defined outside? | `allOperandsDefinedOutside()` | `dialect/arts/Utils/LoopInvarianceUtils.h` |
| All operands dominate point? | `allOperandsDominate()` | `dialect/arts/Utils/LoopInvarianceUtils.h` |
| Get static trip count | `getStaticTripCount()` | `LoopUtils.h` |
| Is innermost loop? | `isInnermostLoop()` | `LoopUtils.h` |
| Is loop IV? | `isLoopInductionVar()` | `LoopUtils.h` |
| Get loop depth | `getLoopDepth()` | `LoopUtils.h` |
| Find nearest enclosing loop | `findNearestLoop()` | `LoopUtils.h` |
| Trace value to DB alloc | `DbUtils::traceToDbAlloc()` | `DbUtils.h` |
| Get underlying DB allocation | `DbUtils::getUnderlyingDb()` | `DbUtils.h` |
| Extract DB memory access info | `DbUtils::getMemoryAccessInfo()` | `DbUtils.h` |
| Get memref from load/store | `DbUtils::getAccessedMemref()` | `DbUtils.h` |
| Get indices from load/store | `DbUtils::getMemoryAccessIndices()` | `DbUtils.h` |
| Is inside epoch? | `EdtUtils::isInsideEpoch()` | `EdtUtils.h` |
| Classify EDT captures | `EdtEnvManager` / `EdtUtils` | `EdtUtils.h` |
| Combine access modes | `combineAccessModes()` | `Utils.h` |
| Check writer mode | `DbUtils::isWriterMode()` | `DbUtils.h` |
| Get element byte size | `getElementTypeByteSize()` | `Utils.h` |
| Check block layout | `usesBlockLayout()` | `PartitionPredicates.h` |
| Supports halo? | `supportsHaloExtension()` | `PartitionPredicates.h` |
| Is stencil family? | `isStencilFamilyDepPattern()` | `StencilAttributes.h` |
| Get lowering contract | `getLoweringContract()` | `LoweringContractUtils.h` |
| Resolve effective contract | `resolveEffectiveContract()` | `LoweringContractUtils.h` |
| Mark op for deferred removal | `RemovalUtils::markForRemoval()` | `RemovalUtils.h` |
| Clamp dep indices | `clampDepIndices()` | `Utils.h` |

## Loop And IV Placement Rules

- Generic loop IV, nearest-loop, trip-count, and loop-depth helpers live in
  `LoopUtils.h`.
- ARTS/ARTS-RT loop-invariant and hoisting-safety helpers live in
  `include/carts/dialect/arts/Utils/LoopInvarianceUtils.h`.
- SDE structured-access interpretation, such as row-major scalarized access
  decomposition, lives in SDE Analysis/Utils.
- ARTS loop graph state, DB-relative loop windows, and ownership proofs live
  in ARTS Analysis, not in generic loop helpers.
- ARTS-RT LLVM CFG loop hints and pointer-lowering decisions live in ARTS-RT.

## Attribute Constants

Always use centralized constants:

```cpp
AttrNames::Operation::ArtsId
AttrNames::Operation::DepPattern
AttrNames::Operation::PartitionMode
AttrNames::Operation::Stencil::StencilCenterOffset
AttrNames::Operation::Stencil::ElementStride
AttrNames::Operation::Stencil::FootprintMinOffsets
AttrNames::Operation::Stencil::FootprintMaxOffsets
```

See `include/carts/utils/OperationAttributes.h` and
`include/carts/utils/StencilAttributes.h` for the full list.

## File Paths

```text
include/carts/utils/Utils.h
include/carts/utils/LoopUtils.h
include/carts/utils/RemovalUtils.h
include/carts/utils/LocationMetadata.h
include/carts/utils/OperationAttributes.h
include/carts/utils/StencilAttributes.h
include/carts/utils/ValueAnalysis.h
include/carts/dialect/sde/Analysis/AffineAccessUtils.h
include/carts/dialect/sde/Analysis/StructuredOpAnalysis.h
include/carts/dialect/sde/Utils/SDECostModel.h
include/carts/dialect/codir/Utils/CodeletABIUtils.h
include/carts/dialect/arts/Utils/DbUtils.h
include/carts/dialect/arts/Utils/EdtUtils.h
include/carts/dialect/arts/Utils/BlockedAccessUtils.h
include/carts/dialect/arts/Utils/LoweringContractUtils.h
include/carts/dialect/arts/Utils/LoopInvarianceUtils.h
include/carts/dialect/arts/Utils/PartitionPredicates.h
include/carts/dialect/arts/Utils/RuntimeConfig.h
include/carts/dialect/arts/Utils/ValueAnalysisUtils.h
include/carts/dialect/arts/Utils/LaunchPolicyUtils.h
include/carts/dialect/arts-rt/Utils/IdRegistry.h
include/carts/dialect/arts-rt/Utils/RuntimeCallUtils.h
include/carts/dialect/arts-rt/Utils/RtDbUtils.h
```
