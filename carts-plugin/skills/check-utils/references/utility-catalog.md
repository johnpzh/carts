# CARTS Shared Utility Catalog

Quick-reference for searching existing utilities before adding new helpers.
Use this with `carts-check-utils`; the live compiler still wins when docs
disagree.

## Canonical Categories

| Category | Canonical owner | Examples |
|----------|-----------------|----------|
| Value constants/folding/provenance | `include/carts/utils/ValueAnalysis.h` | `isZeroConstant`, `isOneConstant`, `tryFoldConstantIndex`, `sameValue`, `dependsOn`, `stripMemrefViewOps` |
| Index builders/general IR helpers | `include/carts/utils/Utils.h` | `createConstantIndex`, `createZeroIndex`, `createOneIndex`, `dominatesOrInAncestor`, `replaceUses` |
| Loop shape and IV helpers | `include/carts/utils/LoopUtils.h` | `isLoopInductionVar`, `getLoopInductionVar`, `getStaticTripCount`, `findNearestLoop`, `getLoopDepth` |
| Loop invariance and hoisting | `include/carts/utils/LoopInvarianceUtils.h` | `isLoopInvariant`, `findHoistTarget`, `allOperandsDominate`, `isSafeDivRemToHoist` |
| Deferred removal | `include/carts/utils/RemovalUtils.h` | `markForRemoval`, `removeAllMarked`, `replaceWithUndef` |
| SDE access/planning facts | `include/carts/dialect/sde/Analysis` or `include/carts/dialect/sde/Utils` | `AffineAccessUtils`, `StructuredOpAnalysis`, `SDECostModel` |
| CODIR codelet ABI | `include/carts/dialect/codir/Utils` | `CodeletABIUtils` |
| CODIR boundary proof logic | boundary conversion helper | `SdeToCodir/TaskDepSliceUtils.*` |
| ARTS DB mechanics | `include/carts/dialect/arts/Utils/DbUtils.h` | `traceToDbAlloc`, `getUnderlyingDb`, `getMemoryAccessInfo`, `isWriterMode` |
| ARTS EDT mechanics | `include/carts/dialect/arts/Utils/EdtUtils.h` | `EdtEnvManager`, `isInsideEpoch`, `classifyEdtArgAccesses` |
| ARTS partition/block predicates | `include/carts/dialect/arts/Utils` | `BlockedAccessUtils`, `PartitionPredicates`, `LoweringContractUtils` |
| ARTS graph/ownership analysis | `include/carts/dialect/arts/Analysis` | `DbGraph`, `EdtGraph`, `LoopNode`, `DbDistributedEligibility` |
| ARTS-RT runtime ABI | `include/carts/dialect/arts-rt/Utils` | `IdRegistry`, `RuntimeCallUtils`, `RtDbUtils` |

Do not create new files such as `LoopIVUtils`, `DbHelperUtils`, or
`ValueHelpers` unless the category is genuinely new and the patch explains why
the existing owner is wrong.

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
| Replace uses or values in a region | `replaceUses()`, `replaceInRegion()` | `Utils.h` |
| Detect trailing work in a block | `hasWorkAfterInParentBlock()` | `Utils.h` |
| Sort stores in program order | `sortStoresInProgramOrder()` | `Utils.h` |
| Detect undef-like ops | `isUndefLikeOp()` | `Utils.h` |
| Check if op is pure enough for analysis | `isPureOp()` or `isSideEffectFreeArithmeticLikeOp()` | `Utils.h` |
| Check loop invariance | `isLoopInvariant()` | `LoopInvarianceUtils.h` |
| Find loop hoist target | `findHoistTarget()` | `LoopInvarianceUtils.h` |
| Check div/rem hoist safety | `isSafeDivRemToHoist()` | `LoopInvarianceUtils.h` |
| All operands defined outside? | `allOperandsDefinedOutside()` | `LoopInvarianceUtils.h` |
| All operands dominate point? | `allOperandsDominate()` | `LoopInvarianceUtils.h` |
| Get static trip count | `getStaticTripCount()` | `LoopUtils.h` |
| Is innermost loop? | `isInnermostLoop()` | `LoopUtils.h` |
| Is loop IV? | `isLoopInductionVar()` | `LoopUtils.h` |
| Get primary loop IV | `getLoopInductionVar()` | `LoopUtils.h` |
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
| Simplify IR (CSE) | `simplifyIR()` | `Utils.h` |

## Loop And IV Placement Rules

- Generic loop IV, nearest-loop, trip-count, and loop-depth helpers live in
  `LoopUtils.h`.
- Loop-invariant and hoisting-safety helpers live in
  `LoopInvarianceUtils.h`.
- SDE structured-access interpretation, such as row-major scalarized access
  decomposition, lives in SDE Analysis/Utils.
- ARTS loop graph state, DB-relative loop windows, and ownership proofs live
  in ARTS Analysis, not in generic loop helpers.
- ARTS-RT LLVM CFG loop hints and pointer-lowering decisions live in ARTS-RT.

## Attribute Constants

Always use centralized constants:

```cpp
AttrNames::Operation::ArtsId
AttrNames::Operation::Workers
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
include/carts/utils/LoopInvarianceUtils.h
include/carts/utils/RemovalUtils.h
include/carts/utils/LocationMetadata.h
include/carts/utils/RuntimeConfig.h
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
include/carts/dialect/arts/Utils/PartitionPredicates.h
include/carts/dialect/arts/Utils/LaunchPolicyUtils.h
include/carts/dialect/arts-rt/Utils/IdRegistry.h
include/carts/dialect/arts-rt/Utils/RuntimeCallUtils.h
include/carts/dialect/arts-rt/Utils/RtDbUtils.h
```
