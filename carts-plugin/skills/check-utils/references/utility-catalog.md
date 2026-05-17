# CARTS Shared Utility Catalog

Quick-reference for searching existing utilities before adding new helpers.
Use this with `carts-check-utils`; the live compiler still wins when docs
disagree.

## Canonical Categories

| Category | Canonical owner | Examples |
|----------|-----------------|----------|
| Dialect-neutral value constants/folding | `include/carts/utils/ValueAnalysis.h` | `isZeroConstant`, `isOneConstant`, `tryFoldConstantIndex`, `sameValue`, `dependsOn`, `stripMemrefViewOps` |
| ARTS DB/EDT/runtime-query value folding, provenance, and block-index analysis | `include/carts/dialect/arts/Utils/ValueAnalysisUtils.h` | `tryFoldConstantIndex`, `stripSelectClamp`, `getOffsetStride`, `extractConstantOffset`, `getUnderlyingValue`, `isDerivedFromPtr` |
| ARTS-RT depv/DB pointer value folding and provenance | `include/carts/dialect/arts-rt/Utils/RtDbUtils.h` | `RtDbUtils::tryFoldConstantIndex`, `RtDbUtils::getUnderlyingValue`, `RtDbUtils::isDerivedFromPtr` |
| Index builders/general pure-op predicates | `include/carts/utils/Utils.h` | `createConstantIndex`, `createZeroIndex`, `createOneIndex`, `isSideEffectFreeArithmeticLikeOp` |
| Generic loop trip-count helpers | `include/carts/utils/LoopUtils.h` | `isInnermostLoop`, `getStaticTripCount` |
| ARTS/ARTS-RT loop invariance and hoisting | `include/carts/dialect/arts/Utils/LoopInvarianceUtils.h` | `isLoopInvariant`, `findHoistTarget`, `allOperandsDominate`, `isSafeDivRemToHoist` |
| Deferred removal | `include/carts/utils/RemovalUtils.h` | `markForRemoval`, `removeAllMarked` |
| SDE access/planning facts | `include/carts/dialect/sde/Analysis` or `include/carts/dialect/sde/Utils` | `AffineAccessUtils`, `StructuredOpAnalysis`, `SDECostModel` |
| CODIR codelet ABI | `include/carts/dialect/codir/Utils` | `CodeletABIUtils` |
| CODIR boundary proof logic | boundary conversion helper | `SdeToCodir/TaskDepSliceUtils.*` |
| SDE Polygeist input normalization helpers | `lib/carts/dialect/sde/Conversion/PolygeistToSde/PolygeistToSdeUtils.h` | `materializeDependView`, `clampDepIndices`, `isInsideOmpRegion`, `containsOmpOp` |
| ARTS DB mechanics | `include/carts/dialect/arts/Utils/DbUtils.h` | `traceToDbAlloc`, `getUnderlyingDb`, `getMemoryAccessInfo`, `isWriterMode` |
| ARTS EDT mechanics | `include/carts/dialect/arts/Utils/EdtUtils.h` | `EdtEnvManager`, `isInsideEpoch`, `classifyEdtArgAccesses` |
| ARTS loop structure, partition/block predicates, runtime topology, and source-location IDs | `include/carts/dialect/arts/Utils` | `LoopStructureUtils`, `BlockedAccessUtils`, `PartitionPredicates`, `LoweringContractUtils`, `RuntimeConfig`, `LocationMetadata` |
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
| Strip ARTS block-index select clamps | `stripSelectClamp()` | `dialect/arts/Utils/ValueAnalysisUtils.h` |
| Detect ARTS block-index stride | `getOffsetStride()` | `dialect/arts/Utils/ValueAnalysisUtils.h` |
| Extract ARTS block-relative constant offset | `extractConstantOffset()` | `dialect/arts/Utils/ValueAnalysisUtils.h` |
| Create zero/one index | `createZeroIndex()`, `createOneIndex()` | `Utils.h` |
| Create arbitrary index constant | `createConstantIndex()` | `Utils.h` |
| Check dominance with ancestor regions | keep pass-local unless a second owner appears | current ARTS strip-mining helper |
| Replace uses or values in a region | keep pass-local unless a second owner appears | current EDT rematerialization helper |
| Detect trailing work in a block | keep pass-local unless a second owner appears | current OpenMP-to-SDE helper |
| Detect undef-like ops | `isUndefLikeOp()` | `RuntimeOpUtils.h` |
| Check if op is side-effect-free arithmetic-like | `isSideEffectFreeArithmeticLikeOp()` | `Utils.h` |
| Check ARTS/ARTS-RT loop invariance | `isLoopInvariant()` | `dialect/arts/Utils/LoopInvarianceUtils.h` |
| Find ARTS/ARTS-RT loop hoist target | `findHoistTarget()` | `dialect/arts/Utils/LoopInvarianceUtils.h` |
| Check div/rem hoist safety | `isSafeDivRemToHoist()` | `dialect/arts/Utils/LoopInvarianceUtils.h` |
| All operands defined outside? | `allOperandsDefinedOutside()` | `dialect/arts/Utils/LoopInvarianceUtils.h` |
| All operands dominate point? | `allOperandsDominate()` | `dialect/arts/Utils/LoopInvarianceUtils.h` |
| Get static trip count | `getStaticTripCount()` | `LoopUtils.h` |
| Is innermost loop? | `isInnermostLoop()` | `LoopUtils.h` |
| Get ARTS loop depth | `getLoopDepth()` | `LoopStructureUtils.h` |
| Collect while-loop bounds | `collectWhileBounds()` | `LoopStructureUtils.h` |
| Trace value to DB alloc | `DbUtils::traceToDbAlloc()` | `DbUtils.h` |
| Get underlying DB allocation | `DbUtils::getUnderlyingDb()` | `DbUtils.h` |
| Extract DB memory access info | `DbUtils::getMemoryAccessInfo()` | `DbUtils.h` |
| Get memref from load/store | `DbUtils::getAccessedMemref()` | `DbUtils.h` |
| Get indices from load/store | `DbUtils::getMemoryAccessIndices()` | `DbUtils.h` |
| Is inside epoch? | `EdtUtils::isInsideEpoch()` | `EdtUtils.h` |
| Classify EDT captures | `EdtEnvManager` / `EdtUtils` | `EdtUtils.h` |
| Combine access modes | `combineAccessModes()` | `DbUtils.h` |
| Check writer mode | `DbUtils::isWriterMode()` | `DbUtils.h` |
| Get element byte size | `getElementTypeByteSize()` | `DbUtils.h` |
| Check block layout | `usesBlockLayout()` | `PartitionPredicates.h` |
| Supports halo? | `supportsHaloExtension()` | `PartitionPredicates.h` |
| Is stencil family? | `isStencilFamilyDepPattern()` | `OperationAttributes.h` |
| Get lowering contract | `getLoweringContract()` | `LoweringContractUtils.h` |
| Mark op for deferred removal | `RemovalUtils::markForRemoval()` | `RemovalUtils.h` |
| Clamp SDE Polygeist dep indices | `sde::clampDepIndices()` | `PolygeistToSdeUtils.h` |
| Check OMP nesting during SDE input preparation | `sde::{isInsideOmpRegion,containsOmpOp}` | `PolygeistToSdeUtils.h` |

## Loop Placement Rules

- Generic shared loop utilities stay limited to neutral trip-count and
  innermost-loop queries in `LoopUtils.h`.
- Single-owner loop IV and nearest-loop helpers stay near their SDE consumers
  unless a second real owner appears.
- ARTS loop-depth and while-bound helpers used by DB/EDT analysis live in
  `include/carts/dialect/arts/Utils/LoopStructureUtils.h`.
- ARTS/ARTS-RT loop-invariant and hoisting-safety helpers live in
  `include/carts/dialect/arts/Utils/LoopInvarianceUtils.h`.
- SDE structured-access interpretation, such as row-major scalarized access
  decomposition, lives in SDE Analysis/Utils.
- ARTS loop graph state, DB-relative loop windows, and ownership proofs live
  in ARTS Analysis, not in generic loop helpers.
- ARTS-RT LLVM CFG loop hints and pointer-lowering decisions live in ARTS-RT.

## Attribute Names

For CARTS IR attributes, prefer the owning TableGen definition and generated
ODS accessor:

```cpp
op.getStencilCenterOffsetAttrName()
op.getStencilMinOffsetsAttrName()
op.getStencilMaxOffsetsAttrName()
```

Use `include/carts/utils/OperationAttributes.h` only for remaining shared or
transitional ARTS metadata that is not yet an ODS attr. See
`include/carts/utils/StencilAttributes.h` for ARTS stencil contract helpers,
and `include/carts/utils/ArrayAttrUtils.h` for dialect-neutral i64 array
attribute builders/readers.

## File Paths

```text
include/carts/utils/Utils.h
include/carts/utils/LoopUtils.h
include/carts/utils/RemovalUtils.h
include/carts/utils/ArrayAttrUtils.h
include/carts/utils/OperationAttributes.h
include/carts/utils/StencilAttributes.h
include/carts/utils/ValueAnalysis.h
include/carts/dialect/sde/Analysis/AffineAccessUtils.h
include/carts/dialect/sde/Analysis/StructuredOpAnalysis.h
include/carts/dialect/sde/Utils/SDECostModel.h
lib/carts/dialect/sde/Conversion/PolygeistToSde/PolygeistToSdeUtils.h
include/carts/dialect/codir/Utils/CodeletABIUtils.h
include/carts/dialect/arts/Utils/DbUtils.h
include/carts/dialect/arts/Utils/EdtUtils.h
include/carts/dialect/arts/Utils/BlockedAccessUtils.h
include/carts/dialect/arts/Utils/LocationMetadata.h
include/carts/dialect/arts/Utils/LoweringContractUtils.h
include/carts/dialect/arts/Utils/LoopInvarianceUtils.h
include/carts/dialect/arts/Utils/LoopStructureUtils.h
include/carts/dialect/arts/Utils/PartitionPredicates.h
include/carts/dialect/arts/Utils/RuntimeConfig.h
include/carts/dialect/arts/Utils/ValueAnalysisUtils.h
include/carts/dialect/arts/Utils/LaunchPolicyUtils.h
include/carts/dialect/arts-rt/Utils/IdRegistry.h
include/carts/dialect/arts-rt/Utils/RuntimeCallUtils.h
include/carts/dialect/arts-rt/Utils/RtDbUtils.h
```
