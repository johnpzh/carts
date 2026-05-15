# Known Duplicate & Misplaced Functions

Last updated: 2026-04-10 (audit rev 2)

## Tier 1: Exact Duplicates

### `isOneLikeValue` / `isOneLike`
- **Location 1**: `lib/carts/dialect/arts/Conversion/ArtsToRt/EdtLowering.cpp:128` (static)
- **Location 2**: `lib/carts/dialect/arts/Analysis/db/DbAnalysis.cpp:1534` (lambda in `hasSingleSize`)
- **Implementation**: Checks `ValueAnalysis::isOneConstant()`, then pattern-matches `add(1, sub(x,x))` and `add(1, sub(min(a,b), a))`
- **Both are 100% identical** except parameter naming
- **Target**: `ValueAnalysis::isOneLikeValue()` in `include/carts/utils/ValueAnalysis.h`

### `hasWorkAfterInParentBlock`
- **Location 1**: `lib/carts/dialect/arts/Conversion/OmpToArts/ConvertOpenMPToArts.cpp:86`
- **Location 2**: `lib/carts/dialect/sde/Transforms/ConvertOpenMPToSde.cpp:79`
- **Implementation**: Walks remaining ops in parent block checking for non-terminator work
- **Target**: `Utils.h`

### `isPureOp`
- **Location 1**: `lib/carts/dialect/arts/Transforms/kernel/StencilTilingNDPattern.cpp:80`
- **Location 2**: `lib/carts/dialect/arts/Transforms/PatternDiscovery.cpp:152`
- **Implementation**: Checks isa<> for memory ops, calls MemoryEffectOpInterface
- **Related**: `isSideEffectFreeArithmeticLikeOp()` in `Utils.h` (similar but narrower)
- **Target**: `Utils.h` as `isPureOp()` (broader than existing)

### `sortStoresInProgramOrder`
- **Location 1**: `lib/carts/dialect/arts/Transforms/EdtStructuralOpt.cpp`
- **Location 2**: `lib/carts/dialect/arts/Transforms/EdtAllocaSinking.cpp`
- **Implementation**: Sorts memref::StoreOps by block position
- **Target**: `Utils.h`

### `findHoistTarget`
- **Location 1**: `lib/carts/dialect/arts-rt/Transforms/RuntimeCallOpt.cpp:48` (for func::CallOp)
- **Location 2**: `lib/carts/dialect/arts-rt/Transforms/DataPtrHoistingSupport.cpp:1376` (for Operation*)
- **Implementation**: Finds highest enclosing loop where all operands are loop-invariant
- **Note**: Different signatures but same algorithm
- **Target**: `LoopInvarianceUtils.h` (template or overloads)

### `getMemoryAccessInfo`
- **Location 1**: `lib/carts/dialect/arts-rt/Transforms/DataPtrHoistingSupport.cpp:604`
- **Location 2**: `lib/carts/utils/DbUtils.cpp:855` (CANONICAL)
- **Action**: Replace DataPtrHoistingSupport's version with `DbUtils::getMemoryAccessInfo()`

### `isUndefLikeOp` / undef string check **(NEW — found 2026-04-10 rev 2)**
- **Location 1**: `lib/carts/dialect/arts/Conversion/ArtsToRt/EdtLowering.cpp:120` (static `isUndefLikeOp`)
- **Location 2**: `lib/carts/utils/EdtUtils.cpp:258` (inline check `== "llvm.mlir.undef"`)
- **Location 3**: `lib/carts/dialect/arts/Transforms/EpochOptCpsChain.cpp:450` (inline check `== "llvm.mlir.undef"`)
- **Implementation**: All check if an op is an undef-like value (llvm.mlir.undef, polygeist.undef, arts.undef)
- **Note**: EdtLowering's version is broadest (checks 3 names), other 2 only check llvm.mlir.undef
- **Target**: `Utils.h` as `isUndefLikeOp()` (consolidate all 3 name checks)

## Tier 2: Hardcoded Attribute Strings

### `BlockLoopStripMiningSupport.cpp:18-19`
```cpp
const llvm::StringLiteral kStripMiningGeneratedAttr =
    "arts.block_loop_strip_mining.generated";
```
**Fix**: Add `StripMiningGenerated` to `AttrNames::Operation` in `OperationAttributes.h`

### `RaiseToLinalg.cpp:240-295`
- Line 240-241: `"parallel"`, `"reduction"` in `computeIteratorTypes()`
- Line 275, 285-287, 293, 295: `"stencil"`, `"elementwise"`, `"matmul"`, `"reduction"` in `classifyPattern()`
**Fix**: Use linalg iterator type constants or define as `PatternSemantics` enum

### `EdtLowering.cpp:123-125`
```cpp
return name == "llvm.mlir.undef" || name == "polygeist.undef" || name == "arts.undef";
```
**Fix**: Use `isa<LLVM::UndefOp>()` check or centralize names

### `MetadataAttacher.cpp:62,97` + `MetadataRegistry.cpp:303` + `MetadataManager.cpp:18`
- Uses `ArtsMetadata::IdAttrName` = `"arts.id"` instead of `AttrNames::Operation::ArtsId`
**Fix**: Replace `ArtsMetadata::IdAttrName` with `AttrNames::Operation::ArtsId`

## Tier 3: High-Value Extractions (Not Duplicated, But Misplaced)

| Function | Source | Target | LOC | Risk |
|----------|--------|--------|-----|------|
| `getLoadInfo/getStoreInfo/getStoredValue` | ScalarReplacement.cpp | DbUtils.h | 30 | VeryLow |
| `getForwardedMemrefAliasSource/Result` | RaiseMemRefDimensionality.cpp | Utils.h | 35 | Low |
| `hoistInvariantOpsInLoop` | Hoisting.cpp | LoopInvarianceUtils.h | 36 | Medium |
| `buildLoopInvariantI1Not/And` | DataPtrHoistingSupport.cpp | Utils.h | 15 | VeryLow |
| `isUnitStrideLoop` | PerfectNestLinearizationPattern.cpp | LoopUtils.h | 5 | VeryLow |
| `ensureBlock` | ConvertOpenMPToSde.cpp | Utils.h | 5 | VeryLow |
| `isUndefLikeOp` | EdtLowering.cpp | Utils.h | 6 | VeryLow |
| `tryGetAffineExpr` | RaiseToLinalg.cpp | new AffineUtils.h | 52 | Low |
| `isAccumulationOp` | ScalarReplacement.cpp | Utils.h | 4 | VeryLow |
| `indicesEqual` | ScalarReplacement.cpp | Utils.h | 8 | VeryLow |
| `isZeroIndexConstant/isMinusOneConstant` | DataPtrHoistingSupport.cpp | ValueAnalysis.h | 8 | VeryLow |
| `getConstInt` | SdeIterationSpaceDecomposition.cpp | **DONE** → delegates to `ValueAnalysis::getConstantIndexStripped` | 10 | VeryLow |
| `getOrCreateZero` | MatmulReductionPattern.cpp:81 | Utils.h | 2 | VeryLow |
| `isZeroIndexList` | DbScratchElimination.cpp:66 | Utils.h or ValueAnalysis.h | 5 | VeryLow |
| `regionHasNoWork` | JacobiAlternatingBuffersPattern.cpp:340 | Utils.h | ~8 | VeryLow |
| `isLoopInvariantForRepeat` | EpochOptStructural.cpp:173 | LoopInvarianceUtils.h | ~10 | Low |
| `getConstIndex` | BlockLoopStripMiningSupport.cpp:37 | **DONE** → delegates to `ValueAnalysis::getConstantIndexStripped` | 9 | VeryLow |
| `tryFoldKnownRuntimeShape` | BlockLoopStripMiningSupport.cpp:48 | **DONE** → RuntimeQueryOp merged into `tryFoldConstantIndex`, -110 LOC | 110 | Medium |
| `analyzeLegacyLoop` | BlockLoopStripMiningSupport.cpp:896 | **STAYS** — coupled to ~15 pass-private helpers, not general analysis | 80 | Medium |

## Statistics (rev 2 audit)

| Directory | Static Functions | % of Codebase |
|-----------|-----------------|---------------|
| `dialect/core/Transforms/` | 230 | 59.1% |
| `transforms/` (all subdirs) | 114 | 29.3% |
| `dialect/sde/Transforms/` | 15 | 3.9% |
| `dialect/core/Conversion/ArtsToRt/` | 13 | 3.3% |
| `dialect/core/Conversion/OmpToArts/` | 7 | 1.8% |
| `dialect/core/Conversion/ArtsToLLVM/` | 5 | 1.3% |
| `dialect/rt/Transforms/` | 2 | 0.5% |
| **Total** | **389** | **100%** |

**Summary**: 9 exact duplicates (Tier 1), 4+ hardcoded string violations (Tier 2), 18+ extractable helpers (Tier 3).
