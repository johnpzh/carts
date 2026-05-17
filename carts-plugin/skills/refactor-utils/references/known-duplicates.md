# Known Utility Cleanup Backlog

Last updated: 2026-05-16.

This file is a triage aid, not a mandate to extract everything. Run
`carts-check-utils` first; extract only when the helper has a second consumer or
expresses a dialect invariant.

## Already Consolidated

These helpers have canonical homes. Do not reintroduce local copies.

| Need | Canonical helper |
|------|------------------|
| one-like expression recognition | `ValueAnalysis::isOneLikeValue` |
| trailing work in parent block | `hasWorkAfterInParentBlock` in `Utils.h` |
| loop hoist target | `findHoistTarget` in ARTS `LoopInvarianceUtils.h` |
| DB memory access info | `DbUtils::getMemoryAccessInfo` |
| undef-like op recognition | `isUndefLikeOp` in `Utils.h` |
| strip-mining generated attribute | `AttrNames::Operation::StripMiningGenerated` |

## Watch List

These are still pass-local or pass-area helpers. They should stay where they are
unless a second real consumer appears.

| Helper | Current home | Placement note |
|--------|--------------|----------------|
| `getLoadInfo`, `getStoreInfo`, `getStoredValue`, `isAccumulationOp` | `lib/carts/dialect/arts-rt/Transforms/ScalarReplacement.cpp` | ARTS-RT scalar replacement internals. Extract to ARTS-RT Utils only with another lowering/optimization consumer. |
| `buildLoopInvariantI1Not` | `DataPtrHoistingInternal.h` / `DataPtrHoistingSupport.cpp` | ARTS-RT data-pointer hoisting support, not generic loop invariance. |
| `hoistInvariantOpsInLoop` | `lib/carts/dialect/arts/Transforms/db/Hoisting.cpp` | ARTS DB hoisting policy; keep pass-local until another ARTS transform needs identical behavior. |
| `getForwardedMemrefAliasSource`, `getForwardedMemrefAliasResult` | `lib/carts/dialect/sde/Conversion/PolygeistToSde/MemrefNormalization.cpp` | SDE conversion-specific alias cleanup. Extract to SDE Analysis/Utils only with a second SDE consumer. |
| `tryGetAffineExpr` | `lib/carts/dialect/sde/Analysis/StructuredOpAnalysis.cpp` | SDE structured analysis internals; keep in analysis until exported API is needed. |
| `ensureBlock` | `include/carts/dialect/sde/Transforms/Passes.h` | SDE region-construction convenience used across SDE passes. If it grows, move to SDE Utils. |
| `analyzeSingleDimBlockLoop` | ARTS block-loop strip-mining support | Coupled to block-loop strip-mining internals; not a generic loop helper. |

## Current Static-Helper Risk

A broad `rg` still finds many static helpers across SDE, ARTS, and ARTS-RT.
That is expected while pass internals remain private. The cleanup target is not
"zero static helpers"; it is "zero duplicated helpers or misplaced dialect
facts."

High-value future audits:

- SDE memref/access helpers in `PolygeistToSde` and `StructuredOpAnalysis`.
- ARTS-RT pointer/packing helpers in lowering and data-pointer hoisting.
- ARTS DB loop/window helpers that may belong in ARTS Analysis instead of pass
  support files.
