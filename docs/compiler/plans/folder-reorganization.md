# Folder Reorganization Plan

See [`dialect-stack-migration.md`](./dialect-stack-migration.md) for the
conceptual driver/dialect split that this physical layout supports.

## Objective

Make the source tree match the target dialect ownership:

```text
carts
  sde
  codir
  arts
  arts-rt
```

The current source tree still lives mostly under `include/arts` and `lib/arts`.
That was acceptable while the project had one ARTS-centered compiler layer, but
it now hides ownership boundaries. SDE and CODIR are CARTS dialects, not ARTS
subsystems. ARTS and ARTS-RT are separate target dialect layers.

## Target Layout

Use `carts/dialect` as the compiler dialect root:

```text
include/carts/dialect/
  sde/
    IR/
    Analysis/
    Transforms/
    Conversion/
    Verify/
    Utils/
  codir/
    IR/
    Analysis/
    Transforms/
    Conversion/
    Verify/
    Utils/
  arts/
    IR/
    Analysis/
    Transforms/
    Conversion/
    Verify/
    Utils/
  arts-rt/
    IR/
    Analysis/
    Transforms/
    Conversion/
    Verify/
    Utils/

lib/carts/dialect/
  sde/
    IR/
    Analysis/
    Transforms/
    Conversion/
    Verify/
    Utils/
  codir/
    IR/
    Analysis/
    Transforms/
    Conversion/
    Verify/
    Utils/
  arts/
    IR/
    Analysis/
    Transforms/
    Conversion/
    Verify/
    Utils/
  arts-rt/
    IR/
    Analysis/
    Transforms/
    Conversion/
    Verify/
    Utils/
```

Every dialect has its own `Analysis/` and `Transforms/` area. The detailed
ownership contract is in
[`subdialect-analysis-optimization.md`](./subdialect-analysis-optimization.md).
Every dialect also has its own `Utils/` area. The utility placement contract is
in [`utility-ownership.md`](./utility-ownership.md).

Use `include/carts/support` and `lib/carts/support` later for compiler-wide
utilities that are not ARTS-specific. Existing `include/carts/utils` should be
split only when a utility's owner is clear:

- CARTS semantic utilities move to `carts/support`.
- SDE-only utilities move under `carts/dialect/sde`.
- CODIR-only utilities move under `carts/dialect/codir`.
- ARTS-machine utilities remain under `carts/dialect/arts` or `carts/support`.
- Runtime ABI utilities move under `carts/dialect/arts-rt`.

## Naming Rules

- Folder names use the dialect layer names: `sde`, `codir`, `arts`, `arts-rt`.
- C++ namespaces should migrate toward `mlir::carts::<dialect>` for SDE and
  CODIR, with `mlir::carts::arts` and `mlir::carts::arts_rt` for target
  dialects if the project chooses a unified CARTS namespace.
- MLIR textual names stay concise: `sde`, `codir`, `arts`, and `arts_rt`.
- The current `core` name is retired from documentation and should disappear
  from new source paths. It may remain in old paths during migration only.
- Do not use `arts_sde` or `arts_sde.dep_family` style names for new facts.
  Use `sde.pattern`, `sde.dep_family`, or the owning dialect's equivalent.

## Migration Status

The physical layout migration is **complete**. The source tree now matches
the target exactly:

```text
include/carts/{Dialect.h, dialect/{sde,codir,arts,arts-rt}/, passes/, utils/}
lib/carts/    {                dialect/{sde,codir,arts,arts-rt}/, passes/, utils/}
```

The legacy `include/arts/` and `lib/arts/` umbrellas no longer exist. The
"core" subdirectory was renamed to `arts`; "rt" was renamed to `arts-rt`.

CMake target names follow the same pattern: `MLIRCartsArts`, `MLIRCartsArtsRt`,
`MLIRCartsSde`, `MLIRCartsCodir*`, and the umbrella `MLIRCartsTransforms`.

Both follow-on items are also complete:

- **C++ namespace migration**: `mlir::arts::*` → `mlir::carts::arts::*`;
  `mlir::arts::sde::*` → `mlir::carts::sde::*`; `mlir::arts::rt::*` →
  `mlir::carts::arts_rt::*`. TableGen `cppNamespace` strings track the new
  roots. The legacy `mlir::arts` namespace (which was ambiguous between
  project umbrella and ARTS dialect) is gone.
- **`MLIRCartsTransforms` split**: now an umbrella that aggregates
  `MLIRCartsSdeTransforms`, `MLIRCartsArtsTransforms`, and
  `MLIRCartsArtsRtTransforms`. New consumers may link a single per-dialect
  library; existing consumers keep working through the umbrella name.

## Migration Phases

### Phase 0: Skeleton And Docs — Done

Target folders exist; dialect docs live under `docs/compiler/dialects/`; master
plan and dialect docs point to this subplan.

### Phase 1: CODIR First — Done

Status: the CODIR include/lib skeleton already exists at
`include/carts/dialect/codir/IR/` and `lib/carts/dialect/codir/IR/`, with
`convert-sde-to-codir`, `verify-codir`, and `convert-codir-to-arts` registered
and wired into the default pipeline.

Move the codelet-specific SDE code into the new CODIR folder before broad SDE
renaming:

- `state/codelet/CodeletUtils.*`
- `state/codelet/ConvertToCodelet.cpp`
- `state/codelet/ScalarForwarding.cpp` if it remains codelet-bound
- `state/codelet/TokenModeRefinement.cpp`
- codelet tests under `lib/carts/dialect/sde/test/codelet`

Do not move tensor cleanup as-is. Convert it to memref/CODIR cleanup or delete
it with tensor-carrier removal.

Exit gate:

- CODIR tests own codelet isolation;
- SDE tests own MU/CU/SU planning only.

### Phase 2: Boundary Conversions — Done

CODIR conversion folders live under
`lib/carts/dialect/codir/Conversion/{SdeToCodir,CodirToArts}`; the direct
SDE-to-ARTS conversion is gone from the live source tree.

### Phase 3: SDE Move — Done

SDE IR, analysis, transforms, verify, and conversion all live under
`include/carts/dialect/sde/` and `lib/carts/dialect/sde/`. All `#include`s use
`carts/dialect/sde/...`. No forwarding headers remain.

### Phase 4: ARTS And ARTS-RT Move — Done

`core/` → `arts/` and `rt/` → `arts-rt/` for both `include/` and `lib/`. ARTS
object materialization and ARTS-RT runtime ABI lowering live in physically
separate trees. ARTS depends on the arts-rt dialect for the boundary lowering
in `Conversion/ArtsToRt/`; that dependency is intentional.

### Phase 5: Utility Cleanup — Done

Utilities are reclassified by the earliest owning layer:

- CARTS-shared (used by 2+ dialects or project-wide): `lib/carts/utils/` —
  Debug, LoopUtils, OperationAttributes, PassInstrumentation, RemovalUtils,
  StencilAttributes, Utils, ValueAnalysis, benchmarks, testing.
- SDE-only: `lib/carts/dialect/sde/Utils/` — SDECostModel, plus the existing
  pass-area `Analysis/` headers (AffineAccessUtils, SdeAnalysisUtils,
  StructuredOpAnalysis) which intentionally stay under Analysis/ since they
  back the SDE structured-op analysis.
- CODIR-only: `lib/carts/dialect/codir/Utils/` — CodeletABIUtils.
- ARTS-only: `lib/carts/dialect/arts/Utils/` — DbUtils, EdtUtils, IdRegistry,
  LocationMetadata, LoweringContractUtils, PartitionPredicates,
  BlockedAccessUtils, MetadataAttrNames, MetadataEnums, ARTSCostModel,
  RuntimeConfig.
- ARTS-RT-only: `lib/carts/dialect/arts-rt/Utils/` — LoopInvarianceUtils.

## CMake Strategy

- Add new CMake subdirectories only when a real compilable slice moves.
- Update public include paths directly when moving a dialect slice.
- Avoid one giant include rewrite. Move one dialect or conversion boundary at a
  time.
- Keep generated TableGen target names stable until all users are migrated.

## Verification

For every physical move:

- `dekk carts build`
- focused lit tests for the moved dialect/pass
- `dekk carts pipeline --json` if driver stages or pass registration changed
- `dekk carts skills generate` if docs or command resources changed
- `git diff --check`
