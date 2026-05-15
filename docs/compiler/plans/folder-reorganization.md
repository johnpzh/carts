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
utilities that are not ARTS-specific. Existing `include/arts/utils` should be
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

## Current Compatibility Layout

The current tree maps to the target as follows:

| Current path | Target path | Notes |
|---|---|---|
| `include/arts/dialect/sde` | `include/carts/dialect/sde` | Move after CODIR split starts. |
| `lib/arts/dialect/sde` | `lib/carts/dialect/sde` | SDE should lose final codelet ABI ownership during this move. |
| `lib/arts/dialect/sde/Transforms/state/codelet` | `lib/carts/dialect/codir` | Split into CODIR where the code owns isolation/deps/params. |
| `include/arts/dialect/core` | `include/carts/dialect/arts` | Rename "core" to ARTS once behavior is stable. |
| `lib/arts/dialect/core` | `lib/carts/dialect/arts` | Keep ARTS-machine object logic here. |
| `include/arts/dialect/rt` | `include/carts/dialect/arts-rt` | Runtime ABI bridge. |
| `lib/arts/dialect/rt` | `lib/carts/dialect/arts-rt` | Runtime-call lowering and cleanup. |

## Migration Phases

### Phase 0: Skeleton And Docs

Create the target folders and document ownership. Do not wire them into CMake
until at least one real dialect slice moves.

Exit gate:

- target folders exist;
- dialect docs exist under `docs/compiler/dialects/`;
- master plan and dialect docs point to this subplan;
- `git diff --check` passes.

### Phase 1: CODIR First

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
- codelet tests under `lib/arts/dialect/sde/test/codelet`

Do not move tensor cleanup as-is. Convert it to memref/CODIR cleanup or delete
it with tensor-carrier removal.

Exit gate:

- CODIR tests own codelet isolation;
- SDE tests own MU/CU/SU planning only.

### Phase 2: Boundary Conversions

Create target conversion folders:

- `lib/carts/dialect/sde/Conversion/SdeToCodir`
- `lib/carts/dialect/codir/Conversion/CodirToArts`
- `lib/carts/dialect/arts/Conversion/ArtsToRt`
- `lib/carts/dialect/arts-rt/Conversion/ArtsRtToLLVM`

The direct SDE-to-ARTS conversion is removed from the live source tree. SDE
must cross the boundary through CODIR before ARTS materialization.

Exit gate:

- SDE-to-CODIR and CODIR-to-ARTS lit tests exist;
- direct-codelet tests are migrated to SDE-to-CODIR/CODIR-to-ARTS coverage or
  removed.

### Phase 3: SDE Move

Move SDE IR, analysis, transforms, and verification to
`carts/dialect/sde`. Do this after codelet ownership has been split out, so SDE
does not carry removed codelet ABI files into its new home.

Exit gate:

- includes use `carts/dialect/sde/...` for moved files;
- moved public include paths are updated at the call sites; migration-only
  forwarding headers are not part of the target layout.

### Phase 4: ARTS And ARTS-RT Move

Move current `core` to `arts` and current `rt` to `arts-rt` once direct
materialization is stable.

Exit gate:

- no new source path contains `core` except historical comments or release
  notes;
- ARTS object materialization and ARTS-RT runtime ABI lowering are physically
  separate.

### Phase 5: Utility Cleanup

Move utilities to the earliest owning layer:

- value/equivalence helpers to common CARTS support;
- memref/SDE analysis helpers to SDE;
- codelet capture helpers to CODIR;
- DB/EDT/epoch helpers to ARTS;
- runtime pointer/packing helpers to ARTS-RT.

Exit gate:

- no pass-local utility duplicates a common helper;
- every reusable helper has the narrowest owning dialect utility home;
- no SDE utility references ARTS runtime topology.

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
