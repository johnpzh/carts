# ARTS Materialization Cleanup Plan

## Objective

Make ARTS materialization direct and mechanical: CODIR deps become ARTS acquires,
MU storage becomes ARTS DB allocation, CODIR codelets become ARTS EDTs, and no
ARTS pass rediscovers source-level DB or dependency policy.

## Current Surface

Relevant current areas:

- `ConvertSdeToArts`
- `CreateDbs`
- `VerifyCoreObjectsOnly`
- `VerifyEdtCreated`
- DB/EDT/epoch analyses under the current `core/` source tree

## Target Contract

ARTS may:

- allocate DBs from explicit MU storage;
- acquire DB windows from explicit CODIR deps;
- create EDTs from isolated CODIR codelets and explicit params;
- bind logical worker capacity to ARTS topology;
- refine DB/EDT/epoch mechanics without changing SDE/CODIR policy.

ARTS must not:

- choose owner dims;
- infer dependency-window legality;
- scan raw memrefs to invent DB roots for supported cases;
- recover implicit codelet captures;
- introduce an `arts.db_control` marker.

## Phases

### Phase 1: Direct Materialization Coverage

- Audit every canonical MU/token/codelet path currently lowered in
  `ConvertSdeToArts`.
- Define the equivalent CODIR-to-ARTS lowering shape.
- Preserve current no-subview behavior while moving responsibility to the
  token-local rewrite path.

Exit gate:

- current codelet conversion tests have direct CODIR-to-ARTS equivalents.

### Phase 2: CreateDbs Shrink

- Classify every remaining `CreateDbs` responsibility as:
  - direct materialization should own it;
  - temporary raw-memref compatibility should own it;
  - dead legacy behavior.
- Move direct materialization responsibilities into CODIR-to-ARTS.
- Convert unsupported raw cases into clear diagnostics instead of silent policy
  inference.
- Keep the current Core implementation limited to coarse `db_ref[0]` raw
  memrefs. Blocked/tiled raw memrefs are unsupported at this boundary because
  SDE/CODIR must have already rewritten token-local accesses.

Exit gate:

- `CreateDbs` no longer chooses owner dims, dependency-window legality, or
  codelet captures.

### Phase 3: Remove Remaining Raw Bridge

- Keep Core raw DB indexers deleted.
- Move any remaining coarse raw-memref cases into direct CODIR-to-ARTS
  materialization.
- Delete `CreateDbs` when supported samples and benchmarks no longer require
  raw memref DB discovery.

Exit gate:

- supported benchmarks do not require the coarse raw `CreateDbs` bridge.

### Phase 4: Verifier Hardening

- Replace `VerifyCoreObjectsOnly` conceptually with `VerifyArtsObjectsOnly`.
- Add checks that ARTS EDTs have complete dep/param metadata.
- Add checks that no unsupported source-shaped SDE/CODIR ops survive.

Exit gate:

- invalid tests fail at the boundary, not in late lowering.

## Verification

- Conversion lit tests for MU storage, memory deps, control deps, scalar params,
  releases, and reductions.
- Negative tests for unresolved raw memrefs in supported paths.
- Focused e2e tests for samples that previously used `CreateDbs`.
- Pipeline dumps proving supported paths bypass the coarse raw bridge.
