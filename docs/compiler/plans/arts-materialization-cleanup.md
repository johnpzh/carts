# ARTS Materialization Cleanup Plan

## Objective

Make ARTS materialization direct and mechanical: CODIR deps become
`arts.db_acquire`, MU storage becomes `arts.db_alloc`, CODIR codelets become
`arts.edt`, and no ARTS pass rediscovers source-level DB or dependency policy.

See [`codir-edt-isolation.md`](./codir-edt-isolation.md) for the codelet
conversion paths that feed this materialization.

## Current Surface

Relevant current areas:

- `ConvertCodirToArts`
- retired direct SDE-to-ARTS materialization
- `CreateDbs`
- `VerifyArtsObjectsOnly`
- `VerifyEdtCreated`
- DB/EDT/epoch analyses under the current `core/` source tree

## Target Contract

ARTS may:

- allocate DBs from explicit MU storage via `arts.db_alloc`;
- acquire DB windows from explicit CODIR deps via `arts.db_acquire`;
- create EDTs from isolated CODIR codelets and explicit params via `arts.edt`;
- bind logical worker capacity to ARTS topology;
- refine DB/EDT/epoch mechanics without changing SDE/CODIR policy.

ARTS must not:

- choose owner dims;
- infer dependency-window legality;
- scan raw memrefs to invent DB roots for supported cases;
- recover implicit codelet captures;
- introduce an `arts.lowering_contract` marker (or any equivalent
  late-rediscovery sentinel) for cases SDE/CODIR could have proved directly.

## Phases

### Phase 1: Direct Materialization Coverage

- [x] Audit every canonical MU/token/codelet path previously lowered directly
  from SDE to ARTS.
- [x] Define the equivalent CODIR-to-ARTS lowering shape.
- [x] Preserve current no-subview behavior while moving responsibility to the
  token-local rewrite path.
- [x] Rematerialize sliced CODIR deps inside ARTS EDTs using result types
  inferred from the actual DB payload memref type, so static SDE/CODIR views
  remain valid after `sde.mu_alloc` lowers to dynamic ARTS DB payloads.

Exit gate:

- current codelet conversion tests have direct CODIR-to-ARTS equivalents.

### Phase 2: CreateDbs Shrink

- Classify every remaining `CreateDbs` responsibility as:
  - direct materialization should own it;
  - temporary raw-memref materialization should own it;
  - dead behavior.
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

- [x] Use `VerifyArtsObjectsOnly` as the only CODIR-to-ARTS object verifier.
- [x] Add checks that explicit-param ARTS EDTs have complete dep/param metadata
  and reject nonconstant above-captures, including non-scalar values. External
  `memref.alloca` stack scratch remains legal because it is sunk/cloned into
  the EDT body rather than lowered as an ABI capture.
- [x] Remove the ARTS EDT verifier-bypass attribute; invalid EDT shapes now
  fail at parse/verify time instead of being carried to a later cleanup pass.
- Add checks that no unsupported source-shaped SDE/CODIR ops survive.

Exit gate:

- invalid tests fail at the boundary, not in late lowering.

## Verification

- Conversion lit tests for MU storage, memory deps, control deps, scalar params,
  releases, and reductions.
- 2026-05-15 evidence after verifier hardening: `dekk carts build` and
  `dekk carts test` passed, with the stale raw-physical-layout fixture removed
  because it only exercised the deleted verifier-bypass path.
- Negative tests for unresolved raw memrefs in supported paths.
- Focused e2e tests for samples that previously used `CreateDbs`.
- Pipeline dumps proving supported paths bypass the coarse raw bridge.
