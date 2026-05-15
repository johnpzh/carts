# CODIR And EDT Isolation Plan

## Objective

Create a dedicated CODIR layer that owns codelet ABI formation and guarantees
that every EDT is isolated from enclosing SSA values before `EdtLowering`.

## Target Contract

Every codelet and EDT has a complete creation-time interface:

- memory roots, memrefs, mutable state, MU tokens, DB handles, and control
  edges are deps;
- scalar firstprivate-style values and small immutable captures are params;
- values reconstructable from deps or params are built inside the codelet;
- no codelet or EDT body may reference an SSA value from above unless it enters
  through an explicit dep or param;
- implicit capture is a verifier error.

## Current Surface

Relevant current files and areas:

- `include/arts/dialect/sde/IR/SdeOps.td`
- `lib/arts/dialect/sde/Transforms/state/codelet/`
- `lib/arts/dialect/core/Conversion/SdeToArts/SdeToArtsPatterns.cpp`
- `lib/arts/dialect/core/Conversion/ArtsToRt/EdtLowering.cpp`
- `include/arts/utils/EdtUtils.h`
- `lib/arts/dialect/core/Transforms/verify/VerifyEdtCreated.cpp`

## Proposed CODIR Shape

Start small:

- `codir.codelet`: isolated body with dep and param region arguments.
- `codir.dep`: memory or control dependency derived from SDE MU/control tokens.
- `codir.param`: scalar immutable parameter.
- `codir.launch`: logical launch carrier before ARTS EDT creation.
- `codir.yield`: explicit results or completion values.

Only add more ops when they remove real complexity from conversion or
verification.

## Phases

### Phase 1: Verifier Before Lowering

- Add a verifier helper that checks a region for above captures.
- Reuse existing value utility helpers instead of duplicating "same value"
  checks locally.
- Add negative tests for implicit scalar, memref, and token captures.

Exit gate:

- invalid codelets fail before ARTS lowering.

### Phase 2: CODIR Op Skeleton

- Add CODIR dialect skeleton and minimal ops.
- Move or mirror `sde.cu_codelet` tests into CODIR tests.
- Keep transitional conversion for cases not yet moved.

Exit gate:

- simple memory dep plus scalar param codelet parses, verifies, and prints
  cleanly.

### Phase 3: SDE-To-CODIR Materialization

- Convert SDE MU tokens into CODIR deps.
- Convert scalar captures into CODIR params.
- Rewrite codelet body arguments to use CODIR dep/param block arguments.
- Reject unresolved `sde.mu_dep` at the boundary.

Exit gate:

- current `convert-sde-codelet` coverage has CODIR equivalents.

### Phase 4: CODIR-To-ARTS EDT Creation

- Lower CODIR deps to ARTS DB acquires or control edges.
- Lower CODIR params to EDT params.
- Lower `codir.codelet` to `arts.edt` with complete dep/param metadata.

Exit gate:

- `arts.edt` creation sites enumerate all deps and params.

### Phase 5: EdtLowering Simplification

- Remove capture recovery from `EdtLowering`.
- Make `EdtLowering` consume explicit EDT dep/param metadata only.
- Add verifier checks so any missing dep/param is rejected before ABI lowering.

Exit gate:

- `EdtLowering` tests prove that implicit above values are rejected and explicit
  deps/params lower mechanically.

## Verification

- CODIR parser/printer lit tests.
- CODIR invalid capture lit tests.
- SDE-to-CODIR and CODIR-to-ARTS conversion lit tests.
- Existing EDT lowering tests updated to assert explicit dep/param ABI.
- Focused e2e sample with a memory dep plus scalar firstprivate capture.
