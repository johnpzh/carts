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
- no codelet or explicit-param EDT body may reference a nonconstant SSA value
  from above unless it enters through an explicit dep or param;
- implicit nonconstant capture is a verifier/lowering error; constant-like
  scalars are rematerializable and do not become ABI params.
- external `memref.alloca` scratch is the narrow exception for explicit-param
  EDTs: it remains legal only as task-local stack storage because ARTS
  structural/lowering passes sink or clone it into the EDT body instead of
  treating it as an ABI dependency.

## Current Surface

Relevant current files and areas:

- `include/carts/dialect/sde/IR/SdeOps.td`
- `lib/carts/dialect/sde/Transforms/state/codelet/`
- `lib/carts/dialect/codir/Conversion/SdeToCodir/SdeToCodir.cpp`
- `lib/carts/dialect/codir/Transforms/VerifyCodir.cpp`
- `lib/carts/dialect/codir/Conversion/CodirToArts/CodirToArts.cpp`
- `lib/carts/dialect/arts-rt/Conversion/ArtsToRt/EdtLowering.cpp` (current
  source-tree path; target lives under the renamed `arts` dialect tree)
- `include/carts/utils/EdtUtils.h`
- `lib/carts/dialect/arts/Transforms/verify/VerifyEdtCreated.cpp` (current
  source-tree path; target lives under the renamed `arts` dialect tree)

## Current CODIR Shape

The live CODIR surface is intentionally small:

- `codir.codelet`: isolated body with dependency operands, scalar param
  operands, matching dep/param region arguments, and dep-mode metadata.
- `codir.yield`: explicit results or completion values.

Separate `codir.dep`, `codir.param`, or `codir.launch` ops should only be added
if they remove real complexity from conversion or verification. The current
production path represents deps and params as `codir.codelet` operands.

## Phases

### Phase 1: Verifier Before Lowering

Status: complete. `verify-codir` enforces operand-shape checks and rejects
implicit above-captures.

- [x] Add a verifier helper that checks a region for above captures.
- [x] Reuse existing value utility helpers instead of duplicating "same value"
  checks locally.
- [x] Add negative tests for implicit scalar and memref captures.

Exit gate:

- invalid codelets fail before ARTS lowering.

### Phase 2: CODIR Op Skeleton

Status: complete for the current codelet surface.

- [x] Add CODIR dialect skeleton and minimal ops.
- [x] Move or mirror `sde.cu_codelet` tests into CODIR tests.
- [x] Remove direct codelet lowering once CODIR covers the path.

Exit gate:

- simple memory dep plus scalar param codelet parses, verifies, and prints
  cleanly.

### Phase 3: SDE-To-CODIR Materialization

Status: complete for the current task/codelet dependency surface.

- [x] Convert SDE MU tokens into CODIR deps.
- [x] Convert scalar captures into CODIR params.
- [x] Rewrite codelet body arguments to use CODIR dep/param block arguments.
- [x] Reject unresolved `sde.mu_dep` at the boundary.

Exit gate:

- current SDE-to-CODIR codelet coverage has CODIR-owned regression tests.

### Phase 4: CODIR-To-ARTS EDT Creation

- [x] Lower CODIR deps to `arts.db_acquire` or control edges, allocating storage
  through `arts.db_alloc` when MU storage materializes here.
- [x] Lower CODIR params to EDT params.
- [x] Lower `codir.codelet` to `arts.edt` with complete dep/param metadata.
- [x] Add `arts.edt params(...)` operand segments and verifier coverage for
  explicit scalar params.

Exit gate:

- `arts.edt` creation sites enumerate all deps and params.

### Phase 5: EdtLowering Simplification

- [x] Remove capture recovery from `EdtLowering`. EDTs now reject nonconstant
  above-captures, including non-scalar values, except for external
  `memref.alloca` stack scratch that is sunk/cloned into the task body. EDTs
  consume explicit param block args directly; direct body uses of explicit
  param operands are rejected by the ARTS EDT verifier.
- [x] Make `EdtLowering` consume explicit EDT dep/param metadata for CODIR
  codelets.
- [x] Add verifier/lowering checks so missing nonconstant scalar params are
  rejected before ABI lowering.
- [x] Preserve positional CODIR explicit ABI slots, including undef-like scalar
  params, so `arts.edt params(...)` stays aligned with dep/param block
  arguments through `EdtLowering`.

Exit gate:

- `EdtLowering` tests prove that implicit above values are rejected and explicit
  deps/params lower mechanically.
- Full e2e suite passes with CODIR-origin explicit-param EDTs.

## Verification

- CODIR parser/printer lit tests.
- CODIR invalid capture lit tests.
- SDE-to-CODIR and CODIR-to-ARTS conversion lit tests.
- Existing EDT lowering tests updated to assert explicit dep/param ABI.
- Focused e2e sample with a memory dep plus scalar firstprivate capture.
- 2026-05-15 evidence: focused explicit-param/CODIR lit tests passed after
  removing scalar-capture recovery and the CODIR explicit ABI marker.
- 2026-05-15 strict-boundary evidence: `dekk carts build` and
  `dekk carts test` passed after removing ARTS EDT verifier-bypass support and
  keeping codelet metadata on CODIR-owned attributes.
