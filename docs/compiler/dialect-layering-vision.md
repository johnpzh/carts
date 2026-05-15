# CARTS Dialect Layering

This document records the intended CARTS dialect contract. The live pipeline in
`tools/compile/Compile.cpp` and `dekk carts pipeline --json` remains the source
of truth for the current implementation.

For execution order, milestones, and subplans, start with
[`master-plan.md`](./master-plan.md).
For the target source tree layout, see
[`plans/folder-reorganization.md`](./plans/folder-reorganization.md).
For per-dialect analysis and optimization ownership, see
[`dialects/`](./dialects/) and
[`plans/subdialect-analysis-optimization.md`](./plans/subdialect-analysis-optimization.md).

## One-Line Rule

SDE proves source semantics and authors the MU/CU/SU plan. CODIR freezes that
plan into isolated codelets with explicit deps and params. `arts` binds those
codelets to the abstract ARTS DB/EDT/epoch machine. `arts-rt` lowers the
abstract machine shape to runtime-facing calls.

## Target Stack

The target compiler stack is:

```text
Polygeist -> sde -> codir -> arts -> arts-rt -> LLVM
```

The names are part of the contract:

- `sde` is a CARTS semantic-decomposition dialect. It is not an ARTS dialect.
- `codir` is a CARTS codelet dialect. It is not an ARTS dialect.
- `arts` is the abstract ARTS-machine dialect. It replaces the documentation
  term "Core ARTS"; the current source tree may still use `core/` while the
  rename is staged.
- `arts-rt` is the runtime-facing ARTS bridge. It replaces the documentation
  term "RT"; the textual dialect may continue to use `arts_rt` where MLIR
  syntax requires an underscore.

Namespace cleanup should follow the same rule. The current implementation still
uses names such as `mlir::arts::sde` and paths under `lib/arts/dialect/sde`.
That is an implementation artifact, not the target ownership model. New design
work should keep SDE and CODIR as CARTS-owned dialects and reserve `arts` for
the abstract ARTS machine.

The codelet split is deliberate. Codelet IR is useful enough to stand on its
own: it is the place where isolated task bodies, token-local memory views,
scalar params, and explicit dependency lists are verified before any ARTS EDT
object exists. The IEEE codelet/CODIR paper referenced by issue discussion is
useful prior art, but CARTS should keep the dialect minimal and shaped around
the contracts below.

## SDE

SDE owns OpenMP semantics, structured program analysis, memref access facts,
scheduling-unit intent, approved `sde.pattern` facts, reductions, barrier and
CPS legality, and the memory-unit plan that makes compute units address the
right data slices.

SDE may contain:

- `mu_data`, `mu_alloc`, `mu_dep`, and `mu_token`: memory-unit roots, backing
  storage declarations, source dependency slices, and canonical access-window
  tokens.
- `cu_region`: compute-unit regions before final codelet formation.
- `su_iterate` and `su_barrier`: scheduling-unit iteration spaces, task-shape
  plans, and source-level synchronization intent.
- `resource_query <logical_workers>`: target-neutral logical execution
  capacity for symbolic grain arithmetic.
- Logical work-plan attrs for pattern classification, chunks, access windows,
  reductions, orchestration groups, and diagnostics.

SDE must not contain ARTS-machine concepts: node count, workers per node,
routes, current node/worker, ARTS runtime topology queries, depv layout, DB
pointer layout, concrete EDT placement, or runtime API decisions.

SDE also should not own the final codelet ABI. The current `sde.cu_codelet`
operation is a migration surface and proof vehicle. In the target stack, SDE
authors the MU/CU/SU plan, then SDE-to-CODIR materialization creates isolated
`codir` codelets from that plan.

The names are intentional:

- **MU** means memory unit: data roots, tokens, slices, and access modes.
- **CU** means compute unit: executable computation before ARTS binding.
- **SU** means scheduling unit: iteration spaces, barriers, task topology, and
  orchestration intent. `su_barrier` is a synchronization operation inside this
  scheduling layer.
- **SCU** is not a current dialect unit. If CARTS later needs a scheduling
  control abstraction, it should be introduced deliberately. Do not overload SU
  with ARTS-machine control, and do not move ARTS scheduling mechanics into
  SDE.

## CODIR

CODIR is the codelet dialect. It is the bridge between SDE semantic planning
and ARTS object materialization.

CODIR owns:

- isolated codelet bodies;
- the complete codelet boundary: memory deps, control deps, scalar params,
  yielded values, and local-only values;
- token-local memref views and body rewrites derived from the SDE MU/CU/SU
  plan;
- verification that codelets do not close over values from enclosing regions;
- codelet-local canonicalization that is independent of the ARTS runtime.

CODIR may contain operations such as:

- `codir.codelet`: an `IsolatedFromAbove` body whose region arguments are only
  declared deps and params.
- `codir.launch` or equivalent launch carrier: the scheduling edge that binds a
  codelet to a logical work item before ARTS EDT materialization.
- `codir.dep`: memory/control dependency operands derived from SDE MU tokens
  and control tokens.
- `codir.param`: scalar, immutable, firstprivate-style values.
- `codir.yield`: explicit result or completion values.

The exact op names are design points. The contract is not: every value used by
a codelet must be local, a dep, a param, or a result of a dep/param-local op.
Memrefs and mutable shared state are deps, not params. Scalars and small
immutable captures are params. Values that can be reconstructed inside the
codelet should be reconstructed inside the codelet.

CODIR must not allocate ARTS DBs, choose ARTS worker routes, create ARTS EDTs,
or depend on runtime topology. It consumes the SDE plan and produces a
runtime-independent isolated codelet graph.

## ARTS

`arts` is the abstract ARTS-machine dialect. It mirrors runtime concepts
without exposing runtime ABI calls.

ARTS may contain:

- `arts.edt`: concrete abstract EDT objects created from isolated CODIR
  codelets.
- `arts.db_*`: DB allocation, acquire, release, ref, mode, layout, and access
  windows.
- `arts.epoch_*`: abstract epoch grouping, waits, continuation, and CPS shape.
- typed contract metadata on EDTs, DBs, and epochs while those contracts are
  being materialized or validated.
- ARTS topology and placement queries selected after SDE/CODIR have provided a
  logical work plan.
- local `scf.for` control flow used to implement dispatch or task-local loops.

ARTS must not contain source-level OpenMP carriers, semantic loop-family
rediscovery, loop fusion policy, SDE-style distribution planning, or
pass-local hardcoded string contracts.

The ARTS dialect binds logical worker lanes to the ARTS abstract machine after
SDE and CODIR have produced a logical work/codelet plan. If an ARTS pass needs
to infer owner dims, tile legality, task dependence legality, or codelet
captures from raw source-shaped regions, the missing fact belongs earlier.

## ARTS-RT

`arts-rt` is the runtime ABI bridge.

ARTS-RT may contain:

- EDT create calls.
- dependency record calls.
- state and parameter packing.
- depv addressing.
- DB pointer/GUID GEPs.
- epoch runtime calls.
- low-level cleanup needed before LLVM lowering.

ARTS-RT must not choose task grain, stencil layout, loop distribution, DB
layout, or epoch topology from semantic facts. Those decisions must already be
fixed by SDE, CODIR, and ARTS.

## EDT Isolation Contract

Every EDT must be isolated from enclosing SSA values before `EdtLowering`.
EDT creation must enumerate the complete dependency list and the complete
parameter list.

The rule is:

- memory roots, memrefs, mutable shared state, DB handles, token windows, and
  control dependencies are deps;
- scalar firstprivate-style values and small immutable captures are params;
- loop IVs and constants used by the EDT are either params or reconstructed
  inside the body;
- no operation inside an EDT body may reference an SSA value defined above the
  EDT unless that value is passed through a declared dep or param;
- lowering fails if it finds an implicit capture.

This makes `EdtLowering` simpler. It should lower an already-isolated ARTS EDT
object by emitting runtime params, dep records, and the body function ABI. It
should not rediscover captures, infer missing deps, or repair codelet
boundaries. CODIR is the verifier-enforced staging point that makes this true.

## Current Flow

The live implementation has not reached the target stack yet. Today the
`openmp-to-arts` stage performs SDE planning and then calls `ConvertSdeToArts`
directly:

```text
ConvertOpenMPToSde
PatternAnalysis
SDE transforms
MemoryUnitMaterialization
ConvertSdeToArts
VerifySdeLowered
VerifyCoreObjectsOnly
```

The current boundary has two paths:

1. Canonical MU/CU form lowers directly: `sde.mu_data` or `sde.mu_alloc`
   becomes `arts.db_alloc`, `sde.mu_token` becomes `arts.db_acquire`, and the
   transitional `sde.cu_codelet` becomes `arts.edt`.
2. Remaining raw-memref `su_iterate`/`cu_task` lowering is a compatibility
   bridge for cases that do not yet have token/codelet-local form. It is
   coarse-only; blocked/tiled physical-layout attrs must be consumed by
   SDE/CODIR token-local rewrite before ARTS.

`CreateDbs` is currently the compatibility bridge for raw memrefs that have not
yet been canonicalized into MU tokens and codelets. It may allocate a coarse
whole-storage ARTS DB and redirect direct raw load/store uses through
`db_ref[0]`. It must not choose owner dims, tile geometry, dependency-window
policy, or block-local coordinates by inspecting
task bodies. This is a temporary migration surface, not the target
architecture.

Generated `scf.for` operations are implementation control flow only. They are
not a semantic carrier.

## Target Flow

The production target is the direct SDE/CODIR/ARTS path:

```text
sde
  mu_data / mu_alloc    data root and storage intent
  mu_token              mode + memref slice
  cu_region             planned compute body
  su_iterate            task topology
        |
        | ConvertSdeToCodir
        v
codir
  codir.codelet         isolated body
  codir.dep             memory/control deps
  codir.param           scalar params
        |
        | ConvertCodirToArts
        v
arts
  arts.db_alloc         created from MU storage
  arts.db_acquire       created from codelet deps
  arts.edt              created from codir.codelet with explicit deps/params
        |
        | ConvertArtsToRt
        v
arts-rt
  runtime calls
```

ARTS should not have an `arts.db_control` operation. There should not be an ARTS
operation whose only purpose is to preserve user-provided dependency metadata
until a later pass rediscovers what SDE already knew. The replacement is:

- `sde.mu_dep` carries source-level dependency slices while SDE is still
  planning.
- `sde.mu_token` carries canonical memory access windows.
- `sde.control_token` carries ordering/completion only.
- CODIR turns those into explicit codelet deps.
- ARTS turns those deps directly into DB acquires, control edges, and EDT
  creation operands.

## MU/CU/SU Rewrite Rule

Tiling is not valid unless the MU, CU, and SU all agree.

A blocked or sliced MU is not a drop-in replacement for the original whole
memref. A local payload view uses coordinates relative to the slice, while the
source program's memref indices are usually global element coordinates.
Therefore SDE planning and SDE-to-CODIR materialization must do all pieces
together:

- choose the CU/SU tile and task schedule;
- choose the MU storage layout and token dependency window;
- rewrite codelet/body accesses so they address the token-local view, including
  ND owner dimensions, halo windows, and strided memref views.

Simple one-dimensional owner slices are covered by the current memref-native
boundary path: the boundary consumes SDE owner facts, creates the acquire
window, and keeps the task body on full MU payload coordinates. That is a
correctness bridge, not the final performance architecture. The final path is
token-local CODIR codelet form, then direct ARTS DB/acquire/EDT lowering.

## Work-Plan Contract

The SDE plan on a work unit should include:

- work family: elementwise, stencil, matmul, reduction, wavefront, Jacobi, or
  explicit unsupported diagnostic;
- logical worker capacity or requested logical lanes, expressed as SDE plan
  attrs or `sde.resource_query <logical_workers>`;
- iteration domain: rank, bounds, steps, owner dims, spatial dims, and local
  task-loop shape;
- schedule: source intent and selected logical chunking;
- access plan: per-root read/write mode, offsets, sizes, halo offsets, write
  footprint, owner dims, disjointness proof, and self-read status;
- physical data plan: memory roots, MU token slices, block shape, halo shape,
  layout request, and root data value;
- reduction plan: accumulator, kind, identity, strategy, partial storage
  request, and final exposure;
- orchestration plan: barrier status, timestep/wave group, repetition
  structure, async strategy, and CPS/continuation preference;
- codelet boundary plan: deps, params, token-local views, local values, yielded
  results, and diagnostic reason if no codelet can be formed.

The capture rule is explicit:

- scalar firstprivate-style values can become CODIR params and then EDT params;
- dynamic arrays, memrefs, and mutable shared state become deps;
- values that can be constructed locally inside a codelet should be constructed
  locally rather than captured.

## Migration Plan

1. Add CODIR as a CARTS-owned dialect with an isolated `codir.codelet` contract,
   explicit dep/param lists, and verifier coverage for no implicit captures.
2. Move the codelet-specific SDE operations, tests, and passes into CODIR or
   split them so SDE authors the plan and CODIR materializes/verifies the
   isolated codelet boundary.
3. Teach SDE-to-CODIR materialization to rewrite MU/CU/SU together for memref
   access windows, including ND owner dims, strided accesses, and halo windows.
4. Convert CODIR to ARTS directly: MU storage to DB allocation, codelet memory
   deps to DB acquires, control deps to ARTS ordering, params to EDT params, and
   isolated codelet bodies to EDT bodies.
5. Strengthen ARTS EDT verification and `EdtLowering` so creation-time deps and
   params are mandatory and implicit above captures are rejected.
6. Remove tensor raising/lowering paths once the memref MU/token/CODIR path
   covers task deps, reductions, and codelet-local state.
7. Keep Core raw DB indexers removed, keep `CreateDbs` coarse-only during the
   transition, and then remove it after every supported benchmark reaches
   direct CODIR/ARTS materialization.
8. Keep scope and placement selection out of SDE. SDE may request logical
   capacity; ARTS decides abstract-machine placement; ARTS-RT lowers the chosen
   runtime API shape.

## Placement Rules

- Decisions about source meaning, legality, reductions, chunking, data layout,
  and distribution intent live in SDE.
- Codelet isolation, token-local access rewriting, deps, and params live in
  CODIR.
- DB/EDT/epoch object materialization and ARTS-machine binding live in `arts`.
- Runtime ABI mapping lives in `arts-rt`.
- If a pass needs to recover source semantics from ARTS implementation loops,
  the required fact belongs in SDE or CODIR instead.
- If a utility is reusable, put it in the owning utility namespace instead of a
  local pass helper.
