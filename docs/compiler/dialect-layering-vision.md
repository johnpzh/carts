# CARTS Dialect Layering

This document records the current dialect contract. The live pipeline in
`tools/compile/Compile.cpp` and `dekk carts pipeline --json` remains the source
of truth when pipeline details change.

## One-Line Rule

SDE describes source semantics, logical work, memory-unit dependencies, and the
task/data shape. Core ARTS binds that already-authored shape to the abstract
ARTS machine. RT maps Core objects to runtime API calls.

## SDE

SDE owns OpenMP semantics, structured program analysis, tensor/linalg facts,
scheduling-unit intent, approved `sde.pattern` facts, reductions, task-body
isolation, and the memory-unit plan that makes those tasks address the right
data slices.

SDE may contain:

- `mu_data`, `mu_dep`, `mu_token`: memory-unit handles. `mu_data` names data
  that crosses asynchronous work, `mu_dep` names a source-level memref
  dependency slice from task depend clauses or SDE owner-slice planning, and
  `mu_token` is the canonical access token that lowers almost 1:1 to a Core
  `arts.db_acquire`.
- `cu_region` and `cu_codelet`: compute-unit regions and isolated task bodies.
  A `cu_codelet` consumes only MU tokens plus explicit scalar captures.
- `su_iterate`: scheduling-unit iteration spaces and task-shape plans.
- `su_barrier`: synchronization inside the scheduling-unit layer.
- `resource_query <logical_workers>`: target-neutral logical execution capacity
  for symbolic SDE grain arithmetic.
- Logical work-plan attrs for chunks, access windows, reductions,
  orchestration groups, and fallback diagnostics.

SDE must not contain ARTS-machine concepts: node count, workers per node,
routes, current node/worker, ARTS runtime topology queries, depv layout, DB
pointer layout, or placement decisions.

The names are intentional:

- **MU** means memory unit: data roots, tokens, slices, and access modes.
- **CU** means compute unit: the executable body and scalar captures.
- **SU** means scheduling unit: iteration spaces, barriers, task topology, and
  orchestration intent. `su_barrier` is a synchronization operation inside this
  scheduling layer.
- **SCU** is not a current dialect unit. If CARTS later needs a separate
  scheduling-control abstraction, it should be introduced deliberately instead
  of overloading SU or making SDE aware of ARTS runtime mechanics.

SDE may use machine-neutral logical capacity through
`sde.resource_query <logical_workers>` or compile-time logical-capacity
estimates. Core decides whether those lanes become ARTS runtime workers, static
workers, nodes, or a two-level ARTS mapping.

## Core ARTS

Core is the abstract ARTS-machine layer. It mirrors runtime objects without
exposing runtime ABI calls.

Core may contain:

- `arts.edt`: concrete EDT objects, not source-level parallel regions.
- `arts.db_*`: DB allocation, acquire, release, ref, mode, layout, and access
  windows.
- `arts.epoch_*`: abstract epoch grouping, waits, continuation, and CPS
  structure.
- typed contract metadata on EDTs, DBs, and epochs while those contracts are
  being materialized or validated.
- Core runtime topology queries selected by Core, never by SDE.
- local `scf.for` control flow used to implement dispatch or task-local loops.

`CreateDbs` is currently the compatibility bridge for raw memrefs that have not
yet been canonicalized into MU tokens. It may consume explicit SDE-authored DB
layout and dependency-slice metadata, allocate the corresponding Core DB, and
rewrite remaining raw memref accesses. It must not choose tensor owner dims,
tile geometry, or dependency-window policy by inspecting task bodies. This is a
temporary migration surface, not the target architecture.

Core must not contain source-level parallel carriers, OpenMP semantics,
semantic loop-family rediscovery, loop fusion policy, SDE-style distribution
planning, or pass-local hardcoded string contracts.

Core binds logical worker lanes to the ARTS abstract machine after SDE has
produced a logical work plan.

## RT

RT is the runtime ABI bridge.

RT may contain:

- EDT create calls.
- dependency record calls.
- state and parameter packing.
- depv addressing.
- DB pointer/GUID GEPs.
- epoch runtime calls.
- low-level cleanup needed before LLVM lowering.

RT must not choose task grain, stencil layout, loop distribution, DB layout, or
epoch topology from semantic facts. Those decisions must already be fixed by
SDE and Core.

## Current Flow

The SDE/Core boundary has two paths:

1. OpenMP becomes SDE `su_iterate`/CU/MU structure.
2. SDE planning selects work family, scheduling, chunking, access windows,
   distribution intent, and reduction strategy.
3. Canonical SDE MU/CU form lowers directly: `sde.mu_data` becomes
   `arts.db_alloc`, `sde.mu_token` becomes `arts.db_acquire`, and
   `sde.cu_codelet` becomes `arts.edt`.
4. Legacy raw-memref `su_iterate`/`cu_task` lowering may emit explicit Core
   dependency controls from the SDE physical owner plan. `CreateDbs` is then a
   compatibility bridge that creates the DB object and rewrites raw accesses; it
   must not infer tensor partition policy.
5. `VerifySdeLowered` and `VerifyCoreObjectsOnly` enforce the boundary.
6. Core DB/EDT/epoch passes refine concrete object shape.
7. RT lowering maps Core objects to runtime-facing calls.

Generated `scf.for` operations are implementation control flow only. They are
not a semantic carrier.

## Boundary Diagram

The production target is the direct MU/token/codelet path:

```text
SDE
  sde.mu_data        data root and layout intent
  sde.mu_token       mode + tensor slice
  sde.cu_codelet     isolated compute body
        |
        | ConvertSdeToArts
        v
Core
  arts.db_alloc      created directly from mu_data
  arts.db_acquire    created directly from mu_token
  arts.edt           created directly from cu_codelet
        |
        | Core/RT lowering
        v
Runtime calls
```

The remaining compatibility path exists only when a `su_iterate` body still
reaches Core as raw memref loads/stores before codelet formation:

```text
SDE raw-memref fallback
  sde.su_iterate attrs: pattern, physicalOwnerDims, physicalBlockShape
        |
        | ConvertSdeToArts
        v
Core DB form
  arts.db_alloc      physical DB layout from SDE attrs
  arts.db_acquire    dependency for the raw external memref capture
  arts.db_ref        raw memref body rewritten to DB-relative indices
```

Core no longer has an `arts.db_control` operation. `sde.control_token` is an
SDE ordering/completion edge consumed by SDE synchronization planning; it must
not describe memory roots or DB slices. `sde.mu_dep` is also SDE-only: if it
still feeds a `cu_task` at `ConvertSdeToArts`, conversion fails instead of
creating a Core marker.

## Removed Core Dependency Markers

The SDE replacement is not a new Core op. The SDE-level representation is:

- `sde.mu_dep` carries user-provided memref dependency slices from
  `omp.task depend(...)` and any SDE-authored owner-slice dependency that has
  not yet become a concrete token.
- `sde.mu_token` carries the canonical memref/codelet dependency slice.
- `sde.control_token` carries ordering/completion only.

Every dependency-bearing raw-memref path must produce canonical SDE MU/CU form
before `ConvertSdeToArts`:

1. SDE must attach every shared data root to an MU root. Function arguments,
   globals, allocas, and backed memrefs need a single `mu_data`/backing handle
   that downstream passes can identify without scanning Core memrefs.
2. SDE must materialize ND dependency slices as MU tokens. Offsets, sizes,
   halo windows, pinned indices, owner dims, and physical block shape must stay
   in SDE element space until the boundary.
3. SDE must align CU bodies with MU layout. A tiled `su_iterate` without
   matching MU address-space rewrite is not a real tiling transformation.
   Loads/stores inside the codelet must already address token-local views or be
   explicitly rewritten before Core.
4. SDE must form isolated `cu_codelet` bodies for task and loop dispatch shapes,
   including the current raw `su_iterate` inside `<parallel>` case and
   `cu_task deps(...)` case.
5. `ConvertSdeToArts` must lower that form directly: `mu_data` to `db_alloc`,
   `mu_token` to `db_acquire`, and `cu_codelet` to `edt`, including ND token
   offsets/sizes and release placement.
6. Core may then run DB/EDT/epoch analyses and optimizations over concrete
   objects, but it must not rediscover DB roots, task slices, tensor owner dims,
   or tiling policy from raw memrefs.

The Core op is gone. The remaining migration work is to cover `su_iterate`,
`cu_task`, task depend slices, function-argument memrefs, globals, reductions,
and ND/stencil windows with SDE tokens/codelets, then shrink the raw-memref
portions of `CreateDbs` and the compatibility indexers.

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
  DB layout request, and root data value;
- reduction plan: accumulator, kind, identity, strategy, partial storage
  request, and final exposure;
- orchestration plan: barrier status, timestep/wave group, repetition
  structure, async strategy, and CPS/continuation preference;
- codelet boundary: tokens, scalar captures, local values, yielded results, and
  fallback reason if no codelet can be formed.

The capture rule is explicit:

- scalar firstprivate-style values can become EDT parameters;
- dynamic arrays, memrefs, tensors, and mutable shared state become
  dependencies;
- values that can be constructed locally inside a codelet should be constructed
  locally rather than captured.

## Placement Rules

- Decisions about source meaning, legality, reductions, chunking, and
  distribution intent live in SDE.
- Realization of DB/EDT/epoch object shape lives in SDE when canonical
  MU/token/codelet form is available; Core handles remaining raw-memref
  compatibility and ARTS topology binding.
- Runtime ABI mapping lives in RT.
- If a pass needs to recover source semantics from Core implementation loops,
  the required fact belongs in SDE instead.
- If a utility is reusable, put it in the owning utility namespace instead of a
  local pass helper.
