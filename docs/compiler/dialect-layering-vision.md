# CARTS Dialect Layering

This document records the current dialect contract. The live pipeline in
`tools/compile/Compile.cpp` and `dekk carts pipeline --json` remains the source
of truth when pipeline details change.

## One-Line Rule

SDE describes source semantics, logical work, and the task/data shape. Core ARTS
materializes abstract ARTS objects when SDE has not already authored them
directly. RT maps those objects to runtime API calls.

## SDE

SDE owns OpenMP semantics, structured program analysis, tensor/linalg facts,
scheduling-unit intent, approved `sde.pattern` facts, reductions, task-body
isolation, and the memory-unit plan that makes those tasks address the right
data slices.

SDE may contain:

- `mu_data`, `mu_dep`, `mu_token`: memory-unit handles. `mu_data` names data
  that crosses asynchronous work, and `mu_token` is the canonical SDE access
  token that lowers almost 1:1 to a Core `arts.db_acquire`.
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
tile geometry, or dependency-window policy by inspecting task bodies.

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
4. Legacy raw-memref `su_iterate` lowering may emit explicit Core dependency
   controls from the SDE physical owner plan. `CreateDbs` is then a
   compatibility bridge that creates the DB object and rewrites raw accesses; it
   must not infer tensor partition policy.
5. `VerifySdeLowered` and `VerifyCoreObjectsOnly` enforce the boundary.
6. Core DB/EDT/epoch passes refine concrete object shape.
7. RT lowering maps Core objects to runtime-facing calls.

Generated `scf.for` operations are implementation control flow only. They are
not a semantic carrier.

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
