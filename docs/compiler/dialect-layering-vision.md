# CARTS Dialect Layering

This document records the current dialect contract. The live pipeline in
`tools/compile/Compile.cpp` and `dekk carts pipeline --json` remains the source
of truth when pipeline details change.

## One-Line Rule

SDE describes source semantics and logical work. Core ARTS materializes abstract
ARTS objects: EDTs, DBs, and epochs. RT maps those objects to runtime API calls.

## SDE

SDE owns OpenMP semantics, structured program analysis, tensor/linalg facts,
scheduling-unit intent, dependency families, reductions, and task-body
isolation.

SDE may contain:

- `cu_region`: source compute regions such as `parallel`, `single`, and `task`.
- `su_iterate`: worksharing/taskloop iteration spaces.
- `su_barrier`: source synchronization and SDE barrier decisions.
- `mu_data`, `mu_dep`, `mu_token`: high-level data and dependency handles.
- `cu_codelet`: isolated task bodies with explicit tokens and captures.
- Logical work-plan attrs for chunks, access windows, reductions,
  orchestration groups, and fallback diagnostics.

SDE must not contain ARTS-machine concepts: node count, workers per node,
routes, current node/worker, runtime topology queries, depv layout, DB pointer
layout, or placement decisions.

SDE may use a machine-neutral logical capacity such as requested workers or
logical lanes. Core decides whether those lanes become local workers, nodes, or
a two-level ARTS mapping.

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

The SDE/Core boundary is direct materialization:

1. OpenMP becomes SDE `su_iterate`/CU/MU structure.
2. SDE planning selects work family, scheduling, chunking, access windows,
   distribution intent, and reduction strategy.
3. `ConvertSdeToArts` materializes Core EDT/DB/epoch structure directly from
   the SDE plan.
4. `VerifySdeLowered` and `VerifyCoreObjectsOnly` enforce the boundary.
5. Core DB/EDT/epoch passes refine concrete object shape.
6. RT lowering maps Core objects to runtime-facing calls.

Generated `scf.for` operations are implementation control flow only. They are
not a semantic carrier.

## Work-Plan Contract

The SDE plan on a work unit should include:

- work family: elementwise, stencil, matmul, reduction, wavefront, Jacobi, or
  explicit unsupported diagnostic;
- logical worker count or requested logical lanes;
- iteration domain: rank, bounds, steps, owner dims, spatial dims, and local
  task-loop shape;
- schedule: source intent and selected logical chunking;
- access plan: per-root read/write mode, offsets, sizes, halo offsets, write
  footprint, owner dims, disjointness proof, and self-read status;
- physical data plan: block shape, halo shape, DB layout request, and root data
  value;
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
- Realization of DB/EDT/epoch object shape and ARTS topology binding lives in
  Core.
- Runtime ABI mapping lives in RT.
- If a pass needs to recover source semantics from Core implementation loops,
  the required fact belongs in SDE instead.
- If a utility is reusable, put it in the owning utility namespace instead of a
  local pass helper.
