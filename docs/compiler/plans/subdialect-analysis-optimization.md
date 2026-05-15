# Subdialect Analysis And Optimization Plan

## Objective

Make every CARTS dialect responsible for its own analyses and optimizations.
No dialect should depend on hidden analysis objects from another dialect, and no
downstream dialect should rediscover facts that an upstream dialect could have
materialized explicitly.

## Rule

Each dialect owns five local areas:

```text
IR/
Analysis/
Transforms/
Conversion/
Verify/
Utils/
```

The `Analysis/` directory answers questions that are meaningful for that
dialect's IR. The `Transforms/` directory performs optimizations whose legality
can be proven from that dialect's IR and documented incoming contract.
The `Utils/` directory contains reusable dialect-local helper APIs that are too
small or mechanical to be analyses but too broadly meaningful to live inside a
single pass. Utility placement is governed by
[`utility-ownership.md`](./utility-ownership.md).

## Dialect Ownership Matrix

| Dialect | Analysis owns | Optimizations own |
|---|---|---|
| SDE | Source legality, memref roots/access maps, dependence windows, reductions, pattern facts, logical capacity. | Tiling, fusion, loop/order transforms, distribution intent, barrier/CPS planning, reduction strategy, MU materialization. |
| CODIR | Codelet capture ABI, dep/param shape, token-local access, codelet-local effects. | Codelet-local canonicalization, scalar capture cleanup, token mode refinement, local invariant cleanup, dead dep/param removal. |
| ARTS | DB/EDT/epoch graphs, acquire windows, dependency slots, resource binding, distributed ownership. | DB mode tightening, EDT/epoch orchestration, dependency slot refinement, contract validation, placement refinement. |
| ARTS-RT | Runtime-call purity, packing shape, depv/pointer locality, alias facts, launch overhead accounting. | Runtime-call hoisting, packing cleanup, scalar replacement, pointer hoisting, alias metadata, LLVM-facing cleanup. |

## Boundary Rule

Analysis results cross dialect boundaries only when materialized in IR:

- SDE facts become SDE attrs, MU tokens, plan operands, and reduction/CPS
  contracts.
- CODIR facts become explicit deps, params, token-local views, yielded values,
  and verifier-clean isolated regions.
- ARTS facts become DB/EDT/epoch attrs, dependency slots, placement/resource
  attrs, and object graph structure.
- ARTS-RT facts become runtime ABI operands, metadata, and LLVM-facing attrs.

Do not pass C++ analysis objects across dialect boundaries as an implicit API.

## Documentation Layout

Target docs live under:

```text
docs/compiler/dialects/
  sde/
    README.md
    analysis.md
    optimizations.md
  codir/
    README.md
    analysis.md
    optimizations.md
  arts/
    README.md
    analysis.md
    optimizations.md
  arts-rt/
    README.md
    analysis.md
    optimizations.md
```

Existing compatibility docs under `docs/compiler/sde`, `docs/compiler/core`,
and `docs/compiler/rt` should either link to this target layout or be folded
into it during migration. Avoid duplicating the same ownership rules in many
places.

## Migration Phases

### Phase 1: Documentation Contract

- Add per-dialect analysis and optimization docs.
- Link the master plan and dialect layering docs to the target docs index.
- Mark old docs as compatibility/migration notes where needed.

Exit gate:

- every dialect has documented analysis and optimization ownership.

### Phase 2: Source Skeleton

- Make `Analysis/` and `Transforms/` folders explicit for every target dialect
  under `include/carts/dialect` and `lib/carts/dialect`.
- Keep source files in old locations until a buildable slice is ready.

Exit gate:

- target folders are trackable in git and documented.

### Phase 3: Move Analyses First

- Move analyses before transforms when possible, because transforms should
  depend on dialect-local analysis APIs.
- SDE analysis moves without codelet capture logic.
- CODIR analysis starts with capture/deps/params/token-local access.
- ARTS analysis starts with DB/EDT/epoch object graphs.
- ARTS-RT analysis starts with runtime-call and packing locality.

Exit gate:

- each moved transform uses the target dialect's local analysis headers.

### Phase 4: Move Optimizations

- Move transforms only after their owning analysis is in place.
- Delete or split transforms that span multiple ownership layers.
- Keep compatibility wrappers temporary and clearly named.

Exit gate:

- no optimization lives in a downstream dialect because it needed an upstream
  fact that was not materialized in IR.

## Verification

- focused lit tests for each moved analysis/transform;
- `dekk carts build` after include or CMake moves;
- `dekk carts pipeline --json` after pass registration changes;
- `git diff --check`;
- benchmark evidence only after behavior changes.
