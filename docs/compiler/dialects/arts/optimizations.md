# ARTS Optimizations

ARTS optimizations refine abstract ARTS objects and orchestration mechanics.

Owned optimizations:

- DB mode tightening;
- DB acquire/window refinement from explicit plan facts;
- EDT structural optimization;
- EDT pointer rematerialization;
- dependency-slot localization;
- epoch creation and epoch cleanup;
- contract validation;
- distributed ownership refinement.

Rules:

- Do not infer owner dims, tile sizes, or dependency legality from raw memrefs.
- Do not recover implicit codelet captures.
- Do not move scalar computations across the explicit EDT dep/param ABI.
- Do not introduce `arts.db_control`.
- Keep any remaining `CreateDbs` use coarse-only; blocked/tiled raw memrefs
  must fail at the boundary and be fixed in SDE/CODIR.

Exit facts for ARTS-RT:

- explicit runtime-independent DB/EDT/epoch graph;
- dependency slot layout;
- complete EDT dep and param ABI;
- runtime topology decisions chosen at the ARTS layer.
