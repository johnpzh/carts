# ARTS Analyses

ARTS analyses operate on abstract ARTS objects after SDE/CODIR have fixed the
semantic plan and codelet ABI.

Owned analyses:

- DB analysis: DB roots, modes, aliases, physical layout validation, and
  acquire-window consistency;
- EDT analysis: EDT graph, deps per EDT, params per EDT, launch shape, and
  distribution metadata;
- epoch analysis: epoch grouping, waits, continuations, and CPS materialization
  shape;
- dependency-slot analysis: local dependency slots, DB acquire projection, and
  dep window mapping;
- resource binding analysis: maps logical worker requests to ARTS topology;
- distributed ownership analysis: maps planned owner blocks to ARTS nodes and
  routes without changing the plan.

Emitted facts:

- ARTS DB/EDT/epoch attrs and operands;
- dependency slot layouts;
- validated DB/acquire/window contracts;
- placement/resource binding attrs;
- diagnostics when SDE/CODIR did not provide enough plan information.

ARTS analysis must validate and refine mechanics. It must not invent missing
source policy.
