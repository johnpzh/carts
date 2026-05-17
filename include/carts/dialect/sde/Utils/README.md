# SDE Utils Headers

Reusable SDE-only helper declarations belong here when they express source
semantics, structured access interpretation, PatternAnalysis facts, MU/CU/SU
planning, or target-neutral scheduling intent.

Current utility groups:

- `SDECostModel.h` carries target-neutral scheduling cost inputs.
- `IterationSizingUtils.h` builds SDE trip-count and logical-worker sizing
  values used by SDE scheduling, tiling, and distribution passes.

Do not place CODIR codelet ABI helpers, ARTS runtime object helpers, or
ARTS-RT runtime-call lowering helpers here.
