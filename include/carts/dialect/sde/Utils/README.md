# SDE Utils Headers

Reusable SDE-only helper declarations belong here when they express source
semantics, structured access interpretation, PatternAnalysis facts, MU/CU/SU
planning, or target-neutral scheduling intent.

`SDECostModel.h` is the current header-only utility interface in this tier.

Do not place CODIR codelet ABI helpers, ARTS runtime object helpers, or
ARTS-RT runtime-call lowering helpers here.
