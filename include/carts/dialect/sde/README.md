# SDE Include Target

Target home for the SDE dialect public headers and TableGen files.

SDE owns source semantics, PatternAnalysis facts, MU/CU/SU planning, reductions,
barrier/CPS legality, and target-neutral logical resource requests. It must not
own final codelet ABI, ARTS topology, DB pointer layout, or runtime calls.
