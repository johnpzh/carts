# ARTS-RT Analyses

ARTS-RT analyses operate on runtime-call-shaped IR after ARTS has fixed object
shape.

Owned analyses:

- runtime call purity and hoistability;
- EDT parameter packing shape;
- state packing shape;
- dependency slot and depv addressing locality;
- DB pointer/GUID access locality;
- alias/noalias opportunities for runtime-facing data pointers;
- launch, continuation, and epoch runtime-call overhead accounting.

Emitted facts:

- runtime-call hoisting candidates;
- scalar replacement candidates;
- alias/noalias metadata plans;
- pointer and depv locality facts;
- diagnostics when runtime ABI overhead is the remaining bottleneck.

ARTS-RT analysis must not compensate for missing SDE/CODIR/ARTS object shape.
