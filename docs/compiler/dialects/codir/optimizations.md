# CODIR Optimizations

CODIR optimizations are codelet-local and runtime-neutral.

Owned optimizations:

- scalar capture normalization;
- local reconstruction of values that should not be params;
- token mode refinement;
- codelet-local scalar forwarding;
- codelet-local CSE/canonicalization;
- generic codelet-local invariant cleanup;
- token-local memref access simplification;
- dead dep/param removal when verification proves no use.

Rules:

- Do not choose owner dims or tile sizes.
- Do not allocate ARTS DBs or create ARTS EDTs.
- Do not access runtime topology.
- Do not use tensor cleanup as a recovery path; codelet deps stay memref/token based.

Exit facts for ARTS:

- complete deps;
- complete params;
- isolated body;
- token-local views;
- yielded completion/data values.
