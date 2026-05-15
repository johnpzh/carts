# ARTS-RT Optimizations

ARTS-RT optimizations are ABI and LLVM-facing cleanup passes.

Owned optimizations:

- EDT launch and continuation overhead cleanup;
- `edt_param_pack` and `state_pack` cleanup;
- dependency slot and depv pointer hoisting;
- DB pointer/GUID GEP cleanup;
- runtime call hoisting;
- scalar replacement;
- data pointer hoisting;
- alias scope generation;
- loop vectorization metadata after runtime shape is fixed.

Rules:

- Do not alter task grain or DB layout.
- Do not infer source dependencies.
- Do not recover codelet captures.
- Run these optimizations only after SDE/CODIR/ARTS shape is correct and traces
  show ABI overhead.

Exit facts for LLVM:

- lowered runtime calls;
- simplified scalar/pointer traffic;
- alias and vectorization metadata;
- verified pre-lowered runtime shape.
