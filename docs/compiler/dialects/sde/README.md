# SDE Dialect

SDE is the semantic decomposition dialect. It owns OpenMP/source semantics,
structured memref facts, MU/CU/SU planning, reductions, barrier/CPS legality,
and target-neutral logical resource requests.

SDE must not own codelet ABI, ARTS placement, DB pointer layout, depv layout, or
runtime calls.

Primary docs:

- [`analysis.md`](./analysis.md)
- [`optimizations.md`](./optimizations.md)
- Existing migration notes under [`../../sde/`](../../sde/)
