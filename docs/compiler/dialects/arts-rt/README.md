# ARTS-RT Dialect

ARTS-RT is the runtime ABI bridge. It receives a fixed ARTS DB/EDT/epoch graph
and lowers it to runtime-call-shaped IR before LLVM lowering.

ARTS-RT must not choose semantic task grain, DB layout, owner dims, dependency
legality, or epoch topology.

Primary docs:

- [`analysis.md`](./analysis.md)
- [`optimizations.md`](./optimizations.md)
- Existing migration note: [`../../rt/README.md`](../../rt/README.md)
