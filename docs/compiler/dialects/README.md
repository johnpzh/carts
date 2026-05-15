# CARTS Dialect Documentation

This is the target documentation root for CARTS compiler dialects. Each dialect
owns its own IR, analyses, optimizations, conversions, and verification. Shared
facts may cross dialect boundaries only as explicit IR attributes, operands, or
types that the receiving dialect documents.

Target stack:

```text
Polygeist -> sde -> codir -> arts -> arts-rt -> LLVM
```

## Dialect Ownership

| Dialect | IR | Analysis | Optimizations | Conversion |
|---|---|---|---|---|
| [`sde`](./sde/README.md) | Source semantic plan: MU/CU/SU, patterns, tokens, barriers. | Source legality, memref access, dependence, reductions, pattern facts. | Tiling, fusion, chunking, distribution intent, barrier/CPS planning, reduction strategy. | OpenMP/Polygeist to SDE, SDE to CODIR. |
| [`codir`](./codir/README.md) | Isolated codelets, deps, params, token-local views. | Capture/dependency ABI, token-local access, codelet-local effects. | Codelet canonicalization, scalar capture cleanup, token mode refinement, local invariant cleanup. | CODIR to ARTS. |
| [`arts`](./arts/README.md) | Abstract ARTS DB/EDT/epoch/dependency objects. | DB/EDT/epoch graph, dependency windows, placement/resource binding. | DB mode tightening, EDT/epoch orchestration, dependency-slot refinement, contract validation. | ARTS to ARTS-RT. |
| [`arts-rt`](./arts-rt/README.md) | Runtime-call-shaped ABI IR. | Runtime-call purity, packing, pointer/dependency-slot locality, alias facts. | Runtime call hoisting, packing cleanup, scalar replacement, pointer hoisting, LLVM-facing cleanup. | ARTS-RT to LLVM. |

## Boundary Rule

Analysis objects do not cross dialect boundaries as hidden side channels. If a
later dialect needs a fact, the earlier dialect must materialize that fact in
the IR contract before conversion.

Examples:

- SDE pattern facts become `sde.pattern`, owner dims, block shapes, token
  windows, and reduction plans.
- CODIR capture analysis becomes explicit deps, params, yielded values, and
  token-local view operands.
- ARTS resource analysis becomes ARTS object attributes, dependency slots, DB
  layouts, and epoch/EDT structure.
- ARTS-RT lowering facts become runtime ABI operands, metadata, or LLVM-facing
  attributes.

## Documentation Rule

Every dialect documentation directory should contain:

- `README.md`: responsibilities and boundary rules.
- `analysis.md`: analyses owned by the dialect and facts they may emit.
- `optimizations.md`: transformations owned by the dialect and their legality
  inputs.

The source-tree target layout mirrors this structure with `IR/`, `Analysis/`,
`Transforms/`, `Conversion/`, and `Verify/`.
