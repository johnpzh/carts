# CODIR Planning Notes

CODIR is the proposed CARTS codelet dialect. It sits between SDE and `arts`:

```text
sde -> codir -> arts
```

The purpose is to separate semantic planning from codelet ABI formation. SDE
proves the MU/CU/SU plan. CODIR materializes that plan into isolated codelets
with explicit deps, params, token-local memory views, and verifier-enforced
capture rules. `arts` then lowers the codelets to ARTS DB/EDT/epoch objects.

The target per-dialect docs live under
[`../dialects/codir/`](../dialects/codir/). In particular, CODIR-owned analyses
are listed in [`../dialects/codir/analysis.md`](../dialects/codir/analysis.md)
and CODIR-owned optimizations are listed in
[`../dialects/codir/optimizations.md`](../dialects/codir/optimizations.md).

## Responsibilities

CODIR owns:

- `IsolatedFromAbove` codelet bodies;
- explicit memory deps, control deps, scalar params, and yielded results;
- token-local memref views produced from SDE MU tokens;
- access rewrites that make the codelet body agree with the MU storage layout;
- scalar capture normalization and reconstruction of local values;
- verification that a codelet uses only local values, deps, params, or values
  derived from them.

CODIR does not own:

- OpenMP semantics or source dependence legality;
- physical owner-dim selection, tiling legality, or reduction strategy;
- ARTS worker routes, node placement, DB GUIDs, depv layout, or runtime API
  calls;
- fallback rediscovery of DB roots from raw memrefs.

## Proposed IR Contract

The initial dialect should stay small. Exact op names can change, but the
contract should not:

- `codir.codelet`: isolated body with dep and param region arguments, plus
  one explicit access-mode entry for each dep.
- `codir.dep`: memory or control dependency derived from SDE tokens.
- `codir.param`: scalar immutable parameter.
- `codir.launch`: logical launch carrier before ARTS EDT creation.
- `codir.yield`: explicit results or completion values.

Every external value used by a `codir.codelet` must be represented in the
creation site. Memrefs, mutable state, and token windows are deps with
read/write/readwrite mode metadata. Scalars and small immutable
firstprivate-style captures are params. Values that can be recomputed from deps
or params should be built inside the codelet body.

## EDT Lowering Benefit

With CODIR in place, `ConvertCodirToArts` can create an `arts.edt` whose deps
and params are complete at construction time. `EdtLowering` then becomes a
mechanical ABI lowering pass:

- dep operands become runtime dependency records;
- params become packed EDT parameters;
- the isolated codelet body becomes the EDT body function;
- any implicit capture is a verifier error, not a lowering feature.

## Migration Checklist

- Move transitional `sde.cu_codelet` tests and utilities into CODIR or split
  them so SDE produces the plan and CODIR owns isolation.
- Replace tensor/codelet paths with memref MU/token/CODIR paths.
- Teach SDE-to-CODIR materialization to handle ND owner slices, strided
  accesses, halo windows, reductions, and task depend clauses.
- Add verifier tests for no implicit captures, dep/param arity, token-local
  access shape, and illegal mutable params.
- Convert CODIR to `arts` directly without `arts.db_control` or raw-memref DB
  rediscovery.
- Remove raw-memref `CreateDbs` compatibility once supported benchmarks all
  reach direct CODIR/ARTS materialization.
