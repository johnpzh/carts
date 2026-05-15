# CODIR Dialect

CODIR is the codelet dialect. It receives the SDE MU/CU/SU plan and creates
isolated codelets with explicit deps, params, token-local memref views, yielded
values, and verifier-enforced capture rules.

CODIR must not own OpenMP semantics, owner-dim selection, ARTS worker routes,
DB allocation policy, or runtime ABI calls.

Primary docs:

- [`analysis.md`](./analysis.md)
- [`optimizations.md`](./optimizations.md)
- Existing migration note: [`../../codir/README.md`](../../codir/README.md)
