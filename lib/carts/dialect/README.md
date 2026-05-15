# CARTS Dialect Implementation Root

This is the target implementation root for CARTS dialect layers.

Target subdialects:

- `sde`
- `codir`
- `arts`
- `arts-rt`

Each subdialect owns its local `IR/`, `Analysis/`, `Transforms/`,
`Conversion/`, and `Verify/` implementation folders. Optimizations belong in
the dialect whose IR proves their legality.

The current implementation still compiles from `lib/arts/dialect/...`. Move
implementation files here in staged slices following
`docs/compiler/plans/folder-reorganization.md`.
