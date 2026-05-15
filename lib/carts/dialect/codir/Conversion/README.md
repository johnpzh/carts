# CODIR Conversion Implementation

This directory owns the explicit dialect-boundary conversions in the live
compiler pipeline:

- `SdeToCodir/`: lowers SDE planning/codelet intent into isolated
  `codir.codelet` operations. This is the only CODIR-owned implementation area
  that may inspect SDE ops such as `sde.mu_dep`, `sde.mu_token`, or
  `sde.su_iterate`.
- `CodirToArts/`: lowers verified CODIR codelets into abstract ARTS DB acquire
  and EDT objects. This is the only CODIR implementation area that may create
  ARTS orchestration ops.

Shared conversion-only helpers live in `ConversionUtils.h`. Helpers that depend
on SDE contracts, such as task-dependency slice proof logic, stay under
`SdeToCodir/` instead of `codir/Utils` so the CODIR dialect library remains
isolated from SDE and ARTS.
