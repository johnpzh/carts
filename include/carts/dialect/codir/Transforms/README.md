# CODIR Transform Includes

CODIR transform pass declarations live in `Passes.h` / `Passes.td` together
with CODIR conversion pass declarations, because the MLIR generated pass
registration surface is shared.

Implementation ownership is still separated:

- CODIR-local verification and optimization live under
  `lib/carts/dialect/codir/Transforms/`.
- SDE-to-CODIR and CODIR-to-ARTS boundary conversions live under
  `lib/carts/dialect/codir/Conversion/`.
