# CODIR Conversion Includes

CODIR conversion passes are declared through
`include/carts/dialect/codir/Transforms/Passes.h` because MLIR pass
registration uses one generated CODIR pass surface. Conversion implementation
details stay under `lib/carts/dialect/codir/Conversion/`.

Do not add SDE- or ARTS-dependent helper APIs here unless they are part of an
intentional conversion boundary contract. CODIR IR and CODIR utilities must
remain usable without depending on SDE, ARTS, or ARTS-RT.
