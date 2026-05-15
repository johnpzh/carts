# dialect/arts/IR/ — ARTS Dialect Implementation

## Layout

```
Dialect.cpp            ArtsDialect::initialize() + verifiers + builders
```

## TableGen Files

The ARTS dialect's TableGen definitions live at `include/carts/dialect/arts/IR/`:

```
include/carts/dialect/arts/IR/
  Dialect.td         Dialect definition
  Ops.td             Op definitions (ARTS ops; ARTS-RT ops live in dialect/arts-rt/IR/)
  Attributes.td      Attribute definitions
  Types.td           Type definitions
```

Generated files appear at the corresponding build path:

```
build/include/carts/dialect/arts/IR/
  Ops.h.inc, Ops.cpp.inc                  Op declarations / definitions
  OpsDialect.h.inc                        Dialect class
  OpsAttributes.h.inc                     Attribute classes
  OpsTypes.h.inc                          Type classes
```

The public header `include/carts/Dialect.h` includes the generated headers
from `build/include/carts/dialect/arts/IR/`.
