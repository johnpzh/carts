# dialect/core/IR/ — Core ARTS Dialect Implementation

## Layout

```
Dialect.cpp            ArtsDialect::initialize() + verifiers + builders
```

## TableGen Files

The core dialect's TableGen definitions live at `include/arts/dialect/core/IR/`:

```
include/arts/dialect/core/IR/
  Dialect.td         TableGen dialect definition
  Ops.td             Op definitions (core ops only; 14 rt ops in arts_rt)
  Attributes.td      Attribute definitions
  Types.td           Type definitions
```

Generated files appear at the corresponding build path:

```
build/include/arts/dialect/core/IR/
  Ops.h.inc          Generated op declarations
  Ops.cpp.inc        Generated op definitions
  OpsDialect.h.inc   Generated dialect class
  OpsAttributes.h.inc  Generated attribute classes
  OpsTypes.h.inc     Generated type classes
```

Forwarding headers at `include/arts/` (Dialect.td, Ops.td, Attributes.td,
Types.td) redirect to the new paths for backward compatibility with the
135+ consumer files that include via the old `arts/Ops.h.inc` paths.

The public header `include/arts/Dialect.h` includes from the new generated
paths in `build/include/arts/dialect/core/IR/`.
