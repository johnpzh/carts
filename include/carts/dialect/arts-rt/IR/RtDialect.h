///==========================================================================///
/// File: RtDialect.h
/// Defines the ARTS runtime dialect and its operations.
///==========================================================================///

#ifndef ARTS_RT_DIALECT_H
#define ARTS_RT_DIALECT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "carts/dialect/arts-rt/IR/RtOpsDialect.h.inc"

namespace mlir::arts::rt {} // namespace mlir::arts::rt

#include "carts/dialect/arts-rt/IR/RtOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "carts/dialect/arts-rt/IR/RtOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "carts/dialect/arts-rt/IR/RtOps.h.inc"

#endif // ARTS_RT_DIALECT_H
