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

#include "carts/dialect/arts-rt/IR/ArtsRtOpsDialect.h.inc"

namespace mlir::carts::arts_rt {} // namespace mlir::carts::arts_rt

#include "carts/dialect/arts-rt/IR/ArtsRtOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "carts/dialect/arts-rt/IR/ArtsRtOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "carts/dialect/arts-rt/IR/ArtsRtOps.h.inc"

#endif // ARTS_RT_DIALECT_H
