///==========================================================================///
/// File: SdeDialect.h
/// Defines the Structured Decomposition Environment dialect.
///==========================================================================///

#ifndef ARTS_SDE_DIALECT_H
#define ARTS_SDE_DIALECT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "carts/dialect/sde/IR/SdeOpsDialect.h.inc"

namespace mlir::carts::sde {} // namespace mlir::carts::sde

#define GET_TYPEDEF_CLASSES
#include "carts/dialect/sde/IR/SdeOpsTypes.h.inc"

#include "carts/dialect/sde/IR/SdeOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "carts/dialect/sde/IR/SdeOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "carts/dialect/sde/IR/SdeOps.h.inc"

#endif // ARTS_SDE_DIALECT_H
