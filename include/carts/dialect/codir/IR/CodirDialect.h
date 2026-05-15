///==========================================================================///
/// File: CodirDialect.h
/// Defines the Codelet IR dialect.
///==========================================================================///

#ifndef CARTS_CODIR_DIALECT_H
#define CARTS_CODIR_DIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "carts/dialect/codir/IR/CodirOpsDialect.h.inc"

namespace mlir::carts::codir {} // namespace mlir::carts::codir

#include "carts/dialect/codir/IR/CodirOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "carts/dialect/codir/IR/CodirOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "carts/dialect/codir/IR/CodirOps.h.inc"

#endif // CARTS_CODIR_DIALECT_H
