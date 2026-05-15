///==========================================================================///
/// File: CodirOps.cpp
/// Defines Codelet IR operations.
///==========================================================================///

#include "carts/dialect/codir/IR/CodirDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::carts::codir;

#define GET_OP_CLASSES
#include "carts/dialect/codir/IR/CodirOps.cpp.inc"
