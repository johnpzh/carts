///==========================================================================///
/// File: CodirDialect.cpp
/// Defines the Codelet IR dialect.
///==========================================================================///

#include "carts/dialect/codir/IR/CodirDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::codir;

void CartsCodirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "carts/dialect/codir/IR/CodirOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "carts/dialect/codir/IR/CodirOpsAttributes.cpp.inc"
      >();
}

#include "carts/dialect/codir/IR/CodirOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "carts/dialect/codir/IR/CodirOpsAttributes.cpp.inc"

#include "carts/dialect/codir/IR/CodirOpsEnums.cpp.inc"
