///==========================================================================///
/// File: SdeDialect.cpp
/// Defines the Structured Decomposition Environment dialect.
///==========================================================================///

#include "carts/dialect/sde/IR/SdeDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::carts::sde;

void CartsSdeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "carts/dialect/sde/IR/SdeOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "carts/dialect/sde/IR/SdeOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "carts/dialect/sde/IR/SdeOpsAttributes.cpp.inc"
      >();
}

#include "carts/dialect/sde/IR/SdeOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "carts/dialect/sde/IR/SdeOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "carts/dialect/sde/IR/SdeOpsTypes.cpp.inc"

#include "carts/dialect/sde/IR/SdeOpsEnums.cpp.inc"
