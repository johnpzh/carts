///==========================================================================///
/// File: SdeDialect.cpp
/// Defines the Structured Decomposition Environment dialect.
///==========================================================================///

#include "arts/dialect/sde/IR/SdeDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::arts::sde;

void ArtsSdeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "arts/dialect/sde/IR/SdeOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "arts/dialect/sde/IR/SdeOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "arts/dialect/sde/IR/SdeOpsAttributes.cpp.inc"
      >();
}

#include "arts/dialect/sde/IR/SdeOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "arts/dialect/sde/IR/SdeOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "arts/dialect/sde/IR/SdeOpsTypes.cpp.inc"

#include "arts/dialect/sde/IR/SdeOpsEnums.cpp.inc"
