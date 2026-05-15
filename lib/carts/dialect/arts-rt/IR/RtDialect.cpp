///==========================================================================///
/// File: RtDialect.cpp
/// Defines the ARTS runtime dialect.
///==========================================================================///

#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts_rt;

void ArtsRtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "carts/dialect/arts-rt/IR/RtOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "carts/dialect/arts-rt/IR/RtOpsAttributes.cpp.inc"
      >();
}

#include "carts/dialect/arts-rt/IR/RtOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "carts/dialect/arts-rt/IR/RtOpsAttributes.cpp.inc"

#include "carts/dialect/arts-rt/IR/RtOpsEnums.cpp.inc"
