#include "carts/dialect/codir/Utils/CodeletABIUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::carts::codir {

bool isCodirDependencyType(Type type) {
  return isa<MemRefType, UnrankedMemRefType>(type);
}

bool isCodirScalarParamType(Type type) {
  return type.isIntOrIndexOrFloat();
}

} // namespace mlir::carts::codir
