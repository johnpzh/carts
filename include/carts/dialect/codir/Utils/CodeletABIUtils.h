#ifndef CARTS_DIALECT_CODIR_UTILS_CODELETABIUTILS_H
#define CARTS_DIALECT_CODIR_UTILS_CODELETABIUTILS_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::carts::codir {

bool isCodirDependencyType(Type type);
bool isCodirScalarParamType(Type type);

} // namespace mlir::carts::codir

#endif // CARTS_DIALECT_CODIR_UTILS_CODELETABIUTILS_H
