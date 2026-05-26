#ifndef CARTS_DIALECT_CODIR_UTILS_CODELETABIUTILS_H
#define CARTS_DIALECT_CODIR_UTILS_CODELETABIUTILS_H

#include "carts/dialect/codir/IR/CodirDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include <optional>

namespace mlir::carts::codir {

bool isCodirDependencyType(Type type);
bool isCodirScalarParamType(Type type);

/// True for regionless ops that produce at least one memref result and are
/// therefore candidates for memref-forwarding analysis (cast, subview,
/// reinterpret_cast, polygeist::SubIndexOp, etc.).
bool isMemrefForwardingOp(Operation *op);

/// Return the declared access mode for the given dependency index of a
/// codelet, or nullopt when the codelet has no dep-modes attribute or the
/// index entry is not a CodirAccessModeAttr.
std::optional<CodirAccessMode> getDepAccessMode(CodeletOp codelet,
                                                unsigned depIndex);

} // namespace mlir::carts::codir

#endif // CARTS_DIALECT_CODIR_UTILS_CODELETABIUTILS_H
