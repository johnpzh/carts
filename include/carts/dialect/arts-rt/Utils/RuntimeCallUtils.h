///==========================================================================///
/// File: RuntimeCallUtils.h
///
/// Runtime ABI call classification helpers owned by ARTS-RT.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_RT_UTILS_RUNTIMECALLUTILS_H
#define CARTS_DIALECT_ARTS_RT_UTILS_RUNTIMECALLUTILS_H

#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Types.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <optional>

namespace mlir {
namespace carts::arts_rt {

inline std::optional<types::RuntimeFunction>
getRuntimeFunction(func::CallOp call) {
  if (!call)
    return std::nullopt;
  return types::runtimeFunctionFromName(call.getCallee());
}

inline std::optional<types::RuntimeFunction>
getRuntimeTopologyQueryFunction(func::CallOp call) {
  std::optional<types::RuntimeFunction> fn = getRuntimeFunction(call);
  if (!fn || !types::isRuntimeTopologyQuery(*fn))
    return std::nullopt;
  return fn;
}

inline bool isRuntimeTopologyQueryCall(func::CallOp call) {
  return getRuntimeTopologyQueryFunction(call).has_value();
}

} // namespace carts::arts_rt
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_RT_UTILS_RUNTIMECALLUTILS_H
