///==========================================================================///
/// File: Types.h
///
/// This file defines runtime ABI function and constant metadata used by
/// ARTS-RT lowering.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_RT_CONVERSION_ARTSRTTOLLVM_TYPES_H
#define CARTS_DIALECT_ARTS_RT_CONVERSION_ARTSRTTOLLVM_TYPES_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>
#include <sys/types.h>

#include "carts/dialect/arts/IR/ArtsDialect.h"

namespace mlir {
namespace carts::arts_rt {

namespace types {

/// IDs for all ARTS runtime library (RTL) functions.
#define ARTS_RTL_FUNCTIONS
enum class RuntimeFunction {
#define ARTS_RTL(Enum, ...) Enum,
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Kinds.def"
};
#undef ARTS_RTL_FUNCTIONS

#define ARTS_RTL(Enum, ...)                                                    \
  constexpr auto Enum = RuntimeFunction::Enum;
#define ARTS_RTL_FUNCTIONS
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Kinds.def"
#undef ARTS_RTL_FUNCTIONS

/// Include ARTS runtime constants
#define ARTS_CONSTANTS
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Kinds.def"
#undef ARTS_CONSTANTS

/// Return the canonical symbol name for an ARTS runtime function.
inline llvm::StringRef runtimeFunctionName(RuntimeFunction fn) {
  switch (fn) {
#define ARTS_RTL_FUNCTIONS
#define ARTS_RTL(Enum, Str, ...)                                               \
  case RuntimeFunction::Enum:                                                  \
    return Str;
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Kinds.def"
#undef ARTS_RTL
#undef ARTS_RTL_FUNCTIONS
  }
  llvm_unreachable("unknown ARTS runtime function");
}

/// Resolve an ARTS runtime function enum from a symbol name, if known.
inline std::optional<RuntimeFunction>
runtimeFunctionFromName(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<RuntimeFunction>>(name)
#define ARTS_RTL_FUNCTIONS
#define ARTS_RTL(Enum, Str, ...) .Case(Str, RuntimeFunction::Enum)
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Kinds.def"
#undef ARTS_RTL
#undef ARTS_RTL_FUNCTIONS
      .Default(std::nullopt);
}

/// Runtime topology/identity queries that are safe to treat as pure reads in
/// the current invocation context.
inline bool isRuntimeTopologyQuery(RuntimeFunction fn) {
  switch (fn) {
  case ARTSRTL_arts_get_total_workers:
  case ARTSRTL_arts_get_total_nodes:
  case ARTSRTL_arts_get_current_worker:
  case ARTSRTL_arts_get_current_node:
    return true;
  default:
    return false;
  }
}

} // end namespace types
} // end namespace arts_rt
} // end namespace mlir

/// Specialization of DenseMapInfo for RuntimeFunction enum.
namespace llvm {
template <> struct DenseMapInfo<mlir::carts::arts_rt::types::RuntimeFunction> {
  static mlir::carts::arts_rt::types::RuntimeFunction getEmptyKey() {
    return static_cast<mlir::carts::arts_rt::types::RuntimeFunction>(~0U);
  }

  static mlir::carts::arts_rt::types::RuntimeFunction getTombstoneKey() {
    return static_cast<mlir::carts::arts_rt::types::RuntimeFunction>(~0U - 1);
  }

  static unsigned
  getHashValue(mlir::carts::arts_rt::types::RuntimeFunction val) {
    return static_cast<unsigned>(val);
  }

  static bool isEqual(mlir::carts::arts_rt::types::RuntimeFunction lhs,
                      mlir::carts::arts_rt::types::RuntimeFunction rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

#endif // CARTS_DIALECT_ARTS_RT_CONVERSION_ARTSRTTOLLVM_TYPES_H
