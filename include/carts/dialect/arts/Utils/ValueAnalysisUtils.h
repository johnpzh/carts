///==========================================================================///
/// File: ValueAnalysisUtils.h
///
/// ARTS-owned value analysis extensions.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_UTILS_VALUEANALYSISUTILS_H
#define CARTS_DIALECT_ARTS_UTILS_VALUEANALYSISUTILS_H

#include "carts/Dialect.h"
#include "mlir/IR/Value.h"
#include <optional>

namespace mlir {
namespace carts::arts {

/// Fold constants through generic index expressions plus ARTS runtime queries.
std::optional<int64_t> tryFoldConstantIndex(Value value, unsigned depth = 0);

/// Strip numeric casts, then fold constants through generic index expressions
/// plus ARTS runtime queries.
std::optional<int64_t> getConstantIndexStripped(Value value);

/// Trace through generic pointer/view wrappers plus ARTS DB and EDT boundaries.
Value getUnderlyingValue(Value value);

/// Like getUnderlyingValue but returns the defining operation.
Operation *getUnderlyingOperation(Value value);

/// Return true when value is derived from source through generic wrappers plus
/// ARTS DB/EDT boundaries.
bool isDerivedFromPtr(Value value, Value source);

} // namespace carts::arts
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_UTILS_VALUEANALYSISUTILS_H
