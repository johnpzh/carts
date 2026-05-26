///==========================================================================///
/// File: ArrayAttrUtils.h
///
/// Generic array-attribute helpers shared across dialect layers.
///==========================================================================///

#ifndef CARTS_UTILS_ARRAYATTRUTILS_H
#define CARTS_UTILS_ARRAYATTRUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir {
namespace carts {

inline ArrayAttr buildI64ArrayAttr(MLIRContext *ctx, ArrayRef<int64_t> values) {
  SmallVector<Attribute, 8> attrs;
  attrs.reserve(values.size());
  auto i64Type = IntegerType::get(ctx, 64);
  for (int64_t value : values)
    attrs.push_back(IntegerAttr::get(i64Type, value));
  return ArrayAttr::get(ctx, attrs);
}

inline ArrayAttr buildI64ArrayAttr(Operation *op, ArrayRef<int64_t> values) {
  return buildI64ArrayAttr(op->getContext(), values);
}

inline std::optional<SmallVector<int64_t, 4>> readI64ArrayAttr(ArrayAttr attr) {
  if (!attr)
    return std::nullopt;

  SmallVector<int64_t, 4> values;
  values.reserve(attr.size());
  for (Attribute element : attr) {
    auto intAttr = dyn_cast<IntegerAttr>(element);
    if (!intAttr)
      return std::nullopt;
    values.push_back(intAttr.getInt());
  }
  return values;
}

inline std::optional<SmallVector<int64_t, 4>> readI64ArrayAttr(Operation *op,
                                                               StringRef name) {
  if (!op)
    return std::nullopt;
  return readI64ArrayAttr(op->getAttrOfType<ArrayAttr>(name));
}

inline std::optional<SmallVector<int64_t, 4>>
readI64ArrayAttr(Operation *op, StringAttr name) {
  if (!op || !name)
    return std::nullopt;
  return readI64ArrayAttr(op->getAttrOfType<ArrayAttr>(name));
}

} // namespace carts
} // namespace mlir

#endif // CARTS_UTILS_ARRAYATTRUTILS_H
