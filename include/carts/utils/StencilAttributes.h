///==========================================================================///
/// File: StencilAttributes.h
///
/// Stencil and halo attribute getter/setter functions for CARTS operations.
///==========================================================================///

#ifndef CARTS_UTILS_STENCILATTRIBUTES_H
#define CARTS_UTILS_STENCILATTRIBUTES_H

#include "carts/Dialect.h"
#include "carts/utils/ArrayAttrUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

namespace mlir {
namespace carts {

namespace detail {
enum class StencilAttrKind {
  CenterOffset,
  MinOffsets,
  MaxOffsets,
  SpatialDims,
  OwnerDims,
  BlockShape,
  WriteFootprint,
  SupportedBlockHalo,
};

template <typename OpT>
inline StringAttr getGeneratedStencilAttrName(OpT op, StencilAttrKind kind) {
  switch (kind) {
  case StencilAttrKind::CenterOffset:
    return op.getStencilCenterOffsetAttrName();
  case StencilAttrKind::MinOffsets:
    return op.getStencilMinOffsetsAttrName();
  case StencilAttrKind::MaxOffsets:
    return op.getStencilMaxOffsetsAttrName();
  case StencilAttrKind::SpatialDims:
    return op.getStencilSpatialDimsAttrName();
  case StencilAttrKind::OwnerDims:
    return op.getStencilOwnerDimsAttrName();
  case StencilAttrKind::BlockShape:
    return op.getStencilBlockShapeAttrName();
  case StencilAttrKind::WriteFootprint:
    return op.getStencilWriteFootprintAttrName();
  case StencilAttrKind::SupportedBlockHalo:
    return op.getStencilSupportedBlockHaloAttrName();
  }
  llvm_unreachable("unknown stencil attribute kind");
}

inline StringAttr getStencilAttrName(Operation *op, StencilAttrKind kind) {
  if (!op)
    return nullptr;
  if (auto dbAlloc = dyn_cast<arts::DbAllocOp>(op))
    return getGeneratedStencilAttrName(dbAlloc, kind);
  if (auto dbAcquire = dyn_cast<arts::DbAcquireOp>(op))
    return getGeneratedStencilAttrName(dbAcquire, kind);
  if (auto edt = dyn_cast<arts::EdtOp>(op))
    return getGeneratedStencilAttrName(edt, kind);
  if (auto epoch = dyn_cast<arts::EpochOp>(op))
    return getGeneratedStencilAttrName(epoch, kind);
  return nullptr;
}
} // namespace detail

inline std::optional<int64_t> getStencilCenterOffset(Operation *op) {
  if (!op)
    return std::nullopt;
  StringAttr name =
      detail::getStencilAttrName(op, detail::StencilAttrKind::CenterOffset);
  if (!name)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}

inline void setStencilCenterOffset(Operation *op, int64_t centerOffset) {
  if (!op)
    return;
  StringAttr name =
      detail::getStencilAttrName(op, detail::StencilAttrKind::CenterOffset);
  if (!name)
    return;
  op->setAttr(
      name,
      IntegerAttr::get(IntegerType::get(op->getContext(), 64), centerOffset));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilMinOffsets(Operation *op) {
  StringAttr name =
      detail::getStencilAttrName(op, detail::StencilAttrKind::MinOffsets);
  return name ? readI64ArrayAttr(op, name) : std::nullopt;
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilMaxOffsets(Operation *op) {
  StringAttr name =
      detail::getStencilAttrName(op, detail::StencilAttrKind::MaxOffsets);
  return name ? readI64ArrayAttr(op, name) : std::nullopt;
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilSpatialDims(Operation *op) {
  StringAttr name =
      detail::getStencilAttrName(op, detail::StencilAttrKind::SpatialDims);
  return name ? readI64ArrayAttr(op, name) : std::nullopt;
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilOwnerDims(Operation *op) {
  StringAttr name =
      detail::getStencilAttrName(op, detail::StencilAttrKind::OwnerDims);
  return name ? readI64ArrayAttr(op, name) : std::nullopt;
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilBlockShape(Operation *op) {
  StringAttr name =
      detail::getStencilAttrName(op, detail::StencilAttrKind::BlockShape);
  return name ? readI64ArrayAttr(op, name) : std::nullopt;
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilWriteFootprint(Operation *op) {
  StringAttr name =
      detail::getStencilAttrName(op, detail::StencilAttrKind::WriteFootprint);
  return name ? readI64ArrayAttr(op, name) : std::nullopt;
}

inline bool hasSupportedBlockHalo(Operation *op) {
  StringAttr name = detail::getStencilAttrName(
      op, detail::StencilAttrKind::SupportedBlockHalo);
  return name && op->hasAttr(name);
}

inline void copyStencilContractAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;

  for (detail::StencilAttrKind kind : {
           detail::StencilAttrKind::CenterOffset,
           detail::StencilAttrKind::MinOffsets,
           detail::StencilAttrKind::MaxOffsets,
           detail::StencilAttrKind::SpatialDims,
           detail::StencilAttrKind::OwnerDims,
           detail::StencilAttrKind::BlockShape,
           detail::StencilAttrKind::WriteFootprint,
           detail::StencilAttrKind::SupportedBlockHalo,
       }) {
    StringAttr sourceName = detail::getStencilAttrName(source, kind);
    StringAttr destName = detail::getStencilAttrName(dest, kind);
    if (!destName)
      continue;
    if (sourceName) {
      if (Attribute attr = source->getAttr(sourceName)) {
        dest->setAttr(destName, attr);
        continue;
      }
    }
    dest->removeAttr(destName);
  }
}

} // namespace carts
} // namespace mlir

#endif // CARTS_UTILS_STENCILATTRIBUTES_H
