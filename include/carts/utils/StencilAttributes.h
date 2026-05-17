///==========================================================================///
/// File: StencilAttributes.h
///
/// Stencil and halo attribute getter/setter functions for CARTS operations.
///==========================================================================///

#ifndef CARTS_UTILS_STENCILATTRIBUTES_H
#define CARTS_UTILS_STENCILATTRIBUTES_H

#include "carts/Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir {
namespace carts {
namespace StencilAttrNames {
namespace Operation {
namespace Stencil {
using namespace llvm;
constexpr StringLiteral StencilCenterOffset("stencil_center_offset");
constexpr StringLiteral ElementStride("element_stride");
constexpr StringLiteral LeftHaloArgIdx("left_halo_arg_idx");
constexpr StringLiteral RightHaloArgIdx("right_halo_arg_idx");
constexpr StringLiteral FootprintMinOffsets("stencil_min_offsets");
constexpr StringLiteral FootprintMaxOffsets("stencil_max_offsets");
constexpr StringLiteral SpatialDims("stencil_spatial_dims");
constexpr StringLiteral OwnerDims("stencil_owner_dims");
constexpr StringLiteral BlockShape("stencil_block_shape");
constexpr StringLiteral WriteFootprint("stencil_write_footprint");
constexpr StringLiteral HaloContractId("stencil_halo_contract_id");
constexpr StringLiteral SupportedBlockHalo("stencil_supported_block_halo");
} // namespace Stencil
} // namespace Operation
} // namespace StencilAttrNames

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

inline std::optional<SmallVector<int64_t, 4>>
readI64ArrayAttr(ArrayAttr attr) {
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

inline std::optional<int64_t> getStencilCenterOffset(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(
          ::mlir::carts::StencilAttrNames::Operation::Stencil::StencilCenterOffset))
    return attr.getInt();
  return std::nullopt;
}

inline void setStencilCenterOffset(Operation *op, int64_t centerOffset) {
  if (!op)
    return;
  op->setAttr(
      ::mlir::carts::StencilAttrNames::Operation::Stencil::StencilCenterOffset,
      IntegerAttr::get(IntegerType::get(op->getContext(), 64), centerOffset));
}

inline std::optional<int64_t> getElementStride(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(
          ::mlir::carts::StencilAttrNames::Operation::Stencil::ElementStride))
    return attr.getInt();
  return std::nullopt;
}

inline void setElementStride(Operation *op, int64_t stride) {
  if (!op)
    return;
  op->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::ElementStride,
              IntegerAttr::get(IndexType::get(op->getContext()), stride));
}

inline std::optional<unsigned> getLeftHaloArgIndex(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(
          ::mlir::carts::StencilAttrNames::Operation::Stencil::LeftHaloArgIdx)) {
    int64_t value = attr.getInt();
    if (value >= 0)
      return static_cast<unsigned>(value);
  }
  return std::nullopt;
}

inline std::optional<unsigned> getRightHaloArgIndex(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(
          ::mlir::carts::StencilAttrNames::Operation::Stencil::RightHaloArgIdx)) {
    int64_t value = attr.getInt();
    if (value >= 0)
      return static_cast<unsigned>(value);
  }
  return std::nullopt;
}

inline void setLeftHaloArgIndex(Operation *op, unsigned argIndex) {
  if (!op)
    return;
  op->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::LeftHaloArgIdx,
              IntegerAttr::get(IndexType::get(op->getContext()), argIndex));
}

inline void setRightHaloArgIndex(Operation *op, unsigned argIndex) {
  if (!op)
    return;
  op->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::RightHaloArgIdx,
              IntegerAttr::get(IndexType::get(op->getContext()), argIndex));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilMinOffsets(Operation *op) {
  return readI64ArrayAttr(op,
                          ::mlir::carts::StencilAttrNames::Operation::Stencil::FootprintMinOffsets);
}

inline void setStencilMinOffsets(Operation *op, ArrayRef<int64_t> offsets) {
  if (!op)
    return;
  op->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::FootprintMinOffsets,
              buildI64ArrayAttr(op, offsets));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilMaxOffsets(Operation *op) {
  return readI64ArrayAttr(op,
                          ::mlir::carts::StencilAttrNames::Operation::Stencil::FootprintMaxOffsets);
}

inline void setStencilMaxOffsets(Operation *op, ArrayRef<int64_t> offsets) {
  if (!op)
    return;
  op->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::FootprintMaxOffsets,
              buildI64ArrayAttr(op, offsets));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilSpatialDims(Operation *op) {
  return readI64ArrayAttr(op, ::mlir::carts::StencilAttrNames::Operation::Stencil::SpatialDims);
}

inline void setStencilSpatialDims(Operation *op, ArrayRef<int64_t> dims) {
  if (!op)
    return;
  op->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::SpatialDims,
              buildI64ArrayAttr(op, dims));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilOwnerDims(Operation *op) {
  return readI64ArrayAttr(op, ::mlir::carts::StencilAttrNames::Operation::Stencil::OwnerDims);
}

inline void setStencilOwnerDims(Operation *op, ArrayRef<int64_t> dims) {
  if (!op)
    return;
  op->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::OwnerDims,
              buildI64ArrayAttr(op, dims));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilBlockShape(Operation *op) {
  return readI64ArrayAttr(op, ::mlir::carts::StencilAttrNames::Operation::Stencil::BlockShape);
}

inline void setStencilBlockShape(Operation *op, ArrayRef<int64_t> blockShape) {
  if (!op)
    return;
  op->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::BlockShape,
              buildI64ArrayAttr(op, blockShape));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilWriteFootprint(Operation *op) {
  return readI64ArrayAttr(op, ::mlir::carts::StencilAttrNames::Operation::Stencil::WriteFootprint);
}

inline void setStencilWriteFootprint(Operation *op,
                                     ArrayRef<int64_t> footprint) {
  if (!op)
    return;
  op->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::WriteFootprint,
              buildI64ArrayAttr(op, footprint));
}

inline std::optional<int64_t> getStencilHaloContractId(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(
          ::mlir::carts::StencilAttrNames::Operation::Stencil::HaloContractId))
    return attr.getInt();
  return std::nullopt;
}

inline void setStencilHaloContractId(Operation *op, int64_t id) {
  if (!op)
    return;
  op->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::HaloContractId,
              IntegerAttr::get(IntegerType::get(op->getContext(), 64), id));
}

inline bool hasSupportedBlockHalo(Operation *op) {
  return op && op->hasAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::SupportedBlockHalo);
}

inline void setSupportedBlockHalo(Operation *op, bool enabled = true) {
  if (!op)
    return;
  if (enabled) {
    op->setAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::SupportedBlockHalo,
                UnitAttr::get(op->getContext()));
    return;
  }
  op->removeAttr(::mlir::carts::StencilAttrNames::Operation::Stencil::SupportedBlockHalo);
}

inline void copyStencilContractAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;

  for (StringRef attrName : {
           ::mlir::carts::StencilAttrNames::Operation::Stencil::StencilCenterOffset,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::ElementStride,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::LeftHaloArgIdx,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::RightHaloArgIdx,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::FootprintMinOffsets,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::FootprintMaxOffsets,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::SpatialDims,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::OwnerDims,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::BlockShape,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::WriteFootprint,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::HaloContractId,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::SupportedBlockHalo,
       }) {
    if (Attribute attr = source->getAttr(attrName))
      dest->setAttr(attrName, attr);
    else
      dest->removeAttr(attrName);
  }
}

inline void inheritStencilContractAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;

  for (StringRef attrName : {
           ::mlir::carts::StencilAttrNames::Operation::Stencil::StencilCenterOffset,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::ElementStride,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::LeftHaloArgIdx,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::RightHaloArgIdx,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::FootprintMinOffsets,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::FootprintMaxOffsets,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::SpatialDims,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::OwnerDims,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::BlockShape,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::WriteFootprint,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::HaloContractId,
           ::mlir::carts::StencilAttrNames::Operation::Stencil::SupportedBlockHalo,
       }) {
    if (dest->hasAttr(attrName))
      continue;
    if (Attribute attr = source->getAttr(attrName))
      dest->setAttr(attrName, attr);
  }
}

} // namespace carts
} // namespace mlir

#endif // CARTS_UTILS_STENCILATTRIBUTES_H
