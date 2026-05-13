///==========================================================================///
/// File: StencilAttributes.h
///
/// Stencil and halo attribute getter/setter functions for ARTS operations.
///==========================================================================///

#ifndef CARTS_UTILS_STENCILATTRIBUTES_H
#define CARTS_UTILS_STENCILATTRIBUTES_H

#include "arts/Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir {
namespace arts {
namespace AttrNames {
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
} // namespace AttrNames

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
          AttrNames::Operation::Stencil::StencilCenterOffset))
    return attr.getInt();
  return std::nullopt;
}

inline void setStencilCenterOffset(Operation *op, int64_t centerOffset) {
  if (!op)
    return;
  op->setAttr(
      AttrNames::Operation::Stencil::StencilCenterOffset,
      IntegerAttr::get(IntegerType::get(op->getContext(), 64), centerOffset));
}

inline std::optional<int64_t> getElementStride(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(
          AttrNames::Operation::Stencil::ElementStride))
    return attr.getInt();
  return std::nullopt;
}

inline void setElementStride(Operation *op, int64_t stride) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::Stencil::ElementStride,
              IntegerAttr::get(IndexType::get(op->getContext()), stride));
}

inline std::optional<unsigned> getLeftHaloArgIndex(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(
          AttrNames::Operation::Stencil::LeftHaloArgIdx)) {
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
          AttrNames::Operation::Stencil::RightHaloArgIdx)) {
    int64_t value = attr.getInt();
    if (value >= 0)
      return static_cast<unsigned>(value);
  }
  return std::nullopt;
}

inline void setLeftHaloArgIndex(Operation *op, unsigned argIndex) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::Stencil::LeftHaloArgIdx,
              IntegerAttr::get(IndexType::get(op->getContext()), argIndex));
}

inline void setRightHaloArgIndex(Operation *op, unsigned argIndex) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::Stencil::RightHaloArgIdx,
              IntegerAttr::get(IndexType::get(op->getContext()), argIndex));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilMinOffsets(Operation *op) {
  return readI64ArrayAttr(op,
                          AttrNames::Operation::Stencil::FootprintMinOffsets);
}

inline void setStencilMinOffsets(Operation *op, ArrayRef<int64_t> offsets) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::Stencil::FootprintMinOffsets,
              buildI64ArrayAttr(op, offsets));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilMaxOffsets(Operation *op) {
  return readI64ArrayAttr(op,
                          AttrNames::Operation::Stencil::FootprintMaxOffsets);
}

inline void setStencilMaxOffsets(Operation *op, ArrayRef<int64_t> offsets) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::Stencil::FootprintMaxOffsets,
              buildI64ArrayAttr(op, offsets));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilSpatialDims(Operation *op) {
  return readI64ArrayAttr(op, AttrNames::Operation::Stencil::SpatialDims);
}

inline void setStencilSpatialDims(Operation *op, ArrayRef<int64_t> dims) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::Stencil::SpatialDims,
              buildI64ArrayAttr(op, dims));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilOwnerDims(Operation *op) {
  return readI64ArrayAttr(op, AttrNames::Operation::Stencil::OwnerDims);
}

inline void setStencilOwnerDims(Operation *op, ArrayRef<int64_t> dims) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::Stencil::OwnerDims,
              buildI64ArrayAttr(op, dims));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilBlockShape(Operation *op) {
  return readI64ArrayAttr(op, AttrNames::Operation::Stencil::BlockShape);
}

inline void setStencilBlockShape(Operation *op, ArrayRef<int64_t> blockShape) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::Stencil::BlockShape,
              buildI64ArrayAttr(op, blockShape));
}

inline std::optional<SmallVector<int64_t, 4>>
getStencilWriteFootprint(Operation *op) {
  return readI64ArrayAttr(op, AttrNames::Operation::Stencil::WriteFootprint);
}

inline void setStencilWriteFootprint(Operation *op,
                                     ArrayRef<int64_t> footprint) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::Stencil::WriteFootprint,
              buildI64ArrayAttr(op, footprint));
}

inline std::optional<int64_t> getStencilHaloContractId(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(
          AttrNames::Operation::Stencil::HaloContractId))
    return attr.getInt();
  return std::nullopt;
}

inline void setStencilHaloContractId(Operation *op, int64_t id) {
  if (!op)
    return;
  op->setAttr(AttrNames::Operation::Stencil::HaloContractId,
              IntegerAttr::get(IntegerType::get(op->getContext(), 64), id));
}

inline bool hasSupportedBlockHalo(Operation *op) {
  return op && op->hasAttr(AttrNames::Operation::Stencil::SupportedBlockHalo);
}

inline void setSupportedBlockHalo(Operation *op, bool enabled = true) {
  if (!op)
    return;
  if (enabled) {
    op->setAttr(AttrNames::Operation::Stencil::SupportedBlockHalo,
                UnitAttr::get(op->getContext()));
    return;
  }
  op->removeAttr(AttrNames::Operation::Stencil::SupportedBlockHalo);
}

inline void copyStencilContractAttrs(Operation *source, Operation *dest) {
  if (!source || !dest)
    return;

  for (StringRef attrName : {
           AttrNames::Operation::Stencil::StencilCenterOffset,
           AttrNames::Operation::Stencil::ElementStride,
           AttrNames::Operation::Stencil::LeftHaloArgIdx,
           AttrNames::Operation::Stencil::RightHaloArgIdx,
           AttrNames::Operation::Stencil::FootprintMinOffsets,
           AttrNames::Operation::Stencil::FootprintMaxOffsets,
           AttrNames::Operation::Stencil::SpatialDims,
           AttrNames::Operation::Stencil::OwnerDims,
           AttrNames::Operation::Stencil::BlockShape,
           AttrNames::Operation::Stencil::WriteFootprint,
           AttrNames::Operation::Stencil::HaloContractId,
           AttrNames::Operation::Stencil::SupportedBlockHalo,
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
           AttrNames::Operation::Stencil::StencilCenterOffset,
           AttrNames::Operation::Stencil::ElementStride,
           AttrNames::Operation::Stencil::LeftHaloArgIdx,
           AttrNames::Operation::Stencil::RightHaloArgIdx,
           AttrNames::Operation::Stencil::FootprintMinOffsets,
           AttrNames::Operation::Stencil::FootprintMaxOffsets,
           AttrNames::Operation::Stencil::SpatialDims,
           AttrNames::Operation::Stencil::OwnerDims,
           AttrNames::Operation::Stencil::BlockShape,
           AttrNames::Operation::Stencil::WriteFootprint,
           AttrNames::Operation::Stencil::HaloContractId,
           AttrNames::Operation::Stencil::SupportedBlockHalo,
       }) {
    if (dest->hasAttr(attrName))
      continue;
    if (Attribute attr = source->getAttr(attrName))
      dest->setAttr(attrName, attr);
  }
}

} // namespace arts
} // namespace mlir

#endif // CARTS_UTILS_STENCILATTRIBUTES_H
