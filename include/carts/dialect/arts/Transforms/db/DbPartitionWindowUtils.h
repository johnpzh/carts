///==========================================================================///
/// File: DbPartitionWindowUtils.h
///
/// Shared helpers for turning owner-aligned element windows into
/// dependency-aligned element windows before DB-space block conversion.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_DB_DBPARTITIONWINDOWUTILS_H
#define ARTS_DIALECT_CORE_TRANSFORMS_DB_DBPARTITIONWINDOWUTILS_H

#include "carts/utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace arts {

struct ExpandedElementWindow {
  Value offset;
  Value size;
};

/// Expand an owner-local element window to cover the full dependency span
/// described by [minOffset, maxOffset]. The start is clamped at zero and the
/// end is clamped to totalExtent when it is available.
inline ExpandedElementWindow
expandElementWindowWithHalo(OpBuilder &builder, Location loc, Value elemOffset,
                            Value elemSize, Value totalExtent,
                            int64_t minOffset, int64_t maxOffset) {
  Value zero = arts::createConstantIndex(builder, loc, 0);

  Value start = elemOffset;
  if (minOffset < 0) {
    Value haloLeft = arts::createConstantIndex(builder, loc, -minOffset);
    Value canShift = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::uge, start, haloLeft);
    Value shifted = arith::SubIOp::create(builder, loc, start, haloLeft);
    start = arith::SelectOp::create(builder, loc, canShift, shifted, zero);
  } else if (minOffset > 0) {
    Value shift = arts::createConstantIndex(builder, loc, minOffset);
    start = arith::AddIOp::create(builder, loc, start, shift);
  }

  Value end = arith::AddIOp::create(builder, loc, elemOffset, elemSize);
  if (maxOffset > 0) {
    Value haloRight = arts::createConstantIndex(builder, loc, maxOffset);
    Value endPlusHalo = arith::AddIOp::create(builder, loc, end, haloRight);
    end = totalExtent
              ? arith::MinUIOp::create(builder, loc, endPlusHalo, totalExtent)
              : endPlusHalo;
  } else if (maxOffset < 0) {
    Value shrink = arts::createConstantIndex(builder, loc, -maxOffset);
    Value canShrink = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::uge, end, shrink);
    Value shrunken = arith::SubIOp::create(builder, loc, end, shrink);
    end = arith::SelectOp::create(builder, loc, canShrink, shrunken, zero);
  }

  Value endAfterStart = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::uge, end, start);
  Value rawSize = arith::SubIOp::create(builder, loc, end, start);
  Value size =
      arith::SelectOp::create(builder, loc, endAfterStart, rawSize, zero);
  return ExpandedElementWindow{start, size};
}

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_DB_DBPARTITIONWINDOWUTILS_H
