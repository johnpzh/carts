///==========================================================================///
/// File: AffineAccessUtils.h
///
/// Small SDE-side helpers for recovering structured access coordinates from
/// scalarized affine index expressions.
///==========================================================================///

#ifndef ARTS_DIALECT_SDE_ANALYSIS_AFFINEACCESSUTILS_H
#define ARTS_DIALECT_SDE_ANALYSIS_AFFINEACCESSUTILS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::arts::sde {

struct LinearizedAccess2D {
  Value outer;
  Value inner;
  Value stride;
};

std::optional<LinearizedAccess2D>
decomposeRowMajorLinearizedIndex(Value index, Value requiredStride = Value());

std::optional<SmallVector<Value, 2>>
inferRowMajorFlatShape(Value totalElements, Value stride);

} // namespace mlir::arts::sde

#endif // ARTS_DIALECT_SDE_ANALYSIS_AFFINEACCESSUTILS_H
