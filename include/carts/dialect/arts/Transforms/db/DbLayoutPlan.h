///==========================================================================///
/// File: DbLayoutPlan.h
///
/// Physical DB layout data projected from SDE/CODIR plans.
///
/// SDE owns the tensor/linalg analysis and chooses the physical owner
/// dimensions, block shape, and task slice contracts. ARTS may use this small
/// value type for object materialization and diagnostics, but block-local
/// access rewriting belongs to SDE/CODIR token-local memref lowering.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_TRANSFORMS_DB_DBLAYOUTPLAN_H
#define CARTS_DIALECT_ARTS_TRANSFORMS_DB_DBLAYOUTPLAN_H

#include "carts/Dialect.h"
#include "carts/dialect/arts/Utils/PartitionPredicates.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace carts::arts {

/// Concrete physical layout derived from SDE plan attributes.
struct DbPhysicalLayoutPlan {
  PartitionMode mode = PartitionMode::fine_grained;

  /// N-D block sizes, one per partitioned owner dimension.
  SmallVector<Value> blockSizes;
  /// Original memref/tensor dimension index for each partitioned owner dim.
  SmallVector<unsigned> partitionedDims;

  /// Allocation shape.  `outerSizes` are DB coordinates; `innerSizes` are the
  /// element memref shape stored inside each DB entry.
  SmallVector<Value> outerSizes;
  SmallVector<Value> innerSizes;

  bool isValid() const {
    if (arts::usesBlockLayout(mode))
      return !blockSizes.empty();
    return true;
  }

  explicit DbPhysicalLayoutPlan(PartitionMode m) : mode(m) {}
};

} // namespace carts::arts
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_TRANSFORMS_DB_DBLAYOUTPLAN_H
