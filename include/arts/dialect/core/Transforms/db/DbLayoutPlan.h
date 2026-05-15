///==========================================================================///
/// File: DbLayoutPlan.h
///
/// Physical DB layout data projected from SDE/CODIR plans.
///
/// SDE owns the tensor/linalg analysis and chooses the physical owner
/// dimensions, block shape, and task slice contracts. Core may use this small
/// value type for ARTS object materialization and diagnostics, but block-local
/// access rewriting belongs to SDE/CODIR token-local memref lowering.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTPLAN_H
#define ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTPLAN_H

#include "arts/Dialect.h"
#include "arts/utils/PartitionPredicates.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace arts {

/// Get string name for mode for diagnostics/debugging.
inline StringRef getPartitionModeName(PartitionMode mode) {
  switch (mode) {
  case PartitionMode::coarse:
    return "Coarse";
  case PartitionMode::fine_grained:
    return "FineGrained";
  case PartitionMode::block:
    return "Block";
  case PartitionMode::stencil:
    return "Stencil";
  }
  return "Unknown";
}

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

  unsigned outerRank() const { return outerSizes.size(); }
  unsigned innerRank() const { return innerSizes.size(); }

  Value getBlockSize(unsigned d = 0) const {
    return d < blockSizes.size() ? blockSizes[d] : Value();
  }

  unsigned numPartitionedDims() const {
    return !partitionedDims.empty() ? partitionedDims.size()
                                    : blockSizes.size();
  }

  bool isCoarse() const { return !requiresWorkerBoundsPlanning(mode); }
  bool isElementWise() const { return arts::usesElementLayout(mode); }
  bool usesBlockedLayout() const { return arts::usesBlockLayout(mode); }
  bool isStencil() const { return arts::supportsHaloExtension(mode); }
  bool isBlock() const { return usesBlockedLayout() && !isStencil(); }

  bool isValid() const {
    if (usesBlockedLayout())
      return !blockSizes.empty();
    return true;
  }

  DbPhysicalLayoutPlan() = default;
  explicit DbPhysicalLayoutPlan(PartitionMode m) : mode(m) {}
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTPLAN_H
