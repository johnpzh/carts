///==========================================================================///
/// File: PartitionPredicates.h
///
/// Semantic predicates for PartitionMode enum values.
/// Replaces scattered multi-mode comparisons with named intent.
///==========================================================================///

#ifndef ARTS_UTILS_PARTITIONPREDICATES_H
#define ARTS_UTILS_PARTITIONPREDICATES_H

#include "carts/Dialect.h"

namespace mlir {
namespace arts {

/// Returns true for partition modes that use a blocked (tiled) memory layout.
/// Matches block and stencil modes (both partition into contiguous chunks).
inline bool usesBlockLayout(PartitionMode m) {
  return m == PartitionMode::block || m == PartitionMode::stencil;
}

/// Returns true for partition modes that require worker bounds planning.
/// All modes except coarse need partition offsets/sizes.
inline bool requiresWorkerBoundsPlanning(PartitionMode m) {
  return m != PartitionMode::coarse;
}

/// Returns true for partition modes that support halo extension.
/// Only stencil mode adds halo cells around block boundaries.
inline bool supportsHaloExtension(PartitionMode m) {
  return m == PartitionMode::stencil;
}

/// Returns true for partition modes that use element-level addressing.
inline bool usesElementLayout(PartitionMode m) {
  return m == PartitionMode::fine_grained;
}

} // namespace arts
} // namespace mlir

#endif // ARTS_UTILS_PARTITIONPREDICATES_H
