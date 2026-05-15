///==========================================================================///
/// File: MetadataEnums.h
///
/// Standalone enum definitions for memory access pattern classification.
/// Extracted from MemrefMetadata.h so they remain available after the
/// MemrefMetadata class is removed.
///==========================================================================///

#ifndef ARTS_UTILS_METADATA_METADATAENUMS_H
#define ARTS_UTILS_METADATA_METADATAENUMS_H

namespace mlir {
namespace arts {

///===----------------------------------------------------------------------===///
/// Memory Analysis Enums
///===----------------------------------------------------------------------===///

/// Memory access pattern classification
enum class AccessPatternType {
  Sequential,
  Strided,
  Stencil,
  GatherScatter,
  Random,
  Unknown
};

/// Temporal locality classification based on reuse distance
enum class TemporalLocalityLevel { Excellent, Good, Fair, Poor, Unknown };

/// Spatial locality classification
enum class SpatialLocalityLevel { Excellent, Good, Fair, Poor, Unknown };

/// Per-dimension access pattern classification
enum class DimAccessPatternType {
  Unknown = 0,
  Constant = 1,
  UnitStride = 2,
  Affine = 3,
  NonAffine = 4
};

} // namespace arts
} // namespace mlir

#endif // ARTS_UTILS_METADATA_METADATAENUMS_H
