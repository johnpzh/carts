///==========================================================================///
/// File: MetadataEnums.h
///
/// Standalone enum definitions for memory access pattern classification.
/// Extracted from MemrefMetadata.h so they remain available after the
/// MemrefMetadata class is removed.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_UTILS_METADATAENUMS_H
#define CARTS_DIALECT_ARTS_UTILS_METADATAENUMS_H

namespace mlir {
namespace carts::arts {

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

/// Per-dimension access pattern classification
enum class DimAccessPatternType {
  Unknown = 0,
  Constant = 1,
  UnitStride = 2,
  Affine = 3,
  NonAffine = 4
};

} // namespace carts::arts
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_UTILS_METADATAENUMS_H
