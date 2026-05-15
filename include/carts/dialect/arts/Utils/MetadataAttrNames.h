///==========================================================================///
/// File: MetadataAttrNames.h
///
/// Attribute name constants for loop and memref metadata, preserved after
/// the MemrefMetadata/LoopMetadata class removal for consumers that still
/// reference these string constants.
///==========================================================================///

#ifndef ARTS_UTILS_METADATA_METADATAATTRNAMES_H
#define ARTS_UTILS_METADATA_METADATAATTRNAMES_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace arts {
namespace AttrNames {

namespace LoopMetadata {
using namespace llvm;
constexpr StringLiteral Name = "arts.loop";
constexpr StringLiteral LocationKey = "location_key";
} // namespace LoopMetadata

namespace MemrefMetadata {
using namespace llvm;
constexpr StringLiteral AllocationId = "allocation_id";
} // namespace MemrefMetadata

} // namespace AttrNames
} // namespace arts
} // namespace mlir

#endif // ARTS_UTILS_METADATA_METADATAATTRNAMES_H
