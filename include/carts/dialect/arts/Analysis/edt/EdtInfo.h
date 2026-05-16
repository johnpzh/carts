///==========================================================================///
/// File: EdtInfo.h
///
/// Shared EDT information objects for analysis.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTINFO_H
#define ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTINFO_H

#include <cstdint>

namespace mlir {
namespace carts::arts {

/// Base info aggregated for an EDT region.
struct EdtInfo {
  /// Structural info
  uint64_t maxLoopDepth = 0;
  /// Program order within function (if known)
  unsigned orderIndex = 0;
};
} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTINFO_H
