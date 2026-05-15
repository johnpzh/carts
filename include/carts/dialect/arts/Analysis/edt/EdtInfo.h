///==========================================================================///
/// File: EdtInfo.h
///
/// Shared EDT information objects for analysis.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTINFO_H
#define ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTINFO_H

#include "carts/Dialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>

namespace mlir {
namespace carts::arts {

/// Base info aggregated for an EDT region.
struct EdtInfo {
  /// Structural info
  uint64_t maxLoopDepth = 0;
  /// Program order within function (if known)
  unsigned orderIndex = 0;

  /// Distribution-pattern analysis results owned at EDT analysis level.
  /// Maps top-level loops within this EDT to classified compute patterns.
  llvm::DenseMap<Operation *, EdtDistributionPattern> loopDistributionPatterns;
  /// EDT-level summary pattern (unknown if mixed or not classified).
  EdtDistributionPattern dominantDistributionPattern =
      EdtDistributionPattern::unknown;
};
} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTINFO_H
