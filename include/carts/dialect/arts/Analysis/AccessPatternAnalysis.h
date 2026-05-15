///==========================================================================///
/// File: AccessPatternAnalysis.h
///
/// Shared utilities for analyzing loop-relative memory access bounds.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_ACCESSPATTERNANALYSIS_H
#define ARTS_DIALECT_CORE_ANALYSIS_ACCESSPATTERNANALYSIS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace arts {

/// Captures one memory access index chain and db_ref prefix rank information.
///
/// This structure represents a single memory access operation's index chain,
/// which may include both DbRefOp indices and subsequent memref.load/store
/// indices.
///
/// Example:
///   %ref = arts.db.ref %db[%i, %j] : memref<100x100xf64>
///   %val = memref.load %ref[%k] : memref<?xf64>
///
/// Results in:
///   indexChain = [%i, %j, %k]
///   dbRefPrefix = 2  (first 2 indices are from DbRefOp)
struct AccessIndexInfo {
  llvm::SmallVector<Value, 8> indexChain; ///  Complete chain of index values
  unsigned dbRefPrefix = 0; ///  Number of indices from DbRefOp prefix
};

/// Loop-relative access bounds summary.
///
/// This structure captures the results of analyzing memory access patterns
/// relative to a loop induction variable, primarily for stencil detection
/// and halo region computation.
///
/// Fields:
///   - minOffset/maxOffset: Range of constant offsets found (e.g., -1 to 1)
///   - centerOffset: Normalization shift for uniformly offset stencils
///   - isStencil: True if multiple distinct offsets were found
///   - valid: True if analysis succeeded
///   - hasVariableOffset: True if non-constant offsets were detected
struct AccessBoundsResult {
  int64_t minOffset = 0;    ///  Minimum constant offset found
  int64_t maxOffset = 0;    ///  Maximum constant offset found
  int64_t centerOffset = 0; ///  Normalization center (for shifted stencils)
  bool isStencil = false;   ///  True if min != max (multiple offsets)
  bool valid = false;       ///  True if analysis completed successfully
  bool hasVariableOffset = false; ///  True if non-constant offset detected
};

/// Normalize stencil bounds around zero for uniformly shifted stencils.
void normalizeAccessBounds(AccessBoundsResult &bounds);

/// Analyze access bounds from index chains relative to loopIV + blockBase.
AccessBoundsResult analyzeAccessBoundsFromIndices(
    llvm::ArrayRef<AccessIndexInfo> accesses, Value loopIV, Value blockBase,
    std::optional<unsigned> partitionDim = std::nullopt);

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_ACCESSPATTERNANALYSIS_H
