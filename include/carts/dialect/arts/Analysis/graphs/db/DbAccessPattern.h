///==========================================================================///
/// File: DbAccessPattern.h
///
/// This file defines common types for access pattern classification and
/// stencil bounds analysis used across DbAllocNode, DbAcquireNode, and
/// partitioning passes.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBACCESSPATTERN_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBACCESSPATTERN_H

#include <cstdint>

namespace mlir {
namespace arts {

/// Per-acquire data access pattern classification.
///
/// This enum characterizes how memory accesses within an EDT acquire relate
/// to the partition/iteration space:
///
///   - Unknown: Pattern could not be determined (conservative default)
///   - Uniform: Predictable, aligned access (e.g., A[i] where i is loop IV)
///   - Stencil: Multiple offset accesses (e.g., A[i-1], A[i], A[i+1])
///   - Indexed: Indirect/irregular access (e.g., A[idx[i]])
///
/// These patterns drive partitioning decisions:
///   - Uniform: Standard block partitioning
///   - Stencil: Block partitioning with halo regions
///   - Indexed: May require full-range access or versioning
enum class AccessPattern { Unknown, Uniform, Stencil, Indexed };

/// Stencil bounds information for halo-aware localization.
///
/// Captures the min/max offsets of a stencil pattern to determine halo
/// region sizes. For example, a 1D 3-point stencil accessing A[i-1], A[i],
/// A[i+1] would have minOffset=-1, maxOffset=1.
struct StencilBounds {
  int64_t minOffset = 0, maxOffset = 0;
  bool isStencil = false, valid = false;

  int64_t haloLeft() const { return minOffset < 0 ? -minOffset : 0; }
  int64_t haloRight() const { return maxOffset > 0 ? maxOffset : 0; }
  bool hasHalo() const { return haloLeft() > 0 || haloRight() > 0; }

  /// Factory method to create StencilBounds from analyzed access bounds.
  static StencilBounds create(int64_t min, int64_t max, bool stencil,
                              bool isValid) {
    return {min, max, stencil, isValid};
  }
};

/// Summary of per-acquire access patterns at allocation level.
struct AcquirePatternSummary {
  bool hasUniform = false, hasStencil = false, hasIndexed = false;

  /// Returns true if multiple different patterns are present
  bool isMixed() const {
    int count = static_cast<int>(hasUniform) + static_cast<int>(hasStencil) +
                static_cast<int>(hasIndexed);
    return count > 1;
  }

  /// Get the dominant pattern (if not mixed)
  AccessPattern getDominant() const {
    if (isMixed())
      return AccessPattern::Unknown;
    if (hasStencil)
      return AccessPattern::Stencil;
    if (hasIndexed)
      return AccessPattern::Indexed;
    if (hasUniform)
      return AccessPattern::Uniform;
    return AccessPattern::Unknown;
  }
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBACCESSPATTERN_H
