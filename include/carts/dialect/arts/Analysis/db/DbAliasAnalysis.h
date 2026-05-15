///==========================================================================///
/// File: DbAliasAnalysis.h
/// Alias analysis for DB nodes.
///
/// This module provides alias analysis for database (DB) allocations and
/// acquires, determining whether two memory accesses may reference the same
/// or overlapping memory regions.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_DB_DBALIASANALYSIS_H
#define ARTS_DIALECT_CORE_ANALYSIS_DB_DBALIASANALYSIS_H

#include <string>

namespace mlir {
namespace carts::arts {
class NodeBase;
class DbAcquireNode;
class DbAllocNode;
class DbAcquireOp;

class DbAliasAnalysis {
public:
  DbAliasAnalysis() = default;

  enum class AliasResult { NoAlias, MayAlias, MustAlias };

  AliasResult classifyAlias(const NodeBase &a, const NodeBase &b,
                            const std::string &indent = "");

  /// Overlap classification between DB acquire slices
  enum class OverlapKind { Unknown, Disjoint, Partial, Full };
  OverlapKind estimateOverlap(const DbAcquireNode *a, const DbAcquireNode *b);
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_DB_DBALIASANALYSIS_H
