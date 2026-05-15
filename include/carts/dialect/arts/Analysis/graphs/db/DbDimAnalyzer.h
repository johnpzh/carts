///==========================================================================///
/// File: DbDimAnalyzer.h
///
/// Canonical per-entry / per-dimension partition facts for DbAcquireNode.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBDIMANALYZER_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBDIMANALYZER_H

#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"

namespace mlir {
namespace carts::arts {

class DbDimAnalyzer {
public:
  static DbAcquirePartitionFacts compute(DbAcquireNode *node);
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBDIMANALYZER_H
