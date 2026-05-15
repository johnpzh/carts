///==========================================================================///
/// File: DbPartitionTypes.h
///
/// Small controller-side DB partitioning types shared across the controller
/// and block-plan resolution. These are normalized snapshots, not canonical
/// graph facts and not final rewrite payloads.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_DB_DBPARTITIONTYPES_H
#define ARTS_DIALECT_CORE_TRANSFORMS_DB_DBPARTITIONTYPES_H

#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbAccessPattern.h"
#include <optional>

namespace mlir {
namespace arts {

/// Pass-local per-acquire partition summary used by DbPartitioning.
///
/// Ownership split:
///   - lowering contracts + typed DB IR attrs own canonical semantics
///   - DbAnalysis supplies derived legality/profitability evidence
///   - DbPartitioning builds this normalized snapshot
///   - DbLayoutPlanUtils consumes SDE/CODIR plan attrs to materialize
///     concrete block geometry
struct AcquirePartitionInfo {
  DbAcquireOp acquire;
  PartitionMode mode = PartitionMode::coarse;
  SmallVector<Value> partitionOffsets;
  SmallVector<Value> partitionSizes;
  SmallVector<unsigned> partitionDims;
  AccessPattern accessPattern = AccessPattern::Unknown;
  bool isValid = false;
  bool hasIndirectAccess = false;
  bool hasDistributionContract = false;
  bool preservesDepMode = false;
  bool needsFullRange = false;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_DB_DBPARTITIONTYPES_H
