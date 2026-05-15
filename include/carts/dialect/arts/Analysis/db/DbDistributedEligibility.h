///==========================================================================///
/// File: DbDistributedEligibility.h
///
/// Eligibility analysis for distributed DB ownership marking.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_DB_DBDISTRIBUTEDELIGIBILITY_H
#define ARTS_DIALECT_CORE_ANALYSIS_DB_DBDISTRIBUTEDELIGIBILITY_H

#include "carts/Dialect.h"
#include <optional>

namespace mlir {
namespace arts {

class DbAnalysis;

enum class DistributedDbEligibilityRejectReason {
  None,
  NestedInEdt,
  GlobalAllocType,
  SingleBlock,
  UnsupportedShape,
  StencilReadInternodeUse,
  UnsupportedPtrUsers,
  UnsupportedGuidUsers,
  NonEdtAcquireUse,
  NoInternodeEdtUse,
};

struct DistributedDbEligibilityResult {
  bool eligible = false;
  DistributedDbEligibilityRejectReason reason =
      DistributedDbEligibilityRejectReason::None;
  /// When set, the pass should stamp this distribution kind on the alloc op.
  std::optional<EdtDistributionKind> distributionKind = std::nullopt;
};

const char *toString(DistributedDbEligibilityRejectReason reason);

DistributedDbEligibilityResult
evaluateDistributedDbEligibility(DbAllocOp alloc, DbAnalysis &dbAnalysis);

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_DB_DBDISTRIBUTEDELIGIBILITY_H
