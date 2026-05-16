///==========================================================================///
/// File: DbHeuristics.cpp
///
/// DB heuristic policy: decision recording for DB materialization/refinement
/// diagnostics.
///==========================================================================///

#include "carts/dialect/arts/Analysis/heuristics/DbHeuristics.h"
#include "carts/Dialect.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/utils/LocationMetadata.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

void DbHeuristics::recordDecision(llvm::StringRef heuristic, bool applied,
                                  llvm::StringRef rationale, Operation *op,
                                  const llvm::StringMap<int64_t> &inputs) {
  int64_t artsId = op ? getArtsId(op) : 0;
  int64_t allocId = 0;
  SmallVector<int64_t> affectedDbIds;
  std::string sourceLocation;

  if (op) {
    LocationMetadata locMeta = LocationMetadata::fromLocation(op->getLoc());
    if (locMeta.isValid())
      sourceLocation = locMeta.file + ":" + std::to_string(locMeta.line);

    if (auto acquireOp = dyn_cast<DbAcquireOp>(op)) {
      if (Operation *allocOp =
              DbUtils::getUnderlyingDbAlloc(acquireOp.getSourcePtr())) {
        allocId = getArtsId(allocOp);
        ValueRange offsets = acquireOp.getOffsets();
        ValueRange sizes = acquireOp.getSizes();

        if (!offsets.empty() && !sizes.empty()) {
          int64_t offset = 0;
          int64_t count = 0;
          bool offsetKnown =
              ValueAnalysis::getConstantIndex(offsets[0], offset);
          bool sizeKnown = ValueAnalysis::getConstantIndex(sizes[0], count);
          if (offsetKnown && sizeKnown && allocId != 0) {
            if (count > kMaxAffectedDbIds) {
              /// Cap: store only the range start for very large DB counts.
              affectedDbIds.push_back(allocId + offset);
            } else {
              for (int64_t i = 0; i < count; ++i)
                affectedDbIds.push_back(allocId + offset + i);
            }
          } else if (allocId != 0) {
            affectedDbIds.push_back(allocId);
          }
        } else if (allocId != 0) {
          if (auto dbAlloc = dyn_cast<DbAllocOp>(allocOp)) {
            ValueRange allocSizes = dbAlloc.getSizes();
            if (!allocSizes.empty()) {
              int64_t totalDbs = 1;
              bool allStatic = true;
              for (Value sz : allocSizes) {
                int64_t dim = 0;
                if (ValueAnalysis::getConstantIndex(sz, dim))
                  totalDbs *= dim;
                else
                  allStatic = false;
              }
              if (allStatic) {
                if (totalDbs > kMaxAffectedDbIds) {
                  /// Cap: store only the first element for very large DB
                  /// counts.
                  affectedDbIds.push_back(allocId);
                } else {
                  for (int64_t i = 0; i < totalDbs; ++i)
                    affectedDbIds.push_back(allocId + i);
                }
              }
            }
          }
        }
      }
    }
  }

  decisions.push_back({heuristic.str(), applied, rationale.str(), artsId,
                       allocId, std::move(affectedDbIds), sourceLocation,
                       inputs});
}

llvm::ArrayRef<HeuristicDecision> DbHeuristics::getDecisions() const {
  return decisions;
}
