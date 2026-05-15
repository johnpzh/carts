///==========================================================================///
/// File: DbHeuristics.cpp
///
/// DB heuristic policy: decision recording, JSON export, and delegation to
/// partitioning heuristics.
///==========================================================================///

#include "carts/dialect/arts/Analysis/heuristics/DbHeuristics.h"
#include "carts/Dialect.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/utils/Debug.h"
#include "carts/dialect/arts/Utils/LocationMetadata.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"

ARTS_DEBUG_SETUP(db_heuristics)

using namespace mlir;
using namespace mlir::arts;

namespace {

static std::string extractHeuristicId(llvm::StringRef rationale) {
  if (rationale.starts_with("H1.")) {
    size_t colonPos = rationale.find(':');
    if (colonPos != std::string::npos)
      return rationale.substr(0, colonPos).str();
  }
  return "Fallback";
}

} // namespace

DbHeuristics::DbHeuristics(const RuntimeConfig &machine) : machine(machine) {}

bool DbHeuristics::isSingleNode() const { return machine.isSingleNode(); }

bool DbHeuristics::isValid() const { return machine.isValid(); }

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
            if (count > getMaxOuterDBs()) {
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
                if (totalDbs > getMaxOuterDBs()) {
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

void DbHeuristics::exportDecisionsToJson(llvm::json::OStream &J) const {
  J.attributeArray("heuristic_decisions", [&]() {
    for (const auto &d : decisions) {
      J.object([&]() {
        J.attribute("heuristic", d.heuristic);
        J.attribute("applied", d.applied);
        J.attribute("rationale", d.rationale);
        J.attribute("target_id", d.affectedArtsId);
        if (d.affectedAllocId != 0)
          J.attribute("alloc_id", d.affectedAllocId);
        if (!d.affectedDbIds.empty()) {
          J.attributeArray("affected_db_ids", [&]() {
            for (int64_t id : d.affectedDbIds)
              J.value(id);
          });
        }
        if (!d.sourceLocation.empty())
          J.attribute("source_location", d.sourceLocation);
        if (!d.costModelInputs.empty()) {
          J.attributeObject("cost_model_inputs", [&]() {
            for (const auto &kv : d.costModelInputs)
              J.attribute(kv.first(), kv.second);
          });
        }
      });
    }
  });
}
