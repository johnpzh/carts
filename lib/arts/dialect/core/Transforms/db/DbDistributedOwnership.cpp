///==========================================================================///
/// File: DbDistributedOwnership.cpp
///
/// Marks eligible DbAlloc operations for distributed ownership lowering.
///
/// Example:
///   Before:
///     %db = arts.db_alloc ...      // no ownership marker
///
///   After:
///     %db = arts.db_alloc ... {distributed}
///==========================================================================///

#define GEN_PASS_DEF_DBDISTRIBUTEDOWNERSHIP
#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/db/DbAnalysis.h"
#include "arts/dialect/core/Analysis/db/DbDistributedEligibility.h"
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/OperationAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <cassert>

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_distributed_ownership);

using namespace mlir;
using namespace mlir::arts;

namespace {

struct DbDistributedOwnershipPass
    : public impl::DbDistributedOwnershipBase<DbDistributedOwnershipPass> {
  explicit DbDistributedOwnershipPass(mlir::arts::AnalysisManager *AM)
      : AM(AM) {
    assert(AM && "AnalysisManager must be provided externally");
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *machine = &AM->getRuntimeConfig();
    if (!machine->hasConfigFile() || !machine->hasValidNodeCount() ||
        !machine->hasValidThreads()) {
      module.emitError(
          "invalid ARTS machine configuration for distributed DB ownership");
      signalPassFailure();
      return;
    }

    auto &dbAnalysis = AM->getDbAnalysis();
    dbAnalysis.invalidate();

    unsigned totalAllocs = 0;
    unsigned markedDistributed = 0;
    module.walk([&](DbAllocOp alloc) {
      ++totalAllocs;
      auto eligibility = evaluateDistributedDbEligibility(alloc, dbAnalysis);
      setDistributedDbAllocation(alloc.getOperation(), eligibility.eligible);
      if (eligibility.eligible) {
        ++markedDistributed;
        /// Stamp the distribution kind when the eligibility analysis specifies
        /// one (when a non-default distribution applies).
        if (eligibility.distributionKind)
          setEdtDistributionKind(alloc.getOperation(),
                                 *eligibility.distributionKind);
      } else {
        ARTS_DEBUG("Reject DbAlloc arts.id=" << getArtsId(alloc.getOperation())
                                             << " reason="
                                             << toString(eligibility.reason));
      }
    });

    ARTS_INFO("DbDistributedOwnership marked " << markedDistributed << " / "
                                               << totalAllocs
                                               << " DbAlloc operations");
  }

private:
  mlir::arts::AnalysisManager *AM = nullptr;
};

} // namespace

namespace mlir {
namespace arts {
std::unique_ptr<Pass>
createDbDistributedOwnershipPass(mlir::arts::AnalysisManager *AM) {
  return std::make_unique<DbDistributedOwnershipPass>(AM);
}
} // namespace arts
} // namespace mlir
