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
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/db/DbDistributedEligibility.h"
#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "carts/utils/OperationAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <cassert>

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_distributed_ownership);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

struct DbDistributedOwnershipPass
    : public impl::DbDistributedOwnershipBase<DbDistributedOwnershipPass> {
  explicit DbDistributedOwnershipPass(mlir::carts::arts::AnalysisManager *AM)
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
        alloc->removeAttr(AttrNames::Operation::DistributedRejectReason);
        alloc.removeLocalOnlyAttr();
        /// Stamp the distribution kind when the eligibility analysis specifies
        /// one (when a non-default distribution applies).
        if (eligibility.distributionKind)
          setEdtDistributionKind(alloc.getOperation(),
                                 *eligibility.distributionKind);
      } else {
        alloc->setAttr(
            AttrNames::Operation::DistributedRejectReason,
            StringAttr::get(alloc.getContext(), toString(eligibility.reason)));
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
  mlir::carts::arts::AnalysisManager *AM = nullptr;
};

} // namespace

namespace mlir {
namespace carts::arts {
std::unique_ptr<Pass>
createDbDistributedOwnershipPass(mlir::carts::arts::AnalysisManager *AM) {
  return std::make_unique<DbDistributedOwnershipPass>(AM);
}
} // namespace carts::arts
} // namespace mlir
