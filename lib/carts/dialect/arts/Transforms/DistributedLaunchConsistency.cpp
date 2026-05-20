///==========================================================================///
/// File: DistributedLaunchConsistency.cpp
///
/// Reconciles ARTS EDT placement with distributed DB ownership.
///==========================================================================///

#define GEN_PASS_DEF_DISTRIBUTEDLAUNCHCONSISTENCY
#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/RuntimeOpUtils.h"
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "carts/utils/Debug.h"

ARTS_DEBUG_SETUP(distributed_launch_consistency);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

struct DistributedLaunchConsistencyPass
    : public impl::DistributedLaunchConsistencyBase<
          DistributedLaunchConsistencyPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    unsigned localized = 0;

    module.walk([&](EdtOp edt) {
      if (edt.getConcurrency() != EdtConcurrency::internode)
        return;
      if (!DbUtils::hasLocalOnlyDistributedLaunchDependency(edt))
        return;

      OpBuilder builder(edt);
      Value localRoute = createCurrentNodeRoute(builder, edt.getLoc());
      edt.setConcurrency(EdtConcurrency::intranode);
      edt.getRouteMutable().set(localRoute);
      ++localized;
      ARTS_DEBUG("Localized internode EDT with rejected distributed DB dep: "
                 << edt);
    });

    ARTS_INFO("Distributed launch consistency localized "
              << localized << " EDTs");
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::carts::arts::createDistributedLaunchConsistencyPass() {
  return std::make_unique<DistributedLaunchConsistencyPass>();
}
