///==========================================================================///
/// File: DeadStateCleanup.cpp
///
/// Remove dead dialect-neutral helper IR before SDE planning.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::carts::sde {
#define GEN_PASS_DEF_SDEDEADSTATECLEANUPPASS
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::sde

#include "carts/utils/DeadIrCleanup.h"
#include "carts/utils/Debug.h"

ARTS_DEBUG_SETUP(sde_dead_state_cleanup);

using namespace mlir;
using namespace mlir::carts;

namespace {

struct SdeDeadStateCleanupPass
    : public sde::impl::SdeDeadStateCleanupPassBase<
          SdeDeadStateCleanupPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    unsigned totalRemoved = 0;
    unsigned iterations = 0;

    while (true) {
      ++iterations;
      DeadIrCleanupResult removed =
          runDeadIrCleanup(module, /*removeSymbols=*/true);
      totalRemoved += removed.total();
      if (removed.total() == 0)
        break;
    }

    ARTS_DEBUG("SDE dead-state cleanup removed " << totalRemoved
                                                 << " operations in "
                                                 << iterations
                                                 << " iterations");
  }
};

} // namespace
