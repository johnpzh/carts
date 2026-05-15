///==========================================================================///
/// File: VerifyDbLowered.cpp
///
/// Verification pass that ensures no arts.db operations survive past DB
/// lowering.
///==========================================================================///

#include "carts/Dialect.h"
#define GEN_PASS_DEF_VERIFYDBLOWERED
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct VerifyDbLoweredPass
    : public impl::VerifyDbLoweredBase<VerifyDbLoweredPass> {
  void runOnOperation() override {
    bool found = false;
    getOperation().walk([&](arts::DbAllocOp op) {
      op.emitError("arts.db_alloc survived past DB lowering");
      found = true;
    });
    getOperation().walk([&](arts::DbAcquireOp op) {
      op.emitError("arts.db_acquire survived past DB lowering");
      found = true;
    });
    getOperation().walk([&](arts::DbReleaseOp op) {
      op.emitError("arts.db_release survived past DB lowering");
      found = true;
    });
    if (found)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::arts::createVerifyDbLoweredPass() {
  return std::make_unique<VerifyDbLoweredPass>();
}
