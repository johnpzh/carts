///==========================================================================///
/// File: VerifyEpochLowered.cpp
///
/// Verification pass that ensures no arts.epoch operations survive past
/// epoch lowering.
///==========================================================================///

#include "carts/Dialect.h"
#define GEN_PASS_DEF_VERIFYEPOCHLOWERED
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct VerifyEpochLoweredPass
    : public impl::VerifyEpochLoweredBase<VerifyEpochLoweredPass> {
  void runOnOperation() override {
    bool found = false;
    getOperation().walk([&](arts::EpochOp op) {
      op.emitError("arts.epoch survived past epoch lowering");
      found = true;
    });
    if (found)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::arts::createVerifyEpochLoweredPass() {
  return std::make_unique<VerifyEpochLoweredPass>();
}
