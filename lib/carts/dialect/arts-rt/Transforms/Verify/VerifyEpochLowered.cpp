///==========================================================================///
/// File: VerifyEpochLowered.cpp
///
/// Verification pass that ensures no arts.epoch operations survive past
/// epoch lowering.
///==========================================================================///

#include "carts/dialect/arts-rt/Transforms/Passes.h"
namespace mlir::carts::arts_rt {
#define GEN_PASS_DEF_VERIFYEPOCHLOWERED
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts_rt
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::carts;

namespace {
struct VerifyEpochLoweredPass
    : public arts_rt::impl::VerifyEpochLoweredBase<VerifyEpochLoweredPass> {
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

std::unique_ptr<Pass> mlir::carts::arts_rt::createVerifyEpochLoweredPass() {
  return std::make_unique<VerifyEpochLoweredPass>();
}
