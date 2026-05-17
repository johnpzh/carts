///==========================================================================///
/// File: VerifyEdtLowered.cpp
///
/// Verification pass that ensures no arts.edt operations survive past EDT
/// lowering.
///==========================================================================///

#include "carts/dialect/arts-rt/Transforms/Passes.h"
namespace mlir::carts::arts_rt {
#define GEN_PASS_DEF_VERIFYEDTLOWERED
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts_rt
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::carts;

namespace {
struct VerifyEdtLoweredPass
    : public arts_rt::impl::VerifyEdtLoweredBase<VerifyEdtLoweredPass> {
  void runOnOperation() override {
    bool found = false;
    getOperation().walk([&](arts::EdtOp op) {
      op.emitError("arts.edt survived past EDT lowering");
      found = true;
    });
    if (found)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::carts::arts_rt::createVerifyEdtLoweredPass() {
  return std::make_unique<VerifyEdtLoweredPass>();
}
