///==========================================================================///
/// File: VerifyPreLowered.cpp
///
/// Verification pass that ensures high-level scheduler operations do not
/// survive past the pre-lowering stage.
///
/// By the end of pre-lowering, high-level structural ops should already be
/// converted into runtime-facing forms (e.g., edt_create/create_epoch), so
/// residual arts.edt/arts.epoch indicate a lowering boundary bug.
///==========================================================================///

#include "carts/dialect/arts-rt/Transforms/Passes.h"
namespace mlir::carts::arts_rt {
#define GEN_PASS_DEF_VERIFYPRELOWERED
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts_rt
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::carts;

namespace {
struct VerifyPreLoweredPass
    : public arts_rt::impl::VerifyPreLoweredBase<VerifyPreLoweredPass> {
  void runOnOperation() override {
    bool found = false;
    getOperation().walk([&](Operation *op) {
      if (isa<arts::EdtOp, arts::EpochOp>(op)) {
        op->emitError(
            "high-level scheduler op survived past pre-lowering step");
        found = true;
      }
    });
    if (found)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::carts::arts_rt::createVerifyPreLoweredPass() {
  return std::make_unique<VerifyPreLoweredPass>();
}
