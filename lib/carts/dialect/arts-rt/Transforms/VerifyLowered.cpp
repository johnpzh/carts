///==========================================================================///
/// File: VerifyLowered.cpp
///
/// Verification pass that ensures no ARTS dialect operations (core or
/// runtime) survive past the arts-to-llvm conversion. Any remaining
/// arts.* or arts_rt.* op indicates a lowering bug.
///==========================================================================///

#include "carts/Dialect.h"
#define GEN_PASS_DEF_VERIFYLOWERED
#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct VerifyLoweredPass : public impl::VerifyLoweredBase<VerifyLoweredPass> {
  void runOnOperation() override {
    bool found = false;
    auto *artsRtDialect = getOperation()
                              ->getContext()
                              ->getLoadedDialect<arts::rt::ArtsRtDialect>();
    getOperation().walk([&](Operation *op) {
      if (isArtsOp(op)) {
        op->emitError("high-level ARTS operation survived past arts-to-llvm");
        found = true;
      } else if (artsRtDialect && op->getDialect() == artsRtDialect) {
        op->emitError("arts_rt operation survived past arts-to-llvm");
        found = true;
      }
    });
    if (found)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::arts::createVerifyLoweredPass() {
  return std::make_unique<VerifyLoweredPass>();
}
