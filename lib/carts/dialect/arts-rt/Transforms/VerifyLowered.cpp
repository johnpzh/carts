///==========================================================================///
/// File: VerifyLowered.cpp
///
/// Verification pass that ensures no ARTS dialect operations (core or
/// runtime) survive past ARTS-RT to LLVM conversion. Any remaining
/// arts.* or arts_rt.* op indicates a lowering bug.
///==========================================================================///

#include "carts/Dialect.h"
#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "carts/dialect/arts-rt/Transforms/Passes.h"
namespace mlir::carts::arts_rt {
#define GEN_PASS_DEF_VERIFYLOWERED
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts_rt
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::carts;

namespace {
struct VerifyLoweredPass
    : public arts_rt::impl::VerifyLoweredBase<VerifyLoweredPass> {
  void runOnOperation() override {
    bool found = false;
    auto *artsRtDialect = getOperation()
                              ->getContext()
                              ->getLoadedDialect<arts::rt::ArtsRtDialect>();
    getOperation().walk([&](Operation *op) {
      if (isArtsOp(op)) {
        op->emitError("high-level ARTS operation survived past "
                      "arts-rt-to-llvm");
        found = true;
      } else if (artsRtDialect && op->getDialect() == artsRtDialect) {
        op->emitError("arts_rt operation survived past arts-rt-to-llvm");
        found = true;
      }
    });
    if (found)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::carts::arts_rt::createVerifyLoweredPass() {
  return std::make_unique<VerifyLoweredPass>();
}
