///==========================================================================///
/// File: VerifySdeLowered.cpp
///
/// Verification pass that ensures no sde.* operations survive after boundary
/// conversion. Any remaining SDE op indicates a conversion failure.
///==========================================================================///

#include "arts/Dialect.h"
#include "arts/dialect/sde/IR/SdeDialect.h"
#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_VERIFYSDELOWERED
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::carts;

namespace {

struct VerifySdeLoweredPass
    : public arts::impl::VerifySdeLoweredBase<VerifySdeLoweredPass> {
  void runOnOperation() override {
    auto *sdeDialect = getOperation()
                           ->getContext()
                           ->getLoadedDialect<sde::CartsSdeDialect>();
    bool failed = false;
    getOperation().walk([&](Operation *op) {
      if (sdeDialect && op->getDialect() == sdeDialect) {
        op->emitError() << "SDE operation '" << op->getName()
                        << "' survived past boundary conversion";
        failed = true;
      }
    });
    if (failed)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::carts::sde::createVerifySdeLoweredPass() {
  return std::make_unique<VerifySdeLoweredPass>();
}
