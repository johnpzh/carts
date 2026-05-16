///==========================================================================///
/// File: VerifySdeLowered.cpp
///
/// Verification pass that ensures no sde.* operations survive after boundary
/// conversion. Any remaining SDE op indicates a conversion failure.
///==========================================================================///

#include "carts/Dialect.h"
#include "carts/dialect/sde/IR/SdeDialect.h"
#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::carts::sde {
#define GEN_PASS_DEF_VERIFYSDELOWERED
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::carts;

namespace {

struct VerifySdeLoweredPass
    : public sde::impl::VerifySdeLoweredBase<VerifySdeLoweredPass> {
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
