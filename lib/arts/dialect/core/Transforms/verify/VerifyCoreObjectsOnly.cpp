///==========================================================================///
/// File: VerifyCoreObjectsOnly.cpp
///
/// Verification pass for the target SDE/Core boundary contract.
///==========================================================================///

#include "arts/Dialect.h"
#include "arts/dialect/sde/IR/SdeDialect.h"
#define GEN_PASS_DEF_VERIFYCOREOBJECTSONLY
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct VerifyCoreObjectsOnlyPass
    : public impl::VerifyCoreObjectsOnlyBase<VerifyCoreObjectsOnlyPass> {
  VerifyCoreObjectsOnlyPass() = default;

  void runOnOperation() override {
    auto module = getOperation();
    auto *sdeDialect =
        module->getContext()->getLoadedDialect<arts::sde::ArtsSdeDialect>();
    bool found = false;

    module.walk([&](Operation *op) {
      if (sdeDialect && op->getDialect() == sdeDialect) {
        op->emitError() << "SDE operation '" << op->getName()
                        << "' remains after the SDE/Core boundary";
        found = true;
      }

    });

    if (found)
      signalPassFailure();
  }
};

} /// namespace

std::unique_ptr<Pass> mlir::arts::createVerifyCoreObjectsOnlyPass() {
  return std::make_unique<VerifyCoreObjectsOnlyPass>();
}
