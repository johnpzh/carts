///==========================================================================///
/// File: VerifyEpochCreated.cpp
///
/// Verification pass that ensures arts.epoch operations exist after epoch
/// creation. Warns if EDTs are present but no epochs were created, which
/// may indicate missing synchronization.
///==========================================================================///

#include "carts/dialect/arts/IR/ArtsDialect.h"
#define GEN_PASS_DEF_VERIFYEPOCHCREATED
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::carts;

namespace {
struct VerifyEpochCreatedPass
    : public impl::VerifyEpochCreatedBase<VerifyEpochCreatedPass> {
  void runOnOperation() override {
    auto module = getOperation();
    bool hasEdts = false;
    bool hasEpochs = false;
    module.walk([&](arts::EdtOp) { hasEdts = true; });
    module.walk([&](arts::EpochOp) { hasEpochs = true; });
    if (hasEdts && !hasEpochs)
      module.emitWarning(
          "EDTs found but no arts.epoch operations after epoch creation");
  }
};
} // namespace

std::unique_ptr<Pass> mlir::carts::arts::createVerifyEpochCreatedPass() {
  return std::make_unique<VerifyEpochCreatedPass>();
}
