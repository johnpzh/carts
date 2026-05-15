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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::carts;

namespace {

struct VerifyCoreObjectsOnlyPass
    : public impl::VerifyCoreObjectsOnlyBase<VerifyCoreObjectsOnlyPass> {
  VerifyCoreObjectsOnlyPass() = default;

  void runOnOperation() override {
    auto module = getOperation();
    auto *sdeDialect =
        module->getContext()->getLoadedDialect<sde::CartsSdeDialect>();
    bool found = false;

    module.walk([&](Operation *op) {
      if (sdeDialect && op->getDialect() == sdeDialect) {
        op->emitError() << "SDE operation '" << op->getName()
                        << "' remains after the SDE/Core boundary";
        found = true;
      }

      if (op->getDialect() && op->getDialect()->getNamespace() == "omp") {
        op->emitError() << "OpenMP operation '" << op->getName()
                        << "' remains after the SDE/Core boundary";
        found = true;
      }

      if (isa<scf::ParallelOp>(op)) {
        op->emitError()
            << "scf.parallel remains after the SDE/Core boundary; "
               "parallel work must be materialized as Core objects";
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
