///==========================================================================///
/// File: VerifyArtsObjectsOnly.cpp
///
/// Verification pass for the target CODIR-to-ARTS boundary contract.
///==========================================================================///

#include "carts/Dialect.h"
#include "carts/dialect/sde/IR/SdeDialect.h"
#define GEN_PASS_DEF_VERIFYARTSOBJECTSONLY
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::carts;

namespace {

static LogicalResult verifyArtsObjectsOnly(ModuleOp module) {
  auto *sdeDialect =
      module->getContext()->getLoadedDialect<sde::CartsSdeDialect>();
  bool found = false;

  module.walk([&](Operation *op) {
    if (sdeDialect && op->getDialect() == sdeDialect) {
      op->emitError() << "SDE operation '" << op->getName()
                      << "' remains after the CODIR-to-ARTS boundary";
      found = true;
    }

    if (op->getDialect() && op->getDialect()->getNamespace() == "omp") {
      op->emitError() << "OpenMP operation '" << op->getName()
                      << "' remains after the CODIR-to-ARTS boundary";
      found = true;
    }

    if (isa<scf::ParallelOp>(op)) {
      op->emitError()
          << "scf.parallel remains after the CODIR-to-ARTS boundary; "
             "parallel work must be materialized as ARTS objects";
      found = true;
    }
  });

  return failure(found);
}

struct VerifyArtsObjectsOnlyPass
    : public impl::VerifyArtsObjectsOnlyBase<VerifyArtsObjectsOnlyPass> {
  VerifyArtsObjectsOnlyPass() = default;

  void runOnOperation() override {
    if (failed(verifyArtsObjectsOnly(getOperation())))
      signalPassFailure();
  }
};

} /// namespace

std::unique_ptr<Pass> mlir::carts::arts::createVerifyArtsObjectsOnlyPass() {
  return std::make_unique<VerifyArtsObjectsOnlyPass>();
}
