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
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

static bool isSemanticParallelEdt(arts::EdtOp op) {
  return op && op.getType() == arts::EdtType::parallel;
}

struct VerifyCoreObjectsOnlyPass
    : public impl::VerifyCoreObjectsOnlyBase<VerifyCoreObjectsOnlyPass> {
  VerifyCoreObjectsOnlyPass() = default;

  explicit VerifyCoreObjectsOnlyPass(bool reportOnly) {
    this->reportOnly = reportOnly;
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *sdeDialect =
        module->getContext()->getLoadedDialect<arts::sde::ArtsSdeDialect>();
    bool found = false;

    module.walk([&](Operation *op) {
      if (sdeDialect && op->getDialect() == sdeDialect) {
        if (reportOnly) {
          llvm::errs() << "core-boundary-report: SDE operation '"
                       << op->getName()
                       << "' remains after the SDE/Core boundary\n";
        } else {
          op->emitError() << "SDE operation '" << op->getName()
                          << "' remains after the SDE/Core boundary";
        }
        found = true;
      }

      if (auto forOp = dyn_cast<arts::ForOp>(op)) {
        if (reportOnly) {
          llvm::errs() << "core-boundary-report: Core boundary still contains "
                          "arts.for loop carrier";
          if (auto parentEdt = forOp->getParentOfType<arts::EdtOp>())
            llvm::errs() << " inside parent arts.edt type "
                         << parentEdt.getTypeAttr();
          llvm::errs() << "; current compatibility producer is SDE-to-ARTS "
                          "su_iterate lowering\n";
        } else {
          auto diagnostic = forOp.emitError()
                            << "Core boundary still contains arts.for loop "
                               "carrier";
          if (auto parentEdt = forOp->getParentOfType<arts::EdtOp>())
            diagnostic << " inside parent arts.edt type "
                       << parentEdt.getTypeAttr();
          diagnostic << "; current compatibility producer is SDE-to-ARTS "
                        "su_iterate lowering";
        }
        found = true;
      }

      if (auto edtOp = dyn_cast<arts::EdtOp>(op)) {
        if (isSemanticParallelEdt(edtOp)) {
          if (reportOnly) {
            llvm::errs() << "core-boundary-report: Core boundary still "
                            "contains semantic parallel arts.edt wrapper\n";
          } else {
            edtOp.emitError()
                << "Core boundary still contains semantic parallel arts.edt "
                   "wrapper";
          }
          found = true;
        }
      }
    });

    if (found && !reportOnly)
      signalPassFailure();
  }
};

} /// namespace

std::unique_ptr<Pass>
mlir::arts::createVerifyCoreObjectsOnlyPass(bool reportOnly) {
  return std::make_unique<VerifyCoreObjectsOnlyPass>(reportOnly);
}
