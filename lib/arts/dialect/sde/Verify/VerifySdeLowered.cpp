///==========================================================================///
/// File: VerifySdeLowered.cpp
///
/// Verification pass that ensures no sde.* operations and no transient
/// linalg/tensor carriers survive after SDE-to-ARTS conversion. Any
/// remaining SDE ops or carriers indicate a conversion failure.
///==========================================================================///

#include "arts/Dialect.h"
#include "arts/dialect/sde/IR/SdeDialect.h"
#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_VERIFYSDELOWERED
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

/// Returns true if the op is a transient carrier that should have been erased
/// by ConvertSdeToArts.
static bool isOrphanedCarrierOp(Operation *op) {
  auto underCoreBoundary = [&] {
    return op->getParentOfType<arts::EdtOp>() ||
           op->getParentOfType<arts::EpochOp>();
  };

  // Transient tensor/linalg carriers inside Core-owned work regions should
  // have been materialized before the SDE-to-ARTS boundary verifier runs.
  if (isa<linalg::LinalgOp, bufferization::ToTensorOp, tensor::EmptyOp>(op))
    return underCoreBoundary();

  return false;
}

struct VerifySdeLoweredPass
    : public arts::impl::VerifySdeLoweredBase<VerifySdeLoweredPass> {
  void runOnOperation() override {
    auto *sdeDialect = getOperation()
                           ->getContext()
                           ->getLoadedDialect<arts::sde::ArtsSdeDialect>();
    bool failed = false;
    getOperation().walk([&](Operation *op) {
      if (sdeDialect && op->getDialect() == sdeDialect) {
        op->emitError() << "SDE operation '" << op->getName()
                        << "' survived past SDE-to-ARTS conversion";
        failed = true;
      }
      if (isOrphanedCarrierOp(op)) {
        op->emitError() << "transient carrier '" << op->getName()
                        << "' survived past SDE-to-ARTS conversion";
        failed = true;
      }
    });
    if (failed)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::arts::sde::createVerifySdeLoweredPass() {
  return std::make_unique<VerifySdeLoweredPass>();
}
