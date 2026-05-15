///==========================================================================///
/// File: CodirCodeletOpt.cpp
///
/// CODIR-local cleanup for isolated codelet bodies.
///==========================================================================///

#include "carts/dialect/codir/Transforms/Passes.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::carts::codir {
#define GEN_PASS_DEF_CODIRCODELETOPT
#include "carts/dialect/codir/Transforms/Passes.h.inc"
} // namespace mlir::carts::codir

using namespace mlir;
using namespace mlir::carts;

namespace {

struct CodirCodeletOptPass
    : public codir::impl::CodirCodeletOptBase<CodirCodeletOptPass> {
  static bool isDeadCodeletBodyOp(Operation *op) {
    if (!op || !op->use_empty() || op->getNumRegions() != 0 ||
        op->hasTrait<OpTrait::IsTerminator>())
      return false;
    return isMemoryEffectFree(op);
  }

  void runOnOperation() override {
    SmallVector<codir::CodeletOp> codelets;
    getOperation().walk(
        [&](codir::CodeletOp codelet) { codelets.push_back(codelet); });

    for (codir::CodeletOp codelet : codelets) {
      if (codelet.getBody().empty())
        continue;

      bool changed = true;
      while (changed) {
        changed = false;
        SmallVector<Operation *> deadOps;
        codelet.getBody().walk([&](Operation *op) {
          if (op == codelet.getOperation())
            return;
          if (isDeadCodeletBodyOp(op))
            deadOps.push_back(op);
        });

        for (Operation *op : deadOps) {
          op->erase();
          changed = true;
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::carts::codir::createCodirCodeletOptPass() {
  return std::make_unique<CodirCodeletOptPass>();
}
