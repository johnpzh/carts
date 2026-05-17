///==========================================================================///
/// File: VerifyEdtCreated.cpp
///
/// Verification pass that ensures arts.edt operations exist after OpenMP
/// conversion. Warns if the module has no EDTs, which may indicate a
/// conversion issue.
///==========================================================================///

#include "carts/dialect/arts/IR/ArtsDialect.h"
#define GEN_PASS_DEF_VERIFYEDTCREATED
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::carts;

namespace {
struct VerifyEdtCreatedPass
    : public impl::VerifyEdtCreatedBase<VerifyEdtCreatedPass> {
  void runOnOperation() override {
    bool found = false;
    getOperation().walk([&](arts::EdtOp op) {
      found = true;
      return WalkResult::interrupt();
    });
    if (!found)
      getOperation().emitWarning(
          "no arts.edt operations found after OpenMP conversion");
  }
};
} // namespace

std::unique_ptr<Pass> mlir::carts::arts::createVerifyEdtCreatedPass() {
  return std::make_unique<VerifyEdtCreatedPass>();
}
