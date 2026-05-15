///==========================================================================///
/// File: LoweringContractCleanup.cpp
///
/// Removes arts.lowering_contract metadata ops once all contract consumers
/// have run and before LLVM lowering starts.
///==========================================================================///

#include "carts/Dialect.h"
#define GEN_PASS_DEF_LOWERINGCONTRACTCLEANUP
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "carts/utils/Debug.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

ARTS_DEBUG_SETUP(lowering_contract_cleanup);

using namespace mlir;
using namespace mlir::arts;

namespace {

struct LoweringContractCleanupPass
    : public impl::LoweringContractCleanupBase<LoweringContractCleanupPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<LoweringContractOp, 16> contracts;
    module.walk(
        [&](LoweringContractOp contract) { contracts.push_back(contract); });

    for (LoweringContractOp contract : contracts)
      contract.erase();

    ARTS_DEBUG("LoweringContractCleanup removed "
               << contracts.size() << " lowering_contract op(s)");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::arts::createLoweringContractCleanupPass() {
  return std::make_unique<LoweringContractCleanupPass>();
}
