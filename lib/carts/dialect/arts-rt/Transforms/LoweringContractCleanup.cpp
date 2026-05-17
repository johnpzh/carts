///==========================================================================///
/// File: LoweringContractCleanup.cpp
///
/// Removes arts.lowering_contract metadata ops after ARTS-RT lowering has
/// consumed them and before final LLVM translation starts.
///==========================================================================///

#include "carts/Dialect.h"
#include "carts/dialect/arts-rt/Transforms/Passes.h"
#include "carts/utils/Debug.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::carts::arts_rt {
#define GEN_PASS_DEF_LOWERINGCONTRACTCLEANUP
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts_rt

ARTS_DEBUG_SETUP(lowering_contract_cleanup);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
using namespace mlir::carts::arts_rt;

namespace {

struct LoweringContractCleanupPass
    : public arts_rt::impl::LoweringContractCleanupBase<
          LoweringContractCleanupPass> {
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

std::unique_ptr<Pass>
mlir::carts::arts_rt::createLoweringContractCleanupPass() {
  return std::make_unique<LoweringContractCleanupPass>();
}
