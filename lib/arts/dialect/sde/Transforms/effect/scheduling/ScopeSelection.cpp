///==========================================================================///
/// File: ScopeSelection.cpp
///
/// Cost-model-backed SDE concurrency scope selection. The current
/// implementation preserves explicit scope, lets nested parallel SDE regions
/// inherit the nearest enclosing parallel scope, and otherwise fills missing
/// scope from the active machine topology.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_SCOPESELECTION
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/LoopUtils.h"
#include "arts/utils/costs/SDECostModel.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::arts;

namespace {

static void assignParallelScopes(Operation *operation,
                                 sde::SdeConcurrencyScopeAttr inheritedScope,
                                 sde::SdeConcurrencyScopeAttr defaultScope) {
  auto currentScope = inheritedScope;
  if (auto cuRegion = dyn_cast<sde::SdeCuRegionOp>(operation);
      cuRegion && cuRegion.getKind() == sde::SdeCuKind::parallel) {
    currentScope = cuRegion.getConcurrencyScopeAttr();
    if (!currentScope) {
      currentScope = inheritedScope ? inheritedScope : defaultScope;
      cuRegion.setConcurrencyScopeAttr(currentScope);
    }
  }

  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (Operation &nestedOp : block)
        assignParallelScopes(&nestedOp, currentScope, defaultScope);
}

/// For su_iterate ops with distributed scope, check if remote access cost
/// justifies distribution. If not, override to local.
static void refineDistributedScope(Operation *operation,
                                   sde::SDECostModel &costModel) {
  operation->walk([&](sde::SdeSuIterateOp op) {
    // Only refine if we're inside a distributed parallel region.
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto cuRegion = dyn_cast<sde::SdeCuRegionOp>(parent);
      if (!cuRegion || cuRegion.getKind() != sde::SdeCuKind::parallel)
        continue;
      auto scope = cuRegion.getConcurrencyScope();
      if (!scope || *scope != sde::SdeConcurrencyScope::distributed)
        continue;

      // Estimate remote access count from the loop body.
      int64_t memAccessCount = 0;
      op.getBody().walk([&](Operation *inner) {
        if (isa<memref::LoadOp, memref::StoreOp, tensor::ExtractOp,
                tensor::InsertOp>(inner))
          ++memAccessCount;
      });

      std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation());
      if (!tripCount)
        break;

      // Distributed cost: only (N-1)/N of accesses are remote.
      int nodeCount = std::max(1, costModel.getLogicalNodeCapacity());
      double remoteFraction = (nodeCount - 1.0) / nodeCount;
      double distributedCost =
          memAccessCount * *tripCount *
          (remoteFraction * costModel.getRemoteDataAccessCost() +
           (1.0 - remoteFraction) * costModel.getLocalDataAccessCost());
      double localCost =
          memAccessCount * *tripCount * costModel.getLocalDataAccessCost();

      // If distributed cost exceeds 8x local cost, prefer local.
      // High threshold because distribution enables parallelism benefits
      // that outweigh raw data-movement cost for most workloads.
      if (distributedCost > localCost * 8.0) {
        cuRegion.setConcurrencyScopeAttr(sde::SdeConcurrencyScopeAttr::get(
            cuRegion.getContext(), sde::SdeConcurrencyScope::local));
      }
      break;
    }
  });
}

struct ScopeSelectionPass
    : public arts::impl::ScopeSelectionBase<ScopeSelectionPass> {
  explicit ScopeSelectionPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    if (!costModel)
      return;

    auto desiredScope = sde::SdeConcurrencyScopeAttr::get(
        &getContext(), costModel->isDistributed()
                           ? sde::SdeConcurrencyScope::distributed
                           : sde::SdeConcurrencyScope::local);

    assignParallelScopes(getOperation(), {}, desiredScope);

    // Refine: on distributed machines, check if distributed scope is
    // cost-justified for each loop. Skip on single-node (scope is already
    // local) to avoid overriding explicitly-set distributed scopes in tests.
    if (costModel->isDistributed())
      refineDistributedScope(getOperation(), *costModel);
  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::arts::sde {

std::unique_ptr<Pass> createScopeSelectionPass(sde::SDECostModel *costModel) {
  return std::make_unique<ScopeSelectionPass>(costModel);
}

} // namespace mlir::arts::sde
