///==========================================================================///
/// File: BarrierElimination.cpp
///
/// Eliminate redundant SDE barriers between independent scheduling units.
/// When the write-set of the predecessor loop is provably disjoint from the
/// read-set of the successor loop, the barrier is marked for elimination so
/// ConvertSdeToArts skips generating arts.barrier.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_BARRIERELIMINATION
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/costs/SDECostModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(barrier_elimination);

using namespace mlir;
using namespace mlir::arts;

namespace {

/// Find the su_iterate op that an operation represents. An su_iterate can
/// appear directly or nested inside an su_distribute wrapper.
static sde::SdeSuIterateOp findSuIterate(Operation *op) {
  if (auto suIt = dyn_cast<sde::SdeSuIterateOp>(op))
    return suIt;
  if (auto dist = dyn_cast<sde::SdeSuDistributeOp>(op)) {
    sde::SdeSuIterateOp result;
    dist.getBody().walk([&](sde::SdeSuIterateOp suIt) { result = suIt; });
    return result;
  }
  return {};
}

/// Return true if the operation is an SDE scheduling-unit container
/// (su_iterate or su_distribute wrapping one).
static bool isSuContainer(Operation *op) {
  return isa<sde::SdeSuIterateOp>(op) || isa<sde::SdeSuDistributeOp>(op);
}

struct BarrierEliminationPass
    : public arts::impl::BarrierEliminationBase<BarrierEliminationPass> {
  explicit BarrierEliminationPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    int eliminated = 0;

    getOperation().walk([&](sde::SdeSuBarrierOp barrier) {
      Block *block = barrier->getBlock();
      if (!block)
        return;

      Operation *predOp = nullptr;
      Operation *succOp = nullptr;

      // Walk backward to find predecessor su_iterate or su_distribute
      for (auto it = Block::reverse_iterator(barrier->getIterator());
           it != block->rend(); ++it) {
        if (isSuContainer(&*it)) {
          predOp = &*it;
          break;
        }
        if (!isMemoryEffectFree(&*it))
          break;
      }

      // Walk forward to find successor su_iterate or su_distribute
      for (auto it = std::next(barrier->getIterator()); it != block->end();
           ++it) {
        if (isSuContainer(&*it)) {
          succOp = &*it;
          break;
        }
        if (!isMemoryEffectFree(&*it))
          break;
      }

      if (!predOp || !succOp)
        return;

      auto predecessor = findSuIterate(predOp);
      auto successor = findSuIterate(succOp);
      if (!predecessor || !successor)
        return;

      // Both must have classification (analyzed)
      if (!predecessor.getStructuredClassificationAttr() ||
          !successor.getStructuredClassificationAttr())
        return;

      // Collect root-level memory accesses on both sides of the barrier.
      // Use the outer container's region to capture all memory ops.
      auto predEffects = sde::collectStructuredMemoryEffects(predOp);
      auto succEffects = sde::collectStructuredMemoryEffects(succOp);

      if (predEffects.hasUnknownEffects || succEffects.hasUnknownEffects)
        return;

      if (predEffects.empty() && succEffects.empty())
        return;

      if (!predEffects.hasWriteConflictWith(succEffects)) {
        double syncCost = costModel ? costModel->getTaskSyncCost() : 0.0;
        barrier->setAttr(AttrNames::Operation::BarrierEliminated,
                         UnitAttr::get(barrier.getContext()));
        eliminated++;
        ARTS_DEBUG("Eliminated barrier (sync cost: " << syncCost << ")");
      }
    });

    ARTS_INFO("BarrierElimination: eliminated " << eliminated << " barrier(s)");

  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::arts::sde {

std::unique_ptr<Pass>
createBarrierEliminationPass(sde::SDECostModel *costModel) {
  return std::make_unique<BarrierEliminationPass>(costModel);
}

} // namespace mlir::arts::sde
