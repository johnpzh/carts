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
#include "arts/utils/ValueAnalysis.h"
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

static void setBarrierReason(sde::SdeSuBarrierOp barrier,
                             sde::SdeBarrierReason reason) {
  barrier.setBarrierReasonAttr(
      sde::SdeBarrierReasonAttr::get(barrier.getContext(), reason));
}

static bool sameValueRange(ValueRange lhs, ValueRange rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [lhsValue, rhsValue] : llvm::zip(lhs, rhs))
    if (!ValueAnalysis::sameValue(lhsValue, rhsValue))
      return false;
  return true;
}

static bool haveSameIterationShape(sde::SdeSuIterateOp lhs,
                                   sde::SdeSuIterateOp rhs) {
  return sameValueRange(lhs.getLowerBounds(), rhs.getLowerBounds()) &&
         sameValueRange(lhs.getUpperBounds(), rhs.getUpperBounds()) &&
         sameValueRange(lhs.getSteps(), rhs.getSteps());
}

static bool isUniformRepeatableStage(sde::SdeSuIterateOp op) {
  if (!op || !op.getReductionAccumulators().empty())
    return false;

  if (auto family = op.getDepFamily()) {
    return *family == sde::SdeDepFamily::uniform ||
           *family == sde::SdeDepFamily::elementwise_pipeline;
  }

  auto classification = op.getStructuredClassification();
  return classification &&
         (*classification == sde::SdeStructuredClassification::elementwise ||
          *classification ==
              sde::SdeStructuredClassification::elementwise_pipeline);
}

static bool isOutOfPlaceStencilStage(
    sde::SdeSuIterateOp op,
    const sde::StructuredMemoryEffectSummary &effects) {
  if (!op || effects.hasUnknownEffects)
    return false;
  auto classification = op.getStructuredClassification();
  if (!classification ||
      *classification != sde::SdeStructuredClassification::stencil)
    return false;
  if (sde::hasInPlaceSelfRead(effects))
    return false;
  auto family = op.getDepFamily();
  return !family || *family == sde::SdeDepFamily::stencil_tiling_nd ||
         *family == sde::SdeDepFamily::jacobi_alternating_buffers;
}

static bool isWavefrontFrontierStage(sde::SdeSuIterateOp op) {
  auto family = op.getDepFamily();
  return family && *family == sde::SdeDepFamily::wavefront_2d;
}

static void stampRepeatedTimestepPlan(sde::SdeSuIterateOp op) {
  if (!op)
    return;
  if (!op.getRepetitionStructureAttr())
    op.setRepetitionStructureAttr(sde::SdeRepetitionStructureAttr::get(
        op.getContext(), sde::SdeRepetitionStructure::full_timestep));
  if (!op.getAsyncStrategyAttr())
    op.setAsyncStrategyAttr(sde::SdeAsyncStrategyAttr::get(
        op.getContext(), sde::SdeAsyncStrategy::advance_edt));
}

static bool writesIntersectReads(const sde::StructuredMemoryEffectSummary &lhs,
                                 const sde::StructuredMemoryEffectSummary &rhs) {
  for (Value written : lhs.writes)
    if (rhs.reads.contains(written))
      return true;
  return false;
}

static bool hasAlternatingBufferExchange(
    const sde::StructuredMemoryEffectSummary &lhs,
    const sde::StructuredMemoryEffectSummary &rhs) {
  return writesIntersectReads(lhs, rhs) && writesIntersectReads(rhs, lhs);
}

static void stampJacobiTimestepPlan(sde::SdeSuIterateOp predecessor,
                                    sde::SdeSuIterateOp successor,
                                    bool predecessorIsStencil) {
  stampRepeatedTimestepPlan(predecessor);
  stampRepeatedTimestepPlan(successor);
  sde::SdeSuIterateOp stencil = predecessorIsStencil ? predecessor : successor;
  stencil.setDepFamilyAttr(sde::SdeDepFamilyAttr::get(
      stencil.getContext(), sde::SdeDepFamily::jacobi_alternating_buffers));
}

struct BarrierEliminationPass
    : public arts::impl::BarrierEliminationBase<BarrierEliminationPass> {
  explicit BarrierEliminationPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    int eliminated = 0;

    getOperation().walk([&](sde::SdeSuBarrierOp barrier) {
      setBarrierReason(barrier, sde::SdeBarrierReason::unknown_required);

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
        barrier.setBarrierEliminatedAttr(UnitAttr::get(barrier.getContext()));
        setBarrierReason(barrier, sde::SdeBarrierReason::redundant);
        eliminated++;
        ARTS_DEBUG("Eliminated barrier (sync cost: " << syncCost << ")");
        return;
      }

      if (isWavefrontFrontierStage(predecessor) ||
          isWavefrontFrontierStage(successor)) {
        setBarrierReason(barrier, sde::SdeBarrierReason::wavefront_frontier);
        return;
      }

      if (isUniformRepeatableStage(predecessor) &&
          isUniformRepeatableStage(successor) &&
          haveSameIterationShape(predecessor, successor)) {
        stampRepeatedTimestepPlan(predecessor);
        stampRepeatedTimestepPlan(successor);
        setBarrierReason(barrier,
                         sde::SdeBarrierReason::timestep_stage_boundary);
        return;
      }

      bool predStencil = isOutOfPlaceStencilStage(predecessor, predEffects);
      bool succStencil = isOutOfPlaceStencilStage(successor, succEffects);
      bool predUniform = isUniformRepeatableStage(predecessor);
      bool succUniform = isUniformRepeatableStage(successor);
      if (haveSameIterationShape(predecessor, successor) &&
          hasAlternatingBufferExchange(predEffects, succEffects) &&
          ((predStencil && succUniform) || (predUniform && succStencil))) {
        stampJacobiTimestepPlan(predecessor, successor, predStencil);
        setBarrierReason(barrier,
                         sde::SdeBarrierReason::timestep_stage_boundary);
        return;
      }

      setBarrierReason(barrier, sde::SdeBarrierReason::required_memory);
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
