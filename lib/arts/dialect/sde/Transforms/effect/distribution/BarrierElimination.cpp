///==========================================================================///
/// File: BarrierElimination.cpp
///
/// Eliminate redundant SDE barriers between independent scheduling units.
/// When the write-set of the predecessor loop is provably disjoint from the
/// read-set of the successor loop, the barrier is marked for elimination so
/// boundary materialization omits the downstream synchronization object.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_BARRIERELIMINATION
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "arts/utils/ValueAnalysis.h"
#include "arts/utils/costs/SDECostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  sde::SdeSuIterateOp result;
  bool multiple = false;
  op->walk([&](sde::SdeSuIterateOp suIt) {
    if (result && result.getOperation() != suIt.getOperation()) {
      multiple = true;
      return WalkResult::interrupt();
    }
    result = suIt;
    return WalkResult::advance();
  });
  if (multiple)
    return {};
  return result;
}

/// Return true if the operation is an SDE scheduling-unit container
/// (su_iterate or su_distribute wrapping one).
static bool isSuContainer(Operation *op) {
  return static_cast<bool>(findSuIterate(op));
}

static void setBarrierReason(sde::SdeSuBarrierOp barrier,
                             sde::SdeBarrierReason reason) {
  barrier.setBarrierReasonAttr(
      sde::SdeBarrierReasonAttr::get(barrier.getContext(), reason));
}

static bool haveSameIterationShape(sde::SdeSuIterateOp lhs,
                                   sde::SdeSuIterateOp rhs) {
  return ValueAnalysis::areValueRangesEquivalent(lhs.getLowerBounds(),
                                                 rhs.getLowerBounds()) &&
         ValueAnalysis::areValueRangesEquivalent(lhs.getUpperBounds(),
                                                 rhs.getUpperBounds()) &&
         ValueAnalysis::areValueRangesEquivalent(lhs.getSteps(),
                                                 rhs.getSteps());
}

static bool haveSameIterationBounds(sde::SdeSuIterateOp lhs,
                                    sde::SdeSuIterateOp rhs) {
  return lhs.getLowerBounds().size() == rhs.getLowerBounds().size() &&
         ValueAnalysis::areValueRangesEquivalent(lhs.getLowerBounds(),
                                                 rhs.getLowerBounds()) &&
         ValueAnalysis::areValueRangesEquivalent(lhs.getUpperBounds(),
                                                 rhs.getUpperBounds());
}

static bool isTiledMultipleOfStep(Value candidate, Value baseStep) {
  if (ValueAnalysis::areValuesEquivalent(candidate, baseStep))
    return true;

  auto mul = candidate.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;

  auto isPositiveMultiplier = [](Value value) {
    if (std::optional<int64_t> folded =
            ValueAnalysis::tryFoldConstantIndex(value))
      return *folded >= 1;
    return ValueAnalysis::isConstantAtLeastOne(value) ||
           ValueAnalysis::isProvablyNonZero(value);
  };

  if (ValueAnalysis::areValuesEquivalent(mul.getLhs(), baseStep))
    return isPositiveMultiplier(mul.getRhs());
  if (ValueAnalysis::areValuesEquivalent(mul.getRhs(), baseStep))
    return isPositiveMultiplier(mul.getLhs());
  return false;
}

static bool haveEquivalentOrTiledSteps(sde::SdeSuIterateOp lhs,
                                       sde::SdeSuIterateOp rhs) {
  if (lhs.getSteps().size() != rhs.getSteps().size())
    return false;

  for (auto [lhsStep, rhsStep] : llvm::zip(lhs.getSteps(), rhs.getSteps())) {
    if (ValueAnalysis::areValuesEquivalent(lhsStep, rhsStep))
      continue;
    if (isTiledMultipleOfStep(lhsStep, rhsStep) ||
        isTiledMultipleOfStep(rhsStep, lhsStep))
      continue;
    return false;
  }
  return true;
}

static bool isUniformRepeatableStage(sde::SdeSuIterateOp op) {
  if (!op || !op.getReductionAccumulators().empty())
    return false;

  if (auto family = op.getPattern()) {
    return *family == sde::SdePattern::uniform ||
           *family == sde::SdePattern::elementwise_pipeline;
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
  auto family = op.getPattern();
  return !family || *family == sde::SdePattern::stencil_tiling_nd ||
         *family == sde::SdePattern::jacobi_alternating_buffers;
}

static bool isWavefrontFrontierStage(sde::SdeSuIterateOp op) {
  auto family = op.getPattern();
  return family && *family == sde::SdePattern::wavefront_2d;
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

static bool sameI64ArrayAttr(ArrayAttr lhs, ArrayAttr rhs) {
  if (!lhs || !rhs || lhs.size() != rhs.size())
    return false;
  for (auto [lhsAttr, rhsAttr] : llvm::zip(lhs, rhs)) {
    auto lhsInt = dyn_cast<IntegerAttr>(lhsAttr);
    auto rhsInt = dyn_cast<IntegerAttr>(rhsAttr);
    if (!lhsInt || !rhsInt || lhsInt.getInt() != rhsInt.getInt())
      return false;
  }
  return true;
}

static bool haveSamePhysicalTimestepPlan(sde::SdeSuIterateOp predecessor,
                                         sde::SdeSuIterateOp successor) {
  return sameI64ArrayAttr(predecessor.getPhysicalOwnerDimsAttr(),
                          successor.getPhysicalOwnerDimsAttr()) &&
         sameI64ArrayAttr(predecessor.getPhysicalBlockShapeAttr(),
                          successor.getPhysicalBlockShapeAttr());
}

static bool sameSdeIterationTopology(sde::SdeSuIterateOp lhs,
                                     sde::SdeSuIterateOp rhs) {
  auto lhsTopology = lhs.getIterationTopology();
  auto rhsTopology = rhs.getIterationTopology();
  return lhsTopology && rhsTopology && *lhsTopology == *rhsTopology;
}

static bool haveSdeApprovedTiledTimestepPlan(sde::SdeSuIterateOp lhs,
                                             sde::SdeSuIterateOp rhs) {
  return haveSameIterationBounds(lhs, rhs) &&
         haveSamePhysicalTimestepPlan(lhs, rhs) &&
         sameI64ArrayAttr(lhs.getLogicalWorkerSliceAttr(),
                          rhs.getLogicalWorkerSliceAttr()) &&
         sameSdeIterationTopology(lhs, rhs) &&
         haveEquivalentOrTiledSteps(lhs, rhs);
}

static bool haveCompatibleTimestepIterationPlan(sde::SdeSuIterateOp lhs,
                                                sde::SdeSuIterateOp rhs) {
  if (haveSameIterationShape(lhs, rhs))
    return true;
  return haveSdeApprovedTiledTimestepPlan(lhs, rhs);
}

static bool isTimestepInterstitialOp(Operation *op) {
  if (!op)
    return false;
  if (op->getNumRegions() != 0) {
    auto effects = sde::collectStructuredMemoryEffects(op);
    return !effects.hasUnknownEffects && effects.empty();
  }
  return !sde::hasUnmodeledMemoryEffect(op);
}

static void stampJacobiTimestepPlan(sde::SdeSuIterateOp predecessor,
                                    sde::SdeSuIterateOp successor,
                                    bool predecessorIsStencil,
                                    bool successorIsStencil) {
  stampRepeatedTimestepPlan(predecessor);
  stampRepeatedTimestepPlan(successor);
  if (predecessorIsStencil)
    predecessor.setPatternAttr(sde::SdePatternAttr::get(
        predecessor.getContext(),
        sde::SdePattern::jacobi_alternating_buffers));
  if (successorIsStencil)
    successor.setPatternAttr(sde::SdePatternAttr::get(
        successor.getContext(), sde::SdePattern::jacobi_alternating_buffers));
}

static bool stampTimestepPlanIfRecognized(
    sde::SdeSuIterateOp predecessor, sde::SdeSuIterateOp successor,
    const sde::StructuredMemoryEffectSummary &predEffects,
    const sde::StructuredMemoryEffectSummary &succEffects,
    bool allowStencilStencilPlan, bool allowUniformUniformPlan) {
  if (!predecessor || !successor)
    return false;
  if (predEffects.hasUnknownEffects || succEffects.hasUnknownEffects)
    return false;
  if (!haveCompatibleTimestepIterationPlan(predecessor, successor))
    return false;

  if (isUniformRepeatableStage(predecessor) &&
      isUniformRepeatableStage(successor) &&
      allowUniformUniformPlan &&
      writesIntersectReads(predEffects, succEffects) &&
      haveSamePhysicalTimestepPlan(predecessor, successor)) {
    stampRepeatedTimestepPlan(predecessor);
    stampRepeatedTimestepPlan(successor);
    return true;
  }

  bool predStencil = isOutOfPlaceStencilStage(predecessor, predEffects);
  bool succStencil = isOutOfPlaceStencilStage(successor, succEffects);
  bool predUniform = isUniformRepeatableStage(predecessor);
  bool succUniform = isUniformRepeatableStage(successor);
  if (!hasAlternatingBufferExchange(predEffects, succEffects))
    return false;

  if (predStencil && succStencil && allowStencilStencilPlan) {
    stampRepeatedTimestepPlan(predecessor);
    stampRepeatedTimestepPlan(successor);
    return true;
  }

  if ((predStencil && succUniform) || (predUniform && succStencil)) {
    stampJacobiTimestepPlan(predecessor, successor, predStencil, succStencil);
    return true;
  }

  return false;
}

static bool stampAdjacentTimestepPair(Operation *predOp, Operation *succOp) {
  sde::SdeSuIterateOp predecessor = findSuIterate(predOp);
  sde::SdeSuIterateOp successor = findSuIterate(succOp);
  if (!predecessor || !successor)
    return false;
  if (!predecessor.getStructuredClassificationAttr() ||
      !successor.getStructuredClassificationAttr())
    return false;

  auto predEffects =
      sde::collectStructuredMemoryEffects(predecessor.getOperation());
  auto succEffects = sde::collectStructuredMemoryEffects(successor.getOperation());
  return stampTimestepPlanIfRecognized(predecessor, successor, predEffects,
                                       succEffects,
                                       /*allowStencilStencilPlan=*/false,
                                       /*allowUniformUniformPlan=*/false);
}

static unsigned stampAdjacentTimestepPairsInLoop(scf::ForOp loop) {
  Operation *previousStage = nullptr;
  unsigned stamped = 0;

  for (Operation &op : loop.getBody()->without_terminator()) {
    if (findSuIterate(&op)) {
      if (previousStage && stampAdjacentTimestepPair(previousStage, &op))
        ++stamped;
      previousStage = &op;
      continue;
    }

    if (isTimestepInterstitialOp(&op))
      continue;

    previousStage = nullptr;
  }

  return stamped;
}

struct BarrierEliminationPass
    : public arts::impl::BarrierEliminationBase<BarrierEliminationPass> {
  explicit BarrierEliminationPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    int eliminated = 0;
    unsigned timestepPairsStamped = 0;

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

      if (stampTimestepPlanIfRecognized(predecessor, successor, predEffects,
                                        succEffects,
                                        /*allowStencilStencilPlan=*/true,
                                        /*allowUniformUniformPlan=*/true)) {
        ++timestepPairsStamped;
        setBarrierReason(barrier,
                         sde::SdeBarrierReason::timestep_stage_boundary);
        return;
      }

      setBarrierReason(barrier, sde::SdeBarrierReason::required_memory);
    });

    getOperation().walk([&](scf::ForOp loop) {
      timestepPairsStamped += stampAdjacentTimestepPairsInLoop(loop);
    });

    ARTS_INFO("BarrierElimination: eliminated " << eliminated << " barrier(s)");
    ARTS_INFO("BarrierElimination: stamped " << timestepPairsStamped
                                             << " timestep pair(s)");

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
