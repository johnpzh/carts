///==========================================================================///
/// File: BarrierElimination.cpp
///
/// Plan SDE barriers between scheduling units. Independent barriers are marked
/// for elimination. Required timestep barriers are annotated as stage
/// boundaries, and SDE-owned CPS candidates get explicit completion tokens so
/// Core can consume the plan without re-discovering the dataflow shape.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_BARRIERELIMINATION
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "carts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "carts/utils/ValueAnalysis.h"
#include "carts/dialect/sde/Utils/SDECostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/Statistic.h"

#include <cassert>

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(barrier_elimination);

static llvm::Statistic numCpsCandidateGroups{
    "barrier_elimination", "NumCpsCandidateGroups",
    "Number of SDE CPS candidate groups stamped"};

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::carts;

namespace {

constexpr llvm::StringLiteral kCpsCandidateGroupId = "cps_candidate_group_id";
constexpr llvm::StringLiteral kCpsCandidateStageIndex =
    "cps_candidate_stage_index";
constexpr llvm::StringLiteral kCpsCandidateStageCount =
    "cps_candidate_stage_count";
constexpr llvm::StringLiteral kCpsCandidateRequiresTokenizedDataflow =
    "cps_candidate_requires_tokenized_dataflow";

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

static bool hasCandidateAttrs(sde::SdeSuIterateOp op) {
  return op && (op->hasAttr(kCpsCandidateGroupId) ||
                op->hasAttr(kCpsCandidateStageIndex) ||
                op->hasAttr(kCpsCandidateStageCount) ||
                op->hasAttr(kCpsCandidateRequiresTokenizedDataflow));
}

static bool hasFinalCpsStageAttrs(sde::SdeSuIterateOp op) {
  return op && (op.getCpsGroupIdAttr() || op.getCpsStageIndexAttr() ||
                op.getCpsStageCountAttr());
}

static bool isSdeCpsCandidateStage(sde::SdeSuIterateOp op) {
  if (!op || hasFinalCpsStageAttrs(op))
    return false;
  auto repetition = op.getRepetitionStructure();
  if (!repetition || *repetition != sde::SdeRepetitionStructure::full_timestep)
    return false;
  auto strategy = op.getAsyncStrategy();
  return strategy && *strategy == sde::SdeAsyncStrategy::advance_edt;
}

static void setBarrierReason(sde::SdeSuBarrierOp barrier,
                             sde::SdeBarrierReason reason) {
  barrier.setBarrierReasonAttr(
      sde::SdeBarrierReasonAttr::get(barrier.getContext(), reason));
}

static bool isTimestepBarrier(sde::SdeSuBarrierOp barrier) {
  if (!barrier)
    return false;
  auto reason = barrier.getBarrierReason();
  return reason && *reason == sde::SdeBarrierReason::timestep_stage_boundary;
}

static bool haveSameIterationShape(sde::SdeSuIterateOp lhs,
                                   sde::SdeSuIterateOp rhs) {
  return arts::ValueAnalysis::areValueRangesEquivalent(lhs.getLowerBounds(),
                                                 rhs.getLowerBounds()) &&
         arts::ValueAnalysis::areValueRangesEquivalent(lhs.getUpperBounds(),
                                                 rhs.getUpperBounds()) &&
         arts::ValueAnalysis::areValueRangesEquivalent(lhs.getSteps(),
                                                 rhs.getSteps());
}

static bool haveSameIterationBounds(sde::SdeSuIterateOp lhs,
                                    sde::SdeSuIterateOp rhs) {
  return lhs.getLowerBounds().size() == rhs.getLowerBounds().size() &&
         arts::ValueAnalysis::areValueRangesEquivalent(lhs.getLowerBounds(),
                                                 rhs.getLowerBounds()) &&
         arts::ValueAnalysis::areValueRangesEquivalent(lhs.getUpperBounds(),
                                                 rhs.getUpperBounds());
}

static bool isTiledMultipleOfStep(Value candidate, Value baseStep) {
  if (arts::ValueAnalysis::areValuesEquivalent(candidate, baseStep))
    return true;

  auto mul = candidate.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;

  auto isPositiveMultiplier = [](Value value) {
    if (std::optional<int64_t> folded =
            arts::ValueAnalysis::tryFoldConstantIndex(value))
      return *folded >= 1;
    return arts::ValueAnalysis::isConstantAtLeastOne(value) ||
           arts::ValueAnalysis::isProvablyNonZero(value);
  };

  if (arts::ValueAnalysis::areValuesEquivalent(mul.getLhs(), baseStep))
    return isPositiveMultiplier(mul.getRhs());
  if (arts::ValueAnalysis::areValuesEquivalent(mul.getRhs(), baseStep))
    return isPositiveMultiplier(mul.getLhs());
  return false;
}

static bool haveEquivalentOrTiledSteps(sde::SdeSuIterateOp lhs,
                                       sde::SdeSuIterateOp rhs) {
  if (lhs.getSteps().size() != rhs.getSteps().size())
    return false;

  for (auto [lhsStep, rhsStep] : llvm::zip(lhs.getSteps(), rhs.getSteps())) {
    if (arts::ValueAnalysis::areValuesEquivalent(lhsStep, rhsStep))
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

static Operation *findPreviousSuContainer(Operation *anchor) {
  Block *block = anchor ? anchor->getBlock() : nullptr;
  if (!block)
    return nullptr;
  for (auto it = Block::reverse_iterator(anchor->getIterator());
       it != block->rend(); ++it) {
    if (findSuIterate(&*it))
      return &*it;
    if (!isMemoryEffectFree(&*it))
      break;
  }
  return nullptr;
}

static Operation *findNextSuContainer(Operation *anchor) {
  Block *block = anchor ? anchor->getBlock() : nullptr;
  if (!block)
    return nullptr;
  for (auto it = std::next(anchor->getIterator()); it != block->end(); ++it) {
    if (findSuIterate(&*it))
      return &*it;
    if (!isMemoryEffectFree(&*it))
      break;
  }
  return nullptr;
}

static int64_t findNextCandidateGroupId(ModuleOp module) {
  int64_t nextId = 0;
  module.walk([&](sde::SdeSuIterateOp op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(kCpsCandidateGroupId))
      nextId = std::max(nextId, attr.getInt() + 1);
    if (auto attr = op.getCpsGroupIdAttr())
      nextId = std::max(nextId, attr.getInt() + 1);
  });
  return nextId;
}

static bool canStampCandidatePair(sde::SdeSuIterateOp first,
                                  sde::SdeSuIterateOp second) {
  if (!isSdeCpsCandidateStage(first) || !isSdeCpsCandidateStage(second))
    return false;
  if (hasCandidateAttrs(first) || hasCandidateAttrs(second))
    return false;
  return true;
}

static void stampCandidatePair(sde::SdeSuIterateOp first,
                               sde::SdeSuIterateOp second, int64_t groupId) {
  assert(canStampCandidatePair(first, second) &&
         "candidate pair preconditions must be checked before stamping");

  MLIRContext *ctx = first.getContext();
  auto setStageAttrs = [&](sde::SdeSuIterateOp op, int64_t stageIndex) {
    op->setAttr(kCpsCandidateGroupId,
                IntegerAttr::get(IntegerType::get(ctx, 64), groupId));
    op->setAttr(kCpsCandidateStageIndex,
                IntegerAttr::get(IntegerType::get(ctx, 64), stageIndex));
    op->setAttr(kCpsCandidateStageCount,
                IntegerAttr::get(IntegerType::get(ctx, 64), 2));
    op->setAttr(kCpsCandidateRequiresTokenizedDataflow, UnitAttr::get(ctx));
  };

  setStageAttrs(first, 0);
  setStageAttrs(second, 1);
}

static bool attachStageCompletionToBarrier(Operation *stageContainer,
                                           sde::SdeSuBarrierOp barrier) {
  if (!stageContainer || !barrier)
    return false;
  if (stageContainer->getBlock() != barrier->getBlock())
    return false;

  OpBuilder builder(stageContainer->getContext());
  builder.setInsertionPointAfter(stageContainer);
  auto token = sde::SdeControlTokenOp::create(
      builder, stageContainer->getLoc(),
      sde::CompletionType::get(stageContainer->getContext()));

  SmallVector<Value> tokens(barrier.getTokens().begin(),
                            barrier.getTokens().end());
  tokens.push_back(token.getToken());

  builder.setInsertionPoint(barrier);
  sde::SdeSuBarrierOp::create(builder, barrier.getLoc(), tokens,
                              barrier.getBarrierEliminatedAttr(),
                              barrier.getBarrierReasonAttr());
  barrier.erase();
  return true;
}

static bool insertStageCompletionBarrier(Operation *firstContainer,
                                         Operation *secondContainer) {
  if (!firstContainer || !secondContainer)
    return false;
  if (firstContainer->getBlock() != secondContainer->getBlock())
    return false;
  if (!firstContainer->isBeforeInBlock(secondContainer))
    return false;

  MLIRContext *ctx = firstContainer->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfter(firstContainer);
  auto token = sde::SdeControlTokenOp::create(
      builder, firstContainer->getLoc(), sde::CompletionType::get(ctx));

  SmallVector<Value> tokens{token.getToken()};
  builder.setInsertionPoint(secondContainer);
  sde::SdeSuBarrierOp::create(
      builder, secondContainer->getLoc(), tokens,
      /*barrierEliminated=*/nullptr,
      sde::SdeBarrierReasonAttr::get(
          ctx, sde::SdeBarrierReason::timestep_stage_boundary));
  return true;
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

static bool hasPhysicalTimestepPlan(sde::SdeSuIterateOp op) {
  return op.getPhysicalOwnerDimsAttr() || op.getPhysicalBlockShapeAttr();
}

static bool haveCompatiblePhysicalTimestepPlan(
    sde::SdeSuIterateOp predecessor, sde::SdeSuIterateOp successor) {
  bool predPlanned = hasPhysicalTimestepPlan(predecessor);
  bool succPlanned = hasPhysicalTimestepPlan(successor);
  if (!predPlanned && !succPlanned)
    return true;
  if (predPlanned != succPlanned)
    return false;
  return haveSamePhysicalTimestepPlan(predecessor, successor);
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

static std::optional<SmallVector<int64_t, 4>>
getUniqueStaticWrittenShape(const sde::StructuredMemoryEffectSummary &effects) {
  std::optional<SmallVector<int64_t, 4>> selectedShape;
  for (Value written : effects.writes) {
    Value root = arts::ValueAnalysis::stripMemrefViewOps(written);
    auto memrefType = dyn_cast<MemRefType>(root.getType());
    if (!memrefType || !memrefType.hasStaticShape())
      return std::nullopt;

    SmallVector<int64_t, 4> shape(memrefType.getShape().begin(),
                                  memrefType.getShape().end());
    if (!selectedShape) {
      selectedShape = std::move(shape);
      continue;
    }
    if (*selectedShape != shape)
      return std::nullopt;
  }
  return selectedShape;
}

static bool haveSameStaticWrittenShape(
    const sde::StructuredMemoryEffectSummary &lhs,
    const sde::StructuredMemoryEffectSummary &rhs) {
  auto lhsShape = getUniqueStaticWrittenShape(lhs);
  auto rhsShape = getUniqueStaticWrittenShape(rhs);
  return lhsShape && rhsShape && *lhsShape == *rhsShape;
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

  bool compatibleIterationPlan =
      haveCompatibleTimestepIterationPlan(predecessor, successor);

  if (isUniformRepeatableStage(predecessor) &&
      isUniformRepeatableStage(successor) &&
      compatibleIterationPlan &&
      allowUniformUniformPlan &&
      writesIntersectReads(predEffects, succEffects) &&
      haveCompatiblePhysicalTimestepPlan(predecessor, successor)) {
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

  if (predStencil && succStencil && compatibleIterationPlan &&
      allowStencilStencilPlan) {
    stampRepeatedTimestepPlan(predecessor);
    stampRepeatedTimestepPlan(successor);
    return true;
  }

  if (((predStencil && succUniform) || (predUniform && succStencil)) &&
      (compatibleIterationPlan ||
       haveSameStaticWrittenShape(predEffects, succEffects))) {
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

static unsigned stampBarrierDelimitedCandidates(ModuleOp module,
                                                int64_t &nextGroupId) {
  unsigned stamped = 0;
  SmallVector<sde::SdeSuBarrierOp> barriers;
  module.walk([&](sde::SdeSuBarrierOp barrier) {
    if (isTimestepBarrier(barrier))
      barriers.push_back(barrier);
  });

  for (sde::SdeSuBarrierOp barrier : barriers) {
    Operation *firstContainer = findPreviousSuContainer(barrier);
    Operation *secondContainer = findNextSuContainer(barrier);
    auto first = findSuIterate(firstContainer);
    auto second = findSuIterate(secondContainer);
    if (canStampCandidatePair(first, second) &&
        attachStageCompletionToBarrier(firstContainer, barrier)) {
      stampCandidatePair(first, second, nextGroupId);
      ++stamped;
      ++nextGroupId;
    }
  }
  return stamped;
}

static bool isTransparentBetweenTimestepCandidates(Operation *op) {
  if (!op)
    return false;
  if (isa<sde::SdeYieldOp>(op))
    return true;
  if (auto barrier = dyn_cast<sde::SdeSuBarrierOp>(op))
    return !isTimestepBarrier(barrier);
  return isTimestepInterstitialOp(op);
}

static unsigned stampAdjacentLoopCandidates(scf::ForOp loop,
                                            int64_t &nextGroupId) {
  Operation *previous = nullptr;
  unsigned stamped = 0;

  for (Operation &op : loop.getBody()->without_terminator()) {
    if (findSuIterate(&op)) {
      if (previous) {
        auto first = findSuIterate(previous);
        auto second = findSuIterate(&op);
        if (canStampCandidatePair(first, second) &&
            insertStageCompletionBarrier(previous, &op)) {
          stampCandidatePair(first, second, nextGroupId);
          ++stamped;
          ++nextGroupId;
        }
      }
      previous = &op;
      continue;
    }

    if (isTransparentBetweenTimestepCandidates(&op))
      continue;

    previous = nullptr;
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
    unsigned cpsCandidateGroupsStamped = 0;

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

    int64_t nextGroupId = findNextCandidateGroupId(getOperation());
    cpsCandidateGroupsStamped =
        stampBarrierDelimitedCandidates(getOperation(), nextGroupId);
    getOperation().walk([&](scf::ForOp loop) {
      cpsCandidateGroupsStamped +=
          stampAdjacentLoopCandidates(loop, nextGroupId);
    });

    numCpsCandidateGroups += cpsCandidateGroupsStamped;
    ARTS_INFO("BarrierElimination: eliminated " << eliminated << " barrier(s)");
    ARTS_INFO("BarrierElimination: stamped " << timestepPairsStamped
                                             << " timestep pair(s)");
    ARTS_INFO("BarrierElimination: stamped " << cpsCandidateGroupsStamped
                                             << " CPS candidate group(s)");

  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass>
createBarrierEliminationPass(sde::SDECostModel *costModel) {
  return std::make_unique<BarrierEliminationPass>(costModel);
}

} // namespace mlir::carts::sde
