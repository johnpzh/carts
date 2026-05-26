///==========================================================================///
/// File: BarrierElimination.cpp
///
/// Plan SDE barriers between scheduling units. Independent barriers are marked
/// for elimination. Required timestep barriers are annotated as stage
/// boundaries, and SDE-owned CPS candidates get explicit completion tokens so
/// ARTS can consume the plan without re-discovering the dataflow shape.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::carts::sde {
#define GEN_PASS_DEF_BARRIERELIMINATION
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::sde

#include "carts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "carts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "carts/dialect/sde/Utils/SDECostModel.h"
#include "carts/utils/ArrayAttrUtils.h"
#include "carts/utils/ValueAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/Statistic.h"

#include <cassert>

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(barrier_elimination);

static llvm::Statistic numCpsCandidateGroups{
    "barrier_elimination", "NumCpsCandidateGroups",
    "Number of SDE CPS candidate groups stamped"};

using namespace mlir;
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
  return strategy && *strategy == sde::SdeAsyncStrategy::advance_stage;
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
  return ::mlir::carts::ValueAnalysis::areValueRangesEquivalent(
             lhs.getLowerBounds(), rhs.getLowerBounds()) &&
         ::mlir::carts::ValueAnalysis::areValueRangesEquivalent(
             lhs.getUpperBounds(), rhs.getUpperBounds()) &&
         ::mlir::carts::ValueAnalysis::areValueRangesEquivalent(lhs.getSteps(),
                                                                rhs.getSteps());
}

static bool haveSameIterationBounds(sde::SdeSuIterateOp lhs,
                                    sde::SdeSuIterateOp rhs) {
  return lhs.getLowerBounds().size() == rhs.getLowerBounds().size() &&
         ::mlir::carts::ValueAnalysis::areValueRangesEquivalent(
             lhs.getLowerBounds(), rhs.getLowerBounds()) &&
         ::mlir::carts::ValueAnalysis::areValueRangesEquivalent(
             lhs.getUpperBounds(), rhs.getUpperBounds());
}

static bool isTiledMultipleOfStep(Value candidate, Value baseStep) {
  if (::mlir::carts::ValueAnalysis::areValuesEquivalent(candidate, baseStep))
    return true;

  auto mul = candidate.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;

  auto isPositiveMultiplier = [](Value value) {
    if (std::optional<int64_t> folded =
            ::mlir::carts::ValueAnalysis::tryFoldConstantIndex(value))
      return *folded >= 1;
    return ::mlir::carts::ValueAnalysis::isConstantAtLeastOne(value) ||
           ::mlir::carts::ValueAnalysis::isProvablyNonZero(value);
  };

  if (::mlir::carts::ValueAnalysis::areValuesEquivalent(mul.getLhs(), baseStep))
    return isPositiveMultiplier(mul.getRhs());
  if (::mlir::carts::ValueAnalysis::areValuesEquivalent(mul.getRhs(), baseStep))
    return isPositiveMultiplier(mul.getLhs());
  return false;
}

static bool haveEquivalentOrTiledSteps(sde::SdeSuIterateOp lhs,
                                       sde::SdeSuIterateOp rhs) {
  if (lhs.getSteps().size() != rhs.getSteps().size())
    return false;

  for (auto [lhsStep, rhsStep] : llvm::zip(lhs.getSteps(), rhs.getSteps())) {
    if (::mlir::carts::ValueAnalysis::areValuesEquivalent(lhsStep, rhsStep))
      continue;
    std::optional<int64_t> lhsFolded =
        ::mlir::carts::ValueAnalysis::tryFoldConstantIndex(lhsStep);
    std::optional<int64_t> rhsFolded =
        ::mlir::carts::ValueAnalysis::tryFoldConstantIndex(rhsStep);
    if (lhsFolded && rhsFolded && *lhsFolded == *rhsFolded)
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

static bool
isOutOfPlaceStencilStage(sde::SdeSuIterateOp op,
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
        op.getContext(), sde::SdeAsyncStrategy::advance_stage));
}

static void coarsenRepeatedStencilSlice(sde::SdeSuIterateOp op) {
  if (!op)
    return;
  auto topology = op.getIterationTopology();
  if (!topology || *topology != sde::SdeIterationTopology::owner_tile)
    return;
  ArrayAttr slice = op.getLogicalWorkerSliceAttr();
  if (!slice || slice.size() != 2)
    return;

  SmallVector<Attribute, 2> coarsenedValues(slice.begin(), slice.end());
  auto firstDim = dyn_cast<IntegerAttr>(coarsenedValues.front());
  if (!firstDim || firstDim.getInt() <= 0)
    return;
  coarsenedValues.front() =
      IntegerAttr::get(firstDim.getType(), firstDim.getInt() * 2);
  ArrayAttr coarsened = ArrayAttr::get(op.getContext(), coarsenedValues);
  if (coarsened == slice)
    return;
  op.setLogicalWorkerSliceAttr(coarsened);
  if (ArrayAttr physical = op.getPhysicalBlockShapeAttr()) {
    SmallVector<Attribute, 2> physicalValues(physical.begin(), physical.end());
    if (physicalValues.size() != 2)
      return;
    auto physicalFirstDim = dyn_cast<IntegerAttr>(physicalValues.front());
    if (!physicalFirstDim || physicalFirstDim.getInt() <= 0)
      return;
    physicalValues.front() = IntegerAttr::get(physicalFirstDim.getType(),
                                              physicalFirstDim.getInt() * 2);
    op.setPhysicalBlockShapeAttr(
        ArrayAttr::get(op.getContext(), physicalValues));
  }
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

static bool canUseAdjacentCandidateStage(sde::SdeSuIterateOp op) {
  return isSdeCpsCandidateStage(op) && !hasCandidateAttrs(op);
}

static void stampCandidateGroup(ArrayRef<sde::SdeSuIterateOp> stages,
                                int64_t groupId) {
  assert(stages.size() >= 2 && "candidate groups require multiple stages");
  MLIRContext *ctx = stages.front()->getContext();
  int64_t stageCount = static_cast<int64_t>(stages.size());

  auto setStageAttrs = [&](sde::SdeSuIterateOp op, int64_t stageIndex) {
    op->setAttr(kCpsCandidateGroupId,
                IntegerAttr::get(IntegerType::get(ctx, 64), groupId));
    op->setAttr(kCpsCandidateStageIndex,
                IntegerAttr::get(IntegerType::get(ctx, 64), stageIndex));
    op->setAttr(kCpsCandidateStageCount,
                IntegerAttr::get(IntegerType::get(ctx, 64), stageCount));
    op->setAttr(kCpsCandidateRequiresTokenizedDataflow, UnitAttr::get(ctx));
  };

  for (auto [index, stage] : llvm::enumerate(stages))
    setStageAttrs(stage, static_cast<int64_t>(index));
}

static void stampCandidatePair(sde::SdeSuIterateOp first,
                               sde::SdeSuIterateOp second, int64_t groupId) {
  assert(canStampCandidatePair(first, second) &&
         "candidate pair preconditions must be checked before stamping");
  SmallVector<sde::SdeSuIterateOp, 2> stages{first, second};
  stampCandidateGroup(stages, groupId);
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
  auto token = sde::SdeControlTokenOp::create(builder, firstContainer->getLoc(),
                                              sde::CompletionType::get(ctx));

  SmallVector<Value> tokens{token.getToken()};
  builder.setInsertionPoint(secondContainer);
  sde::SdeSuBarrierOp::create(
      builder, secondContainer->getLoc(), tokens,
      /*barrierEliminated=*/nullptr,
      sde::SdeBarrierReasonAttr::get(
          ctx, sde::SdeBarrierReason::timestep_stage_boundary));
  return true;
}

static bool
writesIntersectReads(const sde::StructuredMemoryEffectSummary &lhs,
                     const sde::StructuredMemoryEffectSummary &rhs) {
  for (Value written : lhs.writes)
    if (rhs.reads.contains(written))
      return true;
  return false;
}

static bool
hasAlternatingBufferExchange(const sde::StructuredMemoryEffectSummary &lhs,
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

static bool haveCompatiblePhysicalTimestepPlan(sde::SdeSuIterateOp predecessor,
                                               sde::SdeSuIterateOp successor) {
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

static bool isPipelineableStructuredClassification(sde::SdeSuIterateOp op) {
  if (!op || !op.getReductionAccumulators().empty())
    return false;
  auto classification = op.getStructuredClassification();
  if (!classification)
    return false;
  switch (*classification) {
  case sde::SdeStructuredClassification::elementwise:
  case sde::SdeStructuredClassification::elementwise_pipeline:
  case sde::SdeStructuredClassification::matmul:
    return true;
  case sde::SdeStructuredClassification::stencil:
  case sde::SdeStructuredClassification::reduction:
    return false;
  }
  return false;
}

static std::optional<unsigned>
getSinglePhysicalOwnerDim(sde::SdeSuIterateOp op) {
  auto ownerDims =
      ::mlir::carts::readI64ArrayAttr(op.getPhysicalOwnerDimsAttr());
  if (!ownerDims || ownerDims->size() != 1 || (*ownerDims)[0] < 0)
    return std::nullopt;
  return static_cast<unsigned>((*ownerDims)[0]);
}

static bool loopIvSelectsOwnerSlice(Value iv, Value ownerIv,
                                    llvm::SmallPtrSetImpl<Value> &seen) {
  if (!iv || !ownerIv || !seen.insert(iv).second)
    return false;
  if (iv == ownerIv || ::mlir::carts::ValueAnalysis::dependsOn(iv, ownerIv))
    return true;

  auto blockArg = dyn_cast<BlockArgument>(iv);
  if (!blockArg)
    return false;

  auto loop = dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!loop || loop.getInductionVar() != iv)
    return false;

  return loopIvSelectsOwnerSlice(loop.getLowerBound(), ownerIv, seen) ||
         loopIvSelectsOwnerSlice(loop.getUpperBound(), ownerIv, seen);
}

static bool loopIvSelectsOwnerSlice(Value iv, Value ownerIv) {
  llvm::SmallPtrSet<Value, 8> seen;
  return loopIvSelectsOwnerSlice(iv, ownerIv, seen);
}

static bool accessMapUsesOwnerSliceAtPhysicalDim(AffineMap map,
                                                 unsigned physicalDim,
                                                 ArrayRef<Value> ivs,
                                                 unsigned ownerLoopDim) {
  if (!map || physicalDim >= map.getNumResults() || ownerLoopDim >= ivs.size())
    return false;
  std::optional<sde::AffineDimOffset> dimOffset =
      sde::extractDimOffset(map.getResult(physicalDim));
  if (!dimOffset || !dimOffset->dim || dimOffset->offset != 0 ||
      *dimOffset->dim >= ivs.size())
    return false;
  return loopIvSelectsOwnerSlice(ivs[*dimOffset->dim], ivs[ownerLoopDim]);
}

static bool
accessEntriesUseOwnerDimForRoot(ArrayRef<sde::MemrefAccessEntry> accesses,
                                ArrayRef<Value> ivs, Value root,
                                unsigned physicalDim, unsigned ownerLoopDim) {
  bool sawRoot = false;
  for (const sde::MemrefAccessEntry &access : accesses) {
    Value accessRoot =
        ::mlir::carts::ValueAnalysis::stripMemrefViewOps(access.memref);
    if (accessRoot != root)
      continue;
    sawRoot = true;
    if (!accessMapUsesOwnerSliceAtPhysicalDim(access.indexingMap, physicalDim,
                                              ivs, ownerLoopDim))
      return false;
  }
  return sawRoot;
}

static bool hasSingleWriteReadIntermediate(
    const sde::StructuredMemoryEffectSummary &predEffects,
    const sde::StructuredMemoryEffectSummary &succEffects, Value &root) {
  if (predEffects.hasUnknownEffects || succEffects.hasUnknownEffects)
    return false;
  if (predEffects.writes.size() != 1)
    return false;

  Value candidate = *predEffects.writes.begin();
  Value rootCandidate =
      ::mlir::carts::ValueAnalysis::stripMemrefViewOps(candidate);
  if (!rootCandidate || !rootCandidate.getDefiningOp<sde::SdeMuAllocOp>())
    return false;
  if (!succEffects.reads.contains(candidate))
    return false;
  if (succEffects.writes.contains(candidate))
    return false;

  for (Value written : succEffects.writes) {
    if (predEffects.reads.contains(written) ||
        predEffects.writes.contains(written))
      return false;
  }

  root = candidate;
  return true;
}

static bool canPipelineThroughTokenLocalMemoryDeps(
    sde::SdeSuIterateOp predecessor, sde::SdeSuIterateOp successor,
    const sde::StructuredMemoryEffectSummary &predEffects,
    const sde::StructuredMemoryEffectSummary &succEffects) {
  if (!isPipelineableStructuredClassification(predecessor) ||
      !isPipelineableStructuredClassification(successor))
    return false;
  if (!haveSdeApprovedTiledTimestepPlan(predecessor, successor))
    return false;

  auto topology = predecessor.getIterationTopology();
  if (!topology || *topology != sde::SdeIterationTopology::owner_strip ||
      !sameSdeIterationTopology(predecessor, successor))
    return false;

  std::optional<unsigned> predOwnerDim = getSinglePhysicalOwnerDim(predecessor);
  std::optional<unsigned> succOwnerDim = getSinglePhysicalOwnerDim(successor);
  if (!predOwnerDim || !succOwnerDim || *predOwnerDim != *succOwnerDim)
    return false;

  Value intermediate;
  if (!hasSingleWriteReadIntermediate(predEffects, succEffects, intermediate))
    return false;

  auto predSummary = sde::analyzeStructuredLoop(predecessor);
  auto succSummary = sde::analyzeStructuredLoop(successor);
  if (!predSummary || !succSummary)
    return false;
  if (predSummary->nest.ivs.empty() || succSummary->nest.ivs.empty())
    return false;

  constexpr unsigned ownerLoopDim = 0;
  return accessEntriesUseOwnerDimForRoot(predSummary->writes,
                                         predSummary->nest.ivs, intermediate,
                                         *predOwnerDim, ownerLoopDim) &&
         accessEntriesUseOwnerDimForRoot(succSummary->reads,
                                         succSummary->nest.ivs, intermediate,
                                         *succOwnerDim, ownerLoopDim);
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
    Value root = ::mlir::carts::ValueAnalysis::stripMemrefViewOps(written);
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

static bool
haveSameStaticWrittenShape(const sde::StructuredMemoryEffectSummary &lhs,
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
        predecessor.getContext(), sde::SdePattern::jacobi_alternating_buffers));
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
      isUniformRepeatableStage(successor) && compatibleIterationPlan &&
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
    coarsenRepeatedStencilSlice(predecessor);
    coarsenRepeatedStencilSlice(successor);
    return true;
  }

  if (((predStencil && succUniform) || (predUniform && succStencil)) &&
      (compatibleIterationPlan ||
       haveSameStaticWrittenShape(predEffects, succEffects))) {
    stampJacobiTimestepPlan(predecessor, successor, predStencil, succStencil);
    coarsenRepeatedStencilSlice(predecessor);
    coarsenRepeatedStencilSlice(successor);
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
  auto succEffects =
      sde::collectStructuredMemoryEffects(successor.getOperation());
  return stampTimestepPlanIfRecognized(predecessor, successor, predEffects,
                                       succEffects,
                                       /*allowStencilStencilPlan=*/true,
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
  unsigned stamped = 0;
  SmallVector<Operation *, 4> candidateContainers;

  auto flushCandidateChain = [&]() {
    if (candidateContainers.size() < 2) {
      candidateContainers.clear();
      return;
    }

    SmallVector<sde::SdeSuIterateOp, 4> stages;
    stages.reserve(candidateContainers.size());
    for (Operation *container : candidateContainers) {
      auto stage = findSuIterate(container);
      if (!canUseAdjacentCandidateStage(stage)) {
        candidateContainers.clear();
        return;
      }
      stages.push_back(stage);
    }

    for (auto index : llvm::seq<size_t>(1, candidateContainers.size()))
      if (!insertStageCompletionBarrier(candidateContainers[index - 1],
                                        candidateContainers[index])) {
        candidateContainers.clear();
        return;
      }

    stampCandidateGroup(stages, nextGroupId);
    ++stamped;
    ++nextGroupId;
    candidateContainers.clear();
  };

  for (Operation &op : loop.getBody()->without_terminator()) {
    if (findSuIterate(&op)) {
      auto stage = findSuIterate(&op);
      if (!canUseAdjacentCandidateStage(stage))
        flushCandidateChain();
      if (canUseAdjacentCandidateStage(stage))
        candidateContainers.push_back(&op);
      continue;
    }

    if (isTransparentBetweenTimestepCandidates(&op))
      continue;

    flushCandidateChain();
  }

  flushCandidateChain();
  return stamped;
}

struct BarrierEliminationPass
    : public sde::impl::BarrierEliminationBase<BarrierEliminationPass> {
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

      if (canPipelineThroughTokenLocalMemoryDeps(predecessor, successor,
                                                 predEffects, succEffects)) {
        barrier.setBarrierEliminatedAttr(UnitAttr::get(barrier.getContext()));
        setBarrierReason(barrier, sde::SdeBarrierReason::required_memory);
        eliminated++;
        ARTS_DEBUG("Eliminated required-memory barrier: token-local DB "
                   "dependencies preserve the inter-phase order");
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
