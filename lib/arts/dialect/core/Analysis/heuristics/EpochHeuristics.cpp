///==========================================================================///
/// File: EpochHeuristics.cpp
///
/// Shared epoch scheduling/structural policy for continuation, CPS, and epoch
/// fusion. Passes consume the decisions from this helper instead of embedding
/// contract- or pattern-specific policy directly in rewrite code.
///==========================================================================///

#include "arts/dialect/core/Analysis/heuristics/EpochHeuristics.h"
#include "arts/dialect/core/Analysis/KernelFamily.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/LoopUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::arts;

namespace {

using AttrNames::Operation::ContinuationForEpoch;

static bool isInsideLoop(Operation *op) {
  Operation *parent = op ? op->getParentOp() : nullptr;
  while (parent) {
    if (isa<arts::ForOp, scf::ForOp, scf::WhileOp, scf::ParallelOp,
            affine::AffineForOp>(parent))
      return true;
    parent = parent->getParentOp();
  }
  return false;
}

static bool isPureArith(Operation *op) {
  return isSideEffectFreeArithmeticLikeOp(op);
}

static bool isCPSSequentialOp(Operation *op) {
  if (!op)
    return false;
  if (isa<scf::ForOp>(op))
    return true;
  if (isa<func::CallOp>(op))
    return true;
  if (isa<memref::LoadOp, memref::StoreOp, memref::AllocaOp, memref::AllocOp,
          memref::DeallocOp, memref::CastOp, memref::SubViewOp>(op))
    return true;
  if (isa<math::AbsFOp, math::ExpOp, math::Exp2Op, math::SqrtOp, math::TanhOp,
          math::LogOp, math::Log2Op, math::PowFOp, math::CeilOp, math::FloorOp,
          math::CosOp, math::SinOp, math::FmaOp, math::RoundOp>(op))
    return true;
  return isa<LoweringContractOp>(op);
}

static bool bodyContainsEpoch(Block &body) {
  for (Operation &op : body.without_terminator()) {
    if (isa<EpochOp>(op))
      return true;
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool found = false;
      ifOp.walk([&](EpochOp) { found = true; });
      if (found)
        return true;
    }
  }
  return false;
}

static bool bodyIsEpochOnly(Block &body) {
  for (Operation &op : body.without_terminator()) {
    if (isa<EpochOp>(op))
      continue;
    if (isa<scf::IfOp>(op))
      continue;
    if (isPureArith(&op))
      continue;
    if (isa<DbAcquireOp, DbReleaseOp, LoweringContractOp>(op))
      continue;
    return false;
  }
  return true;
}

static void collectEpochSlots(Block &body, SmallVectorImpl<EpochSlot> &slots) {
  for (Operation &op : body.without_terminator()) {
    if (auto epoch = dyn_cast<EpochOp>(op)) {
      slots.push_back({epoch, nullptr});
      continue;
    }
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp)
      continue;
    for (Operation &thenOp : ifOp.getThenRegion().front()) {
      if (auto epoch = dyn_cast<EpochOp>(thenOp))
        slots.push_back({epoch, ifOp});
    }
  }
}

static bool epochBodyIsClonable(EpochOp epoch) {
  return epoch && !epoch.getRegion().empty() &&
         !epoch.getRegion().front().without_terminator().empty();
}

static bool epochHasZeroExpectedLocalWork(EpochOp epoch) {
  if (!epoch)
    return false;
  auto expectedLocalWork = epoch->getAttrOfType<IntegerAttr>(
      AttrNames::Operation::Plan::CostExpectedLocalWork);
  return expectedLocalWork && expectedLocalWork.getInt() == 0;
}

static bool epochHasRepetitionStructure(EpochOp epoch,
                                        RepetitionStructure repetition) {
  if (!epoch)
    return false;
  auto repetitionAttr = epoch->getAttrOfType<StringAttr>(
      AttrNames::Operation::Plan::RepetitionStructure);
  return repetitionAttr &&
         repetitionAttr.getValue() == repetitionStructureToString(repetition);
}

static bool epochCreatesCoordinatorWave(EpochOp epoch) {
  if (!epoch)
    return false;

  unsigned nestedEdtCount = 0;
  bool hasNestedEpoch = false;
  epoch.walk([&](Operation *op) {
    if (op == epoch.getOperation())
      return WalkResult::advance();
    if (isa<EpochOp>(op)) {
      hasNestedEpoch = true;
      return WalkResult::interrupt();
    }
    if (isa<EdtOp>(op)) {
      ++nestedEdtCount;
      if (nestedEdtCount > 1)
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasNestedEpoch || nestedEdtCount > 1;
}

template <typename PredicateT>
static bool slotMatches(EpochSlot slot, PredicateT &&predicate) {
  if (predicate(slot.epoch))
    return true;
  if (!slot.wrappingIf)
    return false;

  bool sawEpoch = false;
  bool allMatch = true;
  slot.wrappingIf->walk([&](EpochOp nestedEpoch) {
    sawEpoch = true;
    if (predicate(nestedEpoch))
      return WalkResult::advance();
    allMatch = false;
    return WalkResult::interrupt();
  });
  return sawEpoch && allMatch;
}

static bool loopHasScalarCarryState(scf::ForOp forOp) {
  llvm::SetVector<Value> capturedValues;
  getUsedValuesDefinedAbove(forOp.getRegion(), capturedValues);
  for (Value captured : capturedValues) {
    auto dbAlloc =
        dyn_cast_or_null<DbAllocOp>(DbUtils::getUnderlyingDbAlloc(captured));
    if (dbAlloc)
      continue;
    if (auto *defOp = captured.getDefiningOp()) {
      if (isa<arith::ConstantOp, arith::ConstantIndexOp, LLVM::AddressOfOp>(
              defOp))
        continue;
    }
    if (isa<IndexType, IntegerType, FloatType>(captured.getType()))
      return true;
  }
  return false;
}

using OwnerDimSignature = SmallVector<int64_t, 4>;

static void
appendUniqueOwnerDimSignature(SmallVectorImpl<OwnerDimSignature> &signatures,
                              ArrayRef<int64_t> ownerDims) {
  OwnerDimSignature candidate(ownerDims.begin(), ownerDims.end());
  if (llvm::any_of(signatures, [&](const OwnerDimSignature &existing) {
        return existing == candidate;
      }))
    return;
  signatures.push_back(std::move(candidate));
}

static void collectOwnerDimSignatures(EpochOp epoch,
                                      SmallVectorImpl<OwnerDimSignature> &out) {
  if (!epoch)
    return;

  epoch.walk([&](Operation *op) {
    if (auto ownerDims = getStencilOwnerDims(op))
      appendUniqueOwnerDimSignature(out, *ownerDims);
    if (auto contract = dyn_cast<LoweringContractOp>(op))
      if (auto ownerDims = contract.getOwnerDims())
        appendUniqueOwnerDimSignature(out, *ownerDims);
  });

  llvm::sort(out,
             [](const OwnerDimSignature &lhs, const OwnerDimSignature &rhs) {
               return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                                   rhs.begin(), rhs.end());
             });
}

static bool loopHasHeterogeneousOwnerDims(ArrayRef<EpochSlot> slots) {
  for (EpochSlot slot : slots) {
    if (slotMatches(slot, [](EpochOp epoch) {
          SmallVector<OwnerDimSignature> signatures;
          collectOwnerDimSignatures(epoch, signatures);
          return signatures.size() > 1;
        }))
      return true;
  }
  return false;
}

static bool shouldAvoidCpsChainForLoop(scf::ForOp forOp,
                                       const EpochAsyncLoopDecision &decision) {
  if (decision.slots.empty())
    return false;

  bool allSlotsHaveZeroExpectedLocalWork =
      llvm::all_of(decision.slots, [](const EpochSlot &slot) {
        return slotMatches(slot, epochHasZeroExpectedLocalWork);
      });

  bool coordinatorWaveLoop =
      llvm::any_of(decision.slots, [](const EpochSlot &slot) {
        return slotMatches(slot, epochCreatesCoordinatorWave);
      });
  bool hasScalarCarryState = loopHasScalarCarryState(forOp);
  bool pureOrchestrationLoop =
      !hasScalarCarryState && coordinatorWaveLoop &&
      (decision.slots.size() > 1 || decision.hasSequentialSidecars);
  bool twoSlotCoordinatorLoopWithSidecars =
      decision.slots.size() == 2 && decision.hasInterEpochSidecars &&
      coordinatorWaveLoop && !hasScalarCarryState;

  bool allSlotsFullTimestep =
      llvm::all_of(decision.slots, [](const EpochSlot &slot) {
        return slotMatches(slot, [](EpochOp epoch) {
          return epochHasRepetitionStructure(epoch,
                                             RepetitionStructure::FullTimestep);
        });
      });
  bool multiStageOrSidecarFullTimestepLoop =
      allSlotsFullTimestep &&
      (decision.slots.size() > 1 || decision.hasSequentialSidecars);
  bool hasHeterogeneousOwnerDimLayout =
      allSlotsFullTimestep &&
      (decision.slots.size() > 1 || decision.hasSequentialSidecars) &&
      loopHasHeterogeneousOwnerDims(decision.slots);

  if (twoSlotCoordinatorLoopWithSidecars)
    return true;

  // STREAM fix: If the loop has inter-epoch sequential sidecars (e.g., timing
  // between kernels), allow CPS chain even for orchestration-heavy loops.
  // The sidecars represent per-iteration state that benefits from CPS.
  if (decision.hasInterEpochSidecars && !hasHeterogeneousOwnerDimLayout)
    return false; // Don't avoid CPS chain

  // Heterogeneous owner dims only block CPS chain when inter-epoch sidecars
  // are present (handled above). Tail-only sidecars with heterogeneous dims
  // are safe because each epoch slot is lowered independently within the
  // continuation — no partition state crosses between slots.
  if (hasHeterogeneousOwnerDimLayout && decision.hasInterEpochSidecars)
    return true;

  return (allSlotsHaveZeroExpectedLocalWork &&
          multiStageOrSidecarFullTimestepLoop) ||
         (allSlotsHaveZeroExpectedLocalWork && pureOrchestrationLoop);
}

static void recordAccess(EpochAccessSummary &summary, Value value,
                         EpochAccessMode mode) {
  summary.record(DbUtils::getUnderlyingDbAlloc(value), mode);
}

static void recordAcquire(EpochAccessSummary &summary, Value value) {
  if (Operation *alloc = DbUtils::getUnderlyingDbAlloc(value))
    summary.acquireAllocs.insert(alloc);
}

static bool canFuseSummaries(const EpochAccessSummary &a,
                             const EpochAccessSummary &b) {
  for (const auto &entry : a.allocModes) {
    EpochAccessMode otherMode = b.lookup(entry.first);
    if (otherMode == EpochAccessMode::None)
      continue;
    if (epochAccessModeHasWrite(entry.second) ||
        epochAccessModeHasWrite(otherMode))
      return false;
  }
  return true;
}

static unsigned countSharedAcquireAllocs(const EpochAccessSummary &a,
                                         const EpochAccessSummary &b) {
  unsigned shared = 0;
  for (Operation *alloc : a.acquireAllocs)
    if (b.acquireAllocs.contains(alloc))
      ++shared;
  return shared;
}

static unsigned countTailWorkUnits(ArrayRef<Operation *> tailOps) {
  unsigned units = 0;
  for (Operation *op : tailOps) {
    if (isa<DbReleaseOp, LoweringContractOp>(op))
      continue;
    ++units;
  }
  return units;
}

static void
collectCapturedTailDbAcquireValues(ArrayRef<Operation *> tailOps,
                                   SmallVectorImpl<Value> &capturedValues) {
  DenseSet<Operation *> tailOpSet(tailOps.begin(), tailOps.end());
  DenseSet<Value> seenValues;
  for (Operation *op : tailOps) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || tailOpSet.contains(defOp))
        continue;
      if (!isa<DbAcquireOp>(defOp))
        continue;
      if (seenValues.insert(operand).second)
        capturedValues.push_back(operand);
    }
  }
}

} // namespace

EpochAccessSummary EpochHeuristics::summarizeEpochAccess(EpochOp epoch) {
  EpochAccessSummary summary;
  if (!epoch)
    return summary;

  epoch.walk([&](Operation *op) {
    if (auto acquire = dyn_cast<DbAcquireOp>(op)) {
      recordAcquire(summary, acquire.getSourcePtr());
      return WalkResult::advance();
    }
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      recordAccess(summary, load.getMemRef(), EpochAccessMode::Read);
      return WalkResult::advance();
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      recordAccess(summary, store.getMemRef(), EpochAccessMode::Write);
      return WalkResult::advance();
    }
    if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      recordAccess(summary, load.getMemRef(), EpochAccessMode::Read);
      return WalkResult::advance();
    }
    if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      recordAccess(summary, store.getMemRef(), EpochAccessMode::Write);
      return WalkResult::advance();
    }
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      recordAccess(summary, copy.getSource(), EpochAccessMode::Read);
      recordAccess(summary, copy.getTarget(), EpochAccessMode::Write);
      return WalkResult::advance();
    }
    if (auto dim = dyn_cast<memref::DimOp>(op)) {
      recordAccess(summary, dim.getSource(), EpochAccessMode::Read);
      return WalkResult::advance();
    }

    if (op->hasTrait<OpTrait::IsTerminator>() || op->getNumRegions() != 0)
      return WalkResult::advance();

    if (isa<DbAcquireOp, DbReleaseOp, DbAllocOp, DbFreeOp, DbRefOp,
            LoweringContractOp, EdtOp, memref::CastOp, memref::SubViewOp,
            memref::ReinterpretCastOp>(op))
      return WalkResult::advance();

    for (Value operand : op->getOperands()) {
      if (DbUtils::getUnderlyingDbAlloc(operand))
        recordAccess(summary, operand, EpochAccessMode::ReadWrite);
    }
    return WalkResult::advance();
  });

  return summary;
}

EpochFusionDecision
EpochHeuristics::evaluateEpochFusion(EpochOp first, EpochOp second,
                                     bool continuationEnabled,
                                     const EpochAccessSummary *firstSummary,
                                     const EpochAccessSummary *secondSummary) {
  EpochFusionDecision decision;
  if (!first || !second) {
    decision.rationale = "missing epoch";
    return decision;
  }
  if (first->hasAttr(ContinuationForEpoch) ||
      second->hasAttr(ContinuationForEpoch)) {
    decision.rationale = "continuation-managed epoch must keep its boundary";
    return decision;
  }
  if (!second.getEpochGuid().use_empty()) {
    decision.rationale = "second epoch GUID is used externally";
    return decision;
  }

  EpochAccessSummary ownedFirst;
  EpochAccessSummary ownedSecond;
  if (!firstSummary) {
    ownedFirst = summarizeEpochAccess(first);
    firstSummary = &ownedFirst;
  }
  if (!secondSummary) {
    ownedSecond = summarizeEpochAccess(second);
    secondSummary = &ownedSecond;
  }

  if (!canFuseSummaries(*firstSummary, *secondSummary)) {
    decision.rationale = "effective accesses conflict";
    return decision;
  }

  if (continuationEnabled) {
    EpochContinuationDecision continuation = evaluateContinuation(second);
    if (continuation.eligible) {
      unsigned sharedAcquireAllocs =
          countSharedAcquireAllocs(*firstSummary, *secondSummary);
      unsigned continuationTailPressure =
          continuation.tailWorkUnits +
          continuation.capturedDbAcquireValues.size();
      if (sharedAcquireAllocs < continuationTailPressure) {
        decision.rationale =
            "continuation tail outweighs redundant acquire collapse";
        return decision;
      }
    }
  }

  decision.shouldFuse = true;
  decision.rationale = "eligible";
  return decision;
}

EpochContinuationDecision
EpochHeuristics::evaluateContinuation(EpochOp epoch, EpochOp previousEpoch,
                                      bool continuationEnabled,
                                      const EpochAccessSummary *previousSummary,
                                      const EpochAccessSummary *epochSummary) {
  EpochContinuationDecision decision;
  if (!epoch) {
    decision.rationale = "missing epoch";
    return decision;
  }

  Block *block = epoch->getBlock();
  if (!block) {
    decision.rationale = "epoch is not attached to a block";
    return decision;
  }
  if (isInsideLoop(epoch)) {
    decision.rationale = "loop-contained epochs must use CPS scheduling";
    return decision;
  }
  if (epoch->hasAttr(ContinuationForEpoch)) {
    decision.rationale = "epoch is already continuation-managed";
    return decision;
  }
  if (epoch.getRegion().empty() ||
      epoch.getRegion().front().without_terminator().empty()) {
    decision.rationale = "epoch body is empty";
    return decision;
  }
  if (previousEpoch) {
    EpochFusionDecision fusionDecision =
        evaluateEpochFusion(previousEpoch, epoch, continuationEnabled,
                            previousSummary, epochSummary);
    if (fusionDecision.shouldFuse) {
      decision.rationale = "previous epoch fusion is preferred";
      return decision;
    }
  }

  bool afterEpoch = false;
  for (Operation &op : *block) {
    if (&op == epoch.getOperation()) {
      afterEpoch = true;
      continue;
    }
    if (afterEpoch && !op.hasTrait<OpTrait::IsTerminator>())
      decision.tailOps.push_back(&op);
  }
  if (decision.tailOps.empty()) {
    decision.rationale = "epoch has no tail to outline";
    return decision;
  }
  decision.tailWorkUnits = countTailWorkUnits(decision.tailOps);
  collectCapturedTailDbAcquireValues(decision.tailOps,
                                     decision.capturedDbAcquireValues);

  llvm::DenseSet<Operation *> tailOpSet(decision.tailOps.begin(),
                                        decision.tailOps.end());
  for (Operation *op : decision.tailOps) {
    if (isa<EpochOp, CreateEpochOp>(op)) {
      decision.rationale = "tail contains a nested epoch";
      return decision;
    }

    bool hasNestedEpoch = false;
    op->walk([&](Operation *inner) {
      if (isa<EpochOp, CreateEpochOp>(inner))
        hasNestedEpoch = true;
    });
    if (hasNestedEpoch) {
      decision.rationale = "tail contains nested epoch-like operations";
      return decision;
    }

    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) {
        decision.rationale = "tail captures a block argument";
        return decision;
      }
      if (tailOpSet.contains(defOp))
        continue;
      if (!isa<DbAcquireOp>(defOp)) {
        decision.rationale = "tail captures non-DB external state";
        return decision;
      }
    }
  }

  if (Operation *terminator = block->getTerminator()) {
    for (Value operand : terminator->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp();
          defOp && tailOpSet.contains(defOp)) {
        decision.rationale =
            "block terminator depends on values defined in the tail";
        return decision;
      }
    }
  }

  decision.eligible = true;
  decision.rationale = "eligible";
  return decision;
}

EpochLoopDriverDecision
EpochHeuristics::evaluateCPSLoopDriver(scf::ForOp forOp) {
  EpochLoopDriverDecision decision;
  if (!forOp) {
    decision.rationale = "missing loop";
    return decision;
  }
  if (forOp.getNumResults() > 0 || !forOp.getInitArgs().empty()) {
    decision.rationale = "loop has iter_args";
    return decision;
  }
  if (isInsideLoop(forOp)) {
    decision.rationale = "loop is nested";
    return decision;
  }

  Block &body = *forOp.getBody();
  if (!bodyContainsEpoch(body)) {
    decision.rationale = "loop body has no epochs";
    return decision;
  }
  if (!bodyIsEpochOnly(body)) {
    decision.rationale = "loop body has non-epoch side effects";
    return decision;
  }

  if (auto tripCount = arts::getStaticTripCount(forOp.getOperation());
      tripCount && *tripCount < 2) {
    decision.rationale = "trip count is less than 2";
    return decision;
  }

  bool alreadyMarked = false;
  forOp.walk([&](EpochOp epoch) {
    if (epoch->hasAttr(ContinuationForEpoch))
      alreadyMarked = true;
  });
  if (alreadyMarked) {
    decision.rationale = "loop already contains continuation-managed epochs";
    return decision;
  }

  decision.eligible = true;
  decision.rationale = "eligible";
  return decision;
}

EpochChainDecision EpochHeuristics::evaluateCPSChain(scf::ForOp forOp) {
  EpochChainDecision decision;
  if (!forOp) {
    decision.rationale = "missing loop";
    return decision;
  }
  if (forOp.getNumResults() > 0 || !forOp.getInitArgs().empty()) {
    decision.rationale = "loop has iter_args";
    return decision;
  }
  if (isInsideLoop(forOp)) {
    decision.rationale = "loop is nested";
    return decision;
  }

  if (auto tripCount = arts::getStaticTripCount(forOp.getOperation());
      tripCount && *tripCount < 2) {
    decision.rationale = "trip count is less than 2";
    return decision;
  }

  Block &body = *forOp.getBody();
  collectEpochSlots(body, decision.slots);
  if (decision.slots.empty()) {
    decision.rationale = "loop body has no epoch slots";
    return decision;
  }

  for (EpochSlot &slot : decision.slots) {
    if (slot.epoch->hasAttr(ContinuationForEpoch)) {
      decision.rationale = "loop already contains continuation-managed epochs";
      return decision;
    }
    if (!epochBodyIsClonable(slot.epoch)) {
      decision.rationale = "epoch body is empty or unclonable";
      return decision;
    }
  }

  for (Operation &op : body.without_terminator()) {
    bool isSlot = false;
    for (EpochSlot &slot : decision.slots) {
      if (&op == slot.epoch.getOperation() ||
          (slot.wrappingIf && &op == slot.wrappingIf.getOperation())) {
        isSlot = true;
        break;
      }
    }
    if (isSlot)
      continue;
    if (isPureArith(&op))
      continue;
    if (isa<DbAcquireOp, DbReleaseOp>(op))
      continue;
    if (isCPSSequentialOp(&op)) {
      decision.sequentialOps.push_back(&op);
      continue;
    }
    decision.rationale = "loop body contains unsupported side effects";
    return decision;
  }

  if (auto tripCount = arts::getStaticTripCount(forOp.getOperation())) {
    int64_t repeatedEpochSections =
        *tripCount * static_cast<int64_t>(decision.slots.size());
    int64_t sequentialBookkeeping =
        static_cast<int64_t>(decision.sequentialOps.size());
    if (repeatedEpochSections <= sequentialBookkeeping) {
      decision.rationale =
          "sequential bookkeeping dominates repeated epoch sections";
      return decision;
    }
  }

  decision.eligible = true;
  decision.rationale = "eligible";
  return decision;
}

EpochAsyncLoopDecision
EpochHeuristics::evaluateAsyncLoopStrategy(scf::ForOp forOp) {
  EpochAsyncLoopDecision decision;
  if (!forOp) {
    decision.rationale = "missing loop";
    return decision;
  }
  if (forOp.getNumResults() > 0 || !forOp.getInitArgs().empty()) {
    decision.rationale = "loop has iter_args";
    return decision;
  }
  if (isInsideLoop(forOp)) {
    decision.rationale = "loop is nested";
    return decision;
  }

  if (auto tripCount = arts::getStaticTripCount(forOp.getOperation());
      tripCount && *tripCount < 2) {
    decision.rationale = "trip count is less than 2";
    return decision;
  }

  Block &body = *forOp.getBody();
  collectEpochSlots(body, decision.slots);
  if (decision.slots.empty()) {
    decision.rationale = "loop body has no epoch slots";
    return decision;
  }

  for (EpochSlot &slot : decision.slots) {
    if (slot.epoch->hasAttr(ContinuationForEpoch)) {
      decision.rationale = "loop already contains continuation-managed epochs";
      return decision;
    }
    if (!epochBodyIsClonable(slot.epoch)) {
      decision.rationale = "epoch body is empty or unclonable";
      return decision;
    }
    if (slot.wrappingIf)
      decision.hasIvConditionedEpochSlots = true;
  }

  decision.bodyIsEpochOnly = bodyIsEpochOnly(body);

  DenseSet<Operation *> sequentialSet;
  for (Operation &op : body.without_terminator()) {
    bool isSlot = false;
    for (EpochSlot &slot : decision.slots) {
      if (&op == slot.epoch.getOperation() ||
          (slot.wrappingIf && &op == slot.wrappingIf.getOperation())) {
        isSlot = true;
        break;
      }
    }
    if (isSlot)
      continue;
    if (isPureArith(&op))
      continue;
    if (isa<DbAcquireOp, DbReleaseOp, LoweringContractOp>(op))
      continue;
    if (isCPSSequentialOp(&op)) {
      if (isa<func::CallOp>(op))
        decision.hasCallSidecars = true;
      decision.hasSequentialSidecars = true;
      decision.sequentialOps.push_back(&op);
      sequentialSet.insert(&op);
      continue;
    }
    decision.hasUnsupportedSideEffects = true;
    decision.rationale = "loop body contains unsupported side effects";
    return decision;
  }

  if (!decision.sequentialOps.empty()) {
    unsigned currentSlot = 0;
    for (Operation &op : body.without_terminator()) {
      if (sequentialSet.contains(&op)) {
        if (currentSlot < decision.slots.size() - 1)
          decision.hasInterEpochSidecars = true;
        else
          decision.hasTailSidecars = true;
      }

      for (unsigned slotIdx = currentSlot; slotIdx < decision.slots.size();
           ++slotIdx) {
        Operation *slotOp =
            decision.slots[slotIdx].wrappingIf
                ? decision.slots[slotIdx].wrappingIf.getOperation()
                : decision.slots[slotIdx].epoch.getOperation();
        if (&op == slotOp) {
          currentSlot = slotIdx + 1;
          break;
        }
      }
    }
  }

  /// Host-side call sidecars usually indicate benchmark/timing control loops
  /// or other external coordination that should stay on the blocking path.
  /// Rewriting those loops into CPS chains relaunches scheduler topology
  /// around non-kernel work and has regressed uniform large-thread kernels.
  if (decision.hasCallSidecars) {
    decision.rationale = "loop has host call sidecars";
    return decision;
  }

  decision.eligible = true;
  if (shouldAvoidCpsChainForLoop(forOp, decision)) {
    decision.rationale =
        "loop is orchestration-heavy and has no profitable CPS-chain-local "
        "work";
    return decision;
  }

  if (decision.hasSequentialSidecars) {
    decision.strategy = EpochAsyncLoopStrategy::CpsChain;
    if (decision.hasInterEpochSidecars)
      decision.rationale = "loop has sequential sidecars between epoch slots";
    else if (decision.hasTailSidecars)
      decision.rationale =
          "loop has sequential tail state that needs inner continuation";
    else
      decision.rationale = "loop has sequential sidecars";
    return decision;
  }

  if (decision.bodyIsEpochOnly) {
    if (decision.slots.size() == 1) {
      decision.strategy = EpochAsyncLoopStrategy::AdvanceEdt;
      if (decision.hasIvConditionedEpochSlots) {
        decision.rationale =
            "single-slot epoch-only loop with IV-conditioned slot prefers "
            "advance EDT";
      } else {
        decision.rationale = "single-slot epoch-only loop prefers advance EDT";
      }
    } else {
      decision.strategy = EpochAsyncLoopStrategy::CpsChain;
      decision.rationale =
          "multi-slot epoch-only loop needs CPS chain for intra-iteration "
          "continuations";
    }
    return decision;
  }

  decision.strategy = EpochAsyncLoopStrategy::CpsChain;
  decision.rationale = "loop needs CPS chain";
  return decision;
}
