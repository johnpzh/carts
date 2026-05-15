///==========================================================================///
/// File: EpochHeuristics.cpp
///
/// Shared epoch scheduling/structural policy for continuation and epoch fusion.
/// Passes consume the decisions from this helper instead of embedding
/// contract- or pattern-specific policy directly in rewrite code.
///==========================================================================///

#include "carts/dialect/arts/Analysis/heuristics/EpochHeuristics.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/utils/LoopUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/StencilAttributes.h"
#include "carts/utils/Utils.h"
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
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

using AttrNames::Operation::ContinuationForEpoch;

static bool isInsideLoop(Operation *op) {
  Operation *parent = op ? op->getParentOp() : nullptr;
  while (parent) {
    if (isa<scf::ForOp, scf::WhileOp, scf::ParallelOp,
            affine::AffineForOp>(parent))
      return true;
    parent = parent->getParentOp();
  }
  return false;
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
    decision.rationale =
        "loop-contained epochs are not eligible for finish continuation";
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
