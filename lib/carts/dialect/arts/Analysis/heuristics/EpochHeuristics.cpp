///==========================================================================///
/// File: EpochHeuristics.cpp
///
/// Shared epoch structural policy for epoch fusion.
/// Passes consume the decisions from this helper instead of embedding
/// contract- or pattern-specific policy directly in rewrite code.
///==========================================================================///

#include "carts/dialect/arts/Analysis/heuristics/EpochHeuristics.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

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
                                     const EpochAccessSummary *firstSummary,
                                     const EpochAccessSummary *secondSummary) {
  EpochFusionDecision decision;
  if (!first || !second) {
    decision.rationale = "missing epoch";
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

  decision.shouldFuse = true;
  decision.rationale = "eligible";
  return decision;
}
