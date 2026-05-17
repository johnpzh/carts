///==========================================================================///
/// File: Hoisting.cpp
///
/// Unified hoisting pass for ARTS operations that performs:
/// 1. Epoch acquire hoisting - hoists read-only db_acquire out of worker loops
/// 2. DbRef hoisting - hoists loop-invariant db_ref out of inner loops in EDTs
///
/// Example:
///   Before:
///     scf.for %w = ... {
///       %acq = arts.db_acquire[<in>] ...
///       scf.for %i = ... {
///         %r = arts.db_ref %acq[%i]
///       }
///     }
///
///   After:
///     %acq = arts.db_acquire[<in>] ...   // hoisted
///     scf.for %w = ... {
///       %r = arts.db_ref %acq[%i]        // inner-invariant refs hoisted
///     }
///==========================================================================///

#define GEN_PASS_DEF_HOISTING
#include "carts/Dialect.h"
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "carts/dialect/arts/Utils/LoopInvarianceUtils.h"
#include "carts/dialect/arts/Utils/LoopStructureUtils.h"
#include "carts/utils/Debug.h"
#include "carts/utils/ValueAnalysis.h"
ARTS_DEBUG_SETUP(hoisting);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

///===----------------------------------------------------------------------===///
/// Shared Utilities
///===----------------------------------------------------------------------===///

/// Hoist loop-invariant pure ops (e.g., div/rem/mod/max) out of inner loops.
static bool hoistInvariantOpsInLoop(scf::ForOp loop) {
  bool changed = false;
  SmallVector<Operation *> candidates;

  for (Operation &op : loop.getBody()->getOperations()) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    if (isa<scf::ForOp>(op))
      continue;
    if (op.getNumRegions() != 0)
      continue;

    if (!mlir::isPure(&op) && !isSafeToHoistDivRem(loop, &op))
      continue;

    bool allInvariant = true;
    for (Value operand : op.getOperands()) {
      if (!isLoopInvariant(loop, operand)) {
        allInvariant = false;
        break;
      }
    }
    if (!allInvariant)
      continue;

    candidates.push_back(&op);
  }

  for (Operation *op : candidates) {
    ARTS_DEBUG("Hoisting loop-invariant op: " << *op);
    op->moveBefore(loop);
    changed = true;
  }

  return changed;
}

static bool isWorkerLoop(scf::ForOp loop) {
  bool hasEdt = false;
  loop.walk([&](EdtOp) {
    hasEdt = true;
    return WalkResult::interrupt();
  });
  return hasEdt;
}

template <typename Predicate>
static SmallVector<scf::ForOp> collectLoops(Operation *root,
                                            Predicate &&predicate) {
  SmallVector<scf::ForOp> loops;
  root->walk([&](scf::ForOp loop) {
    if (predicate(loop))
      loops.push_back(loop);
  });
  llvm::sort(loops, [](scf::ForOp lhs, scf::ForOp rhs) {
    return getLoopDepth(lhs) > getLoopDepth(rhs);
  });
  return loops;
}

/// Run a per-loop hoisting function to fixed point across all given loops.
template <typename HoistFn>
static bool hoistFixedPoint(ArrayRef<scf::ForOp> loops, HoistFn &&hoist) {
  bool changed = false;
  for (scf::ForOp loop : loops)
    while (hoist(loop))
      changed = true;
  return changed;
}

/// Process an EDT to hoist loop-invariant pure ops from all loops within it.
static bool hoistInvariantOpsInEdt(EdtOp edt) {
  return hoistFixedPoint(collectLoops(edt, [](scf::ForOp) { return true; }),
                         hoistInvariantOpsInLoop);
}

///===----------------------------------------------------------------------===///
/// Epoch Acquire Hoisting
///===----------------------------------------------------------------------===///
///
/// Hoists loop-invariant db_acquire ops out of worker loops in
/// epochs. Safety: hoist only when ALL runtime operands and partition hints are
/// loop invariant. This pass does not drop or rewrite partition hints.
///
/// BEFORE:
///   arts.epoch {
///     scf.for %w = 0 to %workers {
///       %acq = arts.db_acquire [in] %db    // acquired every iteration
///       arts.edt(%acq) { ... }
///       arts.db_release %acq               // released every iteration
///     }
///   }
///
/// AFTER:
///   arts.epoch {
///     %acq = arts.db_acquire [in] %db      // hoisted: acquired once
///     scf.for %w = 0 to %workers {
///       arts.edt(%acq) { ... }
///     }
///     arts.db_release %acq                 // single release after loop
///   }
///
///===----------------------------------------------------------------------===///

static bool canHoistAcquireOutOfEpoch(DbAcquireOp acq) {
  switch (acq.getMode()) {
  case ArtsMode::in:
  case ArtsMode::inout:
    return true;
  case ArtsMode::out:
  case ArtsMode::uninitialized:
    return false;
  }
  return false;
}

/// Rematerialize loop-invariant operands to dominate the insertion point.
/// Returns false if any operand depends on the loop IV or can't be traced.
static bool rematerializeInvariantOperands(Operation *insertBefore,
                                           DbAcquireOp acq,
                                           DominanceInfo &domInfo) {
  OpBuilder builder(insertBefore);
  Location loc = acq.getLoc();

  auto mustDominate = [&](Value v) -> bool {
    if (!v)
      return true;
    return domInfo.properlyDominates(v, insertBefore);
  };

  /// Source guid/ptr must already dominate (memref values can't be rebuilt).
  if (!mustDominate(acq.getSourcePtr()))
    return false;
  if (Value srcGuid = acq.getSourceGuid()) {
    if (!mustDominate(srcGuid))
      return false;
  }

  auto materialize = [&](Value v) -> Value {
    if (!v)
      return v;
    if (domInfo.properlyDominates(v, insertBefore))
      return v;
    if (isa<MemRefType>(v.getType()))
      return nullptr;
    return ValueAnalysis::traceValueToDominating(v, insertBefore, builder,
                                                 domInfo, loc);
  };

  auto replaceIfNeeded = [&](Value v) -> bool {
    if (!v)
      return true;
    Value newV = materialize(v);
    if (!newV)
      return false;
    if (newV != v)
      acq->replaceUsesOfWith(v, newV);
    return true;
  };

  if (!replaceIfNeeded(acq.getBoundsValid()))
    return false;
  for (Value v : acq.getIndices())
    if (!replaceIfNeeded(v))
      return false;
  for (Value v : acq.getOffsets())
    if (!replaceIfNeeded(v))
      return false;
  for (Value v : acq.getSizes())
    if (!replaceIfNeeded(v))
      return false;
  for (Value v : acq.getPartitionIndices())
    if (!replaceIfNeeded(v))
      return false;
  for (Value v : acq.getPartitionOffsets())
    if (!replaceIfNeeded(v))
      return false;
  for (Value v : acq.getPartitionSizes())
    if (!replaceIfNeeded(v))
      return false;
  for (Value v : acq.getElementOffsets())
    if (!replaceIfNeeded(v))
      return false;
  for (Value v : acq.getElementSizes())
    if (!replaceIfNeeded(v))
      return false;

  return true;
}

/// Checks if there is a db_release outside the loop for the given acquire.
/// Collects all in-loop releases in the output vector.
static bool hasReleaseOutsideLoop(scf::ForOp loop, DbAcquireOp acq,
                                  SmallVectorImpl<DbReleaseOp> &releases) {
  bool outside = false;
  for (OpOperand &use : acq.getPtr().getUses()) {
    if (auto rel = dyn_cast<DbReleaseOp>(use.getOwner())) {
      if (!loop->isAncestor(rel)) {
        outside = true;
        continue;
      }
      if (rel->getParentOfType<EdtOp>())
        continue; /// release inside EDT body is unrelated
      releases.push_back(rel);
    }
  }
  return outside;
}

/// Validates that all uses of the acquire result are safe for hoisting.
/// Safe uses: releases inside loop, EDTs inside loop, casts inside loop.
static bool usesAreSafe(scf::ForOp loop, DbAcquireOp acq) {
  auto checkUses = [&](Value v) -> bool {
    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      if (isa<DbReleaseOp>(owner)) {
        /// Allow releases only inside loop (handled separately).
        if (!loop->isAncestor(owner))
          return false;
        continue;
      }
      if (isa<EdtOp>(owner)) {
        if (!loop->isAncestor(owner))
          return false;
        continue;
      }
      /// Allow casts inside the loop (dependency plumbing).
      if (isa<memref::CastOp, memref::ReinterpretCastOp,
              polygeist::Memref2PointerOp>(owner)) {
        if (!loop->isAncestor(owner))
          return false;
        continue;
      }
      /// Reject any other uses.
      return false;
    }
    return true;
  };

  return checkUses(acq.getPtr()) && checkUses(acq.getGuid());
}

/// Hoists read-only db_acquire operations out of a single worker loop.
/// Returns true if any hoisting was performed.
static bool hoistAcquiresInWorkerLoop(scf::ForOp loop) {
  bool changed = false;
  SmallVector<DbAcquireOp> candidates;

  loop.walk([&](Operation *op) -> WalkResult {
    if (isa<EdtOp>(op))
      return WalkResult::skip();
    if (auto acq = dyn_cast<DbAcquireOp>(op))
      candidates.push_back(acq);
    return WalkResult::advance();
  });

  for (DbAcquireOp acq : candidates) {
    if (acq.getMode() != ArtsMode::in) {
      ARTS_DEBUG("Skip acquire: not read-only");
      continue;
    }
    if (acq->getParentOfType<EdtOp>()) {
      ARTS_DEBUG("Skip acquire: inside EDT");
      continue;
    }
    func::FuncOp func = loop->getParentOfType<func::FuncOp>();
    DominanceInfo domInfo(func);
    if (!rematerializeInvariantOperands(loop, acq, domInfo)) {
      ARTS_DEBUG("Skip acquire: operands not loop-invariant: " << acq);
      continue;
    }
    if (!usesAreSafe(loop, acq)) {
      ARTS_DEBUG("Skip acquire: unsafe uses");
      continue;
    }

    SmallVector<DbReleaseOp> releases;
    if (hasReleaseOutsideLoop(loop, acq, releases))
      continue;

    ARTS_DEBUG("Hoisting read-only db_acquire out of worker loop");
    acq->moveBefore(loop);
    changed = true;

    /// If the acquire had per-iteration releases in the loop body, replace
    /// them with a single release after the loop to avoid double-release.
    if (!releases.empty()) {
      for (DbReleaseOp rel : releases)
        rel.erase();

      OpBuilder builder(loop);
      builder.setInsertionPointAfter(loop);
      DbReleaseOp::create(builder, loop.getLoc(), acq.getPtr());
    }
  }

  return changed;
}

/// Processes an epoch to hoist acquires from all worker loops within it.
/// Returns true if any hoisting was performed.
static bool hoistAcquiresInEpoch(EpochOp epoch) {
  bool changed = false;
  Block &body = epoch.getBody().front();

  /// First, try hoisting loop-invariant coarse acquires out of the epoch
  /// entirely. Write-capable inout acquires are safe here when every user stays
  /// in the epoch: the epoch remains the completion boundary, and the original
  /// access mode is preserved on the single hoisted acquire.
  /// This handles acquires inside conditional regions or EDT dependencies by
  /// moving them to the parent block and releasing after the epoch completes.
  auto hoistOutOfEpoch = [&]() {
    bool localChanged = false;
    SmallVector<DbAcquireOp> candidates;

    epoch.walk([&](DbAcquireOp acq) {
      if (!canHoistAcquireOutOfEpoch(acq))
        return;
      if (acq->getParentOfType<EdtOp>())
        return;
      candidates.push_back(acq);
    });

    if (candidates.empty())
      return false;

    func::FuncOp func = epoch->getParentOfType<func::FuncOp>();
    DominanceInfo domInfo(func);

    for (DbAcquireOp acq : candidates) {
      scf::ForOp loop = acq->getParentOfType<scf::ForOp>();
      if (!loop || !isWorkerLoop(loop))
        continue;

      if (!rematerializeInvariantOperands(epoch, acq, domInfo))
        continue;

      /// Ensure all uses are within the epoch region.
      bool safe = true;
      SmallVector<DbReleaseOp> releasesToRemove;
      auto checkUses = [&](Value v) {
        for (OpOperand &use : v.getUses()) {
          Operation *owner = use.getOwner();
          if (!epoch->isAncestor(owner)) {
            safe = false;
            return;
          }
          if (auto rel = dyn_cast<DbReleaseOp>(owner))
            releasesToRemove.push_back(rel);
        }
      };
      checkUses(acq.getPtr());
      checkUses(acq.getGuid());
      if (!safe)
        continue;

      ARTS_DEBUG("Hoisting read-only db_acquire out of epoch (loop-invariant)");

      /// Remove releases inside the epoch (including those inside EDTs).
      for (DbReleaseOp rel : releasesToRemove)
        rel.erase();

      /// Move acquire before epoch and insert a single release after epoch.
      acq->moveBefore(epoch);
      OpBuilder builder(epoch);
      builder.setInsertionPointAfter(epoch);
      DbReleaseOp::create(builder, epoch.getLoc(), acq.getPtr());
      localChanged = true;
    }

    return localChanged;
  };

  changed |= hoistOutOfEpoch();

  for (Operation &op : body) {
    auto loop = dyn_cast<scf::ForOp>(op);
    if (!loop)
      continue;
    size_t acqCount = 0;
    loop.walk([&](DbAcquireOp) { ++acqCount; });
    ARTS_DEBUG("Epoch loop has " << acqCount << " db_acquire ops");
    if (acqCount == 0)
      continue;
    changed |= hoistAcquiresInWorkerLoop(loop);
  }
  return changed;
}

///===----------------------------------------------------------------------===///
/// DbRef Hoisting
///===----------------------------------------------------------------------===///
///
/// Hoists loop-invariant db_ref ops out of inner loops, reducing repeated
/// db_ref creation inside tight loops.
///
/// BEFORE:
///   scf.for %i = 0 to %N {
///     scf.for %j = 0 to %M {
///       %ref = arts.db_ref %arg0[%i]       // created every inner iteration
///       use %ref
///     }
///   }
///
/// AFTER:
///   scf.for %i = 0 to %N {
///     %ref = arts.db_ref %arg0[%i]         // hoisted to outer loop
///     scf.for %j = 0 to %M {
///       use %ref
///     }
///   }
///
///===----------------------------------------------------------------------===///

/// Checks if a db_ref operation is loop invariant (source and all indices
/// are defined outside the loop).
static bool refInvariantInLoop(scf::ForOp loop, DbRefOp ref) {
  if (!isLoopInvariant(loop, ref.getSource()))
    return false;
  for (Value idx : ref.getIndices()) {
    if (!isLoopInvariant(loop, idx))
      return false;
  }
  return true;
}

/// Hoists loop-invariant db_ref operations out of a single loop.
/// Returns true if any hoisting was performed.
static bool hoistDbRefsInLoop(scf::ForOp loop) {
  bool changed = false;
  SmallVector<DbRefOp> refs;

  for (Operation &op : loop.getBody()->getOperations())
    if (auto ref = dyn_cast<DbRefOp>(&op))
      refs.push_back(ref);

  for (DbRefOp ref : refs) {
    if (!refInvariantInLoop(loop, ref))
      continue;
    if (!loop->isAncestor(ref))
      continue;

    ARTS_DEBUG("Hoisting loop-invariant db_ref");
    ref->moveBefore(loop);
    changed = true;
  }
  return changed;
}

/// Processes an EDT to hoist db_refs from all loops within it.
/// Returns true if any hoisting was performed.
static bool hoistDbRefsInEdt(EdtOp edt) {
  return hoistFixedPoint(collectLoops(edt, [](scf::ForOp) { return true; }),
                         hoistDbRefsInLoop);
}

/// Hoist loop-invariant ops and db_refs in loops that are NOT inside EDTs.
static bool hoistOutsideEdt(func::FuncOp funcOp) {
  auto loops = collectLoops(
      funcOp, [](scf::ForOp loop) { return !loop->getParentOfType<EdtOp>(); });

  bool changed = hoistFixedPoint(loops, hoistInvariantOpsInLoop);
  changed |= hoistFixedPoint(loops, hoistDbRefsInLoop);
  return changed;
}

///===----------------------------------------------------------------------===///
/// Pass Definition
///===----------------------------------------------------------------------===///

namespace {
struct ArtsHoistingPass : public impl::HoistingBase<ArtsHoistingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool changed = false;

    if (enableAcquireHoisting) {
      module.walk(
          [&](EpochOp epoch) { changed |= hoistAcquiresInEpoch(epoch); });
      if (changed)
        ARTS_INFO("Epoch acquire hoisting applied");
    }

    if (enableRefHoisting) {
      bool refChanged = false;
      bool arithChanged = false;
      module.walk(
          [&](EdtOp edt) { arithChanged |= hoistInvariantOpsInEdt(edt); });
      if (arithChanged) {
        ARTS_INFO("Loop-invariant op hoisting applied");
        changed = true;
      }
      module.walk([&](EdtOp edt) { refChanged |= hoistDbRefsInEdt(edt); });
      module.walk(
          [&](func::FuncOp funcOp) { refChanged |= hoistOutsideEdt(funcOp); });
      if (refChanged) {
        ARTS_INFO("DbRef hoisting applied");
        changed = true;
      }
    }
  }
};
} // namespace

namespace mlir {
namespace carts::arts {
std::unique_ptr<Pass> createHoistingPass() {
  return std::make_unique<ArtsHoistingPass>();
}
} // namespace carts::arts
} // namespace mlir
