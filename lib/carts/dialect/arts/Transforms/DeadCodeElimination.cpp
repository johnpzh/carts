///==========================================================================///
/// File: DeadCodeElimination.cpp
///
/// This pass performs dead code elimination for ARTS operations and shared
/// dialect-neutral helper IR.
/// It removes:
/// - Dead arts.undef operations (undef values with no uses)
/// - Trivially empty EDTs (EDTs with only yield/barrier/release ops)
/// - Dead datablocks (db_alloc where both guid and ptr are unused)
/// - Unused acquires (db_acquire with no memory ops and unused guid)
/// - Generic helper cleanup delegated to carts::runDeadIrCleanup
///
/// DB lifetime cleanup that depends on forwarding/cleanup use graphs stays in
/// DbTransformsPass and EdtTransformsPass; this pass only removes raw-dead IR.
///
/// Example:
///   Before:
///     %u = arts.undef ...
///     %x = memref.load %A[%i]
///     // %u, %x unused
///
///   After:
///     // dead ops removed
///==========================================================================///

#define GEN_PASS_DEF_DEADCODEELIMINATION
#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "carts/utils/Debug.h"
#include "carts/utils/DeadIrCleanup.h"
#include "carts/utils/RemovalUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

ARTS_DEBUG_SETUP(dead_code_elimination);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

struct DeadCodeEliminationPass
    : public impl::DeadCodeEliminationBase<DeadCodeEliminationPass> {

  static bool isContractUse(Operation *user) {
    return isa<LoweringContractOp, DbFreeOp, DbReleaseOp>(user);
  }

  static bool hasNonContractUse(Value value) {
    return llvm::any_of(value.getUsers(),
                        [](Operation *user) { return !isContractUse(user); });
  }

  static bool hasSingleExpectedNonContractUse(Value value,
                                              Operation *expectedUser) {
    unsigned count = 0;
    for (Operation *user : value.getUsers()) {
      if (isContractUse(user))
        continue;
      if (user != expectedUser)
        return false;
      ++count;
    }
    return count == 1;
  }

  static void collectContractUsers(Value value,
                                   SmallVectorImpl<Operation *> &contracts) {
    for (Operation *user : value.getUsers()) {
      if (isContractUse(user))
        contracts.push_back(user);
    }
  }

  static bool isTrivialEdtBodyOp(Operation &op) {
    if (isa<arts::YieldOp, arts::BarrierOp, arts::DbReleaseOp>(op))
      return true;

    return false;
  }

  static bool isSideEffectFreeRegionTree(Operation &op) {
    if (isTrivialEdtBodyOp(op))
      return true;

    if (op.getNumRegions() == 0)
      return isMemoryEffectFree(&op);

    if (isa<scf::ForOp, scf::IfOp, scf::ExecuteRegionOp, affine::AffineForOp>(
            &op) &&
        op.use_empty()) {
      for (Region &region : op.getRegions()) {
        for (Block &block : region) {
          for (Operation &nested : block) {
            if (!isSideEffectFreeRegionTree(nested))
              return false;
          }
        }
      }
      return true;
    }

    return false;
  }

  void runOnOperation() override {
    auto module = getOperation();
    bool changed = true;
    unsigned iterations = 0;
    unsigned totalRemoved = 0;

    ARTS_DEBUG("Starting DeadCodeElimination pass");

    /// Iterate until no more changes
    while (changed) {
      changed = false;
      iterations++;

      /// Dialect-neutral helper cleanup.
      DeadIrCleanupResult removedGeneric =
          runDeadIrCleanup(module, /*removeSymbols=*/true);

      /// ARTS-specific DCE
      unsigned removedUndefs = removeDeadUndefs(module);
      unsigned removedEdts = removeEmptyEdts(module);
      unsigned removedDbs = removeDeadDbs(module);
      unsigned removedAcquires = removeUnusedAcquires(module);
      unsigned removedDuplicateReleases = removeDuplicateDbReleases(module);
      /// Conservatively keep EDT dependency operands. Some control-only
      /// dependency edges encode ordering constraints even when the block
      /// argument has no direct memory users (for example Jacobi-style
      /// alternating-buffer pipelines).

      unsigned removed = removedGeneric.total() + removedUndefs + removedEdts +
                         removedDbs + removedAcquires +
                         removedDuplicateReleases;
      totalRemoved += removed;
      changed = (removed > 0);
    }

    ARTS_DEBUG("Completed DCE: removed " << totalRemoved << " operations in "
                                         << iterations << " iterations");
  }

  ///===----------------------------------------------------------------------===///
  /// ARTS-specific Dead Code Elimination
  ///===----------------------------------------------------------------------===///

  /// Remove arts.undef operations whose results have no uses
  /// These are created as placeholders during EDT outlining and may remain
  /// after transformation if not all uses were replaced.
  unsigned removeDeadUndefs(ModuleOp module) {
    RemovalUtils removalMgr;
    module.walk([&](UndefOp undef) { removalMgr.markForRemoval(undef); });
    ARTS_DEBUG(" - Removing " << removalMgr.size() << " undef operations");
    removalMgr.removeAllMarked(module, /*recursive=*/true);
    return 0;
  }

  /// Remove trivially empty EDTs (only yield/barrier/release ops)
  unsigned removeEmptyEdts(ModuleOp module) {
    RemovalUtils removalMgr;
    module.walk([&](EdtOp edt) {
      bool hasWork = false;
      for (Operation &op : edt.getBody().front()) {
        if (isSideEffectFreeRegionTree(op))
          continue;
        hasWork = true;
        break;
      }
      if (hasWork)
        return;

      /// Remove the EDT and any single-use acquires that only feed it.
      for (Value dep : edt.getDependencies()) {
        if (auto acq = dep.getDefiningOp<arts::DbAcquireOp>()) {
          bool ptrOnlyUsedHere =
              acq.getPtr() == dep &&
              hasSingleExpectedNonContractUse(acq.getPtr(), edt.getOperation());
          bool guidUnused = acq.getGuid().use_empty();
          if (ptrOnlyUsedHere && guidUnused) {
            SmallVector<Operation *> contracts;
            collectContractUsers(acq.getPtr(), contracts);
            for (Operation *contract : contracts)
              removalMgr.markForRemoval(contract);
            removalMgr.markForRemoval(acq);
          }
        }
      }

      removalMgr.markForRemoval(edt);
    });
    auto opsToRemoveSize = removalMgr.getOpsToRemove().size();
    removalMgr.removeAllMarked(module, /*recursive=*/true);
    return opsToRemoveSize;
  }

  /// Remove arts.db_alloc operations where both guid and ptr are unused
  unsigned removeDeadDbs(ModuleOp module) {
    RemovalUtils removalMgr;

    module.walk([&](DbAllocOp dbAlloc) {
      if (!hasNonContractUse(dbAlloc.getGuid()) &&
          !hasNonContractUse(dbAlloc.getPtr())) {
        SmallVector<Operation *> contracts;
        collectContractUsers(dbAlloc.getPtr(), contracts);
        for (Operation *contract : contracts)
          removalMgr.markForRemoval(contract);

        /// Also remove associated db_free operations
        for (Operation *user : dbAlloc.getGuid().getUsers()) {
          if (auto freeOp = dyn_cast<DbFreeOp>(user)) {
            ARTS_DEBUG("Removing associated db_free for guid: " << freeOp);
            removalMgr.markForRemoval(freeOp);
          }
        }
        for (Operation *user : dbAlloc.getPtr().getUsers()) {
          if (auto freeOp = dyn_cast<DbFreeOp>(user)) {
            ARTS_DEBUG("Removing associated db_free for ptr: " << freeOp);
            removalMgr.markForRemoval(freeOp);
          }
        }

        ARTS_DEBUG("Removing dead db_alloc: " << dbAlloc);
        removalMgr.markForRemoval(dbAlloc);
      }
    });

    auto opsToRemoveSize = removalMgr.getOpsToRemove().size();
    removalMgr.removeAllMarked(module, /*recursive=*/true);
    return opsToRemoveSize;
  }

  /// Remove unused db_acquire operations whose ptr and guid are both unused.
  /// This handles "phantom acquires" - acquires generated for arrays visible
  /// in a parallel scope but not actually accessed in a particular EDT.
  unsigned removeUnusedAcquires(ModuleOp module) {
    RemovalUtils removalMgr;

    module.walk([&](DbAcquireOp acquire) {
      if (!hasNonContractUse(acquire.getPtr()) &&
          !hasNonContractUse(acquire.getGuid())) {
        /// Also remove associated db_release operations
        for (Operation *user : acquire.getPtr().getUsers()) {
          if (auto releaseOp = dyn_cast<DbReleaseOp>(user)) {
            ARTS_DEBUG("Removing associated db_release: " << releaseOp);
            removalMgr.markForRemoval(releaseOp);
          }
        }

        ARTS_DEBUG("Removing unused acquire: " << acquire);
        SmallVector<Operation *> contracts;
        collectContractUsers(acquire.getPtr(), contracts);
        for (Operation *contract : contracts)
          removalMgr.markForRemoval(contract);
        removalMgr.markForRemoval(acquire);
      }
    });

    auto opsToRemoveSize = removalMgr.getOpsToRemove().size();
    removalMgr.removeAllMarked(module, /*recursive=*/true);
    return opsToRemoveSize;
  }

  /// Remove duplicate db_release operations on the same SSA value within a
  /// block. A repeated release of the same acquire view is never meaningful
  /// and later lowers to duplicate runtime releases on the same guid.
  unsigned removeDuplicateDbReleases(ModuleOp module) {
    SmallVector<Operation *> toRemove;

    module.walk([&](Operation *op) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          llvm::SmallDenseSet<Value, 8> releasedValues;
          for (Operation &nested : block) {
            auto release = dyn_cast<DbReleaseOp>(&nested);
            if (!release)
              continue;
            Value source = release.getSource();
            if (!releasedValues.insert(source).second) {
              ARTS_DEBUG("Removing duplicate db_release: " << release);
              toRemove.push_back(release);
            }
          }
        }
      }
    });

    for (Operation *op : toRemove)
      op->erase();

    return toRemove.size();
  }

};

} // namespace

std::unique_ptr<Pass> mlir::carts::arts::createDCEPass() {
  return std::make_unique<DeadCodeEliminationPass>();
}
