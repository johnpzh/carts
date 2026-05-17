///==========================================================================///
/// File: DeadCodeElimination.cpp
///
/// This pass performs dead code elimination for ARTS and memref operations.
/// It removes:
/// - Dead loads (loads whose results are unused)
/// - Dead stores (stores to allocas that are never loaded)
/// - Dead allocas (allocas with no remaining uses)
/// - Dead arts.undef operations (undef values with no uses)
/// - Trivially empty EDTs (EDTs with only yield/barrier/release ops)
/// - Dead datablocks (db_alloc where both guid and ptr are unused)
/// - Unused acquires (db_acquire with no memory ops and unused guid)
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
#include "carts/Dialect.h"
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "carts/utils/Debug.h"
#include "carts/utils/RemovalUtils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Ops.h"
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

  static bool isForwardingMemrefAliasOp(Operation *op, Value source) {
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(op))
      return viewLike.getViewSource() == source && op->getNumResults() == 1;

    if (auto cast = dyn_cast<memref::CastOp>(op))
      return cast.getSource() == source && op->getNumResults() == 1;

    if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(op))
      return unrealized.getInputs().size() == 1 &&
             unrealized.getInputs().front() == source &&
             unrealized.getOutputs().size() == 1;

    if (auto subindex = dyn_cast<polygeist::SubIndexOp>(op))
      return subindex.getSource() == source && op->getNumResults() == 1;

    return false;
  }

  static bool hasLoadThroughAlias(Value value, DenseSet<Value> &visited) {
    if (!visited.insert(value).second)
      return false;

    for (Operation *user : value.getUsers()) {
      if (auto load = dyn_cast<memref::LoadOp>(user)) {
        if (load.getMemref() == value)
          return true;
      } else if (auto load = dyn_cast<affine::AffineLoadOp>(user)) {
        if (load.getMemref() == value)
          return true;
      } else if (isForwardingMemrefAliasOp(user, value)) {
        if (hasLoadThroughAlias(user->getResult(0), visited))
          return true;
      } else if (!isa<memref::StoreOp, affine::AffineStoreOp>(user)) {
        return true;
      }
    }

    return false;
  }

  static bool onlyMemoryEffectFreeOpsBetween(Operation *producer,
                                             Operation *consumer) {
    if (!producer || !consumer || producer->getBlock() != consumer->getBlock())
      return false;

    for (Operation *cur = producer->getNextNode(); cur && cur != consumer;
         cur = cur->getNextNode()) {
      if (!isMemoryEffectFree(cur))
        return false;
    }
    return producer->isBeforeInBlock(consumer);
  }

  static bool isNoOpSelfStore(memref::StoreOp store) {
    auto load = store.getValueToStore().getDefiningOp<memref::LoadOp>();
    if (!load)
      return false;
    if (load.getMemref() != store.getMemref())
      return false;
    if (!ValueAnalysis::areValueRangesEquivalent(load.getIndices(),
                                                 store.getIndices()))
      return false;
    return onlyMemoryEffectFreeOpsBetween(load.getOperation(),
                                          store.getOperation());
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

      /// Memref DCE
      unsigned removedLoads = removeDeadLoads(module);
      unsigned removedStores = removeDeadStores(module);
      unsigned removedNoOpStores = removeNoOpSelfStores(module);
      unsigned removedAllocas = removeDeadAllocas(module);

      /// ARTS-specific DCE
      unsigned removedUndefs = removeDeadUndefs(module);
      unsigned removedEdts = removeEmptyEdts(module);
      unsigned removedDbs = removeDeadDbs(module);
      unsigned removedAcquires = removeUnusedAcquires(module);
      unsigned removedDuplicateReleases = removeDuplicateDbReleases(module);
      unsigned removedPureOps = removeDeadPureOps(module);
      /// Conservatively keep EDT dependency operands. Some control-only
      /// dependency edges encode ordering constraints even when the block
      /// argument has no direct memory users (for example Jacobi-style
      /// alternating-buffer pipelines).

      unsigned removed = removedLoads + removedStores + removedNoOpStores +
                         removedAllocas + removedUndefs + removedEdts +
                         removedDbs + removedAcquires +
                         removedDuplicateReleases + removedPureOps;
      totalRemoved += removed;
      changed = (removed > 0);
    }

    ARTS_DEBUG("Completed DCE: removed " << totalRemoved << " operations in "
                                         << iterations << " iterations");

    /// Also remove dead symbols (functions/globals)
    removeDeadSymbols(module);
  }

  /// Remove memref.load operations whose results have no uses
  unsigned removeDeadLoads(ModuleOp module) {
    SmallVector<Operation *> toRemove;

    module.walk([&](memref::LoadOp load) {
      if (load.getResult().use_empty()) {
        ARTS_DEBUG("Removing dead load: " << load);
        toRemove.push_back(load);
      }
    });

    for (auto *op : toRemove)
      op->erase();

    return toRemove.size();
  }

  /// Remove memref.store operations to allocas that have no loads
  unsigned removeDeadStores(ModuleOp module) {
    SmallVector<Operation *> toRemove;

    module.walk([&](memref::AllocaOp alloca) {
      bool hasLoads = false;
      SmallVector<memref::StoreOp> stores;

      DenseSet<Value> visited;
      hasLoads = hasLoadThroughAlias(alloca.getResult(), visited);

      for (auto *user : alloca->getUsers()) {
        if (auto store = dyn_cast<memref::StoreOp>(user)) {
          /// Only consider stores TO this alloca (not stores of alloca value)
          if (store.getMemref() == alloca.getResult())
            stores.push_back(store);
        }
      }

      /// If no loads from this alloca, all stores to it are dead
      if (!hasLoads) {
        for (auto store : stores) {
          ARTS_DEBUG("Removing dead store: " << store);
          toRemove.push_back(store);
        }
      }
    });

    for (auto *op : toRemove)
      op->erase();

    return toRemove.size();
  }

  /// Remove stores that write back an unchanged value loaded from the exact
  /// same memref element with no intervening side-effecting operation.
  unsigned removeNoOpSelfStores(ModuleOp module) {
    SmallVector<Operation *> toRemove;

    module.walk([&](memref::StoreOp store) {
      if (isNoOpSelfStore(store)) {
        ARTS_DEBUG("Removing no-op self-store: " << store);
        toRemove.push_back(store);
      }
    });

    for (Operation *op : toRemove)
      op->erase();

    return toRemove.size();
  }

  /// Remove memref.alloca operations with no users
  unsigned removeDeadAllocas(ModuleOp module) {
    SmallVector<Operation *> toRemove;

    module.walk([&](memref::AllocaOp alloca) {
      if (alloca->use_empty()) {
        ARTS_DEBUG("Removing dead alloca: " << alloca);
        toRemove.push_back(alloca);
      }
    });

    for (auto *op : toRemove)
      op->erase();

    return toRemove.size();
  }

  /// Remove trivially-dead, memory-effect-free ops.
  /// This cleans up residual arithmetic/type-size helper chains that remain
  /// after structural rewrites.
  unsigned removeDeadPureOps(ModuleOp module) {
    SmallVector<Operation *> toRemove;

    module.walk([&](Operation *op) {
      if (!op || !op->use_empty())
        return;
      if (op->hasTrait<OpTrait::IsTerminator>())
        return;
      if (!op->getRegions().empty())
        return;
      if (!isMemoryEffectFree(op))
        return;
      toRemove.push_back(op);
    });

    for (Operation *op : toRemove)
      op->erase();
    return toRemove.size();
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

  /// Remove dead symbols (functions/globals) that are private and unused
  void removeDeadSymbols(ModuleOp module) {
    SmallVector<Operation *> toRemove;
    SymbolTableCollection symbolTable;

    /// Collect all symbol uses
    DenseSet<Operation *> usedSymbols;
    module.walk([&](Operation *op) {
      /// Look for symbol references in this operation
      for (auto attr : op->getAttrs()) {
        if (auto symbolRef = dyn_cast<SymbolRefAttr>(attr.getValue())) {
          if (Operation *symbol =
                  symbolTable.lookupNearestSymbolFrom(op, symbolRef))
            usedSymbols.insert(symbol);
        }
      }
    });

    /// Find dead private symbols
    module.walk([&](Operation *op) {
      if (auto symbol = dyn_cast<SymbolOpInterface>(op)) {
        if (symbol.isPrivate() && !usedSymbols.count(op)) {
          ARTS_DEBUG("Removing dead symbol: " << op->getName());
          toRemove.push_back(op);
        }
      }
    });

    for (auto *op : toRemove)
      op->erase();

    if (!toRemove.empty())
      ARTS_DEBUG("Removed " << toRemove.size() << " dead symbols");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::carts::arts::createDCEPass() {
  return std::make_unique<DeadCodeEliminationPass>();
}
