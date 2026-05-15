///==========================================================================///
/// File: EdtTaskBodyCloning.cpp
///==========================================================================///

#include "carts/dialect/arts/Transforms/edt/EdtTaskBodyCloning.h"
#include "carts/utils/EdtUtils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <optional>

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_task_body_cloning);

using namespace mlir;
using namespace mlir::arts;

void mlir::arts::collectExternalValues(
    Block &sourceBlock, Region *boundaryRegion,
    llvm::SetVector<Value> &externalValues,
    const llvm::DenseSet<Operation *> &opsToSkip) {
  Region *sourceRegion = sourceBlock.getParent();

  for (Operation &op : sourceBlock.without_terminator()) {
    if (opsToSkip.contains(&op))
      continue;

    op.walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          Region *ownerRegion = blockArg.getOwner()->getParent();
          if (sourceRegion->isAncestor(ownerRegion))
            continue;
          if (boundaryRegion->isAncestor(ownerRegion))
            continue;
        }
        if (Operation *defOp = operand.getDefiningOp()) {
          if (sourceRegion->isAncestor(defOp->getParentRegion()))
            continue;
          if (!boundaryRegion->isAncestor(defOp->getParentRegion()))
            externalValues.insert(operand);
        }
      }
    });
  }
}

void mlir::arts::cloneExternalAllocasIntoEdt(Region *taskEdtRegion,
                                             Block &taskBlock,
                                             IRMapping &mapper,
                                             OpBuilder &builder) {
  DenseMap<Operation *, SmallVector<Operation *, 4>> usesByAlloca;
  DenseMap<Operation *, unsigned> operationOrder;

  taskBlock.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      auto allocaOp = operand.getDefiningOp<memref::AllocaOp>();
      if (!allocaOp)
        continue;
      if (taskEdtRegion->isAncestor(allocaOp->getParentRegion()))
        continue;
      usesByAlloca[allocaOp.getOperation()].push_back(op);
    }
  });

  if (usesByAlloca.empty())
    return;

  if (func::FuncOp parentFunc =
          taskEdtRegion->getParentOfType<func::FuncOp>()) {
    unsigned ordinal = 0;
    parentFunc.walk([&](Operation *op) { operationOrder[op] = ordinal++; });
  }

  auto sortStoresWithOrder = [&](MutableArrayRef<memref::StoreOp> stores) {
    std::stable_sort(stores.begin(), stores.end(),
                     [&](memref::StoreOp lhs, memref::StoreOp rhs) {
                       Operation *lhsOp = lhs.getOperation();
                       Operation *rhsOp = rhs.getOperation();
                       if (lhsOp == rhsOp)
                         return false;
                       if (lhsOp->getBlock() == rhsOp->getBlock())
                         return lhsOp->isBeforeInBlock(rhsOp);
                       auto lhsIt = operationOrder.find(lhsOp);
                       auto rhsIt = operationOrder.find(rhsOp);
                       if (lhsIt != operationOrder.end() &&
                           rhsIt != operationOrder.end())
                         return lhsIt->second < rhsIt->second;
                       return lhsOp < rhsOp;
                     });
  };

  ARTS_DEBUG("  - Cloning " << usesByAlloca.size()
                            << " external stack allocas into EDT");

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&taskBlock);

  for (const auto &entry : usesByAlloca) {
    auto allocaOp = cast<memref::AllocaOp>(entry.first);
    Value originalMem = allocaOp.getResult();

    bool hasStoreInEdt = false;
    for (Operation *user : entry.second) {
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemRef() == originalMem) {
          hasStoreInEdt = true;
          break;
        }
      }
    }

    bool hasStoreOutsideEdt = false;
    for (Operation *user : allocaOp->getUsers()) {
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemRef() != originalMem)
          continue;
        if (!taskEdtRegion->isAncestor(store->getParentRegion())) {
          hasStoreOutsideEdt = true;
          break;
        }
      }
    }

    /// If the alloca is initialized outside and only read inside the EDT,
    /// keep the original alloca to preserve initialized values.
    Region *allocaRegion = allocaOp->getParentRegion();
    bool allocaVisible =
        allocaRegion && allocaRegion->isAncestor(taskEdtRegion);

    /// Clone safe initialization stores for this alloca when available.
    SmallVector<memref::StoreOp, 4> initStores;
    bool hasUnsafeStore = false;
    for (Operation *user : allocaOp->getUsers()) {
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemRef() != originalMem)
          continue;
        if (!EdtUtils::canCloneAllocaInitStore(store, originalMem)) {
          hasUnsafeStore = true;
          break;
        }
        initStores.push_back(store);
      }
    }
    sortStoresWithOrder(initStores);

    if (hasStoreOutsideEdt && !hasStoreInEdt && allocaVisible &&
        (hasUnsafeStore || initStores.empty())) {
      continue;
    }

    Operation *clonedOp = builder.clone(*allocaOp.getOperation(), mapper);
    auto newAlloca = cast<memref::AllocaOp>(clonedOp);
    Value clonedMem = newAlloca.getResult();

    for (Operation *user : entry.second)
      user->replaceUsesOfWith(originalMem, clonedMem);

    mapper.map(originalMem, clonedMem);

    if (!hasUnsafeStore && !initStores.empty()) {
      IRMapping storeMapper(mapper);
      storeMapper.map(allocaOp.getResult(), newAlloca.getResult());
      Operation *insertBefore = newAlloca->getNextNode();
      func::FuncOp parentFunc = taskEdtRegion->getParentOfType<func::FuncOp>();
      std::optional<DominanceInfo> domInfo;
      if (insertBefore && parentFunc)
        domInfo.emplace(parentFunc);

      builder.setInsertionPointAfter(newAlloca);
      for (memref::StoreOp store : initStores) {
        IRMapping thisStoreMapper(storeMapper);
        bool canCloneStore = true;
        if (insertBefore && domInfo) {
          for (Value operand : store->getOperands()) {
            if (operand == originalMem)
              continue;
            Value mapped = thisStoreMapper.lookupOrDefault(operand);
            if (!mapped) {
              canCloneStore = false;
              break;
            }
            if (domInfo->properlyDominates(mapped, insertBefore) ||
                domInfo->dominates(mapped, insertBefore))
              continue;
            Value rematerialized = ValueAnalysis::traceValueToDominating(
                mapped, insertBefore, builder, *domInfo, store.getLoc());
            if (!rematerialized) {
              canCloneStore = false;
              break;
            }
            thisStoreMapper.map(operand, rematerialized);
          }
        }
        if (!canCloneStore)
          continue;
        builder.clone(*store.getOperation(), thisStoreMapper);
      }
      builder.setInsertionPointToStart(&taskBlock);
    }
  }
}
