///==========================================================================///
/// File: DeadIrCleanup.cpp
///
/// Shared dead-IR cleanup for dialect-neutral helper IR.
///==========================================================================///

#include "carts/utils/DeadIrCleanup.h"

#include "carts/utils/Debug.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/DenseSet.h"

ARTS_DEBUG_SETUP(dead_ir_cleanup);

using namespace mlir;
using namespace mlir::carts;

namespace {

bool isForwardingMemrefAliasOp(Operation *op, Value source) {
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

bool hasLoadThroughAlias(Value value, DenseSet<Value> &visited) {
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

bool onlyMemoryEffectFreeOpsBetween(Operation *producer, Operation *consumer) {
  if (!producer || !consumer || producer->getBlock() != consumer->getBlock())
    return false;

  for (Operation *cur = producer->getNextNode(); cur && cur != consumer;
       cur = cur->getNextNode()) {
    if (!isMemoryEffectFree(cur))
      return false;
  }
  return producer->isBeforeInBlock(consumer);
}

bool isNoOpSelfStore(memref::StoreOp store) {
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

unsigned removeDeadLoads(ModuleOp module) {
  SmallVector<Operation *> toRemove;

  module.walk([&](memref::LoadOp load) {
    if (load.getResult().use_empty()) {
      ARTS_DEBUG("Removing dead load: " << load);
      toRemove.push_back(load);
    }
  });

  for (Operation *op : toRemove)
    op->erase();

  return toRemove.size();
}

unsigned removeDeadStores(ModuleOp module) {
  SmallVector<Operation *> toRemove;

  module.walk([&](memref::AllocaOp alloca) {
    SmallVector<memref::StoreOp> stores;

    DenseSet<Value> visited;
    bool hasLoads = hasLoadThroughAlias(alloca.getResult(), visited);

    for (Operation *user : alloca->getUsers()) {
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemref() == alloca.getResult())
          stores.push_back(store);
      }
    }

    if (!hasLoads) {
      for (memref::StoreOp store : stores) {
        ARTS_DEBUG("Removing dead store: " << store);
        toRemove.push_back(store);
      }
    }
  });

  for (Operation *op : toRemove)
    op->erase();

  return toRemove.size();
}

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

unsigned removeDeadAllocas(ModuleOp module) {
  SmallVector<Operation *> toRemove;

  module.walk([&](memref::AllocaOp alloca) {
    if (alloca->use_empty()) {
      ARTS_DEBUG("Removing dead alloca: " << alloca);
      toRemove.push_back(alloca);
    }
  });

  for (Operation *op : toRemove)
    op->erase();

  return toRemove.size();
}

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

unsigned removeDeadSymbols(ModuleOp module) {
  SmallVector<Operation *> toRemove;
  SymbolTableCollection symbolTable;

  DenseSet<Operation *> usedSymbols;
  module.walk([&](Operation *op) {
    for (NamedAttribute attr : op->getAttrs()) {
      auto symbolRef = dyn_cast<SymbolRefAttr>(attr.getValue());
      if (!symbolRef)
        continue;
      if (Operation *symbol = symbolTable.lookupNearestSymbolFrom(op, symbolRef))
        usedSymbols.insert(symbol);
    }
  });

  module.walk([&](Operation *op) {
    if (auto symbol = dyn_cast<SymbolOpInterface>(op)) {
      if (symbol.isPrivate() && !usedSymbols.count(op)) {
        ARTS_DEBUG("Removing dead symbol: " << op->getName());
        toRemove.push_back(op);
      }
    }
  });

  for (Operation *op : toRemove)
    op->erase();

  return toRemove.size();
}

} // namespace

DeadIrCleanupResult mlir::carts::runDeadIrCleanup(ModuleOp module,
                                                  bool removeSymbols) {
  DeadIrCleanupResult result;
  result.loads = removeDeadLoads(module);
  result.stores = removeDeadStores(module);
  result.noOpStores = removeNoOpSelfStores(module);
  result.allocas = removeDeadAllocas(module);
  result.pureOps = removeDeadPureOps(module);
  if (removeSymbols)
    result.symbols = removeDeadSymbols(module);
  return result;
}
