///==========================================================================///
/// File: EdtAllocaSinking.cpp
///
/// Clone external stack allocas into EDT regions so that task outlining
/// captures a private copy instead of a shared pointer.
///==========================================================================///

#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Utils/EdtUtils.h"
#include "carts/passes/Passes.h"
#include "carts/utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

#define GEN_PASS_DEF_EDTALLOCASINKING
#include "carts/passes/Passes.h.inc"

namespace {

unsigned sinkExternalAllocasInEdt(EdtOp edt) {
  Block &body = edt.getBody().front();
  DenseMap<Operation *, SmallVector<Operation *, 4>> usesByAlloca;
  DenseMap<Operation *, unsigned> operationOrder;

  body.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      auto allocaOp = operand.getDefiningOp<memref::AllocaOp>();
      if (!allocaOp)
        continue;
      if (edt.getBody().isAncestor(allocaOp->getParentRegion()))
        continue;
      usesByAlloca[allocaOp.getOperation()].push_back(op);
    }
  });

  if (usesByAlloca.empty())
    return 0;

  if (func::FuncOp parentFunc = edt->getParentOfType<func::FuncOp>()) {
    unsigned ordinal = 0;
    parentFunc.walk([&](Operation *op) { operationOrder[op] = ordinal++; });
  }

  OpBuilder builder(edt);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&body);

  unsigned sunkAllocas = 0;
  for (const auto &entry : usesByAlloca) {
    auto allocaOp = cast<memref::AllocaOp>(entry.first);
    bool hasStoreInEdt = false;
    for (Operation *user : entry.second) {
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemRef() == allocaOp.getResult()) {
          hasStoreInEdt = true;
          break;
        }
      }
    }

    bool hasUnsafeStore = false;
    SmallVector<memref::StoreOp, 4> initStores;
    for (Operation *user : allocaOp->getUsers()) {
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemRef() != allocaOp.getResult())
          continue;
        /// Stores inside the EDT body are private writes (e.g., OpenMP
        /// private(buffer) lowered by cgeist outside omp.parallel). These
        /// should not prevent sinking — they are the reason we want to sink.
        if (edt.getBody().isAncestor(store->getParentRegion()))
          continue;
        if (!EdtUtils::canCloneAllocaInitStore(store, allocaOp.getResult())) {
          hasUnsafeStore = true;
          break;
        }
        initStores.push_back(store);
        continue;
      }
    }
    std::stable_sort(initStores.begin(), initStores.end(),
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

    if (hasUnsafeStore)
      continue;

    /// For task and single EDTs, variables are shared by default (OpenMP
    /// semantics). If an alloca is written inside the EDT AND has uses
    /// OUTSIDE the EDT (including other sibling EDTs or continuation code),
    /// it is a shared variable — do NOT sink it or the writes will be lost.
    if (hasStoreInEdt) {
      bool hasExternalUses = false;
      for (Operation *user : allocaOp->getUsers()) {
        if (!edt.getBody().isAncestor(user->getParentRegion())) {
          // Check that this external use is not just the init stores we
          // would clone — only non-store external uses count.
          if (!isa<memref::StoreOp>(user) ||
              cast<memref::StoreOp>(user).getMemRef() != allocaOp.getResult()) {
            hasExternalUses = true;
            break;
          }
          // External stores that are also init stores are OK
          bool isInitStore = llvm::is_contained(initStores, user);
          if (!isInitStore) {
            hasExternalUses = true;
            break;
          }
        }
      }
      if (hasExternalUses)
        continue;
    }

    /// If the alloca is only read inside the EDT, sink it only when we can
    /// safely clone its initialization stores. This preserves scalar values
    /// that would otherwise be uninitialized after outlining.
    if (!hasStoreInEdt && initStores.empty())
      continue;

    Operation *clonedOp = builder.clone(*allocaOp.getOperation());
    ++sunkAllocas;
    auto newAlloca = cast<memref::AllocaOp>(clonedOp);
    IRMapping mapping;
    mapping.map(allocaOp.getResult(), newAlloca.getResult());

    for (Operation *user : entry.second)
      user->replaceUsesOfWith(allocaOp.getResult(), newAlloca.getResult());

    if (!initStores.empty()) {
      builder.setInsertionPointAfter(newAlloca);
      for (memref::StoreOp store : initStores)
        builder.clone(*store.getOperation(), mapping);
      builder.setInsertionPointToStart(&body);
    }
  }
  return sunkAllocas;
}

struct EdtAllocaSinkingPass
    : public ::impl::EdtAllocaSinkingBase<EdtAllocaSinkingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](EdtOp edt) { (void)sinkExternalAllocasInEdt(edt); });
  }
};

} // namespace

namespace mlir {
namespace carts::arts {
std::unique_ptr<Pass> createEdtAllocaSinkingPass() {
  return std::make_unique<EdtAllocaSinkingPass>();
}
} // namespace carts::arts
} // namespace mlir
