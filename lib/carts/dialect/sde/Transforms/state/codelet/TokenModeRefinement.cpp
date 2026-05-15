///==========================================================================///
/// File: TokenModeRefinement.cpp
///
/// Downgrade access modes on mu_token ops inside cu_codelet bodies.
///
/// If a token is declared readwrite but the codelet body only reads the
/// corresponding block argument (no tensor.insert and no memref write), the
/// token's mode is downgraded to read. This enables more precise downstream
/// access materialization.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::carts::arts {
#define GEN_PASS_DEF_TOKENMODEREFINE
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mlir::carts::arts;
using namespace mlir::carts;

namespace {

/// Check whether a codelet block argument is written to (tensor.insert,
/// memref.store, or any op that yields a modified version of the arg).
static bool hasWriteUse(BlockArgument arg, sde::SdeCuCodeletOp codelet) {
  if (isa<MemRefType>(arg.getType())) {
    bool written = false;
    codelet.getBody().walk([&](Operation *op) {
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (store.getMemref() == arg)
          written = true;
      }
      if (auto subview = dyn_cast<memref::SubViewOp>(op)) {
        if (subview.getSource() != arg)
          return;
        for (Operation *user : subview.getResult().getUsers()) {
          if (auto store = dyn_cast<memref::StoreOp>(user))
            if (store.getMemref() == subview.getResult())
              written = true;
        }
      }
    });
    return written;
  }

  // Check if this arg is yielded back (indicates a write).
  auto yield = cast<sde::SdeYieldOp>(codelet.getBody().front().getTerminator());
  for (Value yieldedVal : yield.getValues()) {
    // Walk the def chain backwards from each yielded value.
    // If the yielded value depends on a tensor.insert that targets this arg,
    // it's a write.
    SmallVector<Value, 8> worklist;
    DenseSet<Value> visited;
    worklist.push_back(yieldedVal);
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!visited.insert(v).second)
        continue;
      if (v == arg) {
        // The arg itself is yielded directly — this is an identity write,
        // but actually means no modification. Check if it passes through
        // any insert op first.
        continue;
      }
      Operation *defOp = v.getDefiningOp();
      if (!defOp)
        continue;
      if (auto insert = dyn_cast<tensor::InsertOp>(defOp)) {
        if (insert.getDest() == arg)
          return true;
        worklist.push_back(insert.getDest());
      } else if (auto insertSlice = dyn_cast<tensor::InsertSliceOp>(defOp)) {
        if (insertSlice.getDest() == arg)
          return true;
        worklist.push_back(insertSlice.getDest());
      } else {
        // For other ops, check if any operand leads back to the arg
        // through a write chain.
        for (Value operand : defOp->getOperands())
          worklist.push_back(operand);
      }
    }
  }
  return false;
}

struct TokenModeRefinePass
    : public arts::impl::TokenModeRefineBase<TokenModeRefinePass> {
  using arts::impl::TokenModeRefineBase<
      TokenModeRefinePass>::TokenModeRefineBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    module.walk([&](sde::SdeCuCodeletOp codelet) {
      // Downgrading a writable token also requires removing the corresponding
      // codelet result/yield slot. This pass only mutates token modes, so leave
      // result-bearing codelets unchanged.
      if (codelet.getNumResults() != 0)
        return;

      // Map each token operand to its corresponding codelet block argument.
      Block &body = codelet.getBody().front();
      unsigned numTokens = codelet.getTokens().size();

      for (unsigned i = 0; i < numTokens; ++i) {
        Value tokenVal = codelet.getTokens()[i];
        auto tokenOp = tokenVal.getDefiningOp<sde::SdeMuTokenOp>();
        if (!tokenOp)
          continue;

        // Only refine readwrite → read.
        if (tokenOp.getMode() != sde::SdeAccessMode::readwrite)
          continue;

        BlockArgument blockArg = body.getArgument(i);

        // If the block arg has no write uses, downgrade to read.
        if (!hasWriteUse(blockArg, codelet)) {
          tokenOp.setModeAttr(
              sde::SdeAccessModeAttr::get(ctx, sde::SdeAccessMode::read));
        }
      }
    });
  }
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createTokenModeRefinementPass() {
  return std::make_unique<TokenModeRefinePass>();
}

} // namespace mlir::carts::sde
