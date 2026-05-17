///==========================================================================///
/// File: ScalarForwarding.cpp
///
/// Forward constant-initialized rank-0 memref allocas across region
/// boundaries that block standard mem2reg (omp.parallel, cu_region, etc.).
///
/// For each rank-0 memref.alloca initialized with a constant C:
///   (A) loads inside nested regions are replaced with C if all stores
///       inside that region also write C;
///   (B) same-scope loads are replaced with C if all stores before the
///       load (including those inside intervening region-holding ops)
///       also write C.
/// Subsequent canonicalize/DCE passes then fold the always-true/false
/// scf.if conditions and remove dead stores.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::carts::sde {
#define GEN_PASS_DEF_SCALARFORWARDING
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::sde

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(scalar_forwarding);

using namespace mlir;
using namespace mlir::carts;

namespace {

/// Return the constant attribute if \p value is defined by arith.constant.
static Attribute getConstantAttr(Value value) {
  if (auto cst = value.getDefiningOp<arith::ConstantOp>())
    return cst.getValue();
  return {};
}

/// Check whether every memref.store to \p alloca inside \p region writes
/// a value whose constant attribute equals \p expected.
static bool allStoresInRegionMatch(Value alloca, Region &region,
                                   Attribute expected) {
  bool ok = true;
  region.walk([&](memref::StoreOp store) {
    if (!ok)
      return;
    if (store.getMemref() != alloca)
      return;
    Attribute attr = getConstantAttr(store.getValueToStore());
    if (!attr || attr != expected)
      ok = false;
  });
  return ok;
}

/// Find the outermost region between \p inner and \p allocaRegion.
/// Returns nullptr if \p inner IS \p allocaRegion.
static Region *getOutermostEnclosingRegion(Region *inner,
                                           Region *allocaRegion) {
  if (inner == allocaRegion)
    return nullptr;
  Region *prev = inner;
  while (prev->getParentRegion() != allocaRegion) {
    prev = prev->getParentRegion();
    if (!prev)
      return nullptr;
  }
  return prev;
}

struct ScalarForwardingPass
    : public sde::impl::ScalarForwardingBase<ScalarForwardingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    unsigned forwarded = 0;

    module.walk([&](memref::AllocaOp alloca) {
      if (alloca.getType().getRank() != 0)
        return;

      Value allocaVal = alloca.getResult();
      Region *allocaRegion = alloca->getParentRegion();
      Block *allocaBlock = alloca->getBlock();

      // Find the last store in the alloca's own block that is before any
      // region-holding op and writes a constant.  We walk all users and
      // keep the store that (a) is in the same block, (b) is after the
      // alloca, and (c) is earliest — i.e. the initializing store.
      memref::StoreOp initStore = nullptr;
      for (Operation *user : allocaVal.getUsers()) {
        auto store = dyn_cast<memref::StoreOp>(user);
        if (!store || store->getBlock() != allocaBlock)
          continue;
        if (store->isBeforeInBlock(alloca))
          continue;
        if (!initStore || store->isBeforeInBlock(initStore))
          initStore = store;
      }

      if (!initStore)
        return;

      Attribute initAttr = getConstantAttr(initStore.getValueToStore());
      if (!initAttr)
        return;

      // For each load, check if forwarding is safe.  Handles two cases:
      // (A) Loads inside nested regions (omp.parallel, cu_region, etc.)
      // (B) Same-scope loads where intervening region ops block mem2reg
      SmallVector<memref::LoadOp> toForward;
      for (Operation *user : allocaVal.getUsers()) {
        auto load = dyn_cast<memref::LoadOp>(user);
        if (!load)
          continue;

        Region *loadRegion = load->getParentRegion();

        // Case B: same-scope load blocked by intervening region ops.
        if (loadRegion == allocaRegion) {
          if (load->getBlock() != allocaBlock)
            continue;
          if (!initStore->isBeforeInBlock(load))
            continue;

          // All same-block stores before the load must write initAttr.
          bool allMatch = true;
          for (Operation *u : allocaVal.getUsers()) {
            auto st = dyn_cast<memref::StoreOp>(u);
            if (!st || st->getBlock() != allocaBlock)
              continue;
            if (!st->isBeforeInBlock(load))
              continue;
            Attribute a = getConstantAttr(st.getValueToStore());
            if (!a || a != initAttr) {
              allMatch = false;
              break;
            }
          }
          if (!allMatch)
            continue;

          // All stores inside any region-holding op between init and load
          // must also write initAttr.
          bool regionsOk = true;
          for (Operation &op : *allocaBlock) {
            if (!initStore->isBeforeInBlock(&op))
              continue;
            if (!op.isBeforeInBlock(load))
              continue;
            for (Region &r : op.getRegions()) {
              if (!allStoresInRegionMatch(allocaVal, r, initAttr)) {
                regionsOk = false;
                break;
              }
            }
            if (!regionsOk)
              break;
          }
          if (!regionsOk)
            continue;

          toForward.push_back(load);
          continue;
        }

        // Case A: load inside a nested region.
        Region *outerRegion =
            getOutermostEnclosingRegion(loadRegion, allocaRegion);
        if (!outerRegion)
          continue;

        Operation *regionOp = outerRegion->getParentOp();
        if (regionOp->getBlock() != allocaBlock)
          continue;

        // The init store must be before the region entry.
        if (!initStore->isBeforeInBlock(regionOp))
          continue;

        // Check: no store between initStore and regionOp writes a
        // different constant.  We find the LAST store before regionOp
        // and verify it writes the same constant.
        memref::StoreOp lastBefore = nullptr;
        for (Operation *u : allocaVal.getUsers()) {
          auto st = dyn_cast<memref::StoreOp>(u);
          if (!st || st->getBlock() != allocaBlock)
            continue;
          if (!st->isBeforeInBlock(regionOp))
            continue;
          if (!lastBefore || lastBefore->isBeforeInBlock(st))
            lastBefore = st;
        }
        if (lastBefore) {
          Attribute lastAttr = getConstantAttr(lastBefore.getValueToStore());
          if (!lastAttr || lastAttr != initAttr)
            continue;
        }

        // All stores inside the region must write the same constant.
        if (!allStoresInRegionMatch(allocaVal, *outerRegion, initAttr))
          continue;

        toForward.push_back(load);
      }

      if (toForward.empty())
        return;

      Value constVal = initStore.getValueToStore();
      for (memref::LoadOp load : toForward) {
        ARTS_DEBUG("forwarding constant to load: " << *load);
        load.getResult().replaceAllUsesWith(constVal);
        load.erase();
        ++forwarded;
      }
    });

    ARTS_INFO("ScalarForwarding: forwarded " << forwarded << " load(s)");
  }
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createScalarForwardingPass() {
  return std::make_unique<ScalarForwardingPass>();
}

} // namespace mlir::carts::sde
