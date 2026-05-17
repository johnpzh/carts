///==========================================================================///
/// File: ScalarReplacement.cpp
///
/// This pass transforms memory-based reduction patterns in scf.for loops
/// to register-based iter_args patterns to enable LLVM vectorization.
///
/// Before (blocks vectorization due to loop-carried memory dependency):
///   scf.for %i = ... {
///     %old = memref.load %acc[...]
///     %new = arith.addf %old, %val
///     memref.store %new, %acc[...]
///   }
///
/// After (enables vectorization):
///   %init = memref.load %acc[...]
///   %final = scf.for %i = ... iter_args(%acc_reg = %init) {
///     %new = arith.addf %acc_reg, %val
///     scf.yield %new
///   }
///   memref.store %final, %acc[...]
///
///==========================================================================///

#include "carts/dialect/arts-rt/Transforms/Passes.h"

namespace mlir::carts::arts_rt {
#define GEN_PASS_DEF_SCALARREPLACEMENT
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts_rt

#include "carts/utils/Debug.h"
#include "carts/dialect/arts/Utils/LoopInvarianceUtils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/SetVector.h"

ARTS_DEBUG_SETUP(scalar_replacement);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

/// Information about a detected reduction pattern
struct ReductionPattern {
  scf::ForOp forOp;
  Operation *loadOp;          /// memref.load or polygeist.load
  Operation *storeOp;         /// memref.store or polygeist.store
  Value valueToYield;         /// SSA value that replaces the loop-carried store
  Value memref;               /// The memref being accessed
  SmallVector<Value> indices; /// Loop-invariant indices
};

/// Check if all indices are loop-invariant
static bool areIndicesLoopInvariant(ArrayRef<Value> indices, scf::ForOp forOp) {
  for (Value idx : indices) {
    if (!isLoopInvariant(forOp, idx))
      return false;
  }
  return true;
}

/// Get the memref and indices from a load operation
static std::pair<Value, SmallVector<Value>> getLoadInfo(Operation *op) {
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return {load.getMemRef(), SmallVector<Value>(load.getIndices())};

  /// Handle polygeist.load (DynLoadOp)
  if (auto dynLoad = dyn_cast<polygeist::DynLoadOp>(op))
    return {dynLoad.getMemref(), SmallVector<Value>(dynLoad.getIndices())};
  return {Value(), {}};
}

/// Get the memref and indices from a store operation
static std::pair<Value, SmallVector<Value>> getStoreInfo(Operation *op) {
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return {store.getMemRef(), SmallVector<Value>(store.getIndices())};

  /// Handle polygeist.store (DynStoreOp)
  if (auto dynStore = dyn_cast<polygeist::DynStoreOp>(op))
    return {dynStore.getMemref(), SmallVector<Value>(dynStore.getIndices())};
  return {Value(), {}};
}

/// Get the value being stored
static Value getStoredValue(Operation *op) {
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return store.getValue();
  if (auto dynStore = dyn_cast<polygeist::DynStoreOp>(op))
    return dynStore.getValue();
  return Value();
}

/// Check if an operation is an accumulation operation (addf, mulf, etc.)
static bool isAccumulationOp(Operation *op) {
  return isa<arith::AddFOp, arith::MulFOp, arith::AddIOp, arith::MulIOp,
             arith::MaximumFOp, arith::MinimumFOp>(op);
}

/// Check if an operation is a direct child of a block (not inside nested
/// regions)
static bool isDirectChildOf(Operation *op, Block *block) {
  return op->getBlock() == block;
}

/// Detect reduction pattern in a for loop
static std::optional<ReductionPattern>
detectReductionPattern(scf::ForOp forOp) {
  ReductionPattern pattern;
  pattern.forOp = forOp;

  Block *body = forOp.getBody();

  /// Collect loads and stores that are DIRECT children of the loop body
  /// (not inside nested regions - those need to be transformed first)
  SmallVector<Operation *> loads, stores;
  for (Operation &op : body->without_terminator()) {
    if (isa<memref::LoadOp, polygeist::DynLoadOp>(&op))
      loads.push_back(&op);
    else if (isa<memref::StoreOp, polygeist::DynStoreOp>(&op))
      stores.push_back(&op);
  }

  /// For each store, check if there's a matching load with loop-invariant
  /// indices
  for (Operation *storeOp : stores) {
    auto [storeMemref, storeIndices] = getStoreInfo(storeOp);
    if (!storeMemref || storeIndices.empty())
      continue;

    /// Check if indices are loop-invariant
    if (!areIndicesLoopInvariant(storeIndices, forOp))
      continue;

    Value storedValue = getStoredValue(storeOp);
    if (!storedValue)
      continue;

    /// Check if stored value comes from an accumulation operation
    Operation *accOp = storedValue.getDefiningOp();
    if (!accOp)
      continue;

    /// The accumulation op must also be a direct child of the loop body
    if (!isDirectChildOf(accOp, body))
      continue;

    if (!isAccumulationOp(accOp))
      continue;

    /// Check if one operand of the accumulation is loaded from the same
    /// location
    for (Value operand : accOp->getOperands()) {
      Operation *loadOp = operand.getDefiningOp();
      if (!loadOp)
        continue;

      auto [loadMemref, loadIndices] = getLoadInfo(loadOp);
      if (!loadMemref)
        continue;

      /// Check if load is from the same location as store.
      /// Also ensure the load has only one use (the accumulation op)
      /// to avoid dangling references when erasing the old loop.
      /// Additionally, the memref must be defined OUTSIDE the loop so that
      /// we can create new loads/stores outside the loop that use it.
      if (loadMemref == storeMemref &&
          ValueAnalysis::areValueRangesIdentical(loadIndices, storeIndices) &&
          loadOp->hasOneUse() &&
          isLoopInvariant(forOp, storeMemref)) {
        pattern.loadOp = loadOp;
        pattern.storeOp = storeOp;
        pattern.valueToYield = storedValue;
        pattern.memref = storeMemref;
        pattern.indices = storeIndices;

        ARTS_DEBUG("Found reduction pattern in loop at " << forOp.getLoc());

        return pattern;
      }
    }
  }

  return std::nullopt;
}

/// Rank-0 scratch memrefs often survive outlining even when they are used like
/// pure SSA temporaries. Forward same-block stores to later loads so the
/// outlined worker body can stay register-based.
static bool isPromotableScalarScratch(memref::AllocaOp allocaOp) {
  auto type = allocaOp.getType();
  if (type.getRank() != 0)
    return false;
  Type elemType = type.getElementType();
  if (!elemType.isIntOrIndexOrFloat())
    return false;

  for (Operation *user : allocaOp->getUsers()) {
    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      if (!load.getIndices().empty())
        return false;
      continue;
    }
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (!store.getIndices().empty())
        return false;
      continue;
    }
    return false;
  }
  return true;
}

static unsigned forwardScalarScratchLoadsInBlock(Block &block) {
  DenseMap<Value, Value> availableValueByScratch;
  unsigned forwardedLoads = 0;

  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (auto store = dyn_cast<memref::StoreOp>(&op)) {
      auto allocaOp = store.getMemRef().getDefiningOp<memref::AllocaOp>();
      if (allocaOp && isPromotableScalarScratch(allocaOp)) {
        availableValueByScratch[allocaOp.getResult()] = store.getValue();
        continue;
      }
    }

    if (auto load = dyn_cast<memref::LoadOp>(&op)) {
      auto allocaOp = load.getMemRef().getDefiningOp<memref::AllocaOp>();
      if (allocaOp && isPromotableScalarScratch(allocaOp)) {
        auto it = availableValueByScratch.find(allocaOp.getResult());
        if (it != availableValueByScratch.end()) {
          load.getResult().replaceAllUsesWith(it->second);
          load.erase();
          ++forwardedLoads;
          continue;
        }
      }
    }

    for (Region &region : op.getRegions())
      for (Block &nestedBlock : region)
        forwardedLoads += forwardScalarScratchLoadsInBlock(nestedBlock);

    // Invalidate any forwarded values for allocas that may be modified
    // inside the nested regions (e.g., loop-carried accumulators).
    if (!op.getRegions().empty()) {
      SmallVector<Value, 4> toErase;
      for (auto &entry : availableValueByScratch) {
        Value allocaVal = entry.first;
        for (Region &region : op.getRegions()) {
          bool modified = false;
          region.walk([&](memref::StoreOp store) {
            if (store.getMemRef() == allocaVal)
              modified = true;
          });
          if (modified) {
            toErase.push_back(allocaVal);
            break;
          }
        }
      }
      for (Value v : toErase)
        availableValueByScratch.erase(v);
    }
  }

  return forwardedLoads;
}

static unsigned forwardScalarScratchLoads(ModuleOp module) {
  unsigned forwardedLoads = 0;
  module.walk([&](func::FuncOp funcOp) {
    for (Block &block : funcOp.getBlocks())
      forwardedLoads += forwardScalarScratchLoadsInBlock(block);
  });
  return forwardedLoads;
}

static unsigned eraseDeadScalarScratchAllocas(ModuleOp module) {
  llvm::SetVector<memref::AllocaOp> deadAllocas;
  module.walk([&](memref::AllocaOp allocaOp) {
    if (!isPromotableScalarScratch(allocaOp))
      return;

    bool hasLoadUsers = llvm::any_of(allocaOp->getUsers(), [](Operation *user) {
      return isa<memref::LoadOp>(user);
    });
    if (!hasLoadUsers)
      deadAllocas.insert(allocaOp);
  });

  for (memref::AllocaOp allocaOp : deadAllocas) {
    SmallVector<Operation *> stores;
    for (Operation *user : allocaOp->getUsers())
      stores.push_back(user);
    for (Operation *store : stores)
      store->erase();
    allocaOp.erase();
  }

  return deadAllocas.size();
}

/// Transform a reduction pattern to use iter_args
static LogicalResult transformReduction(ReductionPattern &pattern,
                                        IRRewriter &rewriter) {
  scf::ForOp forOp = pattern.forOp;
  Location loc = forOp.getLoc();

  /// Load initial value before the loop
  rewriter.setInsertionPoint(forOp);

  Value initValue;
  if (auto load = dyn_cast<memref::LoadOp>(pattern.loadOp)) {
    initValue = memref::LoadOp::create(rewriter, loc, load.getMemRef(),
                                       load.getIndices());
  } else if (auto dynLoad = dyn_cast<polygeist::DynLoadOp>(pattern.loadOp)) {
    /// For polygeist.load (DynLoadOp), create new load before the loop
    initValue = polygeist::DynLoadOp::create(
        rewriter, loc, dynLoad.getResult().getType(), dynLoad.getMemref(),
        dynLoad.getIndices(), dynLoad.getSizes());
  }
  Operation *initOp = initValue ? initValue.getDefiningOp() : nullptr;

  /// Create new ForOp with iter_args
  SmallVector<Value> iterArgs = {initValue};
  auto newForOp =
      scf::ForOp::create(rewriter, loc, forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), iterArgs);

  /// Copy attributes (like arts.loop metadata)
  newForOp->setAttrs(forOp->getAttrs());

  /// Get the accumulator iter_arg
  Value accArg = newForOp.getRegionIterArg(0);

  /// Clone the loop body, replacing load with iter_arg
  Block *oldBody = forOp.getBody();
  Block *newBody = newForOp.getBody();

  /// Map old values to new values
  IRMapping mapper;
  mapper.map(forOp.getInductionVar(), newForOp.getInductionVar());
  mapper.map(pattern.loadOp->getResult(0), accArg);

  /// Erase the auto-generated yield terminator.
  if (!newBody->empty()) {
    Operation *existingTerminator = &newBody->back();
    if (existingTerminator &&
        existingTerminator->hasTrait<OpTrait::IsTerminator>())
      rewriter.eraseOp(existingTerminator);
  }

  rewriter.setInsertionPointToEnd(newBody);

  /// Clone operations except the load and store.
  for (Operation &op : oldBody->without_terminator()) {
    if (&op == pattern.loadOp)
      continue; /// Skip the load - replaced by iter_arg
    if (&op == pattern.storeOp)
      continue; /// Skip the store - replaced by yield

    rewriter.clone(op, mapper);
  }

  /// Create yield with the accumulated value.
  Value newAccValue = mapper.lookupOrDefault(pattern.valueToYield);
  if (!newAccValue || newAccValue == pattern.valueToYield) {
    /// This can happen if the stored value was not cloned into the new loop body.
    /// Keep the failure path transactional: remove the speculative loop and
    /// its preheader load before giving another pattern a chance to match.
    scf::YieldOp::create(rewriter, loc, ValueRange{accArg});
    rewriter.setInsertionPointAfter(newForOp);
    rewriter.eraseOp(newForOp);
    if (initOp && initOp->use_empty())
      rewriter.eraseOp(initOp);
    return failure();
  }
  scf::YieldOp::create(rewriter, loc, ValueRange{newAccValue});

  /// Store final result after the loop
  rewriter.setInsertionPointAfter(newForOp);
  Value finalValue = newForOp.getResult(0);

  if (auto store = dyn_cast<memref::StoreOp>(pattern.storeOp)) {
    memref::StoreOp::create(rewriter, loc, finalValue, store.getMemRef(),
                            store.getIndices());
  } else if (auto dynStore = dyn_cast<polygeist::DynStoreOp>(pattern.storeOp)) {
    /// For polygeist.store (DynStoreOp), create new store with updated value
    polygeist::DynStoreOp::create(rewriter, loc, finalValue,
                                  dynStore.getMemref(), dynStore.getIndices(),
                                  dynStore.getSizes());
  }

  /// Erase the old loop
  rewriter.eraseOp(forOp);

  ARTS_INFO("Transformed reduction pattern to iter_args at " << loc);

  return success();
}

struct ScalarReplacementPass
    : public arts_rt::impl::ScalarReplacementBase<ScalarReplacementPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    ARTS_INFO_HEADER(ScalarReplacementPass);

    int transformedReductions = 0;

    /// Walk all scf.for loops and try to transform reductions
    /// Do this iteratively as transformations change the IR
    bool changed = true;
    while (changed) {
      changed = false;

      SmallVector<scf::ForOp> forOps;
      module.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

      for (scf::ForOp forOp : forOps) {
        /// Skip loops that already have iter_args
        if (!forOp.getInitArgs().empty())
          continue;

        auto pattern = detectReductionPattern(forOp);
        if (!pattern)
          continue;

        /// Transform the pattern
        IRRewriter rewriter(module.getContext());
        if (succeeded(transformReduction(*pattern, rewriter))) {
          transformedReductions++;
          changed = true;
          break; /// Start over after transformation
        }
      }
    }

    unsigned forwardedScratchLoads = forwardScalarScratchLoads(module);
    unsigned erasedScratchAllocas = eraseDeadScalarScratchAllocas(module);

    ARTS_INFO("Transformed " << transformedReductions << " reduction patterns");
    ARTS_INFO("Forwarded " << forwardedScratchLoads
                           << " scalar scratch loads and erased "
                           << erasedScratchAllocas << " scratch allocas");
    ARTS_INFO_FOOTER(ScalarReplacementPass);
  }
};

} // namespace

///===----------------------------------------------------------------------===///
/// Pass creation and registration
///===----------------------------------------------------------------------===///
namespace mlir {
namespace carts::arts_rt {

std::unique_ptr<Pass> createScalarReplacementPass() {
  return std::make_unique<ScalarReplacementPass>();
}

} // namespace carts::arts_rt
} // namespace mlir
