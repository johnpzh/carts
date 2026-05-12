///==========================================================================///
/// File: EpochOptSupport.cpp
///
/// Shared local helpers for the split EpochOpt implementation.
///==========================================================================///

#include "arts/dialect/core/Transforms/EpochOptInternal.h"

using namespace mlir;
using namespace mlir::arts;

namespace mlir::arts::epoch_opt {
namespace {
constexpr llvm::StringLiteral kAsyncLoopChainPrefix = "async_loop_";
constexpr llvm::StringLiteral kContinuationChainPrefix = "chain_";

bool isCarryRematerializableOp(Operation *op) {
  if (!op)
    return false;

  if (isSideEffectFreeArithmeticLikeOp(op) || mlir::isPure(op))
    return true;

  return isa<memref::LoadOp, memref::CastOp, memref::SubViewOp,
             polygeist::Pointer2MemrefOp, polygeist::Memref2PointerOp,
             LLVM::PtrToIntOp, LLVM::IntToPtrOp>(op);
}

void rematerializeSequentialOpDefs(OpBuilder &builder, Operation *op,
                                   Block &loopBody, IRMapping &mapping) {
  SmallVector<Value> worklist(op->getOperands().begin(), op->getOperands().end());
  SmallVector<Operation *> toClone;
  DenseSet<Value> visitedValues;
  DenseSet<Operation *> scheduled;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visitedValues.insert(current).second ||
        mapping.contains(current) || isa<BlockArgument>(current))
      continue;

    Operation *defOp = current.getDefiningOp();
    if (!defOp || !loopBody.findAncestorOpInBlock(*defOp))
      continue;
    if (!isCarryRematerializableOp(defOp))
      continue;
    if (!scheduled.insert(defOp).second)
      continue;

    toClone.push_back(defOp);
    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }

  for (auto it = toClone.rbegin(); it != toClone.rend(); ++it) {
    Operation *defOp = *it;
    Operation *cloned = builder.clone(*defOp, mapping);
    for (auto [oldRes, newRes] :
         llvm::zip(defOp->getResults(), cloned->getResults()))
      mapping.map(oldRes, newRes);
  }
}

Value rematerializeCarryValueAtAdvanceSite(OpBuilder &builder, Value value,
                                           Block &loopBody,
                                           IRMapping &mapping) {
  if (!value)
    return value;
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  if (isa<BlockArgument>(value))
    return value;

  SmallVector<Operation *> toClone;
  SmallVector<Value> worklist{value};
  DenseSet<Value> visitedValues;
  DenseSet<Operation *> scheduled;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visitedValues.insert(current).second)
      continue;
    if (mapping.contains(current) || isa<BlockArgument>(current))
      continue;

    Operation *defOp = current.getDefiningOp();
    if (!defOp || loopBody.findAncestorOpInBlock(*defOp))
      continue;
    if (!isCarryRematerializableOp(defOp))
      continue;
    if (!scheduled.insert(defOp).second)
      continue;

    toClone.push_back(defOp);
    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }

  for (auto it = toClone.rbegin(); it != toClone.rend(); ++it) {
    Operation *defOp = *it;
    Operation *cloned = builder.clone(*defOp, mapping);
    for (auto [oldRes, newRes] :
         llvm::zip(defOp->getResults(), cloned->getResults()))
      mapping.map(oldRes, newRes);
  }

  return mapping.lookupOrDefault(value);
}

bool isCpsExcludedDepPattern(ArtsDepPattern pattern) {
  switch (pattern) {
  case ArtsDepPattern::wavefront_2d:
    return true;
  default:
    return false;
  }
}

bool hasRelaunchSoundJacobiContract(Operation *op) {
  if (!op)
    return false;
  auto pattern = getEffectiveDepPattern(op);
  if (!pattern || *pattern != ArtsDepPattern::jacobi_alternating_buffers)
    return false;
  if (auto soundAttr = op->getAttrOfType<BoolAttr>(
          AttrNames::Operation::Proof::RelaunchStateSoundness))
    return soundAttr.getValue();
  return false;
}

bool loopContainsExcludedDepPattern(
    scf::ForOp forOp, function_ref<bool(ArtsDepPattern)> isExcluded) {
  if (!forOp)
    return false;

  auto matchesExcludedPattern = [&](Operation *op) {
    auto pattern = getEffectiveDepPattern(op);
    return pattern && isExcluded(*pattern);
  };

  if (matchesExcludedPattern(forOp.getOperation()))
    return true;

  bool found = false;
  forOp.walk([&](Operation *inner) {
    if (!matchesExcludedPattern(inner))
      return WalkResult::advance();
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

} // namespace

llvm::StringRef asyncLoopStrategyToString(EpochAsyncLoopStrategy strategy) {
  switch (strategy) {
  case EpochAsyncLoopStrategy::None:
    return "none";
  case EpochAsyncLoopStrategy::AdvanceEdt:
    return "advance_edt";
  case EpochAsyncLoopStrategy::CpsChain:
    return "cps_chain";
  }
  return "none";
}

llvm::StringRef
asyncLoopStrategyToPlanAttrString(EpochAsyncLoopStrategy strategy) {
  switch (strategy) {
  case EpochAsyncLoopStrategy::AdvanceEdt:
    return "advance_edt";
  case EpochAsyncLoopStrategy::CpsChain:
    return "cps_chain";
  case EpochAsyncLoopStrategy::None:
    return "blocking";
  }
  return "blocking";
}

bool loopContainsCpsChainExcludedDepPattern(scf::ForOp forOp) {
  if (!forOp)
    return false;

  bool sawJacobiPattern = false;
  bool sawRelaunchSoundJacobi = false;
  bool foundExcludedPattern = false;

  auto inspectOp = [&](Operation *op) {
    auto pattern = getEffectiveDepPattern(op);
    if (!pattern)
      return WalkResult::advance();
    if (*pattern == ArtsDepPattern::wavefront_2d) {
      foundExcludedPattern = true;
      return WalkResult::interrupt();
    }
    if (*pattern != ArtsDepPattern::jacobi_alternating_buffers)
      return WalkResult::advance();

    sawJacobiPattern = true;
    if (hasRelaunchSoundJacobiContract(op))
      sawRelaunchSoundJacobi = true;
    return WalkResult::advance();
  };

  if (inspectOp(forOp.getOperation()).wasInterrupted())
    return true;
  forOp.walk([&](Operation *inner) { return inspectOp(inner); });

  if (foundExcludedPattern)
    return true;
  if (!sawJacobiPattern)
    return false;
  return !sawRelaunchSoundJacobi;
}

bool loopContainsCpsDriverExcludedDepPattern(scf::ForOp forOp) {
  return loopContainsExcludedDepPattern(forOp, isCpsExcludedDepPattern);
}

std::string makeAsyncLoopChainId(Operation *op) {
  return std::string(kAsyncLoopChainPrefix.data()) +
         std::to_string(reinterpret_cast<uintptr_t>(op));
}

std::string makeContinuationChainId(Operation *ownerOp, unsigned chainIdx) {
  return makeAsyncLoopChainId(ownerOp) + "_" +
         std::string(kContinuationChainPrefix.data()) +
         std::to_string(chainIdx);
}

bool isSlotOp(Operation *op, ArrayRef<EpochSlot> slots) {
  for (const EpochSlot &slot : slots) {
    if (op == slot.epoch)
      return true;
    if (slot.wrappingIf && op == slot.wrappingIf)
      return true;
  }
  return false;
}

void markClonedSlotEpochs(OpBuilder &attrBuilder, scf::IfOp ifOp) {
  for (Operation &op : ifOp.getThenRegion().front())
    if (auto epoch = dyn_cast<EpochOp>(op))
      epoch->setAttr(ContinuationForEpoch, attrBuilder.getUnitAttr());
  if (!ifOp.getElseRegion().empty()) {
    for (Operation &op : ifOp.getElseRegion().front())
      if (auto epoch = dyn_cast<EpochOp>(op))
        epoch->setAttr(ContinuationForEpoch, attrBuilder.getUnitAttr());
  }
}

Operation *cloneEpochSlot(OpBuilder &slotBuilder, EpochSlot slot,
                          IRMapping &mapping) {
  if (slot.wrappingIf) {
    Operation *clonedIfOp =
        slotBuilder.clone(*slot.wrappingIf.getOperation(), mapping);
    markClonedSlotEpochs(slotBuilder, cast<scf::IfOp>(clonedIfOp));
    return clonedIfOp;
  }

  Operation *clonedEpochOp =
      slotBuilder.clone(*slot.epoch.getOperation(), mapping);
  cast<EpochOp>(clonedEpochOp)
      ->setAttr(ContinuationForEpoch, slotBuilder.getUnitAttr());
  return clonedEpochOp;
}

void cloneNonSlotArith(OpBuilder &builder, Block &body,
                       MutableArrayRef<EpochSlot> slots, IRMapping &mapping,
                       ArrayRef<Operation *> sequentialOps,
                       ArrayRef<Operation *> eagerSequentialOps,
                       bool restrictEagerSequentialOps) {
  DenseSet<Operation *> seqOpSet(sequentialOps.begin(), sequentialOps.end());
  DenseSet<Operation *> eagerSeqOpSet;
  if (restrictEagerSequentialOps)
    eagerSeqOpSet.insert(eagerSequentialOps.begin(), eagerSequentialOps.end());
  else if (eagerSequentialOps.empty())
    eagerSeqOpSet.insert(sequentialOps.begin(), sequentialOps.end());
  else
    eagerSeqOpSet.insert(eagerSequentialOps.begin(), eagerSequentialOps.end());

  auto operandsReady = [&](Operation &op) {
    for (Value operand : op.getOperands()) {
      if (!operand || mapping.contains(operand) || isa<BlockArgument>(operand))
        continue;
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      if (body.findAncestorOpInBlock(*defOp))
        return false;
    }
    return true;
  };

  // Helper: ensure an operand's defining sequential op is in the mapping.
  // When a non-slot op uses a result produced by a sequential op that was
  // skipped, the clone would reference the original (out-of-scope) value.
  // Eagerly clone only the sequential ops that belong to this prefix. Other
  // sidecar loads/stores must not be pulled before their source-order stores.
  // This is recursive: if a sequential op itself depends on another sequential
  // op, we clone the dependency first.
  std::function<void(Operation &)> ensureSeqOpsMapped = [&](Operation &op) {
    for (Value operand : op.getOperands()) {
      if (mapping.contains(operand))
        continue;
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || !seqOpSet.contains(defOp) || !eagerSeqOpSet.contains(defOp))
        continue;
      // Already mapped — skip.
      bool alreadyMapped = llvm::all_of(defOp->getResults(), [&](Value res) {
        return mapping.contains(res);
      });
      if (alreadyMapped)
        continue;
      // Recursively ensure this sequential op's own dependencies are mapped.
      ensureSeqOpsMapped(*defOp);
      Operation *cloned = builder.clone(*defOp, mapping);
      for (auto [oldRes, newRes] :
           llvm::zip(defOp->getResults(), cloned->getResults()))
        mapping.map(oldRes, newRes);
    }
  };

  for (Operation &op : body.without_terminator()) {
    if (isSlotOp(&op, slots))
      continue;
    if (seqOpSet.contains(&op))
      continue;
    ensureSeqOpsMapped(op);
    if (!operandsReady(op))
      continue;
    Operation *cloned = builder.clone(op, mapping);
    for (auto [oldRes, newRes] :
         llvm::zip(op.getResults(), cloned->getResults()))
      mapping.map(oldRes, newRes);
  }
}

void ensureYieldTerminator(Block &block, Location loc) {
  if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>()) {
    OpBuilder endBuilder = OpBuilder::atBlockEnd(&block);
    YieldOp::create(endBuilder, loc);
  }
}

Operation *getLastNonTerminatorOp(Block &block) {
  if (block.empty())
    return nullptr;

  Operation *lastOp = &block.back();
  return lastOp->hasTrait<OpTrait::IsTerminator>() ? lastOp->getPrevNode()
                                                   : lastOp;
}

void emitAdvanceLogic(OpBuilder &builder, Location loc, Value iv,
                      Value tCurrent, Value ub, Value step, Block &body,
                      MutableArrayRef<EpochSlot> slots, StringRef targetChainId,
                      ArrayRef<Operation *> prefixSequentialOps,
                      ArrayRef<Operation *> allSequentialOps,
                      ArrayRef<Value> loopBackParams,
                      const IRMapping *seedMapping) {
  Value tNext = arith::AddIOp::create(builder, loc, tCurrent, step);
  IRMapping advanceMapping;
  if (seedMapping)
    advanceMapping = *seedMapping;
  advanceMapping.map(iv, tNext);
  for (Operation *seqOp : prefixSequentialOps) {
    rematerializeSequentialOpDefs(builder, seqOp, body, advanceMapping);
    Operation *cloned = builder.clone(*seqOp, advanceMapping);
    for (auto [oldRes, newRes] :
         llvm::zip(seqOp->getResults(), cloned->getResults()))
      advanceMapping.map(oldRes, newRes);
  }
  cloneNonSlotArith(builder, body, slots, advanceMapping, allSequentialOps,
                    prefixSequentialOps, /*restrictEagerSequentialOps=*/true);

  auto i64Ty = IntegerType::get(builder.getContext(), 64);
  Value stepI64 = arith::IndexCastOp::create(builder, loc, i64Ty, step);
  Value ubI64 = arith::IndexCastOp::create(builder, loc, i64Ty, ub);
  SmallVector<Value> nextParams = {stepI64, ubI64};
  for (Value v : loopBackParams)
    nextParams.push_back(
        rematerializeCarryValueAtAdvanceSite(builder, v, body, advanceMapping));
  auto advance = CPSAdvanceOp::create(builder, loc, nextParams,
                                      builder.getStringAttr(targetChainId));
  advance->setAttr(CPSAdditiveParams, builder.getUnitAttr());
  advance->setAttr(CPSNumCarry,
                   builder.getI64IntegerAttr(loopBackParams.size()));

  Region &advanceRegion = advance.getEpochBody();
  if (advanceRegion.empty())
    advanceRegion.emplaceBlock();
  Block &advBlock = advanceRegion.front();
  ensureYieldTerminator(advBlock, loc);
  OpBuilder advBuilder = OpBuilder::atBlockTerminator(&advBlock);

  if (slots[0].wrappingIf) {
    auto indexTy = IndexType::get(builder.getContext());
    Value ivPlaceholder = advBlock.addArgument(indexTy, loc);
    advance->setAttr(CPSAdvanceHasIvArg, builder.getUnitAttr());

    IRMapping ifMapping;
    ifMapping.map(iv, ivPlaceholder);
    for (Operation &op : body.without_terminator()) {
      if (isSlotOp(&op, slots))
        continue;
      Operation *cloned = advBuilder.clone(op, ifMapping);
      for (auto [oldRes, newRes] :
           llvm::zip(op.getResults(), cloned->getResults()))
        ifMapping.map(oldRes, newRes);
    }

    cloneEpochSlot(advBuilder, slots[0], ifMapping);
    return;
  }

  Block &epochBody = slots[0].epoch.getBody().front();
  for (Operation &epochOp : epochBody.without_terminator()) {
    Operation *cloned = advBuilder.clone(epochOp, advanceMapping);
    for (auto [oldRes, newRes] :
         llvm::zip(epochOp.getResults(), cloned->getResults()))
      advanceMapping.map(oldRes, newRes);
  }
}

void markAsContinuation(EdtOp edt, OpBuilder &builder, Operation *ownerOp,
                        unsigned chainIdx) {
  std::string chainId = makeContinuationChainId(ownerOp, chainIdx);
  edt->setAttr(ControlDep, builder.getIntegerAttr(builder.getI32Type(), 1));
  edt->setAttr(ContinuationForEpoch, builder.getUnitAttr());
  edt->setAttr(CPSChainId, builder.getStringAttr(chainId));
  edt->setAttr(CPSLoopContinuation, builder.getUnitAttr());
}

void normalizeAsyncLoopPlanAttrs(scf::ForOp forOp,
                                 EpochAsyncLoopStrategy strategy) {
  if (!forOp)
    return;

  StringRef normalizedStrategy = asyncLoopStrategyToPlanAttrString(strategy);
  MLIRContext *ctx = forOp.getContext();
  StringAttr normalizedAttr = StringAttr::get(ctx, normalizedStrategy);

  if (forOp->hasAttr(AsyncStrategy))
    forOp->setAttr(AsyncStrategy, normalizedAttr);

  forOp.walk([&](Operation *op) {
    if (op == forOp.getOperation())
      return WalkResult::advance();
    if (!op->hasAttr(AsyncStrategy))
      return WalkResult::advance();
    op->setAttr(AsyncStrategy, normalizedAttr);
    return WalkResult::advance();
  });
}

void copyNormalizedPlanAttrs(Operation *source, Operation *dest,
                             EpochAsyncLoopStrategy strategy) {
  if (!source || !dest)
    return;
  copyPlanAttrs(source, dest);
  if (!source->hasAttr(AsyncStrategy))
    return;
  dest->setAttr(AsyncStrategy,
                StringAttr::get(dest->getContext(),
                                asyncLoopStrategyToPlanAttrString(strategy)));
}

} // namespace mlir::arts::epoch_opt
