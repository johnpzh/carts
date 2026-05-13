///==========================================================================///
/// File: EpochOptCpsChain.cpp
///
/// CPS-chain epoch loop transform used by EpochOpt.
///==========================================================================///

#include "arts/dialect/core/Transforms/EpochOptInternal.h"
#include "arts/utils/Debug.h"

ARTS_DEBUG_SETUP(epoch_opt);

using namespace mlir;
using namespace mlir::arts;

namespace mlir::arts::epoch_opt {
namespace {

void expandSequentialCloneOps(Block &body, MutableArrayRef<EpochSlot> slots,
                              SmallVectorImpl<Operation *> &ops) {
  DenseSet<Operation *> cloneSet(ops.begin(), ops.end());
  SmallVector<Operation *> worklist(ops.begin(), ops.end());

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || cloneSet.contains(defOp))
        continue;
      if (defOp->getBlock() != &body || isSlotOp(defOp, slots) ||
          !isSideEffectFreeArithmeticLikeOp(defOp)) {
        continue;
      }
      cloneSet.insert(defOp);
      worklist.push_back(defOp);
    }
  }

  ops.clear();
  for (Operation &op : body.without_terminator())
    if (cloneSet.contains(&op))
      ops.push_back(&op);
}

void rematerializeExternalDefs(OpBuilder &builder, Operation *op,
                               Block *loopBody, IRMapping &mapping) {
  SmallVector<Operation *> toClone;
  SmallVector<Value> worklist;
  DenseSet<Value> visited;
  DenseSet<Operation *> scheduled;

  for (Value operand : op->getOperands())
    worklist.push_back(operand);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;
    if (mapping.contains(v))
      continue;
    if (isa<BlockArgument>(v))
      continue;
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      continue;
    if (loopBody->findAncestorOpInBlock(*defOp))
      continue;
    bool clonableReadOnlyDef =
        isa<memref::LoadOp, memref::CastOp, memref::SubViewOp>(defOp);
    if (!isSideEffectFreeArithmeticLikeOp(defOp) && !mlir::isPure(defOp) &&
        !clonableReadOnlyDef)
      continue;
    if (!scheduled.insert(defOp).second)
      continue;
    toClone.push_back(defOp);
    for (Value inner : defOp->getOperands())
      worklist.push_back(inner);
  }

  for (auto it = toClone.rbegin(); it != toClone.rend(); ++it) {
    Operation *defOp = *it;
    Operation *cloned = builder.clone(*defOp, mapping);
    for (auto [oldRes, newRes] :
         llvm::zip(defOp->getResults(), cloned->getResults()))
      mapping.map(oldRes, newRes);
  }
}

void promoteAllocsForCPSChain(scf::ForOp forOp,
                              ArrayRef<Operation *> sequentialOps,
                              MutableArrayRef<EpochSlot> slots) {
  DenseSet<Operation *> promotable;

  auto traceToAllocLike = [](Value v) -> Operation * {
    while (v) {
      if (auto alloc = v.getDefiningOp<memref::AllocOp>())
        return alloc.getOperation();
      if (auto alloca = v.getDefiningOp<memref::AllocaOp>())
        return alloca.getOperation();
      if (auto cast = v.getDefiningOp<memref::CastOp>()) {
        v = cast.getSource();
        continue;
      }
      if (auto subview = v.getDefiningOp<memref::SubViewOp>()) {
        v = subview.getSource();
        continue;
      }
      break;
    }
    return nullptr;
  };

  for (Operation *seqOp : sequentialOps) {
    for (Value operand : seqOp->getOperands()) {
      Operation *allocOp = traceToAllocLike(operand);
      if (!allocOp)
        continue;
      if (forOp.getBody()->findAncestorOpInBlock(*allocOp))
        continue;
      promotable.insert(allocOp);
    }
  }

  if (promotable.empty())
    return;

  for (Operation *allocLikeOp : promotable) {
    auto allocOp = dyn_cast<memref::AllocOp>(allocLikeOp);
    auto allocaOp = dyn_cast<memref::AllocaOp>(allocLikeOp);
    if (!allocOp && !allocaOp)
      continue;

    Location loc = allocLikeOp->getLoc();
    OpBuilder builder(allocLikeOp);

    Value allocValue = allocOp ? allocOp.getResult() : allocaOp.getResult();
    MemRefType memRefType = cast<MemRefType>(allocValue.getType());
    Type elementType = memRefType.getElementType();
    SmallVector<Value> dynamicSizes;
    if (allocOp)
      dynamicSizes.append(allocOp.getDynamicSizes().begin(),
                          allocOp.getDynamicSizes().end());
    else
      dynamicSizes.append(allocaOp.getDynamicSizes().begin(),
                          allocaOp.getDynamicSizes().end());

    unsigned rank = std::max<unsigned>(1, memRefType.getRank());
    SmallVector<Value> elementSizes;
    elementSizes.reserve(rank);
    unsigned dynamicIdx = 0;
    for (unsigned i = 0; i < static_cast<unsigned>(memRefType.getRank()); ++i) {
      if (memRefType.isDynamicDim(i)) {
        elementSizes.push_back(dynamicSizes[dynamicIdx++]);
      } else {
        elementSizes.push_back(arith::ConstantIndexOp::create(
            builder, loc, memRefType.getDimSize(i)));
      }
    }
    if (memRefType.getRank() == 0)
      elementSizes.push_back(arith::ConstantIndexOp::create(builder, loc, 1));
    SmallVector<Value> sizes = {
        arith::ConstantIndexOp::create(builder, loc, 1)};

    Value route = arith::ConstantOp::create(builder, loc, builder.getI32Type(),
                                            builder.getI32IntegerAttr(0));

    // For CPS-promoted allocs (single partition), use static shape.
    // The DB stores whole memrefs, so elementType is the memref itself.
    auto ptrType = MemRefType::get({1}, memRefType);
    auto dbAlloc = DbAllocOp::create(builder, loc, ArtsMode::inout, route,
                                     DbAllocType::heap, DbMode::write,
                                     memRefType, ptrType, sizes, elementSizes);
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value dbView = DbRefOp::create(builder, loc, dbAlloc.getPtr(),
                                   SmallVector<Value>{zero});
    Value replacement = dbView;
    if (replacement.getType() != allocValue.getType())
      replacement =
          memref::CastOp::create(builder, loc, allocValue.getType(), dbView);

    SmallVector<Operation *> deallocOps;
    if (allocOp) {
      for (Operation *user : allocValue.getUsers())
        if (isa<memref::DeallocOp>(user))
          deallocOps.push_back(user);
    }

    allocValue.replaceAllUsesWith(replacement);

    OpBuilder afterLoop(forOp->getNextNode());
    DbFreeOp::create(afterLoop, loc, dbAlloc.getGuid());
    DbFreeOp::create(afterLoop, loc, dbAlloc.getPtr());

    for (Operation *deallocOp : deallocOps)
      deallocOp->erase();

    ARTS_INFO("CPS Chain: Promoted loop-external memref storage to DbAllocOp "
              "at "
              << loc);
    allocLikeOp->erase();
  }
}

Operation *cloneWithResultMapping(OpBuilder &builder, Operation &op,
                                  IRMapping &mapping);

void cloneSequentialOpsWithExternalDefs(OpBuilder &builder,
                                        ArrayRef<Operation *> ops,
                                        Block *loopBody, IRMapping &mapping) {
  for (Operation *op : ops) {
    rematerializeExternalDefs(builder, op, loopBody, mapping);
    cloneWithResultMapping(builder, *op, mapping);
  }
}

Operation *cloneWithResultMapping(OpBuilder &builder, Operation &op,
                                  IRMapping &mapping) {
  Operation *cloned = builder.clone(op, mapping);
  for (auto [oldRes, newRes] : llvm::zip(op.getResults(), cloned->getResults()))
    mapping.map(oldRes, newRes);
  return cloned;
}

void clearAdvanceSite(Block *block, Operation *anchor) {
  SmallVector<Operation *> opsToErase;
  for (Operation *op = anchor ? anchor->getNextNode() : &block->front();
       op && !op->hasTrait<OpTrait::IsTerminator>(); op = op->getNextNode()) {
    opsToErase.push_back(op);
  }
  for (Operation *op : llvm::reverse(opsToErase))
    op->erase();
}

bool continuationCreatesNestedAsyncWork(EdtOp contEdt) {
  bool createsNestedAsyncWork = false;
  contEdt.walk([&](Operation *op) {
    if (op == contEdt.getOperation())
      return WalkResult::advance();
    if (isa<EpochOp, EdtOp>(op)) {
      createsNestedAsyncWork = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return createsNestedAsyncWork;
}

bool isDbRefCast(Operation *defOp) {
  if (isa<DbRefOp>(defOp))
    return true;
  if (auto cast = dyn_cast<memref::CastOp>(defOp))
    return cast.getSource().getDefiningOp<DbRefOp>() != nullptr;
  return false;
}

struct CpsDbInfo {
  DbAllocOp alloc;
  bool isGuid;
};

} // namespace

bool tryCPSChainTransform(scf::ForOp forOp,
                          const EpochAnalysis &epochAnalysis) {
  if (loopContainsCpsChainExcludedDepPattern(forOp)) {
    ARTS_INFO("CPS Chain: skipping loop with specialized dep pattern");
    return false;
  }

  EpochChainDecision chainDecision = epochAnalysis.evaluateCPSChain(forOp);
  if (!chainDecision.eligible) {
    ARTS_DEBUG("CPS Chain: " << chainDecision.rationale << ", skipping");
    return false;
  }

  Block &body = *forOp.getBody();
  SmallVector<EpochSlot> slots = std::move(chainDecision.slots);
  SmallVector<Operation *> sequentialOps =
      std::move(chainDecision.sequentialOps);
  expandSequentialCloneOps(body, slots, sequentialOps);

  Value iv = forOp.getInductionVar();
  unsigned slotCount = slots.size();
  std::string firstChainId = makeContinuationChainId(forOp.getOperation(), 0);

  SmallVector<SmallVector<Operation *>> interEpochOps(slotCount);
  SmallVector<Operation *> tailOps;
  if (!sequentialOps.empty()) {
    DenseSet<Operation *> seqOpSet(sequentialOps.begin(), sequentialOps.end());
    SmallVector<Operation *> slotAnchors;
    slotAnchors.reserve(slotCount);
    for (EpochSlot &slot : slots) {
      slotAnchors.push_back(slot.wrappingIf ? slot.wrappingIf.getOperation()
                                            : slot.epoch.getOperation());
    }
    unsigned nextSlot = 0;
    for (Operation &op : body.without_terminator()) {
      if (nextSlot < slotCount && &op == slotAnchors[nextSlot]) {
        ++nextSlot;
        continue;
      }
      if (isSlotOp(&op, slots))
        continue;
      if (seqOpSet.contains(&op)) {
        if (nextSlot >= slotCount)
          tailOps.push_back(&op);
        else
          interEpochOps[nextSlot].push_back(&op);
      }
    }
  }

  if (!sequentialOps.empty())
    promoteAllocsForCPSChain(forOp, sequentialOps, slots);

  ARTS_INFO("CPS Chain: Found eligible "
            << slotCount << "-epoch loop"
            << (sequentialOps.empty() ? "" : " (with sequential ops in body)"));

  OpBuilder builder(forOp->getContext());
  Location loc = forOp.getLoc();
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  Value step = forOp.getStep();

  builder.setInsertionPoint(forOp);
  auto outerEpoch =
      EpochOp::create(builder, loc, IntegerType::get(builder.getContext(), 64));
  outerEpoch->setAttr(ContinuationForEpoch, builder.getUnitAttr());
  outerEpoch->setAttr(AttrNames::Operation::CPSOuterEpoch,
                      builder.getUnitAttr());
  copyNormalizedPlanAttrs(forOp.getOperation(), outerEpoch.getOperation(),
                          EpochAsyncLoopStrategy::CpsChain);

  if (auto lbCst = getConstantIntValue(lb))
    outerEpoch->setAttr(CPSInitIter, builder.getI64IntegerAttr(*lbCst));
  else
    outerEpoch->setAttr(CPSInitIter, builder.getI64IntegerAttr(0));

  Region &outerRegion = outerEpoch.getRegion();
  if (outerRegion.empty())
    outerRegion.emplaceBlock();
  Block &outerBlock = outerRegion.front();
  ensureYieldTerminator(outerBlock, loc);

  OpBuilder outerBuilder = OpBuilder::atBlockTerminator(&outerBlock);
  IRMapping firstIterMapping;
  firstIterMapping.map(iv, lb);
  cloneSequentialOpsWithExternalDefs(outerBuilder, interEpochOps.front(), &body,
                                     firstIterMapping);
  cloneNonSlotArith(outerBuilder, body, slots, firstIterMapping, sequentialOps,
                    interEpochOps.front(), /*restrictEagerSequentialOps=*/true);
  cloneEpochSlot(outerBuilder, slots.front(), firstIterMapping);

  OpBuilder chainBuilder(outerBuilder);
  EdtOp firstContinuation = nullptr;
  SmallVector<std::pair<Block *, Operation *>> advanceSites;

  for (unsigned i = 0; i < slotCount; ++i) {
    auto contEdt =
        EdtOp::create(chainBuilder, loc, EdtType::task,
                      EdtConcurrency::intranode, SmallVector<Value>{});
    markAsContinuation(contEdt, chainBuilder, forOp.getOperation(), i);
    copyNormalizedPlanAttrs(forOp.getOperation(), contEdt.getOperation(),
                            EpochAsyncLoopStrategy::CpsChain);
    if (!firstContinuation)
      firstContinuation = contEdt;

    Block &contBlock = contEdt.getBody().front();
    ensureYieldTerminator(contBlock, loc);
    OpBuilder contBuilder = OpBuilder::atBlockTerminator(&contBlock);

    Value tCurrent = lb;
    IRMapping contMapping;
    contMapping.map(iv, tCurrent);

    if (i == slotCount - 1) {
      cloneSequentialOpsWithExternalDefs(
          contBuilder, interEpochOps[slotCount - 1], &body, contMapping);
      cloneSequentialOpsWithExternalDefs(contBuilder, tailOps, &body,
                                         contMapping);
      SmallVector<Operation *> eagerSeq;
      eagerSeq.append(interEpochOps[slotCount - 1].begin(),
                      interEpochOps[slotCount - 1].end());
      eagerSeq.append(tailOps.begin(), tailOps.end());
      cloneNonSlotArith(contBuilder, body, slots, contMapping, sequentialOps,
                        eagerSeq, /*restrictEagerSequentialOps=*/true);
      advanceSites.push_back({&contBlock, getLastNonTerminatorOp(contBlock)});
      continue;
    }

    cloneSequentialOpsWithExternalDefs(contBuilder, interEpochOps[i + 1], &body,
                                       contMapping);
    cloneNonSlotArith(contBuilder, body, slots, contMapping, sequentialOps,
                      interEpochOps[i + 1],
                      /*restrictEagerSequentialOps=*/true);

    EpochSlot &nextSlot = slots[i + 1];
    if (nextSlot.wrappingIf) {
      auto clonedIf =
          cast<scf::IfOp>(cloneEpochSlot(contBuilder, nextSlot, contMapping));

      if (clonedIf.getElseRegion().empty()) {
        Value ifCond = clonedIf.getCondition();
        OpBuilder preIfBuilder(clonedIf);
        auto newIf =
            scf::IfOp::create(preIfBuilder, loc, ifCond, /*withElse=*/true);

        Block &oldThenBlock = clonedIf.getThenRegion().front();
        Block &newThenBlock = newIf.getThenRegion().front();
        for (Operation &op :
             llvm::make_early_inc_range(oldThenBlock.without_terminator()))
          op.moveBefore(newThenBlock.getTerminator());

        Block &elseBlock = newIf.getElseRegion().front();
        advanceSites.push_back({&elseBlock, getLastNonTerminatorOp(elseBlock)});

        clonedIf.erase();

        Block &finalThenBlock = newIf.getThenRegion().front();
        chainBuilder.setInsertionPoint(finalThenBlock.getTerminator());
      } else {
        Block &elseBlock = clonedIf.getElseRegion().front();
        advanceSites.push_back({&elseBlock, getLastNonTerminatorOp(elseBlock)});

        Block &thenBlock = clonedIf.getThenRegion().front();
        chainBuilder.setInsertionPoint(thenBlock.getTerminator());
      }
      continue;
    }

    Operation *clonedEpochOp =
        contBuilder.clone(*nextSlot.epoch.getOperation(), contMapping);
    cast<EpochOp>(clonedEpochOp)
        ->setAttr(ContinuationForEpoch, contBuilder.getUnitAttr());
    chainBuilder = contBuilder;
  }

  SmallVector<Value> loopBackParams;
  if (firstContinuation) {
    loopBackParams = EdtUtils::collectPackedValues(firstContinuation);
    if (loopBackParams.empty()) {
      llvm::SetVector<Value> originalCaptured;
      llvm::SetVector<Value> parameters, constants, dbHandles;
      getUsedValuesDefinedAbove(forOp.getRegion(), originalCaptured);
      EdtUtils::classifyUserValues(originalCaptured.getArrayRef(), parameters,
                            constants, dbHandles);

      loopBackParams.reserve(originalCaptured.size());
      for (Value param : parameters) {
        if (auto *defOp = param.getDefiningOp())
          if (isUndefLikeOp(defOp))
            continue;
        loopBackParams.push_back(param);
      }
      loopBackParams.append(dbHandles.begin(), dbHandles.end());
    }
  }

  SmallVector<Value> scalarParams;
  SmallVector<std::pair<Value, CpsDbInfo>> timingDbs;
  SmallVector<std::pair<Value, CpsDbInfo>> dataGuids;
  SmallVector<std::pair<Value, CpsDbInfo>> dataPtrs;
  SmallVector<EdtOp> allContinuations;

  if (firstContinuation) {
    outerEpoch.walk([&](EdtOp edt) {
      if (edt->hasAttr(CPSLoopContinuation))
        allContinuations.push_back(edt);
    });
  }

  DbAllocOp scratchAlloc;
  Value scratchGuidScalar;
  unsigned numTimingDbGuids = 0;
  SmallVector<Value> depGuidValues;

  if (firstContinuation && !loopBackParams.empty()) {
    for (Value v : loopBackParams) {
      if (auto *defOp = v.getDefiningOp()) {
        if (auto dbAlloc = dyn_cast<DbAllocOp>(defOp)) {
          bool isGuid = (v == dbAlloc.getGuid());
          int64_t partCount = DbUtils::computeStaticPartitionCount(dbAlloc);
          if (partCount == 1) {
            timingDbs.push_back({v, {dbAlloc, isGuid}});
          } else if (isGuid) {
            dataGuids.push_back({v, {dbAlloc, isGuid}});
          } else {
            dataPtrs.push_back({v, {dbAlloc, isGuid}});
          }
          continue;
        }
        if (isDbRefCast(defOp)) {
          timingDbs.push_back(
              {v,
               {dyn_cast_or_null<DbAllocOp>(DbUtils::getUnderlyingDbAlloc(v)),
                false}});
          continue;
        }
      }
      scalarParams.push_back(v);
    }

    loopBackParams.assign(scalarParams.begin(), scalarParams.end());

    bool hasDbHandles =
        !timingDbs.empty() || !dataGuids.empty() || !dataPtrs.empty();
    if (hasDbHandles) {
      ARTS_INFO("CPS Chain: DB routing — "
                << timingDbs.size() << " timing, " << dataGuids.size()
                << " data GUID arrays, " << dataPtrs.size()
                << " data PTR arrays (dropped), " << scalarParams.size()
                << " scalars");

      OpBuilder preEpochBuilder(outerEpoch);
      Value zero = arith::ConstantIndexOp::create(preEpochBuilder, loc, 0);
      Value one = arith::ConstantIndexOp::create(preEpochBuilder, loc, 1);

      if (!dataGuids.empty()) {
        Value totalGuidSlots =
            arith::ConstantIndexOp::create(preEpochBuilder, loc, 0);

        SmallVector<std::pair<Value, Value>> guidArrayLayout;
        for (auto &[guidValue, info] : dataGuids) {
          Value count =
              DbNumElementsOp::create(preEpochBuilder, loc, info.alloc);
          guidArrayLayout.push_back({totalGuidSlots, count});
          totalGuidSlots = arith::AddIOp::create(preEpochBuilder, loc,
                                                 totalGuidSlots, count);
        }

        Value route = arith::ConstantOp::create(
            preEpochBuilder, loc, preEpochBuilder.getI32Type(),
            preEpochBuilder.getI32IntegerAttr(0));

        auto i64Type = preEpochBuilder.getI64Type();
        auto scratchPtrType = MemRefType::get({ShapedType::kDynamic}, i64Type);
        scratchAlloc = DbAllocOp::create(
            preEpochBuilder, loc, ArtsMode::inout, route, DbAllocType::heap,
            DbMode::write, i64Type, scratchPtrType, SmallVector<Value>{one},
            SmallVector<Value>{totalGuidSlots});

        Value scratchView =
            DbRefOp::create(preEpochBuilder, loc, scratchAlloc.getPtr(),
                            SmallVector<Value>{zero});

        for (unsigned g = 0; g < dataGuids.size(); ++g) {
          Value guidArray = dataGuids[g].first;
          auto [offset, count] = guidArrayLayout[g];

          auto loop = scf::ForOp::create(
              preEpochBuilder, loc,
              arith::ConstantIndexOp::create(preEpochBuilder, loc, 0), count,
              arith::ConstantIndexOp::create(preEpochBuilder, loc, 1));
          OpBuilder loopBody = OpBuilder::atBlockTerminator(loop.getBody());
          Value i = loop.getInductionVar();
          Value guid =
              memref::LoadOp::create(loopBody, loc, guidArray, ValueRange{i});
          Value scratchIdx = arith::AddIOp::create(loopBody, loc, offset, i);
          memref::StoreOp::create(loopBody, loc, guid, scratchView,
                                  ValueRange{scratchIdx});
        }

        scratchGuidScalar = memref::LoadOp::create(
            preEpochBuilder, loc, scratchAlloc.getGuid(), ValueRange{zero});
      }

      SmallVector<Value> depGuidPrefix;
      if (scratchGuidScalar) {
        depGuidPrefix.push_back(scratchGuidScalar);
        depGuidValues.push_back(scratchGuidScalar);
      }

      for (auto &[ptrValue, info] : timingDbs) {
        if (info.isGuid || !info.alloc)
          continue;
        Value guidScalar = memref::LoadOp::create(
            preEpochBuilder, loc, info.alloc.getGuid(), ValueRange{zero});
        depGuidPrefix.push_back(guidScalar);
        depGuidValues.push_back(guidScalar);
        ++numTimingDbGuids;
      }

      if (!depGuidPrefix.empty()) {
        depGuidPrefix.append(loopBackParams.begin(), loopBackParams.end());
        loopBackParams = std::move(depGuidPrefix);
      }
    }
  }

  bool loopBackParamsStabilized = false;
  if (firstContinuation && !loopBackParams.empty()) {
    for (auto [advanceBlock, anchor] : advanceSites) {
      clearAdvanceSite(advanceBlock, anchor);
      OpBuilder advanceBuilder = OpBuilder::atBlockTerminator(advanceBlock);
      emitAdvanceLogic(advanceBuilder, loc, iv, lb, ub, step, body, slots,
                       firstChainId, interEpochOps.front(), sequentialOps,
                       loopBackParams);
    }
    loopBackParamsStabilized = true;
  } else {
    constexpr unsigned kMaxLoopBackIterations = 8;
    for (unsigned iter = 0; iter < kMaxLoopBackIterations; ++iter) {
      for (auto [advanceBlock, anchor] : advanceSites) {
        clearAdvanceSite(advanceBlock, anchor);
        OpBuilder advanceBuilder = OpBuilder::atBlockTerminator(advanceBlock);
        emitAdvanceLogic(advanceBuilder, loc, iv, lb, ub, step, body, slots,
                         firstChainId, interEpochOps.front(), sequentialOps,
                         loopBackParams);
      }

      if (!firstContinuation)
        break;

      SmallVector<Value> updatedLoopBackParams =
          EdtUtils::collectPackedValues(firstContinuation);
      if (ValueAnalysis::areValueRangesIdentical(loopBackParams,
                                                 updatedLoopBackParams)) {
        loopBackParamsStabilized = true;
        break;
      }
      loopBackParams = std::move(updatedLoopBackParams);
    }
  }
  if (!loopBackParamsStabilized && !advanceSites.empty())
    ARTS_WARN("CPS Chain: loop-back parameter contract reached iteration cap");

  bool hasDbHandles =
      !timingDbs.empty() || !dataGuids.empty() || !dataPtrs.empty();
  if (hasDbHandles && firstContinuation) {
    OpBuilder preEpochBuilder(outerEpoch);
    Value zero = arith::ConstantIndexOp::create(preEpochBuilder, loc, 0);
    Value one = arith::ConstantIndexOp::create(preEpochBuilder, loc, 1);
    SmallVector<Value> contDeps;
    Value scratchDepGuid;
    DenseSet<Operation *> releasedDbRoots;

    auto emitEarlyRelease = [&](Value dbPtr) {
      Operation *underlyingDb = DbUtils::getUnderlyingDb(dbPtr);
      if (!underlyingDb || !releasedDbRoots.insert(underlyingDb).second)
        return;
      DbReleaseOp::create(preEpochBuilder, loc, dbPtr);
    };

    for (auto &[guidValue, info] : dataGuids) {
      if (info.alloc)
        emitEarlyRelease(info.alloc.getPtr());
    }

    for (auto &[ptrValue, info] : timingDbs) {
      if (info.isGuid)
        continue;
      if (!info.alloc) {
        ARTS_WARN("CPS Chain: timing DB has null alloc, skipping dep");
        continue;
      }
      emitEarlyRelease(info.alloc.getPtr());
      auto acq = DbAcquireOp::create(preEpochBuilder, loc, ArtsMode::inout,
                                     info.alloc.getGuid(), info.alloc.getPtr(),
                                     info.alloc.getPtr().getType(),
                                     /*partitionMode=*/std::nullopt,
                                     /*indices=*/SmallVector<Value>{},
                                     /*offsets=*/SmallVector<Value>{zero},
                                     /*sizes=*/SmallVector<Value>{one});
      contDeps.push_back(acq.getPtr());
    }

    DenseSet<Value> routedDataPtrRoots;
    for (auto &[ptrValue, info] : dataPtrs) {
      if (!info.alloc)
        continue;
      Value allocPtr = info.alloc.getPtr();
      if (!allocPtr || !routedDataPtrRoots.insert(allocPtr).second)
        continue;
      emitEarlyRelease(allocPtr);
      auto acq = DbAcquireOp::create(preEpochBuilder, loc, ArtsMode::in,
                                     info.alloc.getGuid(), allocPtr,
                                     allocPtr.getType(),
                                     /*partitionMode=*/std::nullopt,
                                     /*indices=*/SmallVector<Value>{},
                                     /*offsets=*/SmallVector<Value>{zero},
                                     /*sizes=*/SmallVector<Value>{one});
      contDeps.push_back(acq.getPtr());
    }

    if (scratchAlloc) {
      emitEarlyRelease(scratchAlloc.getPtr());
      auto scratchAcq = DbAcquireOp::create(
          preEpochBuilder, loc, ArtsMode::in, scratchAlloc.getGuid(),
          scratchAlloc.getPtr(), scratchAlloc.getPtr().getType(),
          /*partitionMode=*/std::nullopt,
          /*indices=*/SmallVector<Value>{},
          /*offsets=*/SmallVector<Value>{zero},
          /*sizes=*/SmallVector<Value>{one});
      scratchDepGuid = scratchAcq.getGuid();
      contDeps.push_back(scratchAcq.getPtr());
    }

    DenseMap<Value, unsigned> timingPtrToDepIdx;
    DenseMap<Value, unsigned> dataPtrToDepIdx;
    unsigned depIdx = 0;
    for (auto &[ptrValue, info] : timingDbs) {
      if (info.isGuid)
        continue;
      timingPtrToDepIdx[ptrValue] = depIdx++;
    }
    for (auto &[ptrValue, info] : dataPtrs) {
      if (!info.alloc)
        continue;
      Value allocPtr = info.alloc.getPtr();
      if (!allocPtr || dataPtrToDepIdx.contains(allocPtr))
        continue;
      dataPtrToDepIdx[allocPtr] = depIdx++;
    }
    unsigned scratchDepIdx = scratchAlloc ? depIdx : UINT_MAX;

    allContinuations.clear();
    outerEpoch.walk([&](EdtOp edt) {
      if (edt->hasAttr(CPSLoopContinuation))
        allContinuations.push_back(edt);
    });

    for (EdtOp contEdt : allContinuations) {
      Block &contBlock = contEdt.getBody().front();

      for (Value dep : contDeps) {
        contEdt.appendDependency(dep);
        contBlock.addArgument(dep.getType(), loc);
      }
      auto chainId = contEdt->getAttrOfType<StringAttr>(CPSChainId);
      ARTS_INFO("CPS Chain: Added "
                << contDeps.size() << " deps to "
                << (chainId ? chainId.getValue() : "unknown"));

      unsigned numDeps = contDeps.size();
      unsigned numBlockArgs = contBlock.getNumArguments();

      OpBuilder bodyBuilder(&contBlock, contBlock.begin());
      auto isWithinContinuationScope = [&](OpOperand &use) {
        Operation *owner = use.getOwner();
        if (!contEdt->isAncestor(owner))
          return false;
        for (Operation *parent = owner;
             parent && parent != contEdt.getOperation();
             parent = parent->getParentOp()) {
          if (auto edtParent = dyn_cast<EdtOp>(parent)) {
            if (edtParent != contEdt && edtParent->hasAttr(CPSLoopContinuation))
              return false;
          }
        }
        return true;
      };

      DenseSet<Value> replacedAllocPtrs;
      for (auto &[ptrValue, info] : timingDbs) {
        if (info.isGuid)
          continue;
        auto it = timingPtrToDepIdx.find(ptrValue);
        if (it == timingPtrToDepIdx.end())
          continue;
        unsigned argIdx = numBlockArgs - numDeps + it->second;
        BlockArgument depArg = contBlock.getArgument(argIdx);

        if (info.alloc && !replacedAllocPtrs.count(info.alloc.getPtr())) {
          info.alloc.getPtr().replaceUsesWithIf(depArg,
                                                isWithinContinuationScope);
          replacedAllocPtrs.insert(info.alloc.getPtr());
        }

        if (ptrValue != (info.alloc ? info.alloc.getPtr() : Value())) {
          bodyBuilder.setInsertionPoint(&contBlock, contBlock.begin());
          Value czero = arith::ConstantIndexOp::create(bodyBuilder, loc, 0);

          // Extract result type from the BlockArgument's memref element type.
          // depArg is a DB pointer like memref<1xmemref<10xf64>>, so the
          // element type (memref<10xf64>) is the DbRefOp result type.
          auto depArgType = cast<MemRefType>(depArg.getType());
          Type dbViewResultType = depArgType.getElementType();

          // Create DbRefOp with explicit result type (builder can't infer from
          // BlockArgument). Use the 3-arg create<> overload.
          Value dbView = DbRefOp::create(bodyBuilder, loc, dbViewResultType,
                                         depArg, SmallVector<Value>{czero});

          Value replacement = dbView;
          if (replacement.getType() != ptrValue.getType())
            replacement = memref::CastOp::create(bodyBuilder, loc,
                                                 ptrValue.getType(), dbView);
          ptrValue.replaceUsesWithIf(replacement, isWithinContinuationScope);
        }
      }

      DenseSet<Value> replacedDataPtrRoots;
      for (auto &[ptrValue, info] : dataPtrs) {
        if (!info.alloc)
          continue;
        Value allocPtr = info.alloc.getPtr();
        auto it = dataPtrToDepIdx.find(allocPtr);
        if (!allocPtr || it == dataPtrToDepIdx.end())
          continue;
        unsigned argIdx = numBlockArgs - numDeps + it->second;
        BlockArgument depArg = contBlock.getArgument(argIdx);

        if (!replacedDataPtrRoots.count(allocPtr)) {
          allocPtr.replaceUsesWithIf(depArg, isWithinContinuationScope);
          replacedDataPtrRoots.insert(allocPtr);
        }

        if (ptrValue != allocPtr) {
          Value replacement = depArg;
          if (replacement.getType() != ptrValue.getType())
            replacement = memref::CastOp::create(bodyBuilder, loc,
                                                 ptrValue.getType(), depArg);
          ptrValue.replaceUsesWithIf(replacement, isWithinContinuationScope);
        }
      }

      DenseSet<Value> replacedAllocGuids;
      for (auto &[guidValue, info] : timingDbs) {
        if (!info.isGuid)
          continue;
        auto memrefTy = dyn_cast<MemRefType>(guidValue.getType());
        if (!memrefTy)
          continue;
        bodyBuilder.setInsertionPoint(&contBlock, contBlock.begin());
        SmallVector<Value> dynSizes;
        for (int64_t dim : memrefTy.getShape()) {
          if (dim == ShapedType::kDynamic)
            dynSizes.push_back(
                arith::ConstantIndexOp::create(bodyBuilder, loc, 1));
        }
        Value dummyGuid =
            memref::AllocaOp::create(bodyBuilder, loc, memrefTy, dynSizes);
        guidValue.replaceUsesWithIf(dummyGuid, isWithinContinuationScope);
        if (info.alloc && info.alloc.getGuid() != guidValue &&
            !replacedAllocGuids.count(info.alloc.getGuid())) {
          info.alloc.getGuid().replaceUsesWithIf(dummyGuid,
                                                 isWithinContinuationScope);
          replacedAllocGuids.insert(info.alloc.getGuid());
        }
      }

      if (scratchAlloc && scratchDepIdx != UINT_MAX) {
        unsigned scratchArgIdx = numBlockArgs - numDeps + scratchDepIdx;
        BlockArgument scratchArg = contBlock.getArgument(scratchArgIdx);

        bodyBuilder.setInsertionPoint(&contBlock, contBlock.begin());
        auto i64MemrefTy =
            MemRefType::get({ShapedType::kDynamic}, bodyBuilder.getI64Type());
        auto adaptScratchReplacement = [&](Value replacement,
                                           Type targetType) -> Value {
          if (!replacement || replacement.getType() == targetType)
            return replacement;
          return memref::CastOp::create(bodyBuilder, loc, targetType,
                                        replacement)
              .getResult();
        };

        if (scratchDepGuid) {
          Value replacement =
              adaptScratchReplacement(scratchArg, scratchDepGuid.getType());
          scratchDepGuid.replaceUsesWithIf(replacement,
                                           isWithinContinuationScope);
        }

        Value scratchView = scratchArg;
        if (scratchView.getType() != i64MemrefTy)
          scratchView =
              memref::CastOp::create(bodyBuilder, loc, i64MemrefTy, scratchArg);

        Value runningOffset =
            arith::ConstantIndexOp::create(bodyBuilder, loc, 0);
        SmallVector<std::pair<Value, Value>> guidLayoutForBody;
        for (auto &[guidValue, info] : dataGuids) {
          Value count = DbNumElementsOp::create(bodyBuilder, loc, info.alloc);
          guidLayoutForBody.push_back({runningOffset, count});
          runningOffset =
              arith::AddIOp::create(bodyBuilder, loc, runningOffset, count);
        }

        for (unsigned g = 0; g < dataGuids.size(); ++g) {
          Value origGuidArray = dataGuids[g].first;
          auto &guidInfo = dataGuids[g].second;
          auto [offset, count] = guidLayoutForBody[g];
          auto origMemrefTy = dyn_cast<MemRefType>(origGuidArray.getType());
          if (!origMemrefTy)
            continue;

          Value localGuids = memref::AllocOp::create(
              bodyBuilder, loc, i64MemrefTy, ValueRange{count});
          if (auto allocId =
                  DbUtils::resolveRootAllocId(origGuidArray, guidInfo.alloc))
            setDbRootAllocId(localGuids.getDefiningOp(), *allocId);

          auto copyLoop = scf::ForOp::create(
              bodyBuilder, loc,
              arith::ConstantIndexOp::create(bodyBuilder, loc, 0), count,
              arith::ConstantIndexOp::create(bodyBuilder, loc, 1));
          OpBuilder copyBody = OpBuilder::atBlockTerminator(copyLoop.getBody());
          Value ci2 = copyLoop.getInductionVar();
          Value scratchIdx = arith::AddIOp::create(copyBody, loc, offset, ci2);
          Value guidVal = memref::LoadOp::create(copyBody, loc, scratchView,
                                                 ValueRange{scratchIdx});
          memref::StoreOp::create(copyBody, loc, guidVal, localGuids,
                                  ValueRange{ci2});

          Value replacement = localGuids;
          if (replacement.getType() != origMemrefTy) {
            auto castOp = memref::CastOp::create(bodyBuilder, loc, origMemrefTy,
                                                 localGuids);
            if (auto allocId =
                    DbUtils::resolveRootAllocId(origGuidArray, guidInfo.alloc))
              setDbRootAllocId(castOp.getOperation(), *allocId);
            replacement = castOp.getResult();
          }

          if (guidInfo.alloc)
            guidInfo.alloc.getGuid().replaceUsesWithIf(
                replacement, isWithinContinuationScope);
          origGuidArray.replaceUsesWithIf(replacement,
                                          isWithinContinuationScope);
        }
      }
    }

    ARTS_INFO("CPS Chain: DB routing complete — "
              << contDeps.size() << " deps wired to " << allContinuations.size()
              << " continuations");
  }

  if (firstContinuation && hasDbHandles) {
    auto setDepRoutingAttrForCarry = [&](ArrayRef<Value> carryValues) {
      if (!(numTimingDbGuids > 0 || scratchAlloc))
        return;
      int64_t hasScratch = scratchAlloc ? 1 : 0;
      SmallVector<int64_t> routingData = {
          static_cast<int64_t>(numTimingDbGuids), hasScratch};
      for (Value depGuidValue : depGuidValues) {
        int64_t idx = -1;
        for (unsigned i = 0; i < carryValues.size(); ++i) {
          if (carryValues[i] == depGuidValue) {
            idx = i;
            break;
          }
        }
        routingData.push_back(idx);
      }
      outerEpoch.walk([&](CPSAdvanceOp adv) {
        adv->setAttr(CPSDepRouting,
                     DenseI64ArrayAttr::get(adv.getContext(), routingData));
      });
    };

    auto rewriteAdvanceCarry = [&](ArrayRef<Value> carryValues) {
      outerEpoch.walk([&](CPSAdvanceOp adv) {
        auto numCarryAttr = adv->getAttrOfType<IntegerAttr>(CPSNumCarry);
        if (!numCarryAttr)
          return;
        bool isAdditive = adv->hasAttr(CPSAdditiveParams);
        unsigned paramStart = isAdditive ? 2 : 0;
        SmallVector<Value> newNextParams;
        auto oldParams = adv.getNextIterParams();
        for (unsigned i = 0; i < paramStart && i < oldParams.size(); ++i)
          newNextParams.push_back(oldParams[i]);
        newNextParams.append(carryValues.begin(), carryValues.end());

        OpBuilder advBuilder(adv);
        auto targetId = adv.getTargetChainIdAttr();
        auto newAdv = CPSAdvanceOp::create(advBuilder, adv.getLoc(),
                                           newNextParams, targetId);
        for (auto attr : adv->getAttrs())
          if (attr.getName() != adv.getTargetChainIdAttrName())
            newAdv->setAttr(attr.getName(), attr.getValue());
        newAdv->setAttr(CPSNumCarry,
                        advBuilder.getI64IntegerAttr(carryValues.size()));

        Region &oldRegion = adv.getEpochBody();
        Region &newRegion = newAdv.getEpochBody();
        if (!oldRegion.empty())
          newRegion.takeBody(oldRegion);
        adv.erase();
      });
    };

    SmallVector<Value> reorderedCarry =
        EdtUtils::collectPackedValues(firstContinuation);

    ARTS_INFO("CPS Chain: Re-analyzed " << firstChainId << " captures — "
                                        << reorderedCarry.size()
                                        << " carry params");

    bool preserveOriginalCarryContract = numTimingDbGuids > 0 || scratchAlloc;
    if (preserveOriginalCarryContract) {
      bool carryChanged =
          !ValueAnalysis::areValueRangesIdentical(reorderedCarry,
                                                  loopBackParams) ||
                          reorderedCarry.size() != loopBackParams.size();
      if (carryChanged) {
        ARTS_INFO("CPS Chain: dep-routed carry changed after re-analysis; "
                  << "reseeding loop-back carry from the reanalyzed "
                     "continuation state");
        rewriteAdvanceCarry(reorderedCarry);
        loopBackParams.assign(reorderedCarry.begin(), reorderedCarry.end());
      }
      setDepRoutingAttrForCarry(loopBackParams);
    } else if (!ValueAnalysis::areValueRangesIdentical(reorderedCarry,
                                                       loopBackParams)) {
      ARTS_INFO("CPS Chain: carry re-analysis changed order; preserving "
                << "original kickoff carry contract");
      setDepRoutingAttrForCarry(loopBackParams);
    } else if (reorderedCarry.size() != loopBackParams.size()) {
      ARTS_INFO("CPS Chain: carry re-analysis changed arity from "
                << loopBackParams.size() << " to " << reorderedCarry.size()
                << "; preserving original loop-back contract");
      setDepRoutingAttrForCarry(loopBackParams);
    } else {
      rewriteAdvanceCarry(reorderedCarry);
      setDepRoutingAttrForCarry(reorderedCarry);
    }

    bool hasPlanAttrs = false;
    outerEpoch.walk([&](EdtOp edt) {
      if (hasStructuredPlanAttrs(edt.getOperation()))
        hasPlanAttrs = true;
    });
    if (hasPlanAttrs) {
      SmallVector<int64_t> stateSchema = {
          static_cast<int64_t>(scalarParams.size()),
          static_cast<int64_t>(timingDbs.size()),
          static_cast<int64_t>(dataGuids.size()),
          static_cast<int64_t>(dataPtrs.size())};
      SmallVector<int64_t> depSchema = {
          static_cast<int64_t>(numTimingDbGuids),
          static_cast<int64_t>(scratchAlloc ? 1 : 0)};
      for (EdtOp contEdt : allContinuations) {
        contEdt->setAttr(
            AttrNames::Operation::LaunchState::StateSchema,
            DenseI64ArrayAttr::get(contEdt.getContext(), stateSchema));
        contEdt->setAttr(
            AttrNames::Operation::LaunchState::DepSchema,
            DenseI64ArrayAttr::get(contEdt.getContext(), depSchema));
      }
      ARTS_INFO("CPS Chain: Stamped launch state schema on "
                << allContinuations.size() << " continuations");
    }
  }

  if (firstContinuation) {
    allContinuations.clear();
    outerEpoch.walk([&](EdtOp edt) {
      if (edt->hasAttr(CPSLoopContinuation))
        allContinuations.push_back(edt);
    });

    bool requiresGuidDepReconstruction = numTimingDbGuids > 0 || scratchAlloc;
    bool hasCoordinatorContinuation =
        llvm::any_of(allContinuations, continuationCreatesNestedAsyncWork);

    if (requiresGuidDepReconstruction || hasCoordinatorContinuation) {
      unsigned disabledCount = 0;
      outerEpoch.walk([&](CPSAdvanceOp advance) {
        if (!advance->hasAttr(CPSDirectRecreate))
          return;
        advance->removeAttr(CPSDirectRecreate);
        ++disabledCount;
      });

      if (disabledCount) {
        ARTS_INFO("CPS Chain: disabled direct-recreate on "
                  << disabledCount << " advance op(s)"
                  << (requiresGuidDepReconstruction
                          ? " due to GUID/scratch dep reconstruction"
                          : "")
                  << (hasCoordinatorContinuation
                          ? (requiresGuidDepReconstruction
                                 ? " and coordinator continuations"
                                 : " due to coordinator continuations")
                          : ""));
      }
    }
  }

  bool hasEscapedLoopDefs = false;
  forOp.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (forOp->isAncestor(user))
          continue;
        hasEscapedLoopDefs = true;
        ARTS_WARN("CPS Chain: loop-local value still used outside loop before "
                  "erase: "
                  << result << " defined by " << *op << " used by " << *user);
      }
    }
  });
  if (hasEscapedLoopDefs) {
    ARTS_WARN("CPS Chain: aborting erase of transformed loop due to escaped "
              "loop-local values");
    return false;
  }

  forOp.erase();

  if (scratchAlloc) {
    OpBuilder afterEpoch(outerEpoch->getNextNode());
    DbFreeOp::create(afterEpoch, loc, scratchAlloc.getGuid());
    DbFreeOp::create(afterEpoch, loc, scratchAlloc.getPtr());
  }

  ARTS_INFO("CPS Chain: Transformed " << slotCount
                                      << "-epoch loop into continuation chain");
  return true;
}

} // namespace mlir::arts::epoch_opt
