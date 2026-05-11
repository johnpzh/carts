///==========================================================================///
/// File: EpochOptScheduling.cpp
///
/// Scheduling-oriented epoch transforms used by EpochOpt.
///==========================================================================///

#include "arts/dialect/core/Transforms/EpochOptInternal.h"
#include "arts/utils/Debug.h"

ARTS_DEBUG_SETUP(epoch_opt);

using namespace mlir;
using namespace mlir::arts;

namespace mlir::arts::epoch_opt {
namespace {

scf::IfOp ensureElseBranch(Location loc, scf::IfOp ifOp) {
  if (!ifOp.getElseRegion().empty())
    return ifOp;

  OpBuilder ifBuilder(ifOp);
  auto newIf = scf::IfOp::create(ifBuilder, loc, ifOp.getCondition(), true);
  Block &oldThenBlock = ifOp.getThenRegion().front();
  Block &newThenBlock = newIf.getThenRegion().front();
  for (Operation &op :
       llvm::make_early_inc_range(oldThenBlock.without_terminator()))
    op.moveBefore(newThenBlock.getTerminator());
  ifOp.erase();
  return newIf;
}

} // namespace

LogicalResult
transformToContinuation(EpochOp epochOp,
                        const EpochContinuationDecision &decision) {
  OpBuilder builder(epochOp->getContext());
  Location loc = epochOp.getLoc();

  ARTS_DEBUG(
      "  DB acquires captured: " << decision.capturedDbAcquireValues.size());

  SmallVector<Value> deps(decision.capturedDbAcquireValues.begin(),
                          decision.capturedDbAcquireValues.end());

  builder.setInsertionPointAfter(epochOp);
  auto edtOp = EdtOp::create(builder, loc, EdtType::task,
                             EdtConcurrency::intranode, deps);
  edtOp->setAttr(ControlDep, builder.getIntegerAttr(builder.getI32Type(), 1));
  edtOp->setAttr(ContinuationForEpoch, builder.getUnitAttr());

  Block &edtBlock = edtOp.getBody().front();
  for (Value dep : deps)
    edtBlock.addArgument(dep.getType(), loc);
  builder.setInsertionPointToStart(&edtBlock);

  IRMapping valueMapping;
  for (auto [oldDep, blockArg] : llvm::zip(deps, edtBlock.getArguments()))
    valueMapping.map(oldDep, blockArg);

  for (Operation *op : decision.tailOps) {
    Operation *cloned = builder.clone(*op, valueMapping);
    for (auto [oldRes, newRes] :
         llvm::zip(op->getResults(), cloned->getResults()))
      valueMapping.map(oldRes, newRes);
  }

  if (edtBlock.empty() || !edtBlock.back().hasTrait<OpTrait::IsTerminator>())
    YieldOp::create(builder, loc);

  for (Operation *op : llvm::reverse(decision.tailOps))
    op->erase();

  epochOp->setAttr(ContinuationForEpoch, builder.getUnitAttr());

  ARTS_INFO("  Created continuation EDT with " << deps.size()
                                               << " DB deps + 1 control dep");
  return success();
}

bool tryCPSLoopTransform(scf::ForOp forOp, const EpochAnalysis &epochAnalysis) {
  if (loopContainsCpsDriverExcludedDepPattern(forOp)) {
    ARTS_INFO("CPS: skipping loop with specialized dep pattern");
    return false;
  }

  EpochAsyncLoopDecision asyncDecision =
      epochAnalysis.evaluateAsyncLoopStrategy(forOp);
  if (asyncDecision.strategy != EpochAsyncLoopStrategy::AdvanceEdt ||
      asyncDecision.slots.size() != 1) {
    ARTS_DEBUG("CPS: advance_edt requires a single epoch slot, skipping");
    return false;
  }

  EpochLoopDriverDecision decision = epochAnalysis.evaluateCPSLoopDriver(forOp);
  if (!decision.eligible) {
    ARTS_DEBUG("CPS: " << decision.rationale << ", skipping");
    return false;
  }

  ARTS_INFO("CPS: Found eligible time-step loop");

  OpBuilder builder(forOp->getContext());
  Location loc = forOp.getLoc();
  Block &body = *forOp.getBody();
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  Value step = forOp.getStep();
  Value iv = forOp.getInductionVar();
  auto i64Ty = builder.getI64Type();
  std::string chainId = makeAsyncLoopChainId(forOp.getOperation());

  builder.setInsertionPoint(forOp);
  SmallVector<Value> currentStateDeps;
  MutableArrayRef<EpochSlot> slots(asyncDecision.slots);

  auto wrapEpoch =
      EpochOp::create(builder, loc, IntegerType::get(builder.getContext(), 64));
  wrapEpoch->setAttr(AttrNames::Operation::CPSOuterEpoch,
                     builder.getUnitAttr());
  copyNormalizedPlanAttrs(forOp.getOperation(), wrapEpoch.getOperation(),
                          EpochAsyncLoopStrategy::AdvanceEdt);
  if (auto lbCst = getConstantIntValue(lb))
    wrapEpoch->setAttr(CPSInitIter, builder.getI64IntegerAttr(*lbCst));
  else
    wrapEpoch->setAttr(CPSInitIter, builder.getI64IntegerAttr(0));

  Region &epochRegion = wrapEpoch.getRegion();
  if (epochRegion.empty())
    epochRegion.emplaceBlock();
  Block &epochBlock = epochRegion.front();
  if (epochBlock.empty() ||
      !epochBlock.back().hasTrait<OpTrait::IsTerminator>()) {
    OpBuilder yieldBuilder = OpBuilder::atBlockEnd(&epochBlock);
    YieldOp::create(yieldBuilder, loc);
  }

  OpBuilder epochBuilder = OpBuilder::atBlockTerminator(&epochBlock);
  auto kickoffEdt = EdtOp::create(epochBuilder, loc, EdtType::task,
                                  EdtConcurrency::intranode, currentStateDeps);
  kickoffEdt->setAttr(CPSChainId, builder.getStringAttr(chainId));
  copyNormalizedPlanAttrs(forOp.getOperation(), kickoffEdt.getOperation(),
                          EpochAsyncLoopStrategy::AdvanceEdt);
  if (!currentStateDeps.empty())
    kickoffEdt->setAttr(CPSForwardDeps, builder.getUnitAttr());

  Block &edtBlock = kickoffEdt.getBody().front();
  for (Value dep : currentStateDeps)
    edtBlock.addArgument(dep.getType(), loc);

  IRMapping edtMapping;
  for (auto [oldDep, blockArg] :
       llvm::zip(currentStateDeps, edtBlock.getArguments()))
    edtMapping.map(oldDep, blockArg);
  edtMapping.map(iv, lb);

  OpBuilder edtBuilder = OpBuilder::atBlockEnd(&edtBlock);
  cloneNonSlotArith(edtBuilder, body, slots, edtMapping);

  SmallVector<Block *> advanceBlocks;
  OpBuilder *currentBuilder = &edtBuilder;
  IRMapping currentMapping = edtMapping;
  Operation *firstClonedSlotOp =
      cloneEpochSlot(*currentBuilder, slots.front(), currentMapping);

  if (slots.size() == 1) {
    if (slots.front().wrappingIf) {
      auto clonedIf = cast<scf::IfOp>(firstClonedSlotOp);
      clonedIf = ensureElseBranch(loc, clonedIf);
      advanceBlocks.push_back(&clonedIf.getThenRegion().front());
      advanceBlocks.push_back(&clonedIf.getElseRegion().front());
    } else {
      advanceBlocks.push_back(currentBuilder->getBlock());
    }
  } else {
    for (unsigned slotIdx = 1; slotIdx < slots.size(); ++slotIdx) {
      auto contEdt = EdtOp::create(*currentBuilder, loc, EdtType::task,
                                   EdtConcurrency::intranode, currentStateDeps);
      markAsContinuation(contEdt, *currentBuilder, forOp.getOperation(),
                         slotIdx);
      copyNormalizedPlanAttrs(forOp.getOperation(), contEdt.getOperation(),
                              EpochAsyncLoopStrategy::AdvanceEdt);
      if (!currentStateDeps.empty())
        contEdt->setAttr(CPSForwardDeps, builder.getUnitAttr());

      Block &contBlock = contEdt.getBody().front();
      for (Value dep : currentStateDeps)
        contBlock.addArgument(dep.getType(), loc);
      ensureYieldTerminator(contBlock, loc);

      OpBuilder contBuilder = OpBuilder::atBlockTerminator(&contBlock);
      IRMapping contMapping;
      contMapping.map(iv, lb);
      cloneNonSlotArith(contBuilder, body, slots, contMapping);
      Operation *clonedSlotOp =
          cloneEpochSlot(contBuilder, slots[slotIdx], contMapping);

      bool isLastSlot = slotIdx + 1 == slots.size();
      if (!isLastSlot) {
        currentBuilder = &contBuilder;
        currentMapping = std::move(contMapping);
        continue;
      }

      if (slots[slotIdx].wrappingIf) {
        auto clonedIf = cast<scf::IfOp>(clonedSlotOp);
        clonedIf = ensureElseBranch(loc, clonedIf);
        advanceBlocks.push_back(&clonedIf.getThenRegion().front());
        advanceBlocks.push_back(&clonedIf.getElseRegion().front());
      } else {
        advanceBlocks.push_back(&contBlock);
      }
    }
  }

  SmallVector<Value> loopBackParams = collectEdtPackedValues(kickoffEdt);
  auto iterIt = llvm::find(loopBackParams, lb);
  if (iterIt != loopBackParams.end()) {
    unsigned iterIdx = std::distance(loopBackParams.begin(), iterIt);
    kickoffEdt->setAttr(CPSIterCounterParamIdx,
                        builder.getI64IntegerAttr(iterIdx));
  }

  for (Block *advanceBlock : advanceBlocks) {
    SmallVector<Value> advanceStateDeps(currentStateDeps.begin(),
                                        currentStateDeps.end());
    auto appendAdvanceStateDep = [&](Value value) {
      if (!value || !isa<BaseMemRefType>(value.getType()))
        return;
      Operation *underlyingDb = DbUtils::getUnderlyingDb(value);
      if (!isa_and_nonnull<DbAcquireOp, DepDbAcquireOp>(underlyingDb))
        return;
      if (!llvm::is_contained(advanceStateDeps, value))
        advanceStateDeps.push_back(value);
    };
    for (Value value : loopBackParams)
      appendAdvanceStateDep(value);

    ensureYieldTerminator(*advanceBlock, loc);
    OpBuilder advanceSiteBuilder = OpBuilder::atBlockTerminator(advanceBlock);
    auto advanceEdt =
        EdtOp::create(advanceSiteBuilder, loc, EdtType::task,
                      EdtConcurrency::intranode, advanceStateDeps);
    advanceEdt->setAttr(ControlDep,
                        builder.getIntegerAttr(builder.getI32Type(), 1));
    advanceEdt->setAttr(ContinuationForEpoch, builder.getUnitAttr());
    copyNormalizedPlanAttrs(forOp.getOperation(), advanceEdt.getOperation(),
                            EpochAsyncLoopStrategy::AdvanceEdt);
    if (!advanceStateDeps.empty())
      advanceEdt->setAttr(CPSForwardDeps, builder.getUnitAttr());
    Block &advanceBlockBody = advanceEdt.getBody().front();
    for (Value dep : advanceStateDeps)
      advanceBlockBody.addArgument(dep.getType(), loc);
    ensureYieldTerminator(advanceBlockBody, loc);

    OpBuilder advanceBuilder = OpBuilder::atBlockTerminator(&advanceBlockBody);
    Value stepI64 =
        step.getType().isIndex()
            ? arith::IndexCastOp::create(advanceBuilder, loc, i64Ty, step)
                  .getResult()
            : arith::ExtSIOp::create(advanceBuilder, loc, i64Ty, step)
                  .getResult();
    Value ubI64 =
        ub.getType().isIndex()
            ? arith::IndexCastOp::create(advanceBuilder, loc, i64Ty, ub)
                  .getResult()
            : arith::ExtSIOp::create(advanceBuilder, loc, i64Ty, ub)
                  .getResult();
    auto materializeAdvanceCarry = [&](Value value) -> Value {
      if (!isa<BaseMemRefType>(value.getType()))
        return value;
      Value rawPtr = polygeist::Memref2PointerOp::create(
          advanceBuilder, loc,
          LLVM::LLVMPointerType::get(advanceBuilder.getContext()), value);
      return LLVM::PtrToIntOp::create(advanceBuilder, loc, i64Ty, rawPtr);
    };
    SmallVector<Value> localizedLoopBackParams;
    localizedLoopBackParams.reserve(loopBackParams.size());
    for (Value value : loopBackParams) {
      auto depIt = llvm::find(advanceStateDeps, value);
      if (depIt == advanceStateDeps.end()) {
        localizedLoopBackParams.push_back(materializeAdvanceCarry(value));
        continue;
      }
      unsigned depIndex = std::distance(advanceStateDeps.begin(), depIt);
      localizedLoopBackParams.push_back(
          materializeAdvanceCarry(advanceBlockBody.getArgument(depIndex)));
    }
    SmallVector<Value> nextParams = {stepI64, ubI64};
    nextParams.append(localizedLoopBackParams.begin(),
                      localizedLoopBackParams.end());
    auto advance = CPSAdvanceOp::create(advanceBuilder, loc, nextParams,
                                        advanceBuilder.getStringAttr(chainId));
    advance->setAttr(CPSAdditiveParams, builder.getUnitAttr());
    advance->setAttr(CPSDirectRecreate, builder.getUnitAttr());
    advance->setAttr(CPSNumCarry,
                     builder.getI64IntegerAttr(loopBackParams.size()));
    Region &advanceRegion = advance.getEpochBody();
    if (advanceRegion.empty())
      advanceRegion.emplaceBlock();
    ensureYieldTerminator(advanceRegion.front(), loc);
  }

  if (edtBlock.empty() || !edtBlock.back().hasTrait<OpTrait::IsTerminator>())
    YieldOp::create(edtBuilder, loc);

  forOp.erase();

  ARTS_INFO("CPS: Wrapped time-step loop in epoch + async advance EDT");
  return true;
}

} // namespace mlir::arts::epoch_opt
