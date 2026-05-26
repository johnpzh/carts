///==========================================================================///
/// File: EpochOptStructural.cpp
///
/// Structural and local scheduling transforms used by EpochOpt.
///==========================================================================///

#include "carts/dialect/arts/Transforms/EpochOptInternal.h"
#include "carts/utils/Debug.h"
#include "carts/utils/OperationAttributes.h"

ARTS_DEBUG_SETUP(epoch_opt);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace mlir::carts::arts::epoch_opt {
namespace {

SmallVector<unsigned> findEpochCutPoints(ArrayRef<Operation *> ops) {
  SmallVector<unsigned> cutPoints;
  if (ops.size() <= 1)
    return cutPoints;

  unsigned n = ops.size();
  DenseMap<Operation *, unsigned> opIndex;
  for (unsigned k = 0; k < n; ++k)
    opIndex[ops[k]] = k;

  SmallVector<unsigned> minDep(n, n);
  for (unsigned j = 0; j < n; ++j) {
    unsigned depMin = n;

    auto checkOperand = [&](Value operand) {
      if (auto *defOp = operand.getDefiningOp()) {
        auto it = opIndex.find(defOp);
        if (it != opIndex.end())
          depMin = std::min(depMin, it->second);
      }
    };

    for (Value operand : ops[j]->getOperands())
      checkOperand(operand);

    ops[j]->walk([&](Operation *nested) {
      if (nested == ops[j])
        return WalkResult::advance();
      for (Value operand : nested->getOperands())
        checkOperand(operand);
      return WalkResult::advance();
    });

    for (unsigned k = 0; k < j; ++k) {
      if (DbAnalysis::hasDbConflict(ops[j], ops[k]))
        depMin = std::min(depMin, k);
    }

    minDep[j] = depMin;
  }

  SmallVector<unsigned> suffixMin(n);
  suffixMin[n - 1] = minDep[n - 1];
  for (int k = static_cast<int>(n) - 2; k >= 0; --k)
    suffixMin[k] = std::min(minDep[k], suffixMin[k + 1]);

  for (unsigned i = 1; i < n; ++i) {
    if (suffixMin[i] >= i)
      cutPoints.push_back(i);
  }

  return cutPoints;
}

unsigned narrowEpoch(EpochOp epoch) {
  Block &body = epoch.getBody().front();

  SmallVector<Operation *> ops;
  for (Operation &op : body.without_terminator())
    ops.push_back(&op);

  if (ops.size() <= 1)
    return 0;

  SmallVector<unsigned> cutPoints = findEpochCutPoints(ops);
  if (cutPoints.empty())
    return 0;

  ARTS_DEBUG("Found " << cutPoints.size() << " cut point(s) in epoch");

  SmallVector<std::pair<unsigned, unsigned>> groups;
  unsigned start = 0;
  for (unsigned cp : cutPoints) {
    groups.push_back({start, cp});
    start = cp;
  }
  groups.push_back({start, static_cast<unsigned>(ops.size())});

  if (groups.size() <= 1)
    return 0;

  Location loc = epoch.getLoc();
  bool epochGuidUsed = !epoch.getEpochGuid().use_empty();
  if (epochGuidUsed) {
    ARTS_DEBUG("Epoch GUID is used externally, skipping narrowing");
    return 0;
  }

  OpBuilder builder(epoch->getBlock(), std::next(Block::iterator(epoch)));
  for (unsigned g = 1; g < groups.size(); ++g) {
    auto [gStart, gEnd] = groups[g];
    auto newEpoch = EpochOp::create(builder, loc, ValueRange{});
    auto &newBlock = newEpoch.getBody().emplaceBlock();
    OpBuilder innerBuilder(&newBlock, newBlock.begin());
    YieldOp::create(innerBuilder, loc);

    Operation *terminator = newBlock.getTerminator();
    for (unsigned i = gStart; i < gEnd; ++i)
      ops[i]->moveBefore(terminator);

    builder.setInsertionPointAfter(newEpoch);
  }

  ARTS_DEBUG("Split epoch into " << groups.size() << " narrower epochs");
  return groups.size() - 1;
}

void fuseEpochs(EpochOp first, EpochOp second) {
  if (!second.getEpochGuid().use_empty())
    return;

  Block &firstBlock = first.getBody().front();
  Block &secondBlock = second.getBody().front();
  EdtUtils::spliceBodyBeforeTerminator(secondBlock, firstBlock);

  second.erase();
}

bool isKernelTimerTailOp(Operation *op) {
  auto callOp = dyn_cast<func::CallOp>(op);
  if (!callOp)
    return false;
  return callOp.getCallee().starts_with("carts_kernel_timer_");
}

static bool isRankZeroLocalAlloca(Value value) {
  value = ValueAnalysis::stripNumericCasts(value);
  auto alloca = value.getDefiningOp<memref::AllocaOp>();
  return alloca && alloca.getType().getRank() == 0;
}

static bool isAmortizableSetupOp(Operation *op, Value loopIv) {
  if (!op)
    return false;
  if (isSideEffectFreeArithmeticLikeOp(op))
    return llvm::all_of(op->getOperands(), [&](Value operand) {
      return !ValueAnalysis::dependsOn(operand, loopIv);
    });

  if (auto load = dyn_cast<memref::LoadOp>(op))
    return isRankZeroLocalAlloca(load.getMemRef()) &&
           !ValueAnalysis::dependsOn(load.getMemRef(), loopIv);

  if (auto store = dyn_cast<memref::StoreOp>(op))
    return isRankZeroLocalAlloca(store.getMemRef()) &&
           !ValueAnalysis::dependsOn(store.getMemRef(), loopIv) &&
           !ValueAnalysis::dependsOn(store.getValue(), loopIv);

  return false;
}

static bool isAmortizableTailOp(Operation *op, Value loopIv) {
  if (isKernelTimerTailOp(op))
    return true;
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return isRankZeroLocalAlloca(store.getMemRef()) &&
           !ValueAnalysis::dependsOn(store.getMemRef(), loopIv) &&
           !ValueAnalysis::dependsOn(store.getValue(), loopIv);
  return false;
}

static bool isStableRepeatTopology(Operation *op) {
  auto topology = getPlanIterationTopologyAttr(op);
  return topology &&
         (topology.getValue() == ArtsPlanIterationTopology::owner_strip ||
          topology.getValue() == ArtsPlanIterationTopology::owner_tile);
}

static bool hasZeroSliceWideningPressure(Operation *op) {
  auto attr = getPlanCostSliceWideningPressureAttr(op);
  return attr && attr.getInt() == 0;
}

static bool epochHasRepeatStableUniformPlan(EpochOp epochOp) {
  Operation *op = epochOp.getOperation();
  auto depPattern = getDepPattern(op);
  auto repetition = getPlanRepetitionStructureAttr(op);
  return depPattern && isUniformFamilyDepPattern(*depPattern) && repetition &&
         repetition.getValue() == ArtsPlanRepetitionStructure::full_timestep &&
         isStableRepeatTopology(op) && hasZeroSliceWideningPressure(op);
}

static bool acquireHasRepeatStableUniformSlice(DbAcquireOp acquire) {
  if (!acquire)
    return false;
  auto depPattern = getDepPattern(acquire.getOperation());
  return depPattern && isUniformFamilyDepPattern(*depPattern) &&
         !DbAnalysis::isCoarse(acquire);
}

bool canWrapEdtBodyWithRepeatLoop(EdtOp edt) {
  Block &body = edt.getBody().front();
  bool seenRelease = false;
  bool sawRepeatableOp = false;
  for (Operation &op : body.without_terminator()) {
    if (isa<DbReleaseOp>(op)) {
      seenRelease = true;
      continue;
    }
    if (seenRelease)
      return false;
    sawRepeatableOp = true;
  }
  return sawRepeatableOp;
}

void wrapEdtBodyWithRepeatLoop(EdtOp edt, int64_t repeatCount) {
  if (repeatCount <= 1)
    return;

  Block &body = edt.getBody().front();
  SmallVector<Operation *> repeatOps;
  Operation *insertionPoint = body.getTerminator();
  for (Operation &op : body.without_terminator()) {
    if (isa<DbReleaseOp>(op)) {
      insertionPoint = &op;
      break;
    }
    repeatOps.push_back(&op);
  }
  if (repeatOps.empty())
    return;

  OpBuilder builder(insertionPoint);
  Location loc = edt.getLoc();
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value cN = arith::ConstantIndexOp::create(builder, loc, repeatCount);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
  auto repeatFor = scf::ForOp::create(builder, loc, c0, cN, c1);

  Operation *repeatTerminator = repeatFor.getBody()->getTerminator();
  for (Operation *op : repeatOps)
    op->moveBefore(repeatTerminator);
}

bool epochSupportsRepetitionAmortization(EpochOp epochOp) {
  struct RepeatAllocAccessInfo {
    bool hasRead = false;
    bool hasWrite = false;
    bool repeatStableUniformSlices = true;
  };

  DenseMap<Operation *, RepeatAllocAccessInfo> allocAccesses;
  bool sawEdt = false;
  bool safe = true;
  unsigned edtCount = 0;

  epochOp.walk([&](EdtOp edt) {
    sawEdt = true;
    ++edtCount;
    SmallVector<bool> argReads;
    SmallVector<bool> argWrites;
    EdtUtils::classifyArgAccesses(edt, argReads, argWrites);

    ValueRange deps = edt.getDependencies();
    if (deps.size() != argReads.size()) {
      safe = false;
      return WalkResult::interrupt();
    }

    for (unsigned i = 0; i < deps.size(); ++i) {
      auto acq = deps[i].getDefiningOp<DbAcquireOp>();
      if (!acq) {
        safe = false;
        return WalkResult::interrupt();
      }

      Operation *alloc = DbUtils::getUnderlyingDbAlloc(acq.getSourcePtr());
      if (!alloc) {
        safe = false;
        return WalkResult::interrupt();
      }

      if (argReads[i] && acq.getMode() == ArtsMode::out) {
        safe = false;
        return WalkResult::interrupt();
      }
      if (argWrites[i] && acq.getMode() == ArtsMode::in) {
        safe = false;
        return WalkResult::interrupt();
      }

      RepeatAllocAccessInfo &accessInfo = allocAccesses[alloc];
      accessInfo.repeatStableUniformSlices &=
          acquireHasRepeatStableUniformSlice(acq);
      if (argReads[i])
        accessInfo.hasRead = true;
      if (argWrites[i])
        accessInfo.hasWrite = true;
    }
    return WalkResult::advance();
  });

  if (!safe || !sawEdt)
    return false;

  for (const auto &entry : allocAccesses) {
    const RepeatAllocAccessInfo &accessInfo = entry.second;
    if (!(accessInfo.hasRead && accessInfo.hasWrite))
      continue;
    if (!epochHasRepeatStableUniformPlan(epochOp))
      return false;
    if (edtCount > 1 && !accessInfo.repeatStableUniformSlices)
      return false;
  }
  return true;
}

} // namespace

EpochNarrowingCounts narrowEpochScopes(ModuleOp module) {
  EpochNarrowingCounts counts;
  SmallVector<EpochOp> epochs;
  module.walk([&](EpochOp epoch) { epochs.push_back(epoch); });

  for (EpochOp epoch : epochs) {
    if (!epoch->getBlock())
      continue;
    unsigned created = narrowEpoch(epoch);
    if (created == 0)
      continue;
    ++counts.epochsNarrowed;
    counts.newEpochsCreated += created;
  }
  return counts;
}

unsigned processRegionForEpochFusion(Region &region,
                                     const EpochAnalysis &epochAnalysis) {
  unsigned fusedPairs = 0;
  for (Block &block : region) {
    for (auto it = block.begin(); it != block.end();) {
      auto first = dyn_cast<EpochOp>(&*it);
      if (!first) {
        ++it;
        continue;
      }

      EpochAccessSummary firstAccesses =
          epochAnalysis.summarizeEpochAccess(first);
      auto nextIt = std::next(it);
      while (nextIt != block.end()) {
        auto second = dyn_cast<EpochOp>(&*nextIt);
        if (!second)
          break;
        EpochAccessSummary secondAccesses =
            epochAnalysis.summarizeEpochAccess(second);
        EpochFusionDecision decision = epochAnalysis.evaluateEpochFusion(
            first, second, &firstAccesses, &secondAccesses);
        if (!decision.shouldFuse) {
          ARTS_DEBUG("Skipping epoch fusion: " << decision.rationale);
          break;
        }

        ARTS_DEBUG("Fusing consecutive epochs");
        fuseEpochs(first, second);
        firstAccesses.mergeFrom(secondAccesses);
        ++fusedPairs;
        nextIt = std::next(it);
      }
      ++it;
    }

    for (Operation &op : block) {
      for (Region &nested : op.getRegions())
        fusedPairs += processRegionForEpochFusion(nested, epochAnalysis);
    }
  }
  return fusedPairs;
}

bool tryAmortizeRepeatedEpochLoop(EpochOp epochOp) {
  if (!epochOp || !epochOp->getBlock())
    return false;

  scf::ForOp repeatLoop = dyn_cast<scf::ForOp>(epochOp->getParentOp());
  scf::IfOp wrappingIf = nullptr;
  if (!repeatLoop) {
    wrappingIf = dyn_cast<scf::IfOp>(epochOp->getParentOp());
    if (!wrappingIf)
      return false;
    repeatLoop = dyn_cast<scf::ForOp>(wrappingIf->getParentOp());
  }
  if (!repeatLoop)
    return false;
  if (repeatLoop.getNumResults() != 0 || !repeatLoop.getInitArgs().empty())
    return false;
  if (!repeatLoop.getInductionVar().use_empty())
    return false;

  std::optional<int64_t> tripCount =
      ::mlir::carts::getStaticTripCount(repeatLoop.getOperation());
  if (!tripCount || *tripCount < 2)
    return false;

  Value loopIv = repeatLoop.getInductionVar();
  Block *loopBody = repeatLoop.getBody();

  SmallVector<Operation *> prefixOps;
  SmallVector<Operation *> tailOps;

  if (!wrappingIf) {
    if (epochOp->getBlock() != loopBody)
      return false;

    bool seenEpoch = false;
    for (Operation &op : loopBody->without_terminator()) {
      if (&op == epochOp.getOperation()) {
        seenEpoch = true;
        continue;
      }
      if (!seenEpoch) {
        prefixOps.push_back(&op);
        continue;
      }
      tailOps.push_back(&op);
    }
    if (!seenEpoch)
      return false;
  } else {
    if (!wrappingIf.getElseRegion().empty() &&
        !wrappingIf.getElseRegion().front().without_terminator().empty())
      return false;
    if (wrappingIf->getBlock() != loopBody ||
        epochOp->getBlock() != &wrappingIf.getThenRegion().front())
      return false;
    if (ValueAnalysis::dependsOn(wrappingIf.getCondition(), loopIv))
      return false;

    bool seenIf = false;
    for (Operation &op : loopBody->without_terminator()) {
      if (&op == wrappingIf.getOperation()) {
        seenIf = true;
        continue;
      }
      if (seenIf)
        return false;
      prefixOps.push_back(&op);
    }

    bool seenEpoch = false;
    for (Operation &op :
         wrappingIf.getThenRegion().front().without_terminator()) {
      if (&op == epochOp.getOperation()) {
        seenEpoch = true;
        continue;
      }
      if (!seenEpoch) {
        if (!isAmortizableSetupOp(&op, loopIv))
          return false;
        continue;
      }
      tailOps.push_back(&op);
    }
    if (!seenEpoch)
      return false;
  }

  if (!llvm::all_of(prefixOps, [&](Operation *op) {
        return isAmortizableSetupOp(op, loopIv);
      }))
    return false;

  for (Operation *tailOp : tailOps) {
    if (!isAmortizableTailOp(tailOp, loopIv))
      return false;
  }

  if (!epochSupportsRepetitionAmortization(epochOp))
    return false;

  SmallVector<EdtOp> edts;
  epochOp.walk([&](EdtOp edt) { edts.push_back(edt); });
  if (edts.empty())
    return false;
  for (EdtOp edt : edts) {
    if (!canWrapEdtBodyWithRepeatLoop(edt))
      return false;
  }

  for (Operation *prefixOp : prefixOps)
    prefixOp->moveBefore(repeatLoop);
  for (EdtOp edt : edts)
    wrapEdtBodyWithRepeatLoop(edt, *tripCount);
  if (wrappingIf) {
    wrappingIf->moveBefore(repeatLoop);
  } else {
    epochOp->moveBefore(repeatLoop);
    for (Operation *tailOp : tailOps)
      tailOp->moveBefore(repeatLoop);
  }
  if (repeatLoop.getBody()->without_terminator().empty())
    repeatLoop.erase();

  ARTS_INFO("  Amortized repeated epoch loop (trip count = " << *tripCount
                                                             << ")");
  return true;
}

} // namespace mlir::carts::arts::epoch_opt
