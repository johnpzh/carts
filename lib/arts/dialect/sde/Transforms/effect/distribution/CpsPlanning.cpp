///==========================================================================///
/// File: CpsPlanning.cpp
///
/// Mark SDE-owned CPS timestep candidates after barrier analysis. This pass
/// records candidate grouping and makes barrier-delimited stage completion
/// explicit with SDE control tokens; final `cps_chain` remains reserved until
/// SDE has rewritten all scalar/data/control carries.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_CPSPLANNING
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "arts/utils/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"

#include <cassert>

ARTS_DEBUG_SETUP(cps_planning);

using namespace mlir;
using namespace mlir::arts;

static llvm::Statistic numCpsCandidateGroups{
    "cps_planning", "NumCpsCandidateGroups",
    "Number of SDE CPS candidate groups stamped"};

namespace {

constexpr llvm::StringLiteral kCpsCandidateGroupId = "cps_candidate_group_id";
constexpr llvm::StringLiteral kCpsCandidateStageIndex =
    "cps_candidate_stage_index";
constexpr llvm::StringLiteral kCpsCandidateStageCount =
    "cps_candidate_stage_count";
constexpr llvm::StringLiteral kCpsCandidateRequiresTokenizedDataflow =
    "cps_candidate_requires_tokenized_dataflow";

static sde::SdeSuIterateOp findSuIterate(Operation *op) {
  if (auto suIt = dyn_cast_or_null<sde::SdeSuIterateOp>(op))
    return suIt;
  if (auto dist = dyn_cast_or_null<sde::SdeSuDistributeOp>(op)) {
    sde::SdeSuIterateOp result;
    dist.getBody().walk([&](sde::SdeSuIterateOp suIt) { result = suIt; });
    return result;
  }
  sde::SdeSuIterateOp result;
  bool multiple = false;
  if (op) {
    op->walk([&](sde::SdeSuIterateOp suIt) {
      if (result && result.getOperation() != suIt.getOperation()) {
        multiple = true;
        return WalkResult::interrupt();
      }
      result = suIt;
      return WalkResult::advance();
    });
  }
  if (multiple)
    return {};
  return result;
}

static bool hasCandidateAttrs(sde::SdeSuIterateOp op) {
  return op && (op->hasAttr(kCpsCandidateGroupId) ||
                op->hasAttr(kCpsCandidateStageIndex) ||
                op->hasAttr(kCpsCandidateStageCount) ||
                op->hasAttr(kCpsCandidateRequiresTokenizedDataflow));
}

static bool hasFinalCpsStageAttrs(sde::SdeSuIterateOp op) {
  return op && (op.getCpsGroupIdAttr() || op.getCpsStageIndexAttr() ||
                op.getCpsStageCountAttr());
}

static bool isSdeCpsCandidateStage(sde::SdeSuIterateOp op) {
  if (!op || hasFinalCpsStageAttrs(op))
    return false;
  auto repetition = op.getRepetitionStructure();
  if (!repetition || *repetition != sde::SdeRepetitionStructure::full_timestep)
    return false;
  auto strategy = op.getAsyncStrategy();
  return strategy && *strategy == sde::SdeAsyncStrategy::advance_edt;
}

static bool isTimestepBarrier(sde::SdeSuBarrierOp barrier) {
  if (!barrier)
    return false;
  auto reason = barrier.getBarrierReason();
  return reason && *reason == sde::SdeBarrierReason::timestep_stage_boundary;
}

static Operation *findPreviousSuContainer(Operation *anchor) {
  Block *block = anchor ? anchor->getBlock() : nullptr;
  if (!block)
    return nullptr;
  for (auto it = Block::reverse_iterator(anchor->getIterator());
       it != block->rend(); ++it) {
    if (findSuIterate(&*it))
      return &*it;
    if (!isMemoryEffectFree(&*it))
      break;
  }
  return nullptr;
}

static Operation *findNextSuContainer(Operation *anchor) {
  Block *block = anchor ? anchor->getBlock() : nullptr;
  if (!block)
    return nullptr;
  for (auto it = std::next(anchor->getIterator()); it != block->end(); ++it) {
    if (findSuIterate(&*it))
      return &*it;
    if (!isMemoryEffectFree(&*it))
      break;
  }
  return nullptr;
}

static int64_t findNextCandidateGroupId(ModuleOp module) {
  int64_t nextId = 0;
  module.walk([&](sde::SdeSuIterateOp op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(kCpsCandidateGroupId))
      nextId = std::max(nextId, attr.getInt() + 1);
    if (auto attr = op.getCpsGroupIdAttr())
      nextId = std::max(nextId, attr.getInt() + 1);
  });
  return nextId;
}

static bool canStampCandidatePair(sde::SdeSuIterateOp first,
                                  sde::SdeSuIterateOp second) {
  if (!isSdeCpsCandidateStage(first) || !isSdeCpsCandidateStage(second))
    return false;
  if (hasCandidateAttrs(first) || hasCandidateAttrs(second))
    return false;
  return true;
}

static void stampCandidatePair(sde::SdeSuIterateOp first,
                               sde::SdeSuIterateOp second, int64_t groupId) {
  assert(canStampCandidatePair(first, second) &&
         "candidate pair preconditions must be checked before stamping");

  MLIRContext *ctx = first.getContext();
  auto setStageAttrs = [&](sde::SdeSuIterateOp op, int64_t stageIndex) {
    op->setAttr(kCpsCandidateGroupId,
                IntegerAttr::get(IntegerType::get(ctx, 64), groupId));
    op->setAttr(kCpsCandidateStageIndex,
                IntegerAttr::get(IntegerType::get(ctx, 64), stageIndex));
    op->setAttr(kCpsCandidateStageCount,
                IntegerAttr::get(IntegerType::get(ctx, 64), 2));
    op->setAttr(kCpsCandidateRequiresTokenizedDataflow, UnitAttr::get(ctx));
  };

  setStageAttrs(first, 0);
  setStageAttrs(second, 1);
}

static bool attachStageCompletionToBarrier(Operation *stageContainer,
                                           sde::SdeSuBarrierOp barrier) {
  if (!stageContainer || !barrier)
    return false;
  if (stageContainer->getBlock() != barrier->getBlock())
    return false;

  OpBuilder builder(stageContainer->getContext());
  builder.setInsertionPointAfter(stageContainer);
  auto token = sde::SdeControlTokenOp::create(
      builder, stageContainer->getLoc(),
      sde::CompletionType::get(stageContainer->getContext()));

  SmallVector<Value> tokens(barrier.getTokens().begin(),
                            barrier.getTokens().end());
  tokens.push_back(token.getToken());

  builder.setInsertionPoint(barrier);
  sde::SdeSuBarrierOp::create(builder, barrier.getLoc(), tokens,
                              barrier.getBarrierEliminatedAttr(),
                              barrier.getBarrierReasonAttr());
  barrier.erase();
  return true;
}

static unsigned stampBarrierDelimitedCandidates(ModuleOp module,
                                                int64_t &nextGroupId) {
  unsigned stamped = 0;
  SmallVector<sde::SdeSuBarrierOp> barriers;
  module.walk([&](sde::SdeSuBarrierOp barrier) {
    if (isTimestepBarrier(barrier))
      barriers.push_back(barrier);
  });

  for (sde::SdeSuBarrierOp barrier : barriers) {
    Operation *firstContainer = findPreviousSuContainer(barrier);
    Operation *secondContainer = findNextSuContainer(barrier);
    auto first = findSuIterate(firstContainer);
    auto second = findSuIterate(secondContainer);
    if (canStampCandidatePair(first, second) &&
        attachStageCompletionToBarrier(firstContainer, barrier)) {
      stampCandidatePair(first, second, nextGroupId);
      ++stamped;
      ++nextGroupId;
    }
  }
  return stamped;
}

static bool isTransparentBetweenTimestepCandidates(Operation *op) {
  if (!op)
    return false;
  if (isa<sde::SdeYieldOp>(op))
    return true;
  if (auto barrier = dyn_cast<sde::SdeSuBarrierOp>(op))
    return !isTimestepBarrier(barrier);
  if (op->getNumRegions() != 0) {
    auto effects = sde::collectStructuredMemoryEffects(op);
    return !effects.hasUnknownEffects && effects.empty();
  }
  return !sde::hasUnmodeledMemoryEffect(op);
}

static unsigned stampAdjacentLoopCandidates(scf::ForOp loop,
                                            int64_t &nextGroupId) {
  Operation *previous = nullptr;
  unsigned stamped = 0;

  for (Operation &op : loop.getBody()->without_terminator()) {
    if (findSuIterate(&op)) {
      if (previous) {
        auto first = findSuIterate(previous);
        auto second = findSuIterate(&op);
        if (canStampCandidatePair(first, second)) {
          stampCandidatePair(first, second, nextGroupId);
          ++stamped;
          ++nextGroupId;
        }
      }
      previous = &op;
      continue;
    }

    if (isTransparentBetweenTimestepCandidates(&op))
      continue;

    previous = nullptr;
  }

  return stamped;
}

struct CpsPlanningPass : public arts::impl::CpsPlanningBase<CpsPlanningPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    int64_t nextGroupId = findNextCandidateGroupId(module);
    unsigned stamped = stampBarrierDelimitedCandidates(module, nextGroupId);

    module.walk([&](scf::ForOp loop) {
      stamped += stampAdjacentLoopCandidates(loop, nextGroupId);
    });

    numCpsCandidateGroups += stamped;
    ARTS_INFO("CpsPlanning: stamped " << stamped
                                      << " SDE CPS candidate group(s)");
  }
};

} // namespace

namespace mlir::arts::sde {

std::unique_ptr<Pass> createCpsPlanningPass() {
  return std::make_unique<CpsPlanningPass>();
}

} // namespace mlir::arts::sde
