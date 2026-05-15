///==========================================================================///
/// File: VerifySdeCpsPlan.cpp
///==========================================================================///

#include "carts/dialect/sde/IR/SdeDialect.h"
#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_VERIFYSDECPSPLAN
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <iterator>

using namespace mlir;
using namespace mlir::carts;

namespace {

constexpr llvm::StringLiteral kCpsCandidateGroupId = "cps_candidate_group_id";
constexpr llvm::StringLiteral kCpsCandidateStageIndex =
    "cps_candidate_stage_index";
constexpr llvm::StringLiteral kCpsCandidateStageCount =
    "cps_candidate_stage_count";
constexpr llvm::StringLiteral kCpsCandidateRequiresTokenizedDataflow =
    "cps_candidate_requires_tokenized_dataflow";

struct CpsGroupInfo {
  int64_t expectedCount = -1;
  SmallVector<sde::SdeSuIterateOp, 4> stages;
  DenseMap<int64_t, sde::SdeSuIterateOp> byIndex;
};

struct CandidateBoundary {
  Operation *firstContainer = nullptr;
  sde::SdeSuBarrierOp barrier;
};

static bool hasCandidateAttr(sde::SdeSuIterateOp op) {
  return op->hasAttr(kCpsCandidateGroupId) ||
         op->hasAttr(kCpsCandidateStageIndex) ||
         op->hasAttr(kCpsCandidateStageCount) ||
         op->hasAttr(kCpsCandidateRequiresTokenizedDataflow);
}

static bool isTimestepBarrier(sde::SdeSuBarrierOp barrier) {
  if (!barrier)
    return false;
  auto reason = barrier.getBarrierReason();
  return reason &&
         *reason == sde::SdeBarrierReason::timestep_stage_boundary;
}

static LogicalResult verifyCandidateAttrSet(sde::SdeSuIterateOp op) {
  bool hasGroup = op->hasAttr(kCpsCandidateGroupId);
  bool hasIndex = op->hasAttr(kCpsCandidateStageIndex);
  bool hasCount = op->hasAttr(kCpsCandidateStageCount);
  bool hasRequiresTokenizedDataflow =
      op->hasAttr(kCpsCandidateRequiresTokenizedDataflow);

  if (hasGroup || hasIndex || hasCount || hasRequiresTokenizedDataflow) {
    if (!hasGroup || !hasIndex || !hasCount || !hasRequiresTokenizedDataflow)
      return op.emitOpError()
             << "sde.cps candidate plan requires "
                "sde.cps_candidate_group_id, "
                "sde.cps_candidate_stage_index, "
                "sde.cps_candidate_stage_count, and "
                "sde.cps_candidate_requires_tokenized_dataflow together";
    if (!op->getAttrOfType<UnitAttr>(kCpsCandidateRequiresTokenizedDataflow))
      return op.emitOpError()
             << "sde.cps_candidate_requires_tokenized_dataflow must be a "
                "unit attr";

    if (op.getCpsGroupIdAttr() || op.getCpsStageIndexAttr() ||
        op.getCpsStageCountAttr())
      return op.emitOpError()
             << "sde.cps candidate plan cannot also carry a final "
                "sde.cps stage plan";

    auto repetition = op.getRepetitionStructure();
    if (!repetition ||
        *repetition != sde::SdeRepetitionStructure::full_timestep)
      return op.emitOpError()
             << "sde.cps candidate plan requires sde.repetition_structure "
                "full_timestep";

    auto strategy = op.getAsyncStrategy();
    if (!strategy || *strategy != sde::SdeAsyncStrategy::advance_edt)
      return op.emitOpError()
             << "sde.cps candidate plan requires sde.async_strategy "
                "advance_edt until tokenized dataflow exists";

    auto group = op->getAttrOfType<IntegerAttr>(kCpsCandidateGroupId);
    auto index = op->getAttrOfType<IntegerAttr>(kCpsCandidateStageIndex);
    auto count = op->getAttrOfType<IntegerAttr>(kCpsCandidateStageCount);
    if (!group || !index || !count)
      return op.emitOpError()
             << "sde.cps candidate plan attrs must be integer attrs";
    if (group.getInt() < 0)
      return op.emitOpError()
             << "sde.cps_candidate_group_id must be non-negative";
    if (count.getInt() <= 0)
      return op.emitOpError()
             << "sde.cps_candidate_stage_count must be positive";
    if (index.getInt() < 0 || index.getInt() >= count.getInt())
      return op.emitOpError() << "sde.cps_candidate_stage_index must be in "
                                 "[0, sde.cps_candidate_stage_count)";
  }

  return success();
}

static void recordGroup(
    DenseMap<Operation *, DenseMap<int64_t, CpsGroupInfo>> &groupsByScope,
    ModuleOp module, sde::SdeSuIterateOp op, int64_t groupId,
    int64_t stageIndex, int64_t stageCount, bool &hasFailure, StringRef label,
    StringRef stageCountName, StringRef stageIndexName) {
  Operation *scope = op->getParentOfType<func::FuncOp>();
  if (!scope)
    scope = module;

  CpsGroupInfo &group = groupsByScope[scope][groupId];
  group.stages.push_back(op);

  if (group.expectedCount == -1) {
    group.expectedCount = stageCount;
  } else if (group.expectedCount != stageCount) {
    op.emitError() << label << " group " << groupId << " has inconsistent "
                   << stageCountName;
    hasFailure = true;
  }

  auto inserted = group.byIndex.try_emplace(stageIndex, op);
  if (!inserted.second) {
    op.emitError() << label << " group " << groupId << " has duplicate "
                   << stageIndexName << " " << stageIndex;
    hasFailure = true;
  }
}

static void verifyGroupCompleteness(
    DenseMap<Operation *, DenseMap<int64_t, CpsGroupInfo>> &groupsByScope,
    bool &hasFailure, StringRef label, StringRef stageIndexName) {
  for (auto &scopeEntry : groupsByScope) {
    for (auto &groupEntry : scopeEntry.second) {
      int64_t groupId = groupEntry.first;
      CpsGroupInfo &group = groupEntry.second;
      if (group.expectedCount < 0 || group.stages.empty())
        continue;

      if (static_cast<int64_t>(group.stages.size()) != group.expectedCount) {
        group.stages.front().emitError()
            << label << " group " << groupId << " has " << group.stages.size()
            << " stage(s), expected " << group.expectedCount;
        hasFailure = true;
      }

      for (int64_t index = 0; index < group.expectedCount; ++index) {
        if (group.byIndex.contains(index))
          continue;
        group.stages.front().emitError()
            << label << " group " << groupId << " is missing " << stageIndexName
            << " " << index;
        hasFailure = true;
      }
    }
  }
}

static SmallVector<Operation *, 4> collectBlockAncestors(Operation *op) {
  SmallVector<Operation *, 4> ancestors;
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (current->getBlock())
      ancestors.push_back(current);
  }
  return ancestors;
}

static CandidateBoundary findTimestepBarrierBetween(
    sde::SdeSuIterateOp first, sde::SdeSuIterateOp second) {
  SmallVector<Operation *, 4> firstAncestors =
      collectBlockAncestors(first.getOperation());
  SmallVector<Operation *, 4> secondAncestors =
      collectBlockAncestors(second.getOperation());

  for (Operation *firstContainer : firstAncestors) {
    for (Operation *secondContainer : secondAncestors) {
      if (firstContainer->getBlock() != secondContainer->getBlock())
        continue;
      if (!firstContainer->isBeforeInBlock(secondContainer))
        continue;

      for (auto it = std::next(firstContainer->getIterator());
           it != secondContainer->getIterator(); ++it) {
        auto barrier = dyn_cast<sde::SdeSuBarrierOp>(&*it);
        if (isTimestepBarrier(barrier))
          return {firstContainer, barrier};
      }
    }
  }

  return {};
}

static bool isControlTokenProducedBetween(Value token,
                                          Operation *firstContainer,
                                          Operation *barrier) {
  auto producer = token.getDefiningOp<sde::SdeControlTokenOp>();
  if (!producer)
    return false;
  Operation *producerOp = producer.getOperation();
  if (producerOp->getBlock() != firstContainer->getBlock())
    return false;
  return firstContainer->isBeforeInBlock(producerOp) &&
         producerOp->isBeforeInBlock(barrier);
}

static void verifyCandidateBarrierControlEdges(
    DenseMap<Operation *, DenseMap<int64_t, CpsGroupInfo>> &groupsByScope,
    bool &hasFailure) {
  for (auto &scopeEntry : groupsByScope) {
    for (auto &groupEntry : scopeEntry.second) {
      CpsGroupInfo &group = groupEntry.second;
      if (group.expectedCount != 2)
        continue;
      auto firstIt = group.byIndex.find(0);
      auto secondIt = group.byIndex.find(1);
      if (firstIt == group.byIndex.end() || secondIt == group.byIndex.end())
        continue;

      CandidateBoundary boundary =
          findTimestepBarrierBetween(firstIt->second, secondIt->second);
      if (!boundary.barrier) {
        secondIt->second.emitError()
            << "sde.cps candidate stage pair requires sde.control_token "
               "boundary before successor stage";
        hasFailure = true;
        continue;
      }

      bool hasControlEdge = llvm::any_of(
          boundary.barrier.getTokens(), [&](Value token) {
            return isControlTokenProducedBetween(
                token, boundary.firstContainer, boundary.barrier.getOperation());
          });
      if (hasControlEdge)
        continue;

      boundary.barrier.emitError()
          << "sde.cps candidate timestep barrier requires sde.control_token "
             "produced after the previous candidate stage";
      hasFailure = true;
    }
  }
}

struct VerifySdeCpsPlanPass
    : public arts::impl::VerifySdeCpsPlanBase<VerifySdeCpsPlanPass> {
  void runOnOperation() override {
    DenseMap<Operation *, DenseMap<int64_t, CpsGroupInfo>> groupsByScope;
    DenseMap<Operation *, DenseMap<int64_t, CpsGroupInfo>>
        candidateGroupsByScope;
    bool hasFailure = false;
    ModuleOp module = getOperation();

    module.walk([&](sde::SdeSuIterateOp op) {
      if (mlir::failed(verifyCandidateAttrSet(op))) {
        hasFailure = true;
        return;
      }

      if (op.getCpsGroupIdAttr()) {
        recordGroup(groupsByScope, module, op, op.getCpsGroupIdAttr().getInt(),
                    op.getCpsStageIndexAttr().getInt(),
                    op.getCpsStageCountAttr().getInt(), hasFailure, "sde.cps",
                    "sde.cps_stage_count", "sde.cps_stage_index");
      }

      if (hasCandidateAttr(op)) {
        auto group =
            op->getAttrOfType<IntegerAttr>(kCpsCandidateGroupId).getInt();
        auto index =
            op->getAttrOfType<IntegerAttr>(kCpsCandidateStageIndex).getInt();
        auto count =
            op->getAttrOfType<IntegerAttr>(kCpsCandidateStageCount).getInt();
        recordGroup(candidateGroupsByScope, module, op, group, index, count,
                    hasFailure, "sde.cps candidate",
                    "sde.cps_candidate_stage_count",
                    "sde.cps_candidate_stage_index");
      }
    });

    verifyGroupCompleteness(groupsByScope, hasFailure, "sde.cps",
                            "sde.cps_stage_index");
    verifyGroupCompleteness(candidateGroupsByScope, hasFailure,
                            "sde.cps candidate",
                            "sde.cps_candidate_stage_index");
    verifyCandidateBarrierControlEdges(candidateGroupsByScope, hasFailure);

    if (hasFailure)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::carts::sde::createVerifySdeCpsPlanPass() {
  return std::make_unique<VerifySdeCpsPlanPass>();
}
