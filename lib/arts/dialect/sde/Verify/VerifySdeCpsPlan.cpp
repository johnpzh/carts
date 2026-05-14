///==========================================================================///
/// File: VerifySdeCpsPlan.cpp
///==========================================================================///

#include "arts/dialect/sde/IR/SdeDialect.h"
#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_VERIFYSDECPSPLAN
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

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
  SmallVector<arts::sde::SdeSuIterateOp, 4> stages;
  DenseMap<int64_t, arts::sde::SdeSuIterateOp> byIndex;
};

static bool hasCandidateAttr(arts::sde::SdeSuIterateOp op) {
  return op->hasAttr(kCpsCandidateGroupId) ||
         op->hasAttr(kCpsCandidateStageIndex) ||
         op->hasAttr(kCpsCandidateStageCount) ||
         op->hasAttr(kCpsCandidateRequiresTokenizedDataflow);
}

static LogicalResult verifyCandidateAttrSet(arts::sde::SdeSuIterateOp op) {
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

    if (op.getCpsGroupIdAttr() || op.getCpsStageIndexAttr() ||
        op.getCpsStageCountAttr())
      return op.emitOpError()
             << "sde.cps candidate plan cannot also carry a final "
                "sde.cps stage plan";

    auto repetition = op.getRepetitionStructure();
    if (!repetition ||
        *repetition != arts::sde::SdeRepetitionStructure::full_timestep)
      return op.emitOpError()
             << "sde.cps candidate plan requires sde.repetition_structure "
                "full_timestep";

    auto strategy = op.getAsyncStrategy();
    if (!strategy || *strategy != arts::sde::SdeAsyncStrategy::advance_edt)
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
    ModuleOp module, arts::sde::SdeSuIterateOp op, int64_t groupId,
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

struct VerifySdeCpsPlanPass
    : public arts::impl::VerifySdeCpsPlanBase<VerifySdeCpsPlanPass> {
  void runOnOperation() override {
    DenseMap<Operation *, DenseMap<int64_t, CpsGroupInfo>> groupsByScope;
    DenseMap<Operation *, DenseMap<int64_t, CpsGroupInfo>>
        candidateGroupsByScope;
    bool hasFailure = false;
    ModuleOp module = getOperation();

    module.walk([&](arts::sde::SdeSuIterateOp op) {
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

    if (hasFailure)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::arts::sde::createVerifySdeCpsPlanPass() {
  return std::make_unique<VerifySdeCpsPlanPass>();
}
