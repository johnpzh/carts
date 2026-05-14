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

struct CpsGroupInfo {
  int64_t expectedCount = -1;
  SmallVector<arts::sde::SdeSuIterateOp, 4> stages;
  DenseMap<int64_t, arts::sde::SdeSuIterateOp> byIndex;
};

struct VerifySdeCpsPlanPass
    : public arts::impl::VerifySdeCpsPlanBase<VerifySdeCpsPlanPass> {
  void runOnOperation() override {
    DenseMap<Operation *, DenseMap<int64_t, CpsGroupInfo>> groupsByScope;
    bool failed = false;

    getOperation().walk([&](arts::sde::SdeSuIterateOp op) {
      if (!op.getCpsGroupIdAttr())
        return;

      Operation *scope = op->getParentOfType<func::FuncOp>();
      if (!scope)
        scope = getOperation();

      int64_t groupId = op.getCpsGroupIdAttr().getInt();
      int64_t stageIndex = op.getCpsStageIndexAttr().getInt();
      int64_t stageCount = op.getCpsStageCountAttr().getInt();
      CpsGroupInfo &group = groupsByScope[scope][groupId];
      group.stages.push_back(op);

      if (group.expectedCount == -1) {
        group.expectedCount = stageCount;
      } else if (group.expectedCount != stageCount) {
        op.emitError() << "sde.cps group " << groupId
                       << " has inconsistent sde.cps_stage_count";
        failed = true;
      }

      auto inserted = group.byIndex.try_emplace(stageIndex, op);
      if (!inserted.second) {
        op.emitError() << "sde.cps group " << groupId
                       << " has duplicate sde.cps_stage_index " << stageIndex;
        failed = true;
      }
    });

    for (auto &scopeEntry : groupsByScope) {
      for (auto &groupEntry : scopeEntry.second) {
        int64_t groupId = groupEntry.first;
        CpsGroupInfo &group = groupEntry.second;
        if (group.expectedCount < 0 || group.stages.empty())
          continue;

        if (static_cast<int64_t>(group.stages.size()) != group.expectedCount) {
          group.stages.front().emitError()
              << "sde.cps group " << groupId << " has "
              << group.stages.size() << " stage(s), expected "
              << group.expectedCount;
          failed = true;
        }

        for (int64_t index = 0; index < group.expectedCount; ++index) {
          if (group.byIndex.contains(index))
            continue;
          group.stages.front().emitError()
              << "sde.cps group " << groupId
              << " is missing sde.cps_stage_index " << index;
          failed = true;
        }
      }
    }

    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::arts::sde::createVerifySdeCpsPlanPass() {
  return std::make_unique<VerifySdeCpsPlanPass>();
}
