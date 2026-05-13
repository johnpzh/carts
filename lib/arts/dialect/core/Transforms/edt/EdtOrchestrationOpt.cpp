///==========================================================================///
/// File: EdtOrchestrationOpt.cpp
///
/// Recognize repeated sibling waves inside timestep-style loops and stamp a
/// canonical orchestration contract for downstream lowering.
///==========================================================================///

#define GEN_PASS_DEF_EDTORCHESTRATIONOPT
#include "arts/Dialect.h"
#include "arts/dialect/core/Transforms/edt/EdtParallelSplitLowering.h"
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/EdtUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_orchestration_opt);

#include "llvm/ADT/Statistic.h"
static llvm::Statistic numWaveGroupsStamped{
    "edt_orchestration_opt", "NumWaveGroupsStamped",
    "Number of repeated-wave orchestration groups stamped"};
static llvm::Statistic numWaveEdtsStamped{
    "edt_orchestration_opt", "NumWaveEdtsStamped",
    "Number of parallel EDT waves participating in orchestration groups"};

using namespace mlir;
using namespace mlir::arts;

namespace {

static bool isAllowedInterstitialOp(Operation *op) {
  if (isa<DbAcquireOp, LoweringContractOp>(op))
    return true;
  return op->hasTrait<OpTrait::ConstantLike>();
}

struct RepeatedWaveSignature {
  EdtConcurrency concurrency;
  std::optional<int64_t> workers;
  std::optional<int64_t> workersPerNode;
  std::optional<ArtsDepPattern> depPattern;
  std::optional<EdtDistributionKind> distributionKind;
  std::optional<EdtDistributionPattern> distributionPattern;
  Value route;
};

static std::optional<RepeatedWaveSignature>
getRepeatedWaveSignature(EdtOp edt) {
  if (!edt || edt.getType() != EdtType::parallel)
    return std::nullopt;

  ParallelRegionSplitAnalysis analysis =
      ParallelRegionSplitAnalysis::analyze(edt);
  if (!analysis.needsSplit() || analysis.hasWorkBefore() ||
      analysis.hasWorkAfter() || analysis.forOps.size() != 1)
    return std::nullopt;

  ForOp forOp = analysis.forOps.front();
  if (!forOp.getReductionAccumulators().empty())
    return std::nullopt;
  auto repetition = getPlanRepetitionStructureAttr(forOp.getOperation());
  if (!repetition ||
      repetition.getValue() != ArtsPlanRepetitionStructure::full_timestep) {
    return std::nullopt;
  }
  if (edt->hasAttr(AttrNames::Operation::ControlDep) ||
      forOp->hasAttr(AttrNames::Operation::ControlDep))
    return std::nullopt;

  auto depPattern = getDepPattern(forOp.getOperation());
  auto distributionKind = getEdtDistributionKind(forOp.getOperation());
  auto distributionPattern = getEdtDistributionPattern(forOp.getOperation());
  if (!depPattern || !distributionKind || !distributionPattern)
    return std::nullopt;
  if (!isUniformFamilyDepPattern(*depPattern) ||
      *distributionPattern != EdtDistributionPattern::uniform)
    return std::nullopt;

  return RepeatedWaveSignature{edt.getConcurrency(),
                               getWorkers(edt.getOperation()),
                               getWorkersPerNode(edt.getOperation()),
                               depPattern,
                               distributionKind,
                               distributionPattern,
                               edt.getRoute()};
}

static bool compatibleSignature(const RepeatedWaveSignature &lhs,
                                const RepeatedWaveSignature &rhs) {
  if (lhs.concurrency != rhs.concurrency)
    return false;
  if (lhs.workers != rhs.workers || lhs.workersPerNode != rhs.workersPerNode)
    return false;
  if (lhs.depPattern != rhs.depPattern ||
      lhs.distributionKind != rhs.distributionKind ||
      lhs.distributionPattern != rhs.distributionPattern) {
    return false;
  }
  if (static_cast<bool>(lhs.route) != static_cast<bool>(rhs.route))
    return false;
  if (lhs.route && !ValueAnalysis::sameValue(lhs.route, rhs.route))
    return false;
  return true;
}

static void stampWaveGroup(ArrayRef<EdtOp> group, uint64_t groupId) {
  auto *ctx = group.front()->getContext();
  auto kindAttr = StringAttr::get(ctx, AttrNames::Operation::RepeatedWaveGroup);
  auto i64Type = IntegerType::get(ctx, 64);
  auto groupAttr = IntegerAttr::get(i64Type, groupId);
  auto countAttr = IntegerAttr::get(i64Type, group.size());

  for (auto [waveIndex, edt] : llvm::enumerate(group)) {
    auto waveAttr = IntegerAttr::get(i64Type, static_cast<int64_t>(waveIndex));
    edt->setAttr(AttrNames::Operation::Orchestration::Kind, kindAttr);
    edt->setAttr(AttrNames::Operation::Orchestration::GroupId, groupAttr);
    edt->setAttr(AttrNames::Operation::Orchestration::WaveIndex, waveAttr);
    edt->setAttr(AttrNames::Operation::Orchestration::WaveCount, countAttr);

    for (ForOp forOp : EdtUtils::getTopLevelForOps(edt)) {
      forOp->setAttr(AttrNames::Operation::Orchestration::Kind, kindAttr);
      forOp->setAttr(AttrNames::Operation::Orchestration::GroupId, groupAttr);
      forOp->setAttr(AttrNames::Operation::Orchestration::WaveIndex, waveAttr);
      forOp->setAttr(AttrNames::Operation::Orchestration::WaveCount, countAttr);
    }
  }
}

struct EdtOrchestrationOptPass
    : public impl::EdtOrchestrationOptBase<EdtOrchestrationOptPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    uint64_t nextGroupId = 0;

    module.walk([&](scf::ForOp loopOp) {
      SmallVector<EdtOp, 4> currentCluster;
      std::optional<RepeatedWaveSignature> clusterSignature;

      auto flushCluster = [&]() {
        if (currentCluster.size() < 2) {
          currentCluster.clear();
          clusterSignature.reset();
          return;
        }
        stampWaveGroup(currentCluster, nextGroupId++);
        numWaveGroupsStamped += 1;
        numWaveEdtsStamped += currentCluster.size();
        ARTS_DEBUG("Stamped repeated-wave orchestration group with "
                   << currentCluster.size() << " waves");
        currentCluster.clear();
        clusterSignature.reset();
      };

      Block &body = loopOp.getRegion().front();
      for (Operation &op : body.without_terminator()) {
        if (auto edt = dyn_cast<EdtOp>(&op)) {
          if (auto signature = getRepeatedWaveSignature(edt)) {
            if (!currentCluster.empty() &&
                !compatibleSignature(*clusterSignature, *signature))
              flushCluster();
            currentCluster.push_back(edt);
            clusterSignature = *signature;
            continue;
          }
          flushCluster();
          continue;
        }

        if (!currentCluster.empty() && isAllowedInterstitialOp(&op))
          continue;

        flushCluster();
      }

      flushCluster();
    });
  }
};

} // namespace

namespace mlir {
namespace arts {
std::unique_ptr<Pass> createEdtOrchestrationOptPass() {
  return std::make_unique<EdtOrchestrationOptPass>();
}
} // namespace arts
} // namespace mlir
