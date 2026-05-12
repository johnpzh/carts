///==========================================================================///
/// File: EpochOpt.cpp
///
/// Epoch optimization pass driver. Transform implementations live in
/// neighboring epoch-specific compilation units.
///==========================================================================///

#define GEN_PASS_DEF_EPOCHOPT

#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Transforms/EpochOptInternal.h"
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/Debug.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Statistic.h"

ARTS_DEBUG_SETUP(epoch_opt);

using namespace mlir::arts;

using namespace mlir;
using namespace mlir::arts::epoch_opt;

namespace {

struct AsyncLoopTransformSpec {
  llvm::StringLiteral debugLabel;
  EpochAsyncLoopStrategy selectedStrategy;
  bool (*apply)(scf::ForOp, const EpochAnalysis &);
};

unsigned runAsyncLoopTransform(ModuleOp module, EpochAnalysis &epochAnalysis,
                               const AsyncLoopTransformSpec &spec) {
  unsigned transformed = 0;
  SmallVector<scf::ForOp> forOps;
  module.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });
  ARTS_INFO(spec.debugLabel << ": Evaluating " << forOps.size() << " loop(s)");
  for (scf::ForOp forOp : forOps) {
    EpochAsyncLoopDecision asyncDecision =
        epochAnalysis.evaluateAsyncLoopStrategy(forOp);
    if (asyncDecision.strategy != spec.selectedStrategy) {
      ARTS_INFO(spec.debugLabel
                << ": selector chose "
                << asyncLoopStrategyToString(asyncDecision.strategy) << " ("
                << asyncDecision.rationale << "), skipping");
      continue;
    }
    ARTS_INFO(spec.debugLabel << ": Applying to loop (strategy matched)");
    if (spec.apply(forOp, epochAnalysis))
      ++transformed;
  }
  return transformed;
}

struct EpochOptPass : public impl::EpochOptBase<EpochOptPass> {
  EpochOptPass() : EpochOptPass(/*AM=*/nullptr) {}
  EpochOptPass(const EpochOptPass &other)
      : impl::EpochOptBase<EpochOptPass>(other), AM(other.AM),
        numRepeatedEpochLoopsAmortized(
            this, "num-repeated-epoch-loops-amortized",
            "Number of repeated epoch loops amortized"),
        numContinuationEdtsCreated(
            this, "num-continuation-edts-created",
            "Number of continuation EDTs created from epoch tails"),
        numCpsDriverLoopsTransformed(
            this, "num-cps-driver-loops-transformed",
            "Number of loops wrapped in CPS driver form"),
        numCpsChainLoopsTransformed(
            this, "num-cps-chain-loops-transformed",
            "Number of loops transformed into CPS continuation chains"),
        numEpochsNarrowed(this, "num-epochs-narrowed",
                          "Number of epochs split into narrower scopes"),
        numEpochsCreatedByNarrowing(
            this, "num-epochs-created-by-narrowing",
            "Number of new epochs created while narrowing epoch scopes"),
        numEpochPairsFused(this, "num-epoch-pairs-fused",
                           "Number of consecutive epoch pairs fused"),
        numWorkerLoopsFused(this, "num-worker-loops-fused",
                            "Number of worker loop pairs fused inside epochs") {
  }

  explicit EpochOptPass(mlir::arts::AnalysisManager *AM)
      : AM(AM), numRepeatedEpochLoopsAmortized(
                    this, "num-repeated-epoch-loops-amortized",
                    "Number of repeated epoch loops amortized"),
        numContinuationEdtsCreated(
            this, "num-continuation-edts-created",
            "Number of continuation EDTs created from epoch tails"),
        numCpsDriverLoopsTransformed(
            this, "num-cps-driver-loops-transformed",
            "Number of loops wrapped in CPS driver form"),
        numCpsChainLoopsTransformed(
            this, "num-cps-chain-loops-transformed",
            "Number of loops transformed into CPS continuation chains"),
        numEpochsNarrowed(this, "num-epochs-narrowed",
                          "Number of epochs split into narrower scopes"),
        numEpochsCreatedByNarrowing(
            this, "num-epochs-created-by-narrowing",
            "Number of new epochs created while narrowing epoch scopes"),
        numEpochPairsFused(this, "num-epoch-pairs-fused",
                           "Number of consecutive epoch pairs fused"),
        numWorkerLoopsFused(this, "num-worker-loops-fused",
                            "Number of worker loop pairs fused inside epochs") {
  }

  EpochOptPass(mlir::arts::AnalysisManager *AM, bool narrowing, bool fusion,
               bool loopFusion, bool amortization, bool continuation,
               bool cpsDriver, bool cpsChain = false)
      : EpochOptPass(AM) {
    enableEpochNarrowing = narrowing;
    enableEpochFusion = fusion;
    enableLoopFusion = loopFusion;
    enableAmortization = amortization;
    enableContinuation = continuation;
    enableCPSDriver = cpsDriver;
    enableCPSChain = cpsChain;
  }

  mlir::arts::AnalysisManager &getEpochAnalysisManager(ModuleOp module) {
    if (!AM) {
      ownedAM = std::make_unique<mlir::arts::AnalysisManager>(module);
      AM = ownedAM.get();
    }
    return *AM;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool changed = false;
    mlir::arts::AnalysisManager &epochAM = getEpochAnalysisManager(module);
    EpochAnalysis &epochAnalysis = epochAM.getEpochAnalysis();

    ARTS_INFO_HEADER(EpochOptPass);

    SmallVector<scf::ForOp> asyncLoops;
    module.walk([&](scf::ForOp forOp) { asyncLoops.push_back(forOp); });
    for (scf::ForOp forOp : asyncLoops) {
      EpochAsyncLoopDecision asyncDecision =
          epochAnalysis.evaluateAsyncLoopStrategy(forOp);
      normalizeAsyncLoopPlanAttrs(forOp, asyncDecision.strategy);
    }

    // Amortization runs before CPS transforms: if amortization succeeds,
    // the loop is consumed and CPS is unnecessary. If it fails, the loop
    // falls through to CPS chain as before.
    if (enableAmortization) {
      SmallVector<EpochOp> amortEpochOps;
      module.walk([&](EpochOp epochOp) { amortEpochOps.push_back(epochOp); });
      ARTS_INFO("Found " << amortEpochOps.size()
                         << " epoch operations to analyze for amortization");
      unsigned amortized = 0;
      for (EpochOp epochOp : amortEpochOps) {
        if (tryAmortizeRepeatedEpochLoop(epochOp))
          ++amortized;
      }
      if (amortized > 0) {
        numRepeatedEpochLoopsAmortized += amortized;
        ARTS_INFO("Amortized " << amortized << " repeated epoch loop(s)");
        changed = true;
      }
    }

    if (enableCPSChain) {
      static constexpr AsyncLoopTransformSpec kCpsChainSpec = {
          "CPS Chain", EpochAsyncLoopStrategy::CpsChain, &tryCPSChainTransform};
      unsigned chainTransformed =
          runAsyncLoopTransform(module, epochAnalysis, kCpsChainSpec);
      if (chainTransformed > 0) {
        numCpsChainLoopsTransformed += chainTransformed;
        ARTS_INFO("CPS-chain-transformed " << chainTransformed
                                           << " epoch loop(s)");
        changed = true;
      }
    }

    if (enableCPSDriver) {
      static constexpr AsyncLoopTransformSpec kCpsDriverSpec = {
          "CPS Driver", EpochAsyncLoopStrategy::AdvanceEdt,
          &tryCPSLoopTransform};
      unsigned cpsTransformed =
          runAsyncLoopTransform(module, epochAnalysis, kCpsDriverSpec);
      if (cpsTransformed > 0) {
        numCpsDriverLoopsTransformed += cpsTransformed;
        ARTS_INFO("CPS-transformed " << cpsTransformed << " epoch loop(s)");
        changed = true;
      }
    }

    if (enableContinuation) {
      SmallVector<EpochOp> epochOps;
      module.walk([&](EpochOp epochOp) { epochOps.push_back(epochOp); });
      ARTS_INFO("Found " << epochOps.size()
                         << " epoch operations to analyze for scheduling");
      unsigned transformed = 0;
      for (EpochOp epochOp : epochOps) {
        if (!epochOp || !epochOp->getBlock())
          continue;

        EpochContinuationDecision decision = epochAnalysis.evaluateContinuation(
            epochOp, dyn_cast_or_null<EpochOp>(epochOp->getPrevNode()),
            /*continuationEnabled=*/true);
        if (!decision.eligible) {
          ARTS_DEBUG(
              "  Epoch not eligible for continuation: " << decision.rationale);
          continue;
        }

        ARTS_INFO("  Transforming eligible epoch to continuation form");
        if (succeeded(transformToContinuation(epochOp, decision)))
          ++transformed;
      }
      if (transformed > 0) {
        numContinuationEdtsCreated += transformed;
        ARTS_INFO("Transformed " << transformed
                                 << " epochs to continuation form");
        changed = true;
      }
    }

    if (enableEpochNarrowing) {
      EpochNarrowingCounts narrowingCounts = narrowEpochScopes(module);
      if (narrowingCounts.epochsNarrowed != 0) {
        numEpochsNarrowed += narrowingCounts.epochsNarrowed;
        numEpochsCreatedByNarrowing += narrowingCounts.newEpochsCreated;
        ARTS_INFO("Epoch scope narrowing applied");
        changed = true;
      }
    }

    if (enableEpochFusion) {
      unsigned fusedEpochPairs = processRegionForEpochFusion(
          module.getRegion(), epochAnalysis, enableContinuation);
      if (fusedEpochPairs != 0) {
        numEpochPairsFused += fusedEpochPairs;
        ARTS_INFO("Epoch fusion applied");
        changed = true;
      }
    }

    if (enableLoopFusion) {
      unsigned fusedWorkerLoops = 0;
      module.walk([&](EpochOp epoch) {
        fusedWorkerLoops += fuseWorkerLoopsInEpoch(epoch);
      });
      if (fusedWorkerLoops != 0) {
        numWorkerLoopsFused += fusedWorkerLoops;
        ARTS_INFO("Epoch worker loop fusion applied");
        changed = true;
      }
    }

    ARTS_INFO_FOOTER(EpochOptPass);

    if (!changed)
      markAllAnalysesPreserved();
  }

private:
  mlir::arts::AnalysisManager *AM = nullptr;
  Statistic numRepeatedEpochLoopsAmortized;
  Statistic numContinuationEdtsCreated;
  Statistic numCpsDriverLoopsTransformed;
  Statistic numCpsChainLoopsTransformed;
  Statistic numEpochsNarrowed;
  Statistic numEpochsCreatedByNarrowing;
  Statistic numEpochPairsFused;
  Statistic numWorkerLoopsFused;
  std::unique_ptr<mlir::arts::AnalysisManager> ownedAM;
};

} // namespace

namespace mlir {
namespace arts {

std::unique_ptr<Pass> createEpochOptPass() {
  return std::make_unique<EpochOptPass>();
}

std::unique_ptr<Pass> createEpochOptPass(mlir::arts::AnalysisManager *AM) {
  return std::make_unique<EpochOptPass>(AM);
}

std::unique_ptr<Pass> createEpochOptPass(mlir::arts::AnalysisManager *AM,
                                         bool amortization, bool continuation,
                                         bool cpsDriver, bool cpsChain) {
  return std::make_unique<EpochOptPass>(AM, /*narrowing=*/true, /*fusion=*/true,
                                        /*loopFusion=*/true, amortization,
                                        continuation, cpsDriver, cpsChain);
}

std::unique_ptr<Pass>
createEpochOptSchedulingPass(mlir::arts::AnalysisManager *AM, bool amortization,
                             bool continuation, bool cpsDriver, bool cpsChain) {
  return std::make_unique<EpochOptPass>(
      AM, /*narrowing=*/false, /*fusion=*/false, /*loopFusion=*/false,
      amortization, continuation, cpsDriver, cpsChain);
}

} // namespace arts
} // namespace mlir
