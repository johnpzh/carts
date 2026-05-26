///==========================================================================///
/// File: EpochOpt.cpp
///
/// Epoch optimization pass driver. Transform implementations live in
/// neighboring epoch-specific compilation units.
///==========================================================================///

#define GEN_PASS_DEF_EPOCHOPT

#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "EpochOptInternal.h"
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "carts/utils/Debug.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Statistic.h"

ARTS_DEBUG_SETUP(epoch_opt);

using namespace mlir::carts;
using namespace mlir::carts::arts;

using namespace mlir;
using namespace mlir::carts::arts::epoch_opt;

namespace {

struct EpochOptPass : public impl::EpochOptBase<EpochOptPass> {
  EpochOptPass() : EpochOptPass(/*AM=*/nullptr) {}
  EpochOptPass(const EpochOptPass &other)
      : impl::EpochOptBase<EpochOptPass>(other), AM(other.AM),
        numRepeatedEpochLoopsAmortized(
            this, "num-repeated-epoch-loops-amortized",
            "Number of repeated epoch loops amortized"),
        numEpochsNarrowed(this, "num-epochs-narrowed",
                          "Number of epochs split into narrower scopes"),
        numEpochsCreatedByNarrowing(
            this, "num-epochs-created-by-narrowing",
            "Number of new epochs created while narrowing epoch scopes"),
        numEpochPairsFused(this, "num-epoch-pairs-fused",
                           "Number of consecutive epoch pairs fused") {}

  explicit EpochOptPass(mlir::carts::arts::AnalysisManager *AM)
      : AM(AM), numRepeatedEpochLoopsAmortized(
                    this, "num-repeated-epoch-loops-amortized",
                    "Number of repeated epoch loops amortized"),
        numEpochsNarrowed(this, "num-epochs-narrowed",
                          "Number of epochs split into narrower scopes"),
        numEpochsCreatedByNarrowing(
            this, "num-epochs-created-by-narrowing",
            "Number of new epochs created while narrowing epoch scopes"),
        numEpochPairsFused(this, "num-epoch-pairs-fused",
                           "Number of consecutive epoch pairs fused") {}

  EpochOptPass(mlir::carts::arts::AnalysisManager *AM, bool narrowing,
               bool fusion, bool amortization)
      : EpochOptPass(AM) {
    enableEpochNarrowing = narrowing;
    enableEpochFusion = fusion;
    enableAmortization = amortization;
  }

  mlir::carts::arts::AnalysisManager &getEpochAnalysisManager(ModuleOp module) {
    if (!AM) {
      ownedAM = std::make_unique<mlir::carts::arts::AnalysisManager>(module);
      AM = ownedAM.get();
    }
    return *AM;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool changed = false;
    mlir::carts::arts::AnalysisManager &epochAM =
        getEpochAnalysisManager(module);
    EpochAnalysis &epochAnalysis = epochAM.getEpochAnalysis();

    ARTS_INFO_HEADER(EpochOptPass);

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
      unsigned fusedEpochPairs =
          processRegionForEpochFusion(module.getRegion(), epochAnalysis);
      if (fusedEpochPairs != 0) {
        numEpochPairsFused += fusedEpochPairs;
        ARTS_INFO("Epoch fusion applied");
        changed = true;
      }
    }

    ARTS_INFO_FOOTER(EpochOptPass);

    if (!changed)
      markAllAnalysesPreserved();
  }

private:
  mlir::carts::arts::AnalysisManager *AM = nullptr;
  Statistic numRepeatedEpochLoopsAmortized;
  Statistic numEpochsNarrowed;
  Statistic numEpochsCreatedByNarrowing;
  Statistic numEpochPairsFused;
  std::unique_ptr<mlir::carts::arts::AnalysisManager> ownedAM;
};

} // namespace

namespace mlir {
namespace carts::arts {

std::unique_ptr<Pass> createEpochOptPass() {
  return std::make_unique<EpochOptPass>();
}

std::unique_ptr<Pass> createEpochOptPass(mlir::carts::arts::AnalysisManager *AM,
                                         bool amortization) {
  return std::make_unique<EpochOptPass>(AM, /*narrowing=*/true, /*fusion=*/true,
                                        amortization);
}

} // namespace carts::arts
} // namespace mlir
