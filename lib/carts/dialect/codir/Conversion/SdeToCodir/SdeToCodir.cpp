///==========================================================================///
/// File: SdeToCodir.cpp
///
/// Materializes SDE planning intent as isolated CODIR codelets.
///==========================================================================///
#include "carts/dialect/codir/Utils/CodirConversionUtils.h"
#include "carts/dialect/codir/Conversion/Passes.h"
namespace mlir::carts::codir {
#define GEN_PASS_DEF_CONVERTSDETOCODIR
#include "carts/dialect/codir/Conversion/Passes.h.inc"
} // namespace mlir::carts::codir
namespace {
struct ConvertSdeToCodirPass
    : public codir::impl::ConvertSdeToCodirBase<ConvertSdeToCodirPass> {
  void runOnOperation() override {
    SmallVector<sde::SdeCuCodeletOp> codelets;
    getOperation().walk(
        [&](sde::SdeCuCodeletOp codelet) { codelets.push_back(codelet); });

    for (sde::SdeCuCodeletOp sdeCodelet : codelets) {
      SmallVector<Value> deps;
      SmallVector<Attribute> depModes;
      SmallVector<Attribute> depStorageViews;
      SmallVector<SlicedTokenLocalIndexRewrite> localIndexRewrites;
      OpBuilder builder(sdeCodelet);
      if (failed(materializeCodirDeps(sdeCodelet, builder, deps, depModes,
                                      depStorageViews, localIndexRewrites))) {
        signalPassFailure();
        return;
      }

      SmallVector<Value> params(sdeCodelet.getCaptures().begin(),
                                sdeCodelet.getCaptures().end());
      unsigned originalParamCount = params.size();
      appendSlicedTokenOffsetParams(localIndexRewrites, params);
      appendDynamicCodirDepSliceParams(deps, params);
      auto codirCodelet = createCodirCodelet(
          builder, sdeCodelet.getLoc(), builder.getArrayAttr(depModes),
          builder.getArrayAttr(depStorageViews), deps, params);
      codirCodelet.getBody().takeBody(sdeCodelet.getBody());
      Block &body = codirCodelet.getBody().front();
      for (unsigned idx = originalParamCount, e = params.size(); idx < e; ++idx)
        body.addArgument(params[idx].getType(), sdeCodelet.getLoc());
      if (failed(rewriteTokenLocalAccesses(codirCodelet, localIndexRewrites))) {
        signalPassFailure();
        return;
      }
      replaceSdeYieldWithCodirYield(codirCodelet);
      sdeCodelet.erase();
    }

    SmallVector<sde::SdeCuTaskOp> tasks;
    getOperation().walk([&](sde::SdeCuTaskOp task) { tasks.push_back(task); });
    for (sde::SdeCuTaskOp task : tasks) {
      if (failed(convertCuTaskToCodir(task))) {
        signalPassFailure();
        return;
      }
    }

    SuBarrierTokenDepPlan barrierTokenDepPlan;
    collectSuBarrierTokenDepPlans(getOperation(), barrierTokenDepPlan);

    SmallVector<sde::SdeSuIterateOp> iterates;
    getOperation().walk(
        [&](sde::SdeSuIterateOp iterate) { iterates.push_back(iterate); });
    for (sde::SdeSuIterateOp iterate : iterates) {
      if (failed(convertSuIterateToCodir(iterate, &barrierTokenDepPlan))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<sde::SdeSuDistributeOp> distributes;
    getOperation().walk([&](sde::SdeSuDistributeOp distribute) {
      distributes.push_back(distribute);
    });
    for (sde::SdeSuDistributeOp distribute : distributes) {
      if (failed(inlineSdeSuDistribute(distribute))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<sde::SdeCuRegionOp> regions;
    getOperation().walk(
        [&](sde::SdeCuRegionOp region) { regions.push_back(region); });
    for (sde::SdeCuRegionOp region : regions) {
      if (failed(inlineSdeCuRegion(region))) {
        signalPassFailure();
        return;
      }
    }

    bool hasUnresolvedMuDep = false;
    getOperation().walk(
        [&](sde::SdeMuDepOp muDep) {
          muDep.emitOpError()
              << "must be consumed by convert-sde-to-codir before the CODIR "
                 "boundary; materialize it as an sde.mu_token/codelet dep or "
                 "remove the stale dependency declaration";
          hasUnresolvedMuDep = true;
        });
    if (hasUnresolvedMuDep)
      signalPassFailure();
  }
};

} // namespace
std::unique_ptr<Pass> mlir::carts::codir::createConvertSdeToCodirPass() {
  return std::make_unique<ConvertSdeToCodirPass>();
}
