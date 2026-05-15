///==========================================================================///
/// File: TensorCleanup.cpp
///
/// Clean up tensor/linalg artifacts inside cu_codelet bodies left over from
/// the carrier-authoritative pipeline: dead extract_slice, insert_slice,
/// empty tensors, and foldable subset ops.
///
/// Runs upstream canonicalization + folding patterns scoped to each codelet
/// body (GreedyRewriteConfig limited to the codelet region).
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_TENSORCLEANUP
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::carts;

namespace {

struct TensorCleanupPass
    : public arts::impl::TensorCleanupBase<TensorCleanupPass> {
  using arts::impl::TensorCleanupBase<TensorCleanupPass>::TensorCleanupBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    SmallVector<sde::SdeCuCodeletOp> codelets;
    module.walk([&](sde::SdeCuCodeletOp op) { codelets.push_back(op); });

    if (codelets.empty())
      return;

    RewritePatternSet patterns(ctx);

    // Fold tensor subset ops (extract_slice/insert_slice canonicalization).
    tensor::populateFoldTensorSubsetOpPatterns(patterns);
    // Merge consecutive insert/extract slices.
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    // Fold tensor.empty into consumers.
    tensor::populateFoldTensorEmptyPatterns(patterns);

    // Add canonicalization patterns from tensor and linalg dialects.
    tensor::TensorDialect *tensorDialect =
        ctx->getLoadedDialect<tensor::TensorDialect>();
    if (tensorDialect)
      tensorDialect->getCanonicalizationPatterns(patterns);

    linalg::LinalgDialect *linalgDialect =
        ctx->getLoadedDialect<linalg::LinalgDialect>();
    if (linalgDialect)
      linalgDialect->getCanonicalizationPatterns(patterns);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    // Apply patterns scoped to each codelet body region.
    // cu_codelet is IsolatedFromAbove, so we apply directly to its region.
    GreedyRewriteConfig config;
    config.setMaxIterations(4);

    for (sde::SdeCuCodeletOp codelet : codelets) {
      if (failed(applyPatternsGreedily(codelet.getBody(), frozenPatterns,
                                       config))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createTensorCleanupPass() {
  return std::make_unique<TensorCleanupPass>();
}

} // namespace mlir::carts::sde
