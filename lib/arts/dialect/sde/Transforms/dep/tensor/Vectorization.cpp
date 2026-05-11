///==========================================================================///
/// File: Vectorization.cpp
///
/// Vectorize static-shape linalg.generic carriers inside su_iterate bodies
/// using upstream MLIR's linalg::vectorize(). Runs after BarrierElimination
/// (all dep/effect passes need carriers) and before LowerToMemref.
///
/// Only vectorizes carriers with:
/// - All-parallel iterator types (no reduction vectorization yet)
/// - Static shapes on all operands and loop ranges
/// - Tensor semantics (not buffer)
/// Uses sde.vectorize_width attr if present, otherwise defaults based on
/// element type (f32->8, f64->4, i32->8).
///
/// The vector sizes passed to linalg::vectorize() must be >= the static loop
/// ranges. CARTS currently only vectorizes small live tensor expressions here:
/// tensor carriers whose results are intentionally materialized later by
/// LowerToMemref must stay as linalg so their output effects are preserved.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_VECTORIZATION
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(vectorization);

using namespace mlir;
using namespace mlir::arts;

namespace {

/// Check whether a body block is carrier-authoritative: it contains a
/// linalg.generic carrier but NO scalar executable ops (memref.load/store).
static bool isCarrierAuthoritative(Block &body) {
  bool hasCarrier = false;
  bool hasScalar = false;
  for (Operation &op : body.without_terminator()) {
    if (isa<linalg::GenericOp>(op)) {
      hasCarrier = true;
      continue;
    }
    if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      hasScalar = true;
      continue;
    }
    op.walk([&](Operation *nested) {
      if (isa<memref::LoadOp, memref::StoreOp>(nested))
        hasScalar = true;
    });
  }
  return hasCarrier && !hasScalar;
}

/// Check if a linalg.generic can be vectorized:
/// - All iterator types must be parallel (no reduction vectorization yet)
/// - All shapes must be static
/// - Must have tensor semantics
/// - Must have at least one output
static bool canVectorize(linalg::GenericOp generic) {
  constexpr int64_t kMaxVectorElements = 4096;

  for (auto iterType : generic.getIteratorTypesArray()) {
    if (iterType != utils::IteratorType::parallel)
      return false;
  }
  if (generic.hasPureBufferSemantics())
    return false;
  bool hasLiveResult = llvm::any_of(generic.getResults(), [](Value result) {
    return !result.use_empty();
  });
  if (!hasLiveResult)
    return false;
  for (OpOperand &operand : generic->getOpOperands()) {
    auto ty = dyn_cast<RankedTensorType>(operand.get().getType());
    if (!ty || !ty.hasStaticShape())
      return false;
  }
  int64_t vectorElements = 1;
  for (int64_t range : generic.getStaticLoopRanges()) {
    if (ShapedType::isDynamic(range))
      return false;
    if (range <= 0 || vectorElements > kMaxVectorElements / range)
      return false;
    vectorElements *= range;
  }
  return generic.getNumDpsInits() > 0;
}

struct VectorizationPass
    : public arts::impl::VectorizationBase<VectorizationPass> {
  using arts::impl::VectorizationBase<VectorizationPass>::VectorizationBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect vectorizable carriers inside carrier-authoritative su_iterate
    // bodies.
    SmallVector<linalg::GenericOp> candidates;
    module.walk([&](sde::SdeSuIterateOp iterOp) {
      Block *computeBlock = sde::getSuIterateComputeBlock(iterOp);
      if (!computeBlock)
        return;
      if (!isCarrierAuthoritative(*computeBlock))
        return;
      computeBlock->walk([&](linalg::GenericOp generic) {
        if (canVectorize(generic))
          candidates.push_back(generic);
      });
    });

    if (candidates.empty()) {
      ARTS_DEBUG("Vectorization: no vectorizable carriers found");
      return;
    }

    IRRewriter rewriter(&getContext());
    unsigned vectorized = 0;

    for (linalg::GenericOp generic : candidates) {
      // Use the full static loop ranges as vector sizes. linalg::vectorize()
      // requires vector sizes >= iteration space sizes.
      SmallVector<int64_t> vectorSizes = generic.getStaticLoopRanges();
      if (vectorSizes.empty())
        continue;

      SmallVector<bool> scalableDims(vectorSizes.size(), false);
      rewriter.setInsertionPoint(generic);
      auto result =
          linalg::vectorize(rewriter, generic, vectorSizes, scalableDims);
      if (failed(result)) {
        ARTS_DEBUG("Vectorization: vectorize() failed for carrier");
        continue;
      }
      // Replace original op results with vectorized replacements and erase.
      for (auto [oldResult, newResult] :
           llvm::zip(generic.getResults(), result->replacements))
        rewriter.replaceAllUsesWith(oldResult, newResult);
      rewriter.eraseOp(generic);
      ++vectorized;
    }

    ARTS_INFO("Vectorization: vectorized " << vectorized << " of "
                                           << candidates.size()
                                           << " candidate carrier(s)");
  }
};

} // namespace

namespace mlir::arts::sde {

std::unique_ptr<Pass> createVectorizationPass() {
  return std::make_unique<VectorizationPass>();
}

} // namespace mlir::arts::sde
