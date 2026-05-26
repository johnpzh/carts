///==========================================================================///
/// File: Vectorization.cpp
///
/// Plan vectorization for memref-native SDE loop bodies. The pass stamps
/// target-neutral execution hints on sde.su_iterate; it does not raise to
/// tensor/linalg and does not create target vector IR.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::carts::sde {
#define GEN_PASS_DEF_VECTORIZATION
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::sde

#include "carts/dialect/sde/Utils/SDECostModel.h"
#include "carts/utils/Debug.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <optional>

ARTS_DEBUG_SETUP(vectorization);

using namespace mlir;
using namespace mlir::carts;

namespace {

struct MemrefVectorPlan {
  int64_t vectorizeWidth = 0;
  int64_t unrollFactor = 0;
  int64_t interleaveCount = 0;
};

static bool isTargetNeutralCarrierDialect(Operation *op) {
  Dialect *dialect = op->getDialect();
  if (!dialect)
    return false;
  StringRef ns = dialect->getNamespace();
  return ns == "tensor" || ns == "linalg";
}

static bool hasTensorOrLinalgCarrier(Block &block) {
  bool found = false;
  block.walk([&](Operation *op) {
    if (isTargetNeutralCarrierDialect(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static std::optional<unsigned> getScalarElementBits(Type type) {
  if (auto memrefType = dyn_cast<MemRefType>(type))
    type = memrefType.getElementType();
  if (auto vectorType = dyn_cast<VectorType>(type))
    type = vectorType.getElementType();

  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth();
  return std::nullopt;
}

static bool isSupportedElementType(Type type) {
  std::optional<unsigned> bits = getScalarElementBits(type);
  return bits && (*bits == 32 || *bits == 64);
}

static bool hasMemrefReadOrWrite(Block &block) {
  bool found = false;
  block.walk([&](Operation *op) {
    if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static std::optional<unsigned> getDominantElementBits(Block &block) {
  std::optional<unsigned> selectedBits;

  auto recordType = [&](Type type) {
    std::optional<unsigned> bits = getScalarElementBits(type);
    if (!bits || (*bits != 32 && *bits != 64))
      return;
    if (!selectedBits || *bits < *selectedBits)
      selectedBits = bits;
  };

  block.walk([&](memref::StoreOp storeOp) {
    recordType(storeOp.getValueToStore().getType());
  });
  if (selectedBits)
    return selectedBits;

  block.walk([&](memref::LoadOp loadOp) { recordType(loadOp.getType()); });
  return selectedBits;
}

static bool hasUnsupportedMemrefElement(Block &block) {
  bool unsupported = false;
  block.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      if (!isSupportedElementType(loadOp.getType())) {
        unsupported = true;
        return WalkResult::interrupt();
      }
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      if (!isSupportedElementType(storeOp.getValueToStore().getType())) {
        unsupported = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return unsupported;
}

static bool isVectorizableClassification(sde::SdeSuIterateOp op) {
  auto classification = op.getStructuredClassification();
  if (!classification)
    return false;
  return *classification == sde::SdeStructuredClassification::elementwise ||
         *classification ==
             sde::SdeStructuredClassification::elementwise_pipeline ||
         *classification == sde::SdeStructuredClassification::matmul;
}

static std::optional<MemrefVectorPlan>
buildMemrefVectorPlan(sde::SdeSuIterateOp op, sde::SDECostModel *costModel) {
  if (!costModel)
    return std::nullopt;
  if (!isVectorizableClassification(op))
    return std::nullopt;
  if (op.getInPlaceSharedStateAttr())
    return std::nullopt;

  Block *computeBlock = sde::getSuIterateComputeBlock(op);
  if (!computeBlock || hasTensorOrLinalgCarrier(*computeBlock) ||
      !hasMemrefReadOrWrite(*computeBlock) ||
      hasUnsupportedMemrefElement(*computeBlock))
    return std::nullopt;

  std::optional<unsigned> elementBits = getDominantElementBits(*computeBlock);
  if (!elementBits)
    return std::nullopt;

  MemrefVectorPlan plan;
  plan.vectorizeWidth = costModel->getVectorWidthForElementBits(*elementBits);

  bool reuseHeavy = false;
  if (auto pattern = op.getPattern()) {
    if (*pattern == sde::SdePattern::matmul)
      reuseHeavy = true;
  }
  plan.unrollFactor = costModel->getVectorUnrollFactor(reuseHeavy);
  plan.interleaveCount = costModel->getVectorInterleaveCount();

  return plan;
}

static void stampVectorPlan(sde::SdeSuIterateOp op,
                            const MemrefVectorPlan &plan) {
  Type i64 = IntegerType::get(op.getContext(), 64);
  op.setVectorizeWidthAttr(IntegerAttr::get(i64, plan.vectorizeWidth));
  op.setUnrollFactorAttr(IntegerAttr::get(i64, plan.unrollFactor));
  op.setInterleaveCountAttr(IntegerAttr::get(i64, plan.interleaveCount));
}

struct VectorizationPass
    : public sde::impl::VectorizationBase<VectorizationPass> {
  explicit VectorizationPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    unsigned stamped = 0;
    getOperation().walk([&](sde::SdeSuIterateOp op) {
      std::optional<MemrefVectorPlan> plan =
          buildMemrefVectorPlan(op, costModel);
      if (!plan)
        return;
      stampVectorPlan(op, *plan);
      ++stamped;
    });

    ARTS_INFO("Vectorization: stamped memref vector hints on " << stamped
                                                               << " loop(s)");
  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createVectorizationPass(sde::SDECostModel *costModel) {
  return std::make_unique<VectorizationPass>(costModel);
}

} // namespace mlir::carts::sde
