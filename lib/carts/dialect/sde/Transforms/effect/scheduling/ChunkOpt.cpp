///==========================================================================///
/// File: ChunkOpt.cpp
///
/// Cost-model-backed SDE chunk sizing. Preserves explicit source chunk sizes
/// and synthesizes chunks only for one-dimensional dynamic/guided loops,
/// using either constant or symbolic trip-count arithmetic.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_CHUNKOPT
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "carts/utils/LoopUtils.h"
#include "carts/utils/ValueAnalysis.h"
#include "carts/utils/costs/SDECostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/MathExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::carts;

namespace {

static bool isChunkOptimizableSchedule(sde::SdeScheduleKindAttr schedule) {
  if (!schedule)
    return false;
  switch (schedule.getValue()) {
  case sde::SdeScheduleKind::dynamic:
  case sde::SdeScheduleKind::guided:
    return true;
  default:
    return false;
  }
}

static Value getConstantIndex(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

static Value buildLogicalWorkerCapacityValue(OpBuilder &builder,
                                             Location loc) {
  Value logicalWorkers =
      sde::SdeResourceQueryOp::create(
          builder, loc, sde::SdeResourceQueryKind::logicalWorkers)
          .getResult();
  return arith::MaxUIOp::create(builder, loc, logicalWorkers,
                                getConstantIndex(builder, loc, 1));
}

static Value buildTripCountValue(OpBuilder &builder, Location loc,
                                 sde::SdeSuIterateOp op) {
  if (std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation()))
    return getConstantIndex(builder, loc, *tripCount);

  if (op.getLowerBounds().size() != 1 || op.getUpperBounds().size() != 1 ||
      op.getSteps().size() != 1)
    return Value();

  Value lowerBound = op.getLowerBounds().front();
  Value upperBound = op.getUpperBounds().front();
  Value step = op.getSteps().front();
  Value zero = getConstantIndex(builder, loc, 0);
  Value one = getConstantIndex(builder, loc, 1);

  Value span = arith::SubIOp::create(builder, loc, upperBound, lowerBound);

  int64_t constantStep = 0;
  Value safeStep = step;
  if (arts::ValueAnalysis::getConstantIndex(step, constantStep)) {
    if (constantStep <= 0)
      return Value();
  }

  // Keep symbolic trip-count computation signed-safe so negative runtime
  // spans clamp to zero even when the lower bound is zero.
  Value spanIsNegative = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::slt, span, zero);
  Value nonNegativeSpan =
      arith::SelectOp::create(builder, loc, spanIsNegative, zero, span);

  if (!arts::ValueAnalysis::getConstantIndex(step, constantStep)) {
    Value stepIsTooSmall = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sle, step, zero);
    safeStep = arith::SelectOp::create(builder, loc, stepIsTooSmall, one, step);
  }

  return arith::CeilDivSIOp::create(builder, loc, nonNegativeSpan, safeStep);
}

static Value buildSymbolicChunkValue(OpBuilder &builder, Location loc,
                                     sde::SdeSuIterateOp op,
                                     int64_t minIterations) {
  Value tripCount = buildTripCountValue(builder, loc, op);
  if (!tripCount)
    return Value();

  Value one = getConstantIndex(builder, loc, 1);
  Value workerCountValue = buildLogicalWorkerCapacityValue(builder, loc);
  Value minIterationsValue =
      getConstantIndex(builder, loc, std::max<int64_t>(1, minIterations));

  Value clampedTripCount = arith::MaxUIOp::create(builder, loc, tripCount, one);
  Value balancedChunk = arith::CeilDivUIOp::create(
      builder, loc, clampedTripCount, workerCountValue);
  Value preferredChunk =
      arith::MaxUIOp::create(builder, loc, balancedChunk, minIterationsValue);
  return arith::MinUIOp::create(builder, loc, preferredChunk, clampedTripCount);
}

struct ChunkRewrite {
  sde::SdeSuIterateOp op;
  std::optional<int64_t> chunkSize;
};

struct ChunkOptPass : public arts::impl::ChunkOptBase<ChunkOptPass> {
  explicit ChunkOptPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    if (!costModel)
      return;

    SmallVector<ChunkRewrite> rewrites;
    getOperation().walk([&](sde::SdeSuIterateOp op) {
      if (op.getChunkSize() ||
          !isChunkOptimizableSchedule(op.getScheduleAttr()))
        return;

      std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation());
      if (tripCount && *tripCount <= 1)
        return;

      if (tripCount) {
        int64_t workerCount =
            std::max<int64_t>(1, costModel->getLogicalWorkerCapacity());
        int64_t minIterations =
            std::max<int64_t>(1, costModel->getMinIterationsPerWorker());
        int64_t balancedChunk = llvm::divideCeil(*tripCount, workerCount);
        int64_t chunkSize = std::clamp(std::max(minIterations, balancedChunk),
                                       int64_t{1}, *tripCount);
        rewrites.push_back({op, chunkSize});
        return;
      }

      if (op.getLowerBounds().size() != 1 || op.getUpperBounds().size() != 1 ||
          op.getSteps().size() != 1)
        return;

      rewrites.push_back({op, std::nullopt});
    });

    for (ChunkRewrite &rewrite : rewrites) {
      IRRewriter rewriter(rewrite.op.getContext());
      rewriter.setInsertionPoint(rewrite.op);

      Value chunkSize;
      if (rewrite.chunkSize) {
        chunkSize =
            getConstantIndex(rewriter, rewrite.op.getLoc(), *rewrite.chunkSize);
      } else {
        chunkSize = buildSymbolicChunkValue(
            rewriter, rewrite.op.getLoc(), rewrite.op,
            std::max<int64_t>(1, costModel->getMinIterationsPerWorker()));
      }
      if (!chunkSize)
        continue;

      auto newOp = sde::SdeSuIterateOp::create(
          rewriter, rewrite.op.getLoc(), /*resultTypes=*/TypeRange{},
          rewrite.op.getLowerBounds(), rewrite.op.getUpperBounds(),
          rewrite.op.getSteps(), rewrite.op.getScheduleAttr(), chunkSize,
          rewrite.op.getNowaitAttr(), rewrite.op.getReductionAccumulators(),
          rewrite.op.getReductionKindsAttr(),
          rewrite.op.getReductionStrategyAttr(),
          rewrite.op.getStructuredClassificationAttr(),
          rewrite.op.getPatternAttr(),
          rewrite.op.getAccessMinOffsetsAttr(),
          rewrite.op.getAccessMaxOffsetsAttr(), rewrite.op.getOwnerDimsAttr(),
          rewrite.op.getSpatialDimsAttr(), rewrite.op.getWriteFootprintAttr(),
          rewrite.op.getPhysicalOwnerDimsAttr(),
          rewrite.op.getPhysicalBlockShapeAttr(),
          rewrite.op.getLogicalWorkerSliceAttr(),
          rewrite.op.getPhysicalHaloShapeAttr(),
          rewrite.op.getIterationTopologyAttr(),
          rewrite.op.getRepetitionStructureAttr(),
          rewrite.op.getAsyncStrategyAttr(),
          rewrite.op.getCpsGroupIdAttr(), rewrite.op.getCpsStageIndexAttr(),
          rewrite.op.getCpsStageCountAttr(),
          rewrite.op.getDistributionKindAttr(),
          rewrite.op.getInPlaceSafeAttr(),
          rewrite.op.getInPlaceSharedStateAttr(),
          rewrite.op.getVectorizeWidthAttr(),
          rewrite.op.getUnrollFactorAttr(),
          rewrite.op.getInterleaveCountAttr());
      newOp->setAttrs(sde::getRewrittenAttrs(rewrite.op));
      newOp.getBody().takeBody(rewrite.op.getBody());
      rewriter.eraseOp(rewrite.op);
    }
  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createChunkOptPass(sde::SDECostModel *costModel) {
  return std::make_unique<ChunkOptPass>(costModel);
}

} // namespace mlir::carts::sde
