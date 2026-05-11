///==========================================================================///
/// File: ElementwiseFusion.cpp
///
/// Fuse consecutive sibling SDE elementwise scheduling units before crossing
/// into ARTS IR.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_ELEMENTWISEFUSION
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/ValueAnalysis.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::arts;

namespace {

struct ElementwiseStage {
  sde::SdeSuIterateOp op;
  SmallVector<Value, 4> writes;
};

static bool haveSameIterationSpace(sde::SdeSuIterateOp lhs,
                                   sde::SdeSuIterateOp rhs) {
  if (lhs.getLowerBounds().size() != rhs.getLowerBounds().size() ||
      lhs.getUpperBounds().size() != rhs.getUpperBounds().size() ||
      lhs.getSteps().size() != rhs.getSteps().size())
    return false;

  for (auto [a, b] : llvm::zip(lhs.getLowerBounds(), rhs.getLowerBounds())) {
    if (!ValueAnalysis::areValuesEquivalent(a, b))
      return false;
  }
  for (auto [a, b] : llvm::zip(lhs.getUpperBounds(), rhs.getUpperBounds())) {
    if (!ValueAnalysis::areValuesEquivalent(a, b))
      return false;
  }
  for (auto [a, b] : llvm::zip(lhs.getSteps(), rhs.getSteps())) {
    if (!ValueAnalysis::areValuesEquivalent(a, b))
      return false;
  }
  return true;
}

static bool haveCompatibleSchedule(sde::SdeSuIterateOp lhs,
                                   sde::SdeSuIterateOp rhs) {
  if (lhs.getScheduleAttr() != rhs.getScheduleAttr())
    return false;
  if (lhs.getNowaitAttr() != rhs.getNowaitAttr())
    return false;

  Value lhsChunk = lhs.getChunkSize();
  Value rhsChunk = rhs.getChunkSize();
  if (!lhsChunk || !rhsChunk)
    return lhsChunk == rhsChunk;
  return ValueAnalysis::areValuesEquivalent(lhsChunk, rhsChunk);
}

static Value getWriteRoot(Value value) {
  return ValueAnalysis::stripMemrefViewOps(value);
}

/// Trace a carrier tensor operand back to its backing memref root through
/// mu_memref_to_tensor and tensor.cast chains.
static Value traceCarrierOperandToMemref(Value tensorOperand) {
  Value cur = tensorOperand;
  while (cur) {
    if (auto muOp = cur.getDefiningOp<sde::SdeMuMemrefToTensorOp>())
      return getWriteRoot(muOp.getMemref());
    if (auto castOp = cur.getDefiningOp<tensor::CastOp>()) {
      cur = castOp.getSource();
      continue;
    }
    if (auto sliceOp = cur.getDefiningOp<tensor::ExtractSliceOp>()) {
      cur = sliceOp.getSource();
      continue;
    }
    break;
  }
  return {};
}

static bool hasDisjointWrites(ArrayRef<ElementwiseStage> stages) {
  llvm::DenseSet<Value> writtenTargets;
  for (const ElementwiseStage &stage : stages) {
    for (Value target : stage.writes) {
      if (!target || !writtenTargets.insert(target).second)
        return false;
    }
  }
  return true;
}

static bool isElementwiseStage(sde::SdeSuIterateOp op,
                               ElementwiseStage &stage) {
  if (!op || !op.getStructuredClassificationAttr() ||
      *op.getStructuredClassification() !=
          sde::SdeStructuredClassification::elementwise)
    return false;
  if (!op.getReductionAccumulators().empty())
    return false;

  llvm::DenseSet<Value> seenWrites;
  SmallVector<Value, 4> writes;
  bool hasDuplicateWrite = false;

  // Carrier-authoritative path: extract write-set from carrier DPS outputs.
  Block *computeBody = sde::getSuIterateComputeBlock(op);
  linalg::GenericOp carrier;
  for (Operation &inner : computeBody->without_terminator())
    if (auto g = dyn_cast<linalg::GenericOp>(inner)) {
      carrier = g;
      break;
    }

  if (carrier) {
    for (Value output : carrier.getDpsInits()) {
      Value target = traceCarrierOperandToMemref(output);
      if (!target)
        return false;
      if (!seenWrites.insert(target).second) {
        hasDuplicateWrite = true;
        break;
      }
      writes.push_back(target);
    }
  } else {
    // Fallback: walk memref.store ops (dual-rep / stencil path).
    op.getBody().walk([&](memref::StoreOp storeOp) {
      Value target = getWriteRoot(storeOp.getMemref());
      if (!target)
        return;
      if (!seenWrites.insert(target).second) {
        hasDuplicateWrite = true;
        return;
      }
      writes.push_back(target);
    });
  }

  if (writes.empty() || hasDuplicateWrite)
    return false;

  stage = {op, std::move(writes)};
  return true;
}

static bool isSkippableInterStageOp(Operation *op) {
  return op && op->getNumRegions() == 0 && isMemoryEffectFree(op);
}

static sde::SdeSuIterateOp fuseStages(MutableArrayRef<ElementwiseStage> stages,
                                      IRRewriter &rewriter) {
  assert(stages.size() >= 2 && "expected at least two stages");

  sde::SdeSuIterateOp first = stages.front().op;
  Location loc = first.getLoc();
  rewriter.setInsertionPoint(stages.back().op);

  auto fused = sde::SdeSuIterateOp::create(
      rewriter, loc, /*resultTypes=*/TypeRange{}, first.getLowerBounds(),
      first.getUpperBounds(), first.getSteps(), first.getScheduleAttr(),
      first.getChunkSize(), first.getNowaitAttr(),
      first.getReductionAccumulators(), first.getReductionKindsAttr(),
      first.getReductionStrategyAttr(),
      sde::SdeStructuredClassificationAttr::get(
          first.getContext(),
          sde::SdeStructuredClassification::elementwise_pipeline),
      first.getAccessMinOffsetsAttr(), first.getAccessMaxOffsetsAttr(),
      first.getOwnerDimsAttr(), first.getSpatialDimsAttr(),
      first.getWriteFootprintAttr());
  fused->setAttrs(sde::getRewrittenAttrs(first));
  fused.setStructuredClassificationAttr(
      sde::SdeStructuredClassificationAttr::get(
          first.getContext(),
          sde::SdeStructuredClassification::elementwise_pipeline));

  Block &dst = sde::ensureBlock(fused.getBody());
  if (dst.getNumArguments() == 0) {
    for (BlockArgument arg : stages.front().op.getBody().front().getArguments())
      dst.addArgument(arg.getType(), loc);
  }

  // Create inner cu_region <parallel> wrapper to match the per-stage structure.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&dst);
  auto innerCuRegion = sde::SdeCuRegionOp::create(
      rewriter, loc, /*resultTypes=*/TypeRange{},
      sde::SdeCuKindAttr::get(rewriter.getContext(), sde::SdeCuKind::parallel),
      /*concurrency_scope=*/nullptr,
      /*nowait=*/nullptr,
      /*iterArgs=*/ValueRange{});
  Block &innerBlk = sde::ensureBlock(innerCuRegion.getBody());
  rewriter.setInsertionPointToStart(&innerBlk);

  for (ElementwiseStage &stage : stages) {
    Block *srcBody = sde::getSuIterateComputeBlock(stage.op);
    IRMapping mapper;
    // Map the su_iterate block args (induction vars).
    for (auto [srcArg, dstArg] : llvm::zip(
             stage.op.getBody().front().getArguments(), dst.getArguments()))
      mapper.map(srcArg, dstArg);
    for (Operation &nested : srcBody->without_terminator())
      rewriter.clone(nested, mapper);
  }
  sde::SdeYieldOp::create(rewriter, loc, ValueRange{});

  // Yield at su_iterate level.
  rewriter.setInsertionPointAfter(innerCuRegion);
  sde::SdeYieldOp::create(rewriter, loc, ValueRange{});
  return fused;
}

struct ElementwiseFusionPass
    : public arts::impl::ElementwiseFusionBase<ElementwiseFusionPass> {
  void runOnOperation() override {
    bool changed = true;
    while (changed) {
      changed = false;

      SmallVector<Block *> blocks;
      getOperation().walk([&](Operation *op) {
        for (Region &region : op->getRegions())
          for (Block &block : region)
            blocks.push_back(&block);
      });

      for (Block *block : blocks) {
        if (!block)
          continue;

        for (auto it = block->begin(), e = block->end(); it != e; ++it) {
          auto first = dyn_cast<sde::SdeSuIterateOp>(&*it);
          ElementwiseStage firstStage;
          if (!first || !isElementwiseStage(first, firstStage))
            continue;

          SmallVector<ElementwiseStage, 4> stages;
          stages.push_back(std::move(firstStage));
          for (auto nextIt = std::next(it); nextIt != e; ++nextIt) {
            if (isSkippableInterStageOp(&*nextIt))
              continue;

            auto next = dyn_cast<sde::SdeSuIterateOp>(&*nextIt);
            ElementwiseStage nextStage;
            if (!next || !isElementwiseStage(next, nextStage) ||
                !haveSameIterationSpace(first, next) ||
                !haveCompatibleSchedule(first, next))
              break;

            stages.push_back(std::move(nextStage));
            if (!hasDisjointWrites(stages)) {
              stages.pop_back();
              break;
            }
          }

          if (stages.size() < 2)
            continue;

          IRRewriter rewriter(&getContext());
          sde::SdeSuIterateOp fused = fuseStages(stages, rewriter);
          (void)fused;
          for (const ElementwiseStage &stage : stages)
            rewriter.eraseOp(stage.op);
          changed = true;
          break;
        }

        if (changed)
          break;
      }
    }
  }
};

} // namespace

namespace mlir::arts::sde {

std::unique_ptr<Pass> createElementwiseFusionPass() {
  return std::make_unique<ElementwiseFusionPass>();
}

} // namespace mlir::arts::sde
