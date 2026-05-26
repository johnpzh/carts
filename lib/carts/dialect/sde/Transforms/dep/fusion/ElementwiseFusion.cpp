///==========================================================================///
/// File: ElementwiseFusion.cpp
///
/// Fuse consecutive sibling SDE elementwise scheduling units before boundary
/// materialization.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::carts::sde {
#define GEN_PASS_DEF_ELEMENTWISEFUSION
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::sde

#include "carts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "carts/utils/ValueAnalysis.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::carts;

namespace {

struct MemrefAccess {
  Value root;
  SmallVector<Value, 4> indices;
};

struct ElementwiseStage {
  sde::SdeSuIterateOp op;
  Operation *root = nullptr;
  SmallVector<MemrefAccess, 4> reads;
  SmallVector<MemrefAccess, 4> writes;
};

static bool isSkippableInterStageOp(Operation *op) {
  return op && op->getNumRegions() == 0 && isMemoryEffectFree(op);
}

static sde::SdeSuIterateOp getStageSuIterate(Operation *op) {
  if (auto suIter = dyn_cast_or_null<sde::SdeSuIterateOp>(op))
    return suIter;

  auto cuRegion = dyn_cast_or_null<sde::SdeCuRegionOp>(op);
  if (!cuRegion || cuRegion.getKind() != sde::SdeCuKind::parallel ||
      !cuRegion.getBody().hasOneBlock())
    return {};

  sde::SdeSuIterateOp nested;
  for (Operation &inner : cuRegion.getBody().front().without_terminator()) {
    if (auto suIter = dyn_cast<sde::SdeSuIterateOp>(inner)) {
      if (nested)
        return {};
      nested = suIter;
      continue;
    }
    if (isSkippableInterStageOp(&inner))
      continue;
    return {};
  }
  return nested;
}

static bool areStageValuesEquivalent(Value lhs, sde::SdeSuIterateOp lhsStage,
                                     Value rhs, sde::SdeSuIterateOp rhsStage,
                                     unsigned depth = 0);

static bool isCorrespondingStageLocalForIv(Value lhs,
                                           sde::SdeSuIterateOp lhsStage,
                                           Value rhs,
                                           sde::SdeSuIterateOp rhsStage,
                                           unsigned depth) {
  auto lhsArg = dyn_cast<BlockArgument>(lhs);
  auto rhsArg = dyn_cast<BlockArgument>(rhs);
  if (!lhsArg || !rhsArg || lhsArg.getArgNumber() != 0 ||
      rhsArg.getArgNumber() != 0)
    return false;

  auto lhsFor = dyn_cast_or_null<scf::ForOp>(lhsArg.getOwner()->getParentOp());
  auto rhsFor = dyn_cast_or_null<scf::ForOp>(rhsArg.getOwner()->getParentOp());
  if (!lhsFor || !rhsFor)
    return false;
  if (lhsFor->getParentOfType<sde::SdeSuIterateOp>() != lhsStage ||
      rhsFor->getParentOfType<sde::SdeSuIterateOp>() != rhsStage)
    return false;

  if (!areStageValuesEquivalent(lhsFor.getLowerBound(), lhsStage,
                                rhsFor.getLowerBound(), rhsStage, depth + 1))
    return false;
  if (!areStageValuesEquivalent(lhsFor.getUpperBound(), lhsStage,
                                rhsFor.getUpperBound(), rhsStage, depth + 1))
    return false;
  if (!areStageValuesEquivalent(lhsFor.getStep(), lhsStage, rhsFor.getStep(),
                                rhsStage, depth + 1))
    return false;
  return true;
}

static SmallVector<Value, 4> mapValues(ValueRange values, IRMapping &mapping) {
  SmallVector<Value, 4> mapped;
  mapped.reserve(values.size());
  for (Value value : values)
    mapped.push_back(mapping.lookupOrDefault(value));
  return mapped;
}

static void cloneSkippablePrefixOps(Operation *root, sde::SdeSuIterateOp until,
                                    IRRewriter &rewriter, IRMapping &mapping) {
  auto cuRegion = dyn_cast_or_null<sde::SdeCuRegionOp>(root);
  if (!cuRegion || !cuRegion.getBody().hasOneBlock())
    return;

  for (Operation &inner : cuRegion.getBody().front().without_terminator()) {
    if (&inner == until.getOperation())
      return;
    if (!isSkippableInterStageOp(&inner))
      return;
    rewriter.clone(inner, mapping);
  }
}

static bool haveSameIterationSpace(sde::SdeSuIterateOp lhs,
                                   sde::SdeSuIterateOp rhs) {
  if (lhs.getLowerBounds().size() != rhs.getLowerBounds().size() ||
      lhs.getUpperBounds().size() != rhs.getUpperBounds().size() ||
      lhs.getSteps().size() != rhs.getSteps().size())
    return false;

  for (auto [a, b] : llvm::zip(lhs.getLowerBounds(), rhs.getLowerBounds())) {
    if (!::mlir::carts::ValueAnalysis::areValuesEquivalent(a, b))
      return false;
  }
  for (auto [a, b] : llvm::zip(lhs.getUpperBounds(), rhs.getUpperBounds())) {
    if (!::mlir::carts::ValueAnalysis::areValuesEquivalent(a, b))
      return false;
  }
  for (auto [a, b] : llvm::zip(lhs.getSteps(), rhs.getSteps())) {
    if (!::mlir::carts::ValueAnalysis::areValuesEquivalent(a, b))
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
  return ::mlir::carts::ValueAnalysis::areValuesEquivalent(lhsChunk, rhsChunk);
}

static Value getWriteRoot(Value value) {
  return ::mlir::carts::ValueAnalysis::stripMemrefViewOps(value);
}

static bool hasDisjointWrites(ArrayRef<ElementwiseStage> stages) {
  llvm::DenseSet<Value> writtenTargets;
  for (const ElementwiseStage &stage : stages) {
    for (const MemrefAccess &write : stage.writes) {
      if (!write.root || !writtenTargets.insert(write.root).second)
        return false;
    }
  }
  return true;
}

static bool hasWriteToRoot(const ElementwiseStage &stage, Value root) {
  for (const MemrefAccess &write : stage.writes)
    if (write.root == root)
      return true;
  return false;
}

static bool isCorrespondingStageArg(Value lhs, sde::SdeSuIterateOp lhsStage,
                                    Value rhs, sde::SdeSuIterateOp rhsStage) {
  auto lhsArg = dyn_cast<BlockArgument>(lhs);
  auto rhsArg = dyn_cast<BlockArgument>(rhs);
  if (!lhsArg || !rhsArg)
    return false;
  if (lhsArg.getOwner() != &lhsStage.getBody().front() ||
      rhsArg.getOwner() != &rhsStage.getBody().front())
    return false;
  return lhsArg.getArgNumber() == rhsArg.getArgNumber();
}

static bool areStageValuesEquivalent(Value lhs, sde::SdeSuIterateOp lhsStage,
                                     Value rhs, sde::SdeSuIterateOp rhsStage,
                                     unsigned depth) {
  if (!lhs || !rhs || depth > 8)
    return false;
  lhs = ::mlir::carts::ValueAnalysis::stripNumericCasts(lhs);
  rhs = ::mlir::carts::ValueAnalysis::stripNumericCasts(rhs);
  if (::mlir::carts::ValueAnalysis::sameValue(lhs, rhs))
    return true;
  if (isCorrespondingStageArg(lhs, lhsStage, rhs, rhsStage))
    return true;
  if (isCorrespondingStageLocalForIv(lhs, lhsStage, rhs, rhsStage, depth))
    return true;

  auto lhsConst = ::mlir::carts::ValueAnalysis::tryFoldConstantIndex(lhs);
  auto rhsConst = ::mlir::carts::ValueAnalysis::tryFoldConstantIndex(rhs);
  if (lhsConst || rhsConst)
    return lhsConst && rhsConst && *lhsConst == *rhsConst;

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef || lhsDef->getName() != rhsDef->getName() ||
      lhsDef->getNumOperands() != rhsDef->getNumOperands())
    return false;

  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhsDef->getOperands(), rhsDef->getOperands()))
    if (!areStageValuesEquivalent(lhsOperand, lhsStage, rhsOperand, rhsStage,
                                  depth + 1))
      return false;
  return true;
}

static bool areStageAccessIndicesEquivalent(ValueRange lhs,
                                            sde::SdeSuIterateOp lhsStage,
                                            ValueRange rhs,
                                            sde::SdeSuIterateOp rhsStage) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [lhsIndex, rhsIndex] : llvm::zip(lhs, rhs))
    if (!areStageValuesEquivalent(lhsIndex, lhsStage, rhsIndex, rhsStage))
      return false;
  return true;
}

static bool readMatchesProducerWrite(const MemrefAccess &read,
                                     const ElementwiseStage &consumer,
                                     const ElementwiseStage &producer) {
  for (const MemrefAccess &write : producer.writes) {
    if (write.root != read.root)
      continue;
    if (areStageAccessIndicesEquivalent(read.indices, consumer.op,
                                        write.indices, producer.op))
      return true;
  }
  return false;
}

static bool hasUnsafeReadAfterWrite(ArrayRef<ElementwiseStage> stages) {
  for (unsigned consumerIdx = 1; consumerIdx < stages.size(); ++consumerIdx) {
    const ElementwiseStage &consumer = stages[consumerIdx];
    for (const MemrefAccess &read : consumer.reads) {
      for (unsigned producerIdx = 0; producerIdx < consumerIdx; ++producerIdx) {
        const ElementwiseStage &producer = stages[producerIdx];
        if (!hasWriteToRoot(producer, read.root))
          continue;
        if (!readMatchesProducerWrite(read, consumer, producer))
          return true;
      }
    }
  }
  return false;
}

static bool isElementwiseStage(Operation *root, ElementwiseStage &stage) {
  sde::SdeSuIterateOp op = getStageSuIterate(root);
  if (!op || !op.getStructuredClassificationAttr() ||
      *op.getStructuredClassification() !=
          sde::SdeStructuredClassification::elementwise)
    return false;
  if (!op.getReductionAccumulators().empty())
    return false;

  llvm::DenseSet<Value> seenWrites;
  SmallVector<MemrefAccess, 4> reads;
  SmallVector<MemrefAccess, 4> writes;
  bool hasDuplicateWrite = false;

  op.getBody().walk([&](memref::LoadOp loadOp) {
    Value target = getWriteRoot(loadOp.getMemref());
    if (!target)
      return;
    if (sde::isDefinedInside(op.getOperation(), target))
      return;
    reads.push_back(
        MemrefAccess{target, SmallVector<Value, 4>(loadOp.getIndices())});
  });
  op.getBody().walk([&](memref::StoreOp storeOp) {
    Value target = getWriteRoot(storeOp.getMemref());
    if (!target)
      return;
    if (sde::isDefinedInside(op.getOperation(), target))
      return;
    if (!seenWrites.insert(target).second) {
      hasDuplicateWrite = true;
      return;
    }
    writes.push_back(
        MemrefAccess{target, SmallVector<Value, 4>(storeOp.getIndices())});
  });

  if (writes.empty() || hasDuplicateWrite)
    return false;

  stage = {op, root, std::move(reads), std::move(writes)};
  return true;
}

static sde::SdeSuIterateOp fuseStages(MutableArrayRef<ElementwiseStage> stages,
                                      IRRewriter &rewriter) {
  assert(stages.size() >= 2 && "expected at least two stages");

  sde::SdeSuIterateOp first = stages.front().op;
  Location loc = first.getLoc();

  bool wrappedStages = llvm::all_of(stages, [](ElementwiseStage &stage) {
    return stage.root != stage.op.getOperation() &&
           isa<sde::SdeCuRegionOp>(stage.root);
  });

  Block *insertionBlock = nullptr;
  IRMapping fusedOperandMapping;
  if (wrappedStages) {
    rewriter.setInsertionPoint(stages.back().root);
    auto outerCuRegion = sde::SdeCuRegionOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{},
        sde::SdeCuKindAttr::get(rewriter.getContext(),
                                sde::SdeCuKind::parallel),
        /*nowait=*/nullptr,
        /*iterArgs=*/ValueRange{});
    insertionBlock = &sde::ensureBlock(outerCuRegion.getBody());
    rewriter.setInsertionPointToStart(insertionBlock);
    cloneSkippablePrefixOps(stages.front().root, first, rewriter,
                            fusedOperandMapping);
  } else {
    rewriter.setInsertionPoint(stages.back().root);
  }

  SmallVector<Value, 4> lowerBounds =
      mapValues(first.getLowerBounds(), fusedOperandMapping);
  SmallVector<Value, 4> upperBounds =
      mapValues(first.getUpperBounds(), fusedOperandMapping);
  SmallVector<Value, 4> steps =
      mapValues(first.getSteps(), fusedOperandMapping);
  Value chunkSize =
      first.getChunkSize()
          ? fusedOperandMapping.lookupOrDefault(first.getChunkSize())
          : Value{};

  auto fused = sde::SdeSuIterateOp::create(
      rewriter, loc, /*resultTypes=*/TypeRange{}, lowerBounds, upperBounds,
      steps, first.getScheduleAttr(), chunkSize, first.getNowaitAttr(),
      first.getReductionAccumulators(), first.getReductionKindsAttr(),
      first.getReductionStrategyAttr(), first.getPartialReductionAttr(),
      first.getPartialReductionDimsAttr(),
      first.getPartialReductionOwnerDimsAttr(),
      sde::SdeStructuredClassificationAttr::get(
          first.getContext(),
          sde::SdeStructuredClassification::elementwise_pipeline),
      sde::SdePatternAttr::get(first.getContext(),
                               sde::SdePattern::elementwise_pipeline),
      first.getAccessMinOffsetsAttr(), first.getAccessMaxOffsetsAttr(),
      first.getOwnerDimsAttr(), first.getSpatialDimsAttr(),
      first.getWriteFootprintAttr(), first.getPhysicalOwnerDimsAttr(),
      first.getPhysicalBlockShapeAttr(), first.getLogicalWorkerSliceAttr(),
      first.getPhysicalHaloShapeAttr(), first.getIterationTopologyAttr(),
      first.getRepetitionStructureAttr(), first.getAsyncStrategyAttr(),
      first.getCpsGroupIdAttr(), first.getCpsStageIndexAttr(),
      first.getCpsStageCountAttr(), first.getDistributionKindAttr(),
      first.getInPlaceSafeAttr(), first.getInPlaceSharedStateAttr(),
      first.getVectorizeWidthAttr(), first.getUnrollFactorAttr(),
      first.getInterleaveCountAttr());
  fused->setAttrs(sde::getRewrittenAttrs(first));
  fused.setStructuredClassificationAttr(
      sde::SdeStructuredClassificationAttr::get(
          first.getContext(),
          sde::SdeStructuredClassification::elementwise_pipeline));
  fused.setPatternAttr(sde::SdePatternAttr::get(
      first.getContext(), sde::SdePattern::elementwise_pipeline));

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
    cloneSkippablePrefixOps(stage.root, stage.op, rewriter, mapper);
    for (Operation &nested : srcBody->without_terminator())
      rewriter.clone(nested, mapper);
  }
  sde::SdeYieldOp::create(rewriter, loc, ValueRange{});

  // Yield at su_iterate level.
  rewriter.setInsertionPointAfter(innerCuRegion);
  sde::SdeYieldOp::create(rewriter, loc, ValueRange{});

  if (wrappedStages) {
    rewriter.setInsertionPointToEnd(insertionBlock);
    sde::SdeYieldOp::create(rewriter, loc, ValueRange{});
  }

  return fused;
}

struct ElementwiseFusionPass
    : public sde::impl::ElementwiseFusionBase<ElementwiseFusionPass> {
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
          ElementwiseStage firstStage;
          if (!isElementwiseStage(&*it, firstStage))
            continue;
          sde::SdeSuIterateOp first = firstStage.op;

          SmallVector<ElementwiseStage, 4> stages;
          stages.push_back(std::move(firstStage));
          for (auto nextIt = std::next(it); nextIt != e; ++nextIt) {
            if (isSkippableInterStageOp(&*nextIt))
              continue;

            ElementwiseStage nextStage;
            if (!isElementwiseStage(&*nextIt, nextStage) ||
                !haveSameIterationSpace(first, nextStage.op) ||
                !haveCompatibleSchedule(first, nextStage.op))
              break;

            stages.push_back(std::move(nextStage));
            if (!hasDisjointWrites(stages) || hasUnsafeReadAfterWrite(stages)) {
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
            rewriter.eraseOp(stage.root);
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

namespace mlir::carts::sde {

std::unique_ptr<Pass> createElementwiseFusionPass() {
  return std::make_unique<ElementwiseFusionPass>();
}

} // namespace mlir::carts::sde
