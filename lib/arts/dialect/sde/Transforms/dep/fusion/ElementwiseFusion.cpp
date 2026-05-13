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

#include "arts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "arts/utils/ValueAnalysis.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::arts;

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

static sde::SdeSuIterateOp getStageSuIterate(Operation *op) {
  if (auto suIter = dyn_cast_or_null<sde::SdeSuIterateOp>(op))
    return suIter;

  auto cuRegion = dyn_cast_or_null<sde::SdeCuRegionOp>(op);
  if (!cuRegion || cuRegion.getKind() != sde::SdeCuKind::parallel ||
      !cuRegion.getBody().hasOneBlock())
    return {};

  sde::SdeSuIterateOp nested;
  for (Operation &inner : cuRegion.getBody().front().without_terminator()) {
    auto suIter = dyn_cast<sde::SdeSuIterateOp>(inner);
    if (!suIter || nested)
      return {};
    nested = suIter;
  }
  return nested;
}

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
                                     unsigned depth = 0) {
  if (!lhs || !rhs || depth > 8)
    return false;
  lhs = ValueAnalysis::stripNumericCasts(lhs);
  rhs = ValueAnalysis::stripNumericCasts(rhs);
  if (ValueAnalysis::sameValue(lhs, rhs))
    return true;
  if (isCorrespondingStageArg(lhs, lhsStage, rhs, rhsStage))
    return true;

  auto lhsConst = ValueAnalysis::tryFoldConstantIndex(lhs);
  auto rhsConst = ValueAnalysis::tryFoldConstantIndex(rhs);
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

  // Carrier-authoritative path: extract write-set from carrier DPS outputs.
  Block *computeBody = sde::getSuIterateComputeBlock(op);
  linalg::GenericOp carrier;
  for (Operation &inner : computeBody->without_terminator())
    if (auto g = dyn_cast<linalg::GenericOp>(inner)) {
      carrier = g;
      break;
    }

  if (carrier) {
    for (Value input : carrier.getDpsInputs()) {
      Value target = sde::traceCarrierTensorToMemrefRoot(input);
      if (!target)
        return false;
      reads.push_back(MemrefAccess{target, {}});
    }
    for (Value output : carrier.getDpsInits()) {
      Value target = sde::traceCarrierTensorToMemrefRoot(output);
      if (!target)
        return false;
      if (!seenWrites.insert(target).second) {
        hasDuplicateWrite = true;
        break;
      }
      writes.push_back(MemrefAccess{target, {}});
    }
  } else {
    // Fallback: walk memref.store ops (dual-rep / stencil path).
    op.getBody().walk([&](memref::LoadOp loadOp) {
      Value target = getWriteRoot(loadOp.getMemref());
      if (!target)
        return;
      reads.push_back(
          MemrefAccess{target, SmallVector<Value, 4>(loadOp.getIndices())});
    });
    op.getBody().walk([&](memref::StoreOp storeOp) {
      Value target = getWriteRoot(storeOp.getMemref());
      if (!target)
        return;
      if (!seenWrites.insert(target).second) {
        hasDuplicateWrite = true;
        return;
      }
      writes.push_back(
          MemrefAccess{target, SmallVector<Value, 4>(storeOp.getIndices())});
    });
  }

  if (writes.empty() || hasDuplicateWrite)
    return false;

  stage = {op, root, std::move(reads), std::move(writes)};
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

  bool wrappedStages = llvm::all_of(stages, [](ElementwiseStage &stage) {
    return stage.root != stage.op.getOperation() &&
           isa<sde::SdeCuRegionOp>(stage.root);
  });

  Block *insertionBlock = nullptr;
  if (wrappedStages) {
    rewriter.setInsertionPoint(stages.back().root);
    auto outerCuRegion = sde::SdeCuRegionOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{},
        sde::SdeCuKindAttr::get(rewriter.getContext(),
                                sde::SdeCuKind::parallel),
        /*concurrency_scope=*/nullptr,
        /*nowait=*/nullptr,
        /*iterArgs=*/ValueRange{});
    insertionBlock = &sde::ensureBlock(outerCuRegion.getBody());
    rewriter.setInsertionPointToStart(insertionBlock);
  } else {
    rewriter.setInsertionPoint(stages.back().root);
  }

  auto fused = sde::SdeSuIterateOp::create(
      rewriter, loc, /*resultTypes=*/TypeRange{}, first.getLowerBounds(),
      first.getUpperBounds(), first.getSteps(), first.getScheduleAttr(),
      first.getChunkSize(), first.getNowaitAttr(),
      first.getReductionAccumulators(), first.getReductionKindsAttr(),
      first.getReductionStrategyAttr(),
      sde::SdeStructuredClassificationAttr::get(
          first.getContext(),
          sde::SdeStructuredClassification::elementwise_pipeline),
      sde::SdeDepFamilyAttr::get(first.getContext(),
                                 sde::SdeDepFamily::elementwise_pipeline),
      first.getAccessMinOffsetsAttr(), first.getAccessMaxOffsetsAttr(),
      first.getOwnerDimsAttr(), first.getSpatialDimsAttr(),
      first.getWriteFootprintAttr(), first.getPhysicalOwnerDimsAttr(),
      first.getPhysicalBlockShapeAttr(), first.getLogicalWorkerSliceAttr(),
      first.getPhysicalHaloShapeAttr(), first.getIterationTopologyAttr(),
      first.getRepetitionStructureAttr(), first.getAsyncStrategyAttr(),
      first.getDistributionKindAttr(), first.getInPlaceSafeAttr(),
      first.getInPlaceSharedStateAttr(), first.getVectorizeWidthAttr(),
      first.getUnrollFactorAttr(), first.getInterleaveCountAttr());
  fused->setAttrs(sde::getRewrittenAttrs(first));
  fused.setStructuredClassificationAttr(
      sde::SdeStructuredClassificationAttr::get(
          first.getContext(),
          sde::SdeStructuredClassification::elementwise_pipeline));
  fused.setDepFamilyAttr(sde::SdeDepFamilyAttr::get(
      first.getContext(), sde::SdeDepFamily::elementwise_pipeline));

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

  if (wrappedStages) {
    rewriter.setInsertionPointToEnd(insertionBlock);
    sde::SdeYieldOp::create(rewriter, loc, ValueRange{});
  }

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

namespace mlir::arts::sde {

std::unique_ptr<Pass> createElementwiseFusionPass() {
  return std::make_unique<ElementwiseFusionPass>();
}

} // namespace mlir::arts::sde
