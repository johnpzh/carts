///==========================================================================///
/// File: EdtRewriter.cpp
///
/// Implements acquire rewriting helpers used by ForLowering.
///
/// Rewrite behavior:
///   1. Single-element or forced-coarse path:
///      db_acquire -> partitioning(<coarse>), offsets[0], sizes[1]
///   2. Block path:
///      db_acquire -> partitioning(<block>), worker offsets/sizes
///   3. Stencil extension (when enabled):
///      expand worker offsets/sizes with halo-aware [start-1, end+1] clamping
///   4. N-D owner hints:
///      carry worker-local partition windows in partition_offsets/sizes and
///      materialize full-rank element_offsets/element_sizes from the semantic
///      contract when runtime slice dependencies are profitable
///==========================================================================///

#include "arts/dialect/core/Transforms/edt/EdtRewriter.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/LoweringContractUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/PartitionPredicates.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::arts;

namespace {

static bool hasConcreteReadHaloContract(const LoweringContractInfo &contract,
                                        ArtsMode effectiveMode) {
  if (effectiveMode != ArtsMode::in)
    return false;
  auto minOffsets = contract.getStaticMinOffsets();
  auto maxOffsets = contract.getStaticMaxOffsets();
  if (!minOffsets || !maxOffsets || minOffsets->size() != maxOffsets->size())
    return false;
  return llvm::any_of(llvm::zip(*minOffsets, *maxOffsets), [](auto pair) {
    auto [minOffset, maxOffset] = pair;
    return minOffset != 0 || maxOffset != 0;
  });
}

struct TaskAcquireWindow {
  Value elementOffset;
  Value elementSize;
  Value dbOffset;
  Value dbSize;
};

static DbAllocOp resolveTaskRootAlloc(Value rootGuid, Value rootPtr,
                                      DbAcquireOp parentAcquire) {
  if (auto allocFromGuid = rootGuid.getDefiningOp<DbAllocOp>())
    return allocFromGuid;
  if (Operation *allocOp = DbUtils::getUnderlyingDbAlloc(rootPtr))
    return dyn_cast<DbAllocOp>(allocOp);
  if (parentAcquire)
    if (Operation *allocOp =
            DbUtils::getUnderlyingDbAlloc(parentAcquire.getPtr()))
      return dyn_cast<DbAllocOp>(allocOp);
  return nullptr;
}

static std::optional<TaskAcquireWindow>
computeTaskAcquireWindow(TaskAcquireSlicePlanInput input) {
  if (!input.AC || !input.parentAcquire || !input.taskAcquire)
    return std::nullopt;

  PartitionMode mode =
      input.taskAcquire.getPartitionMode().value_or(PartitionMode::coarse);
  if (!usesBlockLayout(mode))
    return std::nullopt;

  Value one = input.AC->createIndexConstant(1, input.loc);
  Value zero = input.AC->createIndexConstant(0, input.loc);
  Value blockSpan = one;
  Value totalBlocks;
  DbAllocOp rootAlloc =
      resolveTaskRootAlloc(input.rootGuid, input.rootPtr, input.parentAcquire);

  if (rootAlloc) {
    auto elementSizes = rootAlloc.getElementSizes();
    if (!elementSizes.empty() && elementSizes.front())
      blockSpan = input.AC->castToIndex(elementSizes.front(), input.loc);
    auto outerSizes = rootAlloc.getSizes();
    if (!outerSizes.empty() && outerSizes.front())
      totalBlocks = input.AC->castToIndex(outerSizes.front(), input.loc);
  }
  blockSpan = input.AC->create<arith::MaxUIOp>(input.loc, blockSpan, one);

  /// This helper intentionally reasons in leading-dimension block space only.
  /// It converts the already chosen task-local element window into the single
  /// block window used by DB slice metadata on 1-D / row-owned plans.
  Value elementOffset = input.plannedElementOffset;
  Value elementSize = input.plannedElementSizeSeed;
  const bool useRewrittenWindow =
      shouldUsePartitionSliceAsDepWindow(input.contract, input.taskAcquire);
  const bool hasReadHaloContract =
      hasConcreteReadHaloContract(input.contract, input.effectiveMode);
  const bool applyStencilHalo =
      shouldApplyStencilHalo(input.contract, input.taskAcquire);
  const bool useTaskDepWindow =
      input.usesStencilHalo || applyStencilHalo || hasReadHaloContract;
  const bool preserveParentDepRange =
      shouldPreserveParentDepRange(input.contract, input.taskAcquire);
  const bool useTaskPartitionWindow =
      !input.taskAcquire.getPartitionOffsets().empty() &&
      !input.taskAcquire.getPartitionSizes().empty() &&
      (input.effectiveMode != ArtsMode::in || !preserveParentDepRange ||
       useRewrittenWindow);

  if (useTaskPartitionWindow) {
    auto offsetRange = input.taskAcquire.getPartitionOffsets();
    auto sizeRange = input.taskAcquire.getPartitionSizes();

    if (!offsetRange.empty() && offsetRange.front())
      elementOffset = offsetRange.front();
    if (!sizeRange.empty() && sizeRange.front())
      elementSize = sizeRange.front();
  } else if (useTaskDepWindow) {
    auto offsetRange = useRewrittenWindow
                           ? input.taskAcquire.getPartitionOffsets()
                           : input.taskAcquire.getOffsets();
    auto sizeRange = useRewrittenWindow ? input.taskAcquire.getPartitionSizes()
                                        : input.taskAcquire.getSizes();

    if (!offsetRange.empty() && offsetRange.front())
      elementOffset = offsetRange.front();

    if (!sizeRange.empty() && sizeRange.front())
      elementSize = sizeRange.front();
  } else {
    if (!input.taskAcquire.getSizes().empty() &&
        input.taskAcquire.getSizes().front())
      elementSize = input.taskAcquire.getSizes().front();

    if (input.distributionKind == DistributionKind::TwoLevel &&
        input.effectiveMode != ArtsMode::in)
      elementSize = input.plannedElementSize;

    bool isStencilPattern =
        input.distributionPattern &&
        *input.distributionPattern == EdtDistributionPattern::stencil;
    if (isStencilPattern && mode == PartitionMode::stencil &&
        input.effectiveMode != ArtsMode::in)
      elementSize =
          input.AC->create<arith::AddIOp>(input.loc, elementSize, one);
  }

  elementOffset = input.AC->castToIndex(elementOffset, input.loc);
  elementSize = input.AC->castToIndex(elementSize, input.loc);
  elementSize = input.AC->create<arith::MaxUIOp>(input.loc, elementSize, one);

  Value startBlock =
      input.AC->create<arith::DivUIOp>(input.loc, elementOffset, blockSpan);
  Value endElem =
      input.AC->create<arith::AddIOp>(input.loc, elementOffset, elementSize);
  endElem = input.AC->create<arith::SubIOp>(input.loc, endElem, one);
  Value endBlock =
      input.AC->create<arith::DivUIOp>(input.loc, endElem, blockSpan);

  Value startAboveMax;
  if (totalBlocks) {
    Value maxBlock =
        input.AC->create<arith::SubIOp>(input.loc, totalBlocks, one);
    startAboveMax = input.AC->create<arith::CmpIOp>(
        input.loc, arith::CmpIPredicate::ugt, startBlock, maxBlock);
    Value clampedEnd =
        input.AC->create<arith::MinUIOp>(input.loc, endBlock, maxBlock);
    endBlock = input.AC->create<arith::SelectOp>(input.loc, startAboveMax,
                                                 endBlock, clampedEnd);
  }

  Value blockCount =
      input.AC->create<arith::SubIOp>(input.loc, endBlock, startBlock);
  blockCount = input.AC->create<arith::AddIOp>(input.loc, blockCount, one);
  if (startAboveMax) {
    startBlock = input.AC->create<arith::SelectOp>(input.loc, startAboveMax,
                                                   zero, startBlock);
    blockCount = input.AC->create<arith::SelectOp>(input.loc, startAboveMax,
                                                   zero, blockCount);
  }

  return TaskAcquireWindow{elementOffset, elementSize, startBlock, blockCount};
}

struct TaskElementSlice {
  SmallVector<Value, 4> offsets;
  SmallVector<Value, 4> sizes;
};

static bool shouldMaterializeTaskElementSlice(TaskAcquireSlicePlanInput input) {
  if (!input.AC || !input.parentAcquire || !input.taskAcquire)
    return false;

  PartitionMode mode =
      input.taskAcquire.getPartitionMode().value_or(PartitionMode::coarse);
  if (!usesBlockLayout(mode))
    return false;

  if (input.usesStencilHalo)
    return input.effectiveMode == ArtsMode::in;

  if (hasConcreteReadHaloContract(input.contract, input.effectiveMode))
    return true;

  if (shouldUsePartitionSliceAsDepWindow(input.contract, input.taskAcquire)) {
    /// A single element_offsets/element_sizes pair is only sound when the
    /// entire acquire lowers to one slice shape. Writer-capable stencil
    /// windows expand into multiple DB slots whose predecessor/successor face
    /// slices differ per slot, so their ESD byte ranges must be reconstructed
    /// later from the semantic contract during record-dep lowering.
    return input.effectiveMode == ArtsMode::in;
  }

  if (input.effectiveMode == ArtsMode::in)
    return !shouldPreserveParentDepRange(input.contract, input.taskAcquire);

  /// Uniform block writers carry a single owner-local slice shape. Materialize
  /// it so later lowering can reason from the same worker-local element
  /// contract already attached to the corresponding reads.
  bool isUniformBlock =
      input.distributionPattern &&
      *input.distributionPattern == EdtDistributionPattern::uniform;
  return isUniformBlock && !input.taskAcquire.getPartitionOffsets().empty() &&
         !input.taskAcquire.getPartitionSizes().empty() &&
         !input.contract.spatial.ownerDims.empty();
}

static std::optional<TaskElementSlice>
computeTaskElementSlice(TaskAcquireSlicePlanInput input) {
  if (!shouldMaterializeTaskElementSlice(input))
    return std::nullopt;

  DbAllocOp rootAlloc =
      resolveTaskRootAlloc(input.rootGuid, input.rootPtr, input.parentAcquire);
  SmallVector<Value, 4> blockExtents;
  std::optional<LoweringContractInfo> taskContract =
      getLoweringContract(input.taskAcquire.getPtr());
  if ((!taskContract || (!taskContract->getStaticBlockShape() &&
                         taskContract->spatial.blockShape.empty())))
    if (auto opContract =
            getLoweringContract(input.taskAcquire.getOperation(),
                                input.AC->getBuilder(), input.loc))
      taskContract = *opContract;
  if (taskContract) {
    if (auto staticShape = taskContract->getStaticBlockShape()) {
      for (int64_t dim : *staticShape)
        blockExtents.push_back(input.AC->createIndexConstant(dim, input.loc));
    } else if (!taskContract->spatial.blockShape.empty()) {
      blockExtents.assign(taskContract->spatial.blockShape.begin(),
                          taskContract->spatial.blockShape.end());
    }
  }
  if (blockExtents.empty() && rootAlloc)
    blockExtents.assign(rootAlloc.getElementSizes().begin(),
                        rootAlloc.getElementSizes().end());
  if (blockExtents.empty())
    return std::nullopt;

  auto elementOffsets = input.taskAcquire.getPartitionOffsets();
  auto elementSizes = input.taskAcquire.getPartitionSizes();
  unsigned explicitRank =
      std::min<unsigned>(elementOffsets.size(), elementSizes.size());
  if (explicitRank == 0)
    return std::nullopt;

  LoweringContractInfo contractInfo;
  if (taskContract && !taskContract->spatial.ownerDims.empty())
    contractInfo.spatial.ownerDims = taskContract->spatial.ownerDims;
  else
    contractInfo.spatial.ownerDims = input.contract.spatial.ownerDims;
  /// Element-space task slices need an authoritative mapping from partition
  /// entries back to physical memref dims. Falling back to implicit leading
  /// dims is not a benign heuristic here: it fabricates owner-local slices for
  /// reduction inputs whose accesses do not actually follow the distributed
  /// loop IV. When the contract cannot name owner dims, keep the dependency
  /// full- range and let later passes preserve correctness.
  if (contractInfo.spatial.ownerDims.empty())
    return std::nullopt;
  SmallVector<unsigned, 4> explicitDims =
      resolveContractOwnerDims(contractInfo, explicitRank);
  if (explicitDims.size() != explicitRank)
    return std::nullopt;

  Value zero = input.AC->createIndexConstant(0, input.loc);
  TaskElementSlice slice;
  slice.offsets.reserve(blockExtents.size());
  slice.sizes.reserve(blockExtents.size());
  for (Value extent : blockExtents) {
    slice.offsets.push_back(zero);
    slice.sizes.push_back(input.AC->castToIndex(extent, input.loc));
  }

  for (unsigned i = 0; i < explicitRank; ++i) {
    unsigned dim = explicitDims[i];
    if (dim >= blockExtents.size())
      return std::nullopt;
    slice.offsets[dim] = input.AC->castToIndex(elementOffsets[i], input.loc);
    slice.sizes[dim] = input.AC->castToIndex(elementSizes[i], input.loc);
  }

  return slice;
}

} // namespace

TaskAcquireRewritePlan
mlir::arts::planTaskAcquireRewrite(TaskAcquireRewritePlanInput input) {
  auto makePlan = [&]() {
    return TaskAcquireRewritePlan{
        AcquireRewriteInput{
            input.AC, input.loc, input.parentAcquire, input.effectiveMode,
            input.rootGuid, input.rootPtr, input.acquireOffset,
            input.acquireSize, input.acquireHintSize,
            /*extraOffsets=*/SmallVector<Value, 4>{},
            /*extraSizes=*/SmallVector<Value, 4>{},
            /*extraHintSizes=*/SmallVector<Value, 4>{},
            /*dimensionExtents=*/SmallVector<Value, 4>{},
            /*haloMinOffsets=*/SmallVector<int64_t, 4>{},
            /*haloMaxOffsets=*/SmallVector<int64_t, 4>{}, input.step,
            input.stepIsUnit,
            /*singleElement=*/false,
            /*forceCoarse=*/
            input.forceCoarseRewrite ||
                input.distributionKind == DistributionKind::BlockCyclic,
            /*preserveParentDepRange=*/false,
            /*stencilExtent=*/Value()},
        /*useStencilRewriter=*/false,
        /*refinedTaskBlockShape=*/std::nullopt};
  };

  if (!input.AC || !input.parentAcquire)
    return makePlan();

  Value zero = input.AC->createIndexConstant(0, input.loc);
  Value one = input.AC->createIndexConstant(1, input.loc);
  TaskAcquireRewritePlan plan{makePlan()};

  bool isSingleElement = false;
  bool forceCoarseRewrite = plan.rewriteInput.forceCoarse;
  Value stencilExtent;
  SmallVector<Value, 4> extraOffsets;
  SmallVector<Value, 4> extraSizes;
  SmallVector<Value, 4> extraHintSizes;
  SmallVector<Value, 4> dimensionExtents;
  const LoweringContractInfo &contract = input.contract;

  /// Check if we should apply stencil halo extension. Some pipelines stamp the
  /// full stencil contract on the loop/task shape, so the halo bounds may be
  /// present even before explicit applyStencilHalo enablement.
  bool applyStencilHalo = shouldApplyStencilHalo(contract, input.parentAcquire);
  bool preserveParentDepRange =
      shouldPreserveParentDepRange(contract, input.parentAcquire);

  auto haloMinOffsets = contract.getStaticMinOffsets();
  auto haloMaxOffsets = contract.getStaticMaxOffsets();
  if (!applyStencilHalo && input.effectiveMode == ArtsMode::in &&
      haloMinOffsets && haloMaxOffsets) {
    /// Enable halo-aware task window planning when contract has halo bounds.
    applyStencilHalo = true;
    preserveParentDepRange = false;
  }

  if (Operation *rootAllocOp = DbUtils::getUnderlyingDbAlloc(input.rootPtr)) {
    if (auto dbAlloc = dyn_cast<DbAllocOp>(rootAllocOp)) {
      auto elemSizes = dbAlloc.getElementSizes();
      if (!elemSizes.empty()) {
        std::optional<int64_t> totalElemCount = int64_t{1};
        for (Value dim : elemSizes) {
          auto dimConst = ValueAnalysis::tryFoldConstantIndex(dim);
          if (!dimConst || *dimConst < 0) {
            totalElemCount = std::nullopt;
            break;
          }
          *totalElemCount *= *dimConst;
        }

        isSingleElement = llvm::all_of(elemSizes, [](Value value) {
          int64_t constant = 0;
          return ValueAnalysis::getConstantIndex(value, constant) &&
                 constant == 1;
        });

        /// Keep tiny read-only stencil coefficient tables coarse.
        /// Rewriting these to block slices per task can shift coefficient
        /// indexing and introduce thread-count-dependent numerics.
        if (input.effectiveMode == ArtsMode::in) {
          auto allocPattern = getDbAccessPattern(dbAlloc.getOperation());
          bool tinyReadOnlyStencil =
              allocPattern && *allocPattern == DbAccessPattern::stencil &&
              totalElemCount && *totalElemCount > 0 && *totalElemCount <= 8;
          if (tinyReadOnlyStencil)
            forceCoarseRewrite = true;
        }

        unsigned primaryOwnerDim = 0;
        if (!contract.spatial.ownerDims.empty() &&
            contract.spatial.ownerDims.front() >= 0 &&
            static_cast<size_t>(contract.spatial.ownerDims.front()) <
                elemSizes.size()) {
          primaryOwnerDim =
              static_cast<unsigned>(contract.spatial.ownerDims.front());
        }

        auto outerSizes = dbAlloc.getSizes();
        auto buildTotalOwnerExtent = [&](unsigned ownerSlot,
                                         unsigned physicalDim) -> Value {
          Value extent =
              input.AC->castToIndex(elemSizes[physicalDim], input.loc);
          /// DbAllocOp sizes are stored in owner-dimension order, while
          /// elementSizes uses physical memref dimensions. Halo clamping needs
          /// the total logical extent along the distributed owner dimension,
          /// not the per-block span.
          if (ownerSlot < outerSizes.size() && outerSizes[ownerSlot]) {
            Value blocks =
                input.AC->castToIndex(outerSizes[ownerSlot], input.loc);
            extent = input.AC->create<arith::MulIOp>(input.loc, blocks, extent);
          }
          return extent;
        };

        Value primaryExtent =
            buildTotalOwnerExtent(/*ownerSlot=*/0, primaryOwnerDim);
        if (applyStencilHalo)
          stencilExtent = primaryExtent;
        dimensionExtents.push_back(primaryExtent);

        if (input.distributionKind == DistributionKind::Tiling2D &&
            elemSizes.size() > 1 && !isSingleElement && input.tiling2DGrid &&
            input.tiling2DGrid->colWorkers) {
          Value colWorkers =
              input.AC->castToIndex(input.tiling2DGrid->colWorkers, input.loc);
          auto colWorkersConst =
              ValueAnalysis::tryFoldConstantIndex(colWorkers);
          if (!colWorkersConst || *colWorkersConst > 1) {
            unsigned secondaryOwnerDim = 1;
            if (contract.spatial.ownerDims.size() > 1 &&
                contract.spatial.ownerDims[1] >= 0 &&
                static_cast<size_t>(contract.spatial.ownerDims[1]) <
                    elemSizes.size()) {
              secondaryOwnerDim =
                  static_cast<unsigned>(contract.spatial.ownerDims[1]);
            }

            Value totalCols =
                buildTotalOwnerExtent(/*ownerSlot=*/1, secondaryOwnerDim);
            Value colWorkerId = input.AC->castToIndex(
                input.tiling2DGrid->colWorkerId, input.loc);

            Value colWorkersMinusOne =
                input.AC->create<arith::SubIOp>(input.loc, colWorkers, one);
            Value colAdjusted = input.AC->create<arith::AddIOp>(
                input.loc, totalCols, colWorkersMinusOne);
            Value colChunk = input.AC->create<arith::DivUIOp>(
                input.loc, colAdjusted, colWorkers);
            Value colOffset = input.AC->create<arith::MulIOp>(
                input.loc, colWorkerId, colChunk);

            Value colNeedZero = input.AC->create<arith::CmpIOp>(
                input.loc, arith::CmpIPredicate::uge, colOffset, totalCols);
            Value colRemaining = input.AC->create<arith::SubIOp>(
                input.loc, totalCols, colOffset);
            Value colRemainingNonNeg = input.AC->create<arith::SelectOp>(
                input.loc, colNeedZero, zero, colRemaining);
            Value colCount = input.AC->create<arith::MinUIOp>(
                input.loc, colChunk, colRemainingNonNeg);

            extraOffsets.push_back(colOffset);
            extraSizes.push_back(colCount);
            extraHintSizes.push_back(colCount);
            dimensionExtents.push_back(totalCols);
            preserveParentDepRange = false;

            auto rowWorkersConst =
                input.tiling2DGrid->rowWorkers
                    ? ValueAnalysis::tryFoldConstantIndex(input.AC->castToIndex(
                          input.tiling2DGrid->rowWorkers, input.loc))
                    : std::nullopt;
            SmallVector<int64_t, 4> staticTaskBlockShape;
            bool refinedShape = false;
            if (auto contract =
                    getLoweringContract(input.parentAcquire.getPtr())) {
              if (auto inheritedShape = contract->getStaticBlockShape())
                staticTaskBlockShape.assign(inheritedShape->begin(),
                                            inheritedShape->end());
            } else if (auto opContract = getLoweringContract(
                           input.parentAcquire.getOperation(),
                           input.AC->getBuilder(), input.loc)) {
              if (auto inheritedShape = opContract->getStaticBlockShape())
                staticTaskBlockShape.assign(inheritedShape->begin(),
                                            inheritedShape->end());
            }

            if (rowWorkersConst && *rowWorkersConst > 1 &&
                !staticTaskBlockShape.empty() &&
                primaryOwnerDim < elemSizes.size()) {
              if (auto totalRowsConst = ValueAnalysis::tryFoldConstantIndex(
                      elemSizes[primaryOwnerDim]);
                  totalRowsConst && *totalRowsConst > 0) {
                int64_t rowChunk =
                    (*totalRowsConst + *rowWorkersConst - 1) / *rowWorkersConst;
                staticTaskBlockShape[0] = std::max<int64_t>(1, rowChunk);
                refinedShape = true;
              }
            }

            if (colWorkersConst && *colWorkersConst > 1 &&
                staticTaskBlockShape.size() >= 2 &&
                secondaryOwnerDim < elemSizes.size()) {
              if (auto totalColsConst = ValueAnalysis::tryFoldConstantIndex(
                      elemSizes[secondaryOwnerDim]);
                  totalColsConst && *totalColsConst > 0) {
                int64_t colChunk =
                    (*totalColsConst + *colWorkersConst - 1) / *colWorkersConst;
                staticTaskBlockShape[1] = std::max<int64_t>(1, colChunk);
                refinedShape = true;
              }
            }

            if (refinedShape)
              plan.refinedTaskBlockShape = staticTaskBlockShape;
          }
        }

        /// Carry the owned task tile separately from the dependency window.
        /// Stencil read acquires widen `partition_sizes` when halo expansion is
        /// applied, so recovering the owner block shape later from those sizes
        /// loses the pre-halo chunk extent. Materialize the N-D owner tile
        /// here, from the pre-halo hint sizes, so DbPartitioning can harmonize
        /// read/write allocations on the same worker-local block plan.
        if (!plan.refinedTaskBlockShape &&
            !contract.spatial.ownerDims.empty()) {
          SmallVector<unsigned, 4> ownerDims;
          for (int64_t dim : contract.spatial.ownerDims) {
            if (dim >= 0 && static_cast<size_t>(dim) < elemSizes.size())
              ownerDims.push_back(static_cast<unsigned>(dim));
          }

          if (!ownerDims.empty()) {
            SmallVector<int64_t, 4> staticTaskBlockShape;
            staticTaskBlockShape.reserve(elemSizes.size());
            for (Value extent : elemSizes) {
              auto folded = ValueAnalysis::tryFoldConstantIndex(extent);
              if (!folded) {
                staticTaskBlockShape.clear();
                break;
              }
              staticTaskBlockShape.push_back(std::max<int64_t>(1, *folded));
            }

            SmallVector<Value, 4> ownerHintSizes;
            ownerHintSizes.push_back(input.acquireHintSize);
            ownerHintSizes.append(extraHintSizes.begin(), extraHintSizes.end());
            unsigned mappedOwnerDims =
                std::min<unsigned>(ownerDims.size(), ownerHintSizes.size());

            for (unsigned i = 0;
                 !staticTaskBlockShape.empty() && i < mappedOwnerDims; ++i) {
              auto folded =
                  ValueAnalysis::tryFoldConstantIndex(ownerHintSizes[i]);
              if (!folded) {
                staticTaskBlockShape.clear();
                break;
              }
              staticTaskBlockShape[ownerDims[i]] =
                  std::max<int64_t>(1, *folded);
            }

            if (!staticTaskBlockShape.empty())
              plan.refinedTaskBlockShape = staticTaskBlockShape;
          }
        }
      }
    }
  }

  plan.useStencilRewriter = applyStencilHalo;
  plan.rewriteInput.singleElement = isSingleElement;
  plan.rewriteInput.forceCoarse = forceCoarseRewrite;
  plan.rewriteInput.effectiveMode = input.effectiveMode;
  plan.rewriteInput.preserveParentDepRange = preserveParentDepRange;
  plan.rewriteInput.stencilExtent = stencilExtent;
  plan.rewriteInput.extraOffsets = std::move(extraOffsets);
  plan.rewriteInput.extraSizes = std::move(extraSizes);
  plan.rewriteInput.extraHintSizes = std::move(extraHintSizes);
  plan.rewriteInput.dimensionExtents = std::move(dimensionExtents);
  if (haloMinOffsets)
    plan.rewriteInput.haloMinOffsets.assign(haloMinOffsets->begin(),
                                            haloMinOffsets->end());
  if (haloMaxOffsets)
    plan.rewriteInput.haloMaxOffsets.assign(haloMaxOffsets->begin(),
                                            haloMaxOffsets->end());
  return plan;
}

DbAcquireOp mlir::arts::rewriteAcquire(AcquireRewriteInput &in,
                                       bool applyStencilHalo) {
  Value zero = in.AC->createIndexConstant(0, in.loc);
  Value one = in.AC->createIndexConstant(1, in.loc);
  bool shouldApplyHalo =
      applyStencilHalo && in.haloMinOffsets.size() == in.haloMaxOffsets.size();
  if (!shouldApplyHalo)
    shouldApplyHalo = in.effectiveMode == ArtsMode::in &&
                      !in.haloMinOffsets.empty() &&
                      in.haloMinOffsets.size() == in.haloMaxOffsets.size();

  if (in.singleElement || in.forceCoarse) {
    /// Intentionally drop any worker-local block hints here. For single-node
    /// Seidel-like inout updates with cross-element self-reads, keeping those
    /// hints lets DbPartitioning recover an unsafe block/stencil layout later.
    auto coarseAcquire = in.AC->create<DbAcquireOp>(
        in.loc, in.effectiveMode, in.rootGuid, in.rootPtr,
        in.parentAcquire.getPtr().getType(), PartitionMode::coarse,
        /*indices=*/SmallVector<Value>{},
        /*offsets=*/SmallVector<Value>{zero},
        /*sizes=*/SmallVector<Value>{one},
        /*partition_indices=*/SmallVector<Value>{},
        /*partition_offsets=*/SmallVector<Value>{},
        /*partition_sizes=*/SmallVector<Value>{});
    transferContract(in.parentAcquire.getOperation(),
                     coarseAcquire.getOperation(), in.parentAcquire.getPtr(),
                     coarseAcquire.getPtr(), in.AC->getBuilder(), in.loc);
    if (in.parentAcquire.getPreserveAccessMode())
      coarseAcquire.setPreserveAccessMode();
    if (in.parentAcquire.getPreserveDepEdge())
      coarseAcquire.setPreserveDepEdge();
    return coarseAcquire;
  }

  Value workerSize = in.acquireSize;
  Value workerHintSize = in.acquireHintSize;
  if (!in.stepIsUnit) {
    workerSize = in.AC->create<arith::MulIOp>(in.loc, workerSize, in.step);
    workerHintSize =
        in.AC->create<arith::MulIOp>(in.loc, workerHintSize, in.step);
  }

  SmallVector<Value> workerOffsets = {in.acquireOffset};
  SmallVector<Value> workerSizes = {workerSize};
  SmallVector<Value> workerHintSizes = {workerHintSize};
  for (unsigned i = 0; i < in.extraOffsets.size(); ++i) {
    Value extraOffset = in.extraOffsets[i];
    Value extraSize = i < in.extraSizes.size() ? in.extraSizes[i] : one;
    Value extraHint =
        i < in.extraHintSizes.size() ? in.extraHintSizes[i] : extraSize;
    workerOffsets.push_back(extraOffset ? extraOffset : zero);
    workerSizes.push_back(extraSize ? extraSize : one);
    workerHintSizes.push_back(extraHint ? extraHint : extraSize);
  }

  auto applyHaloToDim = [&](unsigned dim, int64_t minOffset, int64_t maxOffset,
                            Value extent) {
    if (dim >= workerOffsets.size() || !extent)
      return;

    Value start = workerOffsets[dim];
    Value size = workerSizes[dim];
    Value end = in.AC->create<arith::AddIOp>(in.loc, start, size);

    Value haloStart = start;
    if (minOffset < 0) {
      Value leftHalo = in.AC->createIndexConstant(-minOffset, in.loc);
      Value canShiftLeft = in.AC->create<arith::CmpIOp>(
          in.loc, arith::CmpIPredicate::uge, start, leftHalo);
      Value shiftedStart =
          in.AC->create<arith::SubIOp>(in.loc, start, leftHalo);
      haloStart = in.AC->create<arith::SelectOp>(in.loc, canShiftLeft,
                                                 shiftedStart, zero);
    } else if (minOffset > 0) {
      Value shift = in.AC->createIndexConstant(minOffset, in.loc);
      haloStart = in.AC->create<arith::AddIOp>(in.loc, start, shift);
    }

    Value haloEnd = end;
    if (maxOffset > 0) {
      Value rightHalo = in.AC->createIndexConstant(maxOffset, in.loc);
      Value endPlusHalo = in.AC->create<arith::AddIOp>(in.loc, end, rightHalo);
      haloEnd = in.AC->create<arith::MinUIOp>(in.loc, endPlusHalo, extent);
    } else if (maxOffset < 0) {
      Value shrink = in.AC->createIndexConstant(-maxOffset, in.loc);
      Value canShrink = in.AC->create<arith::CmpIOp>(
          in.loc, arith::CmpIPredicate::uge, end, shrink);
      Value shrunkenEnd = in.AC->create<arith::SubIOp>(in.loc, end, shrink);
      haloEnd =
          in.AC->create<arith::SelectOp>(in.loc, canShrink, shrunkenEnd, zero);
    }

    Value haloEndAboveStart = in.AC->create<arith::CmpIOp>(
        in.loc, arith::CmpIPredicate::uge, haloEnd, haloStart);
    Value rawHaloSize =
        in.AC->create<arith::SubIOp>(in.loc, haloEnd, haloStart);
    Value haloSize = in.AC->create<arith::SelectOp>(in.loc, haloEndAboveStart,
                                                    rawHaloSize, zero);
    workerOffsets[dim] = haloStart;
    workerSizes[dim] = haloSize;
    workerHintSizes[dim] = haloSize;
  };

  if (shouldApplyHalo) {
    if (!in.dimensionExtents.empty() && !in.haloMinOffsets.empty() &&
        !in.haloMaxOffsets.empty()) {
      unsigned haloRank = std::min<unsigned>(
          workerOffsets.size(),
          std::min<unsigned>(in.dimensionExtents.size(),
                             std::min<unsigned>(in.haloMinOffsets.size(),
                                                in.haloMaxOffsets.size())));
      for (unsigned dim = 0; dim < haloRank; ++dim)
        applyHaloToDim(dim, in.haloMinOffsets[dim], in.haloMaxOffsets[dim],
                       in.dimensionExtents[dim]);
    } else if (in.stencilExtent) {
      applyHaloToDim(/*dim=*/0, /*minOffset=*/-1, /*maxOffset=*/1,
                     in.stencilExtent);
    }
  }

  SmallVector<Value> partitionOffsets(workerOffsets.begin(),
                                      workerOffsets.end());
  SmallVector<Value> partitionHintSizes(workerHintSizes.begin(),
                                        workerHintSizes.end());

  /// Keep dependency slices in element space at this stage, but preserve the
  /// full DB-space rank already materialized for this worker-local acquire.
  /// Multi-dimensional block/stencil rewrites may clamp one dimension to a
  /// singleton tile and leave later `arts.db_ref` ops to select that local
  /// slot. Collapsing offsets/sizes back to one dimension here violates the
  /// DbAcquire layout contract for N-D sources and breaks downstream
  /// verification on wavefront-like kernels.
  SmallVector<Value> dependencyOffsets(workerOffsets.begin(),
                                       workerOffsets.end());
  SmallVector<Value> dependencySizes(workerHintSizes.begin(),
                                     workerHintSizes.end());

  /// Generic read-only acquires may still preserve the parent dependency
  /// range. Pattern-backed block/stencil contracts must keep the worker-local
  /// dependency slice authoritative so later lowering does not widen them back
  /// to full-range.
  bool useParentDepRange = in.preserveParentDepRange && !shouldApplyHalo &&
                           !in.parentAcquire.getOffsets().empty() &&
                           !in.parentAcquire.getSizes().empty();
  if (useParentDepRange) {
    dependencyOffsets.assign(in.parentAcquire.getOffsets().begin(),
                             in.parentAcquire.getOffsets().end());
    dependencySizes.assign(in.parentAcquire.getSizes().begin(),
                           in.parentAcquire.getSizes().end());
  }

  auto blockAcquire = in.AC->create<DbAcquireOp>(
      in.loc, in.effectiveMode, in.rootGuid, in.rootPtr,
      in.parentAcquire.getPtr().getType(), PartitionMode::block,
      /*indices=*/
      SmallVector<Value>(in.parentAcquire.getIndices().begin(),
                         in.parentAcquire.getIndices().end()),
      /*offsets=*/dependencyOffsets,
      /*sizes=*/dependencySizes,
      /*partition_indices=*/SmallVector<Value>{},
      /*partition_offsets=*/partitionOffsets,
      /*partition_sizes=*/partitionHintSizes,
      /*bounds_valid=*/in.parentAcquire.getBoundsValid(),
      /*element_offsets=*/SmallVector<Value>{},
      /*element_sizes=*/SmallVector<Value>{});
  transferContract(in.parentAcquire.getOperation(), blockAcquire.getOperation(),
                   in.parentAcquire.getPtr(), blockAcquire.getPtr(),
                   in.AC->getBuilder(), in.loc);
  /// Task-local block/stencil rewrites define the authoritative dependency
  /// window for the lowered worker EDT, even when later allocation planning
  /// keeps the backing DB layout coarse.
  blockAcquire.setPreserveDepEdge();
  if (in.parentAcquire.getPreserveAccessMode())
    blockAcquire.setPreserveAccessMode();
  if (in.parentAcquire.getPreserveDepEdge())
    blockAcquire.setPreserveDepEdge();
  return blockAcquire;
}

void mlir::arts::applyTaskAcquireSlicePlan(TaskAcquireSlicePlanInput input) {
  if (!input.AC || !input.taskAcquire)
    return;

  const bool hasReadHaloContract =
      hasConcreteReadHaloContract(input.contract, input.effectiveMode);
  const bool applyStencilHalo =
      shouldApplyStencilHalo(input.contract, input.taskAcquire);
  const bool usePartitionSliceAsDepWindow =
      shouldUsePartitionSliceAsDepWindow(input.contract, input.taskAcquire);
  bool shouldUseStencilWindow = input.usesStencilHalo || applyStencilHalo ||
                                usePartitionSliceAsDepWindow ||
                                hasReadHaloContract;
  bool shouldMaterializeElementSlice = shouldMaterializeTaskElementSlice(input);
  bool shouldUpdateBlockWindow = true;

  /// Read-only acquires already get their task-local window from
  /// rewriteAcquire(). This helper primarily updates DB-space slice metadata
  /// for write-capable acquires, but may also synthesize missing partition
  /// metadata for block/stencil read acquires to avoid conservative widening
  /// back to full-range later.
  if (input.effectiveMode == ArtsMode::in && !shouldUseStencilWindow) {
    PartitionMode mode =
        input.taskAcquire.getPartitionMode().value_or(PartitionMode::coarse);
    bool needsPartitionMetadata =
        usesBlockLayout(mode) &&
        (input.taskAcquire.getPartitionOffsets().empty() ||
         input.taskAcquire.getPartitionSizes().empty());
    const bool preserveParentDepRange =
        shouldPreserveParentDepRange(input.contract, input.taskAcquire);
    shouldUpdateBlockWindow = needsPartitionMetadata || !preserveParentDepRange;
  }
  size_t sourceRank =
      DbUtils::getSizesFromDb(input.taskAcquire.getSourcePtr()).size();
  if (sourceRank > 1 || input.taskAcquire.getOffsets().size() > 1 ||
      input.taskAcquire.getPartitionOffsets().size() > 1) {
    /// computeTaskAcquireWindow() intentionally models leading-dimension block
    /// space only. Reapplying it to N-D sources would silently collapse
    /// already-correct worker-local slices back to 1-D metadata.
    shouldUpdateBlockWindow = false;
  }
  if (!shouldUpdateBlockWindow && !shouldMaterializeElementSlice) {
    input.taskAcquire.getElementOffsetsMutable().clear();
    input.taskAcquire.getElementSizesMutable().clear();
    return;
  }

  OpBuilder::InsertionGuard guard(input.AC->getBuilder());
  input.AC->setInsertionPoint(input.taskAcquire);

  if (shouldUpdateBlockWindow) {
    auto window = computeTaskAcquireWindow(input);
    if (!window)
      return;

    /// Keep partition hint ranges worker-local for two-level write acquires.
    /// DbPartitioning consults partition_* hints; if these remain node-wide
    /// while offsets/sizes are worker-local, later rewrites may widen the
    /// write slice back to node scope.
    if (input.distributionKind == DistributionKind::TwoLevel &&
        input.effectiveMode != ArtsMode::in) {
      if (!input.taskAcquire.getPartitionOffsets().empty())
        input.taskAcquire.getPartitionOffsetsMutable().assign(
            SmallVector<Value>{window->elementOffset});
      if (!input.taskAcquire.getPartitionSizes().empty())
        input.taskAcquire.getPartitionSizesMutable().assign(
            SmallVector<Value>{window->elementSize});
    }
    if (input.effectiveMode == ArtsMode::in) {
      if (input.taskAcquire.getPartitionOffsets().empty())
        input.taskAcquire.getPartitionOffsetsMutable().assign(
            SmallVector<Value>{window->elementOffset});
      if (input.taskAcquire.getPartitionSizes().empty())
        input.taskAcquire.getPartitionSizesMutable().assign(
            SmallVector<Value>{window->elementSize});
    }
    input.taskAcquire.getOffsetsMutable().assign(
        SmallVector<Value>{window->dbOffset});
    input.taskAcquire.getSizesMutable().assign(
        SmallVector<Value>{window->dbSize});
  }

  if (shouldMaterializeElementSlice)
    if (auto slice = computeTaskElementSlice(input)) {
      input.taskAcquire.getElementOffsetsMutable().assign(slice->offsets);
      input.taskAcquire.getElementSizesMutable().assign(slice->sizes);
    }
  if (!shouldMaterializeElementSlice) {
    input.taskAcquire.getElementOffsetsMutable().clear();
    input.taskAcquire.getElementSizesMutable().clear();
  }
}

void mlir::arts::applyTaskAcquireContractMetadata(
    Operation *semanticSourceOp, DbAcquireOp taskAcquire,
    const LoweringContractInfo &contract,
    std::optional<SmallVector<int64_t, 4>> refinedTaskBlockShape,
    OpBuilder &builder, Location loc) {
  if (!semanticSourceOp || !taskAcquire)
    return;

  /// Task-local acquires should inherit the loop's semantic contract when
  /// their parent acquire does not already carry it. This keeps downstream
  /// analyses and lowering mode-agnostic: they can reason from the acquire's
  /// own contract instead of pattern-specific branches in ForLowering.
  inheritSemanticContractAttrs(semanticSourceOp, taskAcquire.getOperation());

  if (!getPartitioningHint(taskAcquire.getOperation()))
    if (auto hint = getPartitioningHint(semanticSourceOp)) {
      auto mode = taskAcquire.getPartitionMode();
      if (mode && usesBlockLayout(*mode)) {
        /// DbPartitioning resolves allocation block plans from task acquires
        /// after ForLowering. Carry the loop-selected owner-span hint onto the
        /// rewritten acquire so lowering and later DB planning use the same
        /// block granularity.
        setPartitioningHint(taskAcquire.getOperation(), *hint);
      }
    }

  std::optional<LoweringContractInfo> currentContract =
      getLoweringContract(taskAcquire.getPtr());
  if (auto opContract =
          getLoweringContract(taskAcquire.getOperation(), builder, loc)) {
    if (!currentContract) {
      currentContract = *opContract;
    } else {
      combineContracts(*currentContract, *opContract);
    }
  }

  SmallVector<int64_t, 4> taskOwnerDims;
  if (currentContract && !currentContract->spatial.ownerDims.empty())
    taskOwnerDims.assign(currentContract->spatial.ownerDims.begin(),
                         currentContract->spatial.ownerDims.end());
  else
    taskOwnerDims.assign(contract.spatial.ownerDims.begin(),
                         contract.spatial.ownerDims.end());

  unsigned supportedOwnerRank = 0;
  if (!taskAcquire.getPartitionOffsets().empty() &&
      !taskAcquire.getPartitionSizes().empty())
    supportedOwnerRank =
        std::min<unsigned>(taskAcquire.getPartitionOffsets().size(),
                           taskAcquire.getPartitionSizes().size());
  else if (!taskAcquire.getOffsets().empty() && !taskAcquire.getSizes().empty())
    supportedOwnerRank = std::min<unsigned>(taskAcquire.getOffsets().size(),
                                            taskAcquire.getSizes().size());

  size_t sourceRank =
      DbUtils::getSizesFromDb(taskAcquire.getSourcePtr()).size();
  if (sourceRank != 0)
    supportedOwnerRank =
        supportedOwnerRank == 0
            ? static_cast<unsigned>(sourceRank)
            : std::min<unsigned>(supportedOwnerRank, sourceRank);

  if (supportedOwnerRank != 0 && !taskOwnerDims.empty() &&
      taskOwnerDims.size() > supportedOwnerRank) {
    SmallVector<int64_t, 4> clampedOwnerDims;
    clampedOwnerDims.reserve(supportedOwnerRank);
    for (int64_t dim : taskOwnerDims) {
      if (dim < 0)
        continue;
      clampedOwnerDims.push_back(dim);
      if (clampedOwnerDims.size() == supportedOwnerRank)
        break;
    }
    if (clampedOwnerDims.empty())
      for (unsigned dim = 0; dim < supportedOwnerRank; ++dim)
        clampedOwnerDims.push_back(static_cast<int64_t>(dim));
    taskOwnerDims = std::move(clampedOwnerDims);
  }

  auto clampRankedStencilVector = [&](SmallVector<int64_t, 4> values) {
    if (supportedOwnerRank != 0 && values.size() > supportedOwnerRank)
      values.resize(supportedOwnerRank);
    return values;
  };

  SmallVector<int64_t, 4> taskHaloMinOffsets;
  SmallVector<int64_t, 4> taskHaloMaxOffsets;
  if (currentContract) {
    if (auto halo = projectHaloWindow(*currentContract)) {
      taskHaloMinOffsets = std::move(halo->first);
      taskHaloMaxOffsets = std::move(halo->second);
    }
  }
  if (taskHaloMinOffsets.empty())
    if (auto haloMins = contract.getStaticMinOffsets())
      taskHaloMinOffsets.assign(haloMins->begin(), haloMins->end());
  if (taskHaloMaxOffsets.empty())
    if (auto haloMaxs = contract.getStaticMaxOffsets())
      taskHaloMaxOffsets.assign(haloMaxs->begin(), haloMaxs->end());
  taskHaloMinOffsets = clampRankedStencilVector(std::move(taskHaloMinOffsets));
  taskHaloMaxOffsets = clampRankedStencilVector(std::move(taskHaloMaxOffsets));

  SmallVector<int64_t, 4> taskWriteFootprint;
  if (currentContract && !currentContract->spatial.writeFootprint.empty()) {
    bool allConstant = true;
    for (Value value : currentContract->spatial.writeFootprint) {
      auto folded = ValueAnalysis::tryFoldConstantIndex(value);
      if (!folded) {
        allConstant = false;
        break;
      }
      taskWriteFootprint.push_back(*folded);
    }
    if (!allConstant)
      taskWriteFootprint.clear();
  }
  taskWriteFootprint = clampRankedStencilVector(std::move(taskWriteFootprint));

  auto chunkMode = taskAcquire.getPartitionMode();
  const bool applyStencilHalo = shouldApplyStencilHalo(contract, taskAcquire);
  const bool usePartitionSliceAsDepWindow =
      shouldUsePartitionSliceAsDepWindow(contract, taskAcquire);
  bool shouldMarkStencilCenter =
      usePartitionSliceAsDepWindow || applyStencilHalo ||
      (chunkMode && *chunkMode == PartitionMode::stencil);
  std::optional<int64_t> taskCenterOffset;
  if (currentContract)
    taskCenterOffset = currentContract->spatial.centerOffset;
  if (shouldMarkStencilCenter && !taskCenterOffset) {
    taskCenterOffset = 1;
    if (!taskHaloMinOffsets.empty())
      taskCenterOffset = std::max<int64_t>(0, -taskHaloMinOffsets.front());
  }

  /// Task acquires must carry the full stencil halo contract before
  /// pre-lowering completes. ConvertArtsToLLVM reconstructs per-slot RO halo
  /// slices from these attrs after block expansion, so dropping min/max
  /// offsets here widens wavefront-like dependencies back to whole-DB edges.
  if (!taskOwnerDims.empty())
    /// Keep the full logical stencil block shape, but clamp owner_dims to the
    /// number of DB dimensions this worker-local acquire actually proves safe
    /// to partition. Row-owned wavefront tasks still operate on 2-D tiles, yet
    /// their late ESD inference must reason in the physical 1-D owner space.
    setStencilOwnerDims(taskAcquire.getOperation(), taskOwnerDims);
  if (!taskHaloMinOffsets.empty())
    setStencilMinOffsets(taskAcquire.getOperation(), taskHaloMinOffsets);
  if (!taskHaloMaxOffsets.empty())
    setStencilMaxOffsets(taskAcquire.getOperation(), taskHaloMaxOffsets);
  if (!taskWriteFootprint.empty())
    setStencilWriteFootprint(taskAcquire.getOperation(), taskWriteFootprint);

  std::optional<SmallVector<int64_t, 4>> taskBlockShape = refinedTaskBlockShape;
  if (!taskBlockShape && currentContract)
    taskBlockShape = currentContract->getStaticBlockShape();

  if (!taskBlockShape) {
    SmallVector<unsigned, 4> ownerDims;
    for (int64_t dim : taskOwnerDims) {
      if (dim >= 0)
        ownerDims.push_back(static_cast<unsigned>(dim));
    }

    if (!ownerDims.empty())
      if (auto alloc = dyn_cast_or_null<DbAllocOp>(
              DbUtils::getUnderlyingDbAlloc(taskAcquire.getSourcePtr()))) {
        SmallVector<int64_t, 4> inferredShape;
        inferredShape.reserve(alloc.getElementSizes().size());
        for (Value extent : alloc.getElementSizes()) {
          auto folded = ValueAnalysis::tryFoldConstantIndex(extent);
          if (!folded) {
            inferredShape.clear();
            break;
          }
          inferredShape.push_back(std::max<int64_t>(1, *folded));
        }

        auto partSizes = taskAcquire.getPartitionSizes();
        unsigned mappedOwnerDims =
            std::min<unsigned>(partSizes.size(), ownerDims.size());
        for (unsigned i = 0; !inferredShape.empty() && i < mappedOwnerDims;
             ++i) {
          if (ownerDims[i] >= inferredShape.size()) {
            inferredShape.clear();
            break;
          }
          auto folded = ValueAnalysis::tryFoldConstantIndex(partSizes[i]);
          if (!folded) {
            inferredShape.clear();
            break;
          }
          inferredShape[ownerDims[i]] = std::max<int64_t>(1, *folded);
        }

        if (!inferredShape.empty())
          taskBlockShape = inferredShape;
      }
  }

  std::optional<SmallVector<int64_t, 4>> contractTaskBlockShape;
  if (taskBlockShape) {
    /// Tiling-2D workers refine the inherited stencil owner shape into a
    /// worker-local owner tile. Reflect that narrower static tile on the
    /// acquire before DbPartitioning runs so the N-D block planner can align
    /// allocation blocks with the actual 2-D worker grid instead of widening
    /// back to the parent stencil tile.
    setStencilBlockShape(taskAcquire.getOperation(), *taskBlockShape);

    SmallVector<int64_t, 4> clampedTaskBlockShape = *taskBlockShape;
    clampedTaskBlockShape =
        clampRankedStencilVector(std::move(clampedTaskBlockShape));
    contractTaskBlockShape = std::move(clampedTaskBlockShape);
  }

  if (currentContract) {
    /// Keep the rewritten pointer contract authoritative even when no static
    /// worker block shape is inferable. Halo windows and owner dims must still
    /// survive on the pointer contract so later DB planning never needs to
    /// rediscover them from graph-only stencil facts.

    LoweringContractInfo updated = *currentContract;
    if (!taskOwnerDims.empty())
      updated.spatial.ownerDims = taskOwnerDims;
    auto clampDynamicRank = [&](auto &values) {
      if (supportedOwnerRank != 0 && values.size() > supportedOwnerRank)
        values.resize(supportedOwnerRank);
    };
    auto clampStaticRank = [&](auto &values) {
      if (supportedOwnerRank != 0 && values.size() > supportedOwnerRank)
        values.resize(supportedOwnerRank);
    };

    clampDynamicRank(updated.spatial.blockShape);
    clampDynamicRank(updated.spatial.minOffsets);
    clampDynamicRank(updated.spatial.maxOffsets);
    clampDynamicRank(updated.spatial.writeFootprint);
    clampStaticRank(updated.spatial.staticBlockShape);
    clampStaticRank(updated.spatial.staticMinOffsets);
    clampStaticRank(updated.spatial.staticMaxOffsets);

    if (contractTaskBlockShape)
      updated.spatial.staticBlockShape = *contractTaskBlockShape;
    if (!taskHaloMinOffsets.empty()) {
      updated.spatial.minOffsets.clear();
      updated.spatial.staticMinOffsets = taskHaloMinOffsets;
    }
    if (!taskHaloMaxOffsets.empty()) {
      updated.spatial.maxOffsets.clear();
      updated.spatial.staticMaxOffsets = taskHaloMaxOffsets;
    }
    if (!taskWriteFootprint.empty()) {
      updated.spatial.writeFootprint.clear();
      for (int64_t value : taskWriteFootprint)
        updated.spatial.writeFootprint.push_back(
            arith::ConstantIndexOp::create(builder, loc, value));
    }
    if (taskCenterOffset)
      updated.spatial.centerOffset = taskCenterOffset;
    if (contractTaskBlockShape)
      updated.spatial.blockShape.clear();
    /// Worker-local acquires are freshly rewritten slices. Even when the
    /// parent acquire carried post-DB refinement markers, the new worker-local
    /// contract still needs a fresh DT-1/DT-2 pass to recover properties such
    /// as narrowableDep on the rewritten slice.
    updated.analysis.postDbRefined = false;
    upsertLoweringContract(builder, loc, taskAcquire.getPtr(), updated);
  }
}
