///==========================================================================///
/// File: DbLayoutPlanUtils.cpp
///==========================================================================///

#include "arts/dialect/core/Transforms/db/DbLayoutPlanUtils.h"

#include "arts/utils/OperationAttributes.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::arts;

namespace {

static Value createIndexConstant(OpBuilder &builder, Location loc,
                                 int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

static Value ceilDivPositiveIndex(OpBuilder &builder, Location loc, Value value,
                                  Value divisor) {
  Value one = createIndexConstant(builder, loc, 1);
  divisor = arith::MaxUIOp::create(builder, loc, divisor, one);
  Value adjusted = arith::AddIOp::create(
      builder, loc, value, arith::SubIOp::create(builder, loc, divisor, one));
  return arith::DivUIOp::create(builder, loc, adjusted, divisor);
}

static SmallVector<Value, 4>
materializeIndexConstants(OpBuilder &builder, Location loc,
                          ArrayRef<int64_t> values) {
  SmallVector<Value, 4> result;
  result.reserve(values.size());
  for (int64_t value : values)
    result.push_back(createIndexConstant(builder, loc, value));
  return result;
}

static std::optional<unsigned> findOwnerPosition(ArrayRef<int64_t> ownerDims,
                                                 unsigned physicalDim) {
  for (auto [idx, dim] : llvm::enumerate(ownerDims))
    if (dim >= 0 && static_cast<unsigned>(dim) == physicalDim)
      return static_cast<unsigned>(idx);
  return std::nullopt;
}

static bool ownerDimsContain(ArrayRef<int64_t> ownerDims, unsigned dim) {
  for (int64_t ownerDim : ownerDims)
    if (ownerDim >= 0 && static_cast<unsigned>(ownerDim) == dim)
      return true;
  return false;
}

static std::optional<int64_t>
getBlockShapeForOwnerDim(ArrayRef<int64_t> blockShape,
                         ArrayRef<int64_t> ownerDims, unsigned ownerSlot) {
  if (blockShape.empty())
    return std::nullopt;
  if (blockShape.size() == ownerDims.size()) {
    if (ownerSlot < blockShape.size() && blockShape[ownerSlot] > 0)
      return blockShape[ownerSlot];
    return std::nullopt;
  }
  if (ownerSlot >= ownerDims.size())
    return std::nullopt;
  int64_t ownerDim = ownerDims[ownerSlot];
  if (ownerDim < 0 || static_cast<size_t>(ownerDim) >= blockShape.size())
    return std::nullopt;
  if (blockShape[static_cast<size_t>(ownerDim)] <= 0)
    return std::nullopt;
  return blockShape[static_cast<size_t>(ownerDim)];
}

static Value blockSizeForPhysicalDim(const DbRewritePlan &plan,
                                     unsigned physicalDim) {
  for (auto [idx, dim] : llvm::enumerate(plan.partitionedDims))
    if (dim == physicalDim && idx < plan.blockSizes.size())
      return plan.blockSizes[idx];
  return Value();
}

static void deriveStencilInfo(Operation *source, DbRewritePlan &plan) {
  if (!source || plan.partitionedDims.empty())
    return;

  unsigned primaryOwnerDim = plan.partitionedDims.front();
  std::optional<SmallVector<int64_t, 4>> stencilOwnerDims =
      getStencilOwnerDims(source);
  std::optional<SmallVector<int64_t, 4>> minOffsets =
      getStencilMinOffsets(source);
  std::optional<SmallVector<int64_t, 4>> maxOffsets =
      getStencilMaxOffsets(source);

  if (stencilOwnerDims && minOffsets && maxOffsets) {
    std::optional<unsigned> ownerPos =
        findOwnerPosition(*stencilOwnerDims, primaryOwnerDim);
    if (ownerPos && *ownerPos < minOffsets->size() &&
        *ownerPos < maxOffsets->size()) {
      StencilInfo info;
      info.haloLeft = std::max<int64_t>(0, -(*minOffsets)[*ownerPos]);
      info.haloRight = std::max<int64_t>(0, (*maxOffsets)[*ownerPos]);
      if (info.hasHalo())
        plan.stencilInfo = info;
      return;
    }
  }

  if (std::optional<SmallVector<int64_t, 4>> haloShape =
          readI64ArrayAttr(getPlanHaloShapeAttr(source))) {
    if (!haloShape->empty() && haloShape->front() > 0) {
      StencilInfo info;
      info.haloLeft = haloShape->front();
      info.haloRight = haloShape->front();
      plan.stencilInfo = info;
    }
  }
}

} // namespace

bool mlir::arts::hasPhysicalDbLayoutPlan(Operation *op) {
  if (!op)
    return false;
  return getPlanOwnerDimsAttr(op) && getPlanPhysicalBlockShapeAttr(op);
}

bool mlir::arts::hasReadOnlySourceLayoutMismatch(
    DbAllocOp sourceAlloc, Operation *taskPlanSource,
    const LoweringContractInfo &contract) {
  if (!sourceAlloc || contract.spatial.ownerDims.empty())
    return false;
  if (!(contract.isStencilFamily() || contract.usesStencilDistribution() ||
        contract.supportsBlockHalo()))
    return false;

  auto sourceOwnerDims =
      readI64ArrayAttr(getPlanOwnerDimsAttr(sourceAlloc.getOperation()));
  auto sourceBlockShape =
      readI64ArrayAttr(getPlanPhysicalBlockShapeAttr(sourceAlloc.getOperation()));
  std::optional<SmallVector<int64_t, 4>> taskOwnerDims =
      taskPlanSource ? readI64ArrayAttr(getPlanOwnerDimsAttr(taskPlanSource))
                     : std::nullopt;
  std::optional<SmallVector<int64_t, 4>> taskPlanBlockShape =
      taskPlanSource
          ? readI64ArrayAttr(getPlanPhysicalBlockShapeAttr(taskPlanSource))
          : std::nullopt;
  auto contractBlockShape = contract.getStaticBlockShape();

  ArrayRef<int64_t> effectiveTaskOwnerDims =
      (taskOwnerDims && !taskOwnerDims->empty()) ? ArrayRef<int64_t>(*taskOwnerDims)
                                                 : contract.spatial.ownerDims;
  ArrayRef<int64_t> effectiveTaskBlockShape =
      (taskPlanBlockShape && !taskPlanBlockShape->empty())
          ? ArrayRef<int64_t>(*taskPlanBlockShape)
          : (contractBlockShape ? ArrayRef<int64_t>(*contractBlockShape)
                                : ArrayRef<int64_t>{});
  if (!sourceOwnerDims || sourceOwnerDims->empty() || !sourceBlockShape ||
      sourceBlockShape->empty() || effectiveTaskBlockShape.empty())
    return false;

  for (auto [ownerSlot, ownerDim] : llvm::enumerate(effectiveTaskOwnerDims)) {
    if (ownerDim < 0)
      return true;
    if (!ownerDimsContain(*sourceOwnerDims, static_cast<unsigned>(ownerDim)))
      return true;

    auto sourceBlock =
        getBlockShapeForOwnerDim(*sourceBlockShape, *sourceOwnerDims, ownerSlot);
    auto taskBlock = getBlockShapeForOwnerDim(
        effectiveTaskBlockShape, effectiveTaskOwnerDims, ownerSlot);
    if (sourceBlock && taskBlock && *sourceBlock != *taskBlock)
      return true;
  }

  return false;
}

std::optional<SmallVector<int64_t, 4>>
mlir::arts::getSourceOwnerBlockShape(DbAllocOp sourceAlloc,
                                     const LoweringContractInfo &contract) {
  if (!sourceAlloc || contract.spatial.ownerDims.empty())
    return std::nullopt;

  auto sourceOwnerDims =
      readI64ArrayAttr(getPlanOwnerDimsAttr(sourceAlloc.getOperation()));
  auto sourceBlockShape =
      readI64ArrayAttr(getPlanPhysicalBlockShapeAttr(sourceAlloc.getOperation()));
  if (!sourceOwnerDims || sourceOwnerDims->empty() || !sourceBlockShape ||
      sourceBlockShape->empty())
    return std::nullopt;

  SmallVector<int64_t, 4> blockShape;
  blockShape.reserve(contract.spatial.ownerDims.size());
  for (auto [ownerSlot, ownerDim] : llvm::enumerate(contract.spatial.ownerDims)) {
    if (ownerDim < 0)
      return std::nullopt;
    if (!ownerDimsContain(*sourceOwnerDims, static_cast<unsigned>(ownerDim)))
      return std::nullopt;
    auto sourceBlock =
        getBlockShapeForOwnerDim(*sourceBlockShape, *sourceOwnerDims, ownerSlot);
    if (!sourceBlock)
      return std::nullopt;
    blockShape.push_back(*sourceBlock);
  }

  if (blockShape.empty())
    return std::nullopt;
  return blockShape;
}

FailureOr<DbRewritePlan>
mlir::arts::resolvePhysicalDbLayoutPlan(Operation *planSource,
                                        ValueRange elementSizes,
                                        OpBuilder &builder, Location loc) {
  if (!hasPhysicalDbLayoutPlan(planSource) || elementSizes.empty())
    return failure();

  std::optional<SmallVector<int64_t, 4>> ownerDims =
      readI64ArrayAttr(getPlanOwnerDimsAttr(planSource));
  std::optional<SmallVector<int64_t, 4>> blockShape =
      readI64ArrayAttr(getPlanPhysicalBlockShapeAttr(planSource));
  if (!ownerDims || ownerDims->empty() || !blockShape || blockShape->empty())
    return failure();

  const unsigned memrefRank = elementSizes.size();
  DbRewritePlan plan(PartitionMode::block);

  llvm::DenseSet<unsigned> seenDims;
  for (int64_t rawDim : *ownerDims) {
    if (rawDim < 0 || static_cast<uint64_t>(rawDim) >= memrefRank)
      return failure();
    unsigned dim = static_cast<unsigned>(rawDim);
    if (!seenDims.insert(dim).second)
      return failure();
    plan.partitionedDims.push_back(dim);
  }

  SmallVector<Value, 4> blockShapeValues =
      materializeIndexConstants(builder, loc, *blockShape);
  for (auto [ownerSlot, dim] : llvm::enumerate(plan.partitionedDims)) {
    Value blockSize;
    if (blockShapeValues.size() == memrefRank) {
      blockSize = blockShapeValues[dim];
    } else {
      if (ownerSlot < blockShapeValues.size())
        blockSize = blockShapeValues[ownerSlot];
    }
    if (!blockSize)
      return failure();
    auto folded = ValueAnalysis::tryFoldConstantIndex(blockSize);
    if (folded && *folded <= 0)
      return failure();
    plan.blockSizes.push_back(blockSize);
  }

  if (plan.blockSizes.size() != plan.partitionedDims.size())
    return failure();

  for (auto [idx, physicalDim] : llvm::enumerate(plan.partitionedDims)) {
    Value extent = elementSizes[physicalDim];
    Value blockSize = plan.blockSizes[idx];
    if (!extent || !blockSize)
      return failure();
    plan.outerSizes.push_back(
        ceilDivPositiveIndex(builder, loc, extent, blockSize));
  }

  plan.innerSizes.reserve(memrefRank);
  for (unsigned dim = 0; dim < memrefRank; ++dim) {
    if (Value blockSize = blockSizeForPhysicalDim(plan, dim)) {
      plan.innerSizes.push_back(blockSize);
      continue;
    }
    plan.innerSizes.push_back(elementSizes[dim]);
  }

  deriveStencilInfo(planSource, plan);
  if (!plan.isValid())
    return failure();
  return plan;
}
