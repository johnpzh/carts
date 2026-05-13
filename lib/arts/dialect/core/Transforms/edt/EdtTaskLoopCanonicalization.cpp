///==========================================================================///
/// File: EdtTaskLoopCanonicalization.cpp
///==========================================================================///

#include "arts/dialect/core/Transforms/edt/EdtTaskLoopCanonicalization.h"
#include "arts/dialect/core/Analysis/heuristics/DistributionHeuristics.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_task_loop_canonicalization);

using namespace mlir;
using namespace mlir::arts;

namespace {

static bool valueDependsOn(Value value, Value root,
                           DenseMap<Value, bool> &memo) {
  if (!value || !root)
    return false;
  if (value == root)
    return true;
  auto it = memo.find(value);
  if (it != memo.end())
    return it->second;

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    memo[value] = false;
    return false;
  }

  for (Value operand : defOp->getOperands()) {
    if (valueDependsOn(operand, root, memo)) {
      memo[value] = true;
      return true;
    }
  }
  memo[value] = false;
  return false;
}

static bool isRank0StackSlot(Value value) {
  value = ValueAnalysis::stripMemrefViewOps(value);
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 0)
    return false;
  return value.getDefiningOp<memref::AllocaOp>() != nullptr;
}

static bool isSafeDuplicableTaskLocalPrefixOp(Operation *op) {
  if (!op || isa<scf::ForOp>(op))
    return false;
  if (isMemoryEffectFree(op))
    return true;
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return isRank0StackSlot(store.getMemref());
  return false;
}

static bool collectSingleNestedForWithPrefix(
    Block &block, scf::ForOp &nestedLoop,
    SmallVectorImpl<Operation *> &prefixOps) {
  nestedLoop = nullptr;
  prefixOps.clear();

  for (Operation &op : block.without_terminator()) {
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      if (nestedLoop)
        return false;
      nestedLoop = forOp;
      continue;
    }
    if (nestedLoop)
      return false;
    prefixOps.push_back(&op);
  }

  return nestedLoop != nullptr;
}

static bool valueDefinedInRegion(Value value, Region &region) {
  Operation *defOp = value.getDefiningOp();
  return defOp && region.isAncestor(defOp->getParentRegion());
}

static bool loopBoundsDefinedInRegion(scf::ForOp loop, Region &region) {
  return valueDefinedInRegion(loop.getLowerBound(), region) ||
         valueDefinedInRegion(loop.getUpperBound(), region) ||
         valueDefinedInRegion(loop.getStep(), region);
}

static SmallVector<Value> collectLoopSinkAccessIndices(Operation *op) {
  SmallVector<Value> indices;
  std::optional<DbUtils::MemoryAccessInfo> access =
      DbUtils::getMemoryAccessInfo(op);
  if (!access)
    return indices;

  SmallVector<Value, 4> subindices;
  Value current = ValueAnalysis::stripMemrefViewOps(access->memref);
  while (current) {
    auto subindex = current.getDefiningOp<polygeist::SubIndexOp>();
    if (!subindex)
      break;
    subindices.push_back(subindex.getIndex());
    current = ValueAnalysis::stripMemrefViewOps(subindex.getSource());
  }

  for (Value index : llvm::reverse(subindices))
    indices.push_back(index);

  bool terminalSubindexDeref =
      !subindices.empty() && access->indices.size() == 1;
  if (terminalSubindexDeref) {
    if (auto folded = ValueAnalysis::tryFoldConstantIndex(access->indices[0]))
      terminalSubindexDeref = *folded == 0;
    else
      terminalSubindexDeref = false;
  }

  if (!terminalSubindexDeref)
    indices.append(access->indices.begin(), access->indices.end());
  return indices;
}

static bool taskLoopIvIsContiguousMemrefDim(scf::ForOp iterLoop,
                                            Value globalIdx) {
  int64_t lastIndexHits = 0;
  int64_t nonLastIndexHits = 0;
  DenseMap<Value, bool> memo;

  iterLoop.walk([&](Operation *op) {
    SmallVector<Value> indices = collectLoopSinkAccessIndices(op);
    if (indices.empty())
      return;
    for (auto [idx, indexValue] : llvm::enumerate(indices)) {
      if (!valueDependsOn(indexValue, globalIdx, memo))
        continue;
      if (idx + 1 == indices.size())
        ++lastIndexHits;
      else
        ++nonLastIndexHits;
    }
  });

  return lastIndexHits > 0 && nonLastIndexHits == 0;
}

} // namespace

Operation *mlir::arts::sinkTaskLoopToContiguousInnerDim(
    scf::ForOp iterLoop, Value globalIdx, DistributionKind kind) {
  if (!iterLoop || !globalIdx)
    return nullptr;
  if (kind == DistributionKind::BlockCyclic ||
      kind == DistributionKind::Tiling2D)
    return nullptr;
  if (iterLoop.getNumResults() != 0)
    return nullptr;

  SmallVector<Operation *, 8> iterPrefixOps;
  scf::ForOp middleLoop;
  if (!collectSingleNestedForWithPrefix(*iterLoop.getBody(), middleLoop,
                                        iterPrefixOps))
    return nullptr;
  if (middleLoop.getNumResults() != 0 ||
      loopBoundsDefinedInRegion(middleLoop, iterLoop.getRegion()))
    return nullptr;

  SmallVector<Operation *, 8> middlePrefixOps;
  scf::ForOp innerLoop;
  if (!collectSingleNestedForWithPrefix(*middleLoop.getBody(), innerLoop,
                                        middlePrefixOps))
    return nullptr;
  if (innerLoop.getNumResults() != 0 ||
      loopBoundsDefinedInRegion(innerLoop, iterLoop.getRegion()) ||
      loopBoundsDefinedInRegion(innerLoop, middleLoop.getRegion()))
    return nullptr;

  for (Operation *op : iterPrefixOps)
    if (!isSafeDuplicableTaskLocalPrefixOp(op))
      return nullptr;
  for (Operation *op : middlePrefixOps)
    if (!isSafeDuplicableTaskLocalPrefixOp(op))
      return nullptr;

  if (!taskLoopIvIsContiguousMemrefDim(iterLoop, globalIdx))
    return nullptr;

  OpBuilder builder(iterLoop);
  Location loc = iterLoop.getLoc();
  scf::ForOp newMiddleLoop =
      scf::ForOp::create(builder, loc, middleLoop.getLowerBound(),
                         middleLoop.getUpperBound(), middleLoop.getStep());

  builder.setInsertionPointToStart(newMiddleLoop.getBody());
  scf::ForOp newInnerLoop =
      scf::ForOp::create(builder, loc, innerLoop.getLowerBound(),
                         innerLoop.getUpperBound(), innerLoop.getStep());

  builder.setInsertionPointToStart(newInnerLoop.getBody());
  scf::ForOp newIterLoop =
      scf::ForOp::create(builder, loc, iterLoop.getLowerBound(),
                         iterLoop.getUpperBound(), iterLoop.getStep());

  builder.setInsertionPointToStart(newIterLoop.getBody());
  IRMapping mapper;
  mapper.map(iterLoop.getInductionVar(), newIterLoop.getInductionVar());
  mapper.map(middleLoop.getInductionVar(), newMiddleLoop.getInductionVar());
  mapper.map(innerLoop.getInductionVar(), newInnerLoop.getInductionVar());

  for (Operation *op : iterPrefixOps)
    builder.clone(*op, mapper);
  for (Operation *op : middlePrefixOps)
    builder.clone(*op, mapper);
  for (Operation &op : innerLoop.getBody()->without_terminator())
    builder.clone(op, mapper);

  iterLoop->erase();
  ARTS_DEBUG("Sank task-local chunk loop to contiguous memref dimension");
  return newMiddleLoop.getOperation();
}
