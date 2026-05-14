///==========================================================================///
/// File: LoopAnalysis.cpp
/// Implementation of LoopAnalysis with LoopNode integration
///==========================================================================///

#include "arts/dialect/core/Analysis/loop/LoopAnalysis.h"
#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/graphs/db/DbNode.h"
#include "arts/utils/Debug.h"
#include "arts/utils/LoopUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include <algorithm>

ARTS_DEBUG_SETUP(loop_analysis);

using namespace mlir;
using namespace mlir::arts;

LoopAnalysis::LoopAnalysis(AnalysisManager &analysisManager)
    : ArtsAnalysis(analysisManager), module(analysisManager.getModule()) {}

void LoopAnalysis::invalidate() {
  loopNodes.clear();
  built = false;
}

void LoopAnalysis::ensureAnalyzed() {
  if (!built)
    run();
}

void LoopAnalysis::run() {
  ARTS_DEBUG_HEADER(LoopAnalysis);

  built = true;
  loopNodes.clear();
  module.walk([&](Operation *op) {
    if (!isLoopOperation(op))
      return;
    if (auto whileOp = dyn_cast<scf::WhileOp>(op))
      ARTS_DEBUG("Discovered scf.while at " << whileOp.getLoc());
    loopNodes[op] = std::make_unique<LoopNode>(op, this);
  });

  if (loopNodes.empty())
    ARTS_WARN("No loops found in module");
  else
    ARTS_INFO("Discovered " << loopNodes.size() << " loops");

  ARTS_DEBUG_FOOTER(LoopAnalysis);
}

///===----------------------------------------------------------------------===///
/// LoopNode Access (NEW - Metadata-integrated API)
///===----------------------------------------------------------------------===///

bool LoopAnalysis::isLoopOperation(Operation *op) const {
  return isa<LoopLikeOpInterface>(op);
}

LoopNode *LoopAnalysis::getOrCreateLoopNode(Operation *loopOp) {
  ensureAnalyzed();
  if (!isLoopOperation(loopOp))
    return nullptr;

  auto it = loopNodes.find(loopOp);
  if (it != loopNodes.end())
    return it->second.get();

  auto newNode = std::make_unique<LoopNode>(loopOp, this);
  LoopNode *ptr = newNode.get();
  loopNodes[loopOp] = std::move(newNode);
  return ptr;
}

LoopNode *LoopAnalysis::getLoopNode(Operation *loopOp) {
  ensureAnalyzed();
  auto it = loopNodes.find(loopOp);
  return it != loopNodes.end() ? it->second.get() : nullptr;
}

void LoopAnalysis::collectEnclosingLoops(
    Operation *op, SmallVectorImpl<LoopNode *> &enclosingLoops) {
  ensureAnalyzed();
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (isLoopOperation(p)) {
      if (LoopNode *node = getLoopNode(p)) {
        enclosingLoops.push_back(node);
      }
    }
  }
  /// Reverse to get outermost-to-innermost order
  std::reverse(enclosingLoops.begin(), enclosingLoops.end());
}

void LoopAnalysis::collectLoopsInOperation(Operation *op,
                                           SmallVectorImpl<LoopNode *> &loops) {
  ensureAnalyzed();
  op->walk([&](Operation *childOp) {
    if (isLoopOperation(childOp)) {
      if (LoopNode *node = getOrCreateLoopNode(childOp))
        loops.push_back(node);
    }
  });
}

void LoopAnalysis::collectForLoopsInOperation(
    Operation *op, SmallVectorImpl<LoopNode *> &loops) {
  ensureAnalyzed();
  op->walk([&](Operation *childOp) {
    if (isa<scf::ForOp>(childOp)) {
      if (LoopNode *node = getOrCreateLoopNode(childOp))
        loops.push_back(node);
    }
  });
}

std::optional<int64_t> LoopAnalysis::getStaticTripCount(Operation *loopOp) {
  ensureAnalyzed();
  if (!loopOp || !isLoopOperation(loopOp))
    return std::nullopt;

  if (auto affineFor = dyn_cast<affine::AffineForOp>(loopOp)) {
    if (affineFor.hasConstantBounds()) {
      int64_t lb = affineFor.getConstantLowerBound();
      int64_t ub = affineFor.getConstantUpperBound();
      int64_t step = affineFor.getStepAsInt();
      if (step > 0) {
        int64_t span = ub - lb;
        if (span <= 0)
          return 0;
        return (span + step - 1) / step;
      }
    }
    return std::nullopt;
  }
  return arts::getStaticTripCount(loopOp);
}

std::optional<int64_t>
LoopAnalysis::estimateStaticPerfectNestedWork(Operation *loopOp, int64_t cap) {
  ensureAnalyzed();
  if (!loopOp || !isLoopOperation(loopOp))
    return std::nullopt;
  if (LoopNode *loopNode = getOrCreateLoopNode(loopOp))
    return loopNode->estimateStaticPerfectNestedWork(cap);
  return std::nullopt;
}

bool LoopAnalysis::operationHasDistributedDbContract(Operation *op) {
  ensureAnalyzed();
  return getAnalysisManager().getDbAnalysis().operationHasDistributedDbContract(
      op);
}

bool LoopAnalysis::operationHasPeerInferredPartitionDims(Operation *op) {
  ensureAnalyzed();
  return getAnalysisManager()
      .getDbAnalysis()
      .operationHasPeerInferredPartitionDims(op);
}

template <typename LoopOpType>
void LoopAnalysis::collectLoopsInOperation(Operation *op,
                                           SmallVectorImpl<LoopNode *> &loops) {
  ensureAnalyzed();
  op->walk([&](Operation *childOp) {
    if (isa<LoopOpType>(childOp)) {
      if (LoopNode *node = getOrCreateLoopNode(childOp))
        loops.push_back(node);
    }
  });
}

template void LoopAnalysis::collectLoopsInOperation<scf::ForOp>(
    Operation *op, SmallVectorImpl<LoopNode *> &loops);
template void LoopAnalysis::collectLoopsInOperation<scf::WhileOp>(
    Operation *op, SmallVectorImpl<LoopNode *> &loops);

LoopNode *LoopAnalysis::findEnclosingLoopDrivenBy(Operation *op, Value idx) {
  ensureAnalyzed();
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (!isLoopOperation(parent))
      continue;
    if (LoopNode *node = getLoopNode(parent)) {
      if (node->dependsOnInductionVar(idx))
        return node;
    }
  }
  return nullptr;
}
