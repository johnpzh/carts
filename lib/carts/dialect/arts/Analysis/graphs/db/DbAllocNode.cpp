///==========================================================================///
/// File: DbAllocNode.cpp
///
/// Implementation of DbAlloc node
///==========================================================================///

#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAliasAnalysis.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/MetadataEnums.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/dialect/arts/Utils/PartitionPredicates.h"
#include "carts/utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::arts;

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_alloc_node);

namespace {

template <typename NodeT>
static void collectAcquireNodesRecursive(NodeT *root,
                                         SmallVectorImpl<NodeT *> &out) {
  if (!root)
    return;
  out.push_back(root);
  root->forEachChildNode([&](NodeBase *child) {
    auto *nested = dyn_cast<NodeT>(child);
    if (!nested)
      return;
    collectAcquireNodesRecursive(nested, out);
  });
}

template <typename NodeT>
static SmallVector<NodeT *, 16>
collectAllAcquireNodes(const SmallVector<std::unique_ptr<DbAcquireNode>, 4>
                           &topLevelAcquireNodes) {
  SmallVector<NodeT *, 16> allAcquireNodes;
  for (const auto &acqNode : topLevelAcquireNodes)
    collectAcquireNodesRecursive(static_cast<NodeT *>(acqNode.get()),
                                 allAcquireNodes);
  return allAcquireNodes;
}

} // namespace

///===----------------------------------------------------------------------===///
/// DbAllocNode
///===----------------------------------------------------------------------===///
DbAllocNode::DbAllocNode(DbAllocOp op, DbAnalysis *analysis)
    : dbAllocOp(op), op(op.getOperation()), analysis(analysis) {
  assert(op && "DbAllocOp is required to create DbAllocNode");
  assert(op.getOperation() && "Operation is required to create DbAllocNode");

  if (Value address = dbAllocOp.getAddress()) {
    auto &stringAnalysis = analysis->getAnalysisManager().getStringAnalysis();
    isStringBacked = stringAnalysis.isStringMemRef(address);
  }

}

void DbAllocNode::print(llvm::raw_ostream &os) const {
  os << "DbAllocNode (" << getHierId() << ")";
  os << "\n";
}

void DbAllocNode::forEachChildNode(
    const std::function<void(NodeBase *)> &fn) const {
  for (const auto &n : acquireNodes)
    fn(n.get());
}

DbAcquireNode *DbAllocNode::getOrCreateAcquireNode(DbAcquireOp op) {
  auto it = acquireMap.find(op);
  if (it != acquireMap.end())
    return it->second;
  std::string childId =
      (getHierId() + "." + std::to_string(nextChildId++)).str();
  auto newNode =
      std::make_unique<DbAcquireNode>(op, this, this, analysis, childId);
  DbAcquireNode *ptr = newNode.get();
  acquireNodes.push_back(std::move(newNode));
  acquireMap[op] = ptr;
  return ptr;
}

DbAcquireNode *DbAllocNode::findAcquireNode(DbAcquireOp op) const {
  auto it = acquireMap.find(op);
  return it != acquireMap.end() ? it->second : nullptr;
}

SmallVector<DbAcquireNode *, 16> DbAllocNode::collectAllAcquireNodes() {
  return ::collectAllAcquireNodes<DbAcquireNode>(acquireNodes);
}

SmallVector<const DbAcquireNode *, 16>
DbAllocNode::collectAllAcquireNodes() const {
  return ::collectAllAcquireNodes<const DbAcquireNode>(acquireNodes);
}

bool DbAllocNode::isParallelFriendly() const {
  if (isStringDatablock())
    return false;

  if (accessedInParallelLoop && *accessedInParallelLoop)
    return true;

  SmallVector<const DbAcquireNode *, 16> allAcquireNodes =
      collectAllAcquireNodes();

  /// Offset/size hints on acquires indicate parallel partition intent even
  /// when loop metadata is missing.
  for (const DbAcquireNode *acqNode : allAcquireNodes) {
    if (!acqNode)
      continue;
    DbAcquireOp acqOp = acqNode->getDbAcquireOp();
    if (!acqOp)
      continue;
    if (!acqOp.getPartitionIndices().empty() ||
        !acqOp.getPartitionOffsets().empty())
      return true;
  }

  return false;
}

bool DbAllocNode::canBePartitioned() {
  ARTS_DEBUG("canBePartitioned for alloc " << hierId);

  auto skip = [&](StringRef reason) {
    ARTS_DEBUG("  SKIP alloc " << hierId << ": " << reason);
    return false;
  };

  SmallVector<DbAcquireNode *, 16> allAcquireNodes = collectAllAcquireNodes();

  if (hasNonAffineAccesses && *hasNonAffineAccesses) {
    /// Check if any acquire has explicit partition hints. If hints exist, trust
    /// the authored task partition plan over the static non-affine
    /// classification (e.g., indirect indexing A[B[i]] can still be chunkable
    /// when the parallel work structure is valid).
    bool hasAcquireWithHints = false;
    for (DbAcquireNode *acqNode : allAcquireNodes) {
      if (!acqNode)
        continue;
      DbAcquireOp acqOp = acqNode->getDbAcquireOp();
      if (acqOp && (!acqOp.getPartitionIndices().empty() ||
                    !acqOp.getPartitionOffsets().empty())) {
        hasAcquireWithHints = true;
        ARTS_DEBUG(
            "  Found acquire with partition hints - allowing non-affine alloc");
        break;
      }
    }

    if (hasAcquireWithHints)
      ARTS_DEBUG("  Non-affine access with hints - using fallback");
    else
      ARTS_DEBUG("  Non-affine access - using fine-grained fallback");
  }
  ARTS_DEBUG_REGION({
    ARTS_DBGS() << "  accessedInParallelLoop="
                << (accessedInParallelLoop
                        ? (*accessedInParallelLoop ? "true" : "false")
                        : "nullopt")
                << "\n";
    ARTS_DBGS() << "  hasLoopCarriedDeps="
                << (hasLoopCarriedDeps
                        ? (*hasLoopCarriedDeps ? "true" : "false")
                        : "nullopt")
                << "\n";
    ARTS_DBGS() << "  hasUniformAccess="
                << (hasUniformAccess ? (*hasUniformAccess ? "true" : "false")
                                     : "nullopt")
                << "\n";
    ARTS_DBGS() << "  dominantAccessPattern="
                << (dominantAccessPattern ? std::to_string(static_cast<int>(
                                                *dominantAccessPattern))
                                          : "nullopt")
                << "\n";
  });

  bool allocLevelOk = false;
  if (accessedInParallelLoop && *accessedInParallelLoop) {
    if (!hasLoopCarriedDeps || !*hasLoopCarriedDeps) {
      ARTS_DEBUG("  alloc-level OK (parallel loop, no loop-carried deps)");
      allocLevelOk = true;
    }
  }

  if (!allocLevelOk && hasUniformAccess && *hasUniformAccess) {
    if (dominantAccessPattern) {
      auto pattern = *dominantAccessPattern;
      if (pattern == AccessPatternType::Sequential ||
          pattern == AccessPatternType::Strided) {
        ARTS_DEBUG(
            "  alloc-level OK (uniform access, sequential/strided pattern)");
        allocLevelOk = true;
      }
    }
  }

  if (!allocLevelOk)
    ARTS_DEBUG("  alloc-level metadata inconclusive - deferring to "
               "acquire-level validation");

  if (allAcquireNodes.empty()) {
    ARTS_DEBUG("  No acquire nodes - trivially partitionable");
    return true;
  }

  ARTS_DEBUG("  Checking " << allAcquireNodes.size() << " acquire children...");
  bool anyPassed = false, anyHasStencilBounds = false;
  bool allFailed = true, anyIndirectAccess = false, anyAcquireWithHints = false;

  for (DbAcquireNode *acqNode : allAcquireNodes) {
    if (!acqNode)
      continue;

    DbAcquireOp acqOp = acqNode->getDbAcquireOp();
    if (acqOp && (!acqOp.getPartitionIndices().empty() ||
                  !acqOp.getPartitionOffsets().empty()))
      anyAcquireWithHints = true;
    if (acqNode->hasIndirectAccess())
      anyIndirectAccess = true;

    bool passed = acqNode->canBePartitioned();
    if (passed) {
      anyPassed = true;
      allFailed = false;
      if (auto bounds = acqNode->getStencilBounds()) {
        if (bounds->valid && bounds->isStencil) {
          anyHasStencilBounds = true;
          ARTS_DEBUG("  Acquire has stencil bounds: minOffset="
                     << bounds->minOffset
                     << ", maxOffset=" << bounds->maxOffset);
        }
      }
    } else {
      ARTS_DEBUG("  Acquire failed canBePartitioned()");
    }
  }

  AcquirePatternSummary patterns = summarizeAcquirePatterns();
  ARTS_DEBUG("  Access patterns: hasUniform="
             << patterns.hasUniform << ", hasStencil=" << patterns.hasStencil
             << ", hasIndexed=" << patterns.hasIndexed
             << ", isMixed=" << patterns.isMixed());

  /// Non-uniform access (stencil patterns like A[i-1], A[i], A[i+1]) can
  /// still be partitioned if the acquire-level analysis succeeds.
  if (anyHasStencilBounds) {
    ARTS_DEBUG("  Stencil bounds detected - allowing partition for ESD mode");
    return true;
  }

  if (allFailed && anyIndirectAccess && anyAcquireWithHints) {
    ARTS_DEBUG("  All acquires failed but indirect access with hints "
               "- allowing partitioning for versioning");
    return true;
  }

  if (allFailed &&
      (anyIndirectAccess || (hasNonAffineAccesses && *hasNonAffineAccesses))) {
    ARTS_DEBUG("  All acquires failed but non-affine/indirect access present "
               "- allowing partitioning with full-range acquires");
    return true;
  }

  if (allFailed)
    return skip("all acquire children failed canBePartitioned()");

  if (anyPassed)
    ARTS_DEBUG("  At least one acquire passed validation");

  ARTS_DEBUG("  PASS: alloc " << hierId << " can be partitioned");
  return true;
}

bool DbAllocNode::hasSingleWriter() const {
  /// A single DbAcquireOp can still represent many writers if it is
  /// partitioned by dynamic loop hints. Treat as multi-writer.
  int writeCount = 0;

  SmallVector<const DbAcquireNode *, 16> allAcquireNodes =
      collectAllAcquireNodes();
  for (const DbAcquireNode *acqNode : allAcquireNodes) {
    if (!acqNode)
      continue;
    DbAcquireOp acqOp = acqNode->getDbAcquireOp();
    if (!acqOp)
      continue;

    ArtsMode mode = acqOp.getMode();
    if (!DbUtils::isWriterMode(mode))
      continue;

    writeCount++;
    if (writeCount > 1)
      return false;

    if (!DbAnalysis::hasStaticHints(acqOp))
      return false;
  }

  if (writeCount <= 1) {
    ARTS_DEBUG("Single writer detected -> no overlap possible");
    return true;
  }

  return false;
}

bool DbAllocNode::hasDynamicWriterOffsets() const {
  SmallVector<const DbAcquireNode *, 16> allAcquireNodes =
      collectAllAcquireNodes();
  for (const DbAcquireNode *acqNode : allAcquireNodes) {
    if (!acqNode)
      continue;
    DbAcquireOp acqOp = acqNode->getDbAcquireOp();
    if (!acqOp)
      continue;

    if (!DbUtils::isWriterMode(acqOp.getMode()))
      continue;

    if (!DbAnalysis::hasStaticHints(acqOp)) {
      ARTS_DEBUG("Writer acquire has dynamic offset/size hints");
      return true;
    }
  }

  return false;
}

bool DbAllocNode::allAcquiresWorkerIndexed() const {
  SmallVector<const DbAcquireNode *, 16> allAcquireNodes =
      collectAllAcquireNodes();
  if (allAcquireNodes.empty())
    return true;

  for (const DbAcquireNode *acqNode : allAcquireNodes) {
    if (!acqNode)
      continue;
    if (!acqNode->isWorkerIndexedAccess())
      return false;
  }

  return true;
}

bool DbAllocNode::canProveNonOverlapping() const {
  SmallVector<const DbAcquireNode *, 16> allAcquireNodes =
      collectAllAcquireNodes();

  if (allAcquireNodes.empty())
    return true;

  if (allAcquireNodes.size() == 1) {
    const auto *single = allAcquireNodes.front();
    if (!single)
      return true;
    if (DbAnalysis::hasStaticHints(single->getDbAcquireOp()))
      return true;
  }

  if (hasSingleWriter())
    return true;

  if (hasDynamicWriterOffsets())
    return false;

  bool isPartitioned = false;
  if (dbAllocOp) {
    auto &mutableAlloc = const_cast<DbAllocOp &>(dbAllocOp);
    if (auto mode = getPartitionMode(mutableAlloc.getOperation()))
      isPartitioned = requiresWorkerBoundsPlanning(*mode);
  }
  if (!isPartitioned)
    return false;

  if (allAcquiresWorkerIndexed())
    return true;

  DbAliasAnalysis *aliasAnalysis =
      analysis ? analysis->getAliasAnalysis() : nullptr;

  for (size_t i = 0; i < allAcquireNodes.size(); ++i) {
    for (size_t j = i + 1; j < allAcquireNodes.size(); ++j) {
      const DbAcquireNode *acqA = allAcquireNodes[i];
      const DbAcquireNode *acqB = allAcquireNodes[j];
      if (!acqA || !acqB)
        continue;

      if (aliasAnalysis) {
        auto overlap =
            aliasAnalysis->estimateOverlap(const_cast<DbAcquireNode *>(acqA),
                                           const_cast<DbAcquireNode *>(acqB));
        if (overlap == DbAliasAnalysis::OverlapKind::Disjoint)
          continue;
      }

      return false;
    }
  }

  return true;
}

AcquirePatternSummary DbAllocNode::summarizeAcquirePatterns() const {
  ARTS_DEBUG("summarizeAcquirePatterns() for alloc " << hierId);
  AcquirePatternSummary summary;
  bool hasBlockHint = false;
  bool hasIndirect = false;

  SmallVector<const DbAcquireNode *, 16> allAcquireNodes =
      collectAllAcquireNodes();

  ARTS_DEBUG("  Analyzing " << allAcquireNodes.size() << " acquire nodes");
  for (const DbAcquireNode *acqNode : allAcquireNodes) {
    if (!acqNode)
      continue;
    if (DbAcquireOp acqOp = acqNode->getDbAcquireOp()) {
      auto mode = acqOp.getPartitionModeOr();
      if (usesBlockLayout(mode))
        hasBlockHint = true;
      if (!acqOp.getPartitionOffsets().empty() ||
          !acqOp.getPartitionSizes().empty())
        hasBlockHint = true;
    }
    auto pattern = acqNode->getAccessPattern();
    ARTS_DEBUG("  Acquire node pattern: "
               << (pattern == AccessPattern::Uniform   ? "Uniform"
                   : pattern == AccessPattern::Stencil ? "Stencil"
                   : pattern == AccessPattern::Indexed ? "Indexed"
                                                       : "Unknown"));
    switch (pattern) {
    case AccessPattern::Uniform:
      summary.hasUniform = true;
      break;
    case AccessPattern::Stencil:
      summary.hasStencil = true;
      break;
    case AccessPattern::Indexed:
      summary.hasIndexed = true;
      break;
    default:
      break;
    }
    if (!hasIndirect && acqNode->hasIndirectAccess())
      hasIndirect = true;
  }

  if (hasNonAffineAccesses && *hasNonAffineAccesses) {
    /// Treat indirect access as indexed even when block hints are present.
    /// This enables block partitioning with full-range acquires.
    if (hasIndirect || !hasBlockHint)
      summary.hasIndexed = true;
  } else if (!summary.hasIndexed && hasIndirect) {
    summary.hasIndexed = true;
  }

  ARTS_DEBUG("  Summary: hasUniform=" << summary.hasUniform
                                      << ", hasStencil=" << summary.hasStencil
                                      << ", hasIndexed=" << summary.hasIndexed);
  return summary;
}
