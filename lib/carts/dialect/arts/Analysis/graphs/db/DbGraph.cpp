///==========================================================================///
/// File: DbGraph.cpp
/// Implementation of DbGraph for DB operation analysis.
///==========================================================================///
#include "carts/dialect/arts/Analysis/graphs/db/DbGraph.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAliasAnalysis.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/LocationMetadata.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <string>
using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_graph);

std::string DbGraph::generateAllocId(unsigned id) {
  std::string s;
  if (id == 0)
    return "A";
  unsigned n = id;
  while (n > 0) {
    unsigned rem = (n - 1) % 26;
    s = char('A' + rem) + s;
    n = (n - 1) / 26;
  }
  return s;
}

DbGraph::DbGraph(func::FuncOp func, DbAnalysis *analysis)
    : func(func), analysis(analysis) {
  assert(analysis && "Analysis cannot be null");
  ARTS_DEBUG("Creating DB Graph for function: " << func.getName().str());
}

DbGraph::~DbGraph() = default;

void DbGraph::build() {
  if (built.load(std::memory_order_acquire) &&
      !needsRebuild.load(std::memory_order_acquire))
    return;
  ARTS_DEBUG("Building DB hierarchy");
  invalidate();
  collectNodes();
  computeOpOrder();
  computeMetrics();
  built.store(true, std::memory_order_release);
  needsRebuild.store(false, std::memory_order_release);
}

void DbGraph::invalidate() {
  allocNodes.clear();
  acquireNodeMap.clear();
  opOrder.clear();
  nextAllocId = 1;
  built.store(false, std::memory_order_release);
  needsRebuild.store(true, std::memory_order_release);
}

DbAllocNode *DbGraph::getDbAllocNode(DbAllocOp op) const {
  auto it = allocNodes.find(op);
  return it != allocNodes.end() ? it->second.get() : nullptr;
}

DbAcquireNode *DbGraph::getDbAcquireNode(DbAcquireOp op) const {
  auto it = acquireNodeMap.find(op);
  return it != acquireNodeMap.end() ? it->second : nullptr;
}

DbAllocNode *DbGraph::getOrCreateAllocNode(DbAllocOp op) {
  if (auto node = getDbAllocNode(op))
    return node;

  /// Verify operation is valid before creating node
  Operation *opPtr = op.getOperation();
  if (!opPtr) {
    ARTS_ERROR("Cannot create DbAllocNode for null operation");
    return nullptr;
  }

  /// Verify required elementType attribute exists
  Type elementType = op.getElementType();
  if (!elementType) {
    ARTS_ERROR("DbAllocOp missing elementType attribute: " << *opPtr);
    return nullptr;
  }

  auto newNode = std::make_unique<DbAllocNode>(op, analysis);
  newNode->setHierId(generateAllocId(nextAllocId++));
  DbAllocNode *ptr = newNode.get();
  allocNodes[op] = std::move(newNode);
  return ptr;
}

DbAcquireNode *DbGraph::getOrCreateAcquireNode(DbAcquireOp op) {
  /// Verify operation is valid
  Operation *opPtr = op.getOperation();
  if (!opPtr) {
    ARTS_ERROR("Cannot create DbAcquireNode: operation pointer is null");
    return nullptr;
  }

  if (auto node = getDbAcquireNode(op))
    return node;

  Value ptr = op.getPtr();
  if (!ptr) {
    ARTS_ERROR("DbAcquireOp ptr is null");
    return nullptr;
  }

  Value sourcePtr = op.getSourcePtr();
  if (!sourcePtr) {
    ARTS_ERROR("DbAcquireOp sourcePtr is null");
    return nullptr;
  }

  /// Use getUnderlyingDbAlloc to trace through acquires to find the ultimate
  /// DbAllocOp
  Operation *allocOp = DbUtils::getUnderlyingDbAlloc(sourcePtr);
  if (!allocOp) {
    ARTS_ERROR("Cannot get underlying DB alloc for acquire");
    return nullptr;
  }

  if (!isa<DbAllocOp>(allocOp)) {
    ARTS_ERROR("getUnderlyingDbAlloc returned non-DbAllocOp operation: "
               << allocOp->getName());
    return nullptr;
  }

  DbAllocOp dbAllocOp = cast<DbAllocOp>(allocOp);

  /// Get the immediate underlying DB from sourcePtr to determine if this is a
  /// nested acquire
  Operation *immediateUnderlying = DbUtils::getUnderlyingDb(sourcePtr, 0);
  if (!immediateUnderlying) {
    ARTS_ERROR("Cannot get immediate underlying DB for acquire");
    return nullptr;
  }

  if (auto parentAcquireOp = dyn_cast<DbAcquireOp>(immediateUnderlying)) {
    /// This is a nested acquire - parent is another acquire
    /// Check for cycle before recursing
    if (parentAcquireOp == op) {
      ARTS_ERROR("Cycle detected: DbAcquireOp points to itself");
      return nullptr;
    }

    DbAcquireNode *parentAcquire = getOrCreateAcquireNode(parentAcquireOp);
    if (!parentAcquire) {
      ARTS_ERROR("Failed to get or create parent acquire node");
      return nullptr;
    }
    /// Parent acquire must exist now; its root alloc is tracked inside the node
    DbAcquireNode *acquireNode = parentAcquire->getOrCreateAcquireNode(op);
    if (!acquireNode) {
      ARTS_ERROR("Failed to create acquire node from parent");
      return nullptr;
    }
    acquireNodeMap[op] = acquireNode;
    return acquireNode;
  } else {
    /// Direct acquire from alloc
    DbAllocNode *allocNode = getOrCreateAllocNode(dbAllocOp);
    if (!allocNode) {
      ARTS_ERROR("Failed to get or create alloc node");
      return nullptr;
    }
    DbAcquireNode *acquireNode = allocNode->getOrCreateAcquireNode(op);
    if (!acquireNode) {
      ARTS_ERROR("Failed to create acquire node");
      return nullptr;
    }
    acquireNodeMap[op] = acquireNode;
    return acquireNode;
  }

  /// Should never reach here - all cases handled above
  ARTS_ERROR("Unexpected control flow in getOrCreateAcquireNode");
  return nullptr;
}

void DbGraph::forEachAllocNode(
    const std::function<void(DbAllocNode *)> &fn) const {
  for (const auto &pair : allocNodes)
    fn(pair.second.get());
}

void DbGraph::forEachAcquireNode(
    const std::function<void(DbAcquireNode *)> &fn) const {
  for (const auto &pair : acquireNodeMap)
    fn(pair.second);
}

/// Walk through the function and collect all DB operations
/// (alloc/acquire/release) into graph nodes
void DbGraph::collectNodes() {
  ARTS_DEBUG("[Phase 1] Collecting nodes");

  func.walk([&](Operation *op) {
    if (auto alloc = dyn_cast<DbAllocOp>(op)) {
      getOrCreateAllocNode(alloc);
      return;
    }
    if (auto acq = dyn_cast<DbAcquireOp>(op))
      getOrCreateAcquireNode(acq);
  });

  ARTS_DEBUG("[Phase 1] Collected " << allocNodes.size() << " allocations, and "
                                    << acquireNodeMap.size() << " acquires");
  ARTS_DEBUG("[Phase 1] collectNodes() completed");
}

/// Compute operation order based on the order of operations in the function.
void DbGraph::computeOpOrder() {
  unsigned idx = 0;
  func.walk([&](Operation *op) { opOrder[op] = idx++; });
}

/// Compute per-allocation metrics and global summary metrics.
/// This includes timing intervals, critical spans/paths, loop depths,
/// and peak resource usage across all allocations.
void DbGraph::computeMetrics() {
  /// Process each allocation to compute per-node metrics
  for (const auto &pair : allocNodes)
    computeAllocMetrics(pair.first, pair.second.get());
}

/// Compute metrics for a single allocation.
/// This processes all acquire/release child nodes and derives timing intervals,
/// critical spans, paths, loop depths, and lifetime classification.
void DbGraph::computeAllocMetrics(DbAllocOp alloc, DbAllocNode *allocNode) {
  DbAllocNode &info = *allocNode;
  /// Initialize allocation-level counters
  info.beginIndex = getOpOrder(alloc.getOperation());
  info.numAcquires = 0;
  info.inCount = 0;
  info.outCount = 0;
  info.totalAccessBytes = 0;

  /// Collect all acquire nodes (including nested)
  SmallVector<DbAcquireNode *, 16> allAcquireNodes;
  std::function<void(DbAcquireNode *)> collectAcqRec;
  collectAcqRec = [&](DbAcquireNode *acq) {
    allAcquireNodes.push_back(acq);
    processAcquireNode(acq, info);
    acq->forEachChildNode([&](NodeBase *child) {
      if (auto *nested = dyn_cast<DbAcquireNode>(child))
        collectAcqRec(nested);
    });
  };
  allocNode->forEachChildNode([&](NodeBase *child) {
    if (auto *acq = dyn_cast<DbAcquireNode>(child))
      collectAcqRec(acq);
  });

  /// Sort acquire nodes by program order for interval processing
  llvm::sort(allAcquireNodes,
             [](const DbAcquireNode *a, const DbAcquireNode *b) {
               return a->beginIndex < b->beginIndex;
             });

  /// Compute the allocation's end index as the latest release
  uint64_t lastEnd = info.beginIndex;
  for (auto *node : allAcquireNodes)
    lastEnd = std::max(lastEnd, node->endIndex);
  info.endIndex = lastEnd;

  /// Compute derived lifetime and locality metrics
  computeLoopDepth(info, allAcquireNodes);
}

/// Process an acquire node: compute timing interval, link to release,
/// and aggregate access statistics into the parent allocation.
void DbGraph::processAcquireNode(DbAcquireNode *acq, DbAllocNode &info) {
  /// Compute the lifetime interval [beginIndex, endIndex] in program order
  auto acquireOp = cast<DbAcquireOp>(acq->getOp());
  unsigned beginIndex = getOpOrder(acquireOp.getOperation());
  unsigned endIndex = beginIndex;

  /// Use the precomputed release from the acquire node
  DbReleaseOp releaseOp = acq->getDbReleaseOp();
  if (releaseOp) {
    endIndex = getOpOrder(releaseOp.getOperation());
  }

  /// Update timing info in the acquire node
  acq->beginIndex = beginIndex;
  acq->endIndex = endIndex;

  /// NOTE: No longer adding to alloc->acquireNodes - graph structure is in
  /// allocNode->acquireNodes! The caller builds a local list for computation
  /// purposes.

  /// Aggregate access statistics
  ++info.numAcquires;
  info.inCount += acq->inCount;
  info.outCount += acq->outCount;
  info.totalAccessBytes += acq->estimatedBytes;
}

/// Compute maximum loop depth for an allocation.
/// It captures the deepest loop nesting among all acquires of this
/// allocation as a proxy for locality/pressure heuristics.
void DbGraph::computeLoopDepth(
    DbAllocNode &info, const SmallVectorImpl<DbAcquireNode *> &acquireNodes) {
  auto *loopAnalysis = analysis->getLoopAnalysis();
  assert(loopAnalysis && "Loop analysis required for loop depth");
  unsigned maxDepth = 0;
  for (auto *node : acquireNodes) {
    SmallVector<LoopNode *, 4> enclosingLoops;
    loopAnalysis->collectEnclosingLoops(node->getOp(), enclosingLoops);
    maxDepth = std::max<unsigned>(maxDepth, enclosingLoops.size());
  }
  info.maxLoopDepth = maxDepth;
}

llvm::json::Value DbGraph::exportToJsonValue() const {
  using namespace llvm::json;

  /// ArtsMate DB entities export
  Array dbs;

  for (const auto &kv : allocNodes) {
    DbAllocNode *allocNode = kv.second.get();
    DbAllocOp allocOp = allocNode->getDbAllocOp();

    Object db;

    /// Partitioning mode
    PartitionMode partitionMode =
        DbAnalysis::getPartitionModeFromStructure(allocNode->getDbAllocOp());
    db["partitioning"] = stringifyPartitionMode(partitionMode);

    /// Helper: extract static value or null for dynamic
    auto extractSizes = [](OperandRange values) -> Array {
      Array arr;
      for (Value v : values) {
        if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
          arr.push_back(cst.value());
        /// Dynamic dimension
        else
          arr.push_back(nullptr);
      }
      return arr;
    };

    /// Outer sizes (DB grid dimensions)
    Array outerSizes = extractSizes(allocOp.getSizes());
    /// Coarse allocation = single DB
    if (outerSizes.empty())
      outerSizes.push_back(1);
    db["outer_sizes"] = std::move(outerSizes);

    /// Inner sizes (each DB's payload)
    Array innerSizes = extractSizes(allocOp.getElementSizes());
    db["inner_sizes"] = std::move(innerSizes);

    /// Element type
    Type elemType = allocOp.getElementType();
    std::string dtype;
    if (auto intType = dyn_cast<IntegerType>(elemType))
      dtype = "i" + std::to_string(intType.getWidth());
    else if (auto floatType = dyn_cast<FloatType>(elemType))
      dtype = "f" + std::to_string(floatType.getWidth());
    else
      dtype = "unknown";
    db["dtype"] = dtype;

    /// Access mode from DbAllocOp
    ArtsMode mode = allocOp.getMode();
    if (mode == ArtsMode::in)
      db["access_mode"] = "in";
    else if (mode == ArtsMode::out)
      db["access_mode"] = "out";
    else
      db["access_mode"] = "inout";

    /// Heuristic decision (populated by DbHeuristics)
    db["heuristic"] = nullptr;

    /// Source location
    auto loc = LocationMetadata::fromLocation(allocOp->getLoc());
    if (loc.isValid())
      db["loc"] = loc.file + ":" + std::to_string(loc.line);
    else
      db["loc"] = "unknown";

    /// Static analysis data from DbAllocNode
    Object staticAnalysis;

    /// Lifetime metrics
    staticAnalysis["begin_index"] = (int64_t)allocNode->beginIndex;
    staticAnalysis["end_index"] = (int64_t)allocNode->endIndex;

    /// Access metrics
    staticAnalysis["total_access_bytes"] = (int64_t)allocNode->totalAccessBytes;
    staticAnalysis["num_acquires"] = (int64_t)allocNode->numAcquires;
    staticAnalysis["max_loop_depth"] = (int64_t)allocNode->maxLoopDepth;

    /// Parallelization and partitioning hints
    staticAnalysis["is_parallel_friendly"] = allocNode->isParallelFriendly();
    staticAnalysis["can_be_partitioned"] = allocNode->canBePartitioned();
    staticAnalysis["is_fine_grained"] =
        DbAnalysis::isFineGrained(allocNode->getDbAllocOp());

    /// Twin-diff analysis
    staticAnalysis["has_single_writer"] = allocNode->hasSingleWriter();
    staticAnalysis["all_acquires_worker_indexed"] =
        allocNode->allAcquiresWorkerIndexed();
    staticAnalysis["can_prove_non_overlapping"] =
        allocNode->canProveNonOverlapping();

    /// String datablock flag
    staticAnalysis["is_string_backed"] = allocNode->isStringDatablock();

    /// Memory footprint
    if (allocNode->memoryFootprint.has_value())
      staticAnalysis["memory_footprint_bytes"] = *allocNode->memoryFootprint;

    /// Access pattern
    if (allocNode->dominantAccessPattern.has_value()) {
      std::string pattern;
      switch (*allocNode->dominantAccessPattern) {
      case AccessPatternType::Unknown:
        pattern = "unknown";
        break;
      case AccessPatternType::Sequential:
        pattern = "sequential";
        break;
      case AccessPatternType::Strided:
        pattern = "strided";
        break;
      case AccessPatternType::Stencil:
        pattern = "stencil";
        break;
      case AccessPatternType::Random:
        pattern = "random";
        break;
      case AccessPatternType::GatherScatter:
        pattern = "gather_scatter";
        break;
      }
      staticAnalysis["access_pattern"] = pattern;
    }

    db["static_analysis"] = std::move(staticAnalysis);

    /// Export child acquire nodes
    Array acquires;
    allocNode->forEachChildNode([&](NodeBase *child) {
      auto *acqNode = dyn_cast<DbAcquireNode>(child);
      if (!acqNode)
        return;

      DbAcquireOp acqOp = acqNode->getDbAcquireOp();
      if (!acqOp)
        return;

      Object acq;

      /// Access mode
      ArtsMode acqMode = acqOp.getMode();
      if (acqMode == ArtsMode::in)
        acq["access_mode"] = "in";
      else if (acqMode == ArtsMode::out)
        acq["access_mode"] = "out";
      else
        acq["access_mode"] = "inout";

      /// Source location
      auto acqLoc = LocationMetadata::fromLocation(acqOp->getLoc());
      if (acqLoc.isValid())
        acq["loc"] = acqLoc.file + ":" + std::to_string(acqLoc.line);

      /// Acquire offsets and sizes (which datablocks are being acquired)
      auto extractValues = [](OperandRange values) -> Array {
        Array arr;
        for (Value v : values) {
          if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
            arr.push_back(cst.value());
          else
            arr.push_back(nullptr);
        }
        return arr;
      };

      Array offsets = extractValues(acqOp.getOffsets());
      if (!offsets.empty())
        acq["offsets"] = std::move(offsets);

      Array sizes = extractValues(acqOp.getSizes());
      if (!sizes.empty())
        acq["sizes"] = std::move(sizes);

      acquires.push_back(std::move(acq));
    });

    if (!acquires.empty())
      db["acquires"] = std::move(acquires);

    dbs.push_back(std::move(db));
  }

  return llvm::json::Value(std::move(dbs));
}
