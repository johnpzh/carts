///==========================================================================///
/// File: DbGraph.h
///
/// Defines DbGraph as a lightweight derived hierarchy/cache for datablock
/// operations.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBGRAPH_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBGRAPH_H

#include "carts/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include <atomic>
#include <functional>
#include <memory>
#include <string>

namespace mlir {
namespace carts::arts {

/// Forward declarations
class DbAnalysis;

class DbAllocNode;
class DbAcquireNode;

/// DbGraph maintains the hierarchy of DbAlloc/DbAcquire nodes plus metrics.
/// It is an analysis projection over canonical IR contracts and DB operations,
/// not an independent semantic source of truth.
class DbGraph {
public:
  DbGraph(func::FuncOp func, DbAnalysis *analysis);
  ~DbGraph();

  void build();
  void invalidate();
  llvm::json::Value exportToJsonValue(bool includeAnalysis = false) const;

  /// Retrieve node accessors by concrete op type
  DbAllocNode *getDbAllocNode(DbAllocOp op) const;
  DbAcquireNode *getDbAcquireNode(DbAcquireOp op) const;

  /// Convenience methods for getting or creating specific node types
  DbAllocNode *getOrCreateAllocNode(DbAllocOp op);
  DbAcquireNode *getOrCreateAcquireNode(DbAcquireOp op);

  /// Apply a function to every node (alloc/acquire)
  void forEachAllocNode(const std::function<void(DbAllocNode *)> &fn) const;
  void forEachAcquireNode(const std::function<void(DbAcquireNode *)> &fn) const;

  /// Get operation order for ordering comparisons
  unsigned getOpOrder(Operation *op) const {
    auto it = opOrder.find(op);
    return it == opOrder.end() ? 0u : it->second;
  }

  /// Get DbAnalysis instance
  DbAnalysis *getAnalysis() const { return analysis; }

private:
  func::FuncOp func;
  DbAnalysis *analysis;

  /// Node maps
  DenseMap<DbAllocOp, std::unique_ptr<DbAllocNode>> allocNodes;
  DenseMap<DbAcquireOp, DbAcquireNode *> acquireNodeMap;

  unsigned nextAllocId = 1;

  /// Private helpers
  void collectNodes();
  void computeOpOrder();
  void computeMetrics();
  std::string generateAllocId(unsigned id);

  /// Metrics computation helpers
  void computeAllocMetrics(DbAllocOp alloc, DbAllocNode *allocNode);
  void processAcquireNode(DbAcquireNode *acq, DbAllocNode &info);
  void computeLoopDepth(DbAllocNode &info,
                        const SmallVectorImpl<DbAcquireNode *> &acquireNodes);
private:
  DenseMap<Operation *, unsigned> opOrder;
  std::atomic<bool> built{false};
  std::atomic<bool> needsRebuild{true};
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBGRAPH_H
