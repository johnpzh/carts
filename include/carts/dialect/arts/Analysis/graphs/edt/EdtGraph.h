///==========================================================================///
/// File: EdtGraph.h
///
/// Defines EdtGraph for EDT analysis.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_EDT_EDTGRAPH_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_EDT_EDTGRAPH_H

#include "carts/Dialect.h" // For EdtOp
#include "carts/dialect/arts/Analysis/graphs/base/EdgeBase.h"
#include "carts/dialect/arts/Analysis/graphs/base/NodeBase.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbGraph.h"
#include "carts/dialect/arts/Analysis/graphs/edt/EdtEdge.h"
#include "carts/dialect/arts/Analysis/graphs/edt/EdtNode.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <memory>

namespace mlir {
namespace carts::arts {

class EdtDepEdge;
class EdtAnalysis;

/// Represents task dependencies with edges labeled by data blocks.
class EdtGraph {
public:
  EdtGraph(func::FuncOp func, DbGraph *dbGraph, EdtAnalysis *EA);

  void build();
  void buildNodesOnly();
  void invalidate();
  void print(llvm::raw_ostream &os);
  llvm::json::Value exportToJsonValue(bool includeAnalysis = false) const;
  void exportToJson(llvm::raw_ostream &os, bool includeAnalysis = false) const;
  NodeBase *getEntryNode() const;
  NodeBase *getOrCreateNode(Operation *op);
  NodeBase *getNode(Operation *op) const;
  EdtNode *getEdtNode(EdtOp edt) const;
  void forEachNode(const std::function<void(NodeBase *)> &fn) const;
  bool addEdge(NodeBase *from, NodeBase *to, EdgeBase *edge);

  /// Edt-specific methods
  bool isEdtReachable(EdtOp from, EdtOp to);
  void getDeterministicTopologicalOrder(
      SmallVectorImpl<EdtNode *> &topoOrder,
      SmallVectorImpl<EdtNode *> &leftoverNodes) const;
  func::FuncOp getFunction() const { return func; }
  bool hasDbGraph() const { return dbGraph != nullptr; }
  DbGraph *getDbGraph() const { return dbGraph; }
  size_t size() const { return nodes.size(); }
  uint64_t getVersion() const {
    return version.load(std::memory_order_relaxed);
  }

  /// Check if two EDTs are independent under the current memory-root model.
  /// Shared read-only roots are allowed; any shared writable root is treated
  /// as a dependency that prevents barrier elision.
  bool areEdtsIndependent(EdtOp a, EdtOp b);

private:
  func::FuncOp func;
  DbGraph *dbGraph;
  EdtAnalysis *edtAnalysis;
  DenseMap<EdtOp, std::unique_ptr<EdtNode>> edtNodes;
  SmallVector<NodeBase *, 8> nodes;
  DenseMap<std::pair<NodeBase *, NodeBase *>, std::unique_ptr<EdgeBase>> edges;
  std::atomic<bool> isBuilt{false};
  std::atomic<bool> needsRebuild{true};
  std::atomic<uint64_t> version{0};

  /// Private helpers
  void collectNodes();
  void linkEdtsToLoops();
  void buildDependencies();
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_EDT_EDTGRAPH_H
