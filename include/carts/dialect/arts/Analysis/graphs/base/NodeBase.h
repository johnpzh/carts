///==========================================================================///
/// File: NodeBase.h
///
/// Abstract base class for graph nodes.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_BASE_NODEBASE_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_BASE_NODEBASE_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace arts {

class EdgeBase;

/// Abstract base class for all nodes (e.g., allocs, deps, tasks).
class NodeBase {
public:
  virtual ~NodeBase() = default;

  enum class NodeKind { DbAlloc, DbAcquire, EdtTask };

  /// Get the hierarchical ID (e.g., "A.1" for deps).
  virtual StringRef getHierId() const = 0;

  /// Print node details.
  virtual void print(llvm::raw_ostream &os) const = 0;

  /// Get the underlying MLIR operation.
  virtual Operation *getOp() const = 0;

  /// LLVM-style RTTI support
  virtual NodeKind getKind() const = 0;

  /// Add incoming/outgoing edges.
  virtual void addInEdge(EdgeBase *edge) = 0;
  virtual void addOutEdge(EdgeBase *edge) = 0;

  /// Get incoming/outgoing edges.
  virtual const DenseSet<EdgeBase *> &getInEdges() const = 0;
  virtual const DenseSet<EdgeBase *> &getOutEdges() const = 0;

protected:
  /// Protected for derived classes.
  DenseSet<EdgeBase *> inEdges, outEdges;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_BASE_NODEBASE_H