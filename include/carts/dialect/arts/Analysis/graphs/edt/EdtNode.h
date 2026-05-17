///==========================================================================///
/// File: EdtNode.h
///
/// Defines EDT-specific nodes derived from NodeBase.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_EDT_EDTNODE_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_EDT_EDTNODE_H

#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Analysis/edt/EdtInfo.h"
#include "carts/dialect/arts/Analysis/graphs/base/NodeBase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace carts::arts {

/// Forward declarations
class LoopNode;

////===----------------------------------------------------------------------===////
/// EdtNode - represents an EDT operation
////===----------------------------------------------------------------------===////
class EdtNode : public NodeBase {
public:
  explicit EdtNode(EdtOp op);

  StringRef getHierId() const override { return hierId; }
  void setHierId(std::string id) { hierId = std::move(id); }
  Operation *getOp() const override {
    return const_cast<EdtOp &>(edtOp).getOperation();
  }

  void addInEdge(EdgeBase *edge) override { inEdges.insert(edge); }
  void addOutEdge(EdgeBase *edge) override { outEdges.insert(edge); }
  const DenseSet<EdgeBase *> &getInEdges() const override { return inEdges; }
  const DenseSet<EdgeBase *> &getOutEdges() const override { return outEdges; }

  EdtOp getEdtOp() const { return edtOp; }
  EdtInfo &getInfo() { return info; }
  const EdtInfo &getInfo() const { return info; }

  /// Loop association helpers
  void setAssociatedLoops(ArrayRef<LoopNode *> loops) {
    enclosingLoops.assign(loops.begin(), loops.end());
  }

  ArrayRef<LoopNode *> getAssociatedLoops() const { return enclosingLoops; }

  NodeKind getKind() const override { return NodeKind::EdtTask; }
  static bool classof(const NodeBase *N) {
    return N->getKind() == NodeKind::EdtTask;
  }

private:
  EdtOp edtOp;
  std::string hierId;
  EdtInfo info;

  /// Optional association with one or more enclosing loops
  SmallVector<LoopNode *, 4> enclosingLoops;
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_EDT_EDTNODE_H
