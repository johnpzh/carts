///==========================================================================///
/// File: EdtEdge.h
///
/// Defines EDT-specific edges for graph analysis.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_EDT_EDTEDGE_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_EDT_EDTEDGE_H

#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/graphs/base/EdgeBase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>

namespace mlir {
namespace arts {

class DbAllocNode;
class DbAcquireNode;

/// Data dependency types carried by EDT edges.
enum class DbDepType { RAW, WAR, WAW, RAR };

struct DbEdge {
  DbAcquireNode *producer = nullptr;
  DbAcquireNode *consumer = nullptr;
  DbDepType depType = DbDepType::RAW;

  bool operator==(const DbEdge &other) const {
    return producer == other.producer && consumer == other.consumer &&
           depType == other.depType;
  }
};

class EdtDepEdge : public EdgeBase {
public:
  EdtDepEdge(NodeBase *from, NodeBase *to, const DbEdge &edge);

  NodeBase *getFrom() const override { return from; }
  NodeBase *getTo() const override { return to; }
  EdgeKind getKind() const override { return EdgeKind::Dep; }
  StringRef getType() const override { return typeLabel; }
  void print(llvm::raw_ostream &os) const override;

  ArrayRef<DbEdge> getEdges() const { return dbEdges.getArrayRef(); }
  void appendEdge(const DbEdge &edge) { dbEdges.insert(edge); }

  static bool classof(const EdgeBase *E) {
    return E->getKind() == EdgeKind::Dep;
  }

private:
  NodeBase *from, *to;
  SetVector<DbEdge, SmallVector<DbEdge, 2>> dbEdges;
  std::string typeLabel;
};

} // namespace arts
} // namespace mlir

namespace llvm {
template <> struct DenseMapInfo<mlir::arts::DbEdge> {
  using DbEdge = mlir::arts::DbEdge;
  static inline DbEdge getEmptyKey() {
    return {reinterpret_cast<mlir::arts::DbAcquireNode *>(-1),
            reinterpret_cast<mlir::arts::DbAcquireNode *>(-1),
            mlir::arts::DbDepType::RAW};
  }
  static inline DbEdge getTombstoneKey() {
    return {reinterpret_cast<mlir::arts::DbAcquireNode *>(-2),
            reinterpret_cast<mlir::arts::DbAcquireNode *>(-2),
            mlir::arts::DbDepType::RAW};
  }
  static unsigned getHashValue(const DbEdge &edge) {
    return DenseMapInfo<void *>::getHashValue(edge.producer) ^
           (DenseMapInfo<void *>::getHashValue(edge.consumer) << 1) ^
           static_cast<unsigned>(edge.depType);
  }
  static bool isEqual(const DbEdge &lhs, const DbEdge &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_EDT_EDTEDGE_H
