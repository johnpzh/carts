///==========================================================================///
/// File: EdtEdge.h
///
/// Defines EDT-specific edges for graph analysis.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_EDT_EDTEDGE_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_EDT_EDTEDGE_H

#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/graphs/base/EdgeBase.h"
#include "llvm/ADT/DenseMapInfo.h"
#include <cstdint>
#include <string>

namespace mlir {
namespace carts::arts {

class DbAllocNode;
class DbAcquireNode;

/// Data dependency types carried by EDT edges.
enum class DbDepType { RAW, WAR, WAW };

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

  static bool classof(const EdgeBase *E) {
    return E->getKind() == EdgeKind::Dep;
  }

private:
  NodeBase *from, *to;
  std::string typeLabel;
};

} // namespace carts::arts
} // namespace mlir

namespace llvm {
template <> struct DenseMapInfo<mlir::carts::arts::DbEdge> {
  using DbEdge = mlir::carts::arts::DbEdge;
  static inline DbEdge getEmptyKey() {
    return {reinterpret_cast<mlir::carts::arts::DbAcquireNode *>(-1),
            reinterpret_cast<mlir::carts::arts::DbAcquireNode *>(-1),
            mlir::carts::arts::DbDepType::RAW};
  }
  static inline DbEdge getTombstoneKey() {
    return {reinterpret_cast<mlir::carts::arts::DbAcquireNode *>(-2),
            reinterpret_cast<mlir::carts::arts::DbAcquireNode *>(-2),
            mlir::carts::arts::DbDepType::RAW};
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
