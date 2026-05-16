///==========================================================================///
/// File: EdtEdge.cpp
/// Implementation of EDT edges for graph analysis.
///==========================================================================///

#include "carts/dialect/arts/Analysis/graphs/edt/EdtEdge.h"
#include "carts/dialect/arts/Analysis/graphs/base/NodeBase.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"

using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {
std::string inferLabel(const DbEdge &edge) {
  auto getLabel = [](DbAcquireNode *node) -> std::string {
    if (!node)
      return "";
    if (auto *alloc = node->getRootAlloc())
      return alloc->getHierId().str();
    return "";
  };
  std::string label = getLabel(edge.producer);
  if (!label.empty())
    return label;
  label = getLabel(edge.consumer);
  return label.empty() ? "db" : label;
}
} // namespace

EdtDepEdge::EdtDepEdge(NodeBase *from, NodeBase *to, const DbEdge &edge)
    : from(from), to(to) {
  assert(from && "Source node cannot be null");
  assert(to && "Destination node cannot be null");
  typeLabel = inferLabel(edge);
}
