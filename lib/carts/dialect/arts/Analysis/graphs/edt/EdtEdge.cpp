///==========================================================================///
/// File: EdtEdge.cpp
/// Implementation of EDT edges for graph analysis.
///==========================================================================///

#include "carts/dialect/arts/Analysis/graphs/edt/EdtEdge.h"
#include "carts/dialect/arts/Analysis/graphs/base/NodeBase.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"

using namespace mlir::arts;

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
  dbEdges.insert(edge);
  typeLabel = inferLabel(edge);
}

void EdtDepEdge::print(llvm::raw_ostream &os) const {
  os << "EDT DEPENDENCY EDGE:\n"
     << "  - From: " << from->getHierId() << "\n"
     << "  - To: " << to->getHierId() << "\n";
  for (const auto &edge : dbEdges) {
    std::string label = inferLabel(edge);
    os << "  - Dep: " << label << " [";
    switch (edge.depType) {
    case DbDepType::RAW:
      os << "RAW";
      break;
    case DbDepType::WAR:
      os << "WAR";
      break;
    case DbDepType::WAW:
      os << "WAW";
      break;
    case DbDepType::RAR:
      os << "RAR";
      break;
    }
    os << "]\n";
  }
}
