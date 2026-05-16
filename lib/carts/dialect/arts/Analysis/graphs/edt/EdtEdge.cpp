///==========================================================================///
/// File: EdtEdge.cpp
/// Implementation of EDT edges for graph analysis.
///==========================================================================///

#include "carts/dialect/arts/Analysis/graphs/edt/EdtEdge.h"
#include "carts/dialect/arts/Analysis/graphs/base/NodeBase.h"

using namespace mlir::carts;
using namespace mlir::carts::arts;

EdtDepEdge::EdtDepEdge(NodeBase *from, NodeBase *to, const DbEdge &)
    : from(from), to(to) {
  assert(from && "Source node cannot be null");
  assert(to && "Destination node cannot be null");
}
