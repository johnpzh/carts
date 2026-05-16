///==========================================================================///
/// File: EdtNode.cpp
/// Implementation of EDT nodes for graph analysis.
///==========================================================================///

#include "carts/dialect/arts/Analysis/graphs/edt/EdtNode.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

EdtNode::EdtNode(EdtOp op) : NodeBase(), edtOp(op) {
  assert(edtOp.getOperation() && "Operation must always be available");
}
