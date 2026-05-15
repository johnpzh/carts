///==========================================================================///
/// File: EdtNode.cpp
/// Implementation of EDT nodes for graph analysis.
///==========================================================================///

#include "carts/dialect/arts/Analysis/graphs/edt/EdtNode.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/edt/EdtAnalysis.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "carts/dialect/arts/Analysis/loop/LoopNode.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mlir::arts;

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_node);

EdtNode::EdtNode(EdtOp op, EdtAnalysis *EA)
    : NodeBase(), edtOp(op), edtAnalysis(EA) {
  assert(edtOp.getOperation() && "Operation must always be available");
}

void EdtNode::print(llvm::raw_ostream &os) const {
  os << "EdtNode (" << getHierId() << ")\n";
}

bool EdtNode::hasNestedEdts() const {
  if (!edtOp)
    return false;
  EdtOp op = edtOp;
  bool found = false;
  op.getRegion().walk([&](EdtOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

bool EdtNode::hasNestedTaskEdts() const {
  if (!edtOp)
    return false;
  EdtOp op = edtOp;
  bool found = false;
  op.walk([&](EdtOp nestedEdt) {
    if (nestedEdt == op)
      return WalkResult::advance();
    if (nestedEdt.getType() == EdtType::task) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}
