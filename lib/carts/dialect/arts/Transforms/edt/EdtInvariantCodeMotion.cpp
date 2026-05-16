///==========================================================================///
/// File: EdtInvariantCodeMotion.cpp
///
/// Implementation of EdtInvariantCodeMotion pass.
///
/// Before:
///   arts.edt <task> {
///     %c = arith.constant 64 : index
///     %u = arith.addi %n, %c
///     ... use %u ...
///   }
///
/// After:
///   %c = arith.constant 64 : index
///   %u = arith.addi %n, %c
///   arts.edt <task> { ... use %u ... }
///==========================================================================///

/// Dialects
#include "carts/utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <queue>

/// Arts
#include "carts/Dialect.h"
#include "carts/dialect/arts/Transforms/edt/EdtICM.h"
#include "carts/utils/LoopInvarianceUtils.h"

/// Others
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallPtrSet.h"

/// LLVM support
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_invariant_code_motion);

namespace mlir {
namespace carts::arts {

bool canBeHoistedFromEdt(Region &edtRegion, Operation *op) {
  /// Do not move terminators.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;

  /// Do not move other ARTS ops
  if (isArtsOp(op))
    return false;

  /// Do not move pointers or memref operations
  for (Value result : op->getResults()) {
    if (isa<MemRefType>(result.getType()) ||
        isa<LLVM::LLVMPointerType>(result.getType()))
      return false;
  }

  /// If it is memory effect free and speculatable, it is invariant.
  if (!mlir::isPure(op))
    return false;

  /// Walk the nested operations and check that all used values are defined
  /// outside of the EDT region
  auto walkFn = [&](Operation *child) {
    for (Value operand : child->getOperands()) {
      /// Ignore values defined in a nested region
      if (op->isAncestor(operand.getParentRegion()->getParentOp()))
        continue;
      if (!arts::isDefinedOutside(edtRegion, operand))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  return !op->walk(walkFn).wasInterrupted();
}

uint64_t moveEdtInvariantCode(EdtOp edtOp) {
  uint64_t numMoved = 0;
  Region &region = edtOp.getRegion();

  /// Define the callback to actually move the operation.
  auto moveOutOfEdt = [&](Operation *op) { op->moveBefore(edtOp); };

  std::queue<Operation *> worklist;
  /// Add top-level operations in the EDT body to the worklist.
  for (Operation &op : region.getOps())
    worklist.push(&op);

  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop();

    /// Skip ops that have already been moved or processed
    if (op->getParentRegion() != &region)
      continue;

    /// Check if the op should be moved and can be hoisted
    ARTS_INFO("Checking op:\n" << "  - " << *op);
    if (!canBeHoistedFromEdt(region, op))
      continue;

    ARTS_INFO("  - Moving EDT-invariant op");
    moveOutOfEdt(op);
    ++numMoved;

    /// Since the op has been moved, we need to check its users within the
    /// top-level of the edt region.
    for (Operation *user : op->getUsers())
      if (user->getParentRegion() == &region)
        worklist.push(user);
  }

  return numMoved;
}

} // namespace carts::arts
} // namespace mlir
