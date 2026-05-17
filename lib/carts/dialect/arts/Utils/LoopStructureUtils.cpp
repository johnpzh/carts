///==========================================================================///
/// File: LoopStructureUtils.cpp
///
/// ARTS loop-structure helpers used by DB/EDT analysis and transforms.
///==========================================================================///

#include "carts/dialect/arts/Utils/LoopStructureUtils.h"

#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
namespace carts::arts {

unsigned getLoopDepth(Operation *op) {
  unsigned depth = 0;
  for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
       parent = parent->getParentOp()) {
    if (isa<LoopLikeOpInterface>(parent) || isa<omp::WsloopOp>(parent))
      ++depth;
  }
  return depth;
}

void collectWhileBounds(Value cond, Value iterArg, SmallVector<Value> &bounds) {
  if (!cond)
    return;
  cond = ValueAnalysis::stripNumericCasts(cond);
  if (auto andOp = cond.getDefiningOp<arith::AndIOp>()) {
    collectWhileBounds(andOp.getLhs(), iterArg, bounds);
    collectWhileBounds(andOp.getRhs(), iterArg, bounds);
    return;
  }
  if (auto cmp = cond.getDefiningOp<arith::CmpIOp>()) {
    auto lhs = ValueAnalysis::stripNumericCasts(cmp.getLhs());
    auto rhs = ValueAnalysis::stripNumericCasts(cmp.getRhs());
    auto pred = cmp.getPredicate();
    auto isLessPred =
        pred == arith::CmpIPredicate::slt || pred == arith::CmpIPredicate::ult;
    if (lhs == iterArg && isLessPred) {
      bounds.push_back(cmp.getRhs());
      return;
    }
    if (rhs == iterArg && (pred == arith::CmpIPredicate::sgt ||
                           pred == arith::CmpIPredicate::ugt)) {
      bounds.push_back(cmp.getLhs());
      return;
    }
  }
}

} // namespace carts::arts
} // namespace mlir
