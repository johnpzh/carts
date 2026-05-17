///==========================================================================///
/// File: LoopInvarianceUtils.cpp
///
/// ARTS loop invariance checking and hoisting safety utilities.
///==========================================================================///

#include "carts/dialect/arts/Utils/LoopInvarianceUtils.h"

#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace carts::arts {

bool isLoopInvariant(scf::ForOp loop, Value v) {
  if (!v)
    return true;
  if (loop.isDefinedOutsideOfLoop(v))
    return true;
  Value stripped = ValueAnalysis::stripNumericCasts(v);
  return ValueAnalysis::isValueConstant(stripped);
}

bool isSafeToHoistDivRem(scf::ForOp loop, Operation *op) {
  if (auto div = dyn_cast<arith::DivUIOp>(op)) {
    Value denom = div.getRhs();
    return isLoopInvariant(loop, denom) &&
           ValueAnalysis::isProvablyNonZero(denom);
  }
  if (auto rem = dyn_cast<arith::RemUIOp>(op)) {
    Value denom = rem.getRhs();
    return isLoopInvariant(loop, denom) &&
           ValueAnalysis::isProvablyNonZero(denom);
  }
  return false;
}

bool isSafeDivRemToHoist(Operation *op, scf::ForOp loop,
                         DominanceInfo &domInfo) {
  if (!loop || !loop->isAncestor(op))
    return false;
  Value denom;
  if (auto div = dyn_cast<arith::DivUIOp>(op))
    denom = div.getRhs();
  else if (auto rem = dyn_cast<arith::RemUIOp>(op))
    denom = rem.getRhs();
  else
    return false;

  if (!ValueAnalysis::isProvablyNonZero(denom))
    return false;
  if (!allOperandsDominate(op, loop, domInfo))
    return false;
  return true;
}

bool isDefinedOutside(Region &region, Value value) {
  return !region.isAncestor(value.getParentRegion());
}

bool allOperandsDefinedOutside(Operation *op, Region &region) {
  for (Value operand : op->getOperands()) {
    if (!isDefinedOutside(region, operand))
      return false;
  }
  return true;
}

bool allOperandsDominate(Operation *op, Operation *insertionPoint,
                         DominanceInfo &domInfo) {
  for (Value operand : op->getOperands()) {
    if (!domInfo.properlyDominates(operand, insertionPoint))
      return false;
  }
  return true;
}

scf::ForOp findHoistTarget(Operation *op, Operation *addrOp,
                           DominanceInfo &domInfo) {
  scf::ForOp target = nullptr;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      continue;
    Region &loopRegion = loop.getRegion();
    if (!allOperandsDefinedOutside(addrOp, loopRegion))
      break;
    if (!allOperandsDominate(addrOp, loop, domInfo))
      break;
    target = loop;
  }
  return target;
}

} // namespace carts::arts
} // namespace mlir
