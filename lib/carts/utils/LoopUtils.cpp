///==========================================================================///
/// File: LoopUtils.cpp
///
/// Implementation of non-inline loop utility functions.
///==========================================================================///

#include "carts/utils/LoopUtils.h"
#include "carts/dialect/arts/Analysis/loop/LoopNode.h"
#include "carts/dialect/arts/Utils/BlockedAccessUtils.h"
#include "carts/utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringRef.h"
#include <limits>

namespace mlir {
namespace arts {

Value getLoopInductionVar(Operation *op) {
  if (!op)
    return Value();

  if (auto scfFor = dyn_cast<scf::ForOp>(op))
    return scfFor.getInductionVar();

  if (auto loopLike = dyn_cast<LoopLikeOpInterface>(op)) {
    if (auto ivs = loopLike.getLoopInductionVars(); ivs && !ivs->empty())
      return ivs->front();
  }

  return Value();
}

bool isProvablyZeroLoopLowerBound(Value lb) {
  lb = ValueAnalysis::stripNumericCasts(lb);
  if (ValueAnalysis::isZeroConstant(lb))
    return true;

  auto select = lb.getDefiningOp<arith::SelectOp>();
  if (!select)
    return false;

  Value trueVal = ValueAnalysis::stripNumericCasts(select.getTrueValue());
  Value falseVal = ValueAnalysis::stripNumericCasts(select.getFalseValue());
  auto cmp = ValueAnalysis::stripNumericCasts(select.getCondition())
                 .getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return false;

  Value lhs = ValueAnalysis::stripNumericCasts(cmp.getLhs());
  Value rhs = ValueAnalysis::stripNumericCasts(cmp.getRhs());
  auto pred = cmp.getPredicate();

  auto matchesZeroClamp = [&](Value zeroArm, Value otherArm) {
    if (!ValueAnalysis::isZeroConstant(zeroArm) ||
        !arts::isKnownNonPositive(otherArm))
      return false;
    return ((pred == arith::CmpIPredicate::slt ||
             pred == arith::CmpIPredicate::ult) &&
            ValueAnalysis::sameValue(lhs, otherArm) &&
            ValueAnalysis::isZeroConstant(rhs)) ||
           ((pred == arith::CmpIPredicate::sgt ||
             pred == arith::CmpIPredicate::ugt) &&
            ValueAnalysis::sameValue(rhs, otherArm) &&
            ValueAnalysis::isZeroConstant(lhs));
  };

  return matchesZeroClamp(trueVal, falseVal) ||
         matchesZeroClamp(falseVal, trueVal);
}

bool isLoopFullRange(LoopNode *loop, Value dimSize) {
  if (!loop || !dimSize)
    return false;

  Value lb = loop->getLowerBound();
  Value step = loop->getStep();
  Value ub = loop->getUpperBound();
  if (!lb || !step || !ub)
    return false;

  if (!isProvablyZeroLoopLowerBound(lb))
    return false;
  if (!ValueAnalysis::isOneConstant(ValueAnalysis::stripNumericCasts(step)))
    return false;
  return arts::areEquivalentOwnedSliceExtents(ub, dimSize);
}

static std::optional<int64_t> getStaticTripCount(scf::ForOp loop) {
  if (!loop)
    return std::nullopt;
  std::optional<llvm::APInt> tripCount = loop.getStaticTripCount();
  if (!tripCount || !tripCount->isSignedIntN(64))
    return std::nullopt;
  return tripCount->getSExtValue();
}

static std::optional<int64_t> getStaticTripCount(LoopLikeOpInterface loop) {
  auto lowerBounds = loop.getLoopLowerBounds();
  auto upperBounds = loop.getLoopUpperBounds();
  auto steps = loop.getLoopSteps();
  if (!lowerBounds || !upperBounds || !steps || lowerBounds->size() != 1 ||
      upperBounds->size() != 1 || steps->size() != 1)
    return std::nullopt;

  auto lowerBound = ValueAnalysis::getConstantIndex(lowerBounds->front());
  auto upperBound = ValueAnalysis::getConstantIndex(upperBounds->front());
  auto step = ValueAnalysis::getConstantIndex(steps->front());
  if (!lowerBound || !upperBound || !step || *step <= 0)
    return std::nullopt;

  int64_t span = std::max<int64_t>(0, *upperBound - *lowerBound);
  return (span + *step - 1) / *step;
}

static std::optional<int64_t> getStaticTripCount(affine::AffineForOp loop) {
  if (!loop.hasConstantLowerBound() || !loop.hasConstantUpperBound())
    return std::nullopt;
  int64_t step = loop.getStep().getSExtValue();
  if (step <= 0)
    return std::nullopt;
  int64_t span = std::max<int64_t>(0, loop.getConstantUpperBound() -
                                          loop.getConstantLowerBound());
  return (span + step - 1) / step;
}

static std::optional<int64_t> getStaticTripCount(omp::WsloopOp loop) {
  if (!loop)
    return std::nullopt;

  auto loopNest = dyn_cast_or_null<omp::LoopNestOp>(loop.getWrappedLoop());
  if (!loopNest)
    return std::nullopt;

  auto lowerBounds = loopNest.getLoopLowerBounds();
  auto upperBounds = loopNest.getLoopUpperBounds();
  auto steps = loopNest.getLoopSteps();
  if (lowerBounds.empty() || lowerBounds.size() != upperBounds.size() ||
      lowerBounds.size() != steps.size())
    return std::nullopt;

  auto lowerBound = ValueAnalysis::tryFoldConstantIndex(lowerBounds.front());
  auto upperBound = ValueAnalysis::tryFoldConstantIndex(upperBounds.front());
  auto step = ValueAnalysis::tryFoldConstantIndex(steps.front());
  if (!lowerBound || !upperBound || !step || *step <= 0)
    return std::nullopt;

  int64_t span = std::max<int64_t>(0, *upperBound - *lowerBound);
  return (span + *step - 1) / *step;
}

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

bool containsLoop(EdtOp edt) {
  bool found = false;
  edt.getBody().walk([&](Operation *op) {
    if (op != edt.getOperation() &&
        isa<LoopLikeOpInterface, affine::AffineForOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

std::optional<int64_t> getStaticTripCount(Operation *loopOp) {
  if (!loopOp)
    return std::nullopt;
  if (auto scfFor = dyn_cast<scf::ForOp>(loopOp))
    return getStaticTripCount(scfFor);
  if (auto affineFor = dyn_cast<affine::AffineForOp>(loopOp))
    return getStaticTripCount(affineFor);
  if (auto wsloop = dyn_cast<omp::WsloopOp>(loopOp))
    return getStaticTripCount(wsloop);
  if (auto loopLike = dyn_cast<LoopLikeOpInterface>(loopOp))
    return getStaticTripCount(loopLike);
  return std::nullopt;
}

int64_t getRepeatedParentTripProduct(Operation *op, int64_t maxProduct) {
  if (!op)
    return 1;

  int64_t product = 1;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<func::FuncOp>(parent))
      break;
    auto tripOpt = getStaticTripCount(parent);
    if (!tripOpt || *tripOpt <= 1)
      continue;
    if (*tripOpt >= maxProduct / product)
      return maxProduct;
    product *= *tripOpt;
    if (product >= maxProduct)
      return maxProduct;
  }
  return product;
}

bool hasFloatingPointType(Type type) {
  if (!type)
    return false;
  if (type.isF16() || type.isBF16() || type.isF32() || type.isF64() ||
      type.isF80() || type.isF128())
    return true;
  if (auto vectorType = dyn_cast<VectorType>(type))
    return hasFloatingPointType(vectorType.getElementType());
  return false;
}

bool operationTouchesFloatingPoint(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (hasFloatingPointType(operand.getType()))
      return true;
  }
  for (Value result : op->getResults()) {
    if (hasFloatingPointType(result.getType()))
      return true;
  }
  return false;
}

} // namespace arts
} // namespace mlir
