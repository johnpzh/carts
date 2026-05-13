///==========================================================================///
/// File: IterationSpaceDecomposition.cpp
///
/// Decompose scheduling-unit iteration spaces into interior and boundary
/// regions so downstream SDE passes see a branch-free interior loop that
/// can be tiled, distributed, and classified without control-flow noise.
///
/// Current decomposition:
///   Neighborhood boundary peeling — splits an inner loop that guards
///   boundary rows/columns via if/else into separate boundary loops and a
///   branch-free interior loop.
///
///   Before:
///     for i:
///       for j:
///         if (boundary(i,j)) boundary(...)
///         else               interior(...)
///
///   After:
///     for i:
///       if (boundary_row(i))
///         for j: boundary(...)
///       else
///         for j = 0..1:       boundary(...)
///         for j = 1..M-1:     interior(...)   // branch-free
///         for j = M-1..M:     boundary(...)
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_ITERATIONSPACEDECOMPOSITION
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/Debug.h"
#include "arts/utils/Utils.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

ARTS_DEBUG_SETUP(iteration_space_decomposition);

using namespace mlir;
using namespace mlir::arts;

namespace {

struct BoundaryPeelingMatch {
  scf::ForOp innerLoop;
  scf::IfOp finalIf;
  Value rowIsFirst;
  Value rowIsLast;
  int64_t lowerConst = 0;
  int64_t upperConst = 0;
  SmallVector<Operation *> preludeOps;
};

static bool yieldsConstantBool(Block &block, bool expected) {
  auto yield = dyn_cast<scf::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  return ValueAnalysis::isConstantBool(yield.getOperand(0), expected);
}

static bool matchEqConst(Value condition, Value value, int64_t expectedConst) {
  auto cmp = condition.getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq)
    return false;

  value = ValueAnalysis::stripNumericCasts(value);
  Value lhs = ValueAnalysis::stripNumericCasts(cmp.getLhs());
  Value rhs = ValueAnalysis::stripNumericCasts(cmp.getRhs());
  auto lhsConst = ValueAnalysis::getConstantIndexStripped(lhs);
  auto rhsConst = ValueAnalysis::getConstantIndexStripped(rhs);
  if (lhs == value && rhsConst && *rhsConst == expectedConst)
    return true;
  if (rhs == value && lhsConst && *lhsConst == expectedConst)
    return true;
  return false;
}

static bool branchHasStencilBody(Block &block) {
  int loadCount = 0;
  int storeCount = 0;

  block.walk([&](Operation *op) {
    if (isa<scf::ForOp, scf::IfOp>(op))
      return WalkResult::advance();
    if (isa<memref::LoadOp>(op))
      loadCount++;
    if (isa<memref::StoreOp>(op))
      storeCount++;
    return WalkResult::advance();
  });

  return loadCount >= 4 && storeCount >= 1;
}

static bool branchHasBoundaryBody(Block &block) {
  int storeCount = 0;
  block.walk([&](Operation *op) {
    if (isa<memref::StoreOp>(op))
      storeCount++;
    return WalkResult::advance();
  });
  return storeCount >= 1;
}

static bool matchBoundaryPattern(scf::ForOp loop, BoundaryPeelingMatch &match) {
  auto fail = [&](llvm::StringRef reason) {
    ARTS_DEBUG("IterationSpaceDecomposition no-match:" << reason);
    return false;
  };
  match = {};
  if (!loop || loop.getNumResults() != 0)
    return fail("loop missing or has results");

  auto lowerConst = ValueAnalysis::tryFoldConstantIndex(loop.getLowerBound());
  auto upperConst = ValueAnalysis::tryFoldConstantIndex(loop.getUpperBound());
  auto stepConst = ValueAnalysis::tryFoldConstantIndex(loop.getStep());
  if (!lowerConst || !upperConst || !stepConst || *stepConst != 1)
    return fail("bounds are not constant step-1");
  if ((*upperConst - *lowerConst) <= 2)
    return fail("trip count too small");

  SmallVector<Operation *> ops;
  for (Operation &op : loop.getBody()->without_terminator())
    ops.push_back(&op);
  if (ops.empty())
    return fail("loop body is empty");

  auto finalIf = dyn_cast<scf::IfOp>(ops.back());
  if (!finalIf || finalIf.getNumResults() != 0 || !finalIf.elseBlock())
    return fail("missing final if");
  if (!branchHasBoundaryBody(finalIf.getThenRegion().front()) ||
      !branchHasStencilBody(finalIf.getElseRegion().front()))
    return fail("final if branches do not look like boundary/stencil bodies");

  Value lastCondValue = finalIf.getCondition();
  auto lastCondIf = lastCondValue.getDefiningOp<scf::IfOp>();
  if (!lastCondIf || lastCondIf.getNumResults() != 1 || !lastCondIf.elseBlock())
    return fail("final if condition is not the second boundary if");
  if (!yieldsConstantBool(lastCondIf.getThenRegion().front(), true))
    return fail("second boundary if then-branch is not constant true");

  auto mergedOr = lastCondIf.getCondition().getDefiningOp<arith::OrIOp>();
  if (!mergedOr)
    return fail("second boundary predicate is not an or");

  auto andOp = mergedOr.getRhs().getDefiningOp<arith::AndIOp>();
  Value firstCondValue = mergedOr.getLhs();
  if (!andOp) {
    andOp = mergedOr.getLhs().getDefiningOp<arith::AndIOp>();
    firstCondValue = mergedOr.getRhs();
  }
  if (!andOp)
    return fail("second boundary predicate is missing and arm");

  auto firstCondIf = firstCondValue.getDefiningOp<scf::IfOp>();
  if (!firstCondIf || firstCondIf.getNumResults() != 1 ||
      !firstCondIf.elseBlock())
    return fail("first boundary predicate is not an if");
  if (!yieldsConstantBool(firstCondIf.getThenRegion().front(), true))
    return fail("first boundary if then-branch is not constant true");

  Value rowIsFirst = firstCondIf.getCondition();
  Value rowIsLast = Value();
  for (Value operand : andOp->getOperands()) {
    if (operand == rowIsFirst)
      continue;
    if (auto xori = operand.getDefiningOp<arith::XOrIOp>()) {
      for (Value xorOperand : xori->getOperands()) {
        if (xorOperand == firstCondIf.getResult(0)) {
          for (Value andOperand : andOp->getOperands()) {
            if (andOperand != operand) {
              rowIsLast = andOperand;
              break;
            }
          }
          break;
        }
      }
    }
  }
  if (!rowIsLast)
    return fail("could not recover row-last predicate");

  Value innerIv = loop.getInductionVar();
  Value innerCompareValue = Value();
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (auto cast = dyn_cast<arith::IndexCastOp>(&op))
      if (cast.getIn() == innerIv) {
        innerCompareValue = cast.getOut();
        break;
      }
  }
  if (!innerCompareValue)
    innerCompareValue = innerIv;

  if (!matchEqConst(
          firstCondIf.getElseRegion().front().getTerminator()->getOperand(0),
          innerCompareValue, *lowerConst))
    return fail("first boundary else compare is not iv == lower");
  if (!matchEqConst(
          lastCondIf.getElseRegion().front().getTerminator()->getOperand(0),
          innerCompareValue, *upperConst - 1))
    return fail("second boundary else compare is not iv == upper-1");

  match.innerLoop = loop;
  match.finalIf = finalIf;
  match.rowIsFirst = rowIsFirst;
  match.rowIsLast = rowIsLast;
  match.lowerConst = *lowerConst;
  match.upperConst = *upperConst;
  match.preludeOps = std::move(ops);
  match.preludeOps.pop_back();
  return true;
}

static SmallVector<Operation *>
collectRequiredPreludeOps(ArrayRef<Operation *> preludeOps, Block &branchBody) {
  llvm::DenseSet<Operation *> preludeSet(preludeOps.begin(), preludeOps.end());
  llvm::DenseSet<Operation *> required;
  SmallVector<Value> worklist;

  auto enqueueValue = [&](Value value) {
    if (!value)
      return;
    if (Operation *def = value.getDefiningOp())
      if (preludeSet.contains(def) && required.insert(def).second)
        worklist.append(def->operand_begin(), def->operand_end());
  };

  for (Operation &op : branchBody.without_terminator())
    for (Value operand : op.getOperands())
      enqueueValue(operand);

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    enqueueValue(value);
  }

  SmallVector<Operation *> ordered;
  for (Operation *op : preludeOps)
    if (required.contains(op))
      ordered.push_back(op);
  return ordered;
}

static void cloneLoopSegment(OpBuilder &builder, BoundaryPeelingMatch &match,
                             Block &branchBody, int64_t lowerConst,
                             int64_t upperConst) {
  if (lowerConst >= upperConst)
    return;

  Location loc = match.innerLoop.getLoc();
  Value lower = arts::createConstantIndex(builder, loc, lowerConst);
  Value upper = arts::createConstantIndex(builder, loc, upperConst);
  auto segment =
      scf::ForOp::create(builder, loc, lower, upper, match.innerLoop.getStep());
  builder.setInsertionPointToStart(segment.getBody());

  IRMapping mapping;
  mapping.map(match.innerLoop.getInductionVar(), segment.getInductionVar());

  for (Operation *op : collectRequiredPreludeOps(match.preludeOps, branchBody))
    builder.clone(*op, mapping);
  for (Operation &op : branchBody.without_terminator())
    builder.clone(op, mapping);

  builder.setInsertionPointAfter(segment);
}

static bool peelBoundaryLoop(BoundaryPeelingMatch &match) {
  Location loc = match.innerLoop.getLoc();
  OpBuilder builder(match.innerLoop);
  auto func = match.innerLoop->getParentOfType<func::FuncOp>();
  if (!func)
    return false;
  DominanceInfo domInfo(func);

  Value rowIsFirst = ValueAnalysis::traceValueToDominating(
      match.rowIsFirst, match.innerLoop, builder, domInfo, loc);
  Value rowIsLast = ValueAnalysis::traceValueToDominating(
      match.rowIsLast, match.innerLoop, builder, domInfo, loc);
  if (!rowIsFirst || !rowIsLast)
    return false;

  Value rowBoundary = arith::OrIOp::create(builder, loc, rowIsFirst, rowIsLast);
  auto splitIf =
      scf::IfOp::create(builder, loc, TypeRange{}, rowBoundary, true);

  builder.setInsertionPointToStart(&splitIf.getThenRegion().front());
  cloneLoopSegment(builder, match, match.finalIf.getThenRegion().front(),
                   match.lowerConst, match.upperConst);

  builder.setInsertionPointToStart(&splitIf.getElseRegion().front());
  cloneLoopSegment(builder, match, match.finalIf.getThenRegion().front(),
                   match.lowerConst, match.lowerConst + 1);
  cloneLoopSegment(builder, match, match.finalIf.getElseRegion().front(),
                   match.lowerConst + 1, match.upperConst - 1);
  cloneLoopSegment(builder, match, match.finalIf.getThenRegion().front(),
                   match.upperConst - 1, match.upperConst);

  match.innerLoop.erase();
  return true;
}

struct IterationSpaceDecompositionPass
    : public arts::impl::IterationSpaceDecompositionBase<
          IterationSpaceDecompositionPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp loop) { loops.push_back(loop); });

    int rewrites = 0;
    for (scf::ForOp loop : loops) {
      if (!loop)
        continue;
      BoundaryPeelingMatch match;
      if (!matchBoundaryPattern(loop, match))
        continue;
      if (peelBoundaryLoop(match)) {
        rewrites++;
        ARTS_INFO("Applied iteration space decomposition");
      }
    }

    ARTS_INFO("IterationSpaceDecomposition: applied " << rewrites
                                                      << " rewrite(s)");
  }
};

} // namespace

namespace mlir::arts::sde {

std::unique_ptr<Pass> createIterationSpaceDecompositionPass() {
  return std::make_unique<IterationSpaceDecompositionPass>();
}

} // namespace mlir::arts::sde
