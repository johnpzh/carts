///==========================================================================///
/// File: LoopNode.cpp
///
/// Implementation of LoopNode for SCF loop operations with integrated metadata.
///==========================================================================///

#include "carts/dialect/arts/Analysis/loop/LoopNode.h"
#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "carts/utils/LoopUtils.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

static Value getWhileInductionVar(scf::WhileOp whileOp) {
  if (!whileOp)
    return Value();

  Block &before = whileOp.getBefore().front();
  if (before.getNumArguments() == 0)
    return Value();

  auto condition = dyn_cast<scf::ConditionOp>(before.getTerminator());
  if (!condition)
    return before.getArgument(0);

  Value cond = condition.getCondition();
  for (BlockArgument arg : before.getArguments()) {
    SmallVector<Value> bounds;
    collectWhileBounds(cond, arg, bounds);
    if (!bounds.empty())
      return arg;
  }

  return before.getArgument(0);
}

static Value getPrimaryLoopInductionVar(LoopLikeOpInterface loopLike) {
  if (auto ivs = loopLike.getLoopInductionVars(); ivs && !ivs->empty())
    return ivs->front();
  return Value();
}

static Value getPrimaryLoopLowerBound(LoopLikeOpInterface loopLike) {
  if (auto bounds = loopLike.getLoopLowerBounds(); bounds && !bounds->empty())
    return ValueAnalysis::getValueFromFoldResult(bounds->front());
  return Value();
}

static Value getPrimaryLoopUpperBound(LoopLikeOpInterface loopLike) {
  if (auto bounds = loopLike.getLoopUpperBounds(); bounds && !bounds->empty())
    return ValueAnalysis::getValueFromFoldResult(bounds->front());
  return Value();
}

static Value getPrimaryLoopStep(LoopLikeOpInterface loopLike) {
  if (auto steps = loopLike.getLoopSteps(); steps && !steps->empty())
    return ValueAnalysis::getValueFromFoldResult(steps->front());
  return Value();
}

static std::optional<int64_t>
getPrimaryLoopBoundConstant(LoopLikeOpInterface loopLike, bool upperBound) {
  auto bounds = upperBound ? loopLike.getLoopUpperBounds()
                           : loopLike.getLoopLowerBounds();
  if (!bounds || bounds->empty())
    return std::nullopt;
  return ValueAnalysis::getConstantIndex(bounds->front());
}

static std::optional<int64_t>
getPrimaryLoopStepConstant(LoopLikeOpInterface loopLike) {
  auto steps = loopLike.getLoopSteps();
  if (!steps || steps->empty())
    return std::nullopt;
  return ValueAnalysis::getConstantIndex(steps->front());
}

static Region *getPrimaryLoopRegion(Operation *loopOp) {
  if (!loopOp)
    return nullptr;
  if (auto scfFor = dyn_cast<scf::ForOp>(loopOp))
    return &scfFor.getRegion();
  if (auto affineFor = dyn_cast<affine::AffineForOp>(loopOp))
    return &affineFor.getRegion();
  if (auto loopLike = dyn_cast<LoopLikeOpInterface>(loopOp)) {
    SmallVector<Region *> regions = loopLike.getLoopRegions();
    if (regions.size() == 1)
      return regions.front();
  }
  return nullptr;
}

static bool haveEquivalentFoldResults(OpFoldResult lhs, OpFoldResult rhs) {
  auto lhsValue = llvm::dyn_cast<Value>(lhs);
  auto rhsValue = llvm::dyn_cast<Value>(rhs);
  if (lhsValue && rhsValue)
    return ValueAnalysis::sameValue(lhsValue, rhsValue);
  auto lhsAttr = llvm::dyn_cast<Attribute>(lhs);
  auto rhsAttr = llvm::dyn_cast<Attribute>(rhs);
  return lhsAttr && rhsAttr && lhsAttr == rhsAttr;
}

static bool isTransparentNestedWorkOp(Operation *op) {
  if (!op || op->hasTrait<OpTrait::IsTerminator>())
    return true;
  if (isa<LoopLikeOpInterface, affine::AffineForOp>(op))
    return true;
  return op->getNumRegions() == 0 && mlir::isPure(op);
}

static Operation *getSingleDirectNestedLoop(Operation *loopOp) {
  if (!loopOp)
    return nullptr;

  Region *bodyRegion = getPrimaryLoopRegion(loopOp);
  if (!bodyRegion)
    return nullptr;

  if (!bodyRegion || bodyRegion->empty())
    return nullptr;

  Operation *nestedLoop = nullptr;
  for (Operation &op : bodyRegion->front().without_terminator()) {
    if (isa<LoopLikeOpInterface, affine::AffineForOp>(op)) {
      if (nestedLoop)
        return nullptr;
      nestedLoop = &op;
      continue;
    }
    if (!isTransparentNestedWorkOp(&op))
      return nullptr;
  }
  return nestedLoop;
}

LoopNode::LoopNode(Operation *loopOp, LoopAnalysis *loopAnalysis)
    : loopOp(loopOp), loopAnalysis(loopAnalysis) {
  hierId = "loop@" + std::to_string(reinterpret_cast<uintptr_t>(loopOp));
}

void LoopNode::print(llvm::raw_ostream &os) const {
  os << "LoopNode(" << hierId << ")";

  llvm::TypeSwitch<Operation *>(loopOp)
      .Case<scf::ForOp>([&](auto) { os << " scf.for"; })
      .Case<scf::ParallelOp>([&](auto) { os << " scf.parallel"; })
      .Case<scf::WhileOp>([&](auto) { os << " scf.while"; });
  os << "\n";
}

///===----------------------------------------------------------------------===///
/// Induction Variable Analysis Implementation
///===----------------------------------------------------------------------===///

Value LoopNode::getInductionVar() const {
  return llvm::TypeSwitch<Operation *, Value>(loopOp)
      .Case<scf::ForOp>([](auto op) { return op.getInductionVar(); })
      .Case<scf::ParallelOp>([](auto op) -> Value {
        return op.getInductionVars().empty() ? Value()
                                             : op.getInductionVars()[0];
      })
      .Case<scf::WhileOp>([](auto op) { return getWhileInductionVar(op); })
      .Default([](Operation *op) {
        if (auto loopLike = dyn_cast<LoopLikeOpInterface>(op))
          return getPrimaryLoopInductionVar(loopLike);
        return Value();
      });
}

bool LoopNode::dependsOnInductionVar(Value v) {
  /// Check cache first
  auto it = ivDepCache.find(v);
  if (it != ivDepCache.end())
    return it->second;

  Value iv = getInductionVar();
  if (!iv) {
    ivDepCache[v] = false;
    return false;
  }

  /// Direct match
  if (v == iv) {
    ivDepCache[v] = true;
    return true;
  }

  /// Walk def-use chain
  Operation *defOp = v.getDefiningOp();
  if (!defOp) {
    ivDepCache[v] = false;
    return false;
  }

  /// Check operands recursively
  bool depends = llvm::any_of(defOp->getOperands(), [this](Value operand) {
    return dependsOnInductionVar(operand);
  });

  ivDepCache[v] = depends;
  return depends;
}

bool LoopNode::dependsOnInductionVarNormalized(Value v) {
  return dependsOnInductionVar(ValueAnalysis::stripNumericCasts(v));
}

static bool dependsOnLoopInitImpl(Value value, Value base, unsigned depth) {
  if (!value || !base || depth > 8)
    return false;
  if (value == base)
    return true;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(parentOp)) {
      if (blockArg == forOp.getInductionVar())
        return dependsOnLoopInitImpl(forOp.getLowerBound(), base, depth + 1);
    } else if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(parentOp)) {
      unsigned idx = blockArg.getArgNumber();
      if (idx < whileOp.getInits().size())
        return dependsOnLoopInitImpl(whileOp.getInits()[idx], base, depth + 1);
    } else if (auto loopLike =
                   dyn_cast_or_null<LoopLikeOpInterface>(parentOp)) {
      unsigned idx = blockArg.getArgNumber();
      if (auto lbs = loopLike.getLoopLowerBounds(); lbs && idx < lbs->size()) {
        if (Value lb = ValueAnalysis::getValueFromFoldResult((*lbs)[idx]))
          return dependsOnLoopInitImpl(lb, base, depth + 1);
      }
    } else if (auto parallelOp = dyn_cast_or_null<scf::ParallelOp>(parentOp)) {
      auto ivs = parallelOp.getInductionVars();
      for (auto it : llvm::enumerate(ivs)) {
        if (blockArg == it.value()) {
          auto lbs = parallelOp.getLowerBound();
          if (it.index() < lbs.size())
            return dependsOnLoopInitImpl(lbs[it.index()], base, depth + 1);
          break;
        }
      }
    }
    return false;
  }

  Operation *op = value.getDefiningOp();
  if (!op)
    return false;

  if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
           arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::IndexCastOp,
           arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp,
           arith::TruncIOp>(op))
    return false;

  for (Value operand : op->getOperands())
    if (dependsOnLoopInitImpl(operand, base, depth + 1))
      return true;

  return false;
}

bool LoopNode::dependsOnLoopInit(Value value, Value base) {
  return dependsOnLoopInitImpl(value, base, 0);
}

bool LoopNode::dependsOnLoopInitNormalized(Value value, Value base) {
  Value vStripped = ValueAnalysis::stripNumericCasts(value);
  Value bStripped = ValueAnalysis::stripNumericCasts(base);
  return dependsOnLoopInitImpl(vStripped, bStripped, 0);
}

bool LoopNode::isValueLoopInvariant(Value v) {
  if (!loopOp)
    return false;
  v = ValueAnalysis::stripNumericCasts(v);

  llvm::SmallPtrSet<Value, 16> visited;
  std::function<bool(Value)> isInvariant = [&](Value val) -> bool {
    if (!val)
      return false;
    if (!visited.insert(val).second)
      return true;

    if (ValueAnalysis::isValueConstant(val))
      return true;

    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      Block *owner = blockArg.getOwner();
      if (!owner)
        return false;
      Operation *parentOp = owner->getParentOp();
      if (parentOp && (parentOp == loopOp || loopOp->isAncestor(parentOp)))
        return false;
      return true;
    }

    Operation *defOp = val.getDefiningOp();
    if (!defOp)
      return false;

    if (!loopOp->isAncestor(defOp)) {
      for (Value operand : defOp->getOperands())
        if (!isInvariant(operand))
          return false;
      return true;
    }

    if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
             arith::DivUIOp, arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp,
             arith::TruncIOp>(defOp))
      return false;

    for (Value operand : defOp->getOperands())
      if (!isInvariant(operand))
        return false;
    return true;
  };

  return isInvariant(v);
}

LoopNode::IVExpr LoopNode::analyzeIndexExpr(Value index) {
  IVExpr result;
  Value iv = getInductionVar();

  if (!iv || !dependsOnInductionVar(index))
    return result;

  result.dependsOnIV = true;

  /// Direct IV use: offset=0, multiplier=1
  if (index == iv) {
    result.offset = 0;
    result.multiplier = 1;
    return result;
  }

  Operation *defOp = index.getDefiningOp();
  if (!defOp)
    return result;

  /// Handle arith::AddIOp: iv + constant or constant + iv
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    if (lhs == iv) {
      if (auto constVal = ValueAnalysis::getConstantValue(rhs)) {
        result.offset = *constVal;
        result.multiplier = 1;
      }
    } else if (rhs == iv) {
      if (auto constVal = ValueAnalysis::getConstantValue(lhs)) {
        result.offset = *constVal;
        result.multiplier = 1;
      }
    }
    return result;
  }

  /// Handle arith::SubIOp: iv - constant
  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    if (subOp.getLhs() == iv) {
      if (auto constVal = ValueAnalysis::getConstantValue(subOp.getRhs())) {
        result.offset = -(*constVal);
        result.multiplier = 1;
      }
    }
    return result;
  }

  /// Handle arith::MulIOp: iv * constant or constant * iv
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();

    if (lhs == iv) {
      if (auto constVal = ValueAnalysis::getConstantValue(rhs)) {
        result.offset = 0;
        result.multiplier = *constVal;
      }
    } else if (rhs == iv) {
      if (auto constVal = ValueAnalysis::getConstantValue(lhs)) {
        result.offset = 0;
        result.multiplier = *constVal;
      }
    }
    return result;
  }
  return result;
}

std::optional<int64_t> LoopNode::getLowerBoundConstant() const {
  return llvm::TypeSwitch<Operation *, std::optional<int64_t>>(loopOp)
      .Case<scf::ForOp>([](auto op) -> std::optional<int64_t> {
        if (auto constOp =
                op.getLowerBound()
                    .template getDefiningOp<arith::ConstantIndexOp>())
          return constOp.value();
        return std::nullopt;
      })
      .Case<scf::WhileOp>([this](auto op) -> std::optional<int64_t> {
        Value iv = getInductionVar();
        auto arg = dyn_cast<BlockArgument>(iv);
        if (!arg)
          return std::nullopt;
        unsigned argIndex = arg.getArgNumber();
        if (argIndex >= op.getInits().size())
          return std::nullopt;
        Value init = ValueAnalysis::stripNumericCasts(op.getInits()[argIndex]);
        return ValueAnalysis::getConstantValue(init);
      })
      .Default([](Operation *op) -> std::optional<int64_t> {
        if (auto loopLike = dyn_cast<LoopLikeOpInterface>(op))
          return getPrimaryLoopBoundConstant(loopLike, /*upperBound=*/false);
        return std::nullopt;
      });
}

std::optional<int64_t> LoopNode::getUpperBoundConstant() const {
  return llvm::TypeSwitch<Operation *, std::optional<int64_t>>(loopOp)
      .Case<scf::ForOp>([](auto op) -> std::optional<int64_t> {
        if (auto constOp =
                op.getUpperBound()
                    .template getDefiningOp<arith::ConstantIndexOp>())
          return constOp.value();
        return std::nullopt;
      })
      .Case<scf::WhileOp>([this](auto op) -> std::optional<int64_t> {
        Value iv = getInductionVar();
        auto arg = dyn_cast<BlockArgument>(iv);
        if (!arg)
          return std::nullopt;

        Block &before = op.getBefore().front();
        auto condition = dyn_cast<scf::ConditionOp>(before.getTerminator());
        if (!condition)
          return std::nullopt;

        SmallVector<Value> bounds;
        collectWhileBounds(condition.getCondition(), arg, bounds);
        if (bounds.empty())
          return std::nullopt;

        std::optional<int64_t> minBound;
        for (Value bound : bounds) {
          Value stripped = ValueAnalysis::stripNumericCasts(bound);
          auto cst = ValueAnalysis::getConstantValue(stripped);
          if (!cst)
            return std::nullopt;
          if (!minBound || *cst < *minBound)
            minBound = *cst;
        }
        return minBound;
      })
      .Default([](Operation *op) -> std::optional<int64_t> {
        if (auto loopLike = dyn_cast<LoopLikeOpInterface>(op))
          return getPrimaryLoopBoundConstant(loopLike, /*upperBound=*/true);
        return std::nullopt;
      });
}

std::optional<int64_t> LoopNode::getStepConstant() const {
  return llvm::TypeSwitch<Operation *, std::optional<int64_t>>(loopOp)
      .Case<scf::ForOp>([](auto op) -> std::optional<int64_t> {
        if (auto constOp =
                op.getStep().template getDefiningOp<arith::ConstantIndexOp>())
          return constOp.value();
        return std::nullopt;
      })
      .Default([](Operation *op) -> std::optional<int64_t> {
        if (auto loopLike = dyn_cast<LoopLikeOpInterface>(op))
          return getPrimaryLoopStepConstant(loopLike);
        return std::nullopt;
      });
}

std::optional<int64_t>
LoopNode::estimateStaticPerfectNestedWork(int64_t cap) const {
  if (!loopAnalysis)
    return std::nullopt;

  int64_t capped = std::max<int64_t>(1, cap);
  int64_t product = 1;
  Operation *current = loopOp;

  while (Operation *nested = getSingleDirectNestedLoop(current)) {
    auto tripCount = loopAnalysis->getStaticTripCount(nested);
    if (!tripCount || *tripCount <= 0)
      return product;

    if (product >= capped)
      return capped;
    if (*tripCount >= capped / product)
      return capped;

    product *= *tripCount;
    current = nested;
  }

  return product;
}

Value LoopNode::getLowerBound() const {
  return llvm::TypeSwitch<Operation *, Value>(loopOp)
      .Case<scf::ForOp>([](auto op) { return op.getLowerBound(); })
      .Default([](Operation *op) {
        if (auto loopLike = dyn_cast<LoopLikeOpInterface>(op))
          return getPrimaryLoopLowerBound(loopLike);
        return Value();
      });
}

Value LoopNode::getUpperBound() const {
  return llvm::TypeSwitch<Operation *, Value>(loopOp)
      .Case<scf::ForOp>([](auto op) { return op.getUpperBound(); })
      .Default([](Operation *op) {
        if (auto loopLike = dyn_cast<LoopLikeOpInterface>(op))
          return getPrimaryLoopUpperBound(loopLike);
        return Value();
      });
}

Value LoopNode::getStep() const {
  return llvm::TypeSwitch<Operation *, Value>(loopOp)
      .Case<scf::ForOp>([](auto op) { return op.getStep(); })
      .Default([](Operation *op) {
        if (auto loopLike = dyn_cast<LoopLikeOpInterface>(op))
          return getPrimaryLoopStep(loopLike);
        return Value();
      });
}

int LoopNode::getNestingDepth() const {
  int depth = 0;
  for (Operation *parent = loopOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<LoopLikeOpInterface, scf::WhileOp, affine::AffineForOp>(parent))
      ++depth;
  }
  return depth;
}

///===----------------------------------------------------------------------===///
/// Loop Classification Methods
///===----------------------------------------------------------------------===///

bool LoopNode::isInnermostLoop() const {
  if (!loopOp)
    return false;

  bool hasNested = false;
  loopOp->walk([&](Operation *op) {
    if (op == loopOp)
      return;
    if (isa<LoopLikeOpInterface, scf::WhileOp>(op))
      hasNested = true;
  });
  return !hasNested;
}

bool LoopNode::hasEdt() const {
  if (!loopOp)
    return false;

  bool foundEdt = false;
  loopOp->walk([&](EdtOp) { foundEdt = true; });
  return foundEdt;
}

bool LoopNode::hasDistributedDbContract() const {
  return loopAnalysis && loopOp &&
         loopAnalysis->operationHasDistributedDbContract(loopOp);
}

bool LoopNode::hasPeerInferredPartitionDims() const {
  return loopAnalysis && loopOp &&
         loopAnalysis->operationHasPeerInferredPartitionDims(loopOp);
}

bool LoopNode::haveCompatibleBounds(LoopNode *a, LoopNode *b) {
  if (!a || !b)
    return false;

  Operation *opA = a->getLoopOp();
  Operation *opB = b->getLoopOp();

  return llvm::TypeSwitch<Operation *, bool>(opA)
      .Case<scf::ForOp>([&](auto forA) {
        auto forB = dyn_cast<scf::ForOp>(opB);
        if (!forB)
          return false;
        return ValueAnalysis::sameValue(forA.getLowerBound(),
                                        forB.getLowerBound()) &&
               ValueAnalysis::sameValue(forA.getUpperBound(),
                                        forB.getUpperBound()) &&
               ValueAnalysis::sameValue(forA.getStep(), forB.getStep());
      })
      .Default([&](Operation *genericA) {
        auto loopA = dyn_cast<LoopLikeOpInterface>(genericA);
        auto loopB = dyn_cast<LoopLikeOpInterface>(opB);
        if (!loopA || !loopB)
          return false;

        auto lbA = loopA.getLoopLowerBounds();
        auto lbB = loopB.getLoopLowerBounds();
        auto ubA = loopA.getLoopUpperBounds();
        auto ubB = loopB.getLoopUpperBounds();
        auto stepA = loopA.getLoopSteps();
        auto stepB = loopB.getLoopSteps();
        if (!lbA || !lbB || !ubA || !ubB || !stepA || !stepB ||
            lbA->size() != lbB->size() || ubA->size() != ubB->size() ||
            stepA->size() != stepB->size())
          return false;

        for (auto [lhs, rhs] : llvm::zip(*lbA, *lbB)) {
          if (!haveEquivalentFoldResults(lhs, rhs))
            return false;
        }
        for (auto [lhs, rhs] : llvm::zip(*ubA, *ubB)) {
          if (!haveEquivalentFoldResults(lhs, rhs))
            return false;
        }
        for (auto [lhs, rhs] : llvm::zip(*stepA, *stepB)) {
          if (!haveEquivalentFoldResults(lhs, rhs))
            return false;
        }
        return true;
      });
}
