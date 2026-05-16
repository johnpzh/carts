///==========================================================================///
/// File: ValueAnalysis.cpp
///
/// Static analysis utilities for MLIR Values, constants, and casts.
///==========================================================================///

#include "carts/utils/ValueAnalysis.h"
#include "carts/Dialect.h"
#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/CSE.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <cassert>

namespace mlir {
namespace carts::arts {

static std::optional<bool> proveValuesEqualWithBounds(Value lhs, Value rhs) {
  if (!lhs || !rhs || !lhs.getType().isIndex() || !rhs.getType().isIndex())
    return std::nullopt;
  if (auto equal = ValueBoundsConstraintSet::areEqual(
          ValueBoundsConstraintSet::Variable(lhs),
          ValueBoundsConstraintSet::Variable(rhs));
      succeeded(equal)) {
    return *equal;
  }
  return std::nullopt;
}

static std::optional<bool> proveValueAtLeast(Value value, int64_t minValue) {
  if (!value || !value.getType().isIndex())
    return std::nullopt;
  if (auto lowerBound = ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::LB, ValueBoundsConstraintSet::Variable(value));
      succeeded(lowerBound)) {
    return *lowerBound >= minValue;
  }
  return std::nullopt;
}

static std::optional<bool> proveValueNonZero(Value value) {
  if (!value || !value.getType().isIndex())
    return std::nullopt;
  if (auto atLeastOne = proveValueAtLeast(value, 1); atLeastOne && *atLeastOne)
    return true;
  if (auto upperBound = ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::UB, ValueBoundsConstraintSet::Variable(value),
          /*stopCondition=*/nullptr,
          /*closedUB=*/true);
      succeeded(upperBound)) {
    return *upperBound <= -1;
  }
  return std::nullopt;
}

static std::optional<int64_t>
proveConstantOffsetWithBounds(Value idx, Value loopIV, Value chunkOffset) {
  idx = ValueAnalysis::stripNumericCasts(idx);
  loopIV = ValueAnalysis::stripNumericCasts(loopIV);
  chunkOffset = ValueAnalysis::stripNumericCasts(chunkOffset);

  auto isIndexValue = [](Value v) { return v && v.getType().isIndex(); };
  if (!isIndexValue(idx))
    return std::nullopt;

  SmallVector<Value, 3> operands;
  operands.push_back(idx);

  Builder builder(idx.getContext());
  AffineExpr deltaExpr = builder.getAffineDimExpr(0);
  unsigned dimCount = 1;

  if (isIndexValue(loopIV)) {
    deltaExpr = deltaExpr - builder.getAffineDimExpr(dimCount);
    operands.push_back(loopIV);
    ++dimCount;
  }
  if (isIndexValue(chunkOffset)) {
    deltaExpr = deltaExpr - builder.getAffineDimExpr(dimCount);
    operands.push_back(chunkOffset);
    ++dimCount;
  }

  if (operands.size() == 1)
    return std::nullopt;

  if (auto delta = ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::EQ,
          ValueBoundsConstraintSet::Variable(
              AffineMap::get(dimCount, 0, deltaExpr), operands));
      succeeded(delta)) {
    return *delta;
  }
  return std::nullopt;
}

///===----------------------------------------------------------------------===///
/// Value Cloning Utilities
///===----------------------------------------------------------------------===///

static std::optional<int64_t> tryFoldTypeSizeBytes(polygeist::TypeSizeOp tsOp) {
  auto sourceAttr = tsOp.getSourceAttr();
  if (!sourceAttr)
    return std::nullopt;

  Type sourceType = sourceAttr.getValue();
  if (!sourceType)
    return std::nullopt;

  ModuleOp module = tsOp->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;

  DataLayout layout(module);

  /// Polygeist can emit `typeSize` over memref wrappers while the original C
  /// semantic is `sizeof(pointer)`. Canonicalization does not fold this case
  /// yet, so we treat memrefs as pointer-sized here.
  if (isa<MemRefType>(sourceType)) {
    auto ptrTy = LLVM::LLVMPointerType::get(tsOp.getContext());
    return static_cast<int64_t>(layout.getTypeSize(ptrTy));
  }

  if (isa<IntegerType, FloatType>(sourceType) ||
      LLVM::isCompatibleType(sourceType))
    return static_cast<int64_t>(layout.getTypeSize(sourceType));

  return std::nullopt;
}

static std::optional<int64_t> tryFoldScalarMemrefLoad(memref::LoadOp load,
                                                      unsigned depth) {
  if (!load || !load.getIndices().empty() || depth > 32)
    return std::nullopt;

  Value memref = ValueAnalysis::stripMemrefViewOps(load.getMemRef());
  if (!memref)
    return std::nullopt;

  Operation *memrefDef = memref.getDefiningOp();
  if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(memrefDef))
    return std::nullopt;

  for (Operation *cursor = load->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    auto store = dyn_cast<memref::StoreOp>(cursor);
    if (!store || !store.getIndices().empty())
      continue;

    Value storedMemref = ValueAnalysis::stripMemrefViewOps(store.getMemRef());
    if (storedMemref != memref)
      continue;

    return ValueAnalysis::tryFoldConstantIndex(store.getValue(), depth + 1);
  }

  return std::nullopt;
}

static std::optional<int64_t> tryFoldCmpResult(arith::CmpIOp cmp,
                                               unsigned depth) {
  auto lhs = ValueAnalysis::tryFoldConstantIndex(cmp.getLhs(), depth + 1);
  auto rhs = ValueAnalysis::tryFoldConstantIndex(cmp.getRhs(), depth + 1);

  auto evalCmp = [&](int64_t lhsVal, int64_t rhsVal) -> int64_t {
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return lhsVal == rhsVal;
    case arith::CmpIPredicate::ne:
      return lhsVal != rhsVal;
    case arith::CmpIPredicate::slt:
      return lhsVal < rhsVal;
    case arith::CmpIPredicate::sle:
      return lhsVal <= rhsVal;
    case arith::CmpIPredicate::sgt:
      return lhsVal > rhsVal;
    case arith::CmpIPredicate::sge:
      return lhsVal >= rhsVal;
    case arith::CmpIPredicate::ult:
      return static_cast<uint64_t>(lhsVal) < static_cast<uint64_t>(rhsVal);
    case arith::CmpIPredicate::ule:
      return static_cast<uint64_t>(lhsVal) <= static_cast<uint64_t>(rhsVal);
    case arith::CmpIPredicate::ugt:
      return static_cast<uint64_t>(lhsVal) > static_cast<uint64_t>(rhsVal);
    case arith::CmpIPredicate::uge:
      return static_cast<uint64_t>(lhsVal) >= static_cast<uint64_t>(rhsVal);
    }
    llvm_unreachable("unsupported cmp predicate");
  };

  if (lhs && rhs)
    return evalCmp(*lhs, *rhs);

  Value lhsValue = ValueAnalysis::stripNumericCasts(cmp.getLhs());
  Value rhsValue = ValueAnalysis::stripNumericCasts(cmp.getRhs());
  if (ValueAnalysis::sameValue(lhsValue, rhsValue)) {
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::ule:
    case arith::CmpIPredicate::uge:
      return 1;
    case arith::CmpIPredicate::ne:
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::ugt:
      return 0;
    }
  }

  if (auto rhsConst = ValueAnalysis::tryFoldConstantIndex(rhsValue, depth + 1);
      rhsConst && *rhsConst == 0) {
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::ult:
      return 0;
    case arith::CmpIPredicate::uge:
      return 1;
    default:
      break;
    }
  }

  if (auto lhsConst = ValueAnalysis::tryFoldConstantIndex(lhsValue, depth + 1);
      lhsConst && *lhsConst == 0) {
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::ugt:
      return 0;
    case arith::CmpIPredicate::ule:
      return 1;
    default:
      break;
    }
  }

  return std::nullopt;
}

bool ValueAnalysis::canCloneOperation(
    Operation *op, bool allowMemoryEffectFree,
    llvm::function_ref<bool(Operation *)> extraAllowed) {
  if (!op)
    return false;
  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;
  if (allowMemoryEffectFree && isMemoryEffectFree(op))
    return true;
  return extraAllowed ? extraAllowed(op) : false;
}

bool ValueAnalysis::cloneValuesIntoRegion(
    llvm::SetVector<Value> &values, Region *targetRegion, IRMapping &mapper,
    OpBuilder &builder, bool allowMemoryEffectFree,
    llvm::function_ref<bool(Operation *)> extraAllowed) {
  SmallVector<Value> remaining(values.begin(), values.end());
  bool progress = true;

  while (progress && !remaining.empty()) {
    progress = false;
    SmallVector<Value> nextRemaining;

    for (Value val : remaining) {
      if (mapper.contains(val)) {
        progress = true;
        continue;
      }

      Operation *defOp = val.getDefiningOp();
      if (!defOp) {
        nextRemaining.push_back(val);
        continue;
      }

      if (!canCloneOperation(defOp, allowMemoryEffectFree, extraAllowed)) {
        nextRemaining.push_back(val);
        continue;
      }

      bool allOperandsAvailable = true;
      for (Value operand : defOp->getOperands()) {
        if (mapper.contains(operand))
          continue;
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          Region *ownerRegion = blockArg.getOwner()->getParent();
          if (targetRegion->isAncestor(ownerRegion) ||
              ownerRegion->isAncestor(targetRegion))
            continue;
        }
        if (Operation *opDef = operand.getDefiningOp()) {
          Region *ownerRegion = opDef->getParentRegion();
          if (targetRegion->isAncestor(ownerRegion) ||
              ownerRegion->isAncestor(targetRegion))
            continue;
        }
        allOperandsAvailable = false;
        break;
      }

      if (!allOperandsAvailable) {
        nextRemaining.push_back(val);
        continue;
      }

      Operation *clonedOp = builder.clone(*defOp, mapper);
      for (auto [oldResult, newResult] :
           llvm::zip(defOp->getResults(), clonedOp->getResults())) {
        mapper.map(oldResult, newResult);
      }
      progress = true;
    }

    remaining = std::move(nextRemaining);
  }

  return remaining.empty();
}

///===----------------------------------------------------------------------===///
/// Constant Value Analysis
///===----------------------------------------------------------------------===///

/// Maximum recursion depth for value tracing to prevent stack overflow
static constexpr unsigned kMaxTraceDepth = 64;

bool ValueAnalysis::isValueConstant(Value val) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return false;
  return defOp->hasTrait<OpTrait::ConstantLike>();
}

bool ValueAnalysis::getConstantIndex(Value v, int64_t &out) {
  if (!v)
    return false;
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = c.value();
    return true;
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(c.getValue())) {
      out = intAttr.getInt();
      return true;
    }
  }
  return false;
}

std::optional<int64_t> ValueAnalysis::getConstantValue(Value v) {
  int64_t val;
  if (getConstantIndex(v, val))
    return val;
  return std::nullopt;
}

Value ValueAnalysis::getValueFromFoldResult(OpFoldResult ofr) {
  if (auto value = llvm::dyn_cast<Value>(ofr))
    return value;
  return Value();
}

std::optional<int64_t> ValueAnalysis::getConstantIndex(OpFoldResult ofr) {
  if (auto value = llvm::dyn_cast<Value>(ofr))
    return tryFoldConstantIndex(value);
  if (auto attr = llvm::dyn_cast<Attribute>(ofr))
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt();
  return std::nullopt;
}

std::optional<int64_t> ValueAnalysis::tryFoldConstantIndex(Value v,
                                                           unsigned depth) {
  /// Work-distribution and DB-shape calculations routinely build longer
  /// arithmetic chains (nested min/max/div/add trees) before canonicalization.
  /// Keep this bounded, but allow enough depth to recover compile-time
  /// constants from those staged expressions.
  if (!v || depth > 32)
    return std::nullopt;

  int64_t val;
  if (getConstantIndex(v, val))
    return val;

  if (auto typeSizeOp = v.getDefiningOp<polygeist::TypeSizeOp>())
    return tryFoldTypeSizeBytes(typeSizeOp);

  if (auto load = v.getDefiningOp<memref::LoadOp>())
    if (auto folded = tryFoldScalarMemrefLoad(load, depth + 1))
      return folded;

  if (auto query = v.getDefiningOp<RuntimeQueryOp>()) {
    ModuleOp module = query->getParentOfType<ModuleOp>();
    if (!module)
      return std::nullopt;
    switch (query.getKind()) {
    case RuntimeQueryKind::totalWorkers:
      return getRuntimeTotalWorkers(module);
    case RuntimeQueryKind::totalNodes:
      return getRuntimeTotalNodes(module);
    default:
      return std::nullopt;
    }
  }

  if (auto cast = v.getDefiningOp<arith::IndexCastOp>()) {
    return tryFoldConstantIndex(cast.getIn(), depth + 1);
  }
  if (auto cast = v.getDefiningOp<arith::IndexCastUIOp>()) {
    return tryFoldConstantIndex(cast.getIn(), depth + 1);
  }
  if (auto ext = v.getDefiningOp<arith::ExtSIOp>()) {
    return tryFoldConstantIndex(ext.getIn(), depth + 1);
  }
  if (auto ext = v.getDefiningOp<arith::ExtUIOp>()) {
    return tryFoldConstantIndex(ext.getIn(), depth + 1);
  }
  if (auto trunc = v.getDefiningOp<arith::TruncIOp>()) {
    return tryFoldConstantIndex(trunc.getIn(), depth + 1);
  }

  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    auto lhs = tryFoldConstantIndex(add.getLhs(), depth + 1);
    auto rhs = tryFoldConstantIndex(add.getRhs(), depth + 1);
    if (lhs && rhs)
      return *lhs + *rhs;
    return std::nullopt;
  }

  if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
    auto lhs = tryFoldConstantIndex(sub.getLhs(), depth + 1);
    auto rhs = tryFoldConstantIndex(sub.getRhs(), depth + 1);
    if (lhs && rhs)
      return *lhs - *rhs;
    return std::nullopt;
  }

  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    auto lhs = tryFoldConstantIndex(mul.getLhs(), depth + 1);
    auto rhs = tryFoldConstantIndex(mul.getRhs(), depth + 1);
    if (lhs && rhs)
      return *lhs * *rhs;
    return std::nullopt;
  }

  if (auto div = v.getDefiningOp<arith::DivUIOp>()) {
    auto lhs = tryFoldConstantIndex(div.getLhs(), depth + 1);
    auto rhs = tryFoldConstantIndex(div.getRhs(), depth + 1);
    if (lhs && rhs && *rhs != 0) {
      uint64_t ulhs = static_cast<uint64_t>(*lhs);
      uint64_t urhs = static_cast<uint64_t>(*rhs);
      return static_cast<int64_t>(ulhs / urhs);
    }
    return std::nullopt;
  }

  if (auto div = v.getDefiningOp<arith::DivSIOp>()) {
    auto lhs = tryFoldConstantIndex(div.getLhs(), depth + 1);
    auto rhs = tryFoldConstantIndex(div.getRhs(), depth + 1);
    if (lhs && rhs && *rhs != 0)
      return *lhs / *rhs;
    return std::nullopt;
  }

  if (auto max = v.getDefiningOp<arith::MaxUIOp>()) {
    auto lhs = tryFoldConstantIndex(max.getLhs(), depth + 1);
    auto rhs = tryFoldConstantIndex(max.getRhs(), depth + 1);
    if (lhs && rhs)
      return std::max<uint64_t>(static_cast<uint64_t>(*lhs),
                                static_cast<uint64_t>(*rhs));
    return std::nullopt;
  }

  if (auto min = v.getDefiningOp<arith::MinUIOp>()) {
    auto lhs = tryFoldConstantIndex(min.getLhs(), depth + 1);
    auto rhs = tryFoldConstantIndex(min.getRhs(), depth + 1);
    if (lhs && rhs)
      return std::min<uint64_t>(static_cast<uint64_t>(*lhs),
                                static_cast<uint64_t>(*rhs));
    return std::nullopt;
  }

  if (auto cmp = v.getDefiningOp<arith::CmpIOp>())
    return tryFoldCmpResult(cmp, depth + 1);

  if (auto select = v.getDefiningOp<arith::SelectOp>()) {
    auto cond = tryFoldConstantIndex(select.getCondition(), depth + 1);
    if (cond)
      return tryFoldConstantIndex((*cond != 0) ? select.getTrueValue()
                                               : select.getFalseValue(),
                                  depth + 1);

    auto trueValue = tryFoldConstantIndex(select.getTrueValue(), depth + 1);
    auto falseValue = tryFoldConstantIndex(select.getFalseValue(), depth + 1);
    if (trueValue && falseValue && *trueValue == *falseValue)
      return trueValue;
    return std::nullopt;
  }

  return std::nullopt;
}

std::optional<int64_t> ValueAnalysis::getConstantIndexStripped(Value v) {
  if (!v)
    return std::nullopt;
  return tryFoldConstantIndex(stripNumericCasts(v));
}

std::optional<double> ValueAnalysis::getConstantFloat(Value v) {
  if (!v)
    return std::nullopt;
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(c.getValue())) {
      return floatAttr.getValueAsDouble();
    }
  }
  return std::nullopt;
}

bool ValueAnalysis::isConstantEqual(Value v, int64_t val) {
  if (auto cst = getConstantValue(v))
    return *cst == val;
  if (auto cst = getConstantFloat(v))
    return *cst == static_cast<double>(val);
  return false;
}

bool ValueAnalysis::isZeroConstant(Value v) { return isConstantEqual(v, 0); }

bool ValueAnalysis::isOneConstant(Value v) { return isConstantEqual(v, 1); }

bool ValueAnalysis::isConstantBool(Value v, bool expected) {
  v = stripNumericCasts(v);
  auto constant = v.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;

  Attribute attr = constant.getValue();
  if (auto boolAttr = dyn_cast<BoolAttr>(attr))
    return boolAttr.getValue() == expected;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return expected ? intAttr.getValue().isOne() : intAttr.getValue().isZero();
  return false;
}

bool ValueAnalysis::isTrueConstant(Value v) { return isConstantBool(v, true); }

bool ValueAnalysis::sameValue(Value a, Value b) {
  a = stripNumericCasts(a);
  b = stripNumericCasts(b);
  if (a == b)
    return true;
  if (auto equal = proveValuesEqualWithBounds(a, b); equal && *equal)
    return true;
  auto aConst = tryFoldConstantIndex(a);
  auto bConst = tryFoldConstantIndex(b);
  return aConst && bConst && *aConst == *bConst;
}

bool ValueAnalysis::areValueRangesIdentical(ValueRange lhs, ValueRange rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left != right)
      return false;
  return true;
}

bool ValueAnalysis::hasValueRangePrefix(ValueRange values, ValueRange prefix) {
  if (values.size() < prefix.size())
    return false;
  return std::equal(prefix.begin(), prefix.end(), values.begin());
}

bool ValueAnalysis::areValueRangesEquivalent(ValueRange lhs, ValueRange rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (!sameValue(left, right))
      return false;
  return true;
}

Value ValueAnalysis::stripClampOne(Value v) {
  Value cur = stripNumericCasts(v);
  while (auto maxOp = cur.getDefiningOp<arith::MaxUIOp>()) {
    Value lhs = stripNumericCasts(maxOp.getLhs());
    Value rhs = stripNumericCasts(maxOp.getRhs());
    if (isOneConstant(lhs)) {
      cur = rhs;
      continue;
    }
    if (isOneConstant(rhs)) {
      cur = lhs;
      continue;
    }
    break;
  }
  return cur;
}

Value ValueAnalysis::stripSelectClamp(Value value, int maxDepth) {
  if (!value || maxDepth <= 0)
    return value;
  value = stripNumericCasts(value);

  auto selectOp = value.getDefiningOp<arith::SelectOp>();
  if (!selectOp)
    return value;

  Value trueVal = stripNumericCasts(selectOp.getTrueValue());
  Value falseVal = stripNumericCasts(selectOp.getFalseValue());

  if (auto cmp = selectOp.getCondition().getDefiningOp<arith::CmpIOp>()) {
    Value lhs = stripNumericCasts(cmp.getLhs());
    Value rhs = stripNumericCasts(cmp.getRhs());
    auto pred = cmp.getPredicate();

    bool isClampCmp = pred == arith::CmpIPredicate::slt ||
                      pred == arith::CmpIPredicate::ult ||
                      pred == arith::CmpIPredicate::sgt ||
                      pred == arith::CmpIPredicate::ugt;
    if (isClampCmp) {
      /// Match min/max clamp canonical form:
      ///   cmp(lhs, rhs) + select(cond, rhs, lhs)
      if (sameValue(trueVal, rhs) && sameValue(falseVal, lhs))
        return stripSelectClamp(lhs, maxDepth - 1);
      /// Also handle the mirrored operand form.
      if (sameValue(trueVal, lhs) && sameValue(falseVal, rhs))
        return stripSelectClamp(rhs, maxDepth - 1);
    }
  }

  /// Fallback: select(const, x) / select(x, const) behaves like a guarded
  /// clamp in many lowered index expressions; keep the non-constant side.
  auto trueConst = getConstantValue(trueVal);
  auto falseConst = getConstantValue(falseVal);
  if (trueConst && !falseConst)
    return stripSelectClamp(falseVal, maxDepth - 1);
  if (falseConst && !trueConst)
    return stripSelectClamp(trueVal, maxDepth - 1);

  return value;
}

bool ValueAnalysis::areValuesEquivalent(Value a, Value b, int depth) {
  if (!a || !b)
    return false;
  if (a == b)
    return true;
  if (depth > 6)
    return false;

  a = stripNumericCasts(a);
  b = stripNumericCasts(b);
  if (a == b)
    return true;
  if (auto equal = proveValuesEqualWithBounds(a, b); equal && *equal)
    return true;

  auto ca = getConstantValue(a);
  auto cb = getConstantValue(b);
  if (ca && cb)
    return *ca == *cb;

  // Recursive structural equivalence below handles cases that
  // ValueBoundsConstraintSet::areEqual cannot:
  //   - Non-index types (areEqual requires isIndex())
  //   - Commutative operand order (add(x,y) == add(y,x))
  //   - Structurally identical trees from different IR regions where the
  //     Presburger solver lacks sufficient constraints
  Operation *opA = a.getDefiningOp();
  Operation *opB = b.getDefiningOp();
  if (!opA || !opB)
    return false;
  if (opA->getName() != opB->getName())
    return false;
  if (opA->getNumOperands() != opB->getNumOperands())
    return false;

  auto eq = [&](Value x, Value y) {
    return areValuesEquivalent(x, y, depth + 1);
  };

  if (isa<arith::AddIOp, arith::MulIOp, arith::MaxUIOp, arith::MinUIOp>(opA)) {
    if (opA->getNumOperands() == 2) {
      Value a0 = opA->getOperand(0);
      Value a1 = opA->getOperand(1);
      Value b0 = opB->getOperand(0);
      Value b1 = opB->getOperand(1);
      return (eq(a0, b0) && eq(a1, b1)) || (eq(a0, b1) && eq(a1, b0));
    }
  }

  for (unsigned i = 0; i < opA->getNumOperands(); ++i) {
    if (!eq(opA->getOperand(i), opB->getOperand(i)))
      return false;
  }
  return true;
}

bool ValueAnalysis::isConstantAtLeastOne(Value v) {
  if (!v)
    return false;
  v = stripNumericCasts(v);
  // proveValueAtLeast delegates to
  // ValueBoundsConstraintSet::computeConstantBound but only works for
  // index-typed values. The getConstantIndex fallback below covers non-index
  // integer constants that the bounds infrastructure rejects.
  if (auto atLeastOne = proveValueAtLeast(v, 1); atLeastOne)
    return *atLeastOne;
  int64_t val = 0;
  if (getConstantIndex(v, val))
    return val >= 1;
  return false;
}

bool ValueAnalysis::isProvablyNonZero(Value v, unsigned depth) {
  if (!v || depth > 4)
    return false;
  v = stripNumericCasts(v);
  // proveValueNonZero checks LB >= 1 || UB <= -1 via bounds, but only for
  // index types. The isConstantAtLeastOne fallback handles non-index integer
  // constants. Max propagation handles max(a, b) where either operand is
  // provably non-zero — the bounds infrastructure does not propagate through
  // max ops reliably.
  if (auto nonZero = proveValueNonZero(v); nonZero)
    return *nonZero;
  if (isConstantAtLeastOne(v))
    return true;
  if (auto maxui = v.getDefiningOp<arith::MaxUIOp>()) {
    return isProvablyNonZero(maxui.getLhs(), depth + 1) ||
           isProvablyNonZero(maxui.getRhs(), depth + 1);
  }
  if (auto maxsi = v.getDefiningOp<arith::MaxSIOp>()) {
    return isProvablyNonZero(maxsi.getLhs(), depth + 1) ||
           isProvablyNonZero(maxsi.getRhs(), depth + 1);
  }
  return false;
}

///===----------------------------------------------------------------------===///
/// Value Type Conversion and Casting
///===----------------------------------------------------------------------===///

Value ValueAnalysis::stripNumericCasts(Value value) {
  while (true) {
    if (auto idxCast = value.getDefiningOp<arith::IndexCastOp>()) {
      value = idxCast.getIn();
      continue;
    }
    if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
      value = ext.getIn();
      continue;
    }
    if (auto ext = value.getDefiningOp<arith::ExtUIOp>()) {
      value = ext.getIn();
      continue;
    }
    if (auto trunc = value.getDefiningOp<arith::TruncIOp>()) {
      value = trunc.getIn();
      continue;
    }
    if (auto extF = value.getDefiningOp<arith::ExtFOp>()) {
      value = extF.getIn();
      continue;
    }
    if (auto truncF = value.getDefiningOp<arith::TruncFOp>()) {
      value = truncF.getIn();
      continue;
    }
    break;
  }
  return value;
}

Value ValueAnalysis::castToIndex(Value value, OpBuilder &builder,
                                 Location loc) {
  if (!value)
    return value;
  if (value.getType().isIndex())
    return value;
  if (value.getType().isIntOrIndex())
    return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                      value);
  return value;
}

Value ValueAnalysis::ensureIndexType(Value value, OpBuilder &builder,
                                     Location loc) {
  if (!value)
    return Value();
  if (isa<IndexType>(value.getType()))
    return value;
  if (auto intTy = dyn_cast<IntegerType>(value.getType()))
    return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                      value);
  return Value();
}

///===----------------------------------------------------------------------===///
/// Value Dependencies and Analysis
///===----------------------------------------------------------------------===///

bool ValueAnalysis::dependsOn(Value value, Value base, int depth) {
  if (!value || !base || depth > 8)
    return false;
  if (value == base)
    return true;

  if (isa<BlockArgument>(value))
    return false;

  Operation *op = value.getDefiningOp();
  if (!op)
    return false;

  if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
           arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::IndexCastOp,
           arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp,
           arith::TruncIOp, arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp,
           arith::MaxUIOp, arith::SelectOp, arith::CmpIOp, arith::CmpFOp>(op))
    return false;

  for (Value operand : op->getOperands())
    if (dependsOn(operand, base, depth + 1))
      return true;

  return false;
}

std::optional<int64_t> ValueAnalysis::getOffsetStride(Value idx, Value base,
                                                      int depth) {
  if (!idx || !base || depth > 8)
    return std::nullopt;

  idx = stripNumericCasts(idx);
  base = stripNumericCasts(base);

  if (idx == base)
    return int64_t{1};

  if (auto addOp = idx.getDefiningOp<arith::AddIOp>()) {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();
    bool lhsDep = dependsOn(lhs, base, depth + 1);
    bool rhsDep = dependsOn(rhs, base, depth + 1);
    if (lhsDep && !rhsDep)
      return getOffsetStride(lhs, base, depth + 1);
    if (rhsDep && !lhsDep)
      return getOffsetStride(rhs, base, depth + 1);
    return std::nullopt;
  }

  if (auto subOp = idx.getDefiningOp<arith::SubIOp>()) {
    Value lhs = subOp.getLhs();
    Value rhs = subOp.getRhs();
    bool lhsDep = dependsOn(lhs, base, depth + 1);
    bool rhsDep = dependsOn(rhs, base, depth + 1);
    if (lhsDep && !rhsDep)
      return getOffsetStride(lhs, base, depth + 1);
    return std::nullopt;
  }

  if (auto mulOp = idx.getDefiningOp<arith::MulIOp>()) {
    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();
    int64_t constVal = 0;
    if (getConstantIndex(lhs, constVal) && dependsOn(rhs, base, depth + 1))
      return constVal;
    if (getConstantIndex(rhs, constVal) && dependsOn(lhs, base, depth + 1))
      return constVal;
  }

  return std::nullopt;
}

static bool isDerivedFromPtrImpl(Value value, Value source,
                                 SmallPtrSetImpl<Value> &visited,
                                 unsigned depth) {
  if (depth > kMaxTraceDepth)
    return false;
  if (value == source)
    return true;
  if (!visited.insert(value).second)
    return false;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *parentBlock = blockArg.getParentBlock();
    if (parentBlock && parentBlock->getParentOp()) {
      if (auto edt = dyn_cast<EdtOp>(parentBlock->getParentOp())) {
        unsigned argIndex = blockArg.getArgNumber();
        ValueRange deps = edt.getDependencies();
        if (argIndex < deps.size())
          return isDerivedFromPtrImpl(deps[argIndex], source, visited,
                                      depth + 1);
      }
    }
    return false;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;

  auto trace = [&](Value v) -> bool {
    return isDerivedFromPtrImpl(v, source, visited, depth + 1);
  };

  if (auto dbAcquire = dyn_cast<DbAcquireOp>(defOp))
    return trace(dbAcquire.getSourcePtr());
  if (auto dbGep = dyn_cast<::mlir::carts::arts_rt::DbGepOp>(defOp))
    return trace(dbGep.getBasePtr());
  if (auto gepOp = dyn_cast<LLVM::GEPOp>(defOp))
    return trace(gepOp.getBase());
  if (auto ptr2memref = dyn_cast<polygeist::Pointer2MemrefOp>(defOp))
    return trace(ptr2memref.getSource());
  if (auto memref2ptr = dyn_cast<polygeist::Memref2PointerOp>(defOp))
    return trace(memref2ptr.getSource());
  if (auto subview = dyn_cast<memref::SubViewOp>(defOp))
    return trace(subview.getSource());
  if (auto cast = dyn_cast<memref::CastOp>(defOp))
    return trace(cast.getSource());
  if (auto view = dyn_cast<memref::ViewOp>(defOp))
    return trace(view.getSource());
  if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(defOp))
    return trace(reinterpretCast.getSource());

  /// Fallback: trace through first operand (handles polygeist.subindex etc.)
  if (defOp->getNumOperands() > 0)
    return trace(defOp->getOperand(0));

  return false;
}

bool ValueAnalysis::isDerivedFromPtr(Value value, Value source) {
  SmallPtrSet<Value, 16> visited;
  return isDerivedFromPtrImpl(value, source, visited, 0);
}

/// Extract constant offset from an index expression relative to loopIV and
/// chunkOffset. E.g. chunkOffset + loopIV + 5 yields offset = 5.
std::optional<int64_t> ValueAnalysis::extractConstantOffset(Value idx,
                                                            Value loopIV,
                                                            Value chunkOffset) {
  if (auto delta = proveConstantOffsetWithBounds(idx, loopIV, chunkOffset))
    return delta;

  int64_t accumulator = 0;
  Value current = idx;

  auto isBaseValue = [&](Value v) -> bool {
    return v == loopIV || v == chunkOffset;
  };

  auto isBasePattern = [&](Value v) -> bool {
    if (auto addOp = v.getDefiningOp<arith::AddIOp>()) {
      Value lhs = addOp.getLhs();
      Value rhs = addOp.getRhs();
      return (lhs == loopIV && rhs == chunkOffset) ||
             (lhs == chunkOffset && rhs == loopIV);
    }
    return false;
  };

  while (true) {
    if (isBaseValue(current) || isBasePattern(current))
      return accumulator;

    if (auto addOp = current.getDefiningOp<arith::AddIOp>()) {
      Value lhs = addOp.getLhs();
      Value rhs = addOp.getRhs();

      int64_t constVal;
      if (getConstantIndex(rhs, constVal)) {
        accumulator += constVal;
        current = lhs;
      } else if (getConstantIndex(lhs, constVal)) {
        accumulator += constVal;
        current = rhs;
      } else {
        bool lhsIsBase = isBaseValue(lhs) || isBasePattern(lhs);
        bool rhsIsBase = isBaseValue(rhs) || isBasePattern(rhs);

        if (lhsIsBase && rhsIsBase)
          return accumulator;
        if (lhsIsBase)
          current = rhs;
        else if (rhsIsBase)
          current = lhs;
        else
          break;
      }
    } else if (auto subOp = current.getDefiningOp<arith::SubIOp>()) {
      int64_t constVal;
      if (getConstantIndex(subOp.getRhs(), constVal)) {
        accumulator -= constVal;
        current = subOp.getLhs();
        continue;
      }
      break;
    } else if (auto indexCast = current.getDefiningOp<arith::IndexCastOp>()) {
      current = indexCast.getIn();
      continue;
    } else {
      break;
    }
  }

  return std::nullopt;
}

Value ValueAnalysis::stripConstantOffset(Value value, int64_t *outConst) {
  int64_t accumulator = 0;
  Value current = value;

  if (!current) {
    if (outConst)
      *outConst = 0;
    return current;
  }

  while (current) {
    current = stripNumericCasts(current);

    if (auto addOp = current.getDefiningOp<arith::AddIOp>()) {
      int64_t constVal;
      if (getConstantIndex(addOp.getLhs(), constVal)) {
        accumulator += constVal;
        current = addOp.getRhs();
        continue;
      }
      if (getConstantIndex(addOp.getRhs(), constVal)) {
        accumulator += constVal;
        current = addOp.getLhs();
        continue;
      }
    } else if (auto subOp = current.getDefiningOp<arith::SubIOp>()) {
      int64_t constVal;
      if (getConstantIndex(subOp.getRhs(), constVal)) {
        accumulator -= constVal;
        current = subOp.getLhs();
        continue;
      }
    } else if (auto indexCast = current.getDefiningOp<arith::IndexCastOp>()) {
      current = indexCast.getIn();
      continue;
    }
    break;
  }

  if (outConst)
    *outConst = accumulator;
  return current;
}

///===----------------------------------------------------------------------===///
/// Underlying Value Tracing
///===----------------------------------------------------------------------===///

static Value getUnderlyingValueImpl(Value v, SmallPtrSet<Value, 16> &visited,
                                    unsigned depth) {
  if (!v || depth > kMaxTraceDepth || !visited.insert(v).second)
    return nullptr;

  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    Block *owner = blockArg.getOwner();
    if (owner && owner->getParentOp()) {
      if (auto edt = dyn_cast<EdtOp>(owner->getParentOp())) {
        unsigned argIndex = blockArg.getArgNumber();
        ValueRange deps = edt.getDependencies();
        if (argIndex < deps.size())
          return getUnderlyingValueImpl(deps[argIndex], visited, depth + 1);
      }
    }
    return v;
  }

  Operation *op = v.getDefiningOp();
  if (!op)
    return nullptr;

  /// Defensive: reject operations that are detached from any block.  This
  /// catches dangling Value references to erased operations whose memory has
  /// not yet been reclaimed — calling isa<> on such a pointer is UB.
  if (!op->getBlock())
    return nullptr;

  /// Terminal allocations
  if (isa<DbAllocOp, memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(
          op))
    return v;

  /// Helper to recurse through a single source operand
  auto trace = [&](Value source) -> Value {
    return getUnderlyingValueImpl(source, visited, depth + 1);
  };

  if (auto dbAcquire = dyn_cast<DbAcquireOp>(op))
    return trace(dbAcquire.getSourcePtr());
  if (auto dbGep = dyn_cast<::mlir::carts::arts_rt::DbGepOp>(op))
    return trace(dbGep.getBasePtr());
  if (auto dbRef = dyn_cast<DbRefOp>(op))
    return trace(dbRef.getSource());
  if (auto subview = dyn_cast<memref::SubViewOp>(op))
    return trace(subview.getSource());
  if (auto castOp = dyn_cast<memref::CastOp>(op))
    return trace(castOp.getSource());
  if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(op)) {
    ValueRange inputs = unrealized.getInputs();
    return inputs.empty() ? nullptr : trace(inputs.front());
  }
  if (auto view = dyn_cast<memref::ViewOp>(op))
    return trace(view.getSource());
  if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(op))
    return trace(reinterpret.getSource());
  if (auto p2m = dyn_cast<polygeist::Pointer2MemrefOp>(op))
    return trace(p2m.getSource());
  if (auto m2p = dyn_cast<polygeist::Memref2PointerOp>(op))
    return trace(m2p.getSource());
  if (auto gep = dyn_cast<LLVM::GEPOp>(op))
    return trace(gep.getBase());
  if (auto subindex = dyn_cast<polygeist::SubIndexOp>(op))
    return trace(subindex.getSource());

  return nullptr;
}

Value ValueAnalysis::stripMemrefViewOps(Value value) {
  while (value) {
    auto unrealized = value.getDefiningOp<UnrealizedConversionCastOp>();
    if (unrealized) {
      if (unrealized.getInputs().size() != 1)
        break;
      value = unrealized.getInputs().front();
      continue;
    }

    if (auto castOp = value.getDefiningOp<memref::CastOp>()) {
      value = castOp.getSource();
      continue;
    }
    if (auto subviewOp = value.getDefiningOp<memref::SubViewOp>()) {
      value = subviewOp.getSource();
      continue;
    }
    if (auto viewOp = value.getDefiningOp<memref::ViewOp>()) {
      value = viewOp.getSource();
      continue;
    }
    if (auto reinterpretOp =
            value.getDefiningOp<memref::ReinterpretCastOp>()) {
      value = reinterpretOp.getSource();
      continue;
    }

    break;
  }
  return value;
}

Value ValueAnalysis::getUnderlyingValue(Value v) {
  SmallPtrSet<Value, 16> visited;
  return getUnderlyingValueImpl(v, visited, 0);
}

Operation *ValueAnalysis::getUnderlyingOperation(Value v) {
  Value underlyingValue = getUnderlyingValue(v);
  return underlyingValue ? underlyingValue.getDefiningOp() : nullptr;
}

///===----------------------------------------------------------------------===//
/// Value Reconstruction for Dominance
///===----------------------------------------------------------------------===//

Value ValueAnalysis::traceValueToDominating(Value value,
                                            Operation *insertBefore,
                                            OpBuilder &builder,
                                            DominanceInfo &domInfo,
                                            Location loc, unsigned depth) {
  if (!value || depth > 16)
    return nullptr;

  if (domInfo.properlyDominates(value, insertBefore))
    return value;

  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return nullptr;

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;

  auto trace = [&](Value operand) -> Value {
    return traceValueToDominating(operand, insertBefore, builder, domInfo, loc,
                                  depth + 1);
  };

  if (auto op = dyn_cast<arith::AddIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::SubIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::MulIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::DivSIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::DivUIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::CeilDivSIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::CeilDivUIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::RemSIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::RemUIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::MaxSIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::MaxUIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::MinSIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::MinUIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::AndIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::OrIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::XOrIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::ShLIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::ShRUIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);
  if (auto op = dyn_cast<arith::ShRSIOp>(defOp))
    return traceBinaryOp(op, insertBefore, builder, loc, trace);

  if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
    Value lhs = trace(cmpOp.getLhs());
    Value rhs = trace(cmpOp.getRhs());
    if (lhs && rhs) {
      builder.setInsertionPoint(insertBefore);
      return arith::CmpIOp::create(builder, loc, cmpOp.getPredicate(), lhs,
                                   rhs);
    }
    return nullptr;
  }

  if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    Value cond = trace(selectOp.getCondition());
    Value trueVal = trace(selectOp.getTrueValue());
    Value falseVal = trace(selectOp.getFalseValue());
    if (cond && trueVal && falseVal) {
      builder.setInsertionPoint(insertBefore);
      return arith::SelectOp::create(builder, loc, cond, trueVal, falseVal);
    }
    return nullptr;
  }

  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp)) {
    Value inner = trace(castOp.getIn());
    if (!inner)
      return nullptr;
    if (inner.getType() == castOp.getType())
      return inner;
    builder.setInsertionPoint(insertBefore);
    return arith::IndexCastOp::create(builder, loc, castOp.getType(), inner);
  }
  if (auto castOp = dyn_cast<arith::IndexCastUIOp>(defOp)) {
    Value inner = trace(castOp.getIn());
    if (!inner)
      return nullptr;
    if (inner.getType() == castOp.getType())
      return inner;
    builder.setInsertionPoint(insertBefore);
    return arith::IndexCastUIOp::create(builder, loc, castOp.getType(), inner);
  }
  if (auto castOp = dyn_cast<arith::ExtSIOp>(defOp)) {
    Value inner = trace(castOp.getIn());
    if (!inner)
      return nullptr;
    builder.setInsertionPoint(insertBefore);
    return arith::ExtSIOp::create(builder, loc, castOp.getType(), inner);
  }
  if (auto castOp = dyn_cast<arith::ExtUIOp>(defOp)) {
    Value inner = trace(castOp.getIn());
    if (!inner)
      return nullptr;
    builder.setInsertionPoint(insertBefore);
    return arith::ExtUIOp::create(builder, loc, castOp.getType(), inner);
  }
  if (auto castOp = dyn_cast<arith::TruncIOp>(defOp)) {
    Value inner = trace(castOp.getIn());
    if (!inner)
      return nullptr;
    builder.setInsertionPoint(insertBefore);
    return arith::TruncIOp::create(builder, loc, castOp.getType(), inner);
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    builder.setInsertionPoint(insertBefore);
    return arith::ConstantOp::create(builder, loc, constOp.getType(),
                                     constOp.getValue());
  }
  if (auto constIdxOp = dyn_cast<arith::ConstantIndexOp>(defOp)) {
    builder.setInsertionPoint(insertBefore);
    return ::mlir::carts::arts::createConstantIndex(builder, loc, constIdxOp.value());
  }
  if (auto tsOp = dyn_cast<polygeist::TypeSizeOp>(defOp)) {
    builder.setInsertionPoint(insertBefore);
    return polygeist::TypeSizeOp::create(builder, loc, tsOp.getType(),
                                         tsOp.getSourceAttr());
  }

  /// Handle memref.load from scalar wrappers (rank-0 memref.alloca).
  /// These appear when Polygeist wraps scalar values in rank-0 allocas.
  /// We can clone the load at the insertion point as long as the memref source
  /// dominates it and the loaded value is not a memref type (scalar only).
  if (auto loadOp = dyn_cast<memref::LoadOp>(defOp)) {
    auto memType = cast<MemRefType>(loadOp.getMemref().getType());
    if (memType.getRank() == 0 && !isa<MemRefType>(loadOp.getType()) &&
        domInfo.properlyDominates(loadOp.getMemref(), insertBefore)) {
      builder.setInsertionPoint(insertBefore);
      return memref::LoadOp::create(builder, loc, loadOp.getMemref());
    }
  }

  return nullptr;
}

Value ValueAnalysis::traceMinSIWithFallback(
    arith::MinSIOp minOp, Operation *insertBefore, OpBuilder &builder,
    Location loc, llvm::function_ref<Value(Value)> traceValueFn) {
  Value lhsTraced = traceValueFn(minOp.getLhs());
  Value rhsTraced = traceValueFn(minOp.getRhs());
  if (lhsTraced && rhsTraced) {
    if (ValueAnalysis::isZeroConstant(lhsTraced))
      return rhsTraced;
    if (ValueAnalysis::isZeroConstant(rhsTraced))
      return lhsTraced;
    builder.setInsertionPoint(insertBefore);
    return arith::MinSIOp::create(builder, loc, lhsTraced, rhsTraced);
  }
  if (rhsTraced && !lhsTraced) {
    if (ValueAnalysis::isZeroConstant(rhsTraced))
      return nullptr;
    return rhsTraced;
  }
  if (lhsTraced && !rhsTraced) {
    if (ValueAnalysis::isZeroConstant(lhsTraced))
      return nullptr;
    return lhsTraced;
  }
  return nullptr;
}

Value ValueAnalysis::traceMinUIWithFallback(
    arith::MinUIOp minOp, Operation *insertBefore, OpBuilder &builder,
    Location loc, llvm::function_ref<Value(Value)> traceValueFn) {
  Value lhsTraced = traceValueFn(minOp.getLhs());
  Value rhsTraced = traceValueFn(minOp.getRhs());
  if (lhsTraced && rhsTraced) {
    if (ValueAnalysis::isZeroConstant(lhsTraced))
      return rhsTraced;
    if (ValueAnalysis::isZeroConstant(rhsTraced))
      return lhsTraced;
    builder.setInsertionPoint(insertBefore);
    return arith::MinUIOp::create(builder, loc, lhsTraced, rhsTraced);
  }
  if (rhsTraced && !lhsTraced) {
    if (ValueAnalysis::isZeroConstant(rhsTraced))
      return nullptr;
    return rhsTraced;
  }
  if (lhsTraced && !rhsTraced) {
    if (ValueAnalysis::isZeroConstant(lhsTraced))
      return nullptr;
    return lhsTraced;
  }
  return nullptr;
}

Value ValueAnalysis::traceSelectWithFallback(
    arith::SelectOp selectOp, Operation *insertBefore, OpBuilder &builder,
    Location loc, llvm::function_ref<Value(Value)> traceValueFn,
    llvm::function_ref<Value(Value)> traceCondFn) {
  Value trueTraced = traceValueFn(selectOp.getTrueValue());
  Value falseTraced = traceValueFn(selectOp.getFalseValue());

  if (trueTraced && falseTraced) {
    Value cond = traceCondFn(selectOp.getCondition());
    builder.setInsertionPoint(insertBefore);
    if (cond) {
      return arith::SelectOp::create(builder, loc, cond, trueTraced,
                                     falseTraced);
    }
    /// Condition doesn't dominate -- use max of arms as safe upper bound.
    if (isa<IndexType>(trueTraced.getType())) {
      return arith::MaxUIOp::create(builder, loc, trueTraced, falseTraced);
    }
    return arith::MaxSIOp::create(builder, loc, trueTraced, falseTraced);
  }

  if (trueTraced && !falseTraced) {
    if (ValueAnalysis::isZeroConstant(trueTraced))
      return nullptr;
    return trueTraced;
  }
  if (falseTraced && !trueTraced) {
    if (ValueAnalysis::isZeroConstant(falseTraced))
      return nullptr;
    return falseTraced;
  }
  return nullptr;
}

bool ValueAnalysis::isOneLikeValue(Value value) {
  if (isOneConstant(value))
    return true;

  auto addOp = value.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return false;

  Value lhs = addOp.getLhs();
  Value rhs = addOp.getRhs();
  Value other;
  if (isOneConstant(lhs))
    other = rhs;
  else if (isOneConstant(rhs))
    other = lhs;
  else
    return false;

  auto subOp = other.getDefiningOp<arith::SubIOp>();
  if (!subOp)
    return false;

  Value subLhs = subOp.getLhs();
  Value subRhs = subOp.getRhs();
  if (subLhs == subRhs)
    return true;

  if (auto minOp = subLhs.getDefiningOp<arith::MinUIOp>())
    return minOp.getLhs() == subRhs || minOp.getRhs() == subRhs;

  return false;
}

} // namespace carts::arts
} // namespace mlir
