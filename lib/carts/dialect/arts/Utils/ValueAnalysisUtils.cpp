///==========================================================================///
/// File: ValueAnalysisUtils.cpp
///
/// ARTS-owned value analysis extensions.
///==========================================================================///

#include "carts/dialect/arts/Utils/ValueAnalysisUtils.h"

#include "carts/utils/OperationAttributes.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

static constexpr unsigned kMaxTraceDepth = 64;

static std::optional<int64_t> foldRuntimeQuery(Value value, unsigned) {
  auto query = value.getDefiningOp<RuntimeQueryOp>();
  if (!query)
    return std::nullopt;

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

static bool isDerivedFromPtrImpl(Value value, Value source,
                                 llvm::SmallPtrSetImpl<Value> &visited,
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

  auto trace = [&](Value next) -> bool {
    return isDerivedFromPtrImpl(next, source, visited, depth + 1);
  };

  if (auto dbAcquire = dyn_cast<DbAcquireOp>(defOp))
    return trace(dbAcquire.getSourcePtr());
  if (auto dbRef = dyn_cast<DbRefOp>(defOp))
    return trace(dbRef.getSource());
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

  if (defOp->getNumOperands() > 0)
    return trace(defOp->getOperand(0));

  return false;
}

static Value getUnderlyingValueImpl(Value value,
                                    llvm::SmallPtrSetImpl<Value> &visited,
                                    unsigned depth) {
  if (!value || depth > kMaxTraceDepth || !visited.insert(value).second)
    return nullptr;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *owner = blockArg.getOwner();
    if (owner && owner->getParentOp()) {
      if (auto edt = dyn_cast<EdtOp>(owner->getParentOp())) {
        unsigned argIndex = blockArg.getArgNumber();
        ValueRange deps = edt.getDependencies();
        if (argIndex < deps.size())
          return getUnderlyingValueImpl(deps[argIndex], visited, depth + 1);
      }
    }
    return value;
  }

  Operation *op = value.getDefiningOp();
  if (!op)
    return nullptr;
  if (!op->getBlock())
    return nullptr;

  if (isa<DbAllocOp, memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(
          op))
    return value;

  auto trace = [&](Value source) -> Value {
    return getUnderlyingValueImpl(source, visited, depth + 1);
  };

  if (auto dbAcquire = dyn_cast<DbAcquireOp>(op))
    return trace(dbAcquire.getSourcePtr());
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

} // namespace

std::optional<int64_t> mlir::carts::arts::tryFoldConstantIndex(Value value,
                                                               unsigned depth) {
  return ValueAnalysis::tryFoldConstantIndexWith(value, foldRuntimeQuery,
                                                 depth);
}

std::optional<int64_t>
mlir::carts::arts::getConstantIndexStripped(Value value) {
  if (!value)
    return std::nullopt;
  return tryFoldConstantIndex(ValueAnalysis::stripNumericCasts(value));
}

Value mlir::carts::arts::stripSelectClamp(Value value, int maxDepth) {
  if (!value || maxDepth <= 0)
    return value;
  value = ValueAnalysis::stripNumericCasts(value);

  auto selectOp = value.getDefiningOp<arith::SelectOp>();
  if (!selectOp)
    return value;

  Value trueVal = ValueAnalysis::stripNumericCasts(selectOp.getTrueValue());
  Value falseVal = ValueAnalysis::stripNumericCasts(selectOp.getFalseValue());

  if (auto cmp = selectOp.getCondition().getDefiningOp<arith::CmpIOp>()) {
    Value lhs = ValueAnalysis::stripNumericCasts(cmp.getLhs());
    Value rhs = ValueAnalysis::stripNumericCasts(cmp.getRhs());
    auto pred = cmp.getPredicate();

    bool isClampCmp = pred == arith::CmpIPredicate::slt ||
                      pred == arith::CmpIPredicate::ult ||
                      pred == arith::CmpIPredicate::sgt ||
                      pred == arith::CmpIPredicate::ugt;
    if (isClampCmp) {
      /// Match min/max clamp canonical form:
      ///   cmp(lhs, rhs) + select(cond, rhs, lhs)
      if (ValueAnalysis::sameValue(trueVal, rhs) &&
          ValueAnalysis::sameValue(falseVal, lhs))
        return stripSelectClamp(lhs, maxDepth - 1);
      /// Also handle the mirrored operand form.
      if (ValueAnalysis::sameValue(trueVal, lhs) &&
          ValueAnalysis::sameValue(falseVal, rhs))
        return stripSelectClamp(rhs, maxDepth - 1);
    }
  }

  /// Fallback: select(const, x) / select(x, const) behaves like a guarded
  /// clamp in many lowered index expressions; keep the non-constant side.
  auto trueConst = ValueAnalysis::getConstantValue(trueVal);
  auto falseConst = ValueAnalysis::getConstantValue(falseVal);
  if (trueConst && !falseConst)
    return stripSelectClamp(falseVal, maxDepth - 1);
  if (falseConst && !trueConst)
    return stripSelectClamp(trueVal, maxDepth - 1);

  return value;
}

std::optional<int64_t> mlir::carts::arts::getOffsetStride(Value idx, Value base,
                                                          int depth) {
  if (!idx || !base || depth > 8)
    return std::nullopt;

  idx = ValueAnalysis::stripNumericCasts(idx);
  base = ValueAnalysis::stripNumericCasts(base);

  if (idx == base)
    return int64_t{1};

  if (auto addOp = idx.getDefiningOp<arith::AddIOp>()) {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();
    bool lhsDep = ValueAnalysis::dependsOn(lhs, base, depth + 1);
    bool rhsDep = ValueAnalysis::dependsOn(rhs, base, depth + 1);
    if (lhsDep && !rhsDep)
      return getOffsetStride(lhs, base, depth + 1);
    if (rhsDep && !lhsDep)
      return getOffsetStride(rhs, base, depth + 1);
    return std::nullopt;
  }

  if (auto subOp = idx.getDefiningOp<arith::SubIOp>()) {
    Value lhs = subOp.getLhs();
    Value rhs = subOp.getRhs();
    bool lhsDep = ValueAnalysis::dependsOn(lhs, base, depth + 1);
    bool rhsDep = ValueAnalysis::dependsOn(rhs, base, depth + 1);
    if (lhsDep && !rhsDep)
      return getOffsetStride(lhs, base, depth + 1);
    return std::nullopt;
  }

  if (auto mulOp = idx.getDefiningOp<arith::MulIOp>()) {
    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();
    int64_t constVal = 0;
    if (ValueAnalysis::getConstantIndex(lhs, constVal) &&
        ValueAnalysis::dependsOn(rhs, base, depth + 1))
      return constVal;
    if (ValueAnalysis::getConstantIndex(rhs, constVal) &&
        ValueAnalysis::dependsOn(lhs, base, depth + 1))
      return constVal;
  }

  return std::nullopt;
}

Value mlir::carts::arts::getUnderlyingValue(Value value) {
  llvm::SmallPtrSet<Value, 16> visited;
  return getUnderlyingValueImpl(value, visited, 0);
}

Operation *mlir::carts::arts::getUnderlyingOperation(Value value) {
  Value underlying = getUnderlyingValue(value);
  return underlying ? underlying.getDefiningOp() : nullptr;
}

bool mlir::carts::arts::isDerivedFromPtr(Value value, Value source) {
  llvm::SmallPtrSet<Value, 16> visited;
  return isDerivedFromPtrImpl(value, source, visited, 0);
}

std::optional<int64_t>
mlir::carts::arts::extractConstantOffset(Value idx, Value loopIV,
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
      if (ValueAnalysis::getConstantIndex(rhs, constVal)) {
        accumulator += constVal;
        current = lhs;
      } else if (ValueAnalysis::getConstantIndex(lhs, constVal)) {
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
      if (ValueAnalysis::getConstantIndex(subOp.getRhs(), constVal)) {
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
