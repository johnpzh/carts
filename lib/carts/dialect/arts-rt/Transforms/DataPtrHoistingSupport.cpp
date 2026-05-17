///==========================================================================///
/// File: DataPtrHoistingSupport.cpp
///
/// Shared local helpers for the split DataPtrHoisting implementation.
///==========================================================================///

#include "carts/dialect/arts-rt/Transforms/DataPtrHoistingInternal.h"
#include "carts/dialect/arts-rt/Utils/RtDbUtils.h"

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(data_ptr_hoisting);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
using namespace mlir::carts::arts_rt;

namespace mlir::carts::arts_rt::data_ptr_hoisting {

using namespace ::mlir::carts::arts;

/// Check if an LLVM load is loading a data pointer from deps struct.
/// Pattern: llvm.load from a GEP that accesses the ptr field (offset 2) of
/// artsEdtDep_t struct.
bool isDepsPtrLoad(LLVM::LoadOp loadOp) {
  Value addr = loadOp.getAddr();

  /// Check if loading a pointer type (ptr -> ptr)
  auto resultType = loadOp.getResult().getType();
  if (!isa<LLVM::LLVMPointerType>(resultType))
    return false;

  /// Check if the address comes from arts.dep_gep
  if (auto defOp = addr.getDefiningOp()) {
    /// arts.dep_gep returns (guid, ptr) - we want loads from the ptr result
    if (dyn_cast<DepGepOp>(defOp))
      return true;

    /// Also check for GEP chains that access struct fields
    if (auto gepOp = dyn_cast<LLVM::GEPOp>(defOp)) {
      /// GEP into a struct type accessing field 2 (ptr field)
      auto sourceType = gepOp.getElemType();
      if (auto structType = dyn_cast<LLVM::LLVMStructType>(sourceType)) {
        /// artsEdtDep_t is struct { i64, i32, ptr, i32, i1 }
        /// Field 2 is the ptr field
        return true;
      }
    }
  }

  return false;
}

/// Check if an LLVM load is loading a datablock pointer from a db_gep.
bool isDbPtrLoad(LLVM::LoadOp loadOp) {
  auto resultType = loadOp.getResult().getType();
  if (!isa<LLVM::LLVMPointerType>(resultType))
    return false;
  if (auto defOp = loadOp.getAddr().getDefiningOp<DbGepOp>())
    return true;
  return false;
}

bool matchGreaterThanConstIcmp(Value cond, Value value, int64_t threshold) {
  auto cmp =
      ValueAnalysis::stripNumericCasts(cond).getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return false;

  Value lhs = ValueAnalysis::stripNumericCasts(cmp.getLhs());
  Value rhs = ValueAnalysis::stripNumericCasts(cmp.getRhs());
  auto rhsConst = RtDbUtils::tryFoldConstantIndex(rhs);
  auto lhsConst = RtDbUtils::tryFoldConstantIndex(lhs);
  Value normalizedValue = ValueAnalysis::stripNumericCasts(value);

  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::sgt:
    return rhsConst &&
           ValueAnalysis::areValuesEquivalent(lhs, normalizedValue) &&
           *rhsConst >= threshold;
  case arith::CmpIPredicate::uge:
  case arith::CmpIPredicate::sge:
    return rhsConst &&
           ValueAnalysis::areValuesEquivalent(lhs, normalizedValue) &&
           *rhsConst > threshold;
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::slt:
    return lhsConst &&
           ValueAnalysis::areValuesEquivalent(rhs, normalizedValue) &&
           *lhsConst >= threshold;
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::sle:
    return lhsConst &&
           ValueAnalysis::areValuesEquivalent(rhs, normalizedValue) &&
           *lhsConst > threshold;
  default:
    return false;
  }
}

bool hasDominatingBlockSizeGuard(scf::ForOp loop, Value blockSize,
                                 int64_t threshold) {
  if (auto blockSizeConst = RtDbUtils::tryFoldConstantIndex(
          ValueAnalysis::stripNumericCasts(blockSize)))
    return *blockSizeConst > threshold;

  for (Operation *current = loop.getOperation(), *parent = loop->getParentOp();
       parent; current = parent, parent = parent->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(parent);
    if (!ifOp)
      continue;

    Region *containingRegion = current->getParentRegion();
    if (containingRegion != &ifOp.getThenRegion())
      continue;

    if (matchGreaterThanConstIcmp(ifOp.getCondition(), blockSize, threshold))
      return true;
  }

  return false;
}

bool getSingleHopOffsetThreshold(scf::ForOp loop, int64_t &threshold) {
  auto lb = RtDbUtils::tryFoldConstantIndex(
      ValueAnalysis::stripNumericCasts(loop.getLowerBound()));
  auto ub = RtDbUtils::tryFoldConstantIndex(
      ValueAnalysis::stripNumericCasts(loop.getUpperBound()));
  auto step = RtDbUtils::tryFoldConstantIndex(
      ValueAnalysis::stripNumericCasts(loop.getStep()));
  if (!lb || !ub || !step || *step != 1 || *ub <= *lb)
    return false;

  int64_t maxNegative = std::max<int64_t>(0, -*lb);
  int64_t maxPositive = std::max<int64_t>(0, *ub - 1);
  threshold = std::max(maxNegative, maxPositive);
  return true;
}

bool matchBlockedNeighborCarryIndex(Value value, scf::ForOp loop,
                                    BlockedNeighborCarryPattern &pattern) {
  pattern = {};
  value = ValueAnalysis::stripNumericCasts(value);

  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    Value trueValue = ValueAnalysis::stripNumericCasts(select.getTrueValue());
    Value falseValue = ValueAnalysis::stripNumericCasts(select.getFalseValue());
    if (ValueAnalysis::isZeroConstant(trueValue) &&
        !ValueAnalysis::dependsOn(select.getCondition(),
                                  loop.getInductionVar())) {
      pattern.forceZeroCond = select.getCondition();
      pattern.forceZeroWhenTrue = true;
      value = falseValue;
    } else if (ValueAnalysis::isZeroConstant(falseValue) &&
               !ValueAnalysis::dependsOn(select.getCondition(),
                                         loop.getInductionVar())) {
      pattern.forceZeroCond = select.getCondition();
      pattern.forceZeroWhenTrue = false;
      value = trueValue;
    }
  }

  if (auto minOp = value.getDefiningOp<arith::MinUIOp>()) {
    Value lhs = ValueAnalysis::stripNumericCasts(minOp.getLhs());
    Value rhs = ValueAnalysis::stripNumericCasts(minOp.getRhs());
    if (!ValueAnalysis::dependsOn(lhs, loop.getInductionVar()) &&
        ValueAnalysis::dependsOn(rhs, loop.getInductionVar())) {
      pattern.clampMax = lhs;
      value = rhs;
    } else if (!ValueAnalysis::dependsOn(rhs, loop.getInductionVar()) &&
               ValueAnalysis::dependsOn(lhs, loop.getInductionVar())) {
      pattern.clampMax = rhs;
      value = lhs;
    }
  }

  Value divResult = value;
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    Value lhs = ValueAnalysis::stripNumericCasts(sub.getLhs());
    Value rhs = ValueAnalysis::stripNumericCasts(sub.getRhs());
    if (ValueAnalysis::dependsOn(lhs, loop.getInductionVar()) &&
        !ValueAnalysis::dependsOn(rhs, loop.getInductionVar())) {
      pattern.blockBaseOffset = rhs;
      divResult = lhs;
    }
  }

  auto div = divResult.getDefiningOp<arith::DivUIOp>();
  if (!div)
    return false;

  Value blockSize = ValueAnalysis::stripNumericCasts(div.getRhs());
  if (!isLoopInvariant(loop, blockSize))
    return false;

  int64_t threshold = 0;
  if (!getSingleHopOffsetThreshold(loop, threshold) ||
      !hasDominatingBlockSizeGuard(loop, blockSize, threshold))
    return false;

  Value baseGlobalIndex;
  if (!matchLoopInvariantAddend(div.getLhs(), loop.getInductionVar(),
                                baseGlobalIndex))
    return false;
  if (baseGlobalIndex && !isLoopInvariant(loop, baseGlobalIndex))
    return false;
  if (pattern.blockBaseOffset &&
      !isLoopInvariant(loop, pattern.blockBaseOffset))
    return false;
  if (pattern.clampMax && !isLoopInvariant(loop, pattern.clampMax))
    return false;
  if (pattern.forceZeroCond && !isLoopInvariant(loop, pattern.forceZeroCond))
    return false;

  pattern.baseGlobalIndex = baseGlobalIndex;
  pattern.blockSize = blockSize;
  return true;
}

bool matchCarryAdjust(Value value, Value &belowCond, Value &aboveCond) {
  value = ValueAnalysis::stripNumericCasts(value);

  auto highSel = value.getDefiningOp<arith::SelectOp>();
  if (!highSel)
    return false;
  if (!ValueAnalysis::isOneConstant(
          ValueAnalysis::stripNumericCasts(highSel.getTrueValue())))
    return false;

  aboveCond = highSel.getCondition();
  Value lowArm = ValueAnalysis::stripNumericCasts(highSel.getFalseValue());

  if (auto lowSel = lowArm.getDefiningOp<arith::SelectOp>()) {
    if (!ValueAnalysis::isConstantEqual(
            ValueAnalysis::stripNumericCasts(lowSel.getTrueValue()), -1) ||
        !ValueAnalysis::isZeroConstant(
            ValueAnalysis::stripNumericCasts(lowSel.getFalseValue())))
      return false;
    belowCond = lowSel.getCondition();
    return true;
  }

  if (ValueAnalysis::isZeroConstant(lowArm)) {
    belowCond = Value();
    return true;
  }

  return false;
}

bool matchBasePlusCarry(Value value, Value &baseIndex, Value &belowCond,
                        Value &aboveCond) {
  value = ValueAnalysis::stripNumericCasts(value);
  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  auto tryOperands = [&](Value maybeBase, Value maybeAdjust) {
    Value localBelow, localAbove;
    if (!matchCarryAdjust(maybeAdjust, localBelow, localAbove))
      return false;
    baseIndex = ValueAnalysis::stripNumericCasts(maybeBase);
    belowCond = localBelow;
    aboveCond = localAbove;
    return true;
  };

  return tryOperands(add.getLhs(), add.getRhs()) ||
         tryOperands(add.getRhs(), add.getLhs());
}

bool matchNeighborCarryIndex(Value value, scf::ForOp loop,
                             NeighborCarryIndexPattern &pattern) {
  pattern = {};
  value = ValueAnalysis::stripNumericCasts(value);

  if (auto forceZero = value.getDefiningOp<arith::SelectOp>()) {
    if (!ValueAnalysis::isZeroConstant(
            ValueAnalysis::stripNumericCasts(forceZero.getTrueValue())))
      return false;
    pattern.forceZeroCond = forceZero.getCondition();
    value = ValueAnalysis::stripNumericCasts(forceZero.getFalseValue());
  }

  auto tryMatch = [&](Value candidate, Value clampMax) {
    Value baseIndex, belowCond, aboveCond;
    if (!matchBasePlusCarry(candidate, baseIndex, belowCond, aboveCond))
      return false;
    if (!isLoopInvariant(loop, baseIndex))
      return false;
    if (clampMax && !isLoopInvariant(loop, clampMax))
      return false;
    if (pattern.forceZeroCond && !isLoopInvariant(loop, pattern.forceZeroCond))
      return false;
    pattern.baseIndex = baseIndex;
    pattern.belowCond = belowCond;
    pattern.aboveCond = aboveCond;
    pattern.clampMax = clampMax;
    return true;
  };

  if (auto minOp = value.getDefiningOp<arith::MinUIOp>())
    return tryMatch(minOp.getLhs(), minOp.getRhs()) ||
           tryMatch(minOp.getRhs(), minOp.getLhs());

  return tryMatch(value, Value());
}

Value buildLoopInvariantI1Not(OpBuilder &builder, Location loc, Value cond) {
  if (!cond)
    return Value();
  Value trueConst =
      arith::ConstantOp::create(builder, loc, builder.getBoolAttr(true));
  return arith::XOrIOp::create(builder, loc, cond, trueConst);
}

Value buildLoopInvariantI1And(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return arith::AndIOp::create(builder, loc, lhs, rhs);
}

Value buildNormalizedForceZeroCond(OpBuilder &builder, Location loc,
                                   const BlockedNeighborCarryPattern &pattern) {
  if (!pattern.forceZeroCond)
    return Value();
  if (pattern.forceZeroWhenTrue)
    return pattern.forceZeroCond;
  return buildLoopInvariantI1Not(builder, loc, pattern.forceZeroCond);
}

Value buildBlockedNeighborCenterIndex(
    OpBuilder &builder, Location loc,
    const BlockedNeighborCarryPattern &pattern, Value zero) {
  Value baseGlobal = pattern.baseGlobalIndex
                         ? ValueAnalysis::ensureIndexType(
                               pattern.baseGlobalIndex, builder, loc)
                         : zero;
  Value blockSize =
      ValueAnalysis::ensureIndexType(pattern.blockSize, builder, loc);
  if (!blockSize)
    return Value();

  Value centerDiv = arith::DivUIOp::create(builder, loc, baseGlobal, blockSize);
  Value centerIndex = centerDiv;
  if (pattern.blockBaseOffset) {
    Value blockBase =
        ValueAnalysis::ensureIndexType(pattern.blockBaseOffset, builder, loc);
    centerIndex = arith::SubIOp::create(builder, loc, centerIndex, blockBase);
  }
  if (pattern.clampMax) {
    Value clampMax =
        ValueAnalysis::ensureIndexType(pattern.clampMax, builder, loc);
    centerIndex = arith::MinUIOp::create(builder, loc, centerIndex, clampMax);
  }
  if (Value forceZeroCond =
          buildNormalizedForceZeroCond(builder, loc, pattern)) {
    centerIndex =
        arith::SelectOp::create(builder, loc, forceZeroCond, zero, centerIndex);
  }
  return centerIndex;
}

bool canHoistBlockedNeighborCacheAcross(
    scf::ForOp loop, int varyingIndex, ArrayRef<Value> indices,
    const BlockedNeighborCarryPattern &pattern) {
  auto invariantTo = [&](Value value) {
    return !value || isLoopInvariant(loop, value);
  };

  if (!invariantTo(pattern.baseGlobalIndex) ||
      !invariantTo(pattern.blockSize) ||
      !invariantTo(pattern.blockBaseOffset) || !invariantTo(pattern.clampMax) ||
      !invariantTo(pattern.forceZeroCond))
    return false;

  for (auto indexedValue : llvm::enumerate(indices)) {
    if (static_cast<int>(indexedValue.index()) == varyingIndex)
      continue;
    if (!invariantTo(indexedValue.value()))
      return false;
  }
  return true;
}

scf::ForOp
findBlockedNeighborCacheHoistLoop(scf::ForOp loop, int varyingIndex,
                                  ArrayRef<Value> indices,
                                  const BlockedNeighborCarryPattern &pattern) {
  scf::ForOp hoistLoop = loop;
  for (Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto outerLoop = dyn_cast<scf::ForOp>(parent);
    if (!outerLoop)
      continue;
    if (!canHoistBlockedNeighborCacheAcross(outerLoop, varyingIndex, indices,
                                            pattern))
      break;
    hoistLoop = outerLoop;
  }
  return hoistLoop;
}

Value buildDepPtrLoad(OpBuilder &builder, Location loc, DepGepOp depGep,
                      ArrayRef<Value> indices) {
  auto newDepGep = DepGepOp::create(
      builder, loc, depGep.getGuid().getType(), depGep.getPtr().getType(),
      depGep.getDepStruct(), depGep.getOffset(), indices, depGep.getStrides());
  return LLVM::LoadOp::create(builder, loc, depGep.getPtr().getType(),
                              newDepGep.getPtr());
}

Value buildGuardedDepPtrLoad(OpBuilder &builder, Location loc,
                             DepGepOp depGep, ArrayRef<Value> indices,
                             Value guard, Value fallbackPtr) {
  if (!guard)
    return buildDepPtrLoad(builder, loc, depGep, indices);

  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{fallbackPtr.getType()},
                                guard, true);

  {
    OpBuilder::InsertionGuard guardRegion(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    Value loaded = buildDepPtrLoad(builder, loc, depGep, indices);
    scf::YieldOp::create(builder, loc, loaded);
  }

  {
    OpBuilder::InsertionGuard guardRegion(builder);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    scf::YieldOp::create(builder, loc, fallbackPtr);
  }

  return ifOp.getResult(0);
}

bool materializeBlockedNeighborPtrCache(LLVM::LoadOp loadOp, scf::ForOp loop) {
  auto depGep = loadOp.getAddr().getDefiningOp<DepGepOp>();
  if (!depGep)
    return false;

  SmallVector<Value> indices(depGep.getIndices().begin(),
                             depGep.getIndices().end());
  if (indices.empty())
    return false;

  int varyingIndex = -1;
  for (auto it : llvm::enumerate(indices)) {
    if (isLoopInvariant(loop, it.value()))
      continue;
    if (varyingIndex != -1)
      return false;
    varyingIndex = static_cast<int>(it.index());
  }
  if (varyingIndex < 0 ||
      static_cast<size_t>(varyingIndex + 1) != indices.size())
    return false;

  BlockedNeighborCarryPattern pattern;
  if (!matchBlockedNeighborCarryIndex(indices.back(), loop, pattern))
    return false;

  scf::ForOp hoistLoop =
      findBlockedNeighborCacheHoistLoop(loop, varyingIndex, indices, pattern);
  Location loc = loadOp.getLoc();
  OpBuilder beforeLoop(hoistLoop);
  Value zero = createConstantIndex(beforeLoop, loc, 0);
  Value one = createConstantIndex(beforeLoop, loc, 1);
  SmallVector<polygeist::Pointer2MemrefOp> ptr2memrefs;
  for (Operation *user :
       llvm::make_early_inc_range(loadOp.getResult().getUsers()))
    if (auto ptr2memref = dyn_cast<polygeist::Pointer2MemrefOp>(user))
      ptr2memrefs.push_back(ptr2memref);
  Value blockSize =
      ValueAnalysis::ensureIndexType(pattern.blockSize, beforeLoop, loc);
  Value baseGlobal = pattern.baseGlobalIndex
                         ? ValueAnalysis::ensureIndexType(
                               pattern.baseGlobalIndex, beforeLoop, loc)
                         : zero;
  Value centerIndex =
      buildBlockedNeighborCenterIndex(beforeLoop, loc, pattern, zero);
  if (!blockSize || !centerIndex)
    return false;

  Value centerDiv =
      arith::DivUIOp::create(beforeLoop, loc, baseGlobal, blockSize);
  Value blockStart =
      arith::MulIOp::create(beforeLoop, loc, centerDiv, blockSize);

  SmallVector<Value> centerIndices(indices.begin(), indices.end());
  centerIndices.back() = centerIndex;
  Value centerPtr = buildDepPtrLoad(beforeLoop, loc, depGep, centerIndices);

  Value normalizedForceZeroCond =
      buildNormalizedForceZeroCond(beforeLoop, loc, pattern);

  Value negPtr = centerPtr;
  {
    SmallVector<Value> negIndices(indices.begin(), indices.end());
    negIndices.back() =
        arith::SubIOp::create(beforeLoop, loc, centerIndex, one);
    Value canUseNeg = arith::CmpIOp::create(
        beforeLoop, loc, arith::CmpIPredicate::ugt, centerIndex, zero);
    if (normalizedForceZeroCond) {
      Value notSingle =
          buildLoopInvariantI1Not(beforeLoop, loc, normalizedForceZeroCond);
      canUseNeg =
          buildLoopInvariantI1And(beforeLoop, loc, canUseNeg, notSingle);
    }
    negPtr = buildGuardedDepPtrLoad(beforeLoop, loc, depGep, negIndices,
                                    canUseNeg, centerPtr);
  }

  Value posPtr = centerPtr;
  if (!pattern.clampMax)
    return false;
  {
    Value clampMax =
        ValueAnalysis::ensureIndexType(pattern.clampMax, beforeLoop, loc);
    SmallVector<Value> posIndices(indices.begin(), indices.end());
    posIndices.back() =
        arith::AddIOp::create(beforeLoop, loc, centerIndex, one);
    Value canUsePos = arith::CmpIOp::create(
        beforeLoop, loc, arith::CmpIPredicate::ult, centerIndex, clampMax);
    if (normalizedForceZeroCond) {
      Value notSingle =
          buildLoopInvariantI1Not(beforeLoop, loc, normalizedForceZeroCond);
      canUsePos =
          buildLoopInvariantI1And(beforeLoop, loc, canUsePos, notSingle);
    }
    posPtr = buildGuardedDepPtrLoad(beforeLoop, loc, depGep, posIndices,
                                    canUsePos, centerPtr);
  }

  OpBuilder atLoad(loadOp);
  Value loopIv =
      ValueAnalysis::ensureIndexType(loop.getInductionVar(), atLoad, loc);
  Value candidateGlobal =
      arith::AddIOp::create(atLoad, loc, baseGlobal, loopIv);
  Value nextBlockStart =
      arith::AddIOp::create(atLoad, loc, blockStart, blockSize);
  Value belowCond = arith::CmpIOp::create(
      atLoad, loc, arith::CmpIPredicate::ult, candidateGlobal, blockStart);
  Value aboveCond = arith::CmpIOp::create(
      atLoad, loc, arith::CmpIPredicate::uge, candidateGlobal, nextBlockStart);
  Value localIndex =
      arith::SubIOp::create(atLoad, loc, candidateGlobal, blockStart);
  Value negLocal = arith::AddIOp::create(atLoad, loc, localIndex, blockSize);
  Value posLocal = arith::SubIOp::create(atLoad, loc, localIndex, blockSize);
  Value selectedLocal =
      arith::SelectOp::create(atLoad, loc, belowCond, negLocal, localIndex);
  selectedLocal =
      arith::SelectOp::create(atLoad, loc, aboveCond, posLocal, selectedLocal);

  Value selectedPtr = centerPtr;
  Type ptrType = loadOp.getType();
  selectedPtr = LLVM::SelectOp::create(atLoad, loc, ptrType, belowCond, negPtr,
                                       selectedPtr);
  selectedPtr = LLVM::SelectOp::create(atLoad, loc, ptrType, aboveCond, posPtr,
                                       selectedPtr);

  loadOp.replaceAllUsesWith(selectedPtr);
  for (auto ptr2memref : ptr2memrefs) {
    if (!ptr2memref)
      continue;
    for (Operation *user :
         llvm::make_early_inc_range(ptr2memref.getResult().getUsers()))
      rewriteBlockedNeighborLocalIndices(user, loop, pattern.baseGlobalIndex,
                                         pattern.blockSize, selectedLocal);
  }
  loadOp.erase();
  if (depGep->use_empty())
    depGep.erase();
  return true;
}

Value buildNeighborCandidateIndex(OpBuilder &builder, Location loc,
                                  const NeighborCarryIndexPattern &pattern,
                                  int64_t adjust) {
  Value result = pattern.baseIndex;
  if (adjust > 0) {
    result = arith::AddIOp::create(builder, loc, result,
                                   createConstantIndex(builder, loc, adjust));
  } else if (adjust < 0) {
    result = arith::SubIOp::create(builder, loc, result,
                                   createConstantIndex(builder, loc, -adjust));
  }
  if (pattern.clampMax)
    result = arith::MinUIOp::create(builder, loc, result, pattern.clampMax);
  if (pattern.forceZeroCond) {
    Value zero = createConstantIndex(builder, loc, 0);
    result = arith::SelectOp::create(builder, loc, pattern.forceZeroCond, zero,
                                     result);
  }
  return result;
}

std::optional<MemoryAccessInfo> getMemoryAccessInfo(Operation *op) {
  MemoryAccessInfo info;

  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    info.memref = load.getMemRef();
    info.indices.assign(load.getIndices().begin(), load.getIndices().end());
    info.resultType = load.getType();
    info.isLoad = true;
    return info;
  }

  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    info.memref = store.getMemRef();
    info.indices.assign(store.getIndices().begin(), store.getIndices().end());
    info.storedValue = store.getValue();
    info.isStore = true;
    return info;
  }

  if (auto dynLoad = dyn_cast<polygeist::DynLoadOp>(op)) {
    info.memref = dynLoad.getMemref();
    info.indices.assign(dynLoad.getIndices().begin(),
                        dynLoad.getIndices().end());
    info.sizes.assign(dynLoad.getSizes().begin(), dynLoad.getSizes().end());
    info.resultType = dynLoad.getType();
    info.isLoad = true;
    return info;
  }

  if (auto dynStore = dyn_cast<polygeist::DynStoreOp>(op)) {
    info.memref = dynStore.getMemref();
    info.indices.assign(dynStore.getIndices().begin(),
                        dynStore.getIndices().end());
    info.sizes.assign(dynStore.getSizes().begin(), dynStore.getSizes().end());
    info.storedValue = dynStore.getValue();
    info.isStore = true;
    return info;
  }

  return std::nullopt;
}

bool matchBlockedNeighborLocalIndex(Value value, scf::ForOp loop,
                                    Value baseGlobalIndex, Value blockSize) {
  value = ValueAnalysis::stripNumericCasts(value);
  auto rem = value.getDefiningOp<arith::RemUIOp>();
  if (!rem)
    return false;

  if (!ValueAnalysis::areValuesEquivalent(
          ValueAnalysis::stripNumericCasts(rem.getRhs()),
          ValueAnalysis::stripNumericCasts(blockSize)))
    return false;

  Value matchedBase;
  if (!matchLoopInvariantAddend(rem.getLhs(), loop.getInductionVar(),
                                matchedBase))
    return false;
  if (!baseGlobalIndex || !matchedBase)
    return !baseGlobalIndex && !matchedBase;
  return ValueAnalysis::areValuesEquivalent(
      ValueAnalysis::stripNumericCasts(matchedBase),
      ValueAnalysis::stripNumericCasts(baseGlobalIndex));
}

bool rewriteBlockedNeighborLocalIndices(Operation *op, scf::ForOp loop,
                                        Value baseGlobalIndex, Value blockSize,
                                        Value replacementIndex) {
  auto memInfoOr = getMemoryAccessInfo(op);
  if (!memInfoOr)
    return false;

  const MemoryAccessInfo &memInfo = *memInfoOr;
  SmallVector<Value> newIndices(memInfo.indices.begin(), memInfo.indices.end());
  bool changed = false;
  for (Value &index : newIndices) {
    if (!matchBlockedNeighborLocalIndex(index, loop, baseGlobalIndex,
                                        blockSize))
      continue;
    index = replacementIndex;
    changed = true;
  }
  if (!changed)
    return false;

  OpBuilder builder(op);
  Location loc = op->getLoc();
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    auto newLoad =
        memref::LoadOp::create(builder, loc, memInfo.memref, newIndices);
    load.replaceAllUsesWith(newLoad.getResult());
    load.erase();
    return true;
  }
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    memref::StoreOp::create(builder, loc, memInfo.storedValue, memInfo.memref,
                            newIndices);
    store.erase();
    return true;
  }
  if (auto dynLoad = dyn_cast<polygeist::DynLoadOp>(op)) {
    auto newLoad =
        polygeist::DynLoadOp::create(builder, loc, memInfo.resultType,
                                     memInfo.memref, newIndices, memInfo.sizes);
    dynLoad.replaceAllUsesWith(newLoad.getResult());
    dynLoad.erase();
    return true;
  }
  if (auto dynStore = dyn_cast<polygeist::DynStoreOp>(op)) {
    polygeist::DynStoreOp::create(builder, loc, memInfo.storedValue,
                                  memInfo.memref, newIndices, memInfo.sizes);
    dynStore.erase();
    return true;
  }
  return false;
}

bool matchSingleBlockBlockedDepIndex(Value value, scf::ForOp loop,
                                     Value &globalIndex, Value &blockSize,
                                     Value &startBlock) {
  startBlock = Value();
  value = ValueAnalysis::stripNumericCasts(value);

  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    Value trueValue = ValueAnalysis::stripNumericCasts(select.getTrueValue());
    Value falseValue = ValueAnalysis::stripNumericCasts(select.getFalseValue());
    if (ValueAnalysis::isZeroConstant(trueValue) &&
        !ValueAnalysis::dependsOn(select.getCondition(),
                                  loop.getInductionVar())) {
      value = falseValue;
    } else if (ValueAnalysis::isZeroConstant(falseValue) &&
               !ValueAnalysis::dependsOn(select.getCondition(),
                                         loop.getInductionVar())) {
      value = trueValue;
    }
  }

  if (auto minOp = value.getDefiningOp<arith::MinUIOp>()) {
    Value lhs = ValueAnalysis::stripNumericCasts(minOp.getLhs());
    Value rhs = ValueAnalysis::stripNumericCasts(minOp.getRhs());
    if (ValueAnalysis::dependsOn(lhs, loop.getInductionVar()) &&
        isLoopInvariant(loop, rhs)) {
      value = lhs;
    } else if (ValueAnalysis::dependsOn(rhs, loop.getInductionVar()) &&
               isLoopInvariant(loop, lhs)) {
      value = rhs;
    }
  }

  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    Value lhs = ValueAnalysis::stripNumericCasts(sub.getLhs());
    Value rhs = ValueAnalysis::stripNumericCasts(sub.getRhs());
    if (!isLoopInvariant(loop, rhs))
      return false;
    startBlock = rhs;
    value = lhs;
  }

  auto div = value.getDefiningOp<arith::DivUIOp>();
  if (!div)
    return false;

  Value lhs = ValueAnalysis::stripNumericCasts(div.getLhs());
  Value rhs = ValueAnalysis::stripNumericCasts(div.getRhs());
  if (!isLoopInvariant(loop, rhs))
    return false;

  globalIndex = lhs;
  blockSize = rhs;
  return true;
}

Value stripInvariantZeroFallback(Value value, scf::ForOp loop) {
  value = ValueAnalysis::stripNumericCasts(value);
  auto select = value.getDefiningOp<arith::SelectOp>();
  if (!select ||
      ValueAnalysis::dependsOn(select.getCondition(), loop.getInductionVar()))
    return value;

  Value trueValue = ValueAnalysis::stripNumericCasts(select.getTrueValue());
  Value falseValue = ValueAnalysis::stripNumericCasts(select.getFalseValue());
  if (ValueAnalysis::isZeroConstant(trueValue))
    return falseValue;
  if (ValueAnalysis::isZeroConstant(falseValue))
    return trueValue;
  return value;
}

bool materializeSingleBlockBlockedDepView(LLVM::LoadOp loadOp,
                                          scf::ForOp loop) {
  auto depGep = loadOp.getAddr().getDefiningOp<DepGepOp>();
  if (!depGep)
    return false;

  SmallVector<Value> depIndices(depGep.getIndices().begin(),
                                depGep.getIndices().end());
  if (depIndices.empty())
    return false;

  int varyingIndex = -1;
  for (auto indexedValue : llvm::enumerate(depIndices)) {
    if (isLoopInvariant(loop, indexedValue.value()))
      continue;
    if (varyingIndex != -1)
      return false;
    varyingIndex = static_cast<int>(indexedValue.index());
  }
  if (varyingIndex < 0)
    return false;

  Value globalIndex;
  Value blockSize;
  Value startBlock;
  if (!matchSingleBlockBlockedDepIndex(depIndices[varyingIndex], loop,
                                       globalIndex, blockSize, startBlock))
    return false;

  Value baseGlobalIndex;
  if (!matchLoopInvariantAddend(globalIndex, loop.getInductionVar(),
                                baseGlobalIndex))
    return false;

  Value replacementIndex;
  if (startBlock) {
    startBlock = stripInvariantZeroFallback(startBlock, loop);
    auto localIndex =
        arts::extractLocalFromBlockBase(globalIndex, startBlock, blockSize);
    if (!localIndex || !arts::isLoopIvBoundedBy(*localIndex, blockSize))
      return false;
    replacementIndex = *localIndex;
  } else {
    if (baseGlobalIndex ||
        !ValueAnalysis::isZeroConstant(
            ValueAnalysis::stripNumericCasts(loop.getLowerBound())) ||
        !arts::isLoopIvBoundedBy(loop.getInductionVar(), blockSize))
      return false;
    replacementIndex = loop.getInductionVar();
  }

  SmallVector<polygeist::Pointer2MemrefOp> ptr2memrefs;
  for (Operation *user :
       llvm::make_early_inc_range(loadOp.getResult().getUsers()))
    if (auto ptr2memref = dyn_cast<polygeist::Pointer2MemrefOp>(user))
      ptr2memrefs.push_back(ptr2memref);

  OpBuilder beforeLoop(loop);
  Location loc = loadOp.getLoc();
  SmallVector<Value> hoistedIndices(depIndices.begin(), depIndices.end());
  hoistedIndices[varyingIndex] = createConstantIndex(beforeLoop, loc, 0);
  Value hoistedPtr = buildDepPtrLoad(beforeLoop, loc, depGep, hoistedIndices);

  loadOp.replaceAllUsesWith(hoistedPtr);
  for (auto ptr2memref : ptr2memrefs) {
    if (!ptr2memref)
      continue;
    for (Operation *user :
         llvm::make_early_inc_range(ptr2memref.getResult().getUsers()))
      rewriteBlockedNeighborLocalIndices(user, loop, baseGlobalIndex, blockSize,
                                         replacementIndex);
  }

  loadOp.erase();
  if (depGep->use_empty())
    depGep.erase();
  return true;
}

bool getSelectOperands(Value value, Value &cond, Value &trueValue,
                       Value &falseValue) {
  value = ValueAnalysis::stripNumericCasts(value);

  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    cond = select.getCondition();
    trueValue = select.getTrueValue();
    falseValue = select.getFalseValue();
    return true;
  }

  if (auto select = value.getDefiningOp<LLVM::SelectOp>()) {
    cond = select.getCondition();
    trueValue = select.getTrueValue();
    falseValue = select.getFalseValue();
    return true;
  }

  return false;
}

bool matchLowerBoundaryCond(Value cond, Value candidate) {
  auto cmp =
      ValueAnalysis::stripNumericCasts(cond).getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return false;

  Value lhs = ValueAnalysis::stripNumericCasts(cmp.getLhs());
  Value rhs = ValueAnalysis::stripNumericCasts(cmp.getRhs());
  Value base = ValueAnalysis::stripNumericCasts(candidate);

  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return ValueAnalysis::areValuesEquivalent(lhs, base) &&
           ValueAnalysis::isZeroConstant(rhs);
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return ValueAnalysis::areValuesEquivalent(rhs, base) &&
           ValueAnalysis::isZeroConstant(lhs);
  default:
    return false;
  }
}

bool matchUpperBoundaryCond(Value cond, Value candidate) {
  auto cmp =
      ValueAnalysis::stripNumericCasts(cond).getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return false;

  Value lhs = ValueAnalysis::stripNumericCasts(cmp.getLhs());
  Value rhs = ValueAnalysis::stripNumericCasts(cmp.getRhs());
  Value base = ValueAnalysis::stripNumericCasts(candidate);

  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return ValueAnalysis::areValuesEquivalent(lhs, base);
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return ValueAnalysis::areValuesEquivalent(rhs, base);
  default:
    return false;
  }
}

bool tryBuildBoundaryIndexChain(Value firstCond, Value firstValue,
                                Value secondCond, Value secondValue,
                                Value middleValue, BoundarySelectChain &chain) {
  Value middle = ValueAnalysis::stripNumericCasts(middleValue);
  if (matchLowerBoundaryCond(firstCond, middle) &&
      matchUpperBoundaryCond(secondCond, middle)) {
    chain.lowCond = firstCond;
    chain.highCond = secondCond;
    chain.lowValue = firstValue;
    chain.midValue = middleValue;
    chain.highValue = secondValue;
    return true;
  }

  if (matchLowerBoundaryCond(secondCond, middle) &&
      matchUpperBoundaryCond(firstCond, middle)) {
    chain.lowCond = secondCond;
    chain.highCond = firstCond;
    chain.lowValue = secondValue;
    chain.midValue = middleValue;
    chain.highValue = firstValue;
    return true;
  }

  return false;
}

bool matchBoundaryIndexPattern(Value value, scf::ForOp loop,
                               BoundaryIndexPattern &pattern) {
  Value outerCond, outerTrue, outerFalse;
  if (!getSelectOperands(value, outerCond, outerTrue, outerFalse))
    return false;

  BoundarySelectChain chain;
  bool matched = false;

  Value innerCond, innerTrue, innerFalse;
  if (getSelectOperands(outerFalse, innerCond, innerTrue, innerFalse)) {
    matched = tryBuildBoundaryIndexChain(outerCond, outerTrue, innerCond,
                                         innerTrue, innerFalse, chain);
  }

  if (!matched &&
      getSelectOperands(outerTrue, innerCond, innerTrue, innerFalse)) {
    matched = tryBuildBoundaryIndexChain(outerCond, outerFalse, innerCond,
                                         innerTrue, innerFalse, chain);
  }

  if (!matched)
    return false;

  int64_t offset = 0;
  Value baseExpr = ValueAnalysis::stripConstantOffset(
      ValueAnalysis::stripNumericCasts(chain.midValue), &offset);
  if (!baseExpr || !ValueAnalysis::dependsOn(baseExpr, loop.getInductionVar()))
    return false;

  pattern.chain = chain;
  pattern.baseExpr = baseExpr;
  pattern.offset = offset;
  return true;
}

bool matchBoundaryPointerChain(Value value, Value lowCond, Value highCond,
                               BoundarySelectChain &chain) {
  Value outerCond, outerTrue, outerFalse;
  if (!getSelectOperands(value, outerCond, outerTrue, outerFalse))
    return false;

  auto sameCond = [&](Value lhs, Value rhs) {
    return ValueAnalysis::areValuesEquivalent(lhs, rhs);
  };

  Value innerCond, innerTrue, innerFalse;
  if (getSelectOperands(outerFalse, innerCond, innerTrue, innerFalse)) {
    if (sameCond(outerCond, highCond) && sameCond(innerCond, lowCond)) {
      chain.lowCond = innerCond;
      chain.highCond = outerCond;
      chain.lowValue = innerTrue;
      chain.midValue = innerFalse;
      chain.highValue = outerTrue;
      return true;
    }
    if (sameCond(outerCond, lowCond) && sameCond(innerCond, highCond)) {
      chain.lowCond = outerCond;
      chain.highCond = innerCond;
      chain.lowValue = outerTrue;
      chain.midValue = innerFalse;
      chain.highValue = innerTrue;
      return true;
    }
  }

  if (getSelectOperands(outerTrue, innerCond, innerTrue, innerFalse)) {
    if (sameCond(outerCond, lowCond) && sameCond(innerCond, highCond)) {
      chain.lowCond = outerCond;
      chain.highCond = innerCond;
      chain.lowValue = outerFalse;
      chain.midValue = innerFalse;
      chain.highValue = innerTrue;
      return true;
    }
    if (sameCond(outerCond, highCond) && sameCond(innerCond, lowCond)) {
      chain.lowCond = innerCond;
      chain.highCond = outerCond;
      chain.lowValue = innerTrue;
      chain.midValue = innerFalse;
      chain.highValue = outerFalse;
      return true;
    }
  }

  return false;
}

std::optional<LoopWindowAccessPattern> matchLoopWindowAccess(Operation *op,
                                                             scf::ForOp loop) {
  auto memInfoOr = getMemoryAccessInfo(op);
  if (!memInfoOr)
    return std::nullopt;

  const MemoryAccessInfo &memInfo = *memInfoOr;
  auto ptr2memref = memInfo.memref.getDefiningOp<polygeist::Pointer2MemrefOp>();

  for (auto indexedValue : llvm::enumerate(memInfo.indices)) {
    BoundaryIndexPattern indexPattern;
    if (!matchBoundaryIndexPattern(indexedValue.value(), loop, indexPattern))
      continue;

    LoopWindowAccessPattern pattern;
    pattern.memInfo = memInfo;
    pattern.ptr2memref = ptr2memref;
    pattern.indexPattern = indexPattern;
    pattern.partitionDim = indexedValue.index();

    BoundarySelectChain ptrChain;
    if (ptr2memref && matchBoundaryPointerChain(
                          ptr2memref.getSource(), indexPattern.chain.lowCond,
                          indexPattern.chain.highCond, ptrChain)) {
      if (!isLoopInvariant(loop, ptrChain.lowValue) ||
          !isLoopInvariant(loop, ptrChain.midValue) ||
          !isLoopInvariant(loop, ptrChain.highValue))
        continue;
      pattern.ptrChain = ptrChain;
      pattern.hasBoundaryPtrChain = true;
      return pattern;
    }

    /// Generic fallback: when the access memref itself is already invariant
    /// across the loop, version the boundary index math even if pointer
    /// selection was resolved earlier. This keeps loop-window versioning
    /// useful after pointer hoisting/caching and avoids baking benchmark-
    /// specific assumptions into lowering.
    if (!isLoopInvariant(loop, memInfo.memref))
      continue;
    return pattern;
  }

  return std::nullopt;
}

Value chooseBoundaryRegionValue(BoundaryLoopRegion region, int64_t offset,
                                const BoundarySelectChain &chain) {
  switch (region) {
  case BoundaryLoopRegion::Left:
    return offset < 0 ? chain.lowValue : chain.midValue;
  case BoundaryLoopRegion::Middle:
    return chain.midValue;
  case BoundaryLoopRegion::Right:
    return offset > 0 ? chain.highValue : chain.midValue;
  }
  llvm_unreachable("unknown boundary loop region");
}

bool rewriteLoopWindowAccess(Operation *op, scf::ForOp loop,
                             BoundaryLoopRegion region) {
  auto patternOr = matchLoopWindowAccess(op, loop);
  if (!patternOr)
    return false;

  LoopWindowAccessPattern pattern = *patternOr;
  Location loc = op->getLoc();
  OpBuilder builder(op);

  Value selectedIndex = chooseBoundaryRegionValue(
      region, pattern.indexPattern.offset, pattern.indexPattern.chain);
  Value newMemref = pattern.memInfo.memref;
  if (pattern.hasBoundaryPtrChain) {
    Value selectedPtr = chooseBoundaryRegionValue(
        region, pattern.indexPattern.offset, pattern.ptrChain);
    newMemref = polygeist::Pointer2MemrefOp::create(
        builder, loc, pattern.ptr2memref.getType(), selectedPtr);
  }

  SmallVector<Value> newIndices(pattern.memInfo.indices.begin(),
                                pattern.memInfo.indices.end());
  newIndices[pattern.partitionDim] = selectedIndex;

  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    auto newLoad = memref::LoadOp::create(builder, loc, newMemref, newIndices);
    load.replaceAllUsesWith(newLoad.getResult());
    load.erase();
  } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
    memref::StoreOp::create(builder, loc, pattern.memInfo.storedValue,
                            newMemref, newIndices);
    store.erase();
  } else if (auto dynLoad = dyn_cast<polygeist::DynLoadOp>(op)) {
    auto newLoad = polygeist::DynLoadOp::create(
        builder, loc, pattern.memInfo.resultType, newMemref, newIndices,
        pattern.memInfo.sizes);
    dynLoad.replaceAllUsesWith(newLoad.getResult());
    dynLoad.erase();
  } else if (auto dynStore = dyn_cast<polygeist::DynStoreOp>(op)) {
    polygeist::DynStoreOp::create(builder, loc, pattern.memInfo.storedValue,
                                  newMemref, newIndices, pattern.memInfo.sizes);
    dynStore.erase();
  } else {
    return false;
  }

  if (pattern.hasBoundaryPtrChain && pattern.ptr2memref &&
      pattern.ptr2memref->use_empty())
    pattern.ptr2memref.erase();
  return true;
}

scf::ForOp cloneLoopWindowSegment(OpBuilder &builder, scf::ForOp loop,
                                  Value lowerBound, Value upperBound) {
  scf::ForOp newLoop = scf::ForOp::create(builder, loop.getLoc(), lowerBound,
                                          upperBound, loop.getStep());
  newLoop->setAttrs(loop->getAttrs());

  Block *oldBody = loop.getBody();
  Block *newBody = newLoop.getBody();

  IRMapping mapping;
  mapping.map(loop.getInductionVar(), newLoop.getInductionVar());

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(newBody);
  for (Operation &op : oldBody->without_terminator())
    builder.clone(op, mapping);

  return newLoop;
}

int rewriteLoopWindowSegment(scf::ForOp loop, BoundaryLoopRegion region) {
  SmallVector<Operation *> ops;
  for (Operation &op : loop.getBody()->without_terminator())
    ops.push_back(&op);

  int rewritten = 0;
  for (Operation *op : ops) {
    if (rewriteLoopWindowAccess(op, loop, region))
      rewritten++;
  }
  return rewritten;
}

bool versionLoopWindowAccesses(scf::ForOp loop, int &rewrittenAccesses) {
  if (!loop || loop.getNumResults() != 0 || !isInnermostLoop(loop))
    return false;

  auto stepConst = RtDbUtils::tryFoldConstantIndex(loop.getStep());
  if (!stepConst || *stepConst != 1) {
    ARTS_DEBUG("versionLoopWindowAccesses: skip loop "
               << loop << " due to non-unit step");
    return false;
  }

  int64_t leftBand = 0;
  int64_t rightBand = 0;
  Value canonicalBase;
  int matchedAccesses = 0;

  for (Operation &op : loop.getBody()->without_terminator()) {
    auto patternOr = matchLoopWindowAccess(&op, loop);
    if (!patternOr)
      continue;

    const LoopWindowAccessPattern &pattern = *patternOr;
    if (!canonicalBase)
      canonicalBase = pattern.indexPattern.baseExpr;
    else if (!ValueAnalysis::areValuesEquivalent(
                 canonicalBase, pattern.indexPattern.baseExpr)) {
      ARTS_DEBUG("versionLoopWindowAccesses: canonical base mismatch in loop "
                 << loop << " current=" << canonicalBase
                 << " incoming=" << pattern.indexPattern.baseExpr);
      continue;
    }

    leftBand = std::max<int64_t>(
        leftBand, std::max<int64_t>(0, -pattern.indexPattern.offset));
    rightBand = std::max<int64_t>(
        rightBand, std::max<int64_t>(0, pattern.indexPattern.offset));
    matchedAccesses++;
  }

  if (matchedAccesses < 3 || (leftBand == 0 && rightBand == 0))
    return false;

  Location loc = loop.getLoc();
  OpBuilder builder(loop);

  Value zero = createConstantIndex(builder, loc, 0);
  Value lb = loop.getLowerBound();
  Value ub = loop.getUpperBound();
  Value tripCount = arith::SubIOp::create(builder, loc, ub, lb);

  Value leftSize = zero;
  if (leftBand > 0) {
    Value leftBandVal = createConstantIndex(builder, loc, leftBand);
    leftSize = arith::MinUIOp::create(builder, loc, tripCount, leftBandVal);
  }

  Value leftUb = arith::AddIOp::create(builder, loc, lb, leftSize);
  Value remainingAfterLeft =
      arith::SubIOp::create(builder, loc, tripCount, leftSize);

  Value middleSize = remainingAfterLeft;
  if (rightBand > 0) {
    Value rightBandVal = createConstantIndex(builder, loc, rightBand);
    Value hasInterior =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt,
                              remainingAfterLeft, rightBandVal);
    Value trimmed =
        arith::SubIOp::create(builder, loc, remainingAfterLeft, rightBandVal);
    middleSize =
        arith::SelectOp::create(builder, loc, hasInterior, trimmed, zero);
  }

  Value middleUb = arith::AddIOp::create(builder, loc, leftUb, middleSize);

  scf::ForOp leftLoop = cloneLoopWindowSegment(builder, loop, lb, leftUb);
  scf::ForOp middleLoop =
      cloneLoopWindowSegment(builder, loop, leftUb, middleUb);
  scf::ForOp rightLoop = cloneLoopWindowSegment(builder, loop, middleUb, ub);

  rewrittenAccesses +=
      rewriteLoopWindowSegment(leftLoop, BoundaryLoopRegion::Left);
  rewrittenAccesses +=
      rewriteLoopWindowSegment(middleLoop, BoundaryLoopRegion::Middle);
  rewrittenAccesses +=
      rewriteLoopWindowSegment(rightLoop, BoundaryLoopRegion::Right);

  loop.erase();
  return true;
}

bool materializeNeighborPtrCache(LLVM::LoadOp loadOp, scf::ForOp loop) {
  auto depGep = loadOp.getAddr().getDefiningOp<DepGepOp>();
  if (!depGep)
    return false;

  SmallVector<Value> indices(depGep.getIndices().begin(),
                             depGep.getIndices().end());
  if (indices.empty())
    return false;

  int varyingIndex = -1;
  for (auto it : llvm::enumerate(indices)) {
    if (isLoopInvariant(loop, it.value()))
      continue;
    if (varyingIndex != -1)
      return false;
    varyingIndex = static_cast<int>(it.index());
  }
  if (varyingIndex < 0 ||
      static_cast<size_t>(varyingIndex + 1) != indices.size())
    return false;

  NeighborCarryIndexPattern pattern;
  if (!matchNeighborCarryIndex(indices.back(), loop, pattern) ||
      (!pattern.aboveCond && !pattern.belowCond))
    return materializeBlockedNeighborPtrCache(loadOp, loop);

  Location loc = loadOp.getLoc();
  OpBuilder beforeLoop(loop);
  Value zero = createConstantIndex(beforeLoop, loc, 0);

  SmallVector<Value> centerIndices(indices.begin(), indices.end());
  centerIndices.back() =
      buildNeighborCandidateIndex(beforeLoop, loc, pattern, /*adjust=*/0);
  Value centerPtr = buildDepPtrLoad(beforeLoop, loc, depGep, centerIndices);

  Value negPtr = centerPtr;
  if (pattern.belowCond) {
    SmallVector<Value> negIndices(indices.begin(), indices.end());
    negIndices.back() =
        buildNeighborCandidateIndex(beforeLoop, loc, pattern, /*adjust=*/-1);
    Value canUseNeg = arith::CmpIOp::create(
        beforeLoop, loc, arith::CmpIPredicate::ugt, pattern.baseIndex, zero);
    if (pattern.forceZeroCond) {
      Value notSingle =
          buildLoopInvariantI1Not(beforeLoop, loc, pattern.forceZeroCond);
      canUseNeg =
          buildLoopInvariantI1And(beforeLoop, loc, canUseNeg, notSingle);
    }
    negPtr = buildGuardedDepPtrLoad(beforeLoop, loc, depGep, negIndices,
                                    canUseNeg, centerPtr);
  }

  Value posPtr = centerPtr;
  if (pattern.aboveCond) {
    if (!pattern.clampMax)
      return false;
    SmallVector<Value> posIndices(indices.begin(), indices.end());
    posIndices.back() =
        buildNeighborCandidateIndex(beforeLoop, loc, pattern, /*adjust=*/1);
    Value canUsePos =
        arith::CmpIOp::create(beforeLoop, loc, arith::CmpIPredicate::ult,
                              pattern.baseIndex, pattern.clampMax);
    if (pattern.forceZeroCond) {
      Value notSingle =
          buildLoopInvariantI1Not(beforeLoop, loc, pattern.forceZeroCond);
      canUsePos =
          buildLoopInvariantI1And(beforeLoop, loc, canUsePos, notSingle);
    }
    posPtr = buildGuardedDepPtrLoad(beforeLoop, loc, depGep, posIndices,
                                    canUsePos, centerPtr);
  }

  OpBuilder atLoad(loadOp);
  Value selectedPtr = centerPtr;
  Type ptrType = loadOp.getType();
  if (pattern.belowCond)
    selectedPtr = LLVM::SelectOp::create(
        atLoad, loc, ptrType, pattern.belowCond, negPtr, selectedPtr);
  if (pattern.aboveCond)
    selectedPtr = LLVM::SelectOp::create(
        atLoad, loc, ptrType, pattern.aboveCond, posPtr, selectedPtr);

  loadOp.replaceAllUsesWith(selectedPtr);
  loadOp.erase();
  if (depGep->use_empty())
    depGep.erase();
  return true;
}

/// Find the highest loop that can legally hoist a pure, operand-only op.
/// This complements load hoisting for materializations like
/// polygeist.pointer2memref whose sole source pointer has already been hoisted
/// or was defined outside the loop nest to begin with.
scf::ForOp findInvariantOpHoistTarget(Operation *op, DominanceInfo &domInfo) {
  scf::ForOp target = nullptr;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      continue;
    Region &loopRegion = loop.getRegion();
    if (!allOperandsDefinedOutside(op, loopRegion))
      break;
    if (!allOperandsDominate(op, loop, domInfo))
      break;
    target = loop;
  }
  return target;
}

/// Hoist a load and its address computation (GEP) out of loops.
bool hoistLoadOutOfLoop(LLVM::LoadOp loadOp, scf::ForOp targetLoop) {
  /// Get the address defining op (should be a GEP or arts op)
  Value addr = loadOp.getAddr();
  Operation *addrOp = addr.getDefiningOp();

  if (!addrOp)
    return false;
  if (!targetLoop || !targetLoop->isAncestor(loadOp))
    return false;

  if (targetLoop->isAncestor(addrOp))
    addrOp->moveBefore(targetLoop);
  loadOp->moveBefore(targetLoop);

  ARTS_INFO("Hoisted data pointer load: " << loadOp);
  return true;
}

} // namespace mlir::carts::arts_rt::data_ptr_hoisting
