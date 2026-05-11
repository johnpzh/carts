///==========================================================================///
/// File: EdtReductionLowering.cpp
///
/// Shared reduction-lowering helpers used by ForLowering.
///
/// Transformation:
///   BEFORE:
///     arts.for ... reductions(%acc = ...)
///
///   AFTER (conceptual):
///     1) Allocate partial accumulator array [numWorkers]
///     2) Each task EDT writes partial[workerId]
///     3) Result EDT combines partials -> final accumulator DB
///     4) Final result is exposed to host/result users
///
/// The reduction flow preserves the same dependence ordering via EDT creation
/// and dependency recording in downstream lowering.
///==========================================================================///

#include "arts/dialect/core/Transforms/edt/EdtReductionLowering.h"
#include "arts/dialect/core/Transforms/edt/WorkDistributionUtils.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/Debug.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <cassert>

ARTS_DEBUG_SETUP(edt_reduction_lowering);

using namespace mlir;
using namespace mlir::arts;

namespace {

static Block &ensureBlock(Region &region) {
  if (region.empty())
    region.push_back(new Block());
  return region.front();
}

static bool isScalarReductionCarrierType(MemRefType memrefType) {
  if (!memrefType)
    return false;
  if (memrefType.getRank() == 0)
    return true;
  return memrefType.getRank() == 1;
}

static ReductionLoweringStrategy resolveReductionStrategy(ForOp forOp,
                                                          EdtOp parallelEdt) {
  auto getStrategy =
      [](Operation *op) -> std::optional<ReductionLoweringStrategy> {
    if (!op)
      return std::nullopt;

    auto attr = op->getAttrOfType<StringAttr>(
        AttrNames::Operation::Contract::ReductionStrategy);
    if (!attr)
      return std::nullopt;

    StringRef value = attr.getValue();
    namespace ReductionStrategyValue =
        AttrNames::Operation::Contract::ReductionStrategyValue;
    if (value == ReductionStrategyValue::Atomic)
      return ReductionLoweringStrategy::atomic;
    if (value == ReductionStrategyValue::Tree)
      return ReductionLoweringStrategy::tree;
    if (value == ReductionStrategyValue::LocalAccumulate)
      return ReductionLoweringStrategy::localAccumulate;
    return std::nullopt;
  };

  if (auto strategy = getStrategy(forOp.getOperation()))
    return *strategy;
  if (auto strategy = getStrategy(parallelEdt.getOperation()))
    return *strategy;
  return ReductionLoweringStrategy::localAccumulate;
}

static SmallVector<Value> detectPrivateReductionAccumulators(ForOp forOp) {
  SmallVector<Value> privateAccums(forOp.getReductionAccumulators().size(),
                                   Value());
  if (forOp.getReductionAccumulators().size() != 1)
    return privateAccums;

  Value reductionAccum = ValueAnalysis::stripMemrefViewOps(
      forOp.getReductionAccumulators().front());
  auto reductionType = dyn_cast<MemRefType>(reductionAccum.getType());
  if (!isScalarReductionCarrierType(reductionType))
    return privateAccums;

  struct AccessInfo {
    bool hasLoad = false;
    bool hasStore = false;
  };

  DenseMap<Value, AccessInfo> candidateAccesses;
  auto recordCandidate = [&](Value memref, bool isStore) {
    Value base = ValueAnalysis::stripMemrefViewOps(memref);
    if (!base || base == reductionAccum)
      return;

    auto allocaOp = base.getDefiningOp<memref::AllocaOp>();
    if (!allocaOp || forOp->isAncestor(allocaOp))
      return;

    auto memrefType = dyn_cast<MemRefType>(base.getType());
    if (!isScalarReductionCarrierType(memrefType) ||
        memrefType.getElementType() != reductionType.getElementType())
      return;

    AccessInfo &access = candidateAccesses[base];
    if (isStore)
      access.hasStore = true;
    else
      access.hasLoad = true;
  };

  forOp.walk([&](memref::LoadOp loadOp) {
    recordCandidate(loadOp.getMemref(), /*isStore=*/false);
  });
  forOp.walk([&](memref::StoreOp storeOp) {
    recordCandidate(storeOp.getMemRef(), /*isStore=*/true);
  });

  SmallVector<Value> candidates;
  for (const auto &entry : candidateAccesses) {
    if (entry.second.hasLoad && entry.second.hasStore)
      candidates.push_back(entry.first);
  }

  if (candidates.size() == 1)
    privateAccums[0] = candidates.front();

  return privateAccums;
}

static SmallVector<Value, 1> getScalarIndices(ArtsCodegen *AC, Value memref,
                                              Location loc) {
  SmallVector<Value, 1> indices;
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType || memrefType.getRank() == 0)
    return indices;
  indices.push_back(AC->createIndexConstant(0, loc));
  return indices;
}

static Value loadScalarValue(ArtsCodegen *AC, Location loc, Value memref) {
  SmallVector<Value, 1> indices = getScalarIndices(AC, memref, loc);
  return AC->create<memref::LoadOp>(loc, memref, indices);
}

static void storeScalarValue(ArtsCodegen *AC, Location loc, Value value,
                             Value memref) {
  SmallVector<Value, 1> indices = getScalarIndices(AC, memref, loc);
  AC->create<memref::StoreOp>(loc, value, memref, indices);
}

static Value createReductionCombiner(ArtsCodegen *AC, Location loc,
                                     Type elemType, Value lhs, Value rhs,
                                     ReductionCombinerKind kind) {
  bool isFloat = isa<FloatType>(elemType);
  switch (kind) {
  case ReductionCombinerKind::add:
    return isFloat ? AC->create<arith::AddFOp>(loc, lhs, rhs).getResult()
                   : AC->create<arith::AddIOp>(loc, lhs, rhs).getResult();
  case ReductionCombinerKind::mul:
    return isFloat ? AC->create<arith::MulFOp>(loc, lhs, rhs).getResult()
                   : AC->create<arith::MulIOp>(loc, lhs, rhs).getResult();
  case ReductionCombinerKind::min:
    return isFloat ? AC->create<arith::MinimumFOp>(loc, lhs, rhs).getResult()
                   : AC->create<arith::MinSIOp>(loc, lhs, rhs).getResult();
  case ReductionCombinerKind::max:
    return isFloat ? AC->create<arith::MaximumFOp>(loc, lhs, rhs).getResult()
                   : AC->create<arith::MaxSIOp>(loc, lhs, rhs).getResult();
  case ReductionCombinerKind::land:
    assert(!isFloat && "bitwise reductions require integer element types");
    return AC->create<arith::AndIOp>(loc, lhs, rhs).getResult();
  case ReductionCombinerKind::lor:
    assert(!isFloat && "bitwise reductions require integer element types");
    return AC->create<arith::OrIOp>(loc, lhs, rhs).getResult();
  case ReductionCombinerKind::lxor:
    assert(!isFloat && "bitwise reductions require integer element types");
    return AC->create<arith::XOrIOp>(loc, lhs, rhs).getResult();
  }
  llvm_unreachable("unknown reduction combiner kind");
}

/// Create the identity value for a reduction combiner.
/// add/or/xor: 0, mul: 1, min: max representable, max: min representable,
/// and: all-ones.
static Value createReductionIdentity(ArtsCodegen *AC, Type elemType,
                                     Location loc, ReductionCombinerKind kind) {
  if (kind == ReductionCombinerKind::add)
    return AC->createZeroValue(elemType, loc);

  if (auto intTy = dyn_cast<IntegerType>(elemType)) {
    unsigned width = intTy.getWidth();
    switch (kind) {
    case ReductionCombinerKind::mul:
      return AC->create<arith::ConstantIntOp>(loc, 1, width);
    case ReductionCombinerKind::min: {
      APInt maxVal = APInt::getSignedMaxValue(width);
      return AC->create<arith::ConstantIntOp>(loc, maxVal.getSExtValue(),
                                              width);
    }
    case ReductionCombinerKind::max: {
      APInt minVal = APInt::getSignedMinValue(width);
      return AC->create<arith::ConstantIntOp>(loc, minVal.getSExtValue(),
                                              width);
    }
    case ReductionCombinerKind::land:
      return AC->create<arith::ConstantIntOp>(loc, -1, width);
    case ReductionCombinerKind::lor:
    case ReductionCombinerKind::lxor:
      return AC->createZeroValue(elemType, loc);
    default:
      return AC->createZeroValue(elemType, loc);
    }
  }

  if (auto floatTy = dyn_cast<FloatType>(elemType)) {
    const auto &sem = floatTy.getFloatSemantics();
    switch (kind) {
    case ReductionCombinerKind::mul: {
      APFloat one(sem, 1);
      return AC->create<arith::ConstantFloatOp>(loc, floatTy, one);
    }
    case ReductionCombinerKind::min: {
      APFloat posInf = APFloat::getInf(sem, /*Negative=*/false);
      return AC->create<arith::ConstantFloatOp>(loc, floatTy, posInf);
    }
    case ReductionCombinerKind::max: {
      APFloat negInf = APFloat::getInf(sem, /*Negative=*/true);
      return AC->create<arith::ConstantFloatOp>(loc, floatTy, negInf);
    }
    default:
      return AC->createZeroValue(elemType, loc);
    }
  }

  return AC->createZeroValue(elemType, loc);
}

static void lowerLinearResultCombine(ArtsCodegen *AC, Location loc,
                                     Value partialArg, Value finalMemRef,
                                     Type elemType, Value zeroIndex,
                                     Value oneIndex, Value numWorkers,
                                     bool isSingleWorker,
                                     ReductionCombinerKind combinerKind) {
  if (isSingleWorker) {
    Value workerSlot =
        AC->create<DbRefOp>(loc, partialArg, SmallVector<Value>{zeroIndex});
    Value partialValue = loadScalarValue(AC, loc, workerSlot);
    storeScalarValue(AC, loc, partialValue, finalMemRef);
    return;
  }

  Value identity = createReductionIdentity(AC, elemType, loc, combinerKind);
  auto combineLoop = AC->create<scf::ForOp>(loc, zeroIndex, numWorkers,
                                            oneIndex, ValueRange{identity});
  AC->setInsertionPointToStart(combineLoop.getBody());
  Value workerIdx = combineLoop.getInductionVar();
  Value accumulator = combineLoop.getRegionIterArg(0);

  Value workerSlot =
      AC->create<DbRefOp>(loc, partialArg, SmallVector<Value>{workerIdx});
  Value partialValue = loadScalarValue(AC, loc, workerSlot);
  Value combined = createReductionCombiner(AC, loc, elemType, accumulator,
                                           partialValue, combinerKind);
  AC->create<scf::YieldOp>(loc, combined);

  AC->setInsertionPointAfter(combineLoop);
  storeScalarValue(AC, loc, combineLoop.getResult(0), finalMemRef);
}

static void lowerAtomicResultCombine(ArtsCodegen *AC, Location loc,
                                     Value partialArg, Value finalMemRef,
                                     Type elemType, Value zeroIndex,
                                     Value oneIndex, Value numWorkers,
                                     bool isSingleWorker,
                                     ReductionCombinerKind combinerKind) {
  // Atomic path only supports integer add via AtomicAddOp; all others
  // fall back to the linear combine path.
  if (!isa<IntegerType>(elemType) ||
      combinerKind != ReductionCombinerKind::add) {
    lowerLinearResultCombine(AC, loc, partialArg, finalMemRef, elemType,
                             zeroIndex, oneIndex, numWorkers, isSingleWorker,
                             combinerKind);
    return;
  }

  Value identity = createReductionIdentity(AC, elemType, loc, combinerKind);
  storeScalarValue(AC, loc, identity, finalMemRef);

  if (isSingleWorker) {
    Value workerSlot =
        AC->create<DbRefOp>(loc, partialArg, SmallVector<Value>{zeroIndex});
    Value partialValue = loadScalarValue(AC, loc, workerSlot);
    (void)AC->create<AtomicAddOp>(loc, finalMemRef, partialValue);
    return;
  }

  auto combineLoop =
      AC->create<scf::ForOp>(loc, zeroIndex, numWorkers, oneIndex);
  AC->setInsertionPointToStart(combineLoop.getBody());
  Value workerIdx = combineLoop.getInductionVar();
  Value workerSlot =
      AC->create<DbRefOp>(loc, partialArg, SmallVector<Value>{workerIdx});
  Value partialValue = loadScalarValue(AC, loc, workerSlot);
  (void)AC->create<AtomicAddOp>(loc, finalMemRef, partialValue);
  AC->setInsertionPointAfter(combineLoop);
}

static void lowerTreeResultCombine(ArtsCodegen *AC, Location loc,
                                   Value partialArg, Value finalMemRef,
                                   Type elemType, Value zeroIndex,
                                   Value oneIndex, Value numWorkers,
                                   bool isSingleWorker,
                                   ReductionCombinerKind combinerKind) {
  if (isSingleWorker) {
    Value workerSlot =
        AC->create<DbRefOp>(loc, partialArg, SmallVector<Value>{zeroIndex});
    Value partialValue = loadScalarValue(AC, loc, workerSlot);
    storeScalarValue(AC, loc, partialValue, finalMemRef);
    return;
  }

  Value twoIndex = AC->createIndexConstant(2, loc);
  Value identity = createReductionIdentity(AC, elemType, loc, combinerKind);
  auto pairLoop = AC->create<scf::ForOp>(loc, zeroIndex, numWorkers, twoIndex,
                                         ValueRange{identity});
  AC->setInsertionPointToStart(pairLoop.getBody());
  Value pairIdx = pairLoop.getInductionVar();
  Value accumulator = pairLoop.getRegionIterArg(0);
  Value rhsIdx = AC->create<arith::AddIOp>(loc, pairIdx, oneIndex);
  Value hasRhs = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                           rhsIdx, numWorkers);

  Value lhsSlot =
      AC->create<DbRefOp>(loc, partialArg, SmallVector<Value>{pairIdx});
  Value lhsValue = loadScalarValue(AC, loc, lhsSlot);
  auto rhsIf = AC->create<scf::IfOp>(loc, TypeRange{lhsValue.getType()}, hasRhs,
                                     /*withElseRegion=*/true);
  AC->setInsertionPointToStart(&ensureBlock(rhsIf.getThenRegion()));
  Value rhsSlot =
      AC->create<DbRefOp>(loc, partialArg, SmallVector<Value>{rhsIdx});
  Value rhsValue = loadScalarValue(AC, loc, rhsSlot);
  AC->create<scf::YieldOp>(loc, rhsValue);
  AC->setInsertionPointToStart(&ensureBlock(rhsIf.getElseRegion()));
  AC->create<scf::YieldOp>(loc, identity);

  AC->setInsertionPointToEnd(pairLoop.getBody());
  Value pairValue = createReductionCombiner(AC, loc, elemType, lhsValue,
                                            rhsIf.getResult(0), combinerKind);
  Value combined = createReductionCombiner(AC, loc, elemType, accumulator,
                                           pairValue, combinerKind);
  AC->create<scf::YieldOp>(loc, combined);

  AC->setInsertionPointAfter(pairLoop);
  storeScalarValue(AC, loc, pairLoop.getResult(0), finalMemRef);
}

} // namespace

DenseSet<Value> mlir::arts::detectReductionBlockArgs(ForOp forOp) {
  DenseSet<Value> result;
  forOp.walk([&](memref::StoreOp store) {
    if (auto dbRef = store.getMemRef().getDefiningOp<DbRefOp>()) {
      if (auto blockArg = dyn_cast<BlockArgument>(dbRef.getSource())) {
        // Only flag as a reduction accumulator if the stored value is derived
        // from a load of the same memref (accumulation pattern: load → op →
        // store). Stores of constants or unrelated values (e.g., boolean flags)
        // are not reduction accumulators.
        Value storedVal = store.getValueToStore();
        bool isAccumulation = false;
        SmallVector<Value, 8> worklist{storedVal};
        SmallPtrSet<Value, 8> visited;
        while (!worklist.empty()) {
          Value v = worklist.pop_back_val();
          if (!visited.insert(v).second)
            continue;
          if (auto load = v.getDefiningOp<memref::LoadOp>()) {
            if (load.getMemref() == store.getMemRef()) {
              isAccumulation = true;
              break;
            }
          }
          if (auto *defOp = v.getDefiningOp()) {
            for (Value operand : defOp->getOperands())
              worklist.push_back(operand);
          }
        }
        if (isAccumulation) {
          result.insert(blockArg);
          ARTS_DEBUG(
              "  - Detected old accumulator block arg in store: " << blockArg);
        }
      }
    }
  });
  return result;
}

bool mlir::arts::shouldSkipReductionArg(
    BlockArgument parallelArg, const ReductionLoweringInfo &redInfo,
    const DenseSet<Value> &reductionBlockArgs) {
  /// Final result accumulator - only needed by result EDT
  if (llvm::is_contained(redInfo.finalResultArgs, parallelArg)) {
    ARTS_DEBUG("  - Skipping final result accumulator (only for result EDT)");
    return true;
  }

  /// Partial accumulator - handled separately with full array acquisition
  if (llvm::is_contained(redInfo.partialAccumArgs, parallelArg)) {
    ARTS_DEBUG("  - Skipping partial accumulator (acquired separately)");
    return true;
  }

  /// Old accumulator from OpenMP lowering - replaced by partial accumulators
  if (reductionBlockArgs.contains(parallelArg)) {
    ARTS_DEBUG("  - Skipping old accumulator (replaced by worker-based "
               "partials)");
    return true;
  }

  return false;
}

void mlir::arts::collectOldAccumulatorDbRefs(
    ForOp forOp, Block &parallelBlock,
    const DenseSet<Value> &reductionBlockArgs, DenseSet<Operation *> &opsToSkip,
    IRMapping &mapper, Value myAccumulator) {
  forOp.walk([&](DbRefOp dbRef) {
    if (auto blockArg = dyn_cast<BlockArgument>(dbRef.getSource())) {
      if (blockArg.getOwner() == &parallelBlock &&
          reductionBlockArgs.contains(blockArg)) {
        mapper.map(dbRef.getResult(), myAccumulator);
        opsToSkip.insert(dbRef.getOperation());
        ARTS_DEBUG(
            "  - Will skip cloning old accumulator db_ref, map to worker slot");
      }
    }
  });
}

ReductionLoweringInfo mlir::arts::allocatePartialAccumulators(
    ArtsCodegen *AC, ForOp forOp, EdtOp parallelEdt, Location loc,
    bool splitMode, Value workerCountOverride) {
  ReductionLoweringInfo redInfo;
  redInfo.loopLocation = forOp.getLoc();
  redInfo.parentConcurrency = parallelEdt.getConcurrency();
  redInfo.strategy = resolveReductionStrategy(forOp, parallelEdt);
  ValueRange reductionAccums = forOp.getReductionAccumulators();
  if (reductionAccums.empty())
    return redInfo;

  // Resolve combiner kinds from the arts.reduction_kinds attr forwarded by
  // ConvertSdeToArts.  Fall back to add when the attr is absent or shorter.
  if (auto kindsAttr = forOp->getAttrOfType<ArrayAttr>(
          AttrNames::Operation::Contract::ReductionKinds)) {
    for (auto attr : kindsAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        int32_t val = intAttr.getInt();
        if (val >= 0 && val <= 6)
          redInfo.combinerKinds.push_back(
              static_cast<ReductionCombinerKind>(val));
        else
          redInfo.combinerKinds.push_back(ReductionCombinerKind::add);
      } else {
        redInfo.combinerKinds.push_back(ReductionCombinerKind::add);
      }
    }
  }
  // Pad with add for any missing entries.
  while (redInfo.combinerKinds.size() < reductionAccums.size())
    redInfo.combinerKinds.push_back(ReductionCombinerKind::add);
  SmallVector<Value> privateReductionAccums =
      detectPrivateReductionAccumulators(forOp);

  /// Try to find a stack-allocated DB sink in the parent block to mirror the
  /// final reduction result (used by host prints).
  if (!forOp->getBlock()->empty()) {
    for (Operation &op : forOp->getBlock()->getOperations()) {
      if (auto dbAlloc = dyn_cast<DbAllocOp>(&op)) {
        if (dbAlloc.getAllocType() == DbAllocType::stack &&
            dbAlloc.getDbMode() == DbMode::write) {
          redInfo.hostResultPtrs.push_back(dbAlloc.getPtr());
          break;
        }
      }
      if (&op == forOp.getOperation())
        break;
    }
  }

  ARTS_INFO(" - Allocating partial accumulators for " << reductionAccums.size()
                                                      << " reduction(s)");

  OpBuilder::InsertionGuard IG(AC->getBuilder());
  AC->setInsertionPoint(parallelEdt);

  /// Resolve worker count consistently with ForLowering distribution.
  /// Prefer explicit EDT worker annotations to avoid context-sensitive
  /// runtime-query cloning when outlining result EDT bodies.
  Value numWorkers;
  if (workerCountOverride) {
    numWorkers = workerCountOverride;
  } else if (auto explicitWorkers =
                 arts::getWorkers(parallelEdt.getOperation());
             explicitWorkers && *explicitWorkers > 0) {
    numWorkers = AC->createIndexConstant(*explicitWorkers, loc);
  } else {
    numWorkers = WorkDistributionUtils::getTotalWorkers(AC, loc, parallelEdt);
  }

  Block &parallelBlock = parallelEdt.getBody().front();
  ValueRange parentDeps = parallelEdt.getDependencies();

  /// Detect existing reduction handles inside the parallel EDT.
  DenseSet<Value> reductionBlockArgs = detectReductionBlockArgs(forOp);
  SmallVector<BlockArgument> existingResultArgs;
  SmallVector<Value> existingResultPtrs;

  for (auto [idx, dep] : llvm::enumerate(parentDeps)) {
    BlockArgument arg = parallelBlock.getArgument(idx);
    if (reductionBlockArgs.contains(arg)) {
      existingResultArgs.push_back(arg);
      existingResultPtrs.push_back(dep);
    }
  }

  Value routeZero = AC->create<arith::ConstantIntOp>(loc, 0, 32);
  Value sizeOne = AC->createIndexConstant(1, loc);
  Value zeroIndex = AC->createIndexConstant(0, loc);
  auto numWorkersConst = ValueAnalysis::tryFoldConstantIndex(numWorkers);
  bool isSingleWorker = numWorkersConst && *numWorkersConst == 1;
  bool canCombineDirectly =
      splitMode && isSingleWorker &&
      llvm::all_of(existingResultPtrs, [](Value ptr) {
        return static_cast<bool>(DbUtils::traceToDbAlloc(ptr));
      });
  redInfo.combineDirectlyInTask = canCombineDirectly;

  for (auto [idx, redVar] : llvm::enumerate(reductionAccums)) {
    redInfo.reductionVars.push_back(redVar);
    redInfo.privateReductionAccums.push_back(idx < privateReductionAccums.size()
                                                 ? privateReductionAccums[idx]
                                                 : Value());

    auto redMemRef = dyn_cast<MemRefType>(redVar.getType());
    Type elementType =
        redMemRef ? redMemRef.getElementType() : redVar.getType();
    ReductionCombinerKind combKind = redInfo.combinerKinds[idx];
    Value identity = createReductionIdentity(AC, elementType, loc, combKind);
    if (!identity) {
      ARTS_ERROR("Unsupported reduction element type for initialization");
      continue;
    }

    /// Step 1: Reuse existing final result DB if available, otherwise fallback.
    if (idx < existingResultArgs.size()) {
      redInfo.finalResultArgs.push_back(existingResultArgs[idx]);

      if (idx < existingResultPtrs.size()) {
        Value existingPtr = existingResultPtrs[idx];
        auto allocInfo = DbUtils::traceToDbAlloc(existingPtr);
        if (allocInfo) {
          auto [rootGuid, rootPtr] = *allocInfo;
          redInfo.finalResultGuids.push_back(rootGuid);
          redInfo.finalResultPtrs.push_back(rootPtr);
          redInfo.finalResultIsExternal.push_back(true);
          ARTS_DEBUG("  - Traced reduction result to DbAllocOp from parent");
        } else {
          redInfo.finalResultPtrs.push_back(existingPtr);
          redInfo.finalResultIsExternal.push_back(true);
          ARTS_DEBUG(
              "  - Reusing pre-allocated reduction result datablock from "
              "parent (acquire ptr)");
        }
      }
      ARTS_DEBUG(
          "  - Reusing pre-allocated reduction result datablock from parent");
    } else {
      auto finalAllocOp = AC->create<DbAllocOp>(
          loc, ArtsMode::inout, routeZero, DbAllocType::heap, DbMode::write,
          elementType, Value(), ValueRange{sizeOne}, ValueRange{sizeOne});
      Value finalGuid = finalAllocOp.getResult(0);
      Value finalPtr = finalAllocOp.getResult(1);

      SmallVector<Value> finalInitIndices{zeroIndex};
      Value finalMemRef = AC->create<DbRefOp>(loc, finalPtr, finalInitIndices);
      SmallVector<Value> innerIndices{zeroIndex};
      AC->create<memref::StoreOp>(loc, identity, finalMemRef, innerIndices);

      redInfo.finalResultGuids.push_back(finalGuid);
      redInfo.finalResultPtrs.push_back(finalPtr);
      redInfo.finalResultIsExternal.push_back(false);

      if (!splitMode) {
        auto finalDepAcqOp = AC->create<DbAcquireOp>(
            loc, ArtsMode::inout, finalGuid, finalPtr, PartitionMode::coarse,
            /*indices=*/SmallVector<Value>{},
            /*offsets=*/SmallVector<Value>{},
            /*sizes=*/SmallVector<Value>{},
            /*partition_indices=*/SmallVector<Value>{},
            /*partition_offsets=*/SmallVector<Value>{},
            /*partition_sizes=*/SmallVector<Value>{});
        Value finalDepPtr = finalDepAcqOp.getPtr();
        parallelEdt.appendDependency(finalDepPtr);
        BlockArgument finalPtrArg =
            parallelBlock.addArgument(finalDepPtr.getType(), loc);
        redInfo.finalResultArgs.push_back(finalPtrArg);
      }

      ARTS_DEBUG("  - Allocated fallback final result DB for reduction "
                 "variable");
    }

    SmallVector<Value> innerIndices{zeroIndex};

    if (canCombineDirectly) {
      Value finalGuid = redInfo.finalResultGuids[idx];
      Value finalPtr = redInfo.finalResultPtrs[idx];
      Value finalMemRef = AC->create<DbRefOp>(loc, finalPtr, innerIndices);
      AC->create<memref::StoreOp>(loc, identity, finalMemRef, innerIndices);

      redInfo.partialAccumGuids.push_back(finalGuid);
      redInfo.partialAccumPtrs.push_back(finalPtr);

      ARTS_DEBUG("  - Reusing final result DB as the worker-local accumulator");
      continue;
    }

    /// Step 2: Allocate and initialize partial accumulators DB.
    /// Mark as fine-grained so DbPartitioning does not re-partition it —
    /// each worker accesses exactly its own slot by workerId index.
    auto allocOp = AC->create<DbAllocOp>(
        loc, ArtsMode::inout, routeZero, DbAllocType::heap, DbMode::write,
        elementType, Value(), ValueRange{numWorkers}, ValueRange{sizeOne});
    setPartitionMode(allocOp, PartitionMode::fine_grained);
    Value partialGuid = allocOp.getGuid();
    Value partialPtr = allocOp.getPtr();

    Value one = AC->createIndexConstant(1, loc);
    Location loopLoc = redInfo.loopLocation.value_or(loc);

    /// When numWorkers is a compile-time constant 1, emit straight-line init
    /// instead of a loop.  A loop from 0 to 1 would use its induction variable
    /// as a db_ref index, which the verifier rejects for coarse-grained DBs
    /// (sizes[1]) because the IV is not a constant zero.
    if (isSingleWorker) {
      SmallVector<Value> outerIndices{zeroIndex};
      Value partialMemRef = AC->create<DbRefOp>(loc, partialPtr, outerIndices);
      AC->create<memref::StoreOp>(loc, identity, partialMemRef, innerIndices);
    } else {
      auto initLoop =
          AC->create<scf::ForOp>(loopLoc, zeroIndex, numWorkers, one);
      AC->setInsertionPointToStart(initLoop.getBody());
      Value workerIdx = initLoop.getInductionVar();

      SmallVector<Value> outerIndices{workerIdx};
      Value partialMemRef = AC->create<DbRefOp>(loc, partialPtr, outerIndices);
      AC->create<memref::StoreOp>(loc, identity, partialMemRef, innerIndices);

      AC->setInsertionPointAfter(initLoop);
    }

    redInfo.partialAccumGuids.push_back(partialGuid);
    redInfo.partialAccumPtrs.push_back(partialPtr);

    if (!splitMode) {
      SmallVector<Value> partialOffsets{zeroIndex};
      SmallVector<Value> partialSizes{numWorkers};
      auto depAcqOp = AC->create<DbAcquireOp>(
          loc, ArtsMode::inout, partialGuid, partialPtr, PartitionMode::coarse,
          /*indices=*/SmallVector<Value>{}, /*offsets=*/partialOffsets,
          /*sizes=*/partialSizes,
          /*partition_indices=*/SmallVector<Value>{},
          /*partition_offsets=*/SmallVector<Value>{},
          /*partition_sizes=*/SmallVector<Value>{});
      Value depPtr = depAcqOp.getPtr();
      parallelEdt.appendDependency(depPtr);
      BlockArgument partialPtrArg =
          parallelBlock.addArgument(depPtr.getType(), loc);
      redInfo.partialAccumArgs.push_back(partialPtrArg);
    }

    ARTS_DEBUG("  - Allocated and initialized partial accumulator DB for "
               "reduction variable");
  }

  redInfo.numWorkers = numWorkers;
  return redInfo;
}

void mlir::arts::createResultEdt(ArtsCodegen *AC,
                                 ReductionLoweringInfo &redInfo, Location loc) {
  OpBuilder::InsertionGuard IG(AC->getBuilder());
  if (redInfo.reductionVars.empty())
    return;
  if (redInfo.combineDirectlyInTask)
    return;

  ARTS_INFO(" - Creating single result EDT to combine partial accumulators");

  Value numWorkers = redInfo.numWorkers;
  if (!numWorkers) {
    ARTS_ERROR("Missing numWorkers in ReductionInfo");
    return;
  }

  Value zeroIndex = AC->createIndexConstant(0, loc);
  Value sizeOne = AC->createIndexConstant(1, loc);
  auto numWorkersConst = ValueAnalysis::tryFoldConstantIndex(numWorkers);
  bool isSingleWorker = numWorkersConst && *numWorkersConst == 1;

  SmallVector<Value> partialAcqPtrs;
  partialAcqPtrs.reserve(redInfo.partialAccumPtrs.size());
  for (uint64_t i = 0; i < redInfo.partialAccumPtrs.size(); i++) {
    Value partialPtr = redInfo.partialAccumPtrs[i];
    Value partialGuid = i < redInfo.partialAccumGuids.size()
                            ? redInfo.partialAccumGuids[i]
                            : Value();
    if (!partialPtr) {
      ARTS_ERROR("Missing partial accumulator ptr for reduction " << i);
      continue;
    }

    auto acqOp = AC->create<DbAcquireOp>(
        loc, ArtsMode::in, partialGuid, partialPtr, PartitionMode::coarse,
        /*indices=*/SmallVector<Value>{},
        /*offsets=*/SmallVector<Value>{zeroIndex},
        /*sizes=*/SmallVector<Value>{numWorkers},
        /*partition_indices=*/SmallVector<Value>{},
        /*partition_offsets=*/SmallVector<Value>{},
        /*partition_sizes=*/SmallVector<Value>{});

    partialAcqPtrs.push_back(acqOp.getResult(1));
  }

  SmallVector<Value> finalResultAcqPtrs;
  finalResultAcqPtrs.reserve(redInfo.finalResultPtrs.size());
  for (uint64_t i = 0; i < redInfo.finalResultPtrs.size(); i++) {
    Value finalPtr = redInfo.finalResultPtrs[i];
    Value finalGuid = i < redInfo.finalResultGuids.size()
                          ? redInfo.finalResultGuids[i]
                          : Value();
    if (!finalPtr) {
      ARTS_ERROR("Missing final result ptr for reduction " << i);
      continue;
    }

    ArtsMode finalMode = redInfo.strategy == ReductionLoweringStrategy::atomic
                             ? ArtsMode::inout
                             : ArtsMode::out;
    auto acqOp = AC->create<DbAcquireOp>(
        loc, finalMode, finalGuid, finalPtr, PartitionMode::coarse,
        /*indices=*/SmallVector<Value>{},
        /*offsets=*/SmallVector<Value>{zeroIndex},
        /*sizes=*/SmallVector<Value>{sizeOne},
        /*partition_indices=*/SmallVector<Value>{},
        /*partition_offsets=*/SmallVector<Value>{},
        /*partition_sizes=*/SmallVector<Value>{});

    finalResultAcqPtrs.push_back(acqOp.getResult(1));
  }

  uint64_t reductionCount = std::min<uint64_t>(
      redInfo.reductionVars.size(),
      std::min(partialAcqPtrs.size(), finalResultAcqPtrs.size()));
  if (reductionCount != redInfo.reductionVars.size()) {
    ARTS_ERROR("Reduction/result mismatch: reductions="
               << redInfo.reductionVars.size()
               << " partials=" << partialAcqPtrs.size()
               << " finals=" << finalResultAcqPtrs.size());
  }
  if (reductionCount == 0) {
    ARTS_ERROR("No reduction handles available for result EDT");
    return;
  }

  SmallVector<Value> edtDeps;
  edtDeps.reserve(partialAcqPtrs.size() + finalResultAcqPtrs.size());
  edtDeps.append(partialAcqPtrs.begin(), partialAcqPtrs.end());
  edtDeps.append(finalResultAcqPtrs.begin(), finalResultAcqPtrs.end());

  Value routeZero = AC->create<arith::ConstantIntOp>(loc, 0, 32);
  auto resultEdt = AC->create<EdtOp>(
      loc, EdtType::task, redInfo.parentConcurrency, routeZero, edtDeps);

  Block &resultBlock = resultEdt.getBody().front();
  AC->setInsertionPointToStart(&resultBlock);

  SmallVector<BlockArgument> depBlockArgs;
  depBlockArgs.reserve(edtDeps.size());
  for (Value dep : edtDeps)
    depBlockArgs.push_back(resultBlock.addArgument(dep.getType(), loc));

  Location loopLoc = redInfo.loopLocation.value_or(loc);

  for (uint64_t i = 0; i < reductionCount; i++) {
    BlockArgument partialArg = depBlockArgs[i];
    BlockArgument finalArg = depBlockArgs[partialAcqPtrs.size() + i];

    SmallVector<Value> zeroIndices{zeroIndex};
    Value finalMemRef = AC->create<DbRefOp>(loc, finalArg, zeroIndices);

    auto memType = cast<MemRefType>(finalMemRef.getType());
    Type elemType = memType.getElementType();
    ReductionCombinerKind combKind = i < redInfo.combinerKinds.size()
                                         ? redInfo.combinerKinds[i]
                                         : ReductionCombinerKind::add;

    switch (redInfo.strategy) {
    case ReductionLoweringStrategy::localAccumulate:
      lowerLinearResultCombine(AC, loopLoc, partialArg, finalMemRef, elemType,
                               zeroIndex, sizeOne, numWorkers, isSingleWorker,
                               combKind);
      break;
    case ReductionLoweringStrategy::atomic:
      lowerAtomicResultCombine(AC, loopLoc, partialArg, finalMemRef, elemType,
                               zeroIndex, sizeOne, numWorkers, isSingleWorker,
                               combKind);
      break;
    case ReductionLoweringStrategy::tree:
      lowerTreeResultCombine(AC, loopLoc, partialArg, finalMemRef, elemType,
                             zeroIndex, sizeOne, numWorkers, isSingleWorker,
                             combKind);
      break;
    }
  }

  AC->setInsertionPointToEnd(&resultBlock);
  for (BlockArgument blockArg : depBlockArgs)
    AC->create<DbReleaseOp>(loc, blockArg);

  AC->create<YieldOp>(loc);
  ARTS_INFO(" - Result EDT created successfully");
}

void mlir::arts::finalizeReductionAfterEpoch(ArtsCodegen *AC,
                                             ReductionLoweringInfo &redInfo,
                                             Location loc) {
  if (redInfo.reductionVars.empty())
    return;

  Value zeroIdx = AC->createIndexConstant(0, loc);
  SmallVector<Value> zeroIndices{zeroIdx};

  /// Copy final results back to original reduction variables.
  for (auto [idx, redVar] : llvm::enumerate(redInfo.reductionVars)) {
    if (idx >= redInfo.finalResultPtrs.size())
      continue;
    auto redMemRefTy = dyn_cast<MemRefType>(redVar.getType());
    if (!redMemRefTy)
      continue;

    Value finalHandle = redInfo.finalResultPtrs[idx];
    Value finalRef = AC->create<DbRefOp>(loc, finalHandle, zeroIndices);
    Value loaded = AC->create<memref::LoadOp>(loc, finalRef, zeroIndices);
    Value casted = AC->castParameter(redMemRefTy.getElementType(), loaded, loc);

    /// Find a valid store target for the reduction writeback.
    ///
    /// `redVar` is often a memref.cast defined inside the parallel EDT body
    /// which no longer dominates the current insertion point.  Furthermore,
    /// CreateDbs (stage 6) may have already run and wrapped the underlying
    /// alloca in a stack DB, redirecting all reads through a db_ref view.
    /// Storing to the raw alloca would be dead code.
    ///
    /// Strategy: (1) try `redVar` directly, (2) strip view ops to find the
    /// underlying alloca, (3) look for a DbAllocOp wrapping the alloca and
    /// create a db_ref to store through so the value is visible to host reads.
    auto canStoreHere = [&](Value v) -> bool {
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        Region *owner = ba.getOwner()->getParent();
        Region *cur = AC->getBuilder().getBlock()->getParent();
        return owner && cur && owner->isAncestor(cur);
      }
      if (Operation *def = v.getDefiningOp()) {
        Region *defR = def->getParentRegion();
        Region *cur = AC->getBuilder().getBlock()->getParent();
        return defR && cur && defR->isAncestor(cur);
      }
      return false;
    };

    Value storeTarget;
    if (canStoreHere(redVar)) {
      storeTarget = redVar;
    } else {
      /// Strip memref.cast / subview wrappers to find the underlying alloca.
      Value stripped = ValueAnalysis::stripMemrefViewOps(redVar);
      if (stripped != redVar && canStoreHere(stripped)) {
        /// CreateDbs (stage 6) may have already wrapped the alloca in a
        /// stack DB and redirected reads through db_ref.  Storing to the raw
        /// alloca would be dead code.  Scan the alloca's parent block for a
        /// DbAllocOp[stack] with matching element type that was created to
        /// back this alloca, and store through its db_ref view instead.
        auto allocaMemTy = dyn_cast<MemRefType>(stripped.getType());
        if (allocaMemTy) {
          Operation *defOp = stripped.getDefiningOp();
          Block *parentBlock = defOp ? defOp->getBlock() : nullptr;
          if (parentBlock) {
            bool foundAlloca = false;
            for (Operation &op : *parentBlock) {
              if (&op == defOp) {
                foundAlloca = true;
                continue;
              }
              if (!foundAlloca)
                continue;
              if (auto dbAlloc = dyn_cast<DbAllocOp>(&op)) {
                if (dbAlloc.getAllocType() == DbAllocType::stack &&
                    dbAlloc.getAllocatedElementType().getElementType() ==
                        allocaMemTy.getElementType()) {
                  storeTarget =
                      AC->create<DbRefOp>(loc, dbAlloc.getPtr(), zeroIndices);
                  break;
                }
              }
            }
          }
          /// Fallback: store to the raw alloca if no DB wrapping was found.
          if (!storeTarget)
            storeTarget = stripped;
        }
      }
    }
    if (storeTarget)
      AC->create<memref::StoreOp>(loc, casted, storeTarget, zeroIndices);

    if (idx < redInfo.hostResultPtrs.size()) {
      Value hostPtr = redInfo.hostResultPtrs[idx];
      Value hostRef = AC->create<DbRefOp>(loc, hostPtr, zeroIndices);
      AC->create<memref::StoreOp>(loc, casted, hostRef, zeroIndices);
    }
  }

  if (!redInfo.combineDirectlyInTask) {
    for (uint64_t i = 0; i < redInfo.partialAccumGuids.size(); i++) {
      AC->create<DbFreeOp>(loc, redInfo.partialAccumGuids[i]);
      AC->create<DbFreeOp>(loc, redInfo.partialAccumPtrs[i]);
    }
  }

  for (uint64_t i = 0; i < redInfo.finalResultGuids.size(); i++) {
    if (i < redInfo.finalResultIsExternal.size() &&
        redInfo.finalResultIsExternal[i]) {
      ARTS_DEBUG("  - Skipping db_free for external final result DB at index "
                 << i);
      continue;
    }
    AC->create<DbFreeOp>(loc, redInfo.finalResultGuids[i]);
    AC->create<DbFreeOp>(loc, redInfo.finalResultPtrs[i]);
  }
}
