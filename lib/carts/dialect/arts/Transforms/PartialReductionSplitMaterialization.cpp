///==========================================================================///
/// File: PartialReductionSplitMaterialization.cpp
///
/// Materializes ARTS partial-reduction split plans into concrete DB/EDT
/// structure before ARTS-RT lowering.
///==========================================================================///

#define GEN_PASS_DEF_PARTIALREDUCTIONSPLITMATERIALIZATION
#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/RuntimeOpUtils.h"
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "carts/utils/ArrayAttrUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

struct ReductionLoopMatch {
  DbRefOp resultRef;
  scf::ForOp loop;
  memref::StoreOp resultStore;
  arith::AddFOp add;
  memref::LoadOp seedLoad;
};

struct SplitPlan {
  unsigned resultDepIndex = 0;
  unsigned ownerParamIndex = 0;
  int64_t splitFactor = 0;
  int64_t ownerTaskCount = 0;
  int64_t targetWorkerCount = 0;
  int64_t resultElementCount = 1;
  SmallVector<int64_t, 4> ownerDims;
  SmallVector<int64_t, 4> reductionDims;
  SmallVector<int64_t, 4> splitDims;
  ReductionLoopMatch reduction;
};

static bool isInEdtBody(EdtOp parent, Operation *op) {
  EdtOp enclosing = op ? op->getParentOfType<EdtOp>() : EdtOp{};
  return enclosing == parent;
}

static Value createScalarZero(OpBuilder &builder, Location loc, Type type) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return arith::ConstantOp::create(builder, loc, type,
                                     builder.getFloatAttr(floatType, 0.0));
  return {};
}

static FailureOr<SmallVector<Value, 4>>
rematerializeIndicesDominating(ValueRange indices, Operation *insertBefore,
                               OpBuilder &builder, Location loc) {
  Operation *domRoot = insertBefore->getParentOfType<func::FuncOp>();
  if (!domRoot)
    domRoot = insertBefore->getParentOfType<ModuleOp>();
  if (!domRoot)
    return failure();

  DominanceInfo domInfo(domRoot);
  SmallVector<Value, 4> rematerialized;
  rematerialized.reserve(indices.size());
  for (Value index : indices) {
    Value dominating = ValueAnalysis::traceValueToDominating(
        index, insertBefore, builder, domInfo, loc);
    if (!dominating)
      return failure();
    rematerialized.push_back(dominating);
  }
  return rematerialized;
}

static bool hasMultinodeRuntime(EdtOp edt) {
  ModuleOp module = edt->getParentOfType<ModuleOp>();
  std::optional<int64_t> totalNodes = getRuntimeTotalNodes(module);
  return totalNodes && *totalNodes > 1;
}

static std::optional<int64_t> getRuntimeWorkersPerNode(EdtOp edt) {
  ModuleOp module = edt->getParentOfType<ModuleOp>();
  std::optional<int64_t> totalWorkers = getRuntimeTotalWorkers(module);
  std::optional<int64_t> totalNodes = getRuntimeTotalNodes(module);
  if (!totalWorkers || !totalNodes || *totalWorkers <= 0 || *totalNodes <= 0)
    return std::nullopt;
  return (*totalWorkers + *totalNodes - 1) / *totalNodes;
}

static int64_t ceilDivPositiveI64(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
}

static void clearSplitPlanAttrs(EdtOp edt) {
  edt->removeAttr(edt.getPartialReductionSplitRequiredAttrName());
  edt->removeAttr(edt.getPartialReductionSplitDimsAttrName());
  edt->removeAttr(edt.getPartialReductionSplitFactorAttrName());
  edt->removeAttr(edt.getPartialReductionSplitOwnerTaskCountAttrName());
  edt->removeAttr(edt.getPartialReductionSplitTargetWorkerCountAttrName());
}

static void stampEffectiveSplitPlanAttrs(EdtOp edt, const SplitPlan &plan) {
  MLIRContext *ctx = edt.getContext();
  auto i64 = IntegerType::get(ctx, 64);
  edt.setPartialReductionSplitFactorAttr(
      IntegerAttr::get(i64, plan.splitFactor));
  edt.setPartialReductionSplitTargetWorkerCountAttr(
      IntegerAttr::get(i64, plan.targetWorkerCount));
}

static bool reconcileSplitTopology(EdtOp edt, SplitPlan &plan,
                                   bool &distributedTopology) {
  bool multinode = hasMultinodeRuntime(edt);
  bool depsAllowDistributed =
      !DbUtils::hasLocalOnlyDistributedLaunchDependency(edt);
  distributedTopology = multinode && depsAllowDistributed;
  if (distributedTopology || !multinode)
    return true;

  std::optional<int64_t> localWorkers = getRuntimeWorkersPerNode(edt);
  if (!localWorkers || *localWorkers <= 0)
    return true;

  int64_t effectiveTarget = *localWorkers;
  if (effectiveTarget > plan.targetWorkerCount)
    effectiveTarget = plan.targetWorkerCount;
  if (effectiveTarget <= plan.ownerTaskCount) {
    clearSplitPlanAttrs(edt);
    return false;
  }

  int64_t requestedFactor =
      ceilDivPositiveI64(effectiveTarget, plan.ownerTaskCount);
  if (requestedFactor <= 1) {
    clearSplitPlanAttrs(edt);
    return false;
  }
  if (requestedFactor < plan.splitFactor)
    plan.splitFactor = requestedFactor;
  plan.targetWorkerCount = effectiveTarget;
  stampEffectiveSplitPlanAttrs(edt, plan);
  return true;
}

static Value createOwnerOrdinal(OpBuilder &builder, Location loc,
                                scf::ForOp ownerLoop, Value ownerIndex) {
  Value relative = arith::SubIOp::create(builder, loc, ownerIndex,
                                         ownerLoop.getLowerBound());
  return arith::DivUIOp::create(builder, loc, relative, ownerLoop.getStep());
}

static Value createDistributedRoute(OpBuilder &builder, Location loc,
                                    scf::ForOp ownerLoop, Value ownerIndex,
                                    Value tileIndex, int64_t tileCount) {
  Value ownerOrdinal = createOwnerOrdinal(builder, loc, ownerLoop, ownerIndex);
  Value tileCountValue = createConstantIndex(builder, loc, tileCount);
  Value ownerBase =
      arith::MulIOp::create(builder, loc, ownerOrdinal, tileCountValue);
  Value linearIndex = arith::AddIOp::create(builder, loc, ownerBase, tileIndex);
  Value linearIndexI32 = arith::IndexCastOp::create(
      builder, loc, builder.getI32Type(), linearIndex);
  auto totalNodes =
      RuntimeQueryOp::create(builder, loc, RuntimeQueryKind::totalNodes);
  return arith::RemUIOp::create(builder, loc, linearIndexI32,
                                totalNodes.getResult());
}

static void markDistributedRemoteUse(Value dep) {
  Operation *underlying = DbUtils::getUnderlyingDbAlloc(dep);
  if (auto alloc = dyn_cast_or_null<DbAllocOp>(underlying))
    alloc.removeLocalOnlyAttr();
}

static void markMaterializedReductionDistribution(Operation *op,
                                                  bool distributed) {
  if (!op)
    return;
  setEdtDistributionPattern(op, EdtDistributionPattern::uniform);
  setDistributionVersion(op, 1);
  if (distributed)
    setEdtDistributionKind(op, EdtDistributionKind::block_cyclic);
}

static LogicalResult readRequiredI64Array(EdtOp edt, ArrayAttr attr,
                                          StringRef name,
                                          SmallVectorImpl<int64_t> &values) {
  auto parsed = readI64ArrayAttr(attr);
  if (!parsed)
    return edt.emitOpError() << "requires integer array attribute '" << name
                             << "' for partial-reduction split materialization";
  values.assign(parsed->begin(), parsed->end());
  return success();
}

static LogicalResult findResultDependencyIndex(EdtOp edt, SplitPlan &plan) {
  ArrayAttr depMaps = edt.getPartialReductionDepResultDimMapsAttr();
  if (!depMaps)
    return edt.emitOpError()
           << "requires partialReductionDepResultDimMaps to identify the "
              "partial-reduction result dependency";
  if (depMaps.size() != edt.getDependencies().size())
    return edt.emitOpError()
           << "partialReductionDepResultDimMaps entry count (" << depMaps.size()
           << ") must match dependency count (" << edt.getDependencies().size()
           << ")";

  std::optional<unsigned> resultIndex;
  for (auto [idx, attr] : llvm::enumerate(depMaps)) {
    auto mapAttr = dyn_cast<ArrayAttr>(attr);
    auto dims = readI64ArrayAttr(mapAttr);
    if (!dims)
      return edt.emitOpError() << "partialReductionDepResultDimMaps entry #"
                               << idx << " must be an integer array";
    if (dims->empty())
      continue;
    if (llvm::is_contained(*dims, -1))
      continue;

    bool coversOwnerDims = true;
    for (int64_t ownerDim : plan.ownerDims)
      coversOwnerDims &= llvm::is_contained(*dims, ownerDim);
    if (!coversOwnerDims)
      continue;

    if (auto acquire = edt.getDependencies()[idx].getDefiningOp<DbAcquireOp>())
      if (acquire.getMode() == ArtsMode::in)
        continue;

    if (resultIndex)
      return edt.emitOpError()
             << "partial-reduction split materialization found multiple "
                "candidate result dependencies";
    resultIndex = static_cast<unsigned>(idx);
  }

  if (!resultIndex)
    return edt.emitOpError()
           << "partial-reduction split materialization could not identify a "
              "result dependency with owner dims and no reduction-only dims";
  plan.resultDepIndex = *resultIndex;
  return success();
}

static FailureOr<unsigned> findOwnerParamIndex(EdtOp edt) {
  for (Operation *parent = edt->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      continue;
    for (auto [idx, param] : llvm::enumerate(edt.getParams())) {
      if (param == loop.getInductionVar())
        return static_cast<unsigned>(idx);
    }
  }
  return failure();
}

static FailureOr<DbRefOp> findSingleResultRef(EdtOp edt,
                                              unsigned resultDepIndex) {
  Block &body = edt.getBody().front();
  if (resultDepIndex >= body.getNumArguments())
    return failure();
  BlockArgument resultArg = body.getArgument(resultDepIndex);

  SmallVector<DbRefOp, 2> refs;
  edt.getBody().walk([&](DbRefOp ref) {
    if (!isInEdtBody(edt, ref.getOperation()))
      return;
    if (ref.getSource() == resultArg)
      refs.push_back(ref);
  });

  if (refs.size() != 1)
    return failure();
  return refs.front();
}

static FailureOr<ReductionLoopMatch> matchReductionLoop(EdtOp edt,
                                                        SplitPlan plan) {
  FailureOr<DbRefOp> resultRef = findSingleResultRef(edt, plan.resultDepIndex);
  if (failed(resultRef))
    return failure();

  auto resultType = dyn_cast<MemRefType>((*resultRef).getResult().getType());
  if (!resultType || !isa<FloatType>(resultType.getElementType()))
    return failure();

  SmallVector<ReductionLoopMatch, 2> matches;
  edt.getBody().walk([&](scf::ForOp loop) {
    if (!isInEdtBody(edt, loop.getOperation()))
      return;

    SmallVector<memref::StoreOp, 2> stores;
    loop.getBody()->walk([&](memref::StoreOp store) {
      if (!isInEdtBody(edt, store.getOperation()))
        return;
      if (store.getMemref() == (*resultRef).getResult())
        stores.push_back(store);
    });
    if (stores.size() != 1)
      return;

    auto add = stores.front().getValue().getDefiningOp<arith::AddFOp>();
    if (!add)
      return;

    memref::LoadOp seedLoad;
    for (Value operand : add->getOperands()) {
      auto load = operand.getDefiningOp<memref::LoadOp>();
      if (load && load.getMemref() == (*resultRef).getResult()) {
        if (seedLoad)
          return;
        seedLoad = load;
      }
    }
    if (!seedLoad)
      return;

    matches.push_back({*resultRef, loop, stores.front(), add, seedLoad});
  });

  SmallVector<ReductionLoopMatch, 2> rootMatches;
  for (ReductionLoopMatch candidate : matches) {
    bool nestedInAnotherMatch = false;
    for (ReductionLoopMatch other : matches) {
      if (candidate.loop == other.loop)
        continue;
      if (other.loop->isProperAncestor(candidate.loop.getOperation())) {
        nestedInAnotherMatch = true;
        break;
      }
    }
    if (!nestedInAnotherMatch)
      rootMatches.push_back(candidate);
  }

  if (rootMatches.size() != 1)
    return failure();
  return rootMatches.front();
}

static LogicalResult validateSplitPlan(EdtOp edt, SplitPlan &plan) {
  if (!edt.getPartialReductionAttr())
    return edt.emitOpError()
           << "partialReductionSplitRequired requires partialReduction";

  auto splitFactor = edt.getPartialReductionSplitFactor();
  if (!splitFactor || *splitFactor <= 1)
    return edt.emitOpError()
           << "requires static partialReductionSplitFactor > 1";
  plan.splitFactor = *splitFactor;

  auto ownerTaskCount = edt.getPartialReductionSplitOwnerTaskCount();
  if (!ownerTaskCount || *ownerTaskCount <= 0)
    return edt.emitOpError()
           << "requires positive partialReductionSplitOwnerTaskCount";
  plan.ownerTaskCount = *ownerTaskCount;

  auto targetWorkerCount = edt.getPartialReductionSplitTargetWorkerCount();
  if (!targetWorkerCount || *targetWorkerCount <= 0)
    return edt.emitOpError()
           << "requires positive partialReductionSplitTargetWorkerCount";
  plan.targetWorkerCount = *targetWorkerCount;

  if (failed(readRequiredI64Array(edt, edt.getPartialReductionOwnerDimsAttr(),
                                  "partialReductionOwnerDims",
                                  plan.ownerDims)) ||
      failed(readRequiredI64Array(edt, edt.getPartialReductionDimsAttr(),
                                  "partialReductionDims",
                                  plan.reductionDims)) ||
      failed(readRequiredI64Array(edt, edt.getPartialReductionSplitDimsAttr(),
                                  "partialReductionSplitDims", plan.splitDims)))
    return failure();

  if (plan.ownerDims.size() != 1 || plan.reductionDims.empty() ||
      plan.splitDims.empty())
    return edt.emitOpError()
           << "partial-reduction split materialization currently supports "
              "exactly one owner dim and one or more reduction/split dims";
  for (int64_t splitDim : plan.splitDims) {
    if (llvm::is_contained(plan.reductionDims, splitDim))
      continue;
    return edt.emitOpError()
           << "partialReductionSplitDims must be contained in "
              "partialReductionDims for "
              "the supported static materialization shape";
  }

  if (edt.getDependencies().size() < 2)
    return edt.emitOpError()
           << "partial-reduction split materialization requires a result "
              "dependency and at least one reduction input dependency";

  if (failed(findResultDependencyIndex(edt, plan)))
    return failure();

  FailureOr<unsigned> ownerParamIndex = findOwnerParamIndex(edt);
  if (failed(ownerParamIndex))
    return edt.emitOpError()
           << "partial-reduction split materialization requires an enclosing "
              "owner dispatch loop whose induction variable is an EDT param";
  plan.ownerParamIndex = *ownerParamIndex;

  FailureOr<ReductionLoopMatch> reduction = matchReductionLoop(edt, plan);
  if (failed(reduction))
    return edt.emitOpError()
           << "partial-reduction split materialization could not prove the "
              "supported scalar floating add reduction loop";
  plan.reduction = *reduction;

  if (auto workerSlice =
          readI64ArrayAttr(getPlanLogicalWorkerSliceAttr(edt.getOperation()))) {
    if (!workerSlice->empty()) {
      if ((*workerSlice)[0] <= 0)
        return edt.emitOpError()
               << "partial-reduction split materialization requires a "
                  "positive rank-1 result tile length";
      plan.resultElementCount = (*workerSlice)[0];
    }
  }

  return success();
}

static void copyMaterializedEdtAttrs(EdtOp source, EdtOp dest) {
  StringAttr operandSegments =
      EdtOp::getOperandSegmentSizesAttrName(source->getName());
  for (NamedAttribute attr : source->getAttrs()) {
    if (attr.getName() == operandSegments)
      continue;
    if (attr.getName().getValue() == source.getConcurrencyAttrName())
      continue;
    if (attr.getName().getValue() ==
        source.getPartialReductionSplitRequiredAttrName())
      continue;
    dest->setAttr(attr.getName(), attr.getValue());
  }
  dest->removeAttr(dest.getPartialReductionSplitRequiredAttrName());
}

static void addEdtBlockArguments(EdtOp edt, ValueRange deps, ValueRange params,
                                 Location loc) {
  Block &body = edt.getBody().front();
  for (Value dep : deps)
    body.addArgument(dep.getType(), loc);
  for (Value param : params)
    body.addArgument(param.getType(), loc);
}

static void cloneEdtBody(EdtOp source, EdtOp dest) {
  Block &sourceBlock = source.getBody().front();
  Block &destBlock = dest.getBody().front();

  IRMapping mapper;
  for (auto [index, sourceArg] : llvm::enumerate(sourceBlock.getArguments()))
    mapper.map(sourceArg, destBlock.getArgument(index));

  OpBuilder builder(dest.getContext());
  builder.setInsertionPointToStart(&destBlock);
  for (Operation &op : sourceBlock.without_terminator())
    builder.clone(op, mapper);
  YieldOp::create(builder, source.getLoc());
}

static LogicalResult retileSplitWorkerLoop(EdtOp splitEdt,
                                           const SplitPlan &plan,
                                           unsigned originalParamCount) {
  FailureOr<ReductionLoopMatch> match = matchReductionLoop(splitEdt, plan);
  if (failed(match))
    return failure();

  Location loc = splitEdt.getLoc();
  Block &body = splitEdt.getBody().front();
  Value tileArg =
      body.getArgument(splitEdt.getDependencies().size() + originalParamCount);

  OpBuilder loopBuilder((*match).loop);
  Value one = createOneIndex(loopBuilder, loc);
  Value splitFactor = createConstantIndex(loopBuilder, loc, plan.splitFactor);
  Value splitFactorMinusOne =
      createConstantIndex(loopBuilder, loc, plan.splitFactor - 1);
  Value range =
      arith::SubIOp::create(loopBuilder, loc, (*match).loop.getUpperBound(),
                            (*match).loop.getLowerBound());
  Value numerator =
      arith::AddIOp::create(loopBuilder, loc, range, splitFactorMinusOne);
  Value tileSize =
      arith::DivUIOp::create(loopBuilder, loc, numerator, splitFactor);
  Value tileOffset = arith::MulIOp::create(loopBuilder, loc, tileArg, tileSize);
  Value tileBegin = arith::AddIOp::create(
      loopBuilder, loc, (*match).loop.getLowerBound(), tileOffset);
  Value rawTileEnd =
      arith::AddIOp::create(loopBuilder, loc, tileBegin, tileSize);
  Value tileEnd = arith::MinUIOp::create(loopBuilder, loc, rawTileEnd,
                                         (*match).loop.getUpperBound());
  (*match).loop.setLowerBound(tileBegin);
  (*match).loop.setUpperBound(tileEnd);
  (*match).loop.setStep(one);

  auto resultType = cast<MemRefType>((*match).resultRef.getResult().getType());
  Value identity =
      createScalarZero(loopBuilder, loc, resultType.getElementType());
  if (!identity)
    return failure();

  OpBuilder addBuilder((*match).add);
  SmallVector<Value, 4> addLoadIndices((*match).seedLoad.getIndices().begin(),
                                       (*match).seedLoad.getIndices().end());
  FailureOr<SmallVector<Value, 4>> initIndices = rematerializeIndicesDominating(
      addLoadIndices, (*match).loop.getOperation(), loopBuilder, loc);
  if (failed(initIndices))
    return failure();
  memref::StoreOp::create(loopBuilder, loc, identity,
                          (*match).resultRef.getResult(), *initIndices);
  Value current = memref::LoadOp::create(
      addBuilder, loc, (*match).resultRef.getResult(), addLoadIndices);
  for (OpOperand &operand : (*match).add->getOpOperands())
    if (operand.get() == (*match).seedLoad.getResult())
      operand.set(current);

  if (!(*match).seedLoad->use_empty()) {
    OpBuilder seedBuilder((*match).seedLoad);
    Value zero =
        createScalarZero(seedBuilder, loc, resultType.getElementType());
    if (!zero)
      return failure();
    (*match).seedLoad.getResult().replaceAllUsesWith(zero);
  }
  (*match).seedLoad.erase();

  return success();
}

static DbAcquireOp createTileAcquire(OpBuilder &builder, Location loc,
                                     DbAllocOp db, ArtsMode mode,
                                     Type resultDepType, Value ownerIndex,
                                     Value tileIndex) {
  SmallVector<Value> indices{ownerIndex, tileIndex};
  auto acquire = DbAcquireOp::create(builder, loc, mode, db.getGuid(),
                                     db.getPtr(), resultDepType,
                                     PartitionMode::block, std::move(indices),
                                     /*offsets=*/SmallVector<Value>{},
                                     /*sizes=*/SmallVector<Value>{},
                                     /*partitionIndices=*/SmallVector<Value>{},
                                     /*partitionOffsets=*/SmallVector<Value>{},
                                     /*partitionSizes=*/SmallVector<Value>{},
                                     /*boundsValid=*/Value{},
                                     /*elementOffsets=*/SmallVector<Value>{},
                                     /*elementSizes=*/SmallVector<Value>{});
  inheritDistributionAttrs(db.getOperation(), acquire.getOperation());
  if (auto depPattern = db.getDepPatternAttr())
    acquire.setDepPatternAttr(depPattern);
  return acquire;
}

static DbAcquireOp createPartialTileAcquire(OpBuilder &builder, Location loc,
                                            DbAllocOp partialDb,
                                            Type resultDepType,
                                            Value ownerIndex, Value tileIndex) {
  return createTileAcquire(builder, loc, partialDb, ArtsMode::out,
                           resultDepType, ownerIndex, tileIndex);
}

static DbAllocOp createReductionBufferDb(OpBuilder &builder, Location loc,
                                         Value route, Value ownerCount,
                                         Value tileCount, Value elementCount,
                                         int64_t resultElementCount,
                                         Type scalarType, bool distributed) {
  auto db = DbAllocOp::create(
      builder, loc, ArtsMode::inout, route, DbAllocType::heap, DbMode::write,
      scalarType, SmallVector<Value>{ownerCount, tileCount},
      SmallVector<Value>{elementCount}, PartitionMode::block);
  db->setAttr(db.getPlanOwnerDimsAttrName(), buildI64ArrayAttr(db, {0}));
  db->setAttr(db.getPlanPhysicalBlockShapeAttrName(),
              buildI64ArrayAttr(db, {resultElementCount}));
  db->setAttr(db.getPlanLogicalWorkerSliceAttrName(),
              buildI64ArrayAttr(db, {resultElementCount}));
  db.setDepPatternAttr(
      ArtsDepPatternAttr::get(db.getContext(), ArtsDepPattern::reduction));
  markMaterializedReductionDistribution(db.getOperation(), distributed);
  if (distributed) {
    setDistributedDbAllocation(db.getOperation(), /*enabled=*/true);
    db.removeLocalOnlyAttr();
    db->removeAttr(AttrNames::Operation::DistributedRejectReason);
  }
  return db;
}

static Value createScalarTileRef(OpBuilder &builder, Location loc,
                                 Value depArg) {
  Value zero = createZeroIndex(builder, loc);
  return DbRefOp::create(builder, loc, depArg, SmallVector<Value>{zero})
      .getResult();
}

static void copyCombineMetadata(EdtOp source, EdtOp dest) {
  if (auto depPattern = source.getDepPatternAttr())
    dest.setDepPatternAttr(depPattern);
  if (auto reductionStrategy = source.getReductionStrategyAttr())
    dest.setReductionStrategyAttr(reductionStrategy);
  inheritDistributionAttrs(source.getOperation(), dest.getOperation());
}

static LogicalResult createIntermediateCombineBody(EdtOp combineEdt,
                                                   Type scalarType,
                                                   Value elementCount,
                                                   bool hasRightInput) {
  if (!isa<FloatType>(scalarType))
    return failure();

  Location loc = combineEdt.getLoc();
  Block &body = combineEdt.getBody().front();
  BlockArgument outputArg = body.getArgument(0);
  BlockArgument leftArg = body.getArgument(1);

  OpBuilder builder(combineEdt.getContext());
  builder.setInsertionPointToStart(&body);

  Value outputRef = createScalarTileRef(builder, loc, outputArg);
  Value leftRef = createScalarTileRef(builder, loc, leftArg);
  Value rightRef;
  if (hasRightInput)
    rightRef = createScalarTileRef(builder, loc, body.getArgument(2));

  Value zero = createZeroIndex(builder, loc);
  Value one = createOneIndex(builder, loc);
  auto loop = scf::ForOp::create(builder, loc, zero, elementCount, one);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value idx = loop.getInductionVar();
    SmallVector<Value> indices{idx};
    Value accumulated = memref::LoadOp::create(builder, loc, leftRef, indices);
    if (hasRightInput) {
      Value right = memref::LoadOp::create(builder, loc, rightRef, indices);
      accumulated = arith::AddFOp::create(builder, loc, accumulated, right);
    }
    memref::StoreOp::create(builder, loc, accumulated, outputRef, indices);
  }
  YieldOp::create(builder, loc);
  return success();
}

static LogicalResult createFinalCombineBody(EdtOp combineEdt, Type scalarType,
                                            Value elementCount,
                                            bool hasRightInput) {
  if (!isa<FloatType>(scalarType))
    return failure();

  Location loc = combineEdt.getLoc();
  Block &body = combineEdt.getBody().front();
  BlockArgument finalArg = body.getArgument(0);
  BlockArgument leftArg = body.getArgument(1);

  OpBuilder builder(combineEdt.getContext());
  builder.setInsertionPointToStart(&body);

  Value finalRef = createScalarTileRef(builder, loc, finalArg);
  Value leftRef = createScalarTileRef(builder, loc, leftArg);
  Value rightRef;
  if (hasRightInput)
    rightRef = createScalarTileRef(builder, loc, body.getArgument(2));

  Value zero = createZeroIndex(builder, loc);
  Value one = createOneIndex(builder, loc);
  auto loop = scf::ForOp::create(builder, loc, zero, elementCount, one);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value idx = loop.getInductionVar();
    SmallVector<Value> indices{idx};
    Value accumulated = memref::LoadOp::create(builder, loc, finalRef, indices);
    Value left = memref::LoadOp::create(builder, loc, leftRef, indices);
    accumulated = arith::AddFOp::create(builder, loc, accumulated, left);
    if (hasRightInput) {
      Value right = memref::LoadOp::create(builder, loc, rightRef, indices);
      accumulated = arith::AddFOp::create(builder, loc, accumulated, right);
    }
    memref::StoreOp::create(builder, loc, accumulated, finalRef, indices);
  }
  YieldOp::create(builder, loc);
  return success();
}

static LogicalResult createIntermediateCombineEdt(
    OpBuilder &builder, Location loc, EdtOp sourceEdt, DbAllocOp inputDb,
    DbAllocOp outputDb, Type resultDepType, Value ownerIndex, int64_t leftIndex,
    std::optional<int64_t> rightIndex, int64_t outputIndex, Type scalarType,
    Value elementCount, EdtConcurrency concurrency, Value route) {
  Value leftTile = createConstantIndex(builder, loc, leftIndex);
  Value outputTile = createConstantIndex(builder, loc, outputIndex);
  DbAcquireOp output = createTileAcquire(builder, loc, outputDb, ArtsMode::out,
                                         resultDepType, ownerIndex, outputTile);
  DbAcquireOp left = createTileAcquire(builder, loc, inputDb, ArtsMode::in,
                                       resultDepType, ownerIndex, leftTile);

  SmallVector<Value> deps{output.getPtr(), left.getPtr()};
  if (rightIndex) {
    Value rightTile = createConstantIndex(builder, loc, *rightIndex);
    DbAcquireOp right = createTileAcquire(builder, loc, inputDb, ArtsMode::in,
                                          resultDepType, ownerIndex, rightTile);
    deps.push_back(right.getPtr());
  }

  auto combineEdt = EdtOp::create(builder, loc, EdtType::task, concurrency,
                                  route, deps, ValueRange{});
  addEdtBlockArguments(combineEdt, deps, ValueRange{}, loc);
  copyCombineMetadata(sourceEdt, combineEdt);
  markMaterializedReductionDistribution(
      combineEdt.getOperation(), concurrency == EdtConcurrency::internode);
  return createIntermediateCombineBody(combineEdt, scalarType, elementCount,
                                       rightIndex.has_value());
}

static LogicalResult createFinalCombineEdt(OpBuilder &builder, Location loc,
                                           EdtOp sourceEdt, DbAllocOp inputDb,
                                           Value finalDep, Type resultDepType,
                                           Value ownerIndex, int64_t inputCount,
                                           Type scalarType, Value elementCount,
                                           EdtConcurrency concurrency,
                                           Value route) {
  if (inputCount < 1 || inputCount > 2)
    return failure();

  Value leftTile = createZeroIndex(builder, loc);
  DbAcquireOp left = createTileAcquire(builder, loc, inputDb, ArtsMode::in,
                                       resultDepType, ownerIndex, leftTile);

  SmallVector<Value> deps{finalDep, left.getPtr()};
  if (inputCount == 2) {
    Value rightTile = createOneIndex(builder, loc);
    DbAcquireOp right = createTileAcquire(builder, loc, inputDb, ArtsMode::in,
                                          resultDepType, ownerIndex, rightTile);
    deps.push_back(right.getPtr());
  }

  auto combineEdt = EdtOp::create(builder, loc, EdtType::task, concurrency,
                                  route, deps, ValueRange{});
  addEdtBlockArguments(combineEdt, deps, ValueRange{}, loc);
  copyCombineMetadata(sourceEdt, combineEdt);
  markMaterializedReductionDistribution(
      combineEdt.getOperation(), concurrency == EdtConcurrency::internode);
  return createFinalCombineBody(combineEdt, scalarType, elementCount,
                                inputCount == 2);
}

static LogicalResult materializeSplitPlan(EdtOp edt, SplitPlan &plan) {
  Location loc = edt.getLoc();
  OpBuilder builder(edt.getContext());

  scf::ForOp ownerLoop;
  for (Operation *parent = edt->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      continue;
    if (llvm::is_contained(edt.getParams(), loop.getInductionVar())) {
      ownerLoop = loop;
      break;
    }
  }
  if (!ownerLoop)
    return failure();

  auto resultType =
      cast<MemRefType>(plan.reduction.resultRef.getResult().getType());
  Type scalarType = resultType.getElementType();
  if (!isa<FloatType>(scalarType) || resultType.getRank() != 1)
    return failure();

  bool distributedTopology = false;
  if (!reconcileSplitTopology(edt, plan, distributedTopology))
    return success();
  EdtConcurrency materializedConcurrency =
      distributedTopology ? EdtConcurrency::internode : edt.getConcurrency();

  builder.setInsertionPoint(ownerLoop);
  Value route = createCurrentNodeRoute(builder, loc);
  Value ownerCount = createConstantIndex(builder, loc, plan.ownerTaskCount);
  Value splitFactor = createConstantIndex(builder, loc, plan.splitFactor);
  Value resultElementCount =
      createConstantIndex(builder, loc, plan.resultElementCount);
  DbAllocOp partialDb = createReductionBufferDb(
      builder, loc, route, ownerCount, splitFactor, resultElementCount,
      plan.resultElementCount, scalarType, distributedTopology);
  SmallVector<std::pair<DbAllocOp, int64_t>, 4> intermediateLevels;
  for (int64_t currentCount = plan.splitFactor; currentCount > 2;) {
    int64_t nextCount = (currentCount + 1) / 2;
    Value nextCountValue = createConstantIndex(builder, loc, nextCount);
    DbAllocOp nextDb = createReductionBufferDb(
        builder, loc, route, ownerCount, nextCountValue, resultElementCount,
        plan.resultElementCount, scalarType, distributedTopology);
    intermediateLevels.push_back({nextDb, nextCount});
    currentCount = nextCount;
  }

  Value ownerIndex = edt.getParams()[plan.ownerParamIndex];
  Type resultDepType = edt.getDependencies()[plan.resultDepIndex].getType();
  SmallVector<Value> originalDeps(edt.getDependencies().begin(),
                                  edt.getDependencies().end());
  SmallVector<Value> originalParams(edt.getParams().begin(),
                                    edt.getParams().end());
  unsigned originalParamCount = originalParams.size();
  if (distributedTopology)
    for (Value dep : originalDeps)
      markDistributedRemoteUse(dep);

  builder.setInsertionPoint(edt);
  Value ownerOrdinal = createOwnerOrdinal(builder, loc, ownerLoop, ownerIndex);
  Value zero = createZeroIndex(builder, loc);
  Value one = createOneIndex(builder, loc);
  Value splitUpper = createConstantIndex(builder, loc, plan.splitFactor);
  auto splitLoop = scf::ForOp::create(builder, loc, zero, splitUpper, one);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(splitLoop.getBody());
    Value tileIndex = splitLoop.getInductionVar();
    DbAcquireOp partialTile = createPartialTileAcquire(
        builder, loc, partialDb, resultDepType, ownerOrdinal, tileIndex);

    SmallVector<Value> splitDeps = originalDeps;
    splitDeps[plan.resultDepIndex] = partialTile.getPtr();
    SmallVector<Value> splitParams = originalParams;
    splitParams.push_back(tileIndex);

    Value splitRoute =
        distributedTopology
            ? createDistributedRoute(builder, loc, ownerLoop, ownerIndex,
                                     tileIndex, plan.splitFactor)
            : edt.getRoute();
    auto splitEdt =
        EdtOp::create(builder, loc, edt.getType(), materializedConcurrency,
                      splitRoute, splitDeps, splitParams);
    copyMaterializedEdtAttrs(edt, splitEdt);
    markMaterializedReductionDistribution(splitEdt.getOperation(),
                                          distributedTopology);
    addEdtBlockArguments(splitEdt, splitDeps, splitParams, loc);
    cloneEdtBody(edt, splitEdt);
    if (failed(retileSplitWorkerLoop(splitEdt, plan, originalParamCount)))
      return failure();
  }

  builder.setInsertionPointAfter(splitLoop);
  DbAllocOp currentDb = partialDb;
  int64_t currentCount = plan.splitFactor;
  for (auto [nextDb, nextCount] : intermediateLevels) {
    for (int64_t outputIndex = 0; outputIndex < nextCount; ++outputIndex) {
      int64_t leftIndex = outputIndex * 2;
      std::optional<int64_t> rightIndex;
      if (leftIndex + 1 < currentCount)
        rightIndex = leftIndex + 1;
      Value outputTile = createConstantIndex(builder, loc, outputIndex);
      Value combineRoute =
          distributedTopology
              ? createDistributedRoute(builder, loc, ownerLoop, ownerIndex,
                                       outputTile, nextCount)
              : edt.getRoute();
      if (failed(createIntermediateCombineEdt(
              builder, loc, edt, currentDb, nextDb, resultDepType, ownerOrdinal,
              leftIndex, rightIndex, outputIndex, scalarType,
              resultElementCount, materializedConcurrency, combineRoute)))
        return failure();
    }
    currentDb = nextDb;
    currentCount = nextCount;
  }

  Value finalRoute =
      distributedTopology
          ? createDistributedRoute(builder, loc, ownerLoop, ownerIndex,
                                   createZeroIndex(builder, loc), 1)
          : edt.getRoute();
  if (failed(createFinalCombineEdt(
          builder, loc, edt, currentDb, originalDeps[plan.resultDepIndex],
          resultDepType, ownerOrdinal, currentCount, scalarType,
          resultElementCount, materializedConcurrency, finalRoute)))
    return failure();

  edt.erase();
  return success();
}

struct PartialReductionSplitMaterializationPass
    : public impl::PartialReductionSplitMaterializationBase<
          PartialReductionSplitMaterializationPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<EdtOp, 8> worklist;
    module.walk([&](EdtOp edt) {
      if (edt.getPartialReductionSplitRequiredAttr())
        worklist.push_back(edt);
    });

    for (EdtOp edt : worklist) {
      SplitPlan plan;
      if (failed(validateSplitPlan(edt, plan)) ||
          failed(materializeSplitPlan(edt, plan))) {
        edt.emitError()
            << "failed to materialize partial-reduction split plan; leaving "
               "partialReductionSplitRequired for ARTS-RT guard";
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::carts::arts::createPartialReductionSplitMaterializationPass() {
  return std::make_unique<PartialReductionSplitMaterializationPass>();
}
