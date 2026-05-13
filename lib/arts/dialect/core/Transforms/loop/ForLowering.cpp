///==========================================================================///
/// File: ForLowering.cpp
///
/// Lowers arts.for operations within EDT regions (parallel/task)
/// 1. Splits parallel regions at arts.for boundaries
/// 2. Creates EpochOp wrappers with task EDTs for loop iterations
/// 3. Creates continuation parallel EDTs for post-loop work
/// 4. Rewires DB acquires directly to DbAllocOp (not through block arguments)
///
/// Transformation:
///   BEFORE: edt.parallel { work_1; arts.for {...}; work_2 }
///   AFTER:  edt.parallel { work_1 }
///           arts.epoch { scf.for { edt.task {...} }; edt.task<result> }
///           edt.parallel { work_2 }  /// with reacquired DBs from DbAllocOp
///
/// Worker partitioning uses block distribution:
///   - blockSize = ceil(totalIterations / numWorkers)
///   - Each worker processes iterations [start, start + count)
///
/// Example:
///   one `arts.for` inside `arts.edt<parallel>` -> epoch + per-worker task EDTs
///==========================================================================///

#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AccessPatternAnalysis.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/db/DbAnalysis.h"
#include "arts/dialect/core/Analysis/heuristics/DistributionHeuristics.h"
#include "arts/dialect/core/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "arts/dialect/core/Transforms/db/DbLayoutPlanUtils.h"
#include "arts/utils/ValueAnalysis.h"
#define GEN_PASS_DEF_FORLOWERING
#include "arts/dialect/core/Analysis/db/OwnershipProof.h"
#include "arts/dialect/core/Transforms/edt/EdtParallelSplitLowering.h"
#include "arts/dialect/core/Transforms/edt/EdtReductionLowering.h"
#include "arts/dialect/core/Transforms/edt/EdtRewriter.h"
#include "arts/dialect/core/Transforms/edt/EdtTaskBodyCloning.h"
#include "arts/dialect/core/Transforms/edt/EdtTaskLoopCanonicalization.h"
#include "arts/dialect/core/Transforms/edt/EdtTaskLoopLowering.h"
#include "arts/dialect/core/Transforms/edt/WorkDistributionUtils.h"
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/DbUtils.h"
#include "arts/utils/EdtUtils.h"
#include "arts/utils/LoweringContractUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/PartitionPredicates.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <memory>
#include <optional>

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(for_lowering);

using namespace mlir::arts;

using namespace mlir;
using namespace mlir::func;

namespace {

static std::optional<unsigned> inferAcquireMappedDim(DbAnalysis *dbAnalysis,
                                                     DbAcquireOp acquire,
                                                     ForOp forOp) {
  if (!dbAnalysis || !acquire || !forOp)
    return std::nullopt;
  if (auto mappedDim = dbAnalysis->inferLoopMappedDim(acquire, forOp))
    return mappedDim;
  return dbAnalysis->inferLoopMappedDim(acquire.getPtr(), forOp);
}

static bool acquireHasAccessDependingOnLoopIv(DbAcquireOp acquire,
                                              ForOp forOp) {
  if (!acquire || !forOp)
    return false;

  Block &forBody = forOp.getRegion().front();
  if (forBody.getNumArguments() == 0)
    return false;

  Value loopIV = forBody.getArgument(0);
  Region &forRegion = forOp.getRegion();
  llvm::SetVector<Operation *> memOps;
  DbUtils::collectReachableMemoryOps(acquire.getPtr(), memOps,
                                     /*scope=*/nullptr);

  for (Operation *memOp : memOps) {
    Region *memRegion = memOp ? memOp->getParentRegion() : nullptr;
    if (!memRegion)
      continue;
    if (memRegion != &forRegion && !forRegion.isAncestor(memRegion))
      continue;

    std::optional<DbUtils::MemoryAccessInfo> access =
        DbUtils::getMemoryAccessInfo(memOp);
    if (!access || access->indices.empty())
      continue;

    SmallVector<Value> indexChain;
    Value baseMemref = ValueAnalysis::stripMemrefViewOps(access->memref);
    if (auto dbRef = baseMemref.getDefiningOp<DbRefOp>())
      indexChain = DbUtils::collectFullIndexChain(dbRef, memOp);
    if (indexChain.empty())
      indexChain.append(access->indices.begin(), access->indices.end());

    for (Value index : indexChain)
      if (ValueAnalysis::dependsOn(ValueAnalysis::stripNumericCasts(index),
                                   loopIV))
        return true;
  }

  return false;
}

static std::optional<AccessIndexInfo>
buildAccessIndexInfo(Operation *memOp) {
  std::optional<DbUtils::MemoryAccessInfo> access =
      DbUtils::getMemoryAccessInfo(memOp);
  if (!access || access->indices.empty())
    return std::nullopt;

  AccessIndexInfo info;
  Value baseMemref = ValueAnalysis::stripMemrefViewOps(access->memref);
  if (auto dbRef = baseMemref.getDefiningOp<DbRefOp>()) {
    SmallVector<Value> indexChain = DbUtils::collectFullIndexChain(dbRef, memOp);
    if (indexChain.empty())
      return std::nullopt;
    info.dbRefPrefix = dbRef.getIndices().size();
    info.indexChain.append(indexChain.begin(), indexChain.end());
  } else {
    info.dbRefPrefix = 0;
    info.indexChain.append(access->indices.begin(), access->indices.end());
  }

  return info;
}

static bool writeOwnerAccessesTrackLoopIv(Value dep, ForOp forOp,
                                          unsigned ownerDim) {
  if (!dep || !forOp)
    return false;

  Block &forBody = forOp.getRegion().front();
  if (forBody.getNumArguments() == 0)
    return false;

  Value loopIV = forBody.getArgument(0);
  Region &forRegion = forOp.getRegion();
  llvm::SetVector<Operation *> memOps;
  DbUtils::collectReachableMemoryOps(dep, memOps, /*scope=*/nullptr);

  bool sawWrite = false;
  for (Operation *memOp : memOps) {
    Region *memRegion = memOp ? memOp->getParentRegion() : nullptr;
    if (!memRegion)
      continue;
    if (memRegion != &forRegion && !forRegion.isAncestor(memRegion))
      continue;

    std::optional<DbUtils::MemoryAccessInfo> access =
        DbUtils::getMemoryAccessInfo(memOp);
    if (!access || !access->isWrite())
      continue;

    std::optional<AccessIndexInfo> accessInfo = buildAccessIndexInfo(memOp);
    if (!accessInfo)
      return false;

    AccessBoundsResult bounds = analyzeAccessBoundsFromIndices(
        ArrayRef<AccessIndexInfo>(&*accessInfo, 1), loopIV, loopIV, ownerDim);
    if (!bounds.valid || bounds.hasVariableOffset)
      return false;
    sawWrite = true;
  }

  return sawWrite;
}

static std::optional<ArtsMode> inferLoopLocalMode(Value dep, ForOp forOp);

static bool hasOwnerMismatchedWriteDep(ForOp forOp, EdtOp parallelEdt) {
  if (!forOp || !parallelEdt)
    return false;

  ValueRange parentDeps = parallelEdt.getDependencies();
  if (parallelEdt.getRegion().empty())
    return false;
  Block &parallelBlock = parallelEdt.getRegion().front();

  for (auto [idx, parentDep] : llvm::enumerate(parentDeps)) {
    if (idx >= parallelBlock.getNumArguments())
      break;

    auto parentAcqOp = parentDep.getDefiningOp<DbAcquireOp>();
    if (!parentAcqOp)
      continue;

    BlockArgument parallelArg = parallelBlock.getArgument(idx);
    ArtsMode effectiveMode = parentAcqOp.getMode();
    if (effectiveMode == ArtsMode::inout)
      if (auto loopLocalMode = inferLoopLocalMode(parallelArg, forOp))
        effectiveMode = *loopLocalMode;
    if (effectiveMode != ArtsMode::out && effectiveMode != ArtsMode::inout)
      continue;

    auto sourceAlloc = dyn_cast_or_null<DbAllocOp>(
        DbUtils::getUnderlyingDbAlloc(parentAcqOp.getSourcePtr()));
    if (!sourceAlloc)
      continue;
    auto sourcePartitionMode = sourceAlloc.getPartitionMode();
    if (!sourcePartitionMode || !usesBlockLayout(*sourcePartitionMode))
      continue;

    std::optional<LoweringContractInfo> contract =
        resolveAcquireContract(parentAcqOp);
    if (!contract || contract->spatial.ownerDims.empty())
      continue;

    int64_t rawOwnerDim = contract->spatial.ownerDims.front();
    if (rawOwnerDim < 0)
      continue;
    if (static_cast<unsigned>(rawOwnerDim) >=
        sourceAlloc.getElementSizes().size())
      continue;
    if (auto ownerExtent = ValueAnalysis::tryFoldConstantIndex(
            sourceAlloc.getElementSizes()[static_cast<unsigned>(rawOwnerDim)]))
      if (*ownerExtent <= 1)
        continue;

    if (!writeOwnerAccessesTrackLoopIv(
            parallelArg, forOp, static_cast<unsigned>(rawOwnerDim)))
      return true;
  }

  return false;
}

static bool ownerDimsContain(ArrayRef<int64_t> ownerDims, unsigned dim) {
  for (int64_t ownerDim : ownerDims)
    if (ownerDim >= 0 && static_cast<unsigned>(ownerDim) == dim)
      return true;
  return false;
}

static SmallVector<int64_t, 4> getLoopOwnerDimsFromContractOrPlan(
    Operation *op, const std::optional<LoweringContractInfo> &contract) {
  SmallVector<int64_t, 4> ownerDims;
  if (contract && !contract->spatial.ownerDims.empty()) {
    ownerDims.assign(contract->spatial.ownerDims.begin(),
                     contract->spatial.ownerDims.end());
    return ownerDims;
  }
  if (auto planOwnerDims = readI64ArrayAttr(getPlanOwnerDimsAttr(op)))
    ownerDims.assign(planOwnerDims->begin(), planOwnerDims->end());
  return ownerDims;
}

static bool shouldShareCoarseInPlaceDb(ForOp forOp, EdtOp parallelEdt,
                                       DbAcquireOp parentAcquire,
                                       DbAllocOp sourceAlloc,
                                       ArtsMode effectiveTaskMode,
                                       bool parentHasPartitionInfo) {
  if (!forOp || !parallelEdt || !parentAcquire || !sourceAlloc)
    return false;
  if (!forOp.getInPlaceSharedState())
    return false;
  if (parallelEdt.getConcurrency() != EdtConcurrency::intranode)
    return false;
  if (effectiveTaskMode != ArtsMode::inout)
    return false;
  if (parentHasPartitionInfo || parentAcquire.getPreserveDepEdge())
    return false;
  if (hasPhysicalDbLayoutPlan(sourceAlloc.getOperation()) ||
      hasDistributedDbAllocation(sourceAlloc.getOperation()))
    return false;

  if (auto allocMode = sourceAlloc.getPartitionMode())
    if (usesBlockLayout(*allocMode))
      return false;
  if (auto acquireMode = parentAcquire.getPartitionMode())
    if (usesBlockLayout(*acquireMode))
      return false;

  return true;
}

static std::optional<AccessBoundsResult>
inferLoopHaloBoundsFromValue(Value dep, ForOp forOp, unsigned mappedDim) {
  if (!dep || !forOp)
    return std::nullopt;

  Block &forBody = forOp.getRegion().front();
  if (forBody.getNumArguments() == 0)
    return std::nullopt;

  Value loopIV = forBody.getArgument(0);
  Region &forRegion = forOp.getRegion();
  llvm::SetVector<Operation *> memOps;
  DbUtils::collectReachableMemoryOps(dep, memOps, /*scope=*/nullptr);

  SmallVector<AccessIndexInfo, 16> accesses;
  for (Operation *memOp : memOps) {
    Region *memRegion = memOp ? memOp->getParentRegion() : nullptr;
    if (!memRegion)
      continue;
    if (memRegion != &forRegion && !forRegion.isAncestor(memRegion))
      continue;

    std::optional<DbUtils::MemoryAccessInfo> access =
        DbUtils::getMemoryAccessInfo(memOp);
    if (!access || access->indices.empty())
      continue;

    if (std::optional<AccessIndexInfo> info = buildAccessIndexInfo(memOp))
      accesses.push_back(std::move(*info));
  }

  if (accesses.empty())
    return std::nullopt;

  AccessBoundsResult bounds =
      analyzeAccessBoundsFromIndices(accesses, loopIV, loopIV, mappedDim);
  if (!bounds.valid || (bounds.minOffset == 0 && bounds.maxOffset == 0))
    return std::nullopt;
  return bounds;
}

static std::optional<ArtsMode> inferLoopLocalMode(Value dep, ForOp forOp) {
  if (!dep || !forOp)
    return std::nullopt;

  Region &forRegion = forOp.getRegion();
  llvm::SetVector<Operation *> memOps;
  DbUtils::collectReachableMemoryOps(dep, memOps, /*scope=*/nullptr);

  bool hasRead = false;
  bool hasWrite = false;
  for (Operation *memOp : memOps) {
    Region *memRegion = memOp ? memOp->getParentRegion() : nullptr;
    if (!memRegion)
      continue;
    if (memRegion != &forRegion && !forRegion.isAncestor(memRegion))
      continue;

    std::optional<DbUtils::MemoryAccessInfo> access =
        DbUtils::getMemoryAccessInfo(memOp);
    if (!access)
      continue;

    hasRead |= access->isRead();
    hasWrite |= access->isWrite();
  }

  if (hasRead && hasWrite)
    return ArtsMode::inout;
  if (hasRead)
    return ArtsMode::in;
  if (hasWrite)
    return ArtsMode::out;
  return std::nullopt;
}

///===----------------------------------------------------------------------===///
/// LoopInfo - Information about a loop within a parallel EDT
///
/// LoopInfo encapsulates worker partitioning for arts.for inside a parallel
/// EDT. It implements a simple block distribution:
///   - blockSizeCeil = ceil(totalIterations / numWorkers)
///   - start = workerId * blockSizeCeil
///   - count = min(blockSizeCeil, max(0, totalIterations - start))
///   - hasWork = start < totalIterations
///===----------------------------------------------------------------------===///
class LoopInfo {
public:
  LoopInfo(ArtsCodegen *AC, ForOp forOp, Value numWorkers,
           Value runtimeBlockSizeHint = Value())
      : AC(AC), forOp(forOp) {
    lowerBound = forOp.getLowerBound()[0];
    upperBound = forOp.getUpperBound()[0];
    loopStep = forOp.getStep()[0];
    distributionContract =
        resolveLoopDistributionContract(forOp.getOperation());
    totalWorkers = numWorkers;
    LoopChunkPlan chunkPlan = WorkDistributionUtils::planLoopChunking(
        AC, forOp, distributionContract, runtimeBlockSizeHint);
    blockSize = chunkPlan.blockSize;
    chunkLowerBound = chunkPlan.chunkLowerBound;
    totalIterations = chunkPlan.totalIterations;
    totalChunks = chunkPlan.totalChunks;
    useAlignedLowerBound = chunkPlan.useAlignedLowerBound;
    useRuntimeBlockAlignment = chunkPlan.useRuntimeBlockAlignment;
    alignmentBlockSize = chunkPlan.alignmentBlockSize;
    runtimeAlignmentBlockSize = chunkPlan.runtimeAlignmentBlockSize;
  }

  /// Attributes
  ArtsCodegen *AC;

  /// Loop information
  ForOp forOp;
  Value lowerBound, upperBound, loopStep;
  /// Base lower bound used for chunking (may be aligned to block size)
  Value chunkLowerBound;
  bool useAlignedLowerBound = false;
  bool useRuntimeBlockAlignment = false;
  std::optional<int64_t> alignmentBlockSize;
  Value runtimeAlignmentBlockSize;
  /// Distribution information
  Value blockSize, totalWorkers, totalIterations, totalChunks;
  LoweringContractInfo distributionContract;

  /// Distribution strategy and bounds from DistributionHeuristics
  DistributionStrategy strategy;
  DistributionBounds bounds;       ///  Set by computeBounds()
  DistributionBounds insideBounds; ///  Set by recomputeBoundsInside()
};

using ReductionInfo = ReductionLoweringInfo;
using ParallelRegionAnalysis = ParallelRegionSplitAnalysis;
static bool hasRepeatedWaveGroupKind(Operation *op) {
  return hasStringAttrValue(op, AttrNames::Operation::Orchestration::Kind,
                            AttrNames::Operation::RepeatedWaveGroup);
}

static std::optional<int64_t> getI64Attr(Operation *op, StringRef attrName) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return std::nullopt;
}

static std::optional<int64_t> getOrchestrationGroupId(Operation *op) {
  return getI64Attr(op, AttrNames::Operation::Orchestration::GroupId);
}

static std::optional<int64_t> getOrchestrationWaveIndex(Operation *op) {
  return getI64Attr(op, AttrNames::Operation::Orchestration::WaveIndex);
}

static std::optional<int64_t> getOrchestrationWaveCount(Operation *op) {
  return getI64Attr(op, AttrNames::Operation::Orchestration::WaveCount);
}

static bool haveCompatibleOrchestrationContract(EdtOp lhs, EdtOp rhs) {
  if (!lhs || !rhs)
    return false;
  if (lhs.getType() != EdtType::parallel || rhs.getType() != EdtType::parallel)
    return false;
  if (lhs.getConcurrency() != rhs.getConcurrency())
    return false;
  if (getWorkers(lhs.getOperation()) != getWorkers(rhs.getOperation()))
    return false;
  if (getWorkersPerNode(lhs.getOperation()) !=
      getWorkersPerNode(rhs.getOperation()))
    return false;

  Value lhsRoute = lhs.getRoute();
  Value rhsRoute = rhs.getRoute();
  if (static_cast<bool>(lhsRoute) != static_cast<bool>(rhsRoute))
    return false;
  if (lhsRoute && !ValueAnalysis::sameValue(lhsRoute, rhsRoute))
    return false;

  SmallVector<ForOp, 2> lhsForOps = EdtUtils::getTopLevelForOps(lhs);
  SmallVector<ForOp, 2> rhsForOps = EdtUtils::getTopLevelForOps(rhs);
  if (lhsForOps.size() != 1 || rhsForOps.size() != 1)
    return false;

  Operation *lhsFor = lhsForOps.front().getOperation();
  Operation *rhsFor = rhsForOps.front().getOperation();
  return getDepPattern(lhsFor) == getDepPattern(rhsFor) &&
         getEdtDistributionKind(lhsFor) == getEdtDistributionKind(rhsFor) &&
         getEdtDistributionPattern(lhsFor) == getEdtDistributionPattern(rhsFor);
}

static SmallVector<EdtOp, 4> collectOrchestratedWaveGroup(EdtOp seed) {
  SmallVector<EdtOp, 4> groupedEdts;
  if (!seed || !hasRepeatedWaveGroupKind(seed.getOperation()))
    return groupedEdts;

  std::optional<int64_t> groupId = getOrchestrationGroupId(seed.getOperation());
  if (!groupId)
    return groupedEdts;

  Block *parentBlock = seed->getBlock();
  if (!parentBlock)
    return groupedEdts;

  for (Operation &op : *parentBlock) {
    auto candidate = dyn_cast<EdtOp>(&op);
    if (!candidate)
      continue;
    if (!hasRepeatedWaveGroupKind(candidate.getOperation()))
      continue;
    if (getOrchestrationGroupId(candidate.getOperation()) != groupId)
      continue;
    if (!haveCompatibleOrchestrationContract(seed, candidate))
      continue;
    groupedEdts.push_back(candidate);
  }

  llvm::sort(groupedEdts, [](EdtOp lhs, EdtOp rhs) {
    int64_t lhsIndex =
        getOrchestrationWaveIndex(lhs.getOperation()).value_or(0);
    int64_t rhsIndex =
        getOrchestrationWaveIndex(rhs.getOperation()).value_or(0);
    return lhsIndex < rhsIndex;
  });
  return groupedEdts;
}

/// Count memrefs in the loop body that have both reads and writes,
/// which indicates potential loop-carried dependencies.
static int64_t countMemrefsWithLoopCarriedDeps(arts::ForOp forOp) {
  DenseMap<Value, bool> hasRead, hasWrite;
  SmallPtrSet<Value, 8> seenMemrefs;

  forOp.getBody()->walk([&](Operation *op) {
    Value memref;
    if (auto load = dyn_cast<memref::LoadOp>(op))
      memref = load.getMemref();
    else if (auto store = dyn_cast<memref::StoreOp>(op))
      memref = store.getMemRef();
    else
      return;

    seenMemrefs.insert(memref);
    if (isa<memref::LoadOp>(op))
      hasRead[memref] = true;
    else
      hasWrite[memref] = true;
  });

  int64_t count = 0;
  for (Value m : seenMemrefs) {
    if (hasRead.count(m) && hasWrite.count(m))
      ++count;
  }
  return count;
}

/// ForLowering Pass Implementation
struct ForLoweringPass : public impl::ForLoweringBase<ForLoweringPass> {
  explicit ForLoweringPass(mlir::arts::AnalysisManager *AM = nullptr)
      : AM(AM),
        numEdtsLowered(this, "num-edts-lowered",
                       "Number of EDT regions rewritten to lower arts.for"),
        numForOpsLowered(this, "num-arts-for-lowered",
                         "Number of arts.for operations lowered"),
        numEpochsCreated(this, "num-for-epochs-created",
                         "Number of epochs created for lowered arts.for"),
        numTaskEdtsCreated(this, "num-loop-task-edts-created",
                           "Number of worker task EDTs created for lowered "
                           "arts.for"),
        numInlineSingleLaneLowerings(
            this, "num-inline-single-lane-lowerings",
            "Number of lowered loops executed inline because dispatch "
            "collapsed to one lane"),
        numContinuationParallelsCreated(
            this, "num-continuation-parallels-created",
            "Number of continuation parallel EDTs created for post-loop work"),
        numReductionResultEdtsCreated(
            this, "num-reduction-result-edts-created",
            "Number of reduction result EDTs created after loop lowering") {
    assert(AM && "AnalysisManager must be provided externally");
  }
  ForLoweringPass(const ForLoweringPass &other)
      : impl::ForLoweringBase<ForLoweringPass>(other), AM(other.AM),
        numEdtsLowered(this, "num-edts-lowered",
                       "Number of EDT regions rewritten to lower arts.for"),
        numForOpsLowered(this, "num-arts-for-lowered",
                         "Number of arts.for operations lowered"),
        numEpochsCreated(this, "num-for-epochs-created",
                         "Number of epochs created for lowered arts.for"),
        numTaskEdtsCreated(this, "num-loop-task-edts-created",
                           "Number of worker task EDTs created for lowered "
                           "arts.for"),
        numInlineSingleLaneLowerings(
            this, "num-inline-single-lane-lowerings",
            "Number of lowered loops executed inline because dispatch "
            "collapsed to one lane"),
        numContinuationParallelsCreated(
            this, "num-continuation-parallels-created",
            "Number of continuation parallel EDTs created for post-loop work"),
        numReductionResultEdtsCreated(
            this, "num-reduction-result-edts-created",
            "Number of reduction result EDTs created after loop lowering") {}
  void runOnOperation() override;

private:
  ModuleOp module;
  mlir::arts::AnalysisManager *AM = nullptr;
  Statistic numEdtsLowered;
  Statistic numForOpsLowered;
  Statistic numEpochsCreated;
  Statistic numTaskEdtsCreated;
  Statistic numInlineSingleLaneLowerings;
  Statistic numContinuationParallelsCreated;
  Statistic numReductionResultEdtsCreated;

  void gatherLowerableEdts(SmallVectorImpl<EdtOp> &lowerableEdts);
  void lowerCollectedEdts();

  /// Process a parallel EDT containing arts_for operations
  void lowerParallelEdt(EdtOp parallelEdt);
  bool lowerOrchestratedWaveGroup(ArrayRef<EdtOp> groupedEdts);

  /// Clone loop body into task EDT's scf.for
  void cloneLoopBody(ArtsCodegen *AC, ForOp forOp, scf::ForOp chunkLoop,
                     Value chunkOffset, IRMapping &mapper);

  /// Reduction Support

  /// Allocate partial accumulators for reductions (one per worker)
  /// If splitMode is true, skip creating DbAcquireOps as dependencies to the
  /// parallel EDT (used when the parallel will be erased in split pattern)
  ReductionInfo
  allocatePartialAccumulators(ArtsCodegen *AC, ForOp forOp, EdtOp parallelEdt,
                              Location loc, bool splitMode = false,
                              Value workerCountOverride = Value());

  void createResultEdt(ArtsCodegen *AC, ReductionInfo &redInfo, Location loc);

  /// Lower an arts.for with DB acquires rewired directly to DbAllocOp.
  /// This is used when splitting the parallel region - the for body is
  /// extracted outside the parallel EDT and acquires DBs directly.
  void lowerForWithDbRewiring(ArtsCodegen &AC, ForOp forOp,
                              EdtOp originalParallel,
                              ParallelRegionAnalysis &analysis, Location loc,
                              Block *forcedEpochBlock = nullptr,
                              Operation *setupInsertionAnchor = nullptr);

  /// Create task EDT with DB dependencies rewired to DbAllocOp.
  EdtOp createTaskEdtWithRewiring(ArtsCodegen *AC, LoopInfo &loopInfo,
                                  ForOp forOp, Value workerIdPlaceholder,
                                  EdtOp originalParallel,
                                  bool singleDispatchLane,
                                  ReductionInfo &redInfo);
};

} // namespace

/// LoopInfo computes how to distribute loop iterations across workers using
/// block distribution. This ensures balanced work with minimal overhead.

///===----------------------------------------------------------------------===///
/// Pass Entry Point
///===----------------------------------------------------------------------===///

void ForLoweringPass::runOnOperation() {
  module = getOperation();

  /// Phase 1: Gather candidate EDTs carrying arts.for operations.
  SmallVector<EdtOp, 4> lowerableEdts;
  gatherLowerableEdts(lowerableEdts);

  /// Skip pass bookkeeping/logging entirely when no arts.for is present.
  if (lowerableEdts.empty()) {
    ARTS_DEBUG("No arts.for operations found, skipping ForLowering");
    return;
  }

  ARTS_INFO_HEADER(ForLowering);
  ARTS_DEBUG_REGION(module.dump(););

  /// Phase 2-4: Evaluate lowering policy, build rewrite plans, and apply.
  lowerCollectedEdts();

  ARTS_INFO_FOOTER(ForLowering);
  ARTS_DEBUG_REGION(module.dump(););
}

void ForLoweringPass::gatherLowerableEdts(
    SmallVectorImpl<EdtOp> &lowerableEdts) {
  module.walk([&](EdtOp edt) {
    if (edt.getType() != EdtType::parallel && edt.getType() != EdtType::task)
      return;

    bool hasForOps = edt.getBody()
                         .walk([&](ForOp) { return WalkResult::interrupt(); })
                         .wasInterrupted();
    if (hasForOps)
      lowerableEdts.push_back(edt);
  });
}

void ForLoweringPass::lowerCollectedEdts() {
  DenseSet<Operation *> processed;
  while (true) {
    EdtOp parallelEdt;
    module.walk([&](EdtOp edt) {
      if (parallelEdt)
        return WalkResult::interrupt();
      if (processed.contains(edt.getOperation()))
        return WalkResult::advance();
      if (edt.getType() != EdtType::parallel && edt.getType() != EdtType::task)
        return WalkResult::advance();
      bool hasForOps = edt.getBody()
                           .walk([&](ForOp) { return WalkResult::interrupt(); })
                           .wasInterrupted();
      if (!hasForOps)
        return WalkResult::advance();
      parallelEdt = edt;
      return WalkResult::interrupt();
    });

    if (!parallelEdt)
      break;

    SmallVector<EdtOp, 4> groupedEdts =
        collectOrchestratedWaveGroup(parallelEdt);
    if (groupedEdts.size() > 1 &&
        getOrchestrationWaveIndex(parallelEdt.getOperation()).value_or(-1) ==
            0) {
      if (lowerOrchestratedWaveGroup(groupedEdts)) {
        // The grouped EDTs were already erased by cleanupOriginalParallel
        // inside lowerOrchestratedWaveGroup — do not insert into processed.
        continue;
      }
      ARTS_WARN("Falling back to per-wave lowering for unsupported "
                "orchestration group");
    }

    processed.insert(parallelEdt.getOperation());
    lowerParallelEdt(parallelEdt);
  }
}

/// Hoist a value out of nested regions (scf.if + EDT body) to a target block
/// so it dominates code placed before the parallel EDT.
/// The `targetBlock` is the block containing the parallel EDT.
/// Returns the hoisted value, or the original if it already dominates.
static Value hoistValueToBlock(Value val, Block *targetBlock,
                               Operation *insertBefore, OpBuilder &builder) {
  if (!val)
    return val;

  auto *defOp = val.getDefiningOp();
  if (!defOp)
    return val; // block argument — check if it's in the target block

  // If already defined in the target block, no hoisting needed
  if (defOp->getBlock() == targetBlock)
    return val;

  // For arith.select(cond, trueVal, falseVal) inside scf.if then-region,
  // resolve to trueVal when the scf.if condition matches the select condition.
  if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    if (auto ifOp = defOp->getParentOfType<scf::IfOp>()) {
      if (selectOp.getCondition() == ifOp.getCondition() &&
          ifOp.getThenRegion().isAncestor(defOp->getParentRegion()))
        return hoistValueToBlock(selectOp.getTrueValue(), targetBlock,
                                 insertBefore, builder);
    }
  }

  // Recursively hoist operands first
  SmallVector<Value> hoistedOperands;
  for (Value operand : defOp->getOperands())
    hoistedOperands.push_back(
        hoistValueToBlock(operand, targetBlock, insertBefore, builder));

  // Clone the operation into the target block before insertBefore
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertBefore);
  Operation *cloned = builder.clone(*defOp);
  for (auto [idx, operand] : llvm::enumerate(hoistedOperands))
    cloned->setOperand(idx, operand);

  Value result = cloned->getResult(cast<OpResult>(val).getResultNumber());
  return result;
}

/// For ForOps inside scf.if guards within a parallel EDT, hoist bound operands
/// to the block containing the EDT so they dominate epoch code placed before
/// it.
static void hoistForBoundsFromScfIf(ForOp forOp, EdtOp parallelEdt,
                                    OpBuilder &builder) {
  auto ifOp = forOp->getParentOfType<scf::IfOp>();
  if (!ifOp)
    return;

  // Only handle ForOps inside an scf.if that's within this parallel EDT
  if (!parallelEdt.getBody().isAncestor(ifOp->getParentRegion()))
    return;

  Block *targetBlock = parallelEdt->getBlock();
  Operation *insertBefore = parallelEdt.getOperation();

  ARTS_DEBUG("Hoisting ForOp bounds to before parallel EDT");

  // Hoist lower bound, upper bound, step, and chunk size
  for (Value lb : forOp.getLowerBound()) {
    Value hoisted = hoistValueToBlock(lb, targetBlock, insertBefore, builder);
    if (hoisted != lb)
      forOp->replaceUsesOfWith(lb, hoisted);
  }
  for (Value ub : forOp.getUpperBound()) {
    Value hoisted = hoistValueToBlock(ub, targetBlock, insertBefore, builder);
    if (hoisted != ub)
      forOp->replaceUsesOfWith(ub, hoisted);
  }
  for (Value step : forOp.getStep()) {
    Value hoisted = hoistValueToBlock(step, targetBlock, insertBefore, builder);
    if (hoisted != step)
      forOp->replaceUsesOfWith(step, hoisted);
  }
  if (Value chunk = forOp.getChunkSize()) {
    Value hoisted =
        hoistValueToBlock(chunk, targetBlock, insertBefore, builder);
    if (hoisted != chunk)
      forOp->replaceUsesOfWith(chunk, hoisted);
  }
}

void ForLoweringPass::lowerParallelEdt(EdtOp parallelEdt) {
  ARTS_INFO("Lowering parallel EDT with ALWAYS-SPLIT pattern");

  /// Analyze the parallel EDT structure to find arts.for operations
  /// and categorize operations as before/after the for loop
  ParallelRegionAnalysis analysis =
      ParallelRegionAnalysis::analyze(parallelEdt);

  if (!analysis.needsSplit()) {
    ARTS_DEBUG(" - No arts.for operations found, skipping");
    return;
  }

  ++numEdtsLowered;
  ARTS_INFO(" - Found " << analysis.forOps.size() << " arts.for operation(s)");

  /// Analyze which DBs are used by for and post-for operations
  analysis.analyzeDependenciesForSplit(parallelEdt);

  bool hasPreFor = analysis.hasWorkBefore();
  bool hasPostFor = analysis.hasWorkAfter();

  ARTS_INFO(" - Pre-for work: " << (hasPreFor ? "yes" : "no")
                                << ", Post-for work: "
                                << (hasPostFor ? "yes" : "no"));
  ARTS_DEBUG(" - Deps used by for: " << analysis.depsUsedByFor.size());
  ARTS_DEBUG(" - Deps used after for: " << analysis.depsUsedAfterFor.size());

  ArtsCodegen AC(module);
  if (AM)
    AC.setRuntimeConfig(&AM->getRuntimeConfig());
  Location loc = parallelEdt.getLoc();

  /// Extract for-body outside the parallel EDT
  /// The transformation is:
  /// - Pre-for work stays in original parallel (if any)
  /// - For-body becomes: EpochOp { scf.for { task EDTs } + result EDT }
  /// - Post-for work goes into a new continuation parallel (if any)

  /// Step 1: Set insertion point after original parallel EDT
  AC.setInsertionPointAfter(parallelEdt);

  /// Step 1.5: Hoist ForOp bound operands out of scf.if guards so they
  /// dominate the epoch setup code placed before the parallel EDT.
  for (ForOp forOp : analysis.forOps)
    hoistForBoundsFromScfIf(forOp, parallelEdt, AC.getBuilder());

  /// Step 2: Lower each arts.for with DB rewiring (create epoch + tasks)
  /// This creates the lowered for structure AFTER the parallel EDT
  for (ForOp forOp : analysis.forOps)
    lowerForWithDbRewiring(AC, forOp, parallelEdt, analysis, loc);

  /// Step 3: Create continuation parallel for post-for work (if any)
  /// Skip if opsAfterFor only contains DB release cleanup operations.
  /// Keep barrier-only continuations intact because barrier semantics may be
  /// required by timestep-style loops (e.g., stencil updates) between epochs.
  if (hasPostFor) {
    bool onlyCleanup = llvm::all_of(analysis.opsAfterFor, [](Operation *op) {
      return isa<DbReleaseOp>(op);
    });
    if (!onlyCleanup) {
      createContinuationParallel(AC, parallelEdt, analysis, loc);
      ++numContinuationParallelsCreated;
    }
  }

  /// Step 4: Clean up original parallel EDT + dependencies.
  cleanupOriginalParallel(parallelEdt, analysis, hasPreFor);

  ARTS_INFO(" - Parallel EDT lowering complete");
}

bool ForLoweringPass::lowerOrchestratedWaveGroup(ArrayRef<EdtOp> groupedEdts) {
  if (groupedEdts.size() < 2)
    return false;

  EdtOp firstEdt = groupedEdts.front();
  if (auto expectedWaveCount =
          getOrchestrationWaveCount(firstEdt.getOperation())) {
    if (*expectedWaveCount != static_cast<int64_t>(groupedEdts.size()))
      return false;
  }

  SmallVector<ParallelRegionAnalysis, 4> analyses;
  analyses.reserve(groupedEdts.size());
  for (auto [idx, groupedEdt] : llvm::enumerate(groupedEdts)) {
    EdtOp groupedEdtValue = groupedEdt;
    if (!haveCompatibleOrchestrationContract(firstEdt, groupedEdtValue))
      return false;
    if (getOrchestrationWaveIndex(groupedEdtValue.getOperation())
            .value_or(-1) != static_cast<int64_t>(idx)) {
      return false;
    }
    if (auto expectedWaveCount =
            getOrchestrationWaveCount(groupedEdtValue.getOperation())) {
      if (*expectedWaveCount != static_cast<int64_t>(groupedEdts.size()))
        return false;
    }
    ParallelRegionAnalysis analysis =
        ParallelRegionAnalysis::analyze(groupedEdtValue);
    if (!analysis.needsSplit() || analysis.hasWorkBefore() ||
        analysis.hasWorkAfter() || analysis.forOps.size() != 1) {
      return false;
    }
    if (!analysis.forOps.front().getReductionAccumulators().empty())
      return false;
    analyses.push_back(std::move(analysis));
  }

  ArtsCodegen AC(module);
  if (AM)
    AC.setRuntimeConfig(&AM->getRuntimeConfig());
  Location loc = firstEdt.getLoc();
  AC.setInsertionPointAfter(firstEdt);

  auto sharedEpoch = AC.create<EpochOp>(loc, ValueRange{});
  ++numEpochsCreated;
  if (auto kindAttr = firstEdt->getAttrOfType<StringAttr>(
          AttrNames::Operation::Orchestration::Kind)) {
    sharedEpoch->setAttr(AttrNames::Operation::Orchestration::Kind, kindAttr);
  }
  if (auto groupIdAttr = firstEdt->getAttrOfType<IntegerAttr>(
          AttrNames::Operation::Orchestration::GroupId)) {
    sharedEpoch->setAttr(AttrNames::Operation::Orchestration::GroupId,
                         groupIdAttr);
  }
  if (auto waveCountAttr = firstEdt->getAttrOfType<IntegerAttr>(
          AttrNames::Operation::Orchestration::WaveCount)) {
    sharedEpoch->setAttr(AttrNames::Operation::Orchestration::WaveCount,
                         waveCountAttr);
  }
  transferOperationContract(analyses.front().forOps.front().getOperation(),
                            sharedEpoch.getOperation());
  copyPlanAttrs(analyses.front().forOps.front().getOperation(),
                sharedEpoch.getOperation());
  Region &epochRegion = sharedEpoch.getRegion();
  if (epochRegion.empty())
    epochRegion.push_back(new Block());
  Block *epochBlock = &epochRegion.front();

  for (auto [idx, groupedEdt] : llvm::enumerate(groupedEdts)) {
    ++numEdtsLowered;
    ParallelRegionAnalysis &analysis = analyses[idx];
    EdtOp groupedEdtValue = groupedEdt;
    lowerForWithDbRewiring(AC, analysis.forOps.front(), groupedEdtValue,
                           analysis, groupedEdtValue.getLoc(), epochBlock,
                           sharedEpoch.getOperation());
    cleanupOriginalParallel(groupedEdtValue, analysis, /*hasPreFor=*/false);
  }

  AC.setInsertionPointToEnd(epochBlock);
  if (epochBlock->empty() ||
      !epochBlock->back().hasTrait<OpTrait::IsTerminator>()) {
    AC.create<YieldOp>(loc);
  }
  AC.setInsertionPointAfter(sharedEpoch);

  ARTS_INFO(" - Lowered orchestrated wave group with " << groupedEdts.size()
                                                       << " sibling waves");
  return true;
}

void ForLoweringPass::cloneLoopBody(ArtsCodegen *AC, ForOp forOp,
                                    scf::ForOp chunkLoop, Value chunkOffset,
                                    IRMapping &mapper) {
  Location loc = forOp.getLoc();
  AC->setInsertionPointToStart(chunkLoop.getBody());

  /// Compute global index from local iteration variable
  ///  - global_iter = chunkOffset + local_iter
  ///  - global_idx = lowerBound + global_iter * step
  Value localIter = chunkLoop.getInductionVar();
  Value globalIter = AC->create<arith::AddIOp>(loc, chunkOffset, localIter);
  Value globalIterScaled =
      AC->create<arith::MulIOp>(loc, globalIter, forOp.getStep()[0]);
  Value globalIdx = AC->create<arith::AddIOp>(loc, forOp.getLowerBound()[0],
                                              globalIterScaled);

  /// Map arts.for induction variable to computed global index
  Block &forBody = forOp.getRegion().front();
  if (forBody.getNumArguments() > 0) {
    BlockArgument forIV = forBody.getArgument(0);
    mapper.map(forIV, globalIdx);
  }

  /// Collect constants used in the loop body that are defined outside
  /// and clone them inside the EDT region
  SetVector<Value> constantsToClone;
  Region *taskEdtRegion = chunkLoop->getParentRegion();
  for (Operation &op : forBody.without_terminator()) {
    for (Value operand : op.getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (defOp->hasTrait<OpTrait::ConstantLike>()) {
          /// Check if this constant is defined outside the task EDT region
          Region *defRegion = defOp->getParentRegion();
          if (!taskEdtRegion->isAncestor(defRegion)) {
            constantsToClone.insert(operand);
          }
        }
      }
    }
  }

  /// Clone constants inside the task EDT region before cloning operations
  /// Use the builder with the correct insertion point
  for (Value constant : constantsToClone) {
    if (Operation *defOp = constant.getDefiningOp()) {
      Operation *clonedConst = AC->getBuilder().clone(*defOp);
      mapper.map(constant, clonedConst->getResult(0));
    }
  }

  /// Clone all operations from arts.for body into chunk loop
  for (Operation &op : forBody.without_terminator())
    AC->clone(op, mapper);

  ARTS_DEBUG(
      "    Cloned " << std::distance(forBody.without_terminator().begin(),
                                     forBody.without_terminator().end())
                    << " operations into chunk loop");
}

ReductionInfo ForLoweringPass::allocatePartialAccumulators(
    ArtsCodegen *AC, ForOp forOp, EdtOp parallelEdt, Location loc,
    bool splitMode, Value workerCountOverride) {
  return mlir::arts::allocatePartialAccumulators(
      AC, forOp, parallelEdt, loc, splitMode, workerCountOverride);
}

void ForLoweringPass::createResultEdt(ArtsCodegen *AC, ReductionInfo &redInfo,
                                      Location loc) {
  arts::createResultEdt(AC, redInfo, loc);
}

///===----------------------------------------------------------------------===///
/// Parallel Region Splitting Implementation
///===----------------------------------------------------------------------===///
void ForLoweringPass::lowerForWithDbRewiring(ArtsCodegen &AC, ForOp forOp,
                                             EdtOp originalParallel,
                                             ParallelRegionAnalysis &analysis,
                                             Location loc,
                                             Block *forcedEpochBlock,
                                             Operation *setupInsertionAnchor) {
  ARTS_INFO(" - Lowering arts.for with DB rewiring (split pattern)");
  ++numForOpsLowered;

  /// Read distribution strategy selected by EdtDistributionPass.
  DistributionStrategy strategy =
      AM->getEdtHeuristics().resolveLoweringStrategy(originalParallel, forOp);

  Value zero;
  Value one;
  std::optional<LoopInfo> loopInfoStorage;
  Value dispatchWorkers;
  ReductionInfo redInfo;

  {
    OpBuilder::InsertionGuard guard(AC.getBuilder());
    /// Values consumed by split-mode reduction allocation must dominate the
    /// allocs inserted before the original parallel EDT.
    if (setupInsertionAnchor)
      AC.setInsertionPoint(setupInsertionAnchor);
    else
      AC.setInsertionPoint(originalParallel);

    bool forceSingleOwnerLane =
        hasOwnerMismatchedWriteDep(forOp, originalParallel);
    if (forceSingleOwnerLane) {
      // The current block DB plan partitions the primary owner dimension. If a
      // writer loop distributes a different dimension, splitting by workers
      // gives each task the wrong owner slice. Keep the physical layout but
      // execute this loop as one lane until SDE can choose a matching layout.
      setWorkers(forOp.getOperation(), 1);
      setWorkers(originalParallel.getOperation(), 1);
    }

    /// Get numWorkers from explicit attrs or runtime queries.
    Value numWorkers =
        WorkDistributionUtils::getTotalWorkers(&AC, loc, originalParallel);

    /// numDbPartitions: the number of DB partition blocks (= numNodes for
    /// internode). Used to align worker chunk boundaries to DB block
    /// boundaries.
    Value numDbPartitions;
    if (originalParallel.getConcurrency() == EdtConcurrency::internode) {
      numDbPartitions = AC.castToIndex(
          AC.create<RuntimeQueryOp>(loc, RuntimeQueryKind::totalNodes)
              .getResult(),
          loc);
    } else {
      numDbPartitions = numWorkers;
    }

    zero = AC.createIndexConstant(0, loc);
    one = AC.createIndexConstant(1, loc);

    /// Determine loop chunking before building the epoch so we can size the
    /// dispatch loop and any reduction temporaries to the worker lanes that can
    /// actually receive work.
    Value runtimeBlockSize = WorkDistributionUtils::computeDbAlignmentBlockSize(
        forOp, originalParallel, numDbPartitions, &AC, loc,
        AM->getDbAnalysis());
    loopInfoStorage.emplace(&AC, forOp, numWorkers, runtimeBlockSize);
    LoopInfo &loopInfo = *loopInfoStorage;
    loopInfo.strategy = strategy;
    dispatchWorkers = WorkDistributionUtils::getForDispatchWorkerCount(
        &AC, loc, originalParallel, loopInfo.strategy, loopInfo.totalChunks,
        loopInfo.distributionContract);

    /// Allocate reduction accumulators (same as before, but place BEFORE epoch)
    /// Use splitMode=true since we're splitting the parallel EDT and will
    /// create acquires directly in result/task EDTs.
    if (!forOp.getReductionAccumulators().empty()) {
      ARTS_INFO(" - Detected reduction(s), allocating partial accumulators");
      redInfo =
          allocatePartialAccumulators(&AC, forOp, originalParallel, loc,
                                      /*splitMode=*/true, dispatchWorkers);
    }
  }

  /// Treat both a truly single-worker topology and a one-lane dispatch clamp
  /// as the trivial lowering case. The dispatch worker count can remain
  /// symbolically wrapped in min/max arithmetic even when the topology has
  /// already collapsed to one worker, so check the resolved worker topology
  /// as well as the dispatch count itself.
  const bool singleDispatchLane =
      ValueAnalysis::isOneConstant(loopInfoStorage->totalWorkers) ||
      ValueAnalysis::isOneConstant(dispatchWorkers);
  const bool reuseEnclosingEpoch =
      forOp.getReductionAccumulators().empty() &&
      originalParallel->getParentOfType<EpochOp>() &&
      loopInfoStorage->distributionContract.shouldReuseEnclosingEpoch();

  std::optional<EpochOp> forEpoch;
  Block *epochBlock = nullptr;
  if (forcedEpochBlock) {
    epochBlock = forcedEpochBlock;
    AC.setInsertionPointToEnd(epochBlock);
  } else if (reuseEnclosingEpoch) {
    epochBlock = originalParallel->getBlock();
    AC.setInsertionPoint(originalParallel);
  } else {
    /// Create EpochOp wrapper for the for-body at the caller-managed insertion
    /// point so multiple arts.for regions preserve source order.
    forEpoch = AC.create<EpochOp>(loc, ValueRange{});
    ++numEpochsCreated;
    Region &epochRegion = forEpoch->getRegion();
    if (epochRegion.empty())
      epochRegion.push_back(new Block());
    epochBlock = &epochRegion.front();
    AC.setInsertionPointToStart(epochBlock);
  }

  /// Worker dispatch loop - emit only the worker lanes that can receive at
  /// least one chunk for this loop/distribution strategy.
  auto workerLoop = AC.create<scf::ForOp>(loc, zero, dispatchWorkers, one);
  AC.setInsertionPointToStart(workerLoop.getBody());
  Value workerIV = workerLoop.getInductionVar();

  /// Create task EDT with DB rewiring
  createTaskEdtWithRewiring(&AC, *loopInfoStorage, forOp, workerIV,
                            originalParallel, singleDispatchLane, redInfo);
  if (forEpoch) {
    transferOperationContract(forOp.getOperation(), forEpoch->getOperation());
    copyPlanAttrs(forOp.getOperation(), forEpoch->getOperation());
  }

  /// After worker loop, create result EDT (if reductions)
  AC.setInsertionPointAfter(workerLoop);
  if (!redInfo.reductionVars.empty()) {
    createResultEdt(&AC, redInfo, loc);
    ++numReductionResultEdtsCreated;
  }

  if (forEpoch) {
    /// Close epoch with yield
    AC.setInsertionPointToEnd(epochBlock);
    AC.create<YieldOp>(loc);
    /// Set insertion point after epoch so subsequent for loops insert
    /// correctly.
    AC.setInsertionPointAfter(*forEpoch);
  } else if (forcedEpochBlock) {
    AC.setInsertionPointToEnd(epochBlock);
  } else {
    AC.setInsertionPointAfter(workerLoop);
  }
  if (!redInfo.reductionVars.empty())
    finalizeReductionAfterEpoch(&AC, redInfo, loc);

  ARTS_INFO(" - arts.for lowering with DB rewiring complete");
}

EdtOp ForLoweringPass::createTaskEdtWithRewiring(
    ArtsCodegen *AC, LoopInfo &loopInfo, ForOp forOp, Value workerIdPlaceholder,
    EdtOp originalParallel, bool singleDispatchLane, ReductionInfo &redInfo) {
  Location loc = forOp.getLoc();

  ARTS_DEBUG("  Creating task EDT with DB rewiring");

  /// Recreate total workers using shared distribution helper.
  loopInfo.totalWorkers =
      WorkDistributionUtils::getTotalWorkers(AC, loc, originalParallel);

  /// Compute workers-per-node for internode routing and TwoLevel distribution.
  /// Flat may still run in internode mode during debugging/experiments.
  Value workersPerNode;
  if (loopInfo.strategy.kind == DistributionKind::TwoLevel ||
      originalParallel.getConcurrency() == EdtConcurrency::internode) {
    workersPerNode =
        WorkDistributionUtils::getWorkersPerNode(AC, loc, originalParallel);
  }

  /// Compute worker iteration bounds using DistributionHeuristics
  loopInfo.bounds = WorkDistributionUtils::computeBounds(
      AC, loc, loopInfo.strategy, workerIdPlaceholder, loopInfo.totalWorkers,
      workersPerNode, loopInfo.totalIterations, loopInfo.totalChunks,
      loopInfo.blockSize, loopInfo.distributionContract);

  Value chunkOffset = loopInfo.bounds.iterStart;
  ValueRange parentDeps = originalParallel.getDependencies();
  Block &parallelBlock = originalParallel.getRegion().front();

  IRMapping mapper;

  /// Create scf.if to conditionally create task EDT only if worker has work
  auto ifOp = AC->create<scf::IfOp>(loc, loopInfo.bounds.hasWork,
                                    /*withElseRegion=*/false);
  AC->setInsertionPointToStart(&ifOp.getThenRegion().front());

  /// Detect reduction block arguments
  DenseSet<Value> reductionBlockArgs = redInfo.reductionVars.empty()
                                           ? DenseSet<Value>{}
                                           : detectReductionBlockArgs(forOp);

  Value one = AC->createIndexConstant(1, loc);
  Value taskWorkerId = workerIdPlaceholder;
  TaskLoopLoweringInput taskLoopInput{AC,
                                      loc,
                                      loopInfo.strategy,
                                      loopInfo.bounds,
                                      loopInfo.distributionContract,
                                      taskWorkerId,
                                      loopInfo.totalWorkers,
                                      workersPerNode,
                                      loopInfo.upperBound,
                                      loopInfo.lowerBound,
                                      loopInfo.chunkLowerBound
                                          ? loopInfo.chunkLowerBound
                                          : forOp.getLowerBound()[0],
                                      loopInfo.loopStep,
                                      loopInfo.blockSize,
                                      loopInfo.totalIterations,
                                      loopInfo.totalChunks,
                                      loopInfo.alignmentBlockSize,
                                      loopInfo.runtimeAlignmentBlockSize,
                                      loopInfo.useRuntimeBlockAlignment,
                                      loopInfo.useAlignedLowerBound};

  std::unique_ptr<EdtTaskLoopLowering> taskLoopLowering =
      EdtTaskLoopLowering::create(loopInfo.strategy.kind);
  TaskAcquirePlanningResult acquirePlanning =
      taskLoopLowering->planAcquireRewrite(taskLoopInput, chunkOffset);
  Value workerOffsetVal = acquirePlanning.workerBaseOffset;
  Value acquireOffsetVal = acquirePlanning.acquireOffset;
  Value acquireSizeVal = acquirePlanning.acquireSize;
  Value acquireHintSizeVal = acquirePlanning.acquireHintSize;
  Value stepVal = acquirePlanning.step;
  bool stepIsUnit = acquirePlanning.stepIsUnit;
  std::optional<Tiling2DWorkerGrid> tiling2DGrid = acquirePlanning.tiling2DGrid;

  /// Acquire worker-local partial accumulator slot for reductions
  SmallVector<Value> reductionTaskDeps;
  DenseMap<Value, uint64_t> reductionVarIndex;
  std::optional<SmallVector<int64_t, 4>> refinedTaskBlockShape;

  for (uint64_t i = 0; i < redInfo.reductionVars.size(); i++) {
    /// Use partialAccumPtrs (from DbAllocOp) instead of partialAccumArgs
    /// (BlockArguments inside parallel EDT) to avoid use-after-free when
    /// erasing the parallel EDT in split mode
    Value partialPtr = i < redInfo.partialAccumPtrs.size()
                           ? redInfo.partialAccumPtrs[i]
                           : Value();
    Value partialGuid = i < redInfo.partialAccumGuids.size()
                            ? redInfo.partialAccumGuids[i]
                            : Value();
    if (!partialPtr) {
      ARTS_ERROR("Missing partial accumulator ptr for reduction " << i);
      continue;
    }

    auto partialPtrType = cast<MemRefType>(partialPtr.getType());
    auto innerMemrefType = cast<MemRefType>(partialPtrType.getElementType());

    /// Acquire the worker's slice of the partial accumulator array
    /// Source directly from DbAllocOp (partialGuid, partialPtr)
    /// Use chunked partition mode since we're acquiring by worker offset
    /// offsets=[workerId] and sizes=[1] to get this worker's slot
    auto partialAcqOp = AC->create<DbAcquireOp>(
        loc, ArtsMode::inout, partialGuid, partialPtr, innerMemrefType,
        PartitionMode::block, /*indices=*/SmallVector<Value>{},
        /*offsets=*/SmallVector<Value>{workerIdPlaceholder},
        /*sizes=*/SmallVector<Value>{one},
        /*partition_indices=*/SmallVector<Value>{},
        /*partition_offsets=*/SmallVector<Value>{},
        /*partition_sizes=*/SmallVector<Value>{});
    /// Worker-local partial reduction acquires already have the exact slot
    /// access mode and must not be re-inferred later.
    partialAcqOp.setPreserveAccessMode();

    reductionTaskDeps.push_back(partialAcqOp.getResult(1));
    reductionVarIndex[redInfo.reductionVars[i]] = i;

    ARTS_DEBUG("  - Acquired worker-local partial accumulator slot");
  }

  SmallVector<Value> taskDeps;
  SmallVector<std::pair<BlockArgument, Value>> parallelArgToAcquire;
  bool forceOwnerRoute = false;
  ARTS_DEBUG("  - Processing " << parentDeps.size()
                               << " parent dependencies with DB rewiring");

  for (auto [idx, parentDep] : llvm::enumerate(parentDeps)) {
    size_t depIndex = idx;
    auto parentAcqOp = parentDep.getDefiningOp<DbAcquireOp>();
    if (!parentAcqOp) {
      ARTS_DEBUG("    - Dep " << depIndex << ": Not a DbAcquireOp, skipping");
      continue;
    }

    BlockArgument parallelArg = parallelBlock.getArgument(idx);
    if (shouldSkipReductionArg(parallelArg, redInfo, reductionBlockArgs))
      continue;

    /// Rewiring: Trace to DbAllocOp and acquire from there
    auto allocInfo = DbUtils::traceToDbAlloc(parentDep);
    if (!allocInfo) {
      ARTS_ERROR("Could not trace dependency " << depIndex << " to DbAllocOp");
      continue;
    }
    auto [rootGuid, rootPtr] = *allocInfo;
    Value rootGuidValue = rootGuid;
    Value rootPtrValue = rootPtr;

    /// Preserve only authoritative parent partitioning contracts. CreateDbs
    /// materializes default coarse offsets/sizes even when the source had no
    /// explicit DbControl contract; treating those defaults as authoritative
    /// blocks later worker-local rewrite planning and can leave
    /// block-distributed loops with whole-DB coarse dependencies.
    bool parentHasPartitionInfo =
        parentAcqOp.hasExplicitPartitionHints() &&
        (!parentAcqOp.isCoarse() || parentAcqOp.getPreserveAccessMode() ||
         parentAcqOp.getPreserveDepEdge());

    ArtsMode effectiveTaskMode = parentAcqOp.getMode();
    if (effectiveTaskMode == ArtsMode::inout)
      if (auto loopLocalMode = inferLoopLocalMode(parallelArg, forOp))
        effectiveTaskMode = *loopLocalMode;
    auto sourceAlloc = dyn_cast_or_null<DbAllocOp>(
        DbUtils::getUnderlyingDbAlloc(parentAcqOp.getSourcePtr()));
    bool hasSdePhysicalLayoutPlan =
        sourceAlloc && hasPhysicalDbLayoutPlan(sourceAlloc.getOperation());
    bool parentUsesBlockLayout =
        parentAcqOp.getPartitionMode() &&
        usesBlockLayout(*parentAcqOp.getPartitionMode());
    bool preservePlannedBlockRead = effectiveTaskMode == ArtsMode::in &&
                                    parentUsesBlockLayout &&
                                    hasSdePhysicalLayoutPlan;
    std::optional<unsigned> accessMappedDim =
        inferAcquireMappedDim(&AM->getDbAnalysis(), parentAcqOp, forOp);
    std::optional<LoweringContractInfo> loopSemanticContract =
        getSemanticContract(forOp.getOperation());
    SmallVector<int64_t, 4> loopOwnerDims =
        getLoopOwnerDimsFromContractOrPlan(forOp.getOperation(),
                                           loopSemanticContract);
    bool readOnlyOwnerMismatchedDep =
        effectiveTaskMode == ArtsMode::in && preservePlannedBlockRead &&
        accessMappedDim && !loopOwnerDims.empty() &&
        !ownerDimsContain(loopOwnerDims, *accessMappedDim);
    bool forceCoarseReadOnlyDep =
        effectiveTaskMode == ArtsMode::in && !preservePlannedBlockRead &&
        !acquireHasAccessDependingOnLoopIv(parentAcqOp, forOp);

    if (shouldShareCoarseInPlaceDb(forOp, originalParallel, parentAcqOp,
                                   sourceAlloc, effectiveTaskMode,
                                   parentHasPartitionInfo)) {
      /// SDE marks local OpenMP-compatible in-place loops when the source
      /// intentionally shares one backing store across worker iterations.
      /// Rewriting each worker to an inout block acquire over the one coarse
      /// DB maps every task to the same dependency slot and serializes that
      /// shared-memory work. Capture the coarse DB handle as task state
      /// instead; physical owner-strip plans still use normal DB deps.
      parallelArgToAcquire.push_back({parallelArg, rootPtrValue});
      ARTS_DEBUG("    - Sharing coarse in-place DB state for dep " << idx);
      continue;
    }

    DbAcquireOp chunkAcqOp;
    bool chunkUsesStencilHalo = false;

    if (readOnlyOwnerMismatchedDep) {
      /// This loop's owner IV indexes a non-owner dimension of the read-only
      /// DB. The task writes are owner-local through another dependency, but
      /// this input must remain the full parent read window because the backing
      /// DB is physically partitioned along a different dimension.
      chunkAcqOp = AC->create<DbAcquireOp>(
          loc, ArtsMode::in, rootGuidValue, rootPtrValue,
          parentAcqOp.getPtr().getType(),
          parentAcqOp.getPartitionMode().value_or(PartitionMode::coarse),
          SmallVector<Value>(parentAcqOp.getIndices().begin(),
                             parentAcqOp.getIndices().end()),
          SmallVector<Value>(parentAcqOp.getOffsets().begin(),
                             parentAcqOp.getOffsets().end()),
          SmallVector<Value>(parentAcqOp.getSizes().begin(),
                             parentAcqOp.getSizes().end()),
          SmallVector<Value>(parentAcqOp.getPartitionIndices().begin(),
                             parentAcqOp.getPartitionIndices().end()),
          SmallVector<Value>(parentAcqOp.getPartitionOffsets().begin(),
                             parentAcqOp.getPartitionOffsets().end()),
          SmallVector<Value>(parentAcqOp.getPartitionSizes().begin(),
                             parentAcqOp.getPartitionSizes().end()),
          parentAcqOp.getBoundsValid(),
          SmallVector<Value>(parentAcqOp.getElementOffsets().begin(),
                             parentAcqOp.getElementOffsets().end()),
          SmallVector<Value>(parentAcqOp.getElementSizes().begin(),
                             parentAcqOp.getElementSizes().end()));
      transferContract(parentAcqOp.getOperation(), chunkAcqOp.getOperation(),
                       parentAcqOp.getPtr(), chunkAcqOp.getPtr(),
                       AC->getBuilder(), loc);
      chunkAcqOp.copyPartitionSegmentsFrom(parentAcqOp);
      if (parentAcqOp.getPreserveAccessMode())
        chunkAcqOp.setPreserveAccessMode();
      chunkAcqOp.setPreserveDepEdge();

      Value acquirePtr = chunkAcqOp.getResult(1);
      taskDeps.push_back(acquirePtr);
      parallelArgToAcquire.push_back({parallelArg, acquirePtr});
      ARTS_DEBUG("    - Preserved full read window for owner-mismatched dep "
                 << idx);
      continue;
    }

    std::optional<EdtDistributionPattern> distributionPattern =
        loopInfo.distributionContract.getEffectiveDistributionPattern();

    /// Resolve the effective contract by combining acquire + alloc + loop
    /// contracts.
    LoweringContractInfo contract;
    if (auto acquireContract = resolveAcquireContract(parentAcqOp))
      contract = *acquireContract;
    if (!forceCoarseReadOnlyDep)
      if (loopSemanticContract)
        combineContracts(contract, *loopSemanticContract);

    std::optional<unsigned> inferredMappedDim = accessMappedDim;
    /// Plan-driven path: use plan ownerDims when present.
    if (!forceCoarseReadOnlyDep && contract.spatial.ownerDims.empty()) {
      if (auto planOwnerDims =
              readI64ArrayAttr(getPlanOwnerDimsAttr(forOp.getOperation()))) {
        contract.spatial.ownerDims = *planOwnerDims;
        if (!planOwnerDims->empty() && (*planOwnerDims)[0] >= 0)
          inferredMappedDim = static_cast<unsigned>((*planOwnerDims)[0]);
      }
    }
    /// Fallback: infer from DB analysis.
    if (!forceCoarseReadOnlyDep && contract.spatial.ownerDims.empty())
      if (inferredMappedDim) {
        contract.spatial.ownerDims.push_back(
            static_cast<int64_t>(*inferredMappedDim));
      }

    std::optional<unsigned> ownerDimForHalo;
    if (contract.spatial.ownerDims.size() == 1 &&
        contract.spatial.ownerDims[0] >= 0)
      ownerDimForHalo = static_cast<unsigned>(contract.spatial.ownerDims[0]);
    else if (inferredMappedDim)
      ownerDimForHalo = *inferredMappedDim;

    // If proof.halo_legality is proven, the contract's halo shape is
    // already valid -- skip re-inferring from access patterns.
    bool proofTrustedHalo = false;
    if (LoweringContractOp contractOp =
            getLoweringContractOp(parentAcqOp.getSourcePtr())) {
      OwnershipProof proof = readOwnershipProof(contractOp.getOperation());
      proofTrustedHalo = proof.haloLegality;
    }

    if (!proofTrustedHalo && effectiveTaskMode == ArtsMode::in &&
        ownerDimForHalo && contract.spatial.ownerDims.size() <= 1 &&
        !contract.getStaticMinOffsets() && !contract.getStaticMaxOffsets()) {
      /// Prefer the persisted acquire/loop contract. When it is still
      /// incomplete, infer a conservative loop-local halo directly from IR
      /// uses instead of consulting graph-only partition facts here.
      if (auto bounds = inferLoopHaloBoundsFromValue(parentAcqOp.getPtr(),
                                                     forOp, *ownerDimForHalo)) {
        patchContract(contract, {}, {bounds->minOffset}, {bounds->maxOffset});
      }
    }

    TaskAcquireRewritePlanInput planningInput{
        AC, loc, parentAcqOp, effectiveTaskMode, rootGuidValue, rootPtrValue,
        /*forceCoarseRewrite=*/
        forceCoarseReadOnlyDep ||
            (singleDispatchLane && !parentHasPartitionInfo),
        loopInfo.strategy.kind, distributionPattern, tiling2DGrid, contract,
        acquireOffsetVal, acquireSizeVal, acquireHintSizeVal, stepVal,
        stepIsUnit};

    ARTS_DEBUG("    - Planned contract for dep "
               << depIndex << ": applyStencilHalo="
               << shouldApplyStencilHalo(contract, parentAcqOp)
               << ", preserveParentDepRange="
               << shouldPreserveParentDepRange(contract, parentAcqOp)
               << ", parentAcquire=" << parentAcqOp);

    auto createPlannedAcquire =
        [&]() -> std::pair<DbAcquireOp, TaskAcquireRewritePlan> {
      TaskAcquireRewritePlan rewritePlan =
          planTaskAcquireRewrite(planningInput);
      DbAcquireOp rewritten = rewriteAcquire(rewritePlan.rewriteInput,
                                             rewritePlan.useStencilRewriter);
      ARTS_DEBUG("    - Planned task acquire for dep "
                 << depIndex
                 << ": useStencilRewriter=" << rewritePlan.useStencilRewriter
                 << ", acquire=" << rewritten);
      return {rewritten, std::move(rewritePlan)};
    };
    std::optional<SmallVector<int64_t, 4>> plannedTaskBlockShape;

    if (parentHasPartitionInfo) {
      /// USER HINT EXISTS - ForLowering RESPECTS it!
      /// Use parent's partition mode and operands (from DbControlOp via
      /// CreateDbs)
      ARTS_DEBUG("    - Respecting existing DbControlOp hint on allocation");

      /// Reuse the parent acquire's partitioning clause which came from
      /// CreateDbs
      auto parentPartMode = parentAcqOp.getPartitionMode();
      SmallVector<Value> parentIndices(parentAcqOp.getIndices().begin(),
                                       parentAcqOp.getIndices().end());
      SmallVector<Value> parentOffsets(parentAcqOp.getOffsets().begin(),
                                       parentAcqOp.getOffsets().end());
      SmallVector<Value> parentSizes(parentAcqOp.getSizes().begin(),
                                     parentAcqOp.getSizes().end());

      SmallVector<Value> parentPartIndices(
          parentAcqOp.getPartitionIndices().begin(),
          parentAcqOp.getPartitionIndices().end());
      SmallVector<Value> parentPartOffsets(
          parentAcqOp.getPartitionOffsets().begin(),
          parentAcqOp.getPartitionOffsets().end());
      SmallVector<Value> parentPartSizes(
          parentAcqOp.getPartitionSizes().begin(),
          parentAcqOp.getPartitionSizes().end());
      PartitionMode mode = parentPartMode.value_or(PartitionMode::coarse);
      bool hasExplicitRangeHints =
          !parentOffsets.empty() && !parentSizes.empty();

      /// Hinted ranges are usually element-space and still need DB-space slice
      /// planning for rec_dep/depCount correctness.
      if (usesBlockLayout(mode) || hasExplicitRangeHints) {
        auto [plannedAcquire, rewritePlan] = createPlannedAcquire();
        chunkAcqOp = plannedAcquire;
        chunkUsesStencilHalo = rewritePlan.useStencilRewriter;
        plannedTaskBlockShape = rewritePlan.refinedTaskBlockShape;
        if (parentPartMode)
          setPartitionMode(chunkAcqOp.getOperation(), mode);
        if (chunkAcqOp.getIndices().empty() && !parentIndices.empty())
          chunkAcqOp.getIndicesMutable().assign(parentIndices);
        if (chunkAcqOp.getPartitionIndices().empty() &&
            !parentPartIndices.empty())
          chunkAcqOp.getPartitionIndicesMutable().assign(parentPartIndices);
        if (chunkAcqOp.getPartitionOffsets().empty() &&
            !parentPartOffsets.empty())
          chunkAcqOp.getPartitionOffsetsMutable().assign(parentPartOffsets);
        if (chunkAcqOp.getPartitionSizes().empty() && !parentPartSizes.empty())
          chunkAcqOp.getPartitionSizesMutable().assign(parentPartSizes);
        chunkAcqOp.copyPartitionSegmentsFrom(parentAcqOp);
      } else {
        chunkAcqOp = AC->create<DbAcquireOp>(
            loc, parentAcqOp.getMode(), rootGuidValue, rootPtrValue,
            parentAcqOp.getPtr().getType(), mode, parentIndices, parentOffsets,
            parentSizes, parentPartIndices, parentPartOffsets, parentPartSizes);
        transferContract(parentAcqOp.getOperation(), chunkAcqOp.getOperation(),
                         parentAcqOp.getPtr(), chunkAcqOp.getPtr(),
                         AC->getBuilder(), loc);
        chunkAcqOp.copyPartitionSegmentsFrom(parentAcqOp);
      }

    } else {
      /// NO USER HINT - plan strategy-specific acquire rewriting externally.
      /// When the effective dispatch collapses to a single worker, keep the
      /// compiler-generated task acquire coarse. This preserves the original
      /// whole-DB contract instead of manufacturing a one-lane block slice
      /// that later passes may misinterpret as a real partitioning decision.
      auto [plannedAcquire, rewritePlan] = createPlannedAcquire();
      chunkAcqOp = plannedAcquire;
      chunkUsesStencilHalo = rewritePlan.useStencilRewriter;
      plannedTaskBlockShape = rewritePlan.refinedTaskBlockShape;
    }

    applyTaskAcquireContractMetadata(
        forceCoarseReadOnlyDep ? nullptr : forOp.getOperation(), chunkAcqOp,
        planningInput.contract, plannedTaskBlockShape, AC->getBuilder(), loc);
    if (plannedTaskBlockShape)
      refinedTaskBlockShape = *plannedTaskBlockShape;

    applyTaskAcquireSlicePlan(TaskAcquireSlicePlanInput{
        AC, loc, parentAcqOp, chunkAcqOp, effectiveTaskMode, rootGuidValue,
        rootPtrValue, loopInfo.strategy.kind, distributionPattern,
        acquireOffsetVal, loopInfo.bounds.iterCount, acquireHintSizeVal,
        chunkUsesStencilHalo, planningInput.contract});

    if (originalParallel.getConcurrency() == EdtConcurrency::internode &&
        chunkAcqOp.getMode() != ArtsMode::in &&
        DbAnalysis::isCoarse(chunkAcqOp)) {
      auto allocOp = dyn_cast_or_null<DbAllocOp>(
          DbUtils::getUnderlyingDbAlloc(chunkAcqOp.getSourcePtr()));
      if (allocOp && !hasDistributedDbAllocation(allocOp.getOperation())) {
        forceOwnerRoute = true;
        ARTS_DEBUG("    - Coarse write dependency requires owner-local route");
      }
    }

    Value acquirePtr = chunkAcqOp.getResult(1);

    taskDeps.push_back(acquirePtr);
    parallelArgToAcquire.push_back({parallelArg, acquirePtr});
    ARTS_DEBUG("    - Created rewired acquire for dep " << idx);
  }

  /// When the effective dispatch collapses to one non-reduction lane, keep
  /// all distribution/acquire planning but execute the lowered loop body
  /// directly in the dispatch path instead of outlining a task EDT. This
  /// avoids per-iteration task/epoch overhead while preserving the same
  /// partitioning and contract decisions.
  if (singleDispatchLane && redInfo.reductionVars.empty()) {
    ++numInlineSingleLaneLowerings;
    Block &forBody = forOp.getRegion().front();
    Region *directRegion = &ifOp.getThenRegion();
    Block &directBlock = directRegion->front();

    IRMapping directMapper;
    for (auto [parallelArg, acquirePtr] : parallelArgToAcquire)
      directMapper.map(parallelArg, acquirePtr);

    Value insideTotalWorkers =
        WorkDistributionUtils::getTotalWorkers(AC, loc, originalParallel);
    taskLoopInput.totalWorkers = insideTotalWorkers;

    DenseSet<Operation *> opsToSkip;
    SetVector<Value> externalValues;
    collectExternalValues(forBody, directRegion, externalValues, opsToSkip);

    Value origStep = forOp.getStep()[0];
    Value origLowerBound = forOp.getLowerBound()[0];
    Value origUpperBound = forOp.getUpperBound()[0];
    Value chunkLowerBoundVal =
        loopInfo.chunkLowerBound ? loopInfo.chunkLowerBound : origLowerBound;
    Region *forBodyRegion = forBody.getParent();
    std::function<void(Value)> collectWithDeps = [&](Value val) {
      if (externalValues.contains(val))
        return;
      if (Operation *defOp = val.getDefiningOp()) {
        if (forBodyRegion->isAncestor(defOp->getParentRegion()))
          return;
        if (!directRegion->isAncestor(defOp->getParentRegion())) {
          for (Value operand : defOp->getOperands())
            collectWithDeps(operand);
          externalValues.insert(val);
        }
      }
    };

    collectWithDeps(origStep);
    collectWithDeps(origLowerBound);
    collectWithDeps(origUpperBound);
    collectWithDeps(chunkLowerBoundVal);
    collectWithDeps(workerOffsetVal);
    taskLoopLowering->collectExtraExternalValues(taskLoopInput, externalValues);

    SmallVector<Value> valuesToProcess(externalValues.begin(),
                                       externalValues.end());
    for (Value val : valuesToProcess)
      if (Operation *defOp = val.getDefiningOp())
        for (Value operand : defOp->getOperands())
          collectWithDeps(operand);

    auto extraCloneableOps = [](Operation *op) {
      return isa<arts::DbRefOp, memref::LoadOp>(op);
    };
    if (!ValueAnalysis::cloneValuesIntoRegion(
            externalValues, directRegion, directMapper, AC->getBuilder(),
            /*allowMemoryEffectFree=*/true, extraCloneableOps)) {
      for (Value external : externalValues) {
        if (directMapper.contains(external))
          continue;
        if (Operation *defOp = external.getDefiningOp())
          ARTS_DEBUG("  - Uncloned external value op: " << defOp->getName());
        else
          ARTS_DEBUG("  - Uncloned external value (no defining op)");
      }
      ARTS_WARN("Some external values could not be cloned into the single-lane "
                "inline lowering");
    }

    Value stepValMapped = directMapper.lookupOrDefault(origStep);
    Value origLowerBoundVal = directMapper.lookupOrDefault(origLowerBound);
    Value origUpperBoundVal = directMapper.lookupOrDefault(origUpperBound);
    TaskLoopLoweringMappedValues mappedLoopValues;
    mappedLoopValues.step = stepValMapped;
    mappedLoopValues.lowerBound = origLowerBoundVal;
    mappedLoopValues.upperBound = origUpperBoundVal;
    mappedLoopValues.workerBaseOffset =
        directMapper.lookupOrDefault(workerOffsetVal);
    mappedLoopValues.blockSize =
        directMapper.lookupOrDefault(loopInfo.blockSize);
    mappedLoopValues.totalIterations =
        directMapper.lookupOrDefault(loopInfo.totalIterations);
    mappedLoopValues.totalChunks =
        directMapper.lookupOrDefault(loopInfo.totalChunks);

    TaskLoopLoweringResult loweredLoop =
        taskLoopLowering->lower(taskLoopInput, mappedLoopValues);
    loopInfo.insideBounds = loweredLoop.insideBounds;
    scf::ForOp iterLoop = loweredLoop.iterLoop;

    AC->setInsertionPointToStart(iterLoop.getBody());

    Value localIter = iterLoop.getInductionVar();
    Value globalIterScaled =
        AC->create<arith::MulIOp>(loc, localIter, stepValMapped);
    Value globalIdx = AC->create<arith::AddIOp>(loc, loweredLoop.globalBase,
                                                globalIterScaled);

    for (Operation &op : forBody.without_terminator()) {
      for (Value operand : op.getOperands()) {
        if (auto undef = operand.getDefiningOp<LLVM::UndefOp>()) {
          Value undefVal = undef.getResult();
          Type elemType = undefVal.getType();
          Value identity = AC->createZeroValue(elemType, loc);
          if (identity)
            directMapper.map(undefVal, identity);
        }
      }
    }

    if (forBody.getNumArguments() > 0)
      directMapper.map(forBody.getArgument(0), globalIdx);

    for (Operation &op : forBody.without_terminator()) {
      if (opsToSkip.contains(&op))
        continue;
      if (isa<memref::AllocaOp>(op))
        AC->clone(op, directMapper);
    }

    for (Operation &op : forBody.without_terminator()) {
      if (opsToSkip.contains(&op))
        continue;
      if (isa<memref::AllocaOp>(op))
        continue;
      AC->clone(op, directMapper);
    }

    TaskLoopPostCloneInput postCloneInput{AC,
                                          loc,
                                          iterLoop,
                                          globalIdx,
                                          loweredLoop.innerStripeLane,
                                          loweredLoop.innerStripeCount};
    taskLoopLowering->postCloneAdjust(postCloneInput);
    Operation *loopAnchor = iterLoop.getOperation();

    cloneExternalAllocasIntoEdt(directRegion, directBlock, directMapper,
                                AC->getBuilder());
    if (Operation *sunkLoop = sinkTaskLoopToContiguousInnerDim(
            iterLoop, globalIdx, loopInfo.strategy.kind))
      loopAnchor = sunkLoop;

    AC->setInsertionPointAfter(loopAnchor);
    for (Value dep : taskDeps)
      AC->create<DbReleaseOp>(loc, dep);

    AC->setInsertionPointAfter(ifOp);
    return EdtOp();
  }

  /// Create task EDT
  /// Inherit concurrency from parent parallel EDT - if internode, route to
  /// worker node
  EdtConcurrency taskConcurrency = originalParallel.getConcurrency();
  Value routeValue;
  if (taskConcurrency == EdtConcurrency::internode) {
    if (forceOwnerRoute) {
      routeValue = createCurrentNodeRoute(AC->getBuilder(), loc);
      ARTS_DEBUG("  - Keeping coarse write task on the current owner node");
    } else {
      /// Route to destination node from the global worker id:
      ///   nodeId = globalWorkerId / workersPerNode
      /// workersPerNode is always materialized for internode routing.
      if (!workersPerNode)
        workersPerNode =
            WorkDistributionUtils::getWorkersPerNode(AC, loc, originalParallel);
      Value nodeId =
          AC->create<arith::DivUIOp>(loc, workerIdPlaceholder, workersPerNode);
      routeValue = AC->castToInt(AC->Int32, nodeId, loc);
      ARTS_DEBUG("  - Using internode routing by workers-per-node");
    }
  } else {
    routeValue = createCurrentNodeRoute(AC->getBuilder(), loc);
  }
  auto taskEdt = AC->create<EdtOp>(loc, EdtType::task, taskConcurrency,
                                   routeValue, ValueRange{});
  if (taskConcurrency == EdtConcurrency::intranode)
    taskEdt->setAttr(AttrNames::Operation::ReadyLocalLaunch,
                     AC->getBuilder().getUnitAttr());
  ++numTaskEdtsCreated;
  transferOperationContract(forOp.getOperation(), taskEdt.getOperation());
  copyPlanAttrs(forOp.getOperation(), taskEdt.getOperation());
  copyCoreExecutionHintAttrs(forOp.getOperation(), taskEdt.getOperation());
  if (refinedTaskBlockShape)
    setStencilBlockShape(taskEdt.getOperation(), *refinedTaskBlockShape);

  Block &taskBlock = taskEdt.getBody().front();
  AC->setInsertionPointToStart(&taskBlock);

  /// Track where reduction dependencies start
  uint64_t reductionArgStart = taskDeps.size();

  /// Combine regular and reduction dependencies
  taskDeps.append(reductionTaskDeps.begin(), reductionTaskDeps.end());

  /// Add block arguments to task EDT
  for (uint64_t i = 0; i < taskDeps.size(); i++) {
    Value dep = taskDeps[i];
    if (!dep) {
      ARTS_ERROR("Null dependency at index " << i);
      continue;
    }
    BlockArgument taskArg = taskBlock.addArgument(dep.getType(), loc);
    if (i < reductionArgStart)
      mapper.map(dep, taskArg);
  }

  taskEdt.setDependencies(taskDeps);

  /// Map parallelArg → taskArg for cloning pre-for operations
  for (auto [parallelArg, acquirePtr] : parallelArgToAcquire) {
    if (Value taskArg = mapper.lookupOrNull(acquirePtr)) {
      mapper.map(parallelArg, taskArg);
      continue;
    }
    mapper.map(parallelArg, acquirePtr);
  }

  Value insideTotalWorkers =
      WorkDistributionUtils::getTotalWorkers(AC, loc, originalParallel);
  taskLoopInput.totalWorkers = insideTotalWorkers;

  /// Map reduction variables to worker's accumulator slot
  IRMapping redMapper = mapper;
  DenseSet<Operation *> opsToSkip;

  for (auto [redVar, idx] : reductionVarIndex) {
    uint64_t argIndex = reductionArgStart + idx;
    if (argIndex >= taskBlock.getNumArguments())
      continue;

    BlockArgument partialArg = taskBlock.getArgument(argIndex);
    auto zeroIndex = AC->createIndexConstant(0, loc);
    Value myAccumulator =
        AC->create<DbRefOp>(loc, partialArg, SmallVector<Value>{zeroIndex});
    redMapper.map(redVar, myAccumulator);

    collectOldAccumulatorDbRefs(forOp, parallelBlock, reductionBlockArgs,
                                opsToSkip, redMapper, myAccumulator);
  }

  Block &forBody = forOp.getRegion().front();
  Region *taskEdtRegion = taskBlock.getParent();

  /// Collect external values needed by the loop body and induction variable
  SetVector<Value> externalValues;
  collectExternalValues(forBody, taskEdtRegion, externalValues, opsToSkip);

  Value origStep = forOp.getStep()[0];
  Value origLowerBound = forOp.getLowerBound()[0];
  Value origUpperBound = forOp.getUpperBound()[0];
  Value chunkLowerBoundVal =
      loopInfo.chunkLowerBound ? loopInfo.chunkLowerBound : origLowerBound;

  Region *forBodyRegion = forBody.getParent();
  std::function<void(Value)> collectWithDeps = [&](Value val) {
    if (externalValues.contains(val))
      return;
    if (Operation *defOp = val.getDefiningOp()) {
      if (forBodyRegion->isAncestor(defOp->getParentRegion()))
        return;
      if (!taskEdtRegion->isAncestor(defOp->getParentRegion())) {
        for (Value operand : defOp->getOperands())
          collectWithDeps(operand);
        externalValues.insert(val);
      }
    }
  };

  collectWithDeps(origStep);
  collectWithDeps(origLowerBound);
  collectWithDeps(origUpperBound);
  collectWithDeps(chunkLowerBoundVal);
  collectWithDeps(workerOffsetVal);

  taskLoopLowering->collectExtraExternalValues(taskLoopInput, externalValues);

  /// Collect transitive dependencies
  SmallVector<Value> valuesToProcess(externalValues.begin(),
                                     externalValues.end());
  for (Value val : valuesToProcess) {
    if (Operation *defOp = val.getDefiningOp()) {
      for (Value operand : defOp->getOperands())
        collectWithDeps(operand);
    }
  }

  auto extraCloneableOps = [](Operation *op) {
    return isa<arts::DbRefOp, memref::LoadOp>(op);
  };
  if (!ValueAnalysis::cloneValuesIntoRegion(
          externalValues, taskEdtRegion, redMapper, AC->getBuilder(),
          /*allowMemoryEffectFree=*/true, extraCloneableOps)) {
    for (Value external : externalValues) {
      if (redMapper.contains(external))
        continue;
      if (Operation *defOp = external.getDefiningOp())
        ARTS_DEBUG("  - Uncloned external value op: " << defOp->getName());
      else
        ARTS_DEBUG("  - Uncloned external value (no defining op)");
    }
    ARTS_WARN("Some external values could not be cloned - they may need to be "
              "passed as EDT dependencies");
  }

  Value stepValMapped = redMapper.lookupOrDefault(origStep);
  Value origLowerBoundVal = redMapper.lookupOrDefault(origLowerBound);
  Value origUpperBoundVal = redMapper.lookupOrDefault(origUpperBound);
  TaskLoopLoweringMappedValues mappedLoopValues;
  mappedLoopValues.step = stepValMapped;
  mappedLoopValues.lowerBound = origLowerBoundVal;
  mappedLoopValues.upperBound = origUpperBoundVal;
  mappedLoopValues.workerBaseOffset =
      redMapper.lookupOrDefault(workerOffsetVal);
  mappedLoopValues.blockSize = redMapper.lookupOrDefault(loopInfo.blockSize);
  mappedLoopValues.totalIterations =
      redMapper.lookupOrDefault(loopInfo.totalIterations);
  mappedLoopValues.totalChunks =
      redMapper.lookupOrDefault(loopInfo.totalChunks);

  TaskLoopLoweringResult loweredLoop =
      taskLoopLowering->lower(taskLoopInput, mappedLoopValues);
  loopInfo.insideBounds = loweredLoop.insideBounds;
  chunkOffset = loweredLoop.iterStart;
  scf::ForOp iterLoop = loweredLoop.iterLoop;

  if (!reductionVarIndex.empty()) {
    int64_t memrefDeps = countMemrefsWithLoopCarriedDeps(forOp);
    int64_t numReductionVars = reductionVarIndex.size();

    /// Check if ALL loop-carried dependencies are from reduction variables.
    /// If memrefDeps == numReductionVars, all deps are reductions, which is
    /// safe to parallelize because reduction dependencies are handled via
    /// worker-local accumulators and a final reduction phase, NOT via
    /// cross-iteration data flow. If memrefDeps < numReductionVars, some
    /// reductions have no deps (write-only), which is also safe. If
    /// memrefDeps > numReductionVars, there are non-reduction dependencies
    /// (likely stencil patterns), which we must keep sequential.
    if (memrefDeps <= numReductionVars) {
      ARTS_DEBUG("  Reduction-only deps: "
                 << memrefDeps << " memrefs with deps, " << numReductionVars
                 << " reduction vars, no stencil patterns");
    } else {
      ARTS_DEBUG("  memrefsWithLoopCarriedDeps="
                 << memrefDeps << " > reduction vars=" << numReductionVars
                 << " (stencil patterns detected)");
    }
  }

  AC->setInsertionPointToStart(iterLoop.getBody());

  /// Map induction variable
  Value localIter = iterLoop.getInductionVar();
  Value globalIterScaled =
      AC->create<arith::MulIOp>(loc, localIter, stepValMapped);
  Value globalIdx =
      AC->create<arith::AddIOp>(loc, loweredLoop.globalBase, globalIterScaled);

  /// Map undef to identity value
  for (Operation &op : forBody.without_terminator()) {
    for (Value operand : op.getOperands()) {
      if (auto undef = operand.getDefiningOp<LLVM::UndefOp>()) {
        Value undefVal = undef.getResult();
        Type elemType = undefVal.getType();
        Value identity = AC->createZeroValue(elemType, loc);
        if (identity)
          redMapper.map(undefVal, identity);
      }
    }
  }

  if (forBody.getNumArguments() > 0)
    redMapper.map(forBody.getArgument(0), globalIdx);

  /// No per-iteration bounds check needed: loop bounds already clamped.

  /// Clone stack allocations first so subsequent ops can map to them.
  for (Operation &op : forBody.without_terminator()) {
    if (opsToSkip.contains(&op))
      continue;
    if (isa<memref::AllocaOp>(op))
      AC->clone(op, redMapper);
  }

  /// Clone remaining loop body operations.
  for (Operation &op : forBody.without_terminator()) {
    if (opsToSkip.contains(&op))
      continue;
    if (isa<memref::AllocaOp>(op))
      continue;
    AC->clone(op, redMapper);
  }

  TaskLoopPostCloneInput postCloneInput{AC,
                                        loc,
                                        iterLoop,
                                        globalIdx,
                                        loweredLoop.innerStripeLane,
                                        loweredLoop.innerStripeCount};
  taskLoopLowering->postCloneAdjust(postCloneInput);

  /// Ensure stack allocations used inside the EDT are cloned locally.
  cloneExternalAllocasIntoEdt(taskEdtRegion, taskBlock, redMapper,
                              AC->getBuilder());
  if (reductionVarIndex.empty())
    sinkTaskLoopToContiguousInnerDim(iterLoop, globalIdx,
                                     loopInfo.strategy.kind);

  if (!reductionVarIndex.empty()) {
    auto scalarIndices = [&](Value memref) {
      SmallVector<Value> indices;
      auto memrefType = dyn_cast<MemRefType>(memref.getType());
      if (!memrefType || memrefType.getRank() == 0)
        return indices;
      indices.push_back(AC->createIndexConstant(0, loc));
      return indices;
    };

    auto loadScalar = [&](Value memref) {
      SmallVector<Value> indices = scalarIndices(memref);
      return AC->create<memref::LoadOp>(loc, memref, indices);
    };

    auto storeScalar = [&](Value value, Value memref) {
      SmallVector<Value> indices = scalarIndices(memref);
      AC->create<memref::StoreOp>(loc, value, memref, indices);
    };

    for (auto [redVar, idx] : reductionVarIndex) {
      if (idx >= redInfo.privateReductionAccums.size())
        continue;

      Value privateAccum = redInfo.privateReductionAccums[idx];
      if (!privateAccum) {
        ARTS_DEBUG("No private reduction carrier for " << redVar
                                                       << "; body writes the "
                                                          "worker accumulator "
                                                          "directly");
        continue;
      }
      privateAccum = redMapper.lookupOrDefault(privateAccum);
      auto privateType = dyn_cast<MemRefType>(privateAccum.getType());
      if (!privateType ||
          !taskEdtRegion->isAncestor(privateAccum.getParentRegion())) {
        ARTS_WARN("Skipping private reduction carrier rewrite for " << redVar);
        continue;
      }

      AC->setInsertionPoint(iterLoop);
      Type elemType = privateType.getElementType();
      Value identity = AC->createZeroValue(elemType, loc);
      if (!identity) {
        ARTS_WARN("Skipping reduction accumulator initialization for "
                  << redVar);
        continue;
      }

      storeScalar(identity, privateAccum);

      uint64_t argIndex = reductionArgStart + idx;
      if (argIndex >= taskBlock.getNumArguments())
        continue;

      BlockArgument partialArg = taskBlock.getArgument(argIndex);
      Value workerSlot = AC->create<DbRefOp>(
          loc, partialArg, SmallVector<Value>{AC->createIndexConstant(0, loc)});

      AC->setInsertionPointToEnd(&taskBlock);
      Value privateValue = loadScalar(privateAccum);
      storeScalar(privateValue, workerSlot);
    }
  }

  /// Add yield terminator
  AC->setInsertionPointToEnd(&taskBlock);
  if (taskBlock.empty() || !taskBlock.back().hasTrait<OpTrait::IsTerminator>())
    AC->create<YieldOp>(loc);

  /// Release all DB dependencies before terminator
  AC->setInsertionPoint(taskBlock.getTerminator());
  for (uint64_t i = 0; i < taskDeps.size(); i++) {
    BlockArgument dbPtrArg = taskBlock.getArgument(i);
    AC->create<DbReleaseOp>(loc, dbPtrArg);
  }

  AC->setInsertionPointAfter(ifOp);

  return taskEdt;
}

///===----------------------------------------------------------------------===///
/// Pass creation
///===----------------------------------------------------------------------===///
namespace mlir {
namespace arts {
std::unique_ptr<Pass> createForLoweringPass() {
  return std::make_unique<ForLoweringPass>();
}

std::unique_ptr<Pass> createForLoweringPass(mlir::arts::AnalysisManager *AM) {
  return std::make_unique<ForLoweringPass>(AM);
}
} // namespace arts
} // namespace mlir
