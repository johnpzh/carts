///==========================================================================///
/// File: DbAnalysis.cpp
/// Implementation of the DbAnalysis class for DB operation analysis.
///==========================================================================///

#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/AccessPatternAnalysis.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAliasAnalysis.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbGraph.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/EdtUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/dialect/arts/Utils/PartitionPredicates.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_analysis);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

static bool canNarrowDirectReadSlice(const LoweringContractInfo &contract,
                                     const DbAcquirePartitionFacts &facts) {
  auto acquire = facts.acquire;
  if (!acquire || acquire.getMode() != ArtsMode::in)
    return false;

  /// Scope-wide peer consensus is useful for allocation-level partitioning, but
  /// it does not prove that this specific acquire's read footprint is owner-
  /// local. Narrowing a direct read slice on top of peer-inferred dims is
  /// unsound for reduction-style consumers such as B[k][j] in row-owned matmul
  /// tasks: the acquire is direct, yet it still needs the full reduction range.
  if (facts.partitionDimsFromPeers)
    return false;

  bool blockLikeRequest = usesBlockLayout(facts.requestedMode);
  bool hasCanonicalOwnerLayout = !contract.spatial.ownerDims.empty();

  return blockLikeRequest && facts.hasDirectAccess &&
         !facts.hasIndirectAccess && !facts.hasUnmappedPartitionEntry() &&
         hasCanonicalOwnerLayout;
}

static bool refineContractWithFacts(LoweringContractInfo &contract,
                                    const DbAcquirePartitionFacts &facts) {
  bool changed = false;
  const bool hasSeededContract = !contract.empty();
  const bool seededStencilSemantics = contract.hasExplicitStencilContract();

  /// Post-DB analysis should refine the same contract language, not invent a
  /// new semantic family after the pre-DB pattern pipeline failed to match.
  if (hasSeededContract && !contract.pattern.depPattern &&
      facts.depPattern != ArtsDepPattern::unknown) {
    contract.pattern.depPattern = facts.depPattern;
    changed = true;
  }

  if (!contract.pattern.distributionPattern && contract.pattern.depPattern) {
    if (auto derived =
            getDistributionPatternForDepPattern(*contract.pattern.depPattern)) {
      contract.pattern.distributionPattern = *derived;
      changed = true;
    }
  }

  if (seededStencilSemantics) {
    ArrayRef<unsigned> dims = facts.stencilOwnerDims;
    if (dims.empty())
      dims = facts.partitionDims;
    if (!dims.empty()) {
      SmallVector<int64_t, 4> refinedOwnerDims;
      refinedOwnerDims.reserve(dims.size());
      for (unsigned dim : dims)
        refinedOwnerDims.push_back(static_cast<int64_t>(dim));
      if (contract.spatial.ownerDims != refinedOwnerDims &&
          (contract.spatial.ownerDims.empty() ||
           refinedOwnerDims.size() >= contract.spatial.ownerDims.size())) {
        contract.spatial.ownerDims = std::move(refinedOwnerDims);
        changed = true;
      }
    }
  }

  if (seededStencilSemantics && !contract.spatial.supportedBlockHalo &&
      facts.supportedBlockHalo) {
    contract.spatial.supportedBlockHalo = true;
    changed = true;
  }

  if (seededStencilSemantics && !contract.pattern.distributionPattern &&
      facts.accessPattern == AccessPattern::Stencil) {
    contract.pattern.distributionPattern = EdtDistributionPattern::stencil;
    changed = true;
    if (!contract.pattern.depPattern)
      contract.pattern.depPattern = ArtsDepPattern::stencil;
  }

  /// Direct read-only block/stencil accesses with a resolved owner dimension
  /// can safely keep a worker-local dependency slice. Preserve that fact on
  /// the contract so generic acquire rewriting can narrow the dependency
  /// window without pattern-specific branches.
  if (canNarrowDirectReadSlice(contract, facts) &&
      !contract.analysis.narrowableDep) {
    contract.analysis.narrowableDep = true;
    changed = true;
  }

  if (changed)
    contract.analysis.postDbRefined = true;

  return changed;
}

static bool refineContractWithStencilBounds(LoweringContractInfo &contract,
                                            DbAcquireNode *acqNode) {
  if (!acqNode || !contract.hasExplicitStencilContract() ||
      contract.spatial.ownerDims.size() != 1)
    return false;

  auto minOffsets = contract.getStaticMinOffsets();
  auto maxOffsets = contract.getStaticMaxOffsets();
  if (minOffsets && maxOffsets)
    return false;

  auto bounds = acqNode->getStencilBounds();
  if (!bounds || !bounds->valid || !bounds->hasHalo())
    return false;

  contract.spatial.minOffsets.clear();
  contract.spatial.maxOffsets.clear();
  contract.spatial.staticMinOffsets = {bounds->minOffset};
  contract.spatial.staticMaxOffsets = {bounds->maxOffset};
  contract.analysis.postDbRefined = true;
  return true;
}

static LoweringContractInfo buildAcquireContractSeed(DbAcquireOp acquire) {
  LoweringContractInfo contract;
  if (!acquire)
    return contract;
  if (auto semanticContract = getSemanticContract(acquire.getOperation())) {
    contract = *semanticContract;
  }
  if (auto explicitContract = getLoweringContract(acquire.getPtr())) {
    if (contract.empty())
      contract = *explicitContract;
    else
      mergeLoweringContractInfo(contract, *explicitContract);
  }
  return contract;
}

static SmallVector<unsigned, 4>
resolveContractPartitionDims(const LoweringContractInfo &contract) {
  if (contract.spatial.ownerDims.empty())
    return {};

  unsigned rank = 0;
  for (int64_t dim : contract.spatial.ownerDims) {
    if (dim >= 0)
      rank = std::max<unsigned>(rank, static_cast<unsigned>(dim) + 1);
  }
  if (rank == 0)
    rank = static_cast<unsigned>(contract.spatial.ownerDims.size());
  return resolveContractOwnerDims(contract, rank);
}

static AccessPattern
resolveAcquireAccessPattern(const DbAnalysis::AcquireContractSummary &summary) {
  switch (summary.contract.getEffectiveKind()) {
  case ContractKind::Stencil:
    return AccessPattern::Stencil;
  case ContractKind::Elementwise:
  case ContractKind::Matmul:
    return AccessPattern::Uniform;
  case ContractKind::Triangular:
  case ContractKind::Unknown:
    break;
  }

  if (auto distributionPattern =
          summary.contract.getEffectiveDistributionPattern()) {
    switch (*distributionPattern) {
    case EdtDistributionPattern::stencil:
      return AccessPattern::Stencil;
    case EdtDistributionPattern::uniform:
    case EdtDistributionPattern::matmul:
      return AccessPattern::Uniform;
    case EdtDistributionPattern::triangular:
    case EdtDistributionPattern::unknown:
      break;
    }
  }

  if (summary.accessPattern != AccessPattern::Unknown)
    return summary.accessPattern;
  return AccessPattern::Unknown;
}

static void populateSummaryFromCanonicalContract(
    DbAnalysis::AcquireContractSummary &summary, DbAcquireOp acquire) {
  summary.distributionContract = summary.distributionContract ||
                                 summary.contract.hasDistributionContract();

  if (summary.partitionDims.empty()) {
    SmallVector<unsigned, 4> contractDims =
        resolveContractPartitionDims(summary.contract);
    summary.partitionDims.assign(contractDims.begin(), contractDims.end());
  }

  summary.blockHints = summary.blockHints ||
                       !acquire.getPartitionOffsets().empty() ||
                       !acquire.getPartitionSizes().empty();
  summary.fineGrainedEntries = summary.fineGrainedEntries ||
                               acquire.hasAllFineGrainedEntries() ||
                               !acquire.getPartitionIndices().empty();
  summary.accessPattern = resolveAcquireAccessPattern(summary);
}

static void
mergeDerivedFactEvidence(DbAnalysis::AcquireContractSummary &summary,
                         const DbAcquirePartitionFacts &facts) {
  summary.derivedFactEvidence = true;
  summary.indirectAccess = facts.hasIndirectAccess;
  summary.directAccess = facts.hasDirectAccess;
  summary.blockHints = summary.blockHints || facts.hasBlockHints;
  summary.inferredBlockCapability =
      summary.inferredBlockCapability || facts.inferredBlock;
  summary.fineGrainedEntries =
      summary.fineGrainedEntries || facts.hasFineGrainedEntries();
  summary.unmappedPartitionEntry =
      summary.unmappedPartitionEntry || facts.hasUnmappedPartitionEntry();
  summary.distributedContractEntry =
      summary.distributedContractEntry || facts.hasDistributedContractEntries();
  summary.partitionDimsFromPeersFlag =
      summary.partitionDimsFromPeersFlag ||
      (summary.partitionDims.empty() && facts.partitionDimsFromPeers);

  if (summary.partitionDims.empty()) {
    if (!facts.partitionDims.empty()) {
      summary.partitionDims.assign(facts.partitionDims.begin(),
                                   facts.partitionDims.end());
    } else if (auto inferredDim = facts.inferSingleMappedDim()) {
      summary.partitionDims.push_back(*inferredDim);
    }
  }

  summary.derivedAccessPattern = facts.accessPattern;
}

} // namespace

bool DbAnalysis::isTiling2DTaskAcquire(DbAcquireOp acquire) {
  if (!acquire)
    return false;
  if (auto contract = getLoweringContract(acquire.getPtr())) {
    if (contract->pattern.distributionKind &&
        *contract->pattern.distributionKind == EdtDistributionKind::tiling_2d)
      return true;
  }
  if (auto kind = getEdtDistributionKind(acquire.getOperation()))
    return *kind == EdtDistributionKind::tiling_2d;
  auto [edt, blockArg] = EdtUtils::getBlockArgumentForAcquire(acquire);
  (void)blockArg;
  if (!edt)
    return false;
  auto kind = getEdtDistributionKind(edt.getOperation());
  return kind && *kind == EdtDistributionKind::tiling_2d;
}

DbAnalysis::DbAnalysis(AnalysisManager &AM) : ArtsAnalysis(AM) {
  ARTS_DEBUG("Initializing DbAnalysis");
  dbAliasAnalysis = std::make_unique<DbAliasAnalysis>();
}

DbAnalysis::~DbAnalysis() { ARTS_DEBUG("Destroying DbAnalysis"); }

DbGraph &DbAnalysis::getOrCreateGraph(func::FuncOp func) {
  ARTS_INFO("Getting or creating DbGraph for function: " << func.getName());
  // Fast path: shared lock for read.
  {
    std::shared_lock<std::shared_mutex> readLock(graphMutex);
    auto it = functionGraphMap.find(func);
    if (it != functionGraphMap.end())
      return *it->second;
  }
  // Slow path: exclusive lock for write.
  std::unique_lock<std::shared_mutex> writeLock(graphMutex);
  // Re-check after acquiring write lock (another thread may have created it).
  auto &graph = functionGraphMap[func];
  if (graph)
    return *graph;

  ARTS_DEBUG(" - Creating new DbGraph for function: " << func.getName());
  graph = std::make_unique<DbGraph>(func, this);
  graph->build();
  return *graph;
}

bool DbAnalysis::invalidateGraph(func::FuncOp func) {
  ARTS_INFO("Invalidating DbGraph for function: " << func.getName());
  std::unique_lock<std::shared_mutex> writeLock(graphMutex);
  auto it = functionGraphMap.find(func);
  if (it != functionGraphMap.end()) {
    functionGraphMap.erase(it);
    return true;
  }
  return false;
}

void DbAnalysis::invalidate() {
  ARTS_INFO("Invalidating all DbGraphs");
  std::unique_lock<std::shared_mutex> writeLock(graphMutex);
  functionGraphMap.clear();
}

LoopAnalysis *DbAnalysis::getLoopAnalysis() {
  return &getAnalysisManager().getLoopAnalysis();
}

/// Scan the acquire's block for MinUIOp/MinSIOp that refines the given
/// size hint based on an offset dependency.
/// TODO(PERF): refineSizeFromMinInBlock linearly scans the entire acquire
/// block for MinUIOp/MinSIOp. Limit search to forward from acquire op, or
/// cache results per block.
static Value refineSizeFromMinInBlock(DbAcquireOp acquire, Value offset,
                                      Value sizeHint) {
  if (!offset || !sizeHint)
    return Value();
  int64_t sizeConst = 0;
  bool sizeIsConst = ValueAnalysis::getConstantIndex(sizeHint, sizeConst);
  auto isSameConst = [&](Value v) -> bool {
    int64_t val = 0;
    return ValueAnalysis::getConstantIndex(v, val) && val == sizeConst;
  };
  for (Operation &op : *acquire->getBlock()) {
    Value refined;
    auto tryRefine = [&](Value lhs, Value rhs, Value result) {
      bool lhsIsHint = (lhs == sizeHint) || (sizeIsConst && isSameConst(lhs));
      bool rhsIsHint = (rhs == sizeHint) || (sizeIsConst && isSameConst(rhs));
      if (lhsIsHint && ValueAnalysis::dependsOn(rhs, offset))
        refined = result;
      else if (rhsIsHint && ValueAnalysis::dependsOn(lhs, offset))
        refined = result;
    };

    if (auto minOp = dyn_cast<arith::MinUIOp>(&op)) {
      tryRefine(minOp.getLhs(), minOp.getRhs(), minOp.getResult());
    } else if (auto minOp = dyn_cast<arith::MinSIOp>(&op)) {
      tryRefine(minOp.getLhs(), minOp.getRhs(), minOp.getResult());
    }

    if (refined)
      return refined;
  }
  return Value();
}

/// Apply partition dimension information from the canonical acquire summary.
static void
inferPartitionDims(DbAnalysis::AcquirePartitionSummary &summary,
                   const DbAnalysis::AcquireContractSummary *contractSummary) {
  if (!contractSummary || contractSummary->partitionDims.empty())
    return;
  summary.partitionDims.assign(contractSummary->partitionDims.begin(),
                               contractSummary->partitionDims.end());
  if (!summary.partitionOffsets.empty() &&
      summary.partitionDims.size() > summary.partitionOffsets.size()) {
    summary.partitionDims.resize(summary.partitionOffsets.size());
  }
}

/// For tiling_2d acquires with no partition dims, assign sequential dims.
static void
applyTiling2DPartitionDimsFallback(DbAnalysis::AcquirePartitionSummary &summary,
                                   DbAcquireOp acquire) {
  if (!DbAnalysis::isTiling2DTaskAcquire(acquire))
    return;
  if (!summary.partitionDims.empty())
    return;
  if (summary.partitionOffsets.size() < 2)
    return;
  summary.partitionDims.clear();
  for (unsigned d = 0; d < summary.partitionOffsets.size(); ++d)
    summary.partitionDims.push_back(d);
}

/// Analyze partition info for a coarse-mode acquire.
static void analyzeCoarsePartition(
    DbAnalysis::AcquirePartitionSummary &info, DbAcquireOp acquire,
    DbAcquireNode *acqNode,
    const DbAnalysis::AcquireContractSummary *contractSummary) {
  info.isValid = true;
  if (acqNode) {
    Value offset, size;
    if (succeeded(acqNode->computeBlockInfo(offset, size))) {
      info.partitionOffsets.push_back(offset);
      info.partitionSizes.push_back(size);
      unsigned offsetIdx = 0;
      Value repOffset = DbUtils::pickRepresentativePartitionOffset(
          info.partitionOffsets, &offsetIdx);
      Value repSize = DbUtils::pickRepresentativePartitionSize(
          info.partitionSizes, offsetIdx);
      if (Value refined =
              refineSizeFromMinInBlock(acquire, repOffset, repSize)) {
        if (offsetIdx < info.partitionSizes.size())
          info.partitionSizes[offsetIdx] = refined;
        else if (!info.partitionSizes.empty())
          info.partitionSizes[0] = refined;
      }
      inferPartitionDims(info, contractSummary);
    } else {
      info.isValid = false;
    }
  } else {
    info.isValid = false;
  }
}

/// Analyze partition info for a fine-grained-mode acquire.
static void analyzeFineGrainedPartition(
    DbAnalysis::AcquirePartitionSummary &info, DbAcquireOp acquire,
    const DbAnalysis::AcquireContractSummary *contractSummary,
    OpBuilder &builder) {
  if (!acquire.getPartitionIndices().empty()) {
    for (Value idx : acquire.getPartitionIndices())
      info.partitionOffsets.push_back(idx);
    Value one = ::mlir::carts::arts::createOneIndex(builder, acquire.getLoc());
    for (unsigned i = 0; i < info.partitionOffsets.size(); ++i)
      info.partitionSizes.push_back(one);
    info.isValid = true;
    inferPartitionDims(info, contractSummary);
  }
}

/// Analyze partition info for a block/stencil-mode acquire.
static void analyzeBlockStencilPartition(
    DbAnalysis::AcquirePartitionSummary &info, DbAcquireOp acquire,
    DbAcquireNode *acqNode,
    const DbAnalysis::AcquireContractSummary *contractSummary,
    OpBuilder &builder) {
  /// Prefer explicit partition_* hints when available. If they were already
  /// materialized into offsets/sizes (post-rewrite acquires), reuse those so
  /// we do not regress valid block acquires back to coarse.
  ValueRange effectiveOffsets = acquire.getPartitionOffsets();
  ValueRange effectiveSizes = acquire.getPartitionSizes();
  if (effectiveOffsets.empty()) {
    effectiveOffsets = acquire.getOffsets();
    effectiveSizes = acquire.getSizes();
  }

  if (!effectiveOffsets.empty()) {
    for (Value off : effectiveOffsets)
      info.partitionOffsets.push_back(off);
    Value one = ::mlir::carts::arts::createOneIndex(builder, acquire.getLoc());
    for (unsigned i = 0; i < info.partitionOffsets.size(); ++i)
      info.partitionSizes.push_back(
          i < effectiveSizes.size() ? effectiveSizes[i] : one);
    info.isValid = true;

    unsigned offsetIdx = 0;
    Value repOffset = DbUtils::pickRepresentativePartitionOffset(
        info.partitionOffsets, &offsetIdx);
    Value repSize = DbUtils::pickRepresentativePartitionSize(
        info.partitionSizes, offsetIdx);
    if (Value refined = refineSizeFromMinInBlock(acquire, repOffset, repSize)) {
      if (offsetIdx < info.partitionSizes.size())
        info.partitionSizes[offsetIdx] = refined;
      else if (!info.partitionSizes.empty())
        info.partitionSizes[0] = refined;
    }

    /// For read-only stencil inputs, direct materialization already provides
    /// the exact halo-expanded slice in partition_offsets/sizes. Re-running
    /// graph bounds refinement here double-expands that window and breaks the
    /// later ESD lowering contract.
    bool trustExplicitHaloSlice =
        acquire.getMode() == ArtsMode::in &&
        !acquire.getElementOffsets().empty() &&
        acquire.getElementOffsets().size() == acquire.getElementSizes().size();
    if (!trustExplicitHaloSlice) {
      trustExplicitHaloSlice = contractSummary &&
                               acquire.getMode() == ArtsMode::in &&
                               contractSummary->contract.supportsBlockHalo() &&
                               contractSummary->usesStencilSemantics();
    }

    if (acqNode && !trustExplicitHaloSlice) {
      Value loopOffset, loopSize;
      if (succeeded(acqNode->computeBlockInfo(loopOffset, loopSize))) {
        bool useLoopSize = false;
        int64_t hintConst = 0;
        int64_t loopConst = 0;
        Value hintSize = DbUtils::pickRepresentativePartitionSize(
            info.partitionSizes, offsetIdx);
        Value hintOff = DbUtils::pickRepresentativePartitionOffset(
            info.partitionOffsets, &offsetIdx);
        bool hintIsConst =
            hintSize && ValueAnalysis::getConstantIndex(hintSize, hintConst);
        bool loopIsConst = ValueAnalysis::getConstantIndex(loopSize, loopConst);

        bool offsetRelated = false;
        if (hintOff && loopOffset) {
          Value hintOffStripped = ValueAnalysis::stripNumericCasts(hintOff);
          Value loopOff = ValueAnalysis::stripNumericCasts(loopOffset);
          if (hintOffStripped == loopOff)
            offsetRelated = true;
          if (!offsetRelated &&
              (ValueAnalysis::dependsOn(loopOff, hintOffStripped) ||
               ValueAnalysis::dependsOn(hintOffStripped, loopOff)))
            offsetRelated = true;
          if (!offsetRelated) {
            int64_t hintOffConst = 0;
            int64_t loopOffConst = 0;
            if (ValueAnalysis::getConstantIndex(hintOffStripped,
                                                hintOffConst) &&
                ValueAnalysis::getConstantIndex(loopOff, loopOffConst) &&
                hintOffConst == loopOffConst)
              offsetRelated = true;
          }
        }

        bool loopSizeDependsOnOffset =
            hintOff && ValueAnalysis::dependsOn(loopSize, hintOff);

        if (loopSizeDependsOnOffset)
          useLoopSize = true;
        if (offsetRelated && !loopIsConst)
          useLoopSize = true;
        if (offsetRelated && hintIsConst && loopIsConst &&
            hintConst != loopConst)
          useLoopSize = true;

        if (useLoopSize) {
          if (offsetIdx < info.partitionSizes.size())
            info.partitionSizes[offsetIdx] = loopSize;
          else if (!info.partitionSizes.empty())
            info.partitionSizes[0] = loopSize;
          else
            info.partitionSizes.push_back(loopSize);
        }
      }
    }
    inferPartitionDims(info, contractSummary);
    applyTiling2DPartitionDimsFallback(info, acquire);
    return;
  }

  if (acqNode) {
    Value offset, size;
    if (succeeded(acqNode->computeBlockInfo(offset, size))) {
      info.partitionOffsets.push_back(offset);
      info.partitionSizes.push_back(size);
      info.isValid = true;
      inferPartitionDims(info, contractSummary);
      applyTiling2DPartitionDimsFallback(info, acquire);
    } else {
      info.mode = PartitionMode::coarse;
    }
  }
}

DbAnalysis::AcquirePartitionSummary
DbAnalysis::analyzeAcquirePartition(DbAcquireOp acquire, OpBuilder &builder,
                                    const AcquireContractSummary *summary) {
  AcquirePartitionSummary info;
  if (!acquire)
    return info;

  if (auto modeAttr = getPartitionMode(acquire.getOperation()))
    info.mode = *modeAttr;
  else
    info.mode = DbAnalysis::getPartitionModeFromStructure(acquire);
  if (info.mode == PartitionMode::coarse) {
    if (!acquire.getPartitionIndices().empty())
      info.mode = PartitionMode::fine_grained;
    else if (!acquire.getPartitionOffsets().empty() ||
             !acquire.getPartitionSizes().empty())
      info.mode = PartitionMode::block;
  }

  DbAcquireNode *acqNode = nullptr;
  func::FuncOp func = acquire->getParentOfType<func::FuncOp>();
  if (func) {
    DbGraph &graph = getOrCreateGraph(func);
    acqNode = graph.getDbAcquireNode(acquire);
  }

  std::optional<AcquireContractSummary> ownedSummary;
  if (!summary) {
    ownedSummary = getAcquireContractSummary(acquire);
    if (ownedSummary)
      summary = &*ownedSummary;
  }

  if (summary) {
    info.hasIndirectAccess = summary->hasIndirectAccess();
    info.hasDistributionContract = summary->hasDistributionContract();
    info.partitionDimsFromPeers = summary->partitionDimsFromPeers();
  }

  switch (info.mode) {
  case PartitionMode::coarse:
    analyzeCoarsePartition(info, acquire, acqNode, summary);
    break;
  case PartitionMode::fine_grained:
    analyzeFineGrainedPartition(info, acquire, summary, builder);
    break;
  case PartitionMode::block:
  case PartitionMode::stencil:
    analyzeBlockStencilPartition(info, acquire, acqNode, summary, builder);
    break;
  }

  return info;
}

DbAnalysis::AcquireContractSummary
DbAnalysis::buildCanonicalAcquireContractSummary(DbAcquireOp acquire) {
  AcquireContractSummary summary;
  if (!acquire)
    return summary;

  summary.contract = buildAcquireContractSeed(acquire);
  populateSummaryFromCanonicalContract(summary, acquire);
  summary.accessPattern = resolveAcquireAccessPattern(summary);
  return summary;
}

std::optional<DbAnalysis::AcquireContractSummary>
DbAnalysis::getAcquireContractSummary(DbAcquireOp acquire) {
  if (!acquire)
    return std::nullopt;

  AcquireContractSummary summary =
      buildCanonicalAcquireContractSummary(acquire);

  func::FuncOp func = acquire->getParentOfType<func::FuncOp>();
  DbAcquireNode *acqNode = nullptr;
  if (func) {
    DbGraph &graph = getOrCreateGraph(func);
    acqNode = graph.getDbAcquireNode(acquire);
  }

  if (acqNode) {
    const DbAcquirePartitionFacts *facts = &acqNode->getPartitionFacts();
    const bool hasPersistedRefinement = summary.contract.analysis.postDbRefined;
    /// Once post-DB refinement is persisted, treat that contract as
    /// authoritative and avoid rediscovery-driven mutation here.
    if (!hasPersistedRefinement) {
      bool refined = refineContractWithFacts(summary.contract, *facts);
      refined =
          refineContractWithStencilBounds(summary.contract, acqNode) || refined;
      summary.refinedByDbAnalysis = refined;
    }
    populateSummaryFromCanonicalContract(summary, acquire);
    /// FIX-3: Contract persistence is handled by DT-1 in DbTransformsPass
    /// to avoid double-emission when getAcquireContractSummary() is called
    /// multiple times during analysis.
    mergeDerivedFactEvidence(summary, *facts);
  }

  summary.accessPattern = resolveAcquireAccessPattern(summary);
  if (summary.empty())
    return std::nullopt;
  return summary;
}

AccessPattern DbAnalysis::resolveCanonicalAcquireAccessPattern(
    DbAcquireOp acquire, const AcquireContractSummary *summary) {
  if (!acquire)
    return AccessPattern::Unknown;

  std::optional<AcquireContractSummary> ownedSummary;
  if (!summary) {
    ownedSummary = getAcquireContractSummary(acquire);
    if (ownedSummary)
      summary = &*ownedSummary;
  }

  if (summary && summary->accessPattern != AccessPattern::Unknown)
    return summary->accessPattern;
  if (summary && summary->contract.hasExplicitStencilContract())
    return AccessPattern::Stencil;

  return AccessPattern::Unknown;
}

ArtsDepPattern DbAnalysis::resolveCanonicalAcquireDepPattern(
    DbAcquireOp acquire, const AcquireContractSummary *summary) {
  if (!acquire)
    return ArtsDepPattern::unknown;

  std::optional<AcquireContractSummary> ownedSummary;
  if (!summary) {
    ownedSummary = getAcquireContractSummary(acquire);
    if (ownedSummary)
      summary = &*ownedSummary;
  }

  if (summary && summary->contract.pattern.depPattern)
    return *summary->contract.pattern.depPattern;

  return ArtsDepPattern::unknown;
}

bool DbAnalysis::hasCanonicalAcquireStencilSemantics(
    DbAcquireOp acquire, const AcquireContractSummary *summary) {
  if (resolveCanonicalAcquireAccessPattern(acquire, summary) ==
      AccessPattern::Stencil)
    return true;
  ArtsDepPattern depPattern =
      resolveCanonicalAcquireDepPattern(acquire, summary);
  return isStencilFamilyDepPattern(depPattern);
}

ArtsMode DbAnalysis::classifyMemrefUserAccessMode(Operation *op,
                                                  Operation *underlyingOp) {
  if (!op || !underlyingOp)
    return ArtsMode::uninitialized;

  bool hasRead = false;
  bool hasWrite = false;

  // Guard against malformed IR where a load/store operand has the wrong type.
  auto safeGetMemrefUnderlying = [&](Value v) -> Operation * {
    if (!v || !isa<BaseMemRefType>(v.getType()))
      return nullptr;
    return ValueAnalysis::getUnderlyingOperation(v);
  };

  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    hasRead = safeGetMemrefUnderlying(load->getOperand(0)) == underlyingOp;
  } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
    hasWrite = safeGetMemrefUnderlying(store->getOperand(1)) == underlyingOp;
  } else if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
    hasRead = safeGetMemrefUnderlying(load->getOperand(0)) == underlyingOp;
  } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
    hasWrite = safeGetMemrefUnderlying(store->getOperand(1)) == underlyingOp;
  } else if (auto copy = dyn_cast<memref::CopyOp>(op)) {
    hasRead = safeGetMemrefUnderlying(copy.getSource()) == underlyingOp;
    hasWrite = safeGetMemrefUnderlying(copy.getTarget()) == underlyingOp;
  }

  if (hasRead && hasWrite)
    return ArtsMode::inout;
  if (hasWrite)
    return ArtsMode::out;
  if (hasRead)
    return ArtsMode::in;
  return ArtsMode::uninitialized;
}

bool DbAnalysis::opMatchesAccessMode(Operation *op, Operation *underlyingOp,
                                     ArtsMode requestedMode) {
  ArtsMode actualMode = classifyMemrefUserAccessMode(op, underlyingOp);
  if (actualMode == ArtsMode::uninitialized)
    return false;
  if (requestedMode == ArtsMode::inout)
    return true;
  return actualMode == requestedMode;
}

bool DbAnalysis::accessModeCanSeedNestedAcquire(ArtsMode availableMode,
                                                ArtsMode requestedMode) {
  if (requestedMode == ArtsMode::uninitialized)
    return true;
  if (availableMode == ArtsMode::inout)
    return true;
  return availableMode == requestedMode;
}

ArtsMode DbAnalysis::inferEdtAccessMode(Operation *underlyingOp,
                                        EdtOp edt) const {
  if (!underlyingOp || !edt)
    return ArtsMode::uninitialized;

  ArtsMode combined = ArtsMode::uninitialized;
  edt.walk([&](Operation *op) {
    if (op->getParentOfType<EdtOp>() != edt)
      return;
    combined = combineAccessModes(
        combined, DbAnalysis::classifyMemrefUserAccessMode(op, underlyingOp));
  });
  return combined;
}

bool DbAnalysis::operationHasDistributedDbContract(Operation *op) {
  if (!op)
    return false;

  bool found = false;
  op->walk([&](DbAcquireOp acquire) {
    if (found)
      return WalkResult::interrupt();
    if (auto contractSummary = getAcquireContractSummary(acquire);
        contractSummary && contractSummary->hasDistributionContract()) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

bool DbAnalysis::operationHasPeerInferredPartitionDims(Operation *op) {
  if (!op)
    return false;

  bool found = false;
  op->walk([&](DbAcquireOp acquire) {
    if (found)
      return WalkResult::interrupt();
    if (auto contractSummary = getAcquireContractSummary(acquire);
        contractSummary && contractSummary->partitionDimsFromPeers()) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

std::optional<AccessPattern>
DbAnalysis::getAcquireAccessPattern(DbAcquireOp acquire) {
  if (!acquire)
    return std::nullopt;

  func::FuncOp func = acquire->getParentOfType<func::FuncOp>();
  if (!func)
    return std::nullopt;

  DbGraph &graph = getOrCreateGraph(func);
  DbAcquireNode *node = graph.getDbAcquireNode(acquire);
  if (!node)
    return std::nullopt;
  return node->getAccessPattern();
}

DbAllocNode *DbAnalysis::getDbAllocNode(DbAllocOp alloc) {
  auto func = alloc->getParentOfType<func::FuncOp>();
  if (!func)
    return nullptr;
  return getOrCreateGraph(func).getDbAllocNode(alloc);
}

DbAcquireNode *DbAnalysis::getDbAcquireNode(DbAcquireOp acquire) {
  auto func = acquire->getParentOfType<func::FuncOp>();
  if (!func)
    return nullptr;
  return getOrCreateGraph(func).getDbAcquireNode(acquire);
}

bool DbAnalysis::hasDbConflict(Operation *a, Operation *b) {
  return DbUtils::hasSharedWritableRootConflict(a, b);
}

///===----------------------------------------------------------------------===///
/// Partition / Granularity Queries (static)
///===----------------------------------------------------------------------===///

PartitionMode DbAnalysis::getPartitionModeFromStructure(DbAcquireOp acquire) {
  if (auto mode = ::getPartitionMode(acquire.getOperation()))
    return *mode;
  return PartitionMode::coarse;
}

PartitionMode DbAnalysis::getPartitionModeFromStructure(DbAllocOp alloc) {
  if (auto mode = ::getPartitionMode(alloc.getOperation()))
    return *mode;

  if (DbAnalysis::isCoarseGrained(alloc))
    return PartitionMode::coarse;

  return PartitionMode::fine_grained;
}

bool DbAnalysis::isCoarse(DbAcquireOp acquire) {
  return getPartitionModeFromStructure(acquire) == PartitionMode::coarse;
}

bool DbAnalysis::isCoarseGrained(DbAllocOp alloc) {
  if (auto mode = getPartitionMode(alloc.getOperation()))
    return *mode == PartitionMode::coarse;

  return llvm::all_of(alloc.getSizes(), [](Value v) {
    int64_t val;
    return ValueAnalysis::getConstantIndex(v, val) && val == 1;
  });
}

bool DbAnalysis::isFineGrained(DbAllocOp alloc) {
  if (auto mode = getPartitionMode(alloc.getOperation()))
    return *mode == PartitionMode::fine_grained;

  ValueRange elementSizes = alloc.getElementSizes();
  if (elementSizes.empty())
    return false;

  return llvm::all_of(elementSizes, [](Value v) {
    int64_t cst;
    return ValueAnalysis::getConstantIndex(v, cst) && cst == 1;
  });
}

bool DbAnalysis::hasSingleSize(Operation *dbOp) {
  if (!dbOp)
    return false;

  SmallVector<Value> sizes = DbUtils::getSizesFromDb(dbOp);
  if (sizes.empty())
    return true;

  for (Value size : sizes) {
    if (!ValueAnalysis::isOneLikeValue(size))
      return false;
  }
  return true;
}

///===----------------------------------------------------------------------===///
/// Semantic Queries (static)
///===----------------------------------------------------------------------===///

bool DbAnalysis::isSameMemoryObject(Value lhsMemref, Value rhsMemref) {
  lhsMemref = ValueAnalysis::stripNumericCasts(lhsMemref);
  rhsMemref = ValueAnalysis::stripNumericCasts(rhsMemref);

  Operation *lhsRoot = DbUtils::getUnderlyingDbAlloc(lhsMemref);
  Operation *rhsRoot = DbUtils::getUnderlyingDbAlloc(rhsMemref);
  if (lhsRoot && rhsRoot)
    return lhsRoot == rhsRoot;

  lhsRoot = ValueAnalysis::getUnderlyingOperation(lhsMemref);
  rhsRoot = ValueAnalysis::getUnderlyingOperation(rhsMemref);
  if (lhsRoot && rhsRoot)
    return lhsRoot == rhsRoot;

  return lhsMemref == rhsMemref;
}

bool DbAnalysis::hasStaticHints(DbAcquireOp acqOp) {
  /// Check partition hints (element-space) for static values
  Value offset = acqOp.getPartitionOffsets().empty()
                     ? nullptr
                     : acqOp.getPartitionOffsets().front();
  Value size = acqOp.getPartitionSizes().empty()
                   ? nullptr
                   : acqOp.getPartitionSizes().front();
  int64_t val = 0;
  bool offsetConst = !offset || ValueAnalysis::getConstantIndex(offset, val);
  bool sizeConst = !size || ValueAnalysis::getConstantIndex(size, val);
  return offsetConst && sizeConst;
}

std::optional<int64_t> DbAnalysis::getConstantOffsetBetween(Value idx,
                                                            Value base) {
  if (!idx || !base)
    return std::nullopt;

  /// Same value means offset 0
  if (idx == base)
    return 0;

  /// Strip numeric casts (index casts, sign/zero extensions, etc.)
  Value strippedIdx = ValueAnalysis::stripNumericCasts(idx);
  Value strippedBase = ValueAnalysis::stripNumericCasts(base);

  if (strippedIdx == strippedBase)
    return 0;

  /// Check if idx = base + constant
  if (auto addOp = strippedIdx.getDefiningOp<arith::AddIOp>()) {
    int64_t constVal;
    if (addOp.getLhs() == strippedBase &&
        ValueAnalysis::getConstantIndex(addOp.getRhs(), constVal))
      return constVal;
    if (addOp.getRhs() == strippedBase &&
        ValueAnalysis::getConstantIndex(addOp.getLhs(), constVal))
      return constVal;
  }

  /// Check if idx = base - constant
  if (auto subOp = strippedIdx.getDefiningOp<arith::SubIOp>()) {
    int64_t constVal;
    if (subOp.getLhs() == strippedBase &&
        ValueAnalysis::getConstantIndex(subOp.getRhs(), constVal))
      return -constVal;
  }

  /// Check the reverse: base = idx + constant means idx = base - constant
  if (auto addOp = strippedBase.getDefiningOp<arith::AddIOp>()) {
    int64_t constVal;
    if (addOp.getLhs() == strippedIdx &&
        ValueAnalysis::getConstantIndex(addOp.getRhs(), constVal))
      return -constVal;
    if (addOp.getRhs() == strippedIdx &&
        ValueAnalysis::getConstantIndex(addOp.getLhs(), constVal))
      return -constVal;
  }

  if (auto subOp = strippedBase.getDefiningOp<arith::SubIOp>()) {
    int64_t constVal;
    if (subOp.getLhs() == strippedIdx &&
        ValueAnalysis::getConstantIndex(subOp.getRhs(), constVal))
      return constVal;
  }

  return std::nullopt;
}

bool DbAnalysis::hasMultiEntryStencilPattern(DbAcquireOp acquire,
                                             int64_t &minOffset,
                                             int64_t &maxOffset) {
  size_t numEntries = acquire.getNumPartitionEntries();
  if (numEntries < 2)
    return false;

  /// Get indices for all entries
  SmallVector<SmallVector<Value>> allIndices;
  for (size_t i = 0; i < numEntries; ++i) {
    allIndices.push_back(acquire.getPartitionIndicesForEntry(i));
  }

  /// All entries must have the same number of indices (same dimensionality)
  size_t numDims = allIndices[0].size();
  if (numDims == 0)
    return false;
  for (const auto &indices : allIndices) {
    if (indices.size() != numDims)
      return false;
  }

  /// For each dimension, check if indices form a stencil pattern
  /// A stencil pattern means all indices are base +/- small constant
  bool foundStencilDim = false;
  minOffset = 0;
  maxOffset = 0;

  for (size_t dim = 0; dim < numDims; ++dim) {
    /// Try each entry as potential base
    bool dimIsStencil = false;
    int64_t dimMin = 0, dimMax = 0;

    for (size_t baseEntry = 0; baseEntry < numEntries && !dimIsStencil;
         ++baseEntry) {
      Value base = allIndices[baseEntry][dim];
      if (!base)
        continue;

      bool allMatch = true;
      int64_t localMin = 0, localMax = 0;

      for (size_t i = 0; i < numEntries; ++i) {
        Value idx = allIndices[i][dim];
        auto offset = getConstantOffsetBetween(idx, base);
        if (!offset || std::abs(*offset) > 2) {
          /// Not a small constant offset - not stencil in this dimension
          allMatch = false;
          break;
        }
        localMin = std::min(localMin, *offset);
        localMax = std::max(localMax, *offset);
      }

      if (allMatch && (localMin != localMax)) {
        /// Found stencil pattern in this dimension
        dimIsStencil = true;
        dimMin = localMin;
        dimMax = localMax;
      }
    }

    if (dimIsStencil) {
      foundStencilDim = true;
      /// Accumulate bounds (for multi-dimensional stencils, use the widest)
      minOffset = std::min(minOffset, dimMin);
      maxOffset = std::max(maxOffset, dimMax);
    }
  }

  return foundStencilDim;
}
