///==========================================================================///
/// File: DbDimAnalyzer.cpp
///
/// Canonical per-entry / per-dimension partition facts for DbAcquireNode.
///==========================================================================///

#include "carts/dialect/arts/Analysis/graphs/db/DbDimAnalyzer.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbGraph.h"
#include "carts/dialect/arts/Analysis/graphs/db/MemoryAccessClassifier.h"
#include "carts/dialect/arts/Analysis/graphs/db/PartitionBoundsAnalyzer.h"
#include "carts/utils/DbUtils.h"
#include "carts/utils/EdtUtils.h"
#include "carts/utils/LoweringContractUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/PartitionPredicates.h"
#include "carts/utils/ValueAnalysis.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::arts;

namespace {

static PartitionMode getRequestedMode(DbAcquireOp acquire) {
  if (auto mode = getPartitionMode(acquire.getOperation()))
    return *mode;
  return DbAnalysis::getPartitionModeFromStructure(acquire);
}

static std::pair<Value, Value>
pickRepresentativeRange(PartitionMode mode, const PartitionInfo &info) {
  auto pickRepresentative = [](ArrayRef<Value> values) -> Value {
    for (Value value : values) {
      if (!value)
        continue;
      int64_t constant = 0;
      if (!ValueAnalysis::getConstantIndex(
              ValueAnalysis::stripNumericCasts(value), constant))
        return value;
    }
    for (Value value : values) {
      if (value)
        return value;
    }
    return Value();
  };

  Value offset;
  if (usesElementLayout(mode))
    offset = pickRepresentative(info.indices);
  else
    offset = pickRepresentative(info.offsets);

  Value size;
  if (!info.sizes.empty())
    size = pickRepresentative(info.sizes);
  return {offset, size};
}

static SmallVector<PartitionInfo, 2>
collectPartitionEntries(DbAcquireOp acquire) {
  SmallVector<PartitionInfo, 2> entries = acquire.getPartitionInfos();
  if (!entries.empty())
    return entries;

  PartitionMode mode = getRequestedMode(acquire);
  if (!requiresWorkerBoundsPlanning(mode))
    return entries;

  PartitionInfo info;
  info.mode = mode;
  info.indices.append(acquire.getPartitionIndices().begin(),
                      acquire.getPartitionIndices().end());
  info.offsets.append(acquire.getPartitionOffsets().begin(),
                      acquire.getPartitionOffsets().end());
  info.sizes.append(acquire.getPartitionSizes().begin(),
                    acquire.getPartitionSizes().end());
  if (!info.indices.empty() || !info.offsets.empty() || !info.sizes.empty())
    entries.push_back(std::move(info));
  return entries;
}

static unsigned getPartitionRank(DbAcquireNode *node) {
  if (!node)
    return 0;
  DbAllocNode *allocNode = node->getRootAlloc();
  if (!allocNode)
    return 0;
  if (auto memrefType =
          dyn_cast<MemRefType>(allocNode->getDbAllocOp().getElementType()))
    return memrefType.getRank();
  return allocNode->getDbAllocOp().getElementSizes().size();
}

static void initializeDimFacts(DbAcquireNode *node,
                               DbAcquirePartitionFacts &facts) {
  unsigned rank = getPartitionRank(node);
  facts.dims.reserve(rank);

  SmallVector<DimAccessPatternType> allocDimPatterns;
  if (DbAllocNode *allocNode = node->getRootAlloc())
    allocDimPatterns = allocNode->dimAccessPatterns;

  for (unsigned dim = 0; dim < rank; ++dim) {
    DbDimPartitionFact fact;
    fact.dim = dim;
    if (dim < allocDimPatterns.size())
      fact.accessPattern = allocDimPatterns[dim];
    facts.dims.push_back(fact);
  }
}

static void markLeadingDynamicDims(DbAcquireNode *node,
                                   DbAcquirePartitionFacts &facts) {
  DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
  MemoryAccessClassifier::collectAccessOperations(node, dbRefToMemOps);

  for (auto &[dbRef, memOps] : dbRefToMemOps) {
    for (Operation *memOp : memOps) {
      SmallVector<Value> fullChain =
          DbUtils::collectFullIndexChain(dbRef, memOp);
      if (fullChain.empty())
        continue;

      unsigned memrefStart = dbRef.getIndices().size();
      for (unsigned chainIdx = memrefStart; chainIdx < fullChain.size();
           ++chainIdx) {
        int64_t constant = 0;
        if (ValueAnalysis::getConstantIndex(fullChain[chainIdx], constant))
          continue;

        unsigned dim = chainIdx - memrefStart;
        if (dim < facts.dims.size())
          facts.dims[dim].isLeadingDynamic = true;
        break;
      }
    }
  }
}

static bool hasDistributionContract(DbAcquireOp acquire) {
  auto [edt, blockArg] = EdtUtils::getBlockArgumentForAcquire(acquire);
  (void)blockArg;
  if (edt && (getEdtDistributionKind(edt.getOperation()) ||
              getEdtDistributionPattern(edt.getOperation())))
    return true;

  if (auto epoch = acquire->getParentOfType<EpochOp>())
    return getEdtDistributionKind(epoch.getOperation()) ||
           getEdtDistributionPattern(epoch.getOperation());

  return false;
}

static SmallVector<unsigned, 4>
collectMappedEntryDims(ArrayRef<DbPartitionEntryFact> entries) {
  SmallVector<unsigned, 4> mappedDims;
  SmallVector<unsigned, 4> seenDims;
  for (const DbPartitionEntryFact &entry : entries) {
    if (!entry.representativeOffset || !entry.mappedDim)
      return {};
    if (llvm::is_contained(seenDims, *entry.mappedDim))
      return {};

    seenDims.push_back(*entry.mappedDim);
    mappedDims.push_back(*entry.mappedDim);
  }
  return mappedDims;
}

static std::optional<unsigned>
inferMappedDimFromDepPattern(const DbAcquirePartitionFacts &facts) {
  switch (facts.depPattern) {
  case ArtsDepPattern::wavefront_2d:
    /// The wavefront transform commits to leading-dimension row ownership and
    /// rewrites the distributed loop into row-space. Preserve that fact here
    /// so later DB partitioning does not have to rediscover it from raw access
    /// expressions.
    return facts.dims.empty() ? std::nullopt : std::optional<unsigned>(0u);
  case ArtsDepPattern::stencil_tiling_nd:
  case ArtsDepPattern::cross_dim_stencil_3d:
  case ArtsDepPattern::higher_order_stencil:
  case ArtsDepPattern::jacobi_alternating_buffers:
    if (!facts.stencilOwnerDims.empty())
      return facts.stencilOwnerDims.front();
    return std::nullopt;
  case ArtsDepPattern::unknown:
  case ArtsDepPattern::uniform:
  case ArtsDepPattern::stencil:
  case ArtsDepPattern::matmul:
  case ArtsDepPattern::triangular:
  case ArtsDepPattern::elementwise_pipeline:
    return std::nullopt;
  }
}

static bool
shouldPreserveWavefrontRowLocalRange(const DbAcquirePartitionFacts &facts,
                                     const DbPartitionEntryFact &entry) {
  return facts.depPattern == ArtsDepPattern::wavefront_2d && entry.mappedDim &&
         *entry.mappedDim == 0;
}

static SmallVector<unsigned, 4>
collectDistributedPeerDims(DbAcquireNode *node) {
  if (!node || !node->getAnalysis())
    return {};

  DbAcquireOp acquire = node->getDbAcquireOp();
  if (!acquire)
    return {};

  Operation *scope = nullptr;
  if (auto epoch = acquire->getParentOfType<EpochOp>())
    scope = epoch.getOperation();
  else if (auto edt = acquire->getParentOfType<EdtOp>())
    scope = edt.getOperation();
  if (!scope)
    return {};

  func::FuncOp func = acquire->getParentOfType<func::FuncOp>();
  if (!func)
    return {};

  DbGraph &graph = node->getAnalysis()->getOrCreateGraph(func);
  SmallVector<unsigned, 4> agreedDims;
  bool sawCandidate = false;
  bool conflict = false;

  scope->walk([&](DbAcquireOp peer) {
    if (conflict || peer == acquire)
      return WalkResult::advance();

    DbAcquireNode *peerNode = graph.getDbAcquireNode(peer);
    if (!peerNode)
      return WalkResult::advance();

    SmallVector<PartitionInfo, 2> peerEntries = collectPartitionEntries(peer);
    if (peerEntries.empty())
      return WalkResult::advance();

    SmallVector<DbPartitionEntryFact, 2> peerEntryFacts;
    for (const PartitionInfo &peerEntry : peerEntries) {
      DbPartitionEntryFact fact;
      std::tie(fact.representativeOffset, fact.representativeSize) =
          pickRepresentativeRange(peerEntry.mode, peerEntry);
      if (fact.representativeOffset) {
        fact.mappedDim = peerNode->getPartitionOffsetDim(
            fact.representativeOffset, /*requireLeading=*/false);
      }
      peerEntryFacts.push_back(std::move(fact));
    }

    SmallVector<unsigned, 4> peerDims = collectMappedEntryDims(peerEntryFacts);
    if (peerDims.empty())
      return WalkResult::advance();

    if (!sawCandidate) {
      agreedDims = peerDims;
      sawCandidate = true;
      return WalkResult::advance();
    }

    if (agreedDims.size() != peerDims.size() ||
        !llvm::equal(agreedDims, peerDims))
      conflict = true;
    return WalkResult::advance();
  });

  if (conflict || !sawCandidate)
    return {};
  return agreedDims;
}

static void inferPartitionDims(DbAcquireNode *node,
                               DbAcquirePartitionFacts &facts) {
  facts.partitionDims = collectMappedEntryDims(facts.entries);
  if (!facts.partitionDims.empty() || !facts.hasDistributionContract)
    return;

  /// Helper-call-hidden stencil accesses need their own mapped partition dim.
  /// Peer dim inference is too coarse and can synthesize an unsafe local slice
  /// for arrays that only expose their real access pattern after inlining.
  if (facts.accessPattern == AccessPattern::Stencil)
    return;

  /// Coarse acquires often arrive here without any local partition entry even
  /// though the surrounding loop has already committed to an N-D worker
  /// ownership plan. In that case, reuse the scope-wide peer consensus instead
  /// of falling back to an implicit leading dimension later in DB rewriting.
  ///
  /// When local entries do exist, keep the older guard and only consult peer
  /// ownership when at least one entry explicitly preserves the distributed
  /// contract.
  bool shouldInferFromPeers = facts.entries.empty();
  if (!shouldInferFromPeers) {
    shouldInferFromPeers =
        llvm::any_of(facts.entries, [](const DbPartitionEntryFact &entry) {
          return entry.preservesDistributedContract;
        });
  }
  if (!shouldInferFromPeers)
    return;

  SmallVector<unsigned, 4> peerDims = collectDistributedPeerDims(node);
  if (peerDims.empty())
    return;

  /// Peer ownership is only usable when it fits this acquire's own logical
  /// rank. Keep the inference conservative for unrelated helper/coefficient
  /// DBs that happen to share the same scope.
  if (llvm::any_of(peerDims,
                   [&](unsigned dim) { return dim >= facts.dims.size(); }))
    return;

  facts.partitionDims = std::move(peerDims);
  facts.partitionDimsFromPeers = true;
}

} // namespace

DbAcquirePartitionFacts DbDimAnalyzer::compute(DbAcquireNode *node) {
  DbAcquirePartitionFacts facts;
  if (!node)
    return facts;

  DbAcquireOp acquire = node->getDbAcquireOp();
  if (!acquire)
    return facts;

  facts.acquire = acquire;
  facts.requestedMode = getRequestedMode(acquire);
  DbAnalysis::AcquireContractSummary canonicalSummary =
      DbAnalysis::buildCanonicalAcquireContractSummary(acquire);
  if (canonicalSummary.contract.pattern.depPattern)
    facts.depPattern = *canonicalSummary.contract.pattern.depPattern;
  facts.supportedBlockHalo = canonicalSummary.contract.supportsBlockHalo();
  facts.stencilOwnerDims.assign(canonicalSummary.partitionDims.begin(),
                                canonicalSummary.partitionDims.end());
  facts.accessPattern = canonicalSummary.accessPattern != AccessPattern::Unknown
                            ? canonicalSummary.accessPattern
                            : node->getAccessPattern();
  facts.hasIndirectAccess = node->hasIndirectAccess();
  facts.hasDirectAccess = node->hasDirectAccess();
  facts.hasDistributionContract = canonicalSummary.hasDistributionContract() ||
                                  hasDistributionContract(acquire);
  facts.explicitCoarseRequest =
      !requiresWorkerBoundsPlanning(facts.requestedMode) &&
      acquire.getPartitionIndices().empty() &&
      acquire.getPartitionOffsets().empty() &&
      acquire.getPartitionSizes().empty();
  facts.hasBlockHints = !acquire.getPartitionOffsets().empty() ||
                        !acquire.getPartitionSizes().empty();

  initializeDimFacts(node, facts);
  markLeadingDynamicDims(node, facts);

  SmallVector<PartitionInfo, 2> entries = collectPartitionEntries(acquire);
  facts.inferredBlock = llvm::any_of(entries, [](const PartitionInfo &entry) {
    return !entry.offsets.empty() && !entry.sizes.empty();
  });

  llvm::DenseSet<unsigned> fullRangeDims;
  for (const PartitionInfo &entry : entries) {
    DbPartitionEntryFact fact;
    fact.partition = entry;
    std::tie(fact.representativeOffset, fact.representativeSize) =
        pickRepresentativeRange(entry.mode, entry);

    if (fact.representativeOffset) {
      fact.mappedDim = node->getPartitionOffsetDim(fact.representativeOffset,
                                                   /*requireLeading=*/false);
      if (!fact.mappedDim)
        fact.mappedDim = inferMappedDimFromDepPattern(facts);
      fact.needsFullRange = PartitionBoundsAnalyzer::needsFullRange(
          node, fact.representativeOffset);
      fact.preservesDistributedContract =
          PartitionBoundsAnalyzer::shouldPreserveDistributedContract(
              node, fact.representativeOffset);

      if (shouldPreserveWavefrontRowLocalRange(facts, fact)) {
        /// The wavefront transform already committed the task to a leading-row
        /// schedule and rewrote the distributed loop into row-space. The raw
        /// index matcher does not always see that relation through the nested
        /// inner loops, so keep the task-local range instead of forcing the
        /// acquire back to full-range.
        fact.needsFullRange = false;
        fact.preservesDistributedContract = true;
      }
    }

    if (fact.mappedDim && *fact.mappedDim < facts.dims.size()) {
      facts.dims[*fact.mappedDim].selectedByPartitionEntry = true;
      if (fact.needsFullRange)
        fullRangeDims.insert(*fact.mappedDim);
    }

    facts.entries.push_back(std::move(fact));
  }

  inferPartitionDims(node, facts);

  /// Physical owner dims are part of the contract shape, not part of the ESD
  /// transport decision. If analysis could not recover a mapped dim directly
  /// from partition entries, fall back to the explicit owner-dim contract even
  /// when block-halo transport is disabled.
  if (facts.partitionDims.empty() && !facts.stencilOwnerDims.empty()) {
    facts.partitionDims.assign(facts.stencilOwnerDims.begin(),
                               facts.stencilOwnerDims.end());
  }

  for (unsigned dim : fullRangeDims) {
    if (dim < facts.dims.size())
      facts.dims[dim].needsFullRange = true;
  }

  return facts;
}
