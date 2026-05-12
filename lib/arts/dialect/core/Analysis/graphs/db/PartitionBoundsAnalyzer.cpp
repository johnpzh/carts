///==========================================================================///
/// File: PartitionBoundsAnalyzer.cpp
///
/// Implementation of PartitionBoundsAnalyzer -- partition bound computation
/// and eligibility checking extracted from DbAcquireNode.
///==========================================================================///

#include "arts/dialect/core/Analysis/graphs/db/PartitionBoundsAnalyzer.h"
#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AccessPatternAnalysis.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/db/DbAnalysis.h"
#include "arts/dialect/core/Analysis/edt/EdtAnalysis.h"
#include "arts/dialect/core/Analysis/graphs/db/DbNode.h"
#include "arts/dialect/core/Analysis/graphs/db/MemoryAccessClassifier.h"
#include "arts/dialect/core/Analysis/graphs/edt/EdtGraph.h"
#include "arts/dialect/core/Analysis/graphs/edt/EdtNode.h"
#include "arts/dialect/core/Analysis/loop/LoopAnalysis.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/EdtUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/PartitionPredicates.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/Utils.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::arts;
#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(partition_bounds_analyzer);

///===----------------------------------------------------------------------===///
/// Shared partition hint extraction
///===----------------------------------------------------------------------===///

/// Pick a representative value from a range, preferring non-constant values.
/// Returns the chosen value and sets outIdx to its position.
static Value pickRepresentativeValue(ValueRange vals, unsigned &outIdx) {
  outIdx = 0;
  for (unsigned i = 0; i < vals.size(); ++i) {
    Value v = vals[i];
    if (!v)
      continue;
    int64_t c = 0;
    if (!ValueAnalysis::getConstantIndex(ValueAnalysis::stripNumericCasts(v),
                                         c)) {
      outIdx = i;
      return v;
    }
  }
  for (unsigned i = 0; i < vals.size(); ++i) {
    if (vals[i]) {
      outIdx = i;
      return vals[i];
    }
  }
  return Value();
}

/// Pick the partition size corresponding to a given representative index.
static Value pickPartitionSize(ValueRange sizes, unsigned idx) {
  if (sizes.empty())
    return Value();
  if (idx < sizes.size())
    return sizes[idx];
  return sizes.front();
}

AcquirePartitionHints mlir::arts::extractPartitionHints(DbAcquireOp acquire) {
  AcquirePartitionHints hints;
  unsigned idx = 0;

  auto hintMode = acquire.getPartitionMode();
  bool preferPartitionIndices =
      !hintMode || *hintMode == PartitionMode::fine_grained;

  if (preferPartitionIndices && !acquire.getPartitionIndices().empty())
    hints.offset = pickRepresentativeValue(acquire.getPartitionIndices(), idx);
  else if (!acquire.getPartitionOffsets().empty())
    hints.offset = pickRepresentativeValue(acquire.getPartitionOffsets(), idx);
  else if (!acquire.getPartitionIndices().empty())
    hints.offset = pickRepresentativeValue(acquire.getPartitionIndices(), idx);

  hints.size = pickPartitionSize(acquire.getPartitionSizes(), idx);

  /// Fallback to DB-space indices/offsets/sizes only when an explicit
  /// non-coarse partition mode is set (already-partitioned acquires).
  if (!hints.offset && !hints.size) {
    if (auto mode = acquire.getPartitionMode();
        mode && requiresWorkerBoundsPlanning(*mode)) {
      bool preferDbIndices = usesElementLayout(*mode);
      if (preferDbIndices && !acquire.getIndices().empty())
        hints.offset = pickRepresentativeValue(acquire.getIndices(), idx);
      else if (!acquire.getOffsets().empty())
        hints.offset = pickRepresentativeValue(acquire.getOffsets(), idx);
      else if (!acquire.getIndices().empty())
        hints.offset = pickRepresentativeValue(acquire.getIndices(), idx);

      hints.size = pickPartitionSize(acquire.getSizes(), idx);
    }
  }

  return hints;
}

///===----------------------------------------------------------------------===///
/// File-local helpers
///===----------------------------------------------------------------------===///

using AccessBoundsInfo = AccessBoundsResult;

/// Find the best loop node for bounds analysis.
static LoopNode *findBestLoopNode(ArrayRef<LoopNode *> loopNodes,
                                  Value firstDynIdx, Value partitionOffset,
                                  bool offsetIsZero) {
  LoopNode *bestNode = nullptr;
  int bestDepth = -1;
  bool foundPreferred = false;

  for (LoopNode *loopNode : loopNodes) {
    if (!loopNode->dependsOnInductionVar(firstDynIdx))
      continue;
    if (!offsetIsZero &&
        !ValueAnalysis::dependsOn(firstDynIdx, partitionOffset))
      continue;

    Value loopIV = loopNode->getInductionVar();
    bool offsetDependsOnIV = !offsetIsZero && partitionOffset &&
                             ValueAnalysis::dependsOn(partitionOffset, loopIV);
    bool isPreferred = !offsetDependsOnIV;
    int depth = loopNode->getNestingDepth();

    if (isPreferred) {
      if (!foundPreferred || depth > bestDepth) {
        foundPreferred = true;
        bestDepth = depth;
        bestNode = loopNode;
      }
    } else if (!foundPreferred && depth > bestDepth) {
      bestDepth = depth;
      bestNode = loopNode;
    }
  }

  return bestNode;
}

/// Analyze all memory accesses to compute actual bounds relative to block base.
static AccessBoundsInfo
analyzeAccessBounds(DbAcquireNode *node, Value blockOffset, Value loopIV,
                    std::optional<unsigned> partitionDim = std::nullopt) {
  if (!node || !blockOffset || !loopIV)
    return AccessBoundsInfo{};

  DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
  MemoryAccessClassifier::collectAccessOperations(node, dbRefToMemOps);

  SmallVector<AccessIndexInfo, 16> accesses;

  for (auto &[dbRef, memOps] : dbRefToMemOps) {
    for (Operation *memOp : memOps) {
      SmallVector<Value> indexChain =
          DbUtils::collectFullIndexChain(dbRef, memOp);
      if (indexChain.empty())
        continue;
      AccessIndexInfo info;
      info.dbRefPrefix = dbRef.getIndices().size();
      info.indexChain.append(indexChain.begin(), indexChain.end());
      accesses.push_back(std::move(info));
    }
  }

  return analyzeAccessBoundsFromIndices(accesses, loopIV, blockOffset,
                                        partitionDim);
}

static Value normalizePartitionOffsetRoot(Value offset) {
  if (!offset)
    return Value();

  Value normalized = ValueAnalysis::stripNumericCasts(offset);
  normalized = ValueAnalysis::stripConstantOffset(normalized, nullptr);
  normalized = ValueAnalysis::stripNumericCasts(
      ValueAnalysis::stripSelectClamp(normalized));
  return normalized;
}

static std::optional<int64_t>
getPartitionRelativeConstantOffset(Value idx, Value partitionOffset) {
  if (!idx || !partitionOffset)
    return std::nullopt;

  Value normalizedIdx = ValueAnalysis::stripNumericCasts(idx);
  Value normalizedOffset = normalizePartitionOffsetRoot(partitionOffset);
  if (!normalizedOffset)
    return std::nullopt;

  if (!ValueAnalysis::dependsOn(normalizedIdx, normalizedOffset) &&
      !ValueAnalysis::dependsOn(normalizedIdx, partitionOffset) &&
      !ValueAnalysis::sameValue(normalizedIdx, normalizedOffset) &&
      !ValueAnalysis::sameValue(normalizedIdx, partitionOffset) &&
      !ValueAnalysis::areValuesEquivalent(normalizedIdx, normalizedOffset) &&
      !ValueAnalysis::areValuesEquivalent(normalizedIdx, partitionOffset))
    return std::nullopt;

  return ValueAnalysis::extractConstantOffset(normalizedIdx, partitionOffset,
                                              normalizedOffset);
}

/// Collect the set of array dimensions that carry non-zero stencil offsets
/// relative to the loop induction variable / partition offset.
///
/// For example, given accesses A[i-1][j][k], A[i+1][j][k], A[i][j][k]:
///   stencil dims = {0}   (dim 0 has offsets -1, +1)
///
/// This is used to detect orthogonal stencils: when all stencil dimensions are
/// disjoint from the partition dimension, the stencil does not cross partition
/// boundaries and needsFullRange can safely return false.
static llvm::DenseSet<unsigned> collectStencilDims(DbAcquireNode *node,
                                                   Value partitionOffset) {
  llvm::DenseSet<unsigned> stencilDims;
  if (!node || !partitionOffset)
    return stencilDims;

  DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
  MemoryAccessClassifier::collectAccessOperations(node, dbRefToMemOps);

  /// Collect all index chains first.
  SmallVector<AccessIndexInfo, 16> accesses;
  for (auto &[dbRef, memOps] : dbRefToMemOps) {
    for (Operation *memOp : memOps) {
      SmallVector<Value> fullChain =
          DbUtils::collectFullIndexChain(dbRef, memOp);
      if (fullChain.empty())
        continue;
      AccessIndexInfo info;
      info.dbRefPrefix = dbRef.getIndices().size();
      info.indexChain.append(fullChain.begin(), fullChain.end());
      accesses.push_back(std::move(info));
    }
  }

  if (accesses.empty())
    return stencilDims;

  /// For each memref dimension, a "stencil dimension" carries at least one
  /// non-zero constant offset relative to partition offset / loop IV.
  for (const AccessIndexInfo &access : accesses) {
    unsigned memrefStart = access.dbRefPrefix;
    for (unsigned i = memrefStart; i < access.indexChain.size(); ++i) {
      unsigned dim = i - memrefStart;
      Value idx = access.indexChain[i];
      if (!idx)
        continue;

      /// Ask ValueAnalysis to prove the exact constant delta from the
      /// partition root before falling back to bespoke per-dim reasoning.
      auto constOffset =
          getPartitionRelativeConstantOffset(idx, partitionOffset);
      if (constOffset && *constOffset != 0)
        stencilDims.insert(dim);
    }
  }

  return stencilDims;
}

/// Read-only affine stencil accesses can stay block-local even when the
/// partitioned dimension is non-leading. The later stencil-halo rewrite uses
/// graph-derived min/max offsets to widen the dependency window just enough to
/// cover the needed neighbors, so full-range is unnecessary here.
static bool canUseReadOnlyStencilHaloWindow(DbAcquireNode *node,
                                            std::optional<unsigned> mappedDim,
                                            Value partitionOffset) {
  if (!node || !mappedDim || !partitionOffset)
    return false;
  if (MemoryAccessClassifier::hasStores(node) ||
      MemoryAccessClassifier::hasIndirectAccess(node) ||
      node->getAccessPattern() != AccessPattern::Stencil ||
      node->getHasNonConstantOffset())
    return false;

  /// Bounds analysis already measures halo on the partitioned dimension.
  /// If that graph fact says the read-only stencil needs neighbors, the later
  /// stencil rewrite can materialize the exact local halo window without
  /// widening the acquire to the whole allocation.
  if (auto stencilBounds = node->getStencilBounds())
    return stencilBounds->hasHalo();

  llvm::DenseSet<unsigned> stencilDims =
      collectStencilDims(node, partitionOffset);
  return !stencilDims.empty() && stencilDims.contains(*mappedDim);
}

static DbAnalysis::AcquireContractSummary
buildCanonicalAcquireContractSummary(DbAcquireNode *node) {
  if (!node)
    return {};
  return DbAnalysis::buildCanonicalAcquireContractSummary(
      node->getDbAcquireOp());
}

///===----------------------------------------------------------------------===///
/// PartitionBoundsAnalyzer implementation
///===----------------------------------------------------------------------===///

bool PartitionBoundsAnalyzer::hasValidEdtAndAccesses(DbAcquireNode *node) {
  auto skip = [&](StringRef reason) {
    ARTS_DEBUG("Skipping acquire " << node->getDbAcquireOp() << ": " << reason);
    return false;
  };

  if (!node->getDbAcquireOp())
    return skip("invalid acquire operation");

  if (!node->getEdtUserOp()) {
    if (!MemoryAccessClassifier::hasMemoryAccesses(node)) {
      ARTS_DEBUG("No EDT user and no memory accesses - full-range acquire");
      return true;
    }
    DbAllocNode *allocNode = node->getRootAlloc();
    if (allocNode) {
      DbAllocOp allocOp = allocNode->getDbAllocOp();
      if (allocOp && !allocOp.getElementSizes().empty()) {
        ARTS_DEBUG("No EDT user - full-range acquire with alloc sizes");
        return true;
      }
    }
    return skip("no EDT user and missing allocation sizes");
  }

  EdtOp edt = node->getEdtUser();
  if (!edt || !edt.getOperation())
    return skip("missing EDT");

  bool hasMemAccesses = MemoryAccessClassifier::hasMemoryAccesses(node);
  bool isTaskEdt = (edt.getType() == EdtType::task);
  bool isPassThrough = !isTaskEdt && !hasMemAccesses;

  if (!isTaskEdt && !isPassThrough)
    return skip("not a task EDT");

  if (isTaskEdt && !hasMemAccesses)
    return skip("no memory accesses");

  DbAllocNode *allocNode = node->getRootAlloc();
  if (!allocNode)
    return skip("missing allocation node");

  if (!allocNode->isParallelFriendly()) {
    ARTS_DEBUG("Memref not marked parallel-friendly - allowing for heuristic "
               "evaluation");
  }

  AnalysisManager &AM = node->getAnalysis()->getAnalysisManager();
  EdtAnalysis &edtAnalysis = AM.getEdtAnalysis();

  func::FuncOp func = edt->getParentOfType<func::FuncOp>();
  if (!func)
    return skip("unable to find parent function");

  EdtGraph &edtGraph = edtAnalysis.getOrCreateEdtGraph(func);
  EdtNode *edtNode = edtGraph.getEdtNode(edt);
  if (!edtNode)
    return skip("missing EDT node");

  if (isTaskEdt) {
    node->setPartitionInfo(Value(), Value());
  }

  return true;
}

bool PartitionBoundsAnalyzer::computePartitionBounds(DbAcquireNode *node) {
  DbAcquireOp mutableAcquire =
      DbAcquireOp(node->getDbAcquireOp().getOperation());

  node->setPartitionInfo(Value(), Value());
  node->setHasNonConstantOffset(false);

  AcquirePartitionHints hints = extractPartitionHints(mutableAcquire);
  Value partitionOffset = hints.offset;
  Value partitionSize = hints.size;

  node->setPartitionInfo(partitionOffset, partitionSize);

  if (!node->getEdtUserOp())
    return true;

  bool hasPartitionHint = !mutableAcquire.getPartitionIndices().empty() ||
                          !mutableAcquire.getPartitionOffsets().empty() ||
                          !mutableAcquire.getPartitionSizes().empty();
  if (partitionOffset && !canPartitionWithOffset(node, partitionOffset)) {
    if (hasPartitionHint) {
      ARTS_DEBUG("Partition offset not derived from access; "
                 "continuing with partition hint");
    } else {
      ARTS_DEBUG("Partition offset not derived from access; "
                 "allowing for heuristic evaluation");
    }
  }

  if (!partitionOffset || !partitionSize)
    return true;

  Value analysisOffset = partitionOffset;
  analysisOffset = ValueAnalysis::stripConstantOffset(analysisOffset, nullptr);

  node->setOriginalBounds(std::make_pair(partitionOffset, partitionSize));
  AnalysisManager &AM = node->getAnalysis()->getAnalysisManager();
  LoopAnalysis &loopAnalysis = AM.getLoopAnalysis();

  DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
  MemoryAccessClassifier::collectAccessOperations(node, dbRefToMemOps);

  std::optional<unsigned> partitionDim =
      analysisOffset ? getPartitionOffsetDim(node, analysisOffset,
                                             /*requireLeading=*/false)
                     : std::nullopt;

  Value partitionIdx;
  Value firstDynIdx;
  for (auto &[dbRef, memOps] : dbRefToMemOps) {
    for (Operation *memOp : memOps) {
      SmallVector<Value> fullChain =
          DbUtils::collectFullIndexChain(dbRef, memOp);
      if (!partitionIdx && partitionDim) {
        unsigned memrefStart = dbRef.getIndices().size();
        unsigned chainIdx = memrefStart + *partitionDim;
        if (chainIdx < fullChain.size())
          partitionIdx = fullChain[chainIdx];
      }
      for (Value idx : fullChain) {
        int64_t constVal;
        if (!ValueAnalysis::getConstantIndex(idx, constVal)) {
          firstDynIdx = idx;
          break;
        }
      }
      if (partitionIdx || firstDynIdx)
        break;
    }
    if (partitionIdx || firstDynIdx)
      break;
  }

  bool offsetIsZero = ValueAnalysis::isZeroConstant(
      ValueAnalysis::stripNumericCasts(analysisOffset));
  Value loopIdx = partitionIdx ? partitionIdx : firstDynIdx;
  if (!loopIdx) {
    if (offsetIsZero) {
      ARTS_DEBUG("  No dynamic index - allowing zero-offset partition hints");
      node->setComputedBlockInfo(
          std::make_pair(partitionOffset, partitionSize));
      return true;
    }
    ARTS_DEBUG("  No dynamic index - allowing for heuristic evaluation");
    return true;
  }

  EdtOp edt = node->getEdtUser();
  SmallVector<LoopNode *, 4> loopNodes;
  loopAnalysis.collectForLoopsInOperation(edt, loopNodes);

  if (loopNodes.empty()) {
    ARTS_DEBUG("  No for loops - allowing for heuristic evaluation");
    return true;
  }

  LoopNode *blockLoopNode =
      findBestLoopNode(loopNodes, loopIdx, analysisOffset, offsetIsZero);

  if (!blockLoopNode) {
    ARTS_DEBUG("  No block loop found - allowing for heuristic evaluation");
    return true;
  }

  Value loopIV = blockLoopNode->getInductionVar();
  AccessBoundsInfo bounds =
      analyzeAccessBounds(node, analysisOffset, loopIV, partitionDim);

  if (!bounds.valid) {
    if (bounds.hasVariableOffset && partitionDim)
      node->setHasNonConstantOffset(true);
    ARTS_DEBUG(
        "  Access bounds analysis failed - allowing for heuristic evaluation");
    return true;
  }

  DbAcquireOp dbAcquireOp = node->getDbAcquireOp();
  ARTS_DEBUG("  Bounds analysis for acquire "
             << dbAcquireOp << ": minOffset=" << bounds.minOffset
             << ", maxOffset=" << bounds.maxOffset
             << ", isStencil=" << bounds.isStencil);

  node->setStencilBoundsInternal(StencilBounds::create(
      bounds.minOffset, bounds.maxOffset, bounds.isStencil, bounds.valid));

  OpBuilder builder(dbAcquireOp);
  Location loc = dbAcquireOp.getLoc();
  builder.setInsertionPoint(dbAcquireOp);

  if (bounds.valid && bounds.isStencil && bounds.centerOffset != 0)
    setStencilCenterOffset(dbAcquireOp.getOperation(), bounds.centerOffset);

  Value adjustedOffset = partitionOffset;
  Value adjustedSize = partitionSize;

  if (bounds.minOffset < 0) {
    Value absAdj = arts::createConstantIndex(builder, loc, -bounds.minOffset);
    Value cond = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::uge,
                                       partitionOffset, absAdj);
    Value sub = arith::SubIOp::create(builder, loc, partitionOffset, absAdj);
    Value zero = arts::createZeroIndex(builder, loc);
    adjustedOffset = arith::SelectOp::create(builder, loc, cond, sub, zero);
  } else if (bounds.minOffset > 0) {
    Value adj = arts::createConstantIndex(builder, loc, bounds.minOffset);
    adjustedOffset = arith::AddIOp::create(builder, loc, partitionOffset, adj);
  }

  int64_t sizeAdjustment = bounds.maxOffset - bounds.minOffset;
  if (sizeAdjustment != 0) {
    Value adjustment = arts::createConstantIndex(builder, loc, sizeAdjustment);
    adjustedSize =
        arith::AddIOp::create(builder, loc, partitionSize, adjustment);
  }

  node->setComputedBlockInfo(std::make_pair(adjustedOffset, adjustedSize));
  return true;
}

bool PartitionBoundsAnalyzer::canBePartitioned(DbAcquireNode *node) {
  if (!node->getEdtUserOp())
    return false;

  if (!hasValidEdtAndAccesses(node))
    return false;

  if (!computePartitionBounds(node))
    return false;

  bool childFailed = false;
  node->forEachChildNode([&](NodeBase *child) {
    if (childFailed)
      return;
    auto *childAcq = dyn_cast<DbAcquireNode>(child);
    if (childAcq && !canBePartitioned(childAcq)) {
      ARTS_DEBUG("Skipping acquire " << node->getDbAcquireOp()
                                     << ": nested acquire child failed");
      childFailed = true;
    }
  });

  if (childFailed)
    return false;

  ARTS_DEBUG("PASS: acquire can be partitioned: " << node->getDbAcquireOp());
  return true;
}

bool PartitionBoundsAnalyzer::canPartitionWithOffset(DbAcquireNode *node,
                                                     Value offset) {
  return getPartitionOffsetDim(node, offset, /*requireLeading=*/false)
      .has_value();
}

std::optional<unsigned> PartitionBoundsAnalyzer::getPartitionOffsetDim(
    DbAcquireNode *node, Value offset, bool requireLeading) {
  ARTS_DEBUG("getPartitionOffsetDim called with offset="
             << offset << " requireLeading=" << requireLeading);
  if (!offset) {
    ARTS_DEBUG("  -> returning none (no offset)");
    return std::nullopt;
  }
  Value offsetStripped = ValueAnalysis::stripNumericCasts(offset);
  ARTS_DEBUG("  offsetStripped="
             << offsetStripped
             << " isZero=" << ValueAnalysis::isZeroConstant(offsetStripped));

  int64_t constVal;
  if (ValueAnalysis::getConstantIndex(offsetStripped, constVal)) {
    ARTS_DEBUG("  -> returning none (constant offset "
               << constVal << " cannot be partition variable)");
    return std::nullopt;
  }

  int64_t offsetConst = 0;
  Value offsetBase =
      ValueAnalysis::stripConstantOffset(offsetStripped, &offsetConst);
  if (offsetBase && offsetBase != offsetStripped) {
    ARTS_DEBUG("  normalized offset base=" << offsetBase
                                           << " const=" << offsetConst);
    offsetStripped = offsetBase;
  }

  Value normalizedOffset = normalizePartitionOffsetRoot(offsetStripped);
  auto matchesPartitionOffset = [&](Value candidate) -> bool {
    if (!candidate || !normalizedOffset)
      return false;
    if (getPartitionRelativeConstantOffset(candidate, offsetStripped))
      return true;
    Value normalizedCandidate = ValueAnalysis::stripNumericCasts(
        ValueAnalysis::stripSelectClamp(candidate));
    Value normalized = ValueAnalysis::stripNumericCasts(normalizedOffset);
    if (ValueAnalysis::sameValue(normalizedCandidate, normalized) ||
        ValueAnalysis::areValuesEquivalent(normalizedCandidate, normalized))
      return true;
    return ValueAnalysis::dependsOn(normalizedCandidate, normalized);
  };

  auto dependsOnPartitionOffset = [&](Value candidate,
                                      Value &dependencyRoot) -> bool {
    dependencyRoot = Value();
    if (!candidate)
      return false;
    if (ValueAnalysis::dependsOn(candidate, offsetStripped)) {
      dependencyRoot = offsetStripped;
      return true;
    }
    if (normalizedOffset && normalizedOffset != offsetStripped &&
        ValueAnalysis::dependsOn(candidate, normalizedOffset)) {
      dependencyRoot = normalizedOffset;
      return true;
    }
    return false;
  };

  DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
  MemoryAccessClassifier::collectAccessOperations(node, dbRefToMemOps);

  if (dbRefToMemOps.empty()) {
    ARTS_DEBUG("  -> returning none (no visible memory operations)");
    return std::nullopt;
  }
  ARTS_DEBUG("  found " << dbRefToMemOps.size() << " db_ref users with memOps");

  std::optional<unsigned> foundDim;
  for (auto &[dbRef, memOps] : dbRefToMemOps) {
    for (Operation *memOp : memOps) {
      SmallVector<Value> fullChain =
          DbUtils::collectFullIndexChain(dbRef, memOp);
      if (fullChain.empty())
        continue;

      unsigned memrefStart = dbRef.getIndices().size();
      if (memrefStart >= fullChain.size())
        continue;

      for (unsigned i = 0; i < memrefStart; ++i) {
        if (ValueAnalysis::dependsOn(fullChain[i], offsetStripped)) {
          ARTS_DEBUG("  partition offset appears in db_ref index; "
                     "disabling blocked partitioning");
          return std::nullopt;
        }
      }

      int64_t firstDynPos = -1;
      int64_t matchedPos = -1;
      Value matchedIdx;
      for (unsigned i = memrefStart; i < fullChain.size(); ++i) {
        int64_t constVal = 0;
        bool isConst = ValueAnalysis::getConstantIndex(fullChain[i], constVal);
        if (!isConst && firstDynPos < 0)
          firstDynPos = static_cast<int64_t>(i - memrefStart);

        Value dependencyRoot;
        bool directDepends =
            dependsOnPartitionOffset(fullChain[i], dependencyRoot);
        bool matchedOffset =
            directDepends || matchesPartitionOffset(fullChain[i]);
        if (matchedOffset) {
          if (directDepends) {
            if (auto stride = ValueAnalysis::getOffsetStride(fullChain[i],
                                                             dependencyRoot)) {
              if (*stride != 1) {
                ARTS_DEBUG("  partition offset has stride "
                           << *stride << "; disabling blocked partitioning");
                return std::nullopt;
              }
            }
            if (arts::isIndirectIndex(fullChain[i], dependencyRoot)) {
              ARTS_DEBUG("  indirect index detected; disabling blocked "
                         "partitioning");
              return std::nullopt;
            }
          }
          int64_t dim = static_cast<int64_t>(i - memrefStart);
          if (matchedPos >= 0 && matchedPos != dim) {
            if (matchedIdx && fullChain[i] == matchedIdx) {
              ARTS_DEBUG("  same index in multiple dims (diagonal); "
                         "keeping leading dim "
                         << matchedPos);
              continue;
            }
            ARTS_DEBUG("  partition offset maps to multiple dims; "
                       "disabling blocked partitioning");
            return std::nullopt;
          }
          matchedPos = dim;
          matchedIdx = fullChain[i];
          if (requireLeading && firstDynPos >= 0 && matchedPos != firstDynPos) {
            ARTS_DEBUG("  partition offset maps to non-leading dim; "
                       "disabling blocked partitioning");
            return std::nullopt;
          }
        }
      }

      if (matchedPos < 0) {
        ARTS_DEBUG("  -> returning none (offset not in access pattern)");
        return std::nullopt;
      }

      unsigned matchedIdxPos = memrefStart + matchedPos;
      if (matchedIdxPos < fullChain.size()) {
        Value idx = fullChain[matchedIdxPos];
        int64_t constVal;
        if (!ValueAnalysis::getConstantIndex(idx, constVal)) {
          Value dependencyRoot;
          if (dependsOnPartitionOffset(idx, dependencyRoot) &&
              arts::isIndirectIndex(idx, dependencyRoot)) {
            ARTS_DEBUG("  indirect index detected on partitioned dim; "
                       "disabling blocked partitioning");
            return std::nullopt;
          }
        }
      }

      Value idxAtPos =
          matchedIdx ? matchedIdx : fullChain[memrefStart + matchedPos];
      bool dependsOnLoopInit =
          LoopNode::dependsOnLoopInitNormalized(idxAtPos, offsetStripped) ||
          (normalizedOffset && normalizedOffset != offsetStripped &&
           LoopNode::dependsOnLoopInitNormalized(idxAtPos, normalizedOffset));
      if (!dependsOnLoopInit && !matchesPartitionOffset(idxAtPos)) {
        ARTS_DEBUG("  -> returning none (offset not in access pattern)");
        return std::nullopt;
      }

      if (foundDim && *foundDim != static_cast<unsigned>(matchedPos)) {
        ARTS_DEBUG("  partition offset maps to inconsistent dims; "
                   "disabling blocked partitioning");
        return std::nullopt;
      }
      foundDim = static_cast<unsigned>(matchedPos);
    }
  }

  if (!foundDim) {
    ARTS_DEBUG("  -> returning none (no dynamic index match)");
    return std::nullopt;
  }

  ARTS_DEBUG("  -> returning dim " << *foundDim);
  return foundDim;
}

bool PartitionBoundsAnalyzer::needsFullRange(DbAcquireNode *node,
                                             Value partitionOffset) {
  DbAcquireOp acquire = node ? node->getDbAcquireOp() : DbAcquireOp();
  DbAnalysis::AcquireContractSummary contractSummary =
      buildCanonicalAcquireContractSummary(node);
  bool supportedBlockHalo = contractSummary.contract.supportsBlockHalo();
  SmallVector<unsigned, 4> ownerDims(contractSummary.partitionDims.begin(),
                                     contractSummary.partitionDims.end());
  AccessPattern accessPattern =
      contractSummary.accessPattern != AccessPattern::Unknown
          ? contractSummary.accessPattern
          : node->getAccessPattern();
  bool hasIndirectAccess = MemoryAccessClassifier::hasIndirectAccess(node);
  bool hasStores = MemoryAccessClassifier::hasStores(node);
  std::optional<unsigned> mappedDim =
      partitionOffset ? getPartitionOffsetDim(node, partitionOffset,
                                              /*requireLeading=*/false)
                      : std::nullopt;

  if (partitionOffset && contractSummary.contract.analysis.narrowableDep &&
      acquire.getMode() == ArtsMode::in && !hasIndirectAccess && mappedDim &&
      llvm::is_contained(ownerDims, *mappedDim)) {
    ARTS_DEBUG("  needsFullRange: honoring canonical narrowable dependency "
               "slice");
    return false;
  }

  if (partitionOffset) {
    if (mappedDim) {
      DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
      MemoryAccessClassifier::collectAccessOperations(node, dbRefToMemOps);
      bool indirectInPartitionedDim = false;
      for (auto &[dbRef, memOps] : dbRefToMemOps) {
        for (Operation *memOp : memOps) {
          SmallVector<Value> fullChain =
              DbUtils::collectFullIndexChain(dbRef, memOp);
          unsigned memrefStart = dbRef.getIndices().size();
          unsigned idxPos = memrefStart + *mappedDim;
          if (idxPos >= fullChain.size()) {
            indirectInPartitionedDim = true;
            break;
          }
          Value idx = fullChain[idxPos];
          int64_t constVal;
          if (!ValueAnalysis::getConstantIndex(idx, constVal) &&
              arts::isIndirectIndex(idx, partitionOffset)) {
            indirectInPartitionedDim = true;
            break;
          }
        }
        if (indirectInPartitionedDim)
          break;
      }
      if (indirectInPartitionedDim) {
        ARTS_DEBUG("  needsFullRange: indirect access in partitioned dim");
        return true;
      }
    } else if (hasIndirectAccess) {
      ARTS_DEBUG("  needsFullRange: indirect access detected");
      return true;
    }
  } else if (hasIndirectAccess) {
    ARTS_DEBUG("  needsFullRange: indirect access detected");
    return true;
  }

  if (node->getHasNonConstantOffset()) {
    ARTS_DEBUG("  needsFullRange: non-constant offset in partitioned dim");
    return true;
  }

  if (!hasStores && accessPattern == AccessPattern::Stencil &&
      partitionOffset) {
    if (!getPartitionOffsetDim(node, partitionOffset,
                               /*requireLeading=*/true)) {
      if (supportedBlockHalo && mappedDim &&
          llvm::is_contained(ownerDims, *mappedDim)) {
        ARTS_DEBUG("  needsFullRange: preserving non-leading stencil dim from "
                   "N-D halo contract");
      } else if (canUseReadOnlyStencilHaloWindow(node, mappedDim,
                                                 partitionOffset)) {
        ARTS_DEBUG("  needsFullRange: preserving read-only stencil halo on "
                   "non-leading partition dim");
      } else if (mappedDim) {
        /// Orthogonal stencil: stencil dims differ from partition dim, so
        /// stencil does not cross boundaries and full range not needed.
        llvm::DenseSet<unsigned> stencilDims =
            collectStencilDims(node, partitionOffset);
        if (!stencilDims.empty() && !stencilDims.contains(*mappedDim)) {
          ARTS_DEBUG("  needsFullRange: stencil dims "
                     << "are orthogonal to partition dim " << *mappedDim
                     << "; full range not needed");
        } else {
          ARTS_DEBUG(
              "  needsFullRange: stencil access on non-leading partition dim");
          return true;
        }
      } else {
        ARTS_DEBUG(
            "  needsFullRange: stencil access on non-leading partition dim");
        return true;
      }
    }
  }

  if (!partitionOffset || !canPartitionWithOffset(node, partitionOffset)) {
    if (partitionOffset && supportedBlockHalo && mappedDim &&
        llvm::is_contained(ownerDims, *mappedDim)) {
      ARTS_DEBUG("  needsFullRange: trusting N-D halo contract for partition "
                 "offset");
      return false;
    }
    if (canUseReadOnlyStencilHaloWindow(node, mappedDim, partitionOffset)) {
      ARTS_DEBUG("  needsFullRange: trusting graph-derived read-only stencil "
                 "halo on partition dim");
      return false;
    }
    ARTS_DEBUG("  needsFullRange: "
               << (partitionOffset ? "partition offset not in access pattern"
                                   : "no partition offset (needs full range)"));
    return true;
  }

  return false;
}

bool PartitionBoundsAnalyzer::shouldPreserveDistributedContract(
    DbAcquireNode *node, Value partitionOffset) {
  if (!node || !partitionOffset)
    return false;

  DbAcquireOp acquire = node->getDbAcquireOp();
  if (!acquire)
    return false;

  auto acquireMode = getPartitionMode(acquire.getOperation());
  bool explicitBlockMode = acquireMode && usesBlockLayout(*acquireMode);
  if (!explicitBlockMode)
    return false;

  auto [edt, blockArg] = EdtUtils::getBlockArgumentForAcquire(acquire);
  auto distributionKind =
      edt ? getEdtDistributionKind(edt.getOperation()) : std::nullopt;
  auto distributionPattern =
      edt ? getEdtDistributionPattern(edt.getOperation()) : std::nullopt;
  bool hasDistributionContract =
      edt && (distributionKind || distributionPattern);
  if (!hasDistributionContract)
    return false;

  auto ptrTy = dyn_cast<MemRefType>(acquire.getPtr().getType());
  if (!ptrTy)
    return false;

  bool selectsNestedDb = isa<MemRefType>(ptrTy.getElementType());
  bool selectsLeafDb = !selectsNestedDb;
  DbAnalysis::AcquireContractSummary contractSummary =
      buildCanonicalAcquireContractSummary(node);
  bool readOnlyNestedStencilAcquire =
      acquire.getMode() == ArtsMode::in && selectsNestedDb &&
      (contractSummary.accessPattern == AccessPattern::Stencil ||
       contractSummary.usesStencilSemantics()) &&
      !node->hasIndirectAccess() && !node->hasStores();
  auto selectsConstantLeafFromNestedDb = [&]() {
    if (!blockArg || !selectsNestedDb)
      return false;

    bool sawDbRef = false;
    for (Operation *user : blockArg.getUsers()) {
      if (isa<DbReleaseOp>(user))
        continue;

      auto ref = dyn_cast<DbRefOp>(user);
      if (!ref || ref.getSource() != blockArg)
        return false;

      sawDbRef = true;
      for (Value idx : ref.getIndices()) {
        int64_t constantIdx = 0;
        if (!ValueAnalysis::getConstantIndex(idx, constantIdx) ||
            constantIdx != 0) {
          return false;
        }
      }
    }

    return sawDbRef;
  };
  bool hasExplicitOwnerDims =
      !contractSummary.contract.spatial.ownerDims.empty();
  /// ForLowering/H2 rewrites uniform worker tasks to consume an EDT-local
  /// nested DB pointer. After that rewrite the outer block offset no longer
  /// appears in the inner memref access expressions, so raw offset matching
  /// cannot rediscover the worker-local ownership. Keep the explicit block
  /// contract narrow: only task-scoped uniform workers with explicit owner
  /// dims and a single EDT-local leaf selection.
  bool nestedUniformWorkerAcquire =
      edt && edt.getType() == EdtType::task && hasExplicitOwnerDims &&
      selectsConstantLeafFromNestedDb() && !node->hasIndirectAccess() &&
      distributionKind && *distributionKind == EdtDistributionKind::block &&
      distributionPattern &&
      *distributionPattern == EdtDistributionPattern::uniform;

  if (!(selectsLeafDb || readOnlyNestedStencilAcquire ||
        nestedUniformWorkerAcquire))
    return false;

  auto mappedDim =
      getPartitionOffsetDim(node, partitionOffset, /*requireLeading=*/false);
  if (nestedUniformWorkerAcquire)
    return !mappedDim || !canPartitionWithOffset(node, partitionOffset);

  return !mappedDim;
}
