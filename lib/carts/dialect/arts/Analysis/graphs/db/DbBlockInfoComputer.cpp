///==========================================================================///
/// File: DbBlockInfoComputer.cpp
///
/// Implementation of DbBlockInfoComputer -- block offset/size computation
/// extracted from DbAcquireNode.
///==========================================================================///

#include "carts/dialect/arts/Analysis/graphs/db/DbBlockInfoComputer.h"
#include "carts/dialect/arts/Analysis/AccessPatternAnalysis.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/dialect/arts/Analysis/graphs/db/MemoryAccessClassifier.h"
#include "carts/dialect/arts/Analysis/graphs/db/PartitionBoundsAnalyzer.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/LoopStructureUtils.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_block_info_computer);

///===----------------------------------------------------------------------===///
/// File-local helpers
///===----------------------------------------------------------------------===///

using AccessBoundsInfo = AccessBoundsResult;

/// Analyze all memory accesses to compute actual bounds relative to block base.
static AccessBoundsInfo
analyzeAccessBoundsLocal(DbAcquireNode *node, Value blockOffset, Value loopIV,
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

/// Find the best loop node for bounds analysis (same logic as in
/// PartitionBoundsAnalyzer).
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

///===----------------------------------------------------------------------===///
/// DbBlockInfoComputer implementation
///===----------------------------------------------------------------------===///

LogicalResult DbBlockInfoComputer::computeBlockInfo(DbAcquireNode *node,
                                                    Value &blockOffset,
                                                    Value &blockSize) {
  DbAcquireOp dbAcquireOp = node->getDbAcquireOp();
  ARTS_DEBUG("computeBlockInfo for acquire: " << dbAcquireOp);

  auto cached = node->getComputedBlockInfo();
  if (cached) {
    blockOffset = cached->first;
    blockSize = cached->second;
    ARTS_DEBUG("  using cached block info");
    return success();
  }

  auto [partitionOffset, partitionSize] = node->getPartitionInfo();

  auto fullRangeFallback = [&](StringRef reason) -> LogicalResult {
    DbAllocNode *allocNode = node->getRootAlloc();
    if (!allocNode) {
      ARTS_DEBUG("  no allocation node for full-range fallback");
      return failure();
    }
    DbAllocOp allocOp = allocNode->getDbAllocOp();
    if (!allocOp || allocOp.getElementSizes().empty()) {
      ARTS_DEBUG("  missing element sizes for full-range fallback");
      return failure();
    }

    OpBuilder builder(dbAcquireOp);
    Location loc = dbAcquireOp.getLoc();
    builder.setInsertionPoint(dbAcquireOp);

    blockOffset = ::mlir::carts::createZeroIndex(builder, loc);
    blockSize = allocOp.getElementSizes().front();
    ARTS_DEBUG("  full-range fallback (" << reason << ")");
    return success();
  };

  EdtOp edt = node->getEdtUser();
  if (!edt) {
    ARTS_DEBUG("  no EDT user found");
    return fullRangeFallback("no EDT user");
  }

  if (!MemoryAccessClassifier::hasMemoryAccesses(node) && !partitionOffset &&
      !partitionSize)
    return fullRangeFallback("pass-through acquire");

  AnalysisManager &AM = node->getAnalysis()->getAnalysisManager();
  LoopAnalysis &loopAnalysis = AM.getLoopAnalysis();
  SmallVector<LoopNode *> forLoopNodes;
  SmallVector<LoopNode *> whileLoopNodes;
  loopAnalysis.collectLoopsInOperation<scf::ForOp>(edt, forLoopNodes);
  loopAnalysis.collectLoopsInOperation<scf::WhileOp>(edt, whileLoopNodes);

  scf::WhileOp blockWhile;
  if (!whileLoopNodes.empty()) {
    blockWhile = cast<scf::WhileOp>(whileLoopNodes[0]->getLoopOp());
  }
  if (blockWhile)
    ARTS_DEBUG("  found scf.while at " << blockWhile.getLoc());
  if (!forLoopNodes.empty())
    ARTS_DEBUG("  found " << forLoopNodes.size() << " for-like loops in EDT");

  if (partitionOffset && partitionSize) {
    if (succeeded(computeBlockInfoFromHints(node, blockOffset, blockSize))) {
      if (!forLoopNodes.empty()) {
        Value loopOffset, loopSize;
        if (succeeded(computeBlockInfoFromForLoop(node, forLoopNodes,
                                                  loopOffset, loopSize))) {
          bool useLoopSize = false;
          int64_t hintConst = 0;
          int64_t loopConst = 0;
          bool hintIsConst =
              ValueAnalysis::getConstantIndex(blockSize, hintConst);
          bool loopIsConst =
              ValueAnalysis::getConstantIndex(loopSize, loopConst);
          bool offsetRelated = false;
          if (loopOffset && blockOffset) {
            Value loopOffStripped =
                ValueAnalysis::stripNumericCasts(loopOffset);
            Value blockOffStripped =
                ValueAnalysis::stripNumericCasts(blockOffset);
            if (loopOffStripped == blockOffStripped)
              offsetRelated = true;
            if (!offsetRelated &&
                (ValueAnalysis::dependsOn(loopOffset, blockOffset) ||
                 ValueAnalysis::dependsOn(blockOffset, loopOffset)))
              offsetRelated = true;
            if (!offsetRelated) {
              int64_t loopOffConst = 0;
              int64_t blockOffConst = 0;
              if (ValueAnalysis::getConstantIndex(loopOffStripped,
                                                  loopOffConst) &&
                  ValueAnalysis::getConstantIndex(blockOffStripped,
                                                  blockOffConst) &&
                  loopOffConst == blockOffConst)
                offsetRelated = true;
            }
          }

          bool loopSizeDependsOnOffset =
              blockOffset && ValueAnalysis::dependsOn(loopSize, blockOffset);
          bool loopOffsetIsLocalZero =
              loopOffset && ValueAnalysis::isZeroConstant(
                                ValueAnalysis::stripNumericCasts(loopOffset));

          if (loopSizeDependsOnOffset)
            useLoopSize = true;
          if (loopOffsetIsLocalZero)
            useLoopSize = true;
          if (offsetRelated && !loopIsConst)
            useLoopSize = true;
          if (offsetRelated && hintIsConst && loopIsConst &&
              hintConst != loopConst)
            useLoopSize = true;

          if (useLoopSize) {
            blockSize = loopSize;
            ARTS_DEBUG("  overriding blockSize from loops: " << blockSize);
          }
        }
        if (partitionOffset &&
            !ValueAnalysis::dependsOn(blockSize, partitionOffset)) {
          func::FuncOp func = dbAcquireOp->getParentOfType<func::FuncOp>();
          if (func) {
            OpBuilder builder(dbAcquireOp);
            Location loc = dbAcquireOp.getLoc();
            DominanceInfo domInfo(func);
            for (LoopNode *loopNode : forLoopNodes) {
              Value loopUB = loopNode->getUpperBound();
              if (!loopUB)
                continue;
              Value ubDom = ValueAnalysis::traceValueToDominating(
                  loopUB, dbAcquireOp, builder, domInfo, loc);
              if (!ubDom)
                continue;
              if (ValueAnalysis::dependsOn(ubDom, partitionOffset)) {
                blockSize = ubDom;
                ARTS_DEBUG("  overriding blockSize from loop upper bound: "
                           << blockSize);
                break;
              }
            }
          }
        }
      }
      return success();
    }
  }

  if (blockWhile) {
    Value offsetForCheck;
    if (succeeded(computeBlockInfoFromWhile(node, blockWhile, blockOffset,
                                            blockSize, &offsetForCheck))) {
      Value checkOffset = offsetForCheck ? offsetForCheck : blockOffset;
      if (!PartitionBoundsAnalyzer::canPartitionWithOffset(node, checkOffset))
        return fullRangeFallback("scf.while offset not derived from access");
      return success();
    }
    ARTS_DEBUG("  computeBlockInfoFromWhile failed");
    return fullRangeFallback("scf.while block info");
  }

  llvm::sort(forLoopNodes, [](LoopNode *a, LoopNode *b) {
    return a->getNestingDepth() < b->getNestingDepth();
  });

  return computeBlockInfoFromForLoop(node, forLoopNodes, blockOffset,
                                     blockSize);
}

LogicalResult DbBlockInfoComputer::computeBlockInfoFromWhile(
    DbAcquireNode *node, scf::WhileOp whileOp, Value &blockOffset,
    Value &blockSize, Value *offsetForCheck) {
  DbAcquireOp dbAcquireOp = node->getDbAcquireOp();
  if (!whileOp || !dbAcquireOp)
    return failure();
  Block &before = whileOp.getBefore().front();
  auto condition = dyn_cast<scf::ConditionOp>(before.getTerminator());
  if (!condition)
    return failure();

  LoopAnalysis &loopAnalysis =
      node->getAnalysis()->getAnalysisManager().getLoopAnalysis();
  LoopNode *loopNode = loopAnalysis.getOrCreateLoopNode(whileOp);
  if (!loopNode)
    return failure();

  Value loopIV = loopNode->getInductionVar();
  auto ivArg = dyn_cast<BlockArgument>(loopIV);
  if (!ivArg)
    return failure();
  if (ivArg.getOwner() != &before)
    return failure();

  unsigned ivIndex = ivArg.getArgNumber();
  if (ivIndex >= whileOp.getInits().size())
    return failure();
  Value initValue = whileOp.getInits()[ivIndex];
  if (offsetForCheck)
    *offsetForCheck = initValue;

  SmallVector<Value> whileBounds;
  arts::collectWhileBounds(condition.getCondition(), loopIV, whileBounds);
  if (whileBounds.empty())
    return failure();

  OpBuilder builder(dbAcquireOp);
  Location loc = dbAcquireOp.getLoc();
  builder.setInsertionPoint(dbAcquireOp);
  func::FuncOp func = dbAcquireOp->getParentOfType<func::FuncOp>();
  if (!func)
    return failure();
  DominanceInfo domInfo(func);

  Value initDom = ValueAnalysis::traceValueToDominating(initValue, dbAcquireOp,
                                                        builder, domInfo, loc);
  if (!initDom)
    return failure();

  Value startIdx = ValueAnalysis::ensureIndexType(initDom, builder, loc);
  if (!startIdx)
    return failure();

  SmallVector<Value> blockSizes;
  Value initStripped = ValueAnalysis::stripNumericCasts(initValue);
  for (Value bound : whileBounds) {
    Value boundStripped = ValueAnalysis::stripNumericCasts(bound);
    if (auto addOp = boundStripped.getDefiningOp<arith::AddIOp>()) {
      Value lhs = addOp.getLhs();
      Value rhs = addOp.getRhs();
      Value lhsStripped = ValueAnalysis::stripNumericCasts(lhs);
      Value rhsStripped = ValueAnalysis::stripNumericCasts(rhs);
      Value sizeCandidate;
      if (lhsStripped == initStripped)
        sizeCandidate = rhs;
      else if (rhsStripped == initStripped)
        sizeCandidate = lhs;

      if (sizeCandidate) {
        Value sizeDom = ValueAnalysis::traceValueToDominating(
            sizeCandidate, dbAcquireOp, builder, domInfo, loc);
        Value sizeIdx = ValueAnalysis::ensureIndexType(sizeDom, builder, loc);
        if (sizeIdx) {
          blockSizes.push_back(sizeIdx);
          continue;
        }
      }
    }

    Value boundDom = ValueAnalysis::traceValueToDominating(
        bound, dbAcquireOp, builder, domInfo, loc);
    Value boundIdx = ValueAnalysis::ensureIndexType(boundDom, builder, loc);
    if (!boundIdx)
      continue;
    blockSizes.push_back(
        arith::SubIOp::create(builder, loc, boundIdx, startIdx));
  }

  if (blockSizes.empty())
    return failure();

  Value minSize = blockSizes.front();
  for (Value candidate : llvm::drop_begin(blockSizes))
    minSize = arith::MinSIOp::create(builder, loc, minSize, candidate);

  Value zero = ::mlir::carts::createZeroIndex(builder, loc);
  minSize = arith::MaxSIOp::create(builder, loc, minSize, zero);

  auto [partitionOffset, partitionSize] = node->getPartitionInfo();
  std::optional<unsigned> partitionDim =
      partitionOffset ? PartitionBoundsAnalyzer::getPartitionOffsetDim(
                            node, partitionOffset, /*requireLeading=*/false)
                      : std::nullopt;
  if (!partitionDim)
    partitionDim = node->getPartitionFacts().inferSingleMappedDim();
  AccessBoundsInfo bounds =
      analyzeAccessBoundsLocal(node, initValue, loopIV, partitionDim);
  if (!bounds.valid)
    return failure();

  if (!node->getStencilBounds()) {
    node->setStencilBoundsInternal(StencilBounds::create(
        bounds.minOffset, bounds.maxOffset, bounds.isStencil, bounds.valid));
  }

  Value adjustedOffset = startIdx;
  Value adjustedSize = minSize;

  if (bounds.minOffset < 0) {
    Value absAdj =
        ::mlir::carts::createConstantIndex(builder, loc, -bounds.minOffset);
    Value cond = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::uge,
                                       startIdx, absAdj);
    Value sub = arith::SubIOp::create(builder, loc, startIdx, absAdj);
    Value zeroIdx = ::mlir::carts::createZeroIndex(builder, loc);
    adjustedOffset = arith::SelectOp::create(builder, loc, cond, sub, zeroIdx);
  } else if (bounds.minOffset > 0) {
    Value adj =
        ::mlir::carts::createConstantIndex(builder, loc, bounds.minOffset);
    adjustedOffset = arith::AddIOp::create(builder, loc, startIdx, adj);
  }

  int64_t sizeAdjustment = bounds.maxOffset - bounds.minOffset;
  if (sizeAdjustment != 0) {
    Value adjustment =
        ::mlir::carts::createConstantIndex(builder, loc, sizeAdjustment);
    adjustedSize = arith::AddIOp::create(builder, loc, minSize, adjustment);
  }

  blockOffset = adjustedOffset;
  blockSize = adjustedSize;
  return success();
}

LogicalResult DbBlockInfoComputer::computeBlockInfoFromHints(
    DbAcquireNode *node, Value &blockOffset, Value &blockSize) {
  auto [partitionOffset, partitionSize] = node->getPartitionInfo();
  DbAcquireOp dbAcquireOp = node->getDbAcquireOp();

  if (!partitionOffset || !partitionSize)
    return failure();

  ARTS_DEBUG("  computeBlockInfoFromHints: " << partitionOffset << " / "
                                             << partitionSize);
  if (!node->getOriginalBounds())
    node->setOriginalBounds(std::make_pair(partitionOffset, partitionSize));

  std::optional<unsigned> partitionDim =
      PartitionBoundsAnalyzer::getPartitionOffsetDim(node, partitionOffset,
                                                     /*requireLeading=*/false);
  if (!partitionDim)
    partitionDim = node->getPartitionFacts().inferSingleMappedDim();
  if (!partitionDim) {
    ARTS_DEBUG("  offset hints not derived from access; failing");
    return failure();
  }
  Value partitionIdx;
  Value firstDynIdx;
  DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
  MemoryAccessClassifier::collectAccessOperations(node, dbRefToMemOps);
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

  Value analysisOffset = partitionOffset;
  analysisOffset = ValueAnalysis::stripConstantOffset(analysisOffset, nullptr);

  bool offsetIsZero = ValueAnalysis::isZeroConstant(
      ValueAnalysis::stripNumericCasts(analysisOffset));
  Value loopIdx = partitionIdx ? partitionIdx : firstDynIdx;
  if (!loopIdx) {
    if (offsetIsZero) {
      blockOffset = partitionOffset;
      blockSize = partitionSize;
      return success();
    }
    return failure();
  }

  EdtOp edt = node->getEdtUser();
  AnalysisManager &AM = node->getAnalysis()->getAnalysisManager();
  LoopAnalysis &loopAnalysis = AM.getLoopAnalysis();
  SmallVector<LoopNode *> forLoopNodes;
  loopAnalysis.collectForLoopsInOperation(edt, forLoopNodes);

  LoopNode *boundsLoopNode =
      findBestLoopNode(forLoopNodes, loopIdx, analysisOffset, offsetIsZero);

  if (boundsLoopNode) {
    Value loopIV = boundsLoopNode->getInductionVar();
    AccessBoundsInfo bounds =
        analyzeAccessBoundsLocal(node, analysisOffset, loopIV, partitionDim);

    if (bounds.valid) {
      if (!node->getStencilBounds())
        node->setStencilBoundsInternal(
            StencilBounds::create(bounds.minOffset, bounds.maxOffset,
                                  bounds.isStencil, bounds.valid));

      OpBuilder builder(dbAcquireOp);
      Location loc = dbAcquireOp.getLoc();
      builder.setInsertionPoint(dbAcquireOp);

      if (bounds.minOffset != 0) {
        Value adjustment =
            ::mlir::carts::createConstantIndex(builder, loc, bounds.minOffset);
        blockOffset =
            arith::AddIOp::create(builder, loc, partitionOffset, adjustment);
      } else {
        blockOffset = partitionOffset;
      }

      int64_t sizeAdjustment = bounds.maxOffset - bounds.minOffset;
      if (sizeAdjustment != 0) {
        Value adjustment =
            ::mlir::carts::createConstantIndex(builder, loc, sizeAdjustment);
        blockSize =
            arith::AddIOp::create(builder, loc, partitionSize, adjustment);
      } else {
        blockSize = partitionSize;
      }

      return success();
    }
  }

  blockOffset = partitionOffset;
  blockSize = partitionSize;
  return success();
}

LogicalResult DbBlockInfoComputer::computeBlockInfoFromForLoop(
    DbAcquireNode *node, ArrayRef<LoopNode *> loopNodes, Value &blockOffset,
    Value &blockSize, Value *offsetForCheck) {
  DbAcquireOp dbAcquireOp = node->getDbAcquireOp();
  Value ptrArg = node->getUseInEdt();
  EdtOp edtUser = node->getEdtUser();
  if (!ptrArg) {
    ARTS_DEBUG("  computeBlockInfoFromForLoop: no EDT block argument");
    return failure();
  }
  if (!edtUser) {
    ARTS_DEBUG("  computeBlockInfoFromForLoop: no EDT user");
    return failure();
  }

  if (loopNodes.empty()) {
    ARTS_DEBUG("  computeBlockInfoFromForLoop: no scf.for found");
    return failure();
  }

  OpBuilder builder(dbAcquireOp);
  Location loc = dbAcquireOp.getLoc();
  builder.setInsertionPoint(dbAcquireOp);

  auto [partitionOffset, partitionSize] = node->getPartitionInfo();
  std::optional<unsigned> partitionDim =
      partitionOffset ? PartitionBoundsAnalyzer::getPartitionOffsetDim(
                            node, partitionOffset, /*requireLeading=*/false)
                      : std::nullopt;
  if (!partitionDim)
    partitionDim = node->getPartitionFacts().inferSingleMappedDim();

  auto getLoopUpperBound = [&](LoopNode *ln) -> Value {
    return ln ? ln->getUpperBound() : Value();
  };

  for (LoopNode *loopNode : loopNodes) {
    Value offsetCandidate;
    Operation *loopOp = loopNode->getLoopOp();
    if (!loopOp || !isa<scf::ForOp>(loopOp))
      continue;
    Value loopIV = loopNode->getInductionVar();
    auto pickCandidateFromIndex = [&](Value idx) -> Value {
      Value stripped = ValueAnalysis::stripNumericCasts(idx);
      if (stripped == loopIV)
        return ::mlir::carts::createZeroIndex(builder, loc);

      if (auto addOp = stripped.getDefiningOp<arith::AddIOp>()) {
        Value lhs = addOp.getLhs();
        Value rhs = addOp.getRhs();
        bool lhsIV = loopNode->dependsOnInductionVarNormalized(lhs);
        bool rhsIV = loopNode->dependsOnInductionVarNormalized(rhs);
        if (lhsIV && !rhsIV && loopNode->isValueLoopInvariant(rhs))
          return rhs;
        if (rhsIV && !lhsIV && loopNode->isValueLoopInvariant(lhs))
          return lhs;
      }
      return Value();
    };

    for (Operation *user : ptrArg.getUsers()) {
      auto refOp = dyn_cast<DbRefOp>(user);
      if (!refOp || !loopOp->isAncestor(refOp))
        continue;

      SetVector<Operation *> memOps;
      DbUtils::collectReachableMemoryOps(refOp.getResult(), memOps,
                                         &edtUser.getBody());
      for (Operation *memUser : memOps) {
        if (!loopOp->isAncestor(memUser))
          continue;

        SmallVector<Value> memIndices =
            DbUtils::getMemoryAccessIndices(memUser);
        if (memIndices.empty())
          continue;
        Value idx = memIndices.front();
        if (partitionDim && *partitionDim < memIndices.size())
          idx = memIndices[*partitionDim];
        offsetCandidate = pickCandidateFromIndex(idx);

        if (offsetCandidate)
          break;
      }

      if (!offsetCandidate) {
        for (Value refIdx : refOp.getIndices()) {
          offsetCandidate = pickCandidateFromIndex(refIdx);
          if (offsetCandidate)
            break;
        }
      }
      if (offsetCandidate)
        break;
    }

    if (!offsetCandidate) {
      ARTS_DEBUG("  no offset candidate found in loop at " << loopOp->getLoc());
      continue;
    }

    AccessBoundsInfo bounds =
        analyzeAccessBoundsLocal(node, offsetCandidate, loopIV, partitionDim);
    if (!bounds.valid) {
      ARTS_DEBUG("  bounds analysis failed for loop at " << loopOp->getLoc());
      continue;
    }

    if (!node->getStencilBounds())
      node->setStencilBoundsInternal(StencilBounds::create(
          bounds.minOffset, bounds.maxOffset, bounds.isStencil, bounds.valid));

    func::FuncOp func = dbAcquireOp->getParentOfType<func::FuncOp>();
    if (!func)
      continue;
    DominanceInfo domInfo(func);

    Value offsetDom = ValueAnalysis::traceValueToDominating(
        offsetCandidate, dbAcquireOp, builder, domInfo, loc);
    Value loopUpperBound = getLoopUpperBound(loopNode);
    if (!loopUpperBound)
      continue;
    Value sizeDom = ValueAnalysis::traceValueToDominating(
        loopUpperBound, dbAcquireOp, builder, domInfo, loc);
    if (!offsetDom || !sizeDom) {
      ARTS_DEBUG("  offset/size does not dominate for loop at "
                 << loopOp->getLoc());
      continue;
    }

    Value adjustedOffset =
        ValueAnalysis::ensureIndexType(offsetDom, builder, loc);
    Value adjustedSize = ValueAnalysis::ensureIndexType(sizeDom, builder, loc);
    if (!adjustedOffset || !adjustedSize)
      continue;

    if (bounds.minOffset < 0) {
      Value absAdj =
          ::mlir::carts::createConstantIndex(builder, loc, -bounds.minOffset);
      Value cond = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::uge, adjustedOffset, absAdj);
      Value sub = arith::SubIOp::create(builder, loc, adjustedOffset, absAdj);
      Value zeroIdx = ::mlir::carts::createZeroIndex(builder, loc);
      adjustedOffset =
          arith::SelectOp::create(builder, loc, cond, sub, zeroIdx);
    } else if (bounds.minOffset > 0) {
      Value adj =
          ::mlir::carts::createConstantIndex(builder, loc, bounds.minOffset);
      adjustedOffset = arith::AddIOp::create(builder, loc, adjustedOffset, adj);
    }

    int64_t sizeAdjustment = bounds.maxOffset - bounds.minOffset;
    if (sizeAdjustment != 0) {
      Value adjustment =
          ::mlir::carts::createConstantIndex(builder, loc, sizeAdjustment);
      adjustedSize =
          arith::AddIOp::create(builder, loc, adjustedSize, adjustment);
    }

    blockOffset = adjustedOffset;
    blockSize = adjustedSize;
    if (offsetForCheck)
      *offsetForCheck = offsetCandidate;
    ARTS_DEBUG("  selected loop " << loopOp->getLoc() << " blockOffset="
                                  << blockOffset << " blockSize=" << blockSize);

    Value checkOffset = offsetForCheck ? *offsetForCheck : blockOffset;
    std::optional<unsigned> resolvedDim =
        PartitionBoundsAnalyzer::getPartitionOffsetDim(
            node, checkOffset, /*requireLeading=*/false);
    if (!resolvedDim) {
      Value normalizedCheck = ValueAnalysis::stripNumericCasts(checkOffset);
      if (!partitionDim || !ValueAnalysis::isZeroConstant(normalizedCheck)) {
        ARTS_DEBUG("  refined loop-local offset does not map back to a "
                   "partition dim");
        return failure();
      }
      /// Local single-block loops often rewrite the owned slice to zero-based
      /// coordinates. The original acquire hint already identified the
      /// partitioned dimension, so reuse that dimension instead of rejecting a
      /// valid local size refinement just because the localized offset became
      /// zero.
      resolvedDim = partitionDim;
    }
    return success();
  }

  ARTS_DEBUG("  no suitable scf.for loop found for block info");
  return failure();
}
