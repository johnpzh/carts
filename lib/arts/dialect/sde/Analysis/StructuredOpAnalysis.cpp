///==========================================================================///
/// File: StructuredOpAnalysis.cpp
///
/// Reusable structural analysis for SDE scheduling-unit loops.
///==========================================================================///

#include "arts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "arts/dialect/sde/Analysis/SdeAnalysisUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "arts/utils/ValueAnalysis.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace mlir::carts::sde {
namespace {

static SmallVector<Operation *> getBodyOps(Block &body) {
  SmallVector<Operation *> ops;
  for (auto &op : body) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    if (isa<SdeYieldOp>(op))
      continue;
    ops.push_back(&op);
  }
  return ops;
}

static bool isLocalScratchMemref(Value value, Block &scope) {
  Value root = arts::ValueAnalysis::stripMemrefViewOps(value);
  if (!root)
    return false;

  Operation *def = root.getDefiningOp();
  return def && def->getBlock() == &scope;
}

static bool isUsedInside(Operation *ancestor, Value value) {
  if (!ancestor || !value)
    return false;
  for (OpOperand &use : value.getUses())
    if (ancestor->isAncestor(use.getOwner()))
      return true;
  return false;
}

static bool isLocalScratchSideEffectUsedByLoop(Operation *op, Block &scope,
                                               scf::ForOp loop) {
  if (!op)
    return false;

  if (isa<memref::AllocOp, memref::AllocaOp>(op)) {
    for (Value result : op->getResults()) {
      if (!isLocalScratchMemref(result, scope))
        return false;
      if (!isUsedInside(loop.getOperation(), result))
        return false;
    }
    return true;
  }

  auto isLoopScratchAccess = [&](Value memref) {
    Value root = arts::ValueAnalysis::stripMemrefViewOps(memref);
    return isLocalScratchMemref(root, scope) &&
           isUsedInside(loop.getOperation(), root);
  };

  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return isLoopScratchAccess(loadOp.getMemref());

  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return isLoopScratchAccess(storeOp.getMemref());

  if (auto deallocOp = dyn_cast<memref::DeallocOp>(op))
    return isLoopScratchAccess(deallocOp.getMemref());

  return false;
}

static bool isRankZeroScalarMemrefAccess(Operation *op) {
  Value memref;
  if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(op)) {
    if (isa<MemRefType>(loadOp.getResult().getType()))
      return false;
    memref = loadOp.getMemref();
  } else if (auto storeOp = dyn_cast_or_null<memref::StoreOp>(op)) {
    if (isa<MemRefType>(storeOp.getValueToStore().getType()))
      return false;
    memref = storeOp.getMemref();
  } else {
    return false;
  }

  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  return memrefType && memrefType.getRank() == 0;
}

static bool collectInner(Block &body, LoopNestInfo &info) {
  SmallVector<Operation *> ops = getBodyOps(body);

  if (ops.size() == 1) {
    if (auto innerFor = dyn_cast<scf::ForOp>(ops.front())) {
      info.ivs.push_back(innerFor.getInductionVar());
      return collectInner(*innerFor.getBody(), info);
    }
  }

  if (ops.size() > 1) {
    scf::ForOp innerFor = nullptr;
    SmallVector<Operation *> sideEffectsAroundInnerLoop;
    for (Operation *op : ops) {
      if (auto nestedFor = dyn_cast<scf::ForOp>(op)) {
        if (innerFor) {
          innerFor = nullptr;
          break;
        }
        innerFor = nestedFor;
        continue;
      }
      if (!isMemoryEffectFree(op) && !isRankZeroScalarMemrefAccess(op))
        sideEffectsAroundInnerLoop.push_back(op);
    }
    if (innerFor) {
      bool hasSideEffectsAroundInnerLoop = false;
      for (Operation *op : sideEffectsAroundInnerLoop) {
        if (!isLocalScratchSideEffectUsedByLoop(op, body, innerFor)) {
          hasSideEffectsAroundInnerLoop = true;
          break;
        }
      }
      info.ivs.push_back(innerFor.getInductionVar());
      if (hasSideEffectsAroundInnerLoop) {
        info.innermostBody = &body;
        return true;
      }
      return collectInner(*innerFor.getBody(), info);
    }
  }

  info.innermostBody = &body;
  return true;
}

static bool collectPerfectNest(SdeSuIterateOp iterOp, LoopNestInfo &info) {
  info.rootIterOp = iterOp;

  Region &region = iterOp.getBody();
  if (region.empty() || region.front().getNumArguments() == 0)
    return false;

  // SdeSuIterateOp block arguments are laid out as induction variables
  // followed by iter_args.  Only the loop-rank prefix participates in affine
  // access maps; treating carried values as IVs fabricates reduction
  // dimensions for result-bearing elementwise loops.
  unsigned numIvs = iterOp.getLowerBounds().size();
  if (region.front().getNumArguments() < numIvs)
    return false;
  for (BlockArgument arg : region.front().getArguments().take_front(numIvs))
    info.ivs.push_back(arg);
  return collectInner(*getSuIterateComputeBlock(iterOp), info);
}

static std::optional<AffineExpr>
tryGetAffineExpr(Value value, ArrayRef<Value> ivs, MLIRContext *ctx) {
  for (auto [idx, iv] : llvm::enumerate(ivs)) {
    if (value == iv)
      return getAffineDimExpr(idx, ctx);
  }

  auto *defOp = value.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (auto cst = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return getAffineConstantExpr(intAttr.getInt(), ctx);
    return std::nullopt;
  }

  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    auto lhs = tryGetAffineExpr(addOp.getLhs(), ivs, ctx);
    auto rhs = tryGetAffineExpr(addOp.getRhs(), ivs, ctx);
    if (lhs && rhs)
      return *lhs + *rhs;
  }

  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    auto lhs = tryGetAffineExpr(subOp.getLhs(), ivs, ctx);
    auto rhs = tryGetAffineExpr(subOp.getRhs(), ivs, ctx);
    if (lhs && rhs)
      return *lhs - *rhs;
  }

  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    auto lhs = tryGetAffineExpr(mulOp.getLhs(), ivs, ctx);
    auto rhs = tryGetAffineExpr(mulOp.getRhs(), ivs, ctx);
    if (lhs && rhs &&
        (isa<AffineConstantExpr>(*lhs) || isa<AffineConstantExpr>(*rhs)))
      return *lhs * *rhs;
  }

  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp))
    return tryGetAffineExpr(castOp.getIn(), ivs, ctx);

  return std::nullopt;
}

static std::optional<AffineMap> tryBuildIndexingMap(OperandRange indices,
                                                    ArrayRef<Value> ivs,
                                                    MLIRContext *ctx) {
  SmallVector<AffineExpr> exprs;
  exprs.reserve(indices.size());
  for (Value idx : indices) {
    auto expr = tryGetAffineExpr(idx, ivs, ctx);
    if (!expr)
      return std::nullopt;
    exprs.push_back(*expr);
  }
  return AffineMap::get(/*dimCount=*/ivs.size(), /*symbolCount=*/0, exprs, ctx);
}

static bool isLocalScratchAccess(Operation *scope, Value memref) {
  Value root = arts::ValueAnalysis::stripMemrefViewOps(memref);
  return root && isDefinedInside(scope, root);
}

static bool isLocalScratchAllocation(Operation *scope, Operation *op) {
  if (!op || !isa<memref::AllocOp, memref::AllocaOp>(op))
    return false;
  return llvm::all_of(op->getResults(), [&](Value result) {
    return isLocalScratchAccess(scope, result);
  });
}

static bool collectMemrefAccessesImpl(
    Operation *scope, Block &body, ArrayRef<Value> ivs,
    SmallVectorImpl<MemrefAccessEntry> &reads,
    SmallVectorImpl<MemrefAccessEntry> &writes, MLIRContext *ctx,
    bool &sawAccess) {
  for (auto &op : body) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    if (isLocalScratchAllocation(scope, &op))
      continue;

    if (auto deallocOp = dyn_cast<memref::DeallocOp>(&op))
      if (isLocalScratchAccess(scope, deallocOp.getMemref()))
        continue;

    if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
      // Skip pointer-to-memref wrapper loads (e.g., memref.load %wrapper[] :
      // memref<memref<?xi32>>). These are pointer dereferences, not data
      // accesses.  Including them would create linalg.generic inputs with
      // memref-of-memref types, which cannot be raised to tensors.
      if (isa<MemRefType>(loadOp.getResult().getType()))
        continue;
      auto memrefType = dyn_cast<MemRefType>(loadOp.getMemref().getType());
      if (memrefType && memrefType.getRank() == 0)
        continue;
      if (isLocalScratchAccess(scope, loadOp.getMemref()))
        continue;

      auto map = tryBuildIndexingMap(loadOp.getIndices(), ivs, ctx);
      if (!map)
        return false;
      reads.push_back(
          {loadOp.getMemref(), *map, loadOp.getOperation(), /*isRead=*/true});
      sawAccess = true;
      continue;
    }

    if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
      // Skip stores to pointer-to-memref wrappers.
      if (isa<MemRefType>(storeOp.getValueToStore().getType()))
        continue;
      auto memrefType = dyn_cast<MemRefType>(storeOp.getMemref().getType());
      if (memrefType && memrefType.getRank() == 0)
        continue;
      if (isLocalScratchAccess(scope, storeOp.getMemref()))
        continue;

      auto map = tryBuildIndexingMap(storeOp.getIndices(), ivs, ctx);
      if (!map)
        return false;
      writes.push_back({storeOp.getMemref(), *map, storeOp.getOperation(),
                        /*isRead=*/false});
      sawAccess = true;
      continue;
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      bool thenSawAccess = false;
      if (!collectMemrefAccessesImpl(scope, ifOp.getThenRegion().front(), ivs,
                                     reads, writes, ctx, thenSawAccess))
        return false;
      sawAccess |= thenSawAccess;

      if (!ifOp.getElseRegion().empty()) {
        bool elseSawAccess = false;
        if (!collectMemrefAccessesImpl(scope, ifOp.getElseRegion().front(),
                                       ivs, reads, writes, ctx,
                                       elseSawAccess))
          return false;
        sawAccess |= elseSawAccess;
      }
      continue;
    }

    if (auto nestedFor = dyn_cast<scf::ForOp>(&op)) {
      bool nestedSawAccess = false;
      if (!collectMemrefAccessesImpl(scope, *nestedFor.getBody(), ivs, reads,
                                     writes, ctx, nestedSawAccess))
        return false;
      sawAccess |= nestedSawAccess;
      continue;
    }

    if (isKnownPureScalarLibmCall(&op))
      continue;

    if (isMemoryEffectFree(&op))
      continue;

    return false;
  }

  return true;
}

static bool collectMemrefAccesses(Operation *scope, Block &body,
                                  ArrayRef<Value> ivs,
                                  SmallVectorImpl<MemrefAccessEntry> &reads,
                                  SmallVectorImpl<MemrefAccessEntry> &writes,
                                  MLIRContext *ctx) {
  bool sawAccess = false;
  if (!collectMemrefAccessesImpl(scope, body, ivs, reads, writes, ctx,
                                 sawAccess))
    return false;
  return sawAccess;
}

static void
computeIteratorTypes(unsigned numDims, ArrayRef<AffineMap> outputMaps,
                     SmallVectorImpl<utils::IteratorType> &iterTypes) {
  llvm::SmallBitVector dimsInOutputs(numDims, false);
  for (AffineMap map : outputMaps) {
    for (unsigned i = 0; i < map.getNumResults(); ++i) {
      map.getResult(i).walk([&](AffineExpr expr) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
          dimsInOutputs.set(dimExpr.getPosition());
      });
    }
  }

  for (unsigned d = 0; d < numDims; ++d) {
    iterTypes.push_back(dimsInOutputs.test(d) ? utils::IteratorType::parallel
                                              : utils::IteratorType::reduction);
  }
}

// AffineDimOffset and extractDimOffset are defined below the anonymous
// namespace as public functions declared in StructuredOpAnalysis.h.

static void appendNestedForIvs(Block &body, SmallVectorImpl<Value> &ivs) {
  llvm::SmallPtrSet<Value, 8> seen;
  for (Value iv : ivs)
    seen.insert(iv);

  body.walk([&](scf::ForOp loop) {
    Value iv = loop.getInductionVar();
    if (seen.insert(iv).second)
      ivs.push_back(iv);
  });
}

static std::optional<StructuredNeighborhoodInfo>
extractNeighborhoodSummary(ArrayRef<MemrefAccessEntry> reads,
                           unsigned numLoops) {
  if (numLoops == 0)
    return std::nullopt;

  StructuredNeighborhoodInfo info;
  info.minOffsets.assign(numLoops, 0);
  info.maxOffsets.assign(numLoops, 0);
  info.writeFootprint.assign(numLoops, 1);
  for (unsigned dim = 0; dim < numLoops; ++dim) {
    info.ownerDims.push_back(dim);
    info.spatialDims.push_back(dim);
  }

  bool sawNeighborhoodOffset = false;
  for (const MemrefAccessEntry &entry : reads) {
    for (AffineExpr result : entry.indexingMap.getResults()) {
      auto dimOffset = extractDimOffset(result);
      if (!dimOffset || !dimOffset->dim)
        continue;

      unsigned dim = *dimOffset->dim;
      if (dim >= numLoops)
        continue;

      info.minOffsets[dim] = std::min(info.minOffsets[dim], dimOffset->offset);
      info.maxOffsets[dim] = std::max(info.maxOffsets[dim], dimOffset->offset);
      sawNeighborhoodOffset |= dimOffset->offset != 0;
    }
  }

  if (!sawNeighborhoodOffset)
    return std::nullopt;
  return info;
}

static llvm::SmallBitVector getUsedDims(AffineMap map, unsigned numDims) {
  llvm::SmallBitVector used(numDims);
  for (AffineExpr result : map.getResults())
    for (unsigned dim = 0; dim < numDims; ++dim)
      if (result.isFunctionOfDim(dim))
        used.set(dim);
  return used;
}

static bool hasExactDimUse(AffineMap map, const llvm::SmallBitVector &expected,
                           unsigned numDims) {
  return getUsedDims(map, numDims) == expected;
}

static bool hasCanonicalMatmulAccessShape(ArrayRef<MemrefAccessEntry> reads,
                                          ArrayRef<AffineMap> outputMaps,
                                          ArrayRef<utils::IteratorType> iterTypes,
                                          unsigned numDims) {
  SmallVector<unsigned, 2> parallelDims;
  SmallVector<unsigned, 1> reductionDims;
  for (auto [dim, type] : llvm::enumerate(iterTypes)) {
    if (type == utils::IteratorType::parallel)
      parallelDims.push_back(dim);
    else
      reductionDims.push_back(dim);
  }

  if (parallelDims.size() != 2 || reductionDims.size() != 1 || numDims != 3)
    return false;

  llvm::SmallBitVector outputDims(numDims);
  outputDims.set(parallelDims[0]);
  outputDims.set(parallelDims[1]);
  bool hasMatmulOutput = llvm::any_of(outputMaps, [&](AffineMap map) {
    return hasExactDimUse(map, outputDims, numDims);
  });
  if (!hasMatmulOutput)
    return false;

  llvm::SmallBitVector lhsDims(numDims);
  lhsDims.set(parallelDims[0]);
  lhsDims.set(reductionDims[0]);
  llvm::SmallBitVector rhsDims(numDims);
  rhsDims.set(reductionDims[0]);
  rhsDims.set(parallelDims[1]);

  bool hasLhs = false;
  bool hasRhs = false;
  for (const MemrefAccessEntry &read : reads) {
    hasLhs |= hasExactDimUse(read.indexingMap, lhsDims, numDims);
    hasRhs |= hasExactDimUse(read.indexingMap, rhsDims, numDims);
  }
  return hasLhs && hasRhs;
}

static SdeStructuredClassification
classifyPattern(ArrayRef<MemrefAccessEntry> reads,
                ArrayRef<AffineMap> outputMaps,
                ArrayRef<utils::IteratorType> iterTypes, unsigned numDims) {
  unsigned numParallel = 0;
  unsigned numReduction = 0;
  for (utils::IteratorType t : iterTypes) {
    if (t == utils::IteratorType::parallel)
      ++numParallel;
    else
      ++numReduction;
  }

  if (numReduction == 0) {
    for (const auto &entry : reads) {
      if (hasConstantOffsets(entry.indexingMap))
        return SdeStructuredClassification::stencil;
    }
    return SdeStructuredClassification::elementwise;
  }

  if (numParallel == 2 && numReduction == 1 && numDims == 3 &&
      reads.size() >= 2 && !outputMaps.empty() &&
      hasCanonicalMatmulAccessShape(reads, outputMaps, iterTypes, numDims))
    return SdeStructuredClassification::matmul;

  return SdeStructuredClassification::reduction;
}

static bool isConstantZeroIndexingMap(AffineMap indexingMap) {
  if (indexingMap.getNumResults() != 1)
    return false;

  auto constant = dyn_cast<AffineConstantExpr>(indexingMap.getResult(0));
  return constant && constant.getValue() == 0;
}

static bool supportsReductionCarrierSubset(SdeSuIterateOp iterOp,
                                           const LoopNestInfo &nest,
                                           ArrayRef<MemrefAccessEntry> reads,
                                           ArrayRef<MemrefAccessEntry> writes) {
  if (iterOp.getReductionAccumulators().size() != 1 || nest.ivs.size() != 1)
    return false;
  if (writes.size() != 1)
    return false;

  Value reductionAccumulator = iterOp.getReductionAccumulators().front();
  auto reductionType = dyn_cast<MemRefType>(reductionAccumulator.getType());
  if (!reductionType || reductionType.getRank() != 1)
    return false;

  const MemrefAccessEntry &write = writes.front();
  if (write.memref != reductionAccumulator ||
      !isConstantZeroIndexingMap(write.indexingMap))
    return false;

  unsigned matchingReductionReads = 0;
  for (const MemrefAccessEntry &read : reads) {
    if (read.memref != reductionAccumulator)
      continue;
    if (!isConstantZeroIndexingMap(read.indexingMap))
      return false;
    ++matchingReductionReads;
  }

  return matchingReductionReads == 1;
}

static Value normalizeOutputRoot(Value value) {
  if (!value)
    return {};
  if (isa<BaseMemRefType>(value.getType()))
    return arts::ValueAnalysis::stripMemrefViewOps(value);
  return value;
}

static std::optional<SmallVector<int64_t, 4>>
getStaticShape(Value value) {
  Value root = normalizeOutputRoot(value);
  if (!root)
    return std::nullopt;
  auto shapedType = dyn_cast<ShapedType>(root.getType());
  if (!shapedType || !shapedType.hasRank() || !shapedType.hasStaticShape())
    return std::nullopt;

  SmallVector<int64_t, 4> shape;
  shape.reserve(shapedType.getRank());
  for (int64_t dim : shapedType.getShape())
    shape.push_back(dim);
  return shape;
}

static std::optional<std::pair<SmallVector<int64_t, 4>,
                               SmallVector<int64_t, 4>>>
buildLoopPhysicalDimMaps(AffineMap map, unsigned numLoops, unsigned rank) {
  if (map.getNumResults() != rank)
    return std::nullopt;

  SmallVector<int64_t, 4> loopToPhysical(numLoops, -1);
  SmallVector<int64_t, 4> physicalToLoop(rank, -1);
  bool mappedAnyDim = false;

  for (unsigned physicalDim = 0; physicalDim < rank; ++physicalDim) {
    std::optional<AffineDimOffset> dimOffset =
        extractDimOffset(map.getResult(physicalDim));
    if (!dimOffset)
      return std::nullopt;
    if (!dimOffset->dim)
      continue;
    if (dimOffset->offset != 0)
      return std::nullopt;

    unsigned loopDim = *dimOffset->dim;
    if (loopDim >= numLoops)
      return std::nullopt;
    if (loopToPhysical[loopDim] >= 0 ||
        physicalToLoop[physicalDim] >= 0)
      return std::nullopt;
    loopToPhysical[loopDim] = static_cast<int64_t>(physicalDim);
    physicalToLoop[physicalDim] = static_cast<int64_t>(loopDim);
    mappedAnyDim = true;
  }

  if (!mappedAnyDim)
    return std::nullopt;
  return std::make_pair(std::move(loopToPhysical), std::move(physicalToLoop));
}

} // namespace

std::optional<AffineDimOffset> extractDimOffset(AffineExpr expr) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
    return AffineDimOffset{dimExpr.getPosition(), 0};
  if (auto cstExpr = dyn_cast<AffineConstantExpr>(expr))
    return AffineDimOffset{std::nullopt, cstExpr.getValue()};

  auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binExpr)
    return std::nullopt;

  auto lhs = extractDimOffset(binExpr.getLHS());
  auto rhs = extractDimOffset(binExpr.getRHS());
  if (!lhs || !rhs)
    return std::nullopt;

  switch (binExpr.getKind()) {
  case AffineExprKind::Add:
    if (lhs->dim && rhs->dim)
      return std::nullopt;
    return AffineDimOffset{lhs->dim ? lhs->dim : rhs->dim,
                           lhs->offset + rhs->offset};
  default:
    return std::nullopt;
  }
}

bool hasConstantOffsets(AffineMap map) {
  for (AffineExpr result : map.getResults()) {
    auto dimOffset = extractDimOffset(result);
    if (dimOffset && dimOffset->dim && dimOffset->offset != 0)
      return true;
  }
  return false;
}

std::optional<StructuredLoopSummary>
analyzeStructuredLoop(SdeSuIterateOp iterOp) {
  MLIRContext *ctx = iterOp.getContext();
  StructuredLoopSummary summary;
  if (!collectPerfectNest(iterOp, summary.nest))
    return std::nullopt;
  if (!summary.nest.innermostBody || summary.nest.ivs.empty())
    return std::nullopt;
  appendNestedForIvs(*summary.nest.innermostBody, summary.nest.ivs);

  if (!collectMemrefAccesses(iterOp.getOperation(), *summary.nest.innermostBody,
                             summary.nest.ivs, summary.reads, summary.writes,
                             ctx))
    return std::nullopt;

  summary.outputMaps.reserve(summary.writes.size() +
                             iterOp.getReductionAccumulators().size());
  for (const MemrefAccessEntry &write : summary.writes)
    summary.outputMaps.push_back(write.indexingMap);
  for (Value ignored : iterOp.getReductionAccumulators()) {
    (void)ignored;
    summary.outputMaps.push_back(
        AffineMap::get(summary.nest.ivs.size(), 0, {}, ctx));
  }

  if (summary.outputMaps.empty())
    return std::nullopt;

  computeIteratorTypes(summary.nest.ivs.size(), summary.outputMaps,
                       summary.iterTypes);
  summary.classification =
      classifyPattern(summary.reads, summary.outputMaps, summary.iterTypes,
                      summary.nest.ivs.size());
  summary.supportsReductionCarrier = supportsReductionCarrierSubset(
      iterOp, summary.nest, summary.reads, summary.writes);
  return summary;
}

std::optional<StructuredNeighborhoodInfo>
extractNeighborhoodSummary(const StructuredLoopSummary &summary) {
  return extractNeighborhoodSummary(summary.reads, summary.nest.ivs.size());
}

std::optional<StructuredOutputLayoutPlan>
findCompatibleOutputLayoutPlan(const StructuredLoopSummary &summary) {
  if (!summary.nest.rootIterOp || summary.writes.empty() ||
      summary.nest.ivs.empty())
    return std::nullopt;

  std::optional<StructuredOutputLayoutPlan> selected;
  Operation *rootOp = summary.nest.rootIterOp;

  for (const MemrefAccessEntry &write : summary.writes) {
    Value root = normalizeOutputRoot(write.memref);
    if (!root || isDefinedInside(rootOp, root))
      continue;

    std::optional<SmallVector<int64_t, 4>> shape = getStaticShape(root);
    if (!shape || shape->empty())
      return std::nullopt;

    auto maps = buildLoopPhysicalDimMaps(write.indexingMap,
                                         summary.nest.ivs.size(),
                                         shape->size());
    if (!maps)
      return std::nullopt;

    StructuredOutputLayoutPlan candidate;
    candidate.root = root;
    candidate.shape = std::move(*shape);
    candidate.loopDimToPhysicalDim = std::move(maps->first);
    candidate.physicalDimToLoopDim = std::move(maps->second);

    if (!selected) {
      selected = std::move(candidate);
      continue;
    }

    if (candidate.shape != selected->shape ||
        candidate.loopDimToPhysicalDim != selected->loopDimToPhysicalDim ||
        candidate.physicalDimToLoopDim != selected->physicalDimToLoopDim)
      return std::nullopt;
  }

  return selected;
}

std::optional<StructuredOutputLayoutPlan>
findCompatibleOutputLayoutPlan(SdeSuIterateOp op) {
  std::optional<StructuredLoopSummary> summary = analyzeStructuredLoop(op);
  if (!summary)
    return std::nullopt;
  return findCompatibleOutputLayoutPlan(*summary);
}

} // namespace mlir::carts::sde
