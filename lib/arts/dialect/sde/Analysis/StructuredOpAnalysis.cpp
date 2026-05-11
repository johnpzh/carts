///==========================================================================///
/// File: StructuredOpAnalysis.cpp
///
/// Reusable structural analysis for SDE scheduling-unit loops.
///==========================================================================///

#include "arts/dialect/sde/Analysis/StructuredOpAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

namespace mlir::arts::sde {
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
    for (Operation *op : ops) {
      if (auto nestedFor = dyn_cast<scf::ForOp>(op)) {
        if (innerFor) {
          innerFor = nullptr;
          break;
        }
        innerFor = nestedFor;
        continue;
      }
      if (!isMemoryEffectFree(op)) {
        innerFor = nullptr;
        break;
      }
    }
    if (innerFor) {
      info.ivs.push_back(innerFor.getInductionVar());
      return collectInner(*innerFor.getBody(), info);
    }
  }

  info.innermostBody = &body;
  return true;
}

static Block &getComputeBlock(Block &body) {
  // Delegate to the shared helper that looks through cu_region <parallel>.
  // We need the SdeSuIterateOp to call the shared helper, but this function
  // is called from contexts where we only have the block. Inline the same
  // logic to avoid changing all callers.
  for (Operation &inner : body.without_terminator()) {
    if (auto cuRegion = dyn_cast<SdeCuRegionOp>(inner);
        cuRegion && cuRegion.getKind() == SdeCuKind::parallel)
      return cuRegion.getBody().front();
  }
  return body;
}

static bool collectPerfectNest(SdeSuIterateOp iterOp, LoopNestInfo &info) {
  info.rootIterOp = iterOp;

  Region &region = iterOp.getBody();
  if (region.empty() || region.front().getNumArguments() == 0)
    return false;

  llvm::append_range(info.ivs, region.front().getArguments());
  return collectInner(getComputeBlock(region.front()), info);
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

static bool
collectMemrefAccessesImpl(Block &body, ArrayRef<Value> ivs,
                          SmallVectorImpl<MemrefAccessEntry> &reads,
                          SmallVectorImpl<MemrefAccessEntry> &writes,
                          MLIRContext *ctx, bool &sawAccess) {
  for (auto &op : body) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
      // Skip pointer-to-memref wrapper loads (e.g., memref.load %wrapper[] :
      // memref<memref<?xi32>>). These are pointer dereferences, not data
      // accesses.  Including them would create linalg.generic inputs with
      // memref-of-memref types, which cannot be raised to tensors.
      if (isa<MemRefType>(loadOp.getResult().getType()))
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

      auto map = tryBuildIndexingMap(storeOp.getIndices(), ivs, ctx);
      if (!map)
        return false;
      writes.push_back({storeOp.getMemref(), *map, storeOp.getOperation(),
                        /*isRead=*/false});
      sawAccess = true;
      continue;
    }

    if (auto extractOp = dyn_cast<tensor::ExtractOp>(&op)) {
      // Skip extracts that return tensor values (nested tensor types).
      if (isa<TensorType>(extractOp.getResult().getType()))
        continue;

      auto map = tryBuildIndexingMap(extractOp.getIndices(), ivs, ctx);
      if (!map)
        return false;
      reads.push_back({extractOp.getTensor(), *map, extractOp.getOperation(),
                       /*isRead=*/true});
      sawAccess = true;
      continue;
    }

    if (auto insertOp = dyn_cast<tensor::InsertOp>(&op)) {
      auto map = tryBuildIndexingMap(insertOp.getIndices(), ivs, ctx);
      if (!map)
        return false;
      writes.push_back({insertOp.getDest(), *map, insertOp.getOperation(),
                        /*isRead=*/false});
      sawAccess = true;
      continue;
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      bool thenSawAccess = false;
      if (!collectMemrefAccessesImpl(ifOp.getThenRegion().front(), ivs, reads,
                                     writes, ctx, thenSawAccess))
        return false;
      sawAccess |= thenSawAccess;

      if (!ifOp.getElseRegion().empty()) {
        bool elseSawAccess = false;
        if (!collectMemrefAccessesImpl(ifOp.getElseRegion().front(), ivs, reads,
                                       writes, ctx, elseSawAccess))
          return false;
        sawAccess |= elseSawAccess;
      }
      continue;
    }

    if (auto nestedFor = dyn_cast<scf::ForOp>(&op)) {
      bool nestedSawAccess = false;
      if (!collectMemrefAccessesImpl(*nestedFor.getBody(), ivs, reads, writes,
                                     ctx, nestedSawAccess))
        return false;
      sawAccess |= nestedSawAccess;
      continue;
    }

    if (isMemoryEffectFree(&op))
      continue;

    return false;
  }

  return true;
}

static bool collectMemrefAccesses(Block &body, ArrayRef<Value> ivs,
                                  SmallVectorImpl<MemrefAccessEntry> &reads,
                                  SmallVectorImpl<MemrefAccessEntry> &writes,
                                  MLIRContext *ctx) {
  bool sawAccess = false;
  if (!collectMemrefAccessesImpl(body, ivs, reads, writes, ctx, sawAccess))
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
      reads.size() >= 2 && !outputMaps.empty())
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

bool StructuredLoopSummary::supportsLinalgCarrier() const {
  if (classification == SdeStructuredClassification::stencil) {
    // Stencil carriers require a branch-free innermost body: no scf.if or
    // nested scf.for that would make the access pattern ambiguous.
    if (!nest.innermostBody)
      return false;
    for (Operation &op : nest.innermostBody->without_terminator()) {
      if (isa<scf::IfOp, scf::ForOp>(op))
        return false;
    }
    // Stencil carriers require static constant offsets that we can decompose
    // into shifted extract_slice views. Verify every read has dim+const form.
    for (const MemrefAccessEntry &entry : reads) {
      for (AffineExpr result : entry.indexingMap.getResults()) {
        auto dimOffset = extractDimOffset(result);
        if (!dimOffset || !dimOffset->dim)
          return false;
      }
    }
    // Also require at least one write with all-identity maps.
    for (const MemrefAccessEntry &entry : writes) {
      for (AffineExpr result : entry.indexingMap.getResults()) {
        auto dimOffset = extractDimOffset(result);
        if (!dimOffset || !dimOffset->dim || dimOffset->offset != 0)
          return false;
      }
    }
    return true;
  }
  return classification != SdeStructuredClassification::reduction ||
         supportsReductionCarrier;
}

std::optional<StructuredLoopSummary>
analyzeStructuredLoop(SdeSuIterateOp iterOp) {
  MLIRContext *ctx = iterOp.getContext();
  StructuredLoopSummary summary;
  if (!collectPerfectNest(iterOp, summary.nest))
    return std::nullopt;
  if (!summary.nest.innermostBody || summary.nest.ivs.empty())
    return std::nullopt;

  if (!collectMemrefAccesses(*summary.nest.innermostBody, summary.nest.ivs,
                             summary.reads, summary.writes, ctx))
    return std::nullopt;

  summary.outputMaps.reserve(summary.writes.size() +
                             iterOp.getReductionAccumulators().size());
  for (const MemrefAccessEntry &write : summary.writes)
    summary.outputMaps.push_back(write.indexingMap);
  for (size_t i = 0; i < iterOp.getReductionAccumulators().size(); ++i)
    summary.outputMaps.push_back(
        AffineMap::get(summary.nest.ivs.size(), 0, {}, ctx));

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

} // namespace mlir::arts::sde
