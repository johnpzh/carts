///==========================================================================///
/// File: StructuredOpAnalysis.cpp
///
/// Reusable structural analysis for SDE scheduling-unit loops.
///==========================================================================///

#include "carts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "carts/dialect/sde/Analysis/SdeAnalysisUtils.h"

#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::carts;

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
  Value root = ::mlir::carts::ValueAnalysis::stripMemrefViewOps(value);
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

static bool isUsedInsideAny(ArrayRef<Operation *> ancestors, Value value) {
  if (!value)
    return false;
  for (Operation *ancestor : ancestors)
    if (isUsedInside(ancestor, value))
      return true;
  return false;
}

// Polygeist can preserve C local scratch allocation as libc calls instead of
// memref.alloc.  The analyzer may ignore those effects only for exact
// allocator/free pairs whose result cannot escape the analyzed body.
static bool isLibcAllocatorCallee(StringRef callee) {
  return callee == "malloc" || callee == "calloc" || callee == "aligned_alloc";
}

static bool isLibcFreeCallee(StringRef callee) { return callee == "free"; }

static bool hasBaseMemrefType(Value value) {
  return value && isa<BaseMemRefType>(value.getType());
}

static Value stripMemrefViews(Value value) {
  return ::mlir::carts::ValueAnalysis::stripMemrefViewOps(value);
}

static bool sameMemrefRoot(Value lhs, Value rhs) {
  return stripMemrefViews(lhs) == stripMemrefViews(rhs);
}

static bool isLibcFreeCallUsing(func::CallOp call, Value root) {
  return call && isLibcFreeCallee(call.getCallee()) &&
         call->getNumOperands() == 1 && call.getNumResults() == 0 &&
         sameMemrefRoot(call.getOperand(0), root);
}

static bool isLocalLibcAllocatorRoot(Value root, Block &scope) {
  root = stripMemrefViews(root);
  if (!hasBaseMemrefType(root))
    return false;

  auto call = root.getDefiningOp<func::CallOp>();
  return call && call->getBlock() == &scope &&
         isLibcAllocatorCallee(call.getCallee());
}

static bool hasLocalLibcFreeUse(Value root) {
  root = stripMemrefViews(root);
  for (OpOperand &use : root.getUses())
    if (auto call = dyn_cast<func::CallOp>(use.getOwner()))
      if (isLibcFreeCallUsing(call, root))
        return true;
  return false;
}

static bool opIsInsideAny(ArrayRef<Operation *> ancestors, Operation *op) {
  if (!op)
    return false;
  for (Operation *ancestor : ancestors)
    if (ancestor && ancestor->isAncestor(op))
      return true;
  return false;
}

static bool onlyUsedByRegionsOrLocalFree(Value value,
                                         ArrayRef<Operation *> regions,
                                         Block &scope, Value root,
                                         llvm::SmallPtrSetImpl<Value> &seen) {
  if (!value || !seen.insert(value).second)
    return true;

  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (!owner || owner->hasTrait<OpTrait::IsTerminator>())
      return false;

    if (auto storeOp = dyn_cast<memref::StoreOp>(owner))
      if (sameMemrefRoot(storeOp.getValueToStore(), root))
        return false;

    if (auto call = dyn_cast<func::CallOp>(owner)) {
      if (!isLibcFreeCallUsing(call, root))
        return false;
      if (call->getBlock() != &scope && !opIsInsideAny(regions, owner))
        return false;
      continue;
    }

    bool hasDerivedMemrefResult = false;
    for (Value result : owner->getResults())
      hasDerivedMemrefResult |= hasBaseMemrefType(result);
    if (hasDerivedMemrefResult) {
      if (!isMemoryEffectFree(owner))
        return false;
      for (Value result : owner->getResults()) {
        if (!hasBaseMemrefType(result))
          continue;
        if (!onlyUsedByRegionsOrLocalFree(result, regions, scope, root, seen))
          return false;
      }
      continue;
    }

    if (!opIsInsideAny(regions, owner))
      return false;
  }

  return true;
}

static bool onlyUsedByRegionsOrLocalFree(Value root,
                                         ArrayRef<Operation *> regions,
                                         Block &scope) {
  root = stripMemrefViews(root);
  llvm::SmallPtrSet<Value, 8> seen;
  return onlyUsedByRegionsOrLocalFree(root, regions, scope, root, seen);
}

static bool isLocalLibcAllocatorScratch(Value root, Block &scope,
                                        ArrayRef<Operation *> regions) {
  root = stripMemrefViews(root);
  return isLocalLibcAllocatorRoot(root, scope) &&
         isUsedInsideAny(regions, root) && hasLocalLibcFreeUse(root) &&
         onlyUsedByRegionsOrLocalFree(root, regions, scope);
}

static bool isLocalLibcAllocatorScratchCall(Operation *op, Block &scope,
                                            ArrayRef<Operation *> regions) {
  auto call = dyn_cast_or_null<func::CallOp>(op);
  if (!call || call->getBlock() != &scope ||
      !isLibcAllocatorCallee(call.getCallee()) || call.getNumResults() == 0)
    return false;

  for (Value result : call.getResults())
    if (!isLocalLibcAllocatorScratch(result, scope, regions))
      return false;
  return true;
}

static bool isLocalLibcFreeScratchCall(Operation *op, Block &scope,
                                       ArrayRef<Operation *> regions) {
  auto call = dyn_cast_or_null<func::CallOp>(op);
  if (!isLibcFreeCallUsing(call, call && call->getNumOperands() == 1
                                     ? stripMemrefViews(call.getOperand(0))
                                     : Value{}))
    return false;
  Value root = stripMemrefViews(call.getOperand(0));
  return isLocalLibcAllocatorScratch(root, scope, regions);
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
    Value root = ::mlir::carts::ValueAnalysis::stripMemrefViewOps(memref);
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

static bool
isLocalScratchSideEffectUsedByRegions(Operation *op, Block &scope,
                                      ArrayRef<Operation *> regions) {
  if (!op || regions.empty())
    return false;

  if (isLocalLibcAllocatorScratchCall(op, scope, regions) ||
      isLocalLibcFreeScratchCall(op, scope, regions))
    return true;

  for (Operation *region : regions)
    if (auto loop = dyn_cast_or_null<scf::ForOp>(region))
      if (isLocalScratchSideEffectUsedByLoop(op, scope, loop))
        return true;
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

static bool isEffectFreeBoundaryOp(Operation *op) {
  return isMemoryEffectFree(op) || isRankZeroScalarMemrefAccess(op);
}

static bool
boundarySideEffectsAreLocalScratch(ArrayRef<Operation *> sideEffects,
                                   Block &body, ArrayRef<Operation *> regions) {
  for (Operation *op : sideEffects)
    if (!isLocalScratchSideEffectUsedByRegions(op, body, regions))
      return false;
  return true;
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
    SmallVector<scf::ForOp, 2> nestedFors;
    SmallVector<Operation *> sideEffectsAroundInnerLoop;
    for (Operation *op : ops) {
      if (auto nestedFor = dyn_cast<scf::ForOp>(op)) {
        nestedFors.push_back(nestedFor);
        continue;
      }
      if (!isEffectFreeBoundaryOp(op))
        sideEffectsAroundInnerLoop.push_back(op);
    }

    SmallVector<Operation *, 2> nestedForOps;
    nestedForOps.reserve(nestedFors.size());
    for (scf::ForOp nestedFor : nestedFors)
      nestedForOps.push_back(nestedFor.getOperation());

    if (nestedFors.size() == 1) {
      scf::ForOp innerFor = nestedFors.front();
      bool hasUnsupportedSideEffectsAroundInnerLoop =
          !boundarySideEffectsAreLocalScratch(sideEffectsAroundInnerLoop, body,
                                              nestedForOps);
      info.ivs.push_back(innerFor.getInductionVar());
      if (hasUnsupportedSideEffectsAroundInnerLoop) {
        info.innermostBody = &body;
        return true;
      }
      return collectInner(*innerFor.getBody(), info);
    }

    if (nestedFors.size() > 1) {
      if (!boundarySideEffectsAreLocalScratch(sideEffectsAroundInnerLoop, body,
                                              nestedForOps))
        return false;
      info.innermostBody = &body;
      return true;
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

static bool isLocalLibcAllocatorScratchForBody(Operation *op, Block &body) {
  Operation *region = body.getParentOp();
  if (!region)
    return false;

  SmallVector<Operation *, 1> regions{region};
  return isLocalLibcAllocatorScratchCall(op, body, regions) ||
         isLocalLibcFreeScratchCall(op, body, regions);
}

static bool
collectMemrefAccessesImpl(Operation *scope, Block &body, ArrayRef<Value> ivs,
                          SmallVectorImpl<MemrefAccessEntry> &reads,
                          SmallVectorImpl<MemrefAccessEntry> &writes,
                          MLIRContext *ctx, bool &sawAccess) {
  for (auto &op : body) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    if (isLocalScratchEffect(&op, scope))
      continue;

    if (isLocalLibcAllocatorScratchForBody(&op, body))
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
        if (!collectMemrefAccessesImpl(scope, ifOp.getElseRegion().front(), ivs,
                                       reads, writes, ctx, elseSawAccess))
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

static bool hasCanonicalMatmulAccessShape(
    ArrayRef<MemrefAccessEntry> reads, ArrayRef<AffineMap> outputMaps,
    ArrayRef<utils::IteratorType> iterTypes, unsigned numDims) {
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
    return ::mlir::carts::ValueAnalysis::stripMemrefViewOps(value);
  return value;
}

static std::optional<SmallVector<int64_t, 4>> getStaticShape(Value value) {
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

static bool hasAnyExactOwnerSliceAccess(SdeSuIterateOp iterOp, Value root,
                                        ArrayRef<int64_t> ownerDims) {
  if (!iterOp || !root || ownerDims.empty())
    return false;

  SmallVector<Value, 4> ownerIndexValues = collectOwnerIndexValues(iterOp);
  if (ownerIndexValues.empty())
    return false;

  bool sawExactOwnerAccess = false;
  auto checkIndices = [&](Value memref, OperandRange indices) {
    Value base = normalizeOutputRoot(memref);
    if (base != root || indices.empty())
      return;

    for (int64_t rawDim : ownerDims) {
      if (rawDim < 0 || static_cast<size_t>(rawDim) >= indices.size())
        return;
      if (!isExactOwnerIndex(indices[rawDim], ownerIndexValues))
        return;
    }
    sawExactOwnerAccess = true;
  };

  iterOp.getBody().walk([&](Operation *nested) {
    if (sawExactOwnerAccess)
      return WalkResult::interrupt();
    if (auto loadOp = dyn_cast<memref::LoadOp>(nested)) {
      if (!isa<MemRefType>(loadOp.getResult().getType()))
        checkIndices(loadOp.getMemref(), loadOp.getIndices());
      return sawExactOwnerAccess ? WalkResult::interrupt()
                                 : WalkResult::advance();
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(nested)) {
      if (!isa<MemRefType>(storeOp.getValueToStore().getType()))
        checkIndices(storeOp.getMemref(), storeOp.getIndices());
      return sawExactOwnerAccess ? WalkResult::interrupt()
                                 : WalkResult::advance();
    }
    return WalkResult::advance();
  });

  return sawExactOwnerAccess;
}

static std::optional<
    std::pair<SmallVector<int64_t, 4>, SmallVector<int64_t, 4>>>
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
    if (loopToPhysical[loopDim] >= 0 || physicalToLoop[physicalDim] >= 0)
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

bool isOwnerLocalPipelineReduction(SdeSuIterateOp iterOp) {
  if (iterOp.getReductionAccumulators().size() != 0)
    return false;
  if (iterOp.getLowerBounds().size() != 1)
    return false;

  std::optional<LoopIndexedOutputPlan> outputPlan =
      findLoopIndexedOutputPlan(iterOp);
  if (!outputPlan || outputPlan->ownerPhysicalDims.empty())
    return false;

  StructuredMemoryEffectSummary effects =
      collectStructuredMemoryEffects(iterOp.getBody());
  if (effects.hasUnknownEffects || effects.writes.empty() ||
      !effects.writes.contains(outputPlan->root))
    return false;

  // Owner-local pipeline reductions may be materialized with compute-block
  // dependency views. Every external read therefore has to be provably inside
  // the same owner slice, not merely dependent on the owner IV. Triangular
  // kernels such as correlation read both data[i, *] and data[j, *] for
  // j > i; slicing those reads to the i owner block is out of bounds on
  // multinode runs.
  for (Value read : effects.reads) {
    if (isDefinedInside(iterOp.getOperation(), read))
      continue;
    if (!hasAnyExactOwnerSliceAccess(iterOp, read,
                                     outputPlan->ownerPhysicalDims))
      continue;
    if (!allRootAccessesStayWithinOwnerSlice(iterOp, read,
                                             outputPlan->ownerPhysicalDims))
      return false;
  }

  for (Value written : effects.writes) {
    if (isDefinedInside(iterOp.getOperation(), written))
      continue;
    if (!effects.reads.contains(written))
      continue;
    if (!allRootAccessesStayWithinOwnerSlice(iterOp, written,
                                             outputPlan->ownerPhysicalDims))
      return false;
  }

  return true;
}

bool hasDistinctExternalMatmulInputRoots(SdeSuIterateOp iterOp) {
  auto hasDistinctExternalReadOnlyRoots = [&]() {
    StructuredMemoryEffectSummary effects =
        collectStructuredMemoryEffects(iterOp.getBody());
    if (effects.hasUnknownEffects)
      return false;

    llvm::DenseSet<Value> inputRoots;
    iterOp.getBody().walk([&](memref::LoadOp loadOp) {
      if (isa<MemRefType>(loadOp.getResult().getType()))
        return;
      Value root = normalizeOutputRoot(loadOp.getMemref());
      if (!root || isDefinedInside(iterOp.getOperation(), root) ||
          effects.writes.contains(root))
        return;
      inputRoots.insert(root);
    });
    return inputRoots.size() >= 2;
  };

  std::optional<StructuredLoopSummary> summary = analyzeStructuredLoop(iterOp);
  if (!summary)
    return hasDistinctExternalReadOnlyRoots();

  SmallVector<unsigned, 2> parallelDims;
  SmallVector<unsigned, 1> reductionDims;
  for (auto [dim, type] : llvm::enumerate(summary->iterTypes)) {
    if (type == utils::IteratorType::parallel)
      parallelDims.push_back(dim);
    else
      reductionDims.push_back(dim);
  }
  if (parallelDims.size() != 2 || reductionDims.size() != 1 ||
      summary->nest.ivs.size() != 3)
    return hasDistinctExternalReadOnlyRoots();

  unsigned numDims = summary->nest.ivs.size();
  llvm::SmallBitVector lhsDims(numDims);
  lhsDims.set(parallelDims[0]);
  lhsDims.set(reductionDims[0]);
  llvm::SmallBitVector rhsDims(numDims);
  rhsDims.set(reductionDims[0]);
  rhsDims.set(parallelDims[1]);

  Value lhsRoot;
  Value rhsRoot;
  llvm::DenseSet<Value> externalWriteRoots;
  for (const MemrefAccessEntry &write : summary->writes) {
    Value root = normalizeOutputRoot(write.memref);
    if (root && !isDefinedInside(iterOp.getOperation(), root))
      externalWriteRoots.insert(root);
  }

  for (const MemrefAccessEntry &read : summary->reads) {
    Value root = normalizeOutputRoot(read.memref);
    if (!root || isDefinedInside(iterOp.getOperation(), root) ||
        externalWriteRoots.contains(root))
      continue;

    llvm::SmallBitVector used = getUsedDims(read.indexingMap, numDims);
    if (used == lhsDims) {
      if (!lhsRoot)
        lhsRoot = root;
      else if (lhsRoot != root)
        return false;
    }
    if (used == rhsDims) {
      if (!rhsRoot)
        rhsRoot = root;
      else if (rhsRoot != root)
        return false;
    }
  }

  if (lhsRoot && rhsRoot && lhsRoot != rhsRoot)
    return true;
  return hasDistinctExternalReadOnlyRoots();
}

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

  // If no read or write indexing map references any loop IV (every result is a
  // constant), the analyzer cannot prove what this loop is doing — typically
  // the only loads/stores we could see were leaf accesses behind opaque
  // pointer-of-pointer indirection (e.g. `float ****` function arguments).
  // Bail rather than fall through to `classifyPattern`, which would otherwise
  // mark every IV as a reduction dim and hard-block Tiling (see
  // Tiling.cpp::isTilingCandidate, `case reduction: return false`).
  auto isConstantMap = [](AffineMap map) {
    return llvm::all_of(map.getResults(), [](AffineExpr e) {
      return isa<AffineConstantExpr>(e);
    });
  };
  bool readsAllConstant =
      llvm::all_of(summary.reads, [&](const MemrefAccessEntry &e) {
        return isConstantMap(e.indexingMap);
      });
  bool writesAllConstant =
      llvm::all_of(summary.writes, [&](const MemrefAccessEntry &e) {
        return isConstantMap(e.indexingMap);
      });
  if (readsAllConstant && writesAllConstant)
    return std::nullopt;

  computeIteratorTypes(summary.nest.ivs.size(), summary.outputMaps,
                       summary.iterTypes);
  summary.classification =
      classifyPattern(summary.reads, summary.outputMaps, summary.iterTypes,
                      summary.nest.ivs.size());
  if (summary.classification == SdeStructuredClassification::reduction &&
      isOwnerLocalPipelineReduction(iterOp))
    summary.classification = SdeStructuredClassification::elementwise_pipeline;
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

    auto maps = buildLoopPhysicalDimMaps(
        write.indexingMap, summary.nest.ivs.size(), shape->size());
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
