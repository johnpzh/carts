///==========================================================================///
/// File: Tiling.cpp
///
/// Cost-model-driven tiling in SDE. The current implementation focuses on
/// executable loop nests whose tensor-backed linalg carrier proves that
/// strip-mining the outer SDE loop preserves a disjoint write set. The pass
/// rewrites the surrounding sde.su_iterate so the tiled loop nest survives
/// SDE->ARTS lowering while tensor carriers remain an SDE-only concern.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_TILING
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/LoopUtils.h"
#include "arts/utils/ValueAnalysis.h"
#include "arts/utils/costs/SDECostModel.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>

using namespace mlir;
using namespace mlir::arts;

namespace {

static Value stripSimpleMemrefAlias(Value value) {
  while (auto castOp = value.getDefiningOp<memref::CastOp>())
    value = castOp.getSource();
  return value;
}

static Value getConstantIndex(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

static Value buildTripCountValue(OpBuilder &builder, Location loc,
                                 sde::SdeSuIterateOp op) {
  if (std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation()))
    return getConstantIndex(builder, loc, *tripCount);

  if (op.getLowerBounds().size() != 1 || op.getUpperBounds().size() != 1 ||
      op.getSteps().size() != 1)
    return Value();

  Value lowerBound = op.getLowerBounds().front();
  Value upperBound = op.getUpperBounds().front();
  Value step = op.getSteps().front();
  Value zero = getConstantIndex(builder, loc, 0);
  Value one = getConstantIndex(builder, loc, 1);

  Value span = arith::SubIOp::create(builder, loc, upperBound, lowerBound);

  int64_t constantStep = 0;
  Value safeStep = step;
  if (ValueAnalysis::getConstantIndex(step, constantStep)) {
    if (constantStep <= 0)
      return Value();
  } else {
    Value stepIsTooSmall = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sle, step, zero);
    safeStep = arith::SelectOp::create(builder, loc, stepIsTooSmall, one, step);
  }

  Value spanIsNegative = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::slt, span, zero);
  Value nonNegativeSpan =
      arith::SelectOp::create(builder, loc, spanIsNegative, zero, span);
  return arith::CeilDivSIOp::create(builder, loc, nonNegativeSpan, safeStep);
}

static Value buildTileIterationValue(OpBuilder &builder, Location loc,
                                     sde::SdeSuIterateOp op,
                                     sde::SDECostModel &costModel) {
  Value tripCount = buildTripCountValue(builder, loc, op);
  if (!tripCount)
    return Value();

  Value one = getConstantIndex(builder, loc, 1);
  Value workerCountValue = getConstantIndex(
      builder, loc, std::max<int64_t>(1, costModel.getWorkerCount()));
  Value minIterationsValue = getConstantIndex(
      builder, loc,
      std::max<int64_t>(1, costModel.getMinIterationsPerWorker()));

  Value clampedTripCount = arith::MaxUIOp::create(builder, loc, tripCount, one);
  Value balancedTile = arith::CeilDivUIOp::create(
      builder, loc, clampedTripCount, workerCountValue);
  Value preferredTile =
      arith::MaxUIOp::create(builder, loc, balancedTile, minIterationsValue);
  return arith::MinUIOp::create(builder, loc, preferredTile, clampedTripCount);
}

static SmallVector<Value> buildPerDimTripCounts(OpBuilder &builder,
                                                Location loc,
                                                sde::SdeSuIterateOp op) {
  SmallVector<Value> tripCounts;
  unsigned numDims = op.getLowerBounds().size();

  for (unsigned d = 0; d < numDims; ++d) {
    Value lb = op.getLowerBounds()[d];
    Value ub = op.getUpperBounds()[d];
    Value step = op.getSteps()[d];
    Value zero = getConstantIndex(builder, loc, 0);
    Value one = getConstantIndex(builder, loc, 1);

    Value span = arith::SubIOp::create(builder, loc, ub, lb);

    int64_t constantStep = 0;
    Value safeStep = step;
    if (ValueAnalysis::getConstantIndex(step, constantStep)) {
      if (constantStep <= 0)
        return {};
    } else {
      Value stepIsTooSmall = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sle, step, zero);
      safeStep =
          arith::SelectOp::create(builder, loc, stepIsTooSmall, one, step);
    }

    Value spanIsNegative = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::slt, span, zero);
    Value nonNegativeSpan =
        arith::SelectOp::create(builder, loc, spanIsNegative, zero, span);
    tripCounts.push_back(
        arith::CeilDivSIOp::create(builder, loc, nonNegativeSpan, safeStep));
  }
  return tripCounts;
}

static SmallVector<Value>
buildPerDimTileIterations(OpBuilder &builder, Location loc,
                          sde::SdeSuIterateOp op,
                          sde::SDECostModel &costModel) {
  SmallVector<Value> tripCounts = buildPerDimTripCounts(builder, loc, op);
  if (tripCounts.empty())
    return {};

  unsigned numDims = tripCounts.size();
  int workersPerDim =
      std::max(1, static_cast<int>(std::ceil(
                      std::pow(costModel.getWorkerCount(), 1.0 / numDims))));
  int64_t minIter = costModel.getMinIterationsPerWorker();

  SmallVector<Value> tileIterations;
  for (unsigned d = 0; d < numDims; ++d) {
    Value one = getConstantIndex(builder, loc, 1);
    Value workersVal = getConstantIndex(builder, loc, workersPerDim);
    Value minIterVal =
        getConstantIndex(builder, loc, std::max<int64_t>(1, minIter));
    Value clampedTrip =
        arith::MaxUIOp::create(builder, loc, tripCounts[d], one);
    Value balanced =
        arith::CeilDivUIOp::create(builder, loc, clampedTrip, workersVal);
    Value preferred =
        arith::MaxUIOp::create(builder, loc, balanced, minIterVal);
    Value tile = arith::MinUIOp::create(builder, loc, preferred, clampedTrip);
    tileIterations.push_back(tile);
  }
  return tileIterations;
}

static bool isCarrierOp(Operation &op) {
  return isa<bufferization::ToTensorOp, sde::SdeMuMemrefToTensorOp,
             tensor::EmptyOp, linalg::GenericOp, tensor::ExtractSliceOp,
             tensor::InsertSliceOp>(op);
}

static bool isScalarExecutableOp(Operation &op) {
  if (isa<memref::LoadOp, memref::StoreOp>(op))
    return true;
  return op.getNumRegions() == 0 && isMemoryEffectFree(&op);
}

static bool isExecutableInnermostBody(Block &body) {
  for (Operation &op : body.without_terminator()) {
    if (isCarrierOp(op))
      continue;
    if (!isScalarExecutableOp(op))
      return false;
  }
  return true;
}

static bool hasPerfectNestedScalarLoopNest(Block &body, unsigned numLoops) {
  if (numLoops == 0)
    return false;
  if (numLoops == 1)
    return isExecutableInnermostBody(body);

  Block *current = &body;
  for (unsigned depth = 1; depth < numLoops; ++depth) {
    scf::ForOp nestedLoop;
    for (Operation &op : current->without_terminator()) {
      if (isCarrierOp(op))
        continue;
      if (!isa<scf::ForOp>(op) || nestedLoop)
        return false;
      nestedLoop = cast<scf::ForOp>(op);
    }
    if (!nestedLoop || !nestedLoop.getInitArgs().empty())
      return false;
    current = nestedLoop.getBody();
  }

  return isExecutableInnermostBody(*current);
}

static linalg::GenericOp findTensorCarrier(Block &body) {
  linalg::GenericOp tensorGeneric;
  for (Operation &op : body) {
    auto generic = dyn_cast<linalg::GenericOp>(op);
    if (!generic)
      continue;
    if (!llvm::all_of(generic.getDpsInputs(), [](Value operand) {
          return isa<TensorType>(operand.getType());
        }))
      continue;
    if (!llvm::all_of(generic.getDpsInits(), [](Value operand) {
          return isa<TensorType>(operand.getType());
        }))
      continue;
    if (tensorGeneric)
      return nullptr;
    tensorGeneric = generic;
  }
  return tensorGeneric;
}

static bool hasDisjointWriteSet(Block &body, linalg::GenericOp tensorGeneric) {
  unsigned loadCount = 0;
  unsigned storeCount = 0;
  SmallVector<Value> writtenMemrefs;

  body.walk([&](memref::LoadOp) { ++loadCount; });
  body.walk([&](memref::StoreOp storeOp) {
    ++storeCount;
    Value base = stripSimpleMemrefAlias(storeOp.getMemref());
    if (!llvm::is_contained(writtenMemrefs, base))
      writtenMemrefs.push_back(base);
  });

  if (loadCount != tensorGeneric.getNumDpsInputs() ||
      storeCount != tensorGeneric.getNumDpsInits() ||
      writtenMemrefs.size() != static_cast<size_t>(storeCount))
    return false;

  for (auto [outputIndex, output] :
       llvm::enumerate(tensorGeneric.getDpsInits())) {
    AffineMap indexingMap = tensorGeneric.getMatchingIndexingMap(
        tensorGeneric.getDpsInitOperand(outputIndex));
    if (!indexingMap.isProjectedPermutation())
      return false;
  }

  return true;
}

static bool isElementwiseTensorCandidate(Block &body,
                                         linalg::GenericOp tensorGeneric,
                                         unsigned numSuDims) {
  if (tensorGeneric.getNumLoops() == 0)
    return false;
  if (!llvm::all_of(tensorGeneric.getIteratorTypesArray(),
                    [](utils::IteratorType type) {
                      return type == utils::IteratorType::parallel;
                    }))
    return false;
  if (tensorGeneric.getNumDpsInits() == 0)
    return false;
  if (!hasDisjointWriteSet(body, tensorGeneric))
    return false;
  // The su_iterate provides numSuDims induction variables directly as block
  // args. The remaining dimensions must appear as nested scf.for loops.
  unsigned nestedLoops = tensorGeneric.getNumLoops() >= numSuDims
                             ? tensorGeneric.getNumLoops() - numSuDims + 1
                             : 1;
  return hasPerfectNestedScalarLoopNest(body, nestedLoops);
}

static bool isMatmulTensorCandidate(Block &body,
                                    linalg::GenericOp tensorGeneric) {
  if (tensorGeneric.getNumLoops() != 3 || tensorGeneric.getNumDpsInits() != 1)
    return false;

  unsigned numParallel = 0;
  unsigned numReduction = 0;
  for (utils::IteratorType type : tensorGeneric.getIteratorTypesArray()) {
    if (type == utils::IteratorType::parallel)
      ++numParallel;
    else if (type == utils::IteratorType::reduction)
      ++numReduction;
  }
  if (numParallel != 2 || numReduction != 1)
    return false;

  unsigned nonCarrierOps = 0;
  for (Operation &nested : body.without_terminator()) {
    if (isCarrierOp(nested))
      continue;
    if (!isa<scf::ForOp>(nested))
      return false;
    ++nonCarrierOps;
  }
  return nonCarrierOps == 1;
}

static bool isReductionTensorCandidate(Block &body,
                                       linalg::GenericOp tensorGeneric) {
  if (tensorGeneric.getNumLoops() == 0 || tensorGeneric.getNumDpsInits() == 0)
    return false;
  // Must have at least one parallel dim to tile.
  bool hasParallel = false;
  bool hasReduction = false;
  for (utils::IteratorType type : tensorGeneric.getIteratorTypesArray()) {
    if (type == utils::IteratorType::parallel)
      hasParallel = true;
    else if (type == utils::IteratorType::reduction)
      hasReduction = true;
  }
  if (!hasParallel || !hasReduction)
    return false;
  // Must have a single nested scf.for (the loop nest).
  unsigned nonCarrierOps = 0;
  for (Operation &nested : body.without_terminator()) {
    if (isCarrierOp(nested))
      continue;
    if (!isa<scf::ForOp>(nested))
      return false;
    ++nonCarrierOps;
  }
  return nonCarrierOps == 1;
}

/// Return a per-SDE-dim mask: true = parallel (should tile), false = reduction
/// (keep original step). The su_iterate's first dim maps to the first carrier
/// iterator type, etc.
static SmallVector<bool> getParallelDimMask(sde::SdeSuIterateOp op,
                                            linalg::GenericOp tensorGeneric) {
  unsigned numDims = op.getLowerBounds().size();
  SmallVector<bool> mask(numDims, true); // default: tile everything
  if (!tensorGeneric)
    return mask;
  auto iterTypes = tensorGeneric.getIteratorTypesArray();
  for (unsigned d = 0; d < numDims && d < iterTypes.size(); ++d) {
    mask[d] = (iterTypes[d] == utils::IteratorType::parallel);
  }
  return mask;
}

static bool isStencilCandidate(sde::SdeSuIterateOp op, Block &body) {
  if (!op.getAccessMinOffsetsAttr() || !op.getAccessMaxOffsetsAttr())
    return false;
  // Stencils always have at least one nested scf.for for the inner dimension.
  // Count the total loop depth: 1 SDE dim + inner scf.for loops.
  unsigned numSuDims = op.getLowerBounds().size();
  unsigned innerLoops = 0;
  Block *current = &body;
  while (true) {
    scf::ForOp nestedLoop;
    for (Operation &nested : current->without_terminator()) {
      if (isCarrierOp(nested))
        continue;
      if (auto forOp = dyn_cast<scf::ForOp>(nested)) {
        if (nestedLoop)
          return false; // imperfect nest
        nestedLoop = forOp;
      } else if (!isScalarExecutableOp(nested)) {
        return false;
      }
    }
    if (!nestedLoop)
      break;
    ++innerLoops;
    current = nestedLoop.getBody();
  }
  return isExecutableInnermostBody(*current) && (numSuDims + innerLoops) >= 1;
}

static SmallVector<int64_t> getStencilHaloWidths(sde::SdeSuIterateOp op) {
  SmallVector<int64_t> halos;
  ArrayAttr minArr = op.getAccessMinOffsetsAttr();
  ArrayAttr maxArr = op.getAccessMaxOffsetsAttr();
  if (!minArr || !maxArr || minArr.size() != maxArr.size())
    return {};
  for (unsigned d = 0; d < minArr.size(); ++d) {
    int64_t lo = cast<IntegerAttr>(minArr[d]).getInt();
    int64_t hi = cast<IntegerAttr>(maxArr[d]).getInt();
    halos.push_back(std::max<int64_t>(1, hi - lo + 1));
  }
  return halos;
}

/// Check whether a body block is carrier-authoritative: it contains a
/// linalg.generic carrier but NO scalar executable ops (memref.load/store)
/// anywhere in the body, including nested regions (except inside the carrier
/// itself). This distinguishes carrier-authoritative IR (where the carrier IS
/// the computation) from dual-representation IR (where scalar ops coexist).
static bool isCarrierAuthoritative(Block &body) {
  bool hasCarrier = false;
  bool hasScalar = false;
  for (Operation &op : body.without_terminator()) {
    if (isa<linalg::GenericOp>(op)) {
      hasCarrier = true;
      continue; // Don't walk into the carrier's own body.
    }
    if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      hasScalar = true;
      continue;
    }
    // Walk into nested regions (scf.for, etc.) to find scalar ops.
    op.walk([&](Operation *nested) {
      if (isa<memref::LoadOp, memref::StoreOp>(nested))
        hasScalar = true;
    });
  }
  return hasCarrier && !hasScalar;
}

/// Check carrier-authoritative eligibility purely from the carrier's
/// properties. In authoritative mode, the carrier IS the computation — no
/// scalar body to inspect. We check iterator types, output projections, and
/// disjointness directly from the linalg.generic.
static bool
isCarrierAuthoritativeCandidate(linalg::GenericOp carrier,
                                sde::SdeStructuredClassification cls) {
  if (carrier.getNumDpsInits() == 0 || carrier.getNumLoops() == 0)
    return false;

  auto iterTypes = carrier.getIteratorTypesArray();
  switch (cls) {
  case sde::SdeStructuredClassification::elementwise:
    // All iterators must be parallel, all output maps must be projected perms.
    if (!llvm::all_of(iterTypes, [](utils::IteratorType t) {
          return t == utils::IteratorType::parallel;
        }))
      return false;
    for (unsigned i = 0; i < carrier.getNumDpsInits(); ++i) {
      AffineMap map =
          carrier.getMatchingIndexingMap(carrier.getDpsInitOperand(i));
      if (!map.isProjectedPermutation())
        return false;
    }
    return true;

  case sde::SdeStructuredClassification::matmul:
    return carrier.getNumLoops() == 3 && carrier.getNumDpsInits() == 1;

  case sde::SdeStructuredClassification::reduction: {
    bool hasParallel = false, hasReduction = false;
    for (auto t : iterTypes) {
      if (t == utils::IteratorType::parallel)
        hasParallel = true;
      if (t == utils::IteratorType::reduction)
        hasReduction = true;
    }
    return hasParallel && hasReduction;
  }

  default:
    return false;
  }
}

static bool isTilingCandidate(sde::SdeSuIterateOp op, Block &body,
                              linalg::GenericOp tensorGeneric) {
  if (op.getChunkSize())
    return false;
  if (op->getParentOfType<scf::ForOp>())
    return false;
  if (op.getLowerBounds().empty())
    return false;
  if (!op.getStructuredClassificationAttr())
    return false;

  auto classification = *op.getStructuredClassification();

  // Carrier-authoritative path: body has carrier but no scalar ops.
  if (tensorGeneric && isCarrierAuthoritative(body))
    return isCarrierAuthoritativeCandidate(tensorGeneric, classification);

  // Dual-representation path: scalar body alongside carrier.
  unsigned numSuDims = op.getLowerBounds().size();
  switch (classification) {
  case sde::SdeStructuredClassification::stencil:
    return op.getReductionAccumulators().size() == 0 &&
           isStencilCandidate(op, body);
  case sde::SdeStructuredClassification::elementwise:
    if (!tensorGeneric || op.getReductionAccumulators().size() != 0)
      return false;
    return isElementwiseTensorCandidate(body, tensorGeneric, numSuDims);
  case sde::SdeStructuredClassification::matmul:
    if (!tensorGeneric || op.getReductionAccumulators().size() != 0)
      return false;
    return isMatmulTensorCandidate(body, tensorGeneric);
  case sde::SdeStructuredClassification::reduction:
    if (!tensorGeneric)
      return false;
    return isReductionTensorCandidate(body, tensorGeneric);
  default:
    return false;
  }
}

/// Tile a carrier-authoritative body. Instead of creating scf.for tile loops
/// and cloning scalar ops, construct extract_slice → tiled linalg.generic →
/// insert_slice at the su_iterate body level.
///
/// For each carrier operand, the slicing is driven by the operand's indexing
/// map: dimensions that reference a tiled parallel dim get sliced at
/// [tileBase][tileSize], while untiled (reduction/full-range) dimensions
/// keep their full range.
static bool tileCarrierAuthoritative(linalg::GenericOp carrier, Block &srcBody,
                                     OpBuilder &builder, Value tileBase,
                                     Value tileSize, unsigned tileDim,
                                     IRMapping &mapper) {
  Location loc = carrier.getLoc();

  // 1. Clone carrier prep ops (mu_memref_to_tensor, bufferization.to_tensor,
  //    tensor.empty, etc.) but NOT the carrier itself.
  for (Operation &op : srcBody.without_terminator()) {
    if (isa<linalg::GenericOp>(op))
      continue;
    builder.clone(op, mapper);
  }

  auto iterTypes = carrier.getIteratorTypesArray();
  auto indexingMaps = carrier.getIndexingMapsArray();

  // Build a map: carrier loop dim → (isTiled, tileBase, tileSize).
  // Only the tileDim is tiled; others keep full range.
  // For N-dim carriers where the su_iterate provides dim 0 and inner dims
  // are additional parallel dims, only dim tileDim is sliced here.

  // Helper: for a given operand, compute extract_slice offsets/sizes/strides
  // based on its indexing map.
  auto sliceOperand = [&](Value operand, AffineMap map) -> Value {
    Value mapped = mapper.lookupOrDefault(operand);
    auto tensorTy = cast<RankedTensorType>(mapped.getType());
    unsigned rank = tensorTy.getRank();

    SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

    // Initialize sizes to full static extents.
    for (unsigned d = 0; d < rank; ++d)
      sizes.push_back(builder.getIndexAttr(tensorTy.getDimSize(d)));

    // For each result dim of the map, check if it references the tiled
    // carrier dim. If so, slice that operand dim.
    for (unsigned r = 0; r < map.getNumResults(); ++r) {
      AffineExpr expr = map.getResult(r);
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        if (dimExpr.getPosition() == tileDim) {
          offsets[r] = tileBase;
          sizes[r] = tileSize;
        }
      }
    }

    // Check if any dim is actually dynamic (tiled).
    bool needsSlice = false;
    for (unsigned r = 0; r < rank; ++r) {
      if (auto val = dyn_cast<Value>(offsets[r])) {
        needsSlice = true;
        break;
      }
      if (auto val = dyn_cast<Value>(sizes[r])) {
        needsSlice = true;
        break;
      }
    }
    if (!needsSlice)
      return mapped;

    return tensor::ExtractSliceOp::create(builder, loc, mapped, offsets, sizes,
                                          strides);
  };

  // 2. Slice each carrier input.
  SmallVector<Value> slicedInputs;
  for (auto [idx, input] : llvm::enumerate(carrier.getDpsInputs())) {
    AffineMap map = indexingMaps[idx];
    Value sliced = sliceOperand(input, map);
    slicedInputs.push_back(sliced);
  }

  // 3. Slice each carrier output.
  unsigned inputCount = carrier.getNumDpsInputs();
  SmallVector<Value> slicedOutputs;
  SmallVector<Value> fullOutputs; // for insert_slice targets
  for (auto [idx, output] : llvm::enumerate(carrier.getDpsInits())) {
    AffineMap map = indexingMaps[inputCount + idx];
    Value mappedFull = mapper.lookupOrDefault(output);
    fullOutputs.push_back(mappedFull);
    Value sliced = sliceOperand(output, map);
    slicedOutputs.push_back(sliced);
  }

  // 4. Compute result types from sliced outputs.
  SmallVector<Type> resultTypes;
  for (Value out : slicedOutputs)
    resultTypes.push_back(out.getType());

  // 5. Create tiled linalg.generic with sliced operands.
  // The carrier body may reference values outside the carrier (e.g., su_iterate
  // block args). Start from the parent mapper so those external references are
  // remapped to the new body's arguments.
  auto tiledGeneric = linalg::GenericOp::create(
      builder, loc, resultTypes, slicedInputs, slicedOutputs, indexingMaps,
      iterTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        IRMapping bodyMapper(mapper);
        Block &carrierBody = carrier.getRegion().front();
        for (auto [oldArg, newArg] :
             llvm::zip(carrierBody.getArguments(), blockArgs))
          bodyMapper.map(oldArg, newArg);
        for (Operation &op : carrierBody.without_terminator())
          nestedBuilder.clone(op, bodyMapper);
        auto carrierYield = cast<linalg::YieldOp>(carrierBody.getTerminator());
        SmallVector<Value> yieldValues;
        for (Value v : carrierYield.getValues())
          yieldValues.push_back(bodyMapper.lookupOrDefault(v));
        linalg::YieldOp::create(nestedBuilder, nestedLoc, yieldValues);
      });

  // 6. Insert_slice each output result back into the full tensor.
  for (auto [idx, result] : llvm::enumerate(tiledGeneric.getResults())) {
    AffineMap map = indexingMaps[inputCount + idx];
    auto fullTensorTy = cast<RankedTensorType>(fullOutputs[idx].getType());
    unsigned rank = fullTensorTy.getRank();

    SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
    for (unsigned d = 0; d < rank; ++d)
      sizes.push_back(builder.getIndexAttr(fullTensorTy.getDimSize(d)));

    for (unsigned r = 0; r < map.getNumResults(); ++r) {
      AffineExpr expr = map.getResult(r);
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        if (dimExpr.getPosition() == tileDim) {
          offsets[r] = tileBase;
          sizes[r] = tileSize;
        }
      }
    }

    tensor::InsertSliceOp::create(builder, loc, result, fullOutputs[idx],
                                  offsets, sizes, strides);
  }

  return true;
}

static void cloneBodyIntoTileLoop(Block &srcBody, IRMapping &mapper,
                                  OpBuilder &builder) {
  bool authoritative = isCarrierAuthoritative(srcBody);
  for (Operation &op : srcBody.without_terminator()) {
    if (!authoritative && isCarrierOp(op))
      continue;
    builder.clone(op, mapper);
  }
}

struct TilingPass : public arts::impl::TilingBase<TilingPass> {
  explicit TilingPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    if (!costModel)
      return;

    SmallVector<std::pair<sde::SdeSuIterateOp, linalg::GenericOp>> rewrites;
    getOperation().walk([&](sde::SdeSuIterateOp op) {
      Block *body = sde::getSuIterateComputeBlock(op);
      linalg::GenericOp tensorGeneric = findTensorCarrier(*body);
      if (!isTilingCandidate(op, *body, tensorGeneric))
        return;

      std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation());
      if (tripCount && *tripCount <= 1)
        return;
      rewrites.push_back({op, tensorGeneric});
    });

    for (auto [op, tensorGeneric] : rewrites) {
      PatternRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();
      unsigned numDims = op.getLowerBounds().size();

      // Determine which dims are parallel (should tile) vs reduction (skip).
      SmallVector<bool> parallelMask = getParallelDimMask(op, tensorGeneric);

      // Compute per-dim tile iterations.
      SmallVector<Value> perDimTileIter;
      if (numDims == 1) {
        // 1-D fast path: preserves existing static trip count optimization.
        if (!parallelMask[0]) {
          // Single reduction dim -- nothing to tile.
          continue;
        }
        Value tileIterations;
        if (std::optional<int64_t> tripCount =
                getStaticTripCount(op.getOperation())) {
          int64_t balancedTile = llvm::divideCeil(
              *tripCount, std::max<int64_t>(1, costModel->getWorkerCount()));
          int64_t tileCount = std::clamp(
              std::max<int64_t>(balancedTile,
                                costModel->getMinIterationsPerWorker()),
              int64_t{1}, *tripCount);
          if (tileCount <= 1)
            continue;
          tileIterations = getConstantIndex(rewriter, loc, tileCount);
        } else {
          tileIterations =
              buildTileIterationValue(rewriter, loc, op, *costModel);
        }
        if (!tileIterations)
          continue;
        perDimTileIter.push_back(tileIterations);
      } else {
        // N-dim path: distribute workers across dimensions.
        perDimTileIter =
            buildPerDimTileIterations(rewriter, loc, op, *costModel);
        if (perDimTileIter.empty())
          continue;
      }

      // For stencils, enforce halo-aware minimum tile size per dimension.
      if (op.getStructuredClassification() ==
          sde::SdeStructuredClassification::stencil) {
        SmallVector<int64_t> halos = getStencilHaloWidths(op);
        for (unsigned d = 0; d < numDims && d < halos.size(); ++d) {
          Value haloVal = getConstantIndex(rewriter, loc, halos[d]);
          perDimTileIter[d] =
              arith::MaxUIOp::create(rewriter, loc, perDimTileIter[d], haloVal);
        }
        // Cache-friendly tile: prefer tiles that fit in L2 cache
        int64_t elemSize = 8; // default f64
        // Try memref stores first; fall back to carrier output type.
        bool foundElem = false;
        op.getBody().walk([&](memref::StoreOp storeOp) {
          Type elemType = storeOp.getValueToStore().getType();
          if (elemType.isF32() || elemType.isInteger(32))
            elemSize = 4;
          foundElem = true;
          return WalkResult::interrupt();
        });
        if (!foundElem && tensorGeneric && tensorGeneric.getNumDpsInits() > 0) {
          Type outTy = tensorGeneric.getDpsInits()[0].getType();
          if (auto shapedTy = dyn_cast<ShapedType>(outTy)) {
            Type et = shapedTy.getElementType();
            if (et.isF32() || et.isInteger(32))
              elemSize = 4;
          }
        }
        int64_t l2Size = costModel->getL2CacheSize();
        int64_t cacheLineTile =
            l2Size / (elemSize * std::max<unsigned>(1, numDims));
        Value cacheVal = getConstantIndex(rewriter, loc,
                                          std::max<int64_t>(1, cacheLineTile));
        for (unsigned d = 0; d < numDims; ++d) {
          perDimTileIter[d] = arith::MinUIOp::create(
              rewriter, loc, perDimTileIter[d], cacheVal);
        }
      }

      // Validate steps and compute tiled steps per dim.
      // For reduction dims, keep the original step (no tiling).
      SmallVector<Value> tiledSteps;
      bool badStep = false;
      bool anyTiled = false;
      for (unsigned d = 0; d < numDims; ++d) {
        Value originalStep = op.getSteps()[d];
        int64_t constantStep = 0;
        if (ValueAnalysis::getConstantIndex(originalStep, constantStep) &&
            constantStep <= 0) {
          badStep = true;
          break;
        }
        if (parallelMask[d]) {
          tiledSteps.push_back(arith::MulIOp::create(
              rewriter, loc, originalStep, perDimTileIter[d]));
          anyTiled = true;
        } else {
          tiledSteps.push_back(originalStep);
        }
      }
      if (badStep || !anyTiled)
        continue;

      auto newOp = sde::SdeSuIterateOp::create(
          rewriter, loc, /*resultTypes=*/TypeRange{}, op.getLowerBounds(),
          op.getUpperBounds(), ValueRange{tiledSteps}, op.getScheduleAttr(),
          op.getChunkSize(), op.getNowaitAttr(), op.getReductionAccumulators(),
          op.getReductionKindsAttr(), op.getReductionStrategyAttr(),
          op.getStructuredClassificationAttr(), op.getAccessMinOffsetsAttr(),
          op.getAccessMaxOffsetsAttr(), op.getOwnerDimsAttr(),
          op.getSpatialDimsAttr(), op.getWriteFootprintAttr());
      newOp->setAttrs(sde::getRewrittenAttrs(op));

      Block &newBody = sde::ensureBlock(newOp.getBody());
      for (unsigned d = newBody.getNumArguments(); d < numDims; ++d)
        newBody.addArgument(rewriter.getIndexType(), loc);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newBody);

      Block &srcBody = op.getBody().front();
      Block *computeBody = sde::getSuIterateComputeBlock(op);

      // Carrier-authoritative path: tile the carrier directly via
      // extract_slice → tiled linalg.generic → insert_slice.
      // No scf.for tile loop — the carrier encodes the iteration.
      if (tensorGeneric && isCarrierAuthoritative(*computeBody)) {
        // For 1-D: tile dim 0 (the su_iterate dim).
        // Compute dynamic tile size: min(tileStep, ub - base).
        Value tileBase = newBody.getArgument(0);
        Value rem = arith::SubIOp::create(rewriter, loc, op.getUpperBounds()[0],
                                          tileBase);
        Value tileSizeVal =
            arith::MinUIOp::create(rewriter, loc, perDimTileIter[0], rem);
        // Map old body's block args → new body's block args so that any
        // ops cloned from the old body that reference iteration variables
        // will point to the new body's arguments (not the old, soon-erased
        // ones).
        IRMapping carrierMapper;
        for (unsigned d = 0; d < numDims; ++d)
          carrierMapper.map(srcBody.getArgument(d), newBody.getArgument(d));
        tileCarrierAuthoritative(tensorGeneric, *computeBody, rewriter,
                                 tileBase, tileSizeVal, /*tileDim=*/0,
                                 carrierMapper);
      } else {
        // Dual-representation path: build nested scf.for tile loops for
        // parallel dims only. Reduction dims pass through directly.
        IRMapping mapper;
        for (unsigned d = 0; d < numDims; ++d) {
          Value tileBase = newBody.getArgument(d);
          if (!parallelMask[d]) {
            // Reduction dim: map directly, no tile loop.
            mapper.map(srcBody.getArgument(d), tileBase);
            continue;
          }
          Value tileLimit =
              arith::AddIOp::create(rewriter, loc, tileBase, tiledSteps[d]);
          Value tileUpper = arith::MinUIOp::create(rewriter, loc, tileLimit,
                                                   op.getUpperBounds()[d]);
          Value originalStep = op.getSteps()[d];
          auto tileLoop = scf::ForOp::create(rewriter, loc, tileBase, tileUpper,
                                             originalStep);
          mapper.map(srcBody.getArgument(d), tileLoop.getInductionVar());
          rewriter.setInsertionPointToStart(tileLoop.getBody());
        }

        // Clone body into the innermost tile loop. In dual-representation
        // mode, skips carrier ops.
        cloneBodyIntoTileLoop(*computeBody, mapper, rewriter);
      }

      // Yield at the end of the su_iterate body (not inside nested loops).
      rewriter.setInsertionPointToEnd(&newBody);
      sde::SdeYieldOp::create(rewriter, loc, ValueRange{});

      rewriter.eraseOp(op);
    }
  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::arts::sde {

std::unique_ptr<Pass> createTilingPass(sde::SDECostModel *costModel) {
  return std::make_unique<TilingPass>(costModel);
}

} // namespace mlir::arts::sde
