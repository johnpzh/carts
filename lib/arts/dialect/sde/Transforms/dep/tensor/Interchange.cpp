///==========================================================================///
/// LoopInterchange.cpp - Cache-optimal loop interchange via linalg carriers
///
/// Two interchange strategies:
///
/// 1. Carrier-based (Phase 7A): For any classification with a tensor carrier,
///    analyzes indexing maps to compute per-dim stride cost. Dims that appear
///    as the last (rightmost/stride-1) index in more operand maps should be
///    innermost. Uses bubble-sort over parallel dims by cost. Subsumes the
///    original matmul-only j-k to k-j interchange.
///
/// 2. Stencil halo ordering (Phase 7B): For stencil classification (no tensor
///    carrier), reads accessMinOffsets/accessMaxOffsets to compute per-dim halo
///    width. For 3D+ stencils with multiple inner scf.for loops, reorders so
///    the smallest-halo-width dim is outermost (minimizes total halo volume).
///
/// The interchange handles imperfect nests by distributing init ops into a
/// separate loop when the j loop contains both init stores and a reduction
/// loop (e.g., 2mm: C[i,j] = 0 then for k: C[i,j] += A[i,k]*B[k,j]).
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_LOOPINTERCHANGE
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(loop_interchange);

using namespace mlir;
using namespace mlir::arts;

namespace {

static bool isCarrierOp(Operation &op) {
  return isa<bufferization::ToTensorOp, sde::SdeMuMemrefToTensorOp,
             tensor::EmptyOp, linalg::GenericOp, tensor::ExtractSliceOp,
             tensor::InsertSliceOp>(op);
}

/// Check whether a body block is carrier-authoritative: it contains a
/// linalg.generic carrier but NO scalar executable ops (memref.load/store)
/// anywhere in the body, including nested regions (except inside the carrier).
static bool isCarrierAuthoritative(Block &body) {
  bool hasCarrier = false;
  bool hasScalar = false;
  for (Operation &op : body.without_terminator()) {
    if (isa<linalg::GenericOp>(op)) {
      hasCarrier = true;
      continue;
    }
    if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      hasScalar = true;
      continue;
    }
    op.walk([&](Operation *nested) {
      if (isa<memref::LoadOp, memref::StoreOp>(nested))
        hasScalar = true;
    });
  }
  return hasCarrier && !hasScalar;
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

/// Check whether a matmul carrier would benefit from j-k to k-j interchange.
/// Returns true if the reduction dimension is not already outermost among the
/// two inner loops.
///
/// For standard matmul C[i,j] += A[i,k]*B[k,j]:
///   Iterator types: [parallel(i), parallel(j), reduction(k)]
///   The carrier's dims 1 (j) and 2 (k) correspond to the two inner scf.for
///   loops. We want k outermost for stride-1 B[k,j] access on j.
///
/// The carrier's indexing maps tell us the exact dim indices:
///   A: (d0, d1, d2) -> (d0, d2)  => reads at [i, k]
///   B: (d0, d1, d2) -> (d2, d1)  => reads at [k, j]
///   C: (d0, d1, d2) -> (d0, d1)  => reads/writes at [i, j]
///
/// For B's last (innermost) indexing dimension to have stride-1 access,
/// that dimension should be the innermost loop. B's last dim is d1 (j).
/// So we need j innermost, k outermost among the inner pair.
static bool shouldInterchangeMatmul(linalg::GenericOp carrier,
                                    unsigned &jDimIdx, unsigned &kDimIdx) {
  auto iterTypes = carrier.getIteratorTypesArray();
  if (iterTypes.size() != 3)
    return false;

  // Find the reduction dim (k) and the two parallel dims.
  // The first dim (d0) is the outer SDE su_iterate dim.
  SmallVector<unsigned> parallelDims;
  unsigned reductionDim = 0;
  bool foundReduction = false;
  for (unsigned i = 0; i < 3; ++i) {
    if (iterTypes[i] == utils::IteratorType::reduction) {
      if (foundReduction)
        return false;
      reductionDim = i;
      foundReduction = true;
    } else if (iterTypes[i] == utils::IteratorType::parallel) {
      parallelDims.push_back(i);
    }
  }
  if (!foundReduction || parallelDims.size() != 2)
    return false;

  // The scf.for loops inside the su_iterate body follow the carrier dim
  // ordering (excluding d0 which is the su_iterate IV). The first inner
  // scf.for = dim 1, second inner scf.for = dim 2.
  // We need k to be the first inner (outermost of pair) for stride-1 B access.
  // If k is already dim 1, no swap needed.
  if (reductionDim == 1)
    return false;

  // reductionDim == 2 (innermost) and a parallel dim == 1 — need interchange
  jDimIdx = 1; // current outer of the inner pair (parallel/j)
  kDimIdx = 2; // current inner of the inner pair (reduction/k)
  return true;
}

/// Compute a stride cost for each dim in the carrier's indexing maps.
/// A dim that appears as the rightmost (last) index in an operand's affine map
/// has stride-1 access for that operand. Dims with more stride-1 appearances
/// should be innermost for cache efficiency.
///
/// Returns a cost per dim where lower cost = should be more inner.
/// Cost 0 = dim appears as last index in the most maps (best stride-1
/// candidate).
static SmallVector<int64_t> computeStrideCostPerDim(linalg::GenericOp carrier) {
  unsigned numDims = carrier.getNumLoops();
  SmallVector<int64_t> stride1Count(numDims, 0);

  for (AffineMap map : carrier.getIndexingMapsArray()) {
    if (map.getNumResults() == 0)
      continue;
    // The last result of the map is the innermost (stride-1) index
    AffineExpr lastExpr = map.getResult(map.getNumResults() - 1);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(lastExpr))
      stride1Count[dimExpr.getPosition()]++;
  }

  // Invert: dims with more stride-1 hits get lower cost (should be inner)
  int64_t maxCount =
      *std::max_element(stride1Count.begin(), stride1Count.end());
  SmallVector<int64_t> cost(numDims);
  for (unsigned i = 0; i < numDims; ++i)
    cost[i] = maxCount - stride1Count[i];
  return cost;
}

/// Compute the optimal permutation for inner dims (excluding dim 0 which is the
/// SDE su_iterate dim). Only considers parallel dims for reordering; reduction
/// dims stay in their original position relative to parallel dims.
///
/// Returns the permutation as indices into the original dim ordering, or empty
/// if no interchange is beneficial.
static SmallVector<unsigned>
computeOptimalPermutation(linalg::GenericOp carrier) {
  auto iterTypes = carrier.getIteratorTypesArray();
  unsigned numDims = iterTypes.size();
  if (numDims < 2)
    return {};

  // Collect inner dims (all except dim 0 = SDE loop dim)
  SmallVector<unsigned> innerDims;
  for (unsigned i = 1; i < numDims; ++i)
    innerDims.push_back(i);

  if (innerDims.size() < 2)
    return {};

  SmallVector<int64_t> cost = computeStrideCostPerDim(carrier);

  // Separate parallel and reduction dims among the inner dims
  SmallVector<unsigned> parallelInner;
  for (unsigned d : innerDims) {
    if (iterTypes[d] == utils::IteratorType::parallel)
      parallelInner.push_back(d);
  }
  if (parallelInner.size() < 2)
    return {};

  // Bubble-sort parallel dims by cost: highest cost (worst stride) outermost,
  // lowest cost (best stride-1) innermost
  SmallVector<unsigned> sorted = parallelInner;
  for (size_t i = 0; i < sorted.size(); ++i) {
    for (size_t j = 0; j + 1 < sorted.size() - i; ++j) {
      if (cost[sorted[j]] < cost[sorted[j + 1]])
        std::swap(sorted[j], sorted[j + 1]);
    }
  }

  // Check if any reordering happened
  if (sorted == parallelInner)
    return {};

  // Build the full permutation: dim 0 stays, then interleave sorted parallel
  // dims with reduction dims in their original relative order
  SmallVector<unsigned> perm;
  perm.push_back(0);
  unsigned sortedIdx = 0;
  for (unsigned d : innerDims) {
    if (iterTypes[d] == utils::IteratorType::reduction) {
      // Reduction dims keep their position relative to parallel dims
      // For matmul (k outermost), the sorted parallel order handles this
      // since reduction dims aren't in the sorted list
      perm.push_back(d);
    } else {
      perm.push_back(sorted[sortedIdx++]);
    }
  }
  return perm;
}

/// Collect all nested scf.for loops from the su_iterate body into a flat list.
/// Skips carrier ops. Returns false if the nest structure is not a simple
/// linear chain of scf.for loops.
static bool collectInnerForChain(Block &body,
                                 SmallVectorImpl<scf::ForOp> &loops) {
  Block *current = &body;
  while (true) {
    scf::ForOp found;
    for (Operation &op : current->without_terminator()) {
      if (isCarrierOp(op))
        continue;
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (found)
          return false; // multiple loops at same level
        found = forOp;
      }
      // Allow non-carrier, non-for ops (e.g. arith ops for index computation)
    }
    if (!found)
      break;
    loops.push_back(found);
    current = found.getBody();
  }
  return !loops.empty();
}

/// Find the two innermost scf.for loops inside the su_iterate body.
/// The su_iterate body can contain carrier ops + one scf.for (the j loop),
/// and j's body can contain an optional init section + one scf.for (the k
/// loop).
static bool findInnerLoopPair(Block &body, scf::ForOp &jLoop, scf::ForOp &kLoop,
                              SmallVectorImpl<Operation *> &initOps) {
  jLoop = nullptr;
  kLoop = nullptr;

  for (Operation &op : body.without_terminator()) {
    if (isCarrierOp(op))
      continue;
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (jLoop)
        return false; // multiple loops at this level
      jLoop = forOp;
    } else {
      return false; // unexpected non-carrier, non-for op
    }
  }
  if (!jLoop)
    return false;

  // Inside the j loop, find the k loop and any prefix init ops.
  Block *jBody = jLoop.getBody();
  for (Operation &op : jBody->without_terminator()) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (kLoop)
        return false; // multiple nested loops
      kLoop = forOp;
    } else {
      // Everything before the k loop is an init op.
      if (!kLoop)
        initOps.push_back(&op);
      else
        return false; // ops after the inner loop
    }
  }
  return kLoop != nullptr;
}

/// Collect prefix ops from initOps that are used by kLoop's body.
/// These must be rematerialized into the rebuilt loop nest.
static bool
collectRematerializablePrefixOps(ArrayRef<Operation *> initOps,
                                 scf::ForOp kLoop,
                                 SmallVectorImpl<Operation *> &out) {
  DenseSet<Operation *> prefixSet(initOps.begin(), initOps.end());
  DenseSet<Operation *> needed;
  bool supported = true;

  std::function<void(Value)> visitValue = [&](Value value) {
    if (!supported)
      return;
    Operation *defOp = value.getDefiningOp();
    if (!defOp || !prefixSet.contains(defOp))
      return;
    if (!needed.insert(defOp).second)
      return;
    if (!isMemoryEffectFree(defOp)) {
      supported = false;
      return;
    }
    for (Value operand : defOp->getOperands())
      visitValue(operand);
  };

  kLoop.walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      visitValue(operand);
  });

  if (!supported)
    return false;

  for (Operation *op : initOps) {
    if (needed.contains(op))
      out.push_back(op);
  }
  return true;
}

/// Apply the j-k to k-j interchange with init loop distribution.
///
/// BEFORE:
///   su_iterate(i) {
///     for j: [init ops] for k: { body }
///   }
///
/// AFTER:
///   su_iterate(i) {
///     for j: { init ops }        (only if init has side effects)
///     for k: for j: { body }     (interchanged)
///   }
static bool interchangeLoops(scf::ForOp jLoop, scf::ForOp kLoop,
                             ArrayRef<Operation *> initOps) {
  if (kLoop->getParentOp() != jLoop.getOperation()) {
    ARTS_DEBUG("LoopInterchange: k loop is not directly inside j loop");
    return false;
  }

  SmallVector<Operation *, 8> rematerializedPrefixOps;
  if (!collectRematerializablePrefixOps(initOps, kLoop,
                                        rematerializedPrefixOps)) {
    ARTS_DEBUG("LoopInterchange: reduction body captures side-effecting "
               "prefix ops");
    return false;
  }

  OpBuilder builder(jLoop);

  Value lb1 = jLoop.getLowerBound();
  Value ub1 = jLoop.getUpperBound();
  Value step1 = jLoop.getStep();
  Value iv1 = jLoop.getInductionVar();

  Value lb2 = kLoop.getLowerBound();
  Value ub2 = kLoop.getUpperBound();
  Value step2 = kLoop.getStep();
  Value iv2 = kLoop.getInductionVar();

  bool needsInitLoop = llvm::any_of(
      initOps, [](Operation *op) { return !isMemoryEffectFree(op); });

  if (needsInitLoop) {
    scf::ForOp::create(builder, jLoop.getLoc(), lb1, ub1, step1, ValueRange{},
                       [&](OpBuilder &nestedBuilder, Location loc,
                           Value initLoopIV, ValueRange) {
                         IRMapping mapping;
                         mapping.map(iv1, initLoopIV);
                         for (Operation *op : initOps)
                           nestedBuilder.clone(*op, mapping);
                         scf::YieldOp::create(nestedBuilder, loc);
                       });
  }

  scf::ForOp::create(
      builder, jLoop.getLoc(), lb2, ub2, step2, ValueRange{},
      [&](OpBuilder &outerBuilder, Location loc, Value newOuterIV, ValueRange) {
        scf::ForOp::create(
            outerBuilder, loc, lb1, ub1, step1, ValueRange{},
            [&](OpBuilder &innerBuilder, Location innerLoc, Value newInnerIV,
                ValueRange) {
              IRMapping mapping;
              mapping.map(iv1, newInnerIV); // j IV -> new inner IV
              mapping.map(iv2, newOuterIV); // k IV -> new outer IV

              for (Operation *op : rematerializedPrefixOps)
                innerBuilder.clone(*op, mapping);

              for (Operation &op : kLoop.getBody()->without_terminator())
                innerBuilder.clone(op, mapping);

              scf::YieldOp::create(innerBuilder, innerLoc);
            });
        scf::YieldOp::create(outerBuilder, loc);
      });

  ARTS_INFO("SDE loop interchange applied: k-j order (was j-k)");

  jLoop->erase();
  return true;
}

struct LoopInterchangePass
    : public arts::impl::LoopInterchangeBase<LoopInterchangePass> {

  /// Try carrier-based interchange for any classification with a tensor
  /// carrier. Uses stride cost analysis from indexing maps.
  bool tryCarrierInterchange(sde::SdeSuIterateOp op, Block &body,
                             linalg::GenericOp carrier) {
    auto classification = *op.getStructuredClassification();

    // Carrier-authoritative path: permute carrier dims directly via
    // upstream linalg::interchangeGenericOp. No scf.for loops to swap.
    if (isCarrierAuthoritative(body)) {
      SmallVector<unsigned> perm = computeOptimalPermutation(carrier);
      if (perm.empty())
        return false;

      IRRewriter rewriter(carrier.getContext());
      rewriter.setInsertionPoint(carrier);
      auto result = linalg::interchangeGenericOp(rewriter, carrier, perm);
      if (failed(result))
        return false;

      ARTS_INFO("LoopInterchange: carrier-authoritative interchange applied");
      return true;
    }

    // Dual-representation path below: interchange scf.for loops.

    // Matmul-specific path: uses the dedicated shouldInterchangeMatmul
    // analysis which handles the 3-dim (i,j,k) case precisely.
    if (classification == sde::SdeStructuredClassification::matmul) {
      unsigned jDimIdx, kDimIdx;
      if (!shouldInterchangeMatmul(carrier, jDimIdx, kDimIdx))
        return false;

      scf::ForOp jLoop, kLoop;
      SmallVector<Operation *> initOps;
      if (!findInnerLoopPair(body, jLoop, kLoop, initOps))
        return false;

      return interchangeLoops(jLoop, kLoop, initOps);
    }

    // General N-dim path: for any classification with 2+ parallel inner dims,
    // compute optimal permutation from stride cost analysis.
    SmallVector<unsigned> perm = computeOptimalPermutation(carrier);
    if (perm.empty())
      return false;

    // For the general case, we currently support 2-loop interchange only
    // (same structure as matmul: outer + inner scf.for pair).
    scf::ForOp jLoop, kLoop;
    SmallVector<Operation *> initOps;
    if (!findInnerLoopPair(body, jLoop, kLoop, initOps))
      return false;

    // The permutation tells us the optimal dim ordering.
    // If the two inner dims (positions 1 and 2 in perm) are swapped from
    // the original ordering, we should interchange.
    // perm[1] > perm[2] means the original dim at position 1 should be
    // after the original dim at position 2 — so interchange needed.
    if (perm.size() < 3 || perm[1] <= perm[2])
      return false;

    ARTS_DEBUG("LoopInterchange: general N-dim interchange triggered");
    return interchangeLoops(jLoop, kLoop, initOps);
  }

  /// Try stencil halo-based interchange for stencil classification.
  /// Reads accessMinOffsets/accessMaxOffsets, computes per-dim halo width,
  /// and reorders inner loops so smallest-halo-width dim is outermost.
  /// Only applies to 3D+ stencils with 2+ inner scf.for loops.
  bool tryStencilHaloInterchange(sde::SdeSuIterateOp op, Block &body) {
    ArrayAttr minArr = op.getAccessMinOffsetsAttr();
    ArrayAttr maxArr = op.getAccessMaxOffsetsAttr();
    if (!minArr || !maxArr || minArr.size() < 2)
      return false;

    // Compute halo widths per dim
    SmallVector<int64_t> haloWidths;
    for (unsigned d = 0; d < minArr.size(); ++d) {
      int64_t lo = cast<IntegerAttr>(minArr[d]).getInt();
      int64_t hi = cast<IntegerAttr>(maxArr[d]).getInt();
      haloWidths.push_back(hi - lo);
    }

    // Collect inner scf.for loop chain
    SmallVector<scf::ForOp> innerLoops;
    if (!collectInnerForChain(body, innerLoops))
      return false;

    // Need at least 2 inner loops to consider interchange.
    // dim 0 is the su_iterate IV, inner loops correspond to dims 1..N.
    if (innerLoops.size() < 2)
      return false;

    // For the inner loop pair (first two inner loops = dims 1 and 2):
    // Smallest halo should be outermost of the pair to minimize total
    // halo volume when the outer dim is partitioned.
    // innerLoops[0] = current outermost inner (dim 1)
    // innerLoops[1] = next inner (dim 2)
    // Swap if haloWidths[1] > haloWidths[2], i.e., dim 1 has wider halo.
    // (dims are 0-indexed; dim 0 = su_iterate, dim 1 = innerLoops[0], etc.)
    unsigned dim1 = 1;
    unsigned dim2 = 2;
    if (dim2 >= haloWidths.size())
      return false;

    if (haloWidths[dim1] <= haloWidths[dim2])
      return false; // already optimal or equal

    // Build the interchange: swap the two innermost scf.for loops.
    // This is a simpler case: no init-loop distribution needed since
    // stencil bodies are pure computation.
    scf::ForOp outerLoop = innerLoops[0];
    scf::ForOp innerLoop = innerLoops[1];

    if (innerLoop->getParentOp() != outerLoop.getOperation()) {
      ARTS_DEBUG("LoopInterchange: stencil inner loop not directly nested");
      return false;
    }

    // Collect any prefix ops in the outer loop body before the inner loop
    SmallVector<Operation *> prefixOps;
    for (Operation &op : outerLoop.getBody()->without_terminator()) {
      if (&op == innerLoop.getOperation())
        break;
      prefixOps.push_back(&op);
    }

    return interchangeLoops(outerLoop, innerLoop, prefixOps);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    ARTS_INFO_HEADER(LoopInterchangePass);

    int interchangeCount = 0;

    module.walk([&](sde::SdeSuIterateOp op) {
      auto classAttr = op.getStructuredClassificationAttr();
      if (!classAttr)
        return;

      Block *body = sde::getSuIterateComputeBlock(op);
      auto classification = *op.getStructuredClassification();

      // Phase 7B: Stencil halo ordering (no tensor carrier needed)
      if (classification == sde::SdeStructuredClassification::stencil) {
        if (tryStencilHaloInterchange(op, *body))
          ++interchangeCount;
        return;
      }

      // Phase 7A: Carrier-based interchange for any classification
      linalg::GenericOp carrier = findTensorCarrier(*body);
      if (!carrier)
        return;

      if (tryCarrierInterchange(op, *body, carrier))
        ++interchangeCount;
    });

    ARTS_DEBUG("LoopInterchangePass: interchanged " << interchangeCount
                                                    << " loop nests");
  }
};

} // namespace

namespace mlir::arts::sde {

std::unique_ptr<Pass> createLoopInterchangePass() {
  return std::make_unique<LoopInterchangePass>();
}

} // namespace mlir::arts::sde
