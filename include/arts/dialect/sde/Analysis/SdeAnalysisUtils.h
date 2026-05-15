///==========================================================================///
/// File: SdeAnalysisUtils.h
///
/// Small SDE-owned helpers shared by SDE analyses and transforms.
///==========================================================================///

#ifndef ARTS_DIALECT_SDE_ANALYSIS_SDEANALYSISUTILS_H
#define ARTS_DIALECT_SDE_ANALYSIS_SDEANALYSISUTILS_H

#include "arts/dialect/sde/IR/SdeDialect.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace mlir::carts::sde {

/// Get the computation block inside su_iterate, looking through an optional
/// cu_region <parallel> wrapper.
inline Block *getSuIterateComputeBlock(SdeSuIterateOp op) {
  Block &body = op.getBody().front();
  for (Operation &inner : body.without_terminator()) {
    if (auto cuRegion = dyn_cast<SdeCuRegionOp>(inner);
        cuRegion && cuRegion.getKind() == SdeCuKind::parallel)
      return &cuRegion.getBody().front();
  }
  return &body;
}

/// Root-level memory effects for an SDE structured region. This intentionally
/// stays in SDE because memref reads/writes are SDE scheduling facts, not Core
/// object-graph facts.
struct StructuredMemoryEffectSummary {
  llvm::DenseSet<Value> reads;
  llvm::DenseSet<Value> writes;
  bool hasUnknownEffects = false;

  bool empty() const {
    return reads.empty() && writes.empty() && !hasUnknownEffects;
  }

  bool allWritesAreRead() const {
    if (writes.empty())
      return false;
    for (Value written : writes)
      if (!reads.contains(written))
        return false;
    return true;
  }

  bool hasWriteConflictWith(const StructuredMemoryEffectSummary &rhs) const {
    auto intersects = [](const llvm::DenseSet<Value> &lhs,
                         const llvm::DenseSet<Value> &rhs) {
      for (Value value : lhs)
        if (rhs.contains(value))
          return true;
      return false;
    };
    return intersects(writes, rhs.reads) || intersects(writes, rhs.writes) ||
           intersects(reads, rhs.writes);
  }
};

/// SDE-owned proof that a scheduling unit writes an external tensor/memref
/// through the owner induction variable. This is a semantic distribution fact;
/// later layers may materialize it as a physical storage layout.
struct LoopIndexedOutputPlan {
  Value root;
  SmallVector<int64_t, 4> shape;
  SmallVector<int64_t, 4> ownerPhysicalDims;
};

inline bool isDefinedInside(Operation *ancestor, Value value) {
  if (!ancestor || !value)
    return false;
  Operation *def = value.getDefiningOp();
  return def && ancestor->isAncestor(def);
}

inline bool isKnownPureScalarLibmCallee(StringRef callee) {
  return callee == "tanhf" || callee == "tanh";
}

inline bool isScalarOrVectorValueType(Type type) {
  if (type.isIntOrIndexOrFloat())
    return true;
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType && vectorType.getElementType().isIntOrIndexOrFloat();
}

/// Some C libm calls survive Polygeist/MLIR canonicalization as func.call
/// instead of math dialect ops. For SDE scheduling they are pure scalar
/// computations: they neither read nor write program memory, and allowing them
/// keeps output-only elementwise loops block-partitionable.
inline bool isKnownPureScalarLibmCall(Operation *op) {
  auto call = dyn_cast_or_null<func::CallOp>(op);
  if (!call || !isKnownPureScalarLibmCallee(call.getCallee()))
    return false;

  for (Value operand : call.getOperands())
    if (!isScalarOrVectorValueType(operand.getType()))
      return false;
  for (Value result : call.getResults())
    if (!isScalarOrVectorValueType(result.getType()))
      return false;
  return true;
}

inline bool hasUnmodeledMemoryEffect(Operation *op) {
  if (!op || op->hasTrait<OpTrait::IsTerminator>() || isa<SdeYieldOp>(op))
    return false;

  if (isa<memref::LoadOp, memref::StoreOp>(op))
    return false;

  if (isKnownPureScalarLibmCall(op))
    return false;

  /// Region operations are structural here; nested operations are visited by
  /// the region walk and report their own effects.
  if (op->getNumRegions() != 0)
    return false;

  if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    memEffects.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      if (isa<MemoryEffects::Read, MemoryEffects::Write>(
              effect.getEffect()))
        return true;
    }
    return false;
  }

  return !isMemoryEffectFree(op);
}

inline void collectStructuredMemoryEffects(
    Region &region, StructuredMemoryEffectSummary &summary) {
  region.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      summary.reads.insert(
          arts::ValueAnalysis::stripMemrefViewOps(loadOp.getMemref()));
      return;
    }

    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      summary.writes.insert(
          arts::ValueAnalysis::stripMemrefViewOps(storeOp.getMemref()));
      return;
    }

    if (hasUnmodeledMemoryEffect(op))
      summary.hasUnknownEffects = true;
  });
}

inline bool hasInPlaceSelfRead(const StructuredMemoryEffectSummary &summary) {
  if (summary.writes.empty())
    return false;
  for (Value written : summary.writes)
    if (summary.reads.contains(written))
      return true;
  return false;
}

inline SmallVector<Value, 4> collectOwnerIndexValues(SdeSuIterateOp op) {
  SmallVector<Value, 4> ownerIndexValues;
  if (op.getBody().empty())
    return ownerIndexValues;

  Block &body = op.getBody().front();
  auto ivs = op.getLoopInductionVars();
  if (!ivs || ivs->empty())
    return ownerIndexValues;
  Value loopIv = ivs->front();

  ownerIndexValues.push_back(loopIv);
  Block *computeBlock = getSuIterateComputeBlock(op);
  if (computeBlock && computeBlock != &body) {
    if (auto cuRegion = dyn_cast_or_null<SdeCuRegionOp>(
            computeBlock->getParentOp())) {
      for (auto [idx, iterArg] : llvm::enumerate(cuRegion.getIterArgs())) {
        if (idx >= computeBlock->getNumArguments())
          break;
        if (iterArg == loopIv || arts::ValueAnalysis::dependsOn(iterArg, loopIv))
          ownerIndexValues.push_back(computeBlock->getArgument(idx));
      }
    }
  }

  return ownerIndexValues;
}

inline bool isExactOwnerIndex(Value index, ArrayRef<Value> ownerIndexValues) {
  int64_t constantOffset = 0;
  Value base = arts::ValueAnalysis::stripConstantOffset(
      arts::ValueAnalysis::stripNumericCasts(index), &constantOffset);
  if (constantOffset != 0)
    return false;

  base = arts::ValueAnalysis::stripNumericCasts(base);
  for (Value ownerIndex : ownerIndexValues)
    if (arts::ValueAnalysis::sameValue(base, ownerIndex))
      return true;
  return false;
}

inline bool isOwnerDependentIndex(Value index,
                                  ArrayRef<Value> ownerIndexValues) {
  if (isExactOwnerIndex(index, ownerIndexValues))
    return true;

  Value normalized = arts::ValueAnalysis::stripNumericCasts(index);
  for (Value ownerIndex : ownerIndexValues)
    if (arts::ValueAnalysis::dependsOn(normalized, ownerIndex))
      return true;
  return false;
}

inline SmallVector<int64_t, 4>
collectExactOwnerIndexedPhysicalDims(OperandRange indices,
                                     ArrayRef<Value> ownerIndexValues) {
  SmallVector<int64_t, 4> ownerPhysicalDims;
  for (auto [idx, index] : llvm::enumerate(indices))
    if (isExactOwnerIndex(index, ownerIndexValues))
      ownerPhysicalDims.push_back(static_cast<int64_t>(idx));
  return ownerPhysicalDims;
}

/// Prove that all accesses to an in-place output root stay within the same
/// owner slice of a one-dimensional scheduling unit. This permits blocked
/// storage layout for row-local update kernels such as layernorm while rejecting
/// stencil-like first-dimension offsets (`i +/- 1`) that need halo planning.
inline bool allRootAccessesStayWithinOwnerSlice(SdeSuIterateOp op,
                                                Value root,
                                                ArrayRef<int64_t> ownerDims) {
  if (!op || !root)
    return false;
  if (ownerDims.empty())
    return false;

  SmallVector<Value, 4> ownerIndexValues = collectOwnerIndexValues(op);
  if (ownerIndexValues.empty())
    return false;

  bool sawRootAccess = false;
  auto checkIndices = [&](Value memref, OperandRange indices) {
    Value base = arts::ValueAnalysis::stripMemrefViewOps(memref);
    if (base != root)
      return WalkResult::advance();

    sawRootAccess = true;
    auto memRefType = dyn_cast<MemRefType>(base.getType());
    if (!memRefType || memRefType.getRank() == 0 || indices.empty())
      return WalkResult::interrupt();
    for (int64_t rawDim : ownerDims) {
      if (rawDim < 0 || static_cast<size_t>(rawDim) >= indices.size())
        return WalkResult::interrupt();
      if (!isExactOwnerIndex(indices[rawDim], ownerIndexValues))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };

  WalkResult result = op.getBody().walk([&](Operation *nested) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(nested)) {
      if (isa<MemRefType>(loadOp.getResult().getType()))
        return WalkResult::advance();
      return checkIndices(loadOp.getMemref(), loadOp.getIndices());
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(nested)) {
      if (isa<MemRefType>(storeOp.getValueToStore().getType()))
        return WalkResult::advance();
      return checkIndices(storeOp.getMemref(), storeOp.getIndices());
    }
    return WalkResult::advance();
  });

  return !result.wasInterrupted() && sawRootAccess;
}

inline bool allRootAccessesStayWithinOwnerSlice(SdeSuIterateOp op,
                                                Value root) {
  SmallVector<int64_t, 1> firstDim{0};
  return allRootAccessesStayWithinOwnerSlice(op, root, firstDim);
}

inline std::optional<LoopIndexedOutputPlan>
findLoopIndexedOutputPlan(SdeSuIterateOp op) {
  if (op.getBody().empty())
    return std::nullopt;

  Block &body = op.getBody().front();
  SmallVector<Value, 4> ownerIndexValues = collectOwnerIndexValues(op);
  if (ownerIndexValues.empty())
    return std::nullopt;
  Block *computeBlock = getSuIterateComputeBlock(op);

  std::optional<LoopIndexedOutputPlan> selectedPlan;
  auto visitStore = [&](memref::StoreOp storeOp) {
    Value memref = storeOp.getMemref();
    Value base = arts::ValueAnalysis::stripMemrefViewOps(memref);
    if (isDefinedInside(op.getOperation(), base))
      return WalkResult::advance();

    auto memRefType = dyn_cast<MemRefType>(base.getType());
    if (!memRefType || memRefType.getRank() == 0 || storeOp.getIndices().empty())
      return WalkResult::advance();

    bool indexedByOwner = false;
    for (Value ownerIndex : ownerIndexValues)
      if (isExactOwnerIndex(storeOp.getIndices().front(), ownerIndexValues) ||
          arts::ValueAnalysis::dependsOn(storeOp.getIndices().front(), ownerIndex)) {
        indexedByOwner = true;
        break;
      }
    if (!indexedByOwner)
      return WalkResult::advance();

    SmallVector<int64_t, 4> shape;
    shape.reserve(memRefType.getRank());
    for (int64_t dim : memRefType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return WalkResult::advance();
      shape.push_back(dim);
    }

    selectedPlan = LoopIndexedOutputPlan{base, std::move(shape),
                                         SmallVector<int64_t, 4>{0}};
    return WalkResult::interrupt();
  };

  Block *walkBlock = computeBlock ? computeBlock : &body;
  for (Operation &nested : walkBlock->without_terminator()) {
    if (nested.walk(visitStore).wasInterrupted())
      break;
  }

  return selectedPlan;
}

/// Recover an owner-indexed output plan for direct memref loops whose owner IV
/// maps to any physical output dimension. Unlike findLoopIndexedOutputPlan,
/// this validates all external memref stores in the compute block and rejects
/// mixed output shapes or mixed owner dimensions. That makes it suitable for
/// authoring physical storage layouts from imperfect stencil/update nests.
inline std::optional<LoopIndexedOutputPlan>
findConsistentLoopIndexedOutputPlanWithOwnerDims(SdeSuIterateOp op) {
  if (op.getBody().empty())
    return std::nullopt;

  SmallVector<Value, 4> ownerIndexValues = collectOwnerIndexValues(op);
  if (ownerIndexValues.empty())
    return std::nullopt;

  Block *computeBlock = getSuIterateComputeBlock(op);
  if (!computeBlock)
    return std::nullopt;

  bool rejected = false;
  std::optional<LoopIndexedOutputPlan> selectedPlan;
  auto visitStore = [&](memref::StoreOp storeOp) {
    Value base = arts::ValueAnalysis::stripMemrefViewOps(storeOp.getMemref());
    if (isDefinedInside(op.getOperation(), base))
      return WalkResult::advance();

    auto memRefType = dyn_cast<MemRefType>(base.getType());
    if (!memRefType || memRefType.getRank() == 0 ||
        storeOp.getIndices().empty()) {
      rejected = true;
      return WalkResult::interrupt();
    }

    SmallVector<int64_t, 4> ownerPhysicalDims =
        collectExactOwnerIndexedPhysicalDims(storeOp.getIndices(),
                                             ownerIndexValues);
    if (ownerPhysicalDims.empty()) {
      rejected = true;
      return WalkResult::interrupt();
    }

    SmallVector<int64_t, 4> shape;
    shape.reserve(memRefType.getRank());
    for (int64_t dim : memRefType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        rejected = true;
        return WalkResult::interrupt();
      }
      shape.push_back(dim);
    }

    LoopIndexedOutputPlan candidate{base, std::move(shape),
                                    std::move(ownerPhysicalDims)};
    if (!selectedPlan) {
      selectedPlan = std::move(candidate);
      return WalkResult::advance();
    }

    if (candidate.shape != selectedPlan->shape ||
        candidate.ownerPhysicalDims != selectedPlan->ownerPhysicalDims) {
      rejected = true;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  };

  for (Operation &nested : computeBlock->without_terminator()) {
    if (nested.walk(visitStore).wasInterrupted())
      break;
  }

  if (rejected)
    return std::nullopt;
  return selectedPlan;
}

inline StructuredMemoryEffectSummary collectStructuredMemoryEffects(
    Region &region) {
  StructuredMemoryEffectSummary summary;
  collectStructuredMemoryEffects(region, summary);
  return summary;
}

inline StructuredMemoryEffectSummary collectStructuredMemoryEffects(
    Operation *op) {
  StructuredMemoryEffectSummary summary;
  if (!op)
    return summary;
  for (Region &region : op->getRegions())
    collectStructuredMemoryEffects(region, summary);
  return summary;
}

} // namespace mlir::carts::sde

#endif // ARTS_DIALECT_SDE_ANALYSIS_SDEANALYSISUTILS_H
