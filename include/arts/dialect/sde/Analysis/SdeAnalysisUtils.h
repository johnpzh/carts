///==========================================================================///
/// File: SdeAnalysisUtils.h
///
/// Small SDE-owned helpers shared by SDE analyses and transforms.
///==========================================================================///

#ifndef ARTS_DIALECT_SDE_ANALYSIS_SDEANALYSISUTILS_H
#define ARTS_DIALECT_SDE_ANALYSIS_SDEANALYSISUTILS_H

#include "arts/dialect/sde/IR/SdeDialect.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::arts::sde {

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

/// Trace a carrier tensor operand back to its backing memref root through
/// mu_memref_to_tensor and tensor view/cast chains.
inline Value traceCarrierTensorToMemrefRoot(Value tensorOperand) {
  Value cur = tensorOperand;
  while (cur) {
    if (auto muOp = cur.getDefiningOp<SdeMuMemrefToTensorOp>())
      return ValueAnalysis::stripMemrefViewOps(muOp.getMemref());
    if (auto castOp = cur.getDefiningOp<tensor::CastOp>()) {
      cur = castOp.getSource();
      continue;
    }
    if (auto sliceOp = cur.getDefiningOp<tensor::ExtractSliceOp>()) {
      cur = sliceOp.getSource();
      continue;
    }
    break;
  }
  return {};
}

/// Root-level memory effects for an SDE structured region. This intentionally
/// stays in SDE because carrier tensors and DPS linalg operands are SDE
/// scheduling facts, not Core DB graph facts.
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
/// Core may later materialize it as physical DB layout.
struct LoopIndexedOutputPlan {
  Value root;
  SmallVector<int64_t, 4> shape;
};

inline bool isDefinedInside(Operation *ancestor, Value value) {
  if (!ancestor || !value)
    return false;
  Operation *def = value.getDefiningOp();
  return def && ancestor->isAncestor(def);
}

inline bool hasUnmodeledMemoryEffect(Operation *op) {
  if (!op || op->hasTrait<OpTrait::IsTerminator>() || isa<SdeYieldOp>(op))
    return false;

  if (isa<memref::LoadOp, memref::StoreOp, linalg::GenericOp>(op))
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
          ValueAnalysis::stripMemrefViewOps(loadOp.getMemref()));
      return;
    }

    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      summary.writes.insert(
          ValueAnalysis::stripMemrefViewOps(storeOp.getMemref()));
      return;
    }

    if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      for (Value input : generic.getDpsInputs()) {
        Value root = traceCarrierTensorToMemrefRoot(input);
        if (root)
          summary.reads.insert(root);
      }
      for (Value output : generic.getDpsInits()) {
        Value root = traceCarrierTensorToMemrefRoot(output);
        if (root)
          summary.writes.insert(root);
      }
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

inline std::optional<LoopIndexedOutputPlan>
findLoopIndexedOutputPlan(SdeSuIterateOp op) {
  if (op.getBody().empty())
    return std::nullopt;

  Block &body = op.getBody().front();
  auto ivs = op.getLoopInductionVars();
  if (!ivs || ivs->empty())
    return std::nullopt;
  Value loopIv = ivs->front();

  SmallVector<Value, 4> ownerIndexValues{loopIv};
  Block *computeBlock = getSuIterateComputeBlock(op);
  if (computeBlock && computeBlock != &body) {
    if (auto cuRegion = dyn_cast_or_null<SdeCuRegionOp>(
            computeBlock->getParentOp())) {
      for (auto [idx, iterArg] : llvm::enumerate(cuRegion.getIterArgs())) {
        if (idx >= computeBlock->getNumArguments())
          break;
        if (iterArg == loopIv || ValueAnalysis::dependsOn(iterArg, loopIv))
          ownerIndexValues.push_back(computeBlock->getArgument(idx));
      }
    }
  }

  std::optional<LoopIndexedOutputPlan> selectedPlan;
  auto visitStore = [&](memref::StoreOp storeOp) {
    Value memref = storeOp.getMemref();
    Value base = ValueAnalysis::stripMemrefViewOps(memref);
    if (isDefinedInside(op.getOperation(), base))
      return WalkResult::advance();

    auto memRefType = dyn_cast<MemRefType>(base.getType());
    if (!memRefType || memRefType.getRank() == 0 || storeOp.getIndices().empty())
      return WalkResult::advance();

    bool indexedByOwner = false;
    for (Value ownerIndex : ownerIndexValues) {
      if (ValueAnalysis::dependsOn(storeOp.getIndices().front(), ownerIndex)) {
        indexedByOwner = true;
        break;
      }
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

    selectedPlan = LoopIndexedOutputPlan{base, std::move(shape)};
    return WalkResult::interrupt();
  };

  Block *walkBlock = computeBlock ? computeBlock : &body;
  for (Operation &nested : walkBlock->without_terminator()) {
    if (nested.walk(visitStore).wasInterrupted())
      break;
  }

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

} // namespace mlir::arts::sde

#endif // ARTS_DIALECT_SDE_ANALYSIS_SDEANALYSISUTILS_H
