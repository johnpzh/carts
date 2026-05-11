///==========================================================================///
/// File: Passes.h
///
/// Pass declarations for the SDE (Structured Decomposition Environment)
/// dialect.
///
/// This header is self-contained: it includes all dialect headers required
/// by dependentDialects in SDE Passes.td so that .h.inc can be safely
/// included without manual per-file dependency management.
///==========================================================================///

#ifndef ARTS_DIALECT_SDE_TRANSFORMS_PASSES_H
#define ARTS_DIALECT_SDE_TRANSFORMS_PASSES_H

#include "arts/Dialect.h"
#include "arts/dialect/sde/IR/SdeDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::arts {

#define GEN_PASS_DECL
#include "arts/dialect/sde/Transforms/Passes.h.inc"

} // namespace mlir::arts

namespace mlir::arts::sde {

/// Ensure a region has at least one block, creating an empty one if needed.
inline Block &ensureBlock(Region &region) {
  if (region.empty())
    region.push_back(new Block());
  return region.front();
}

/// Strip the operand segment sizes attribute when cloning/recreating an
/// SdeSuIterateOp so that the builder can recompute it from operands.
inline NamedAttrList getRewrittenAttrs(SdeSuIterateOp op) {
  NamedAttrList attrs(op->getAttrs());
  attrs.erase(op.getOperandSegmentSizesAttrName().getValue());
  return attrs;
}

/// Get the computation block inside su_iterate, looking through an optional
/// cu_region <parallel> wrapper. Returns the innermost block containing
/// user compute ops.
inline Block *getSuIterateComputeBlock(SdeSuIterateOp op) {
  Block &body = op.getBody().front();
  // Check if the body starts with a cu_region <parallel> wrapper.
  for (Operation &inner : body.without_terminator()) {
    if (auto cuRegion = dyn_cast<SdeCuRegionOp>(inner);
        cuRegion && cuRegion.getKind() == SdeCuKind::parallel)
      return &cuRegion.getBody().front();
  }
  return &body;
}

class SDECostModel;

// --- State passes (IR form changes) ---
std::unique_ptr<Pass> createRaiseToLinalgPass();
std::unique_ptr<Pass> createRaiseToTensorPass();
std::unique_ptr<Pass> createRaiseMemrefToTensorPass();
std::unique_ptr<Pass> createScalarForwardingPass();
std::unique_ptr<Pass> createLowerToMemrefPass();
std::unique_ptr<Pass> createConvertToCodeletPass();
std::unique_ptr<Pass> createTensorCleanupPass();
std::unique_ptr<Pass> createTokenModeRefinementPass();

// --- Dep passes (structural transforms) ---
std::unique_ptr<Pass> createLoopInterchangePass();
std::unique_ptr<Pass> createTilingPass(SDECostModel *costModel = nullptr);
std::unique_ptr<Pass>
createStructuredSummariesPass(SDECostModel *costModel = nullptr);
std::unique_ptr<Pass> createElementwiseFusionPass();
std::unique_ptr<Pass> createIterationSpaceDecompositionPass();
std::unique_ptr<Pass> createVectorizationPass();

// --- Effect passes (scheduling decisions) ---
std::unique_ptr<Pass>
createScopeSelectionPass(SDECostModel *costModel = nullptr);
std::unique_ptr<Pass>
createScheduleRefinementPass(SDECostModel *costModel = nullptr);
std::unique_ptr<Pass> createChunkOptPass(SDECostModel *costModel = nullptr);
std::unique_ptr<Pass>
createReductionStrategyPass(SDECostModel *costModel = nullptr);
std::unique_ptr<Pass>
createDistributionPlanningPass(SDECostModel *costModel = nullptr);
std::unique_ptr<Pass>
createBarrierEliminationPass(SDECostModel *costModel = nullptr);

// --- Conversion passes ---
std::unique_ptr<Pass>
createConvertOpenMPToSdePass(SDECostModel *costModel = nullptr);
std::unique_ptr<Pass> createConvertSdeToArtsPass();

// --- Verification ---
std::unique_ptr<Pass> createVerifySdeLoweredPass();

} // namespace mlir::arts::sde

#endif // ARTS_DIALECT_SDE_TRANSFORMS_PASSES_H
