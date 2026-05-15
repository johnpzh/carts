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
#include "arts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "arts/dialect/sde/IR/SdeDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir::carts::sde {
std::unique_ptr<Pass> createMemoryUnitMaterializationPass();
} // namespace mlir::carts::sde

namespace mlir::arts {

#define GEN_PASS_DECL
#include "arts/dialect/sde/Transforms/Passes.h.inc"

} // namespace mlir::arts

namespace mlir::carts::sde {

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

class SDECostModel;

// --- State passes (IR cleanup before planning) ---
std::unique_ptr<Pass> createScalarForwardingPass();
std::unique_ptr<Pass> createMemoryUnitMaterializationPass();

// --- Dep passes (structural transforms) ---
std::unique_ptr<Pass> createPatternAnalysisPass(
    SDECostModel *costModel = nullptr);
std::unique_ptr<Pass> createLoopInterchangePass();
std::unique_ptr<Pass> createTilingPass(SDECostModel *costModel = nullptr);
std::unique_ptr<Pass> createElementwiseFusionPass();
std::unique_ptr<Pass> createVectorizationPass(
    SDECostModel *costModel = nullptr);
std::unique_ptr<Pass> createIterationSpaceDecompositionPass();

// --- Effect passes (scheduling decisions) ---
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
std::unique_ptr<Pass> createVerifySdeCpsPlanPass();
std::unique_ptr<Pass> createVerifySdeLoweredPass();

} // namespace mlir::carts::sde

#endif // ARTS_DIALECT_SDE_TRANSFORMS_PASSES_H
