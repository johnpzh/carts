///==========================================================================///
/// File: StructuredOpAnalysis.h
///
/// Reusable structural loop analysis for SDE scheduling-unit loops.
///
/// This analysis stays on the SDE side of the pipeline so semantic pattern
/// classification can be shared by SDE-owned distribution and tiling planning
/// without depending on target runtime IR.
///
/// Consumes post-LowerAffine IR: `sde.su_iterate` bodies containing
/// `scf.for`, `memref.load`/`store`, and
/// `arith.addi`/`subi`/`muli`/`index_cast`/`constant`. It does **not** match
/// `affine.*` ops; those have been lowered by the `sde-input-normalization`
/// and `initial-cleanup` stages registered in
/// `tools/compile/Compile.cpp`. The `AffineMap` / `AffineExpr` types here
/// come from `mlir/IR/AffineMap.h` (MLIR's IR-level math data structure for
/// linear combinations of dims and symbols) and are reconstructed from
/// post-lowering `arith` chains by `tryGetAffineExpr` and
/// `tryBuildIndexingMap` in `StructuredOpAnalysis.cpp`.
///==========================================================================///

#ifndef ARTS_DIALECT_SDE_ANALYSIS_STRUCTUREDMETHODANALYSIS_H
#define ARTS_DIALECT_SDE_ANALYSIS_STRUCTUREDMETHODANALYSIS_H

#include "carts/dialect/sde/IR/SdeDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::carts::sde {

struct LoopNestInfo {
  SmallVector<Value> ivs;
  Block *innermostBody = nullptr;
  SdeSuIterateOp rootIterOp;
};

struct MemrefAccessEntry {
  Value memref;
  AffineMap indexingMap;
  Operation *op = nullptr;
  bool isRead = false;
};

struct StructuredLoopSummary {
  LoopNestInfo nest;
  SmallVector<MemrefAccessEntry> reads;
  SmallVector<MemrefAccessEntry> writes;
  SmallVector<AffineMap> outputMaps;
  SmallVector<utils::IteratorType> iterTypes;
  SdeStructuredClassification classification =
      SdeStructuredClassification::elementwise;
  bool supportsReductionCarrier = false;
};

struct StructuredNeighborhoodInfo {
  SmallVector<int64_t, 4> ownerDims;
  SmallVector<int64_t, 4> spatialDims;
  SmallVector<int64_t, 4> minOffsets;
  SmallVector<int64_t, 4> maxOffsets;
  SmallVector<int64_t, 4> writeFootprint;
};

struct StructuredOutputLayoutPlan {
  Value root;
  SmallVector<int64_t, 4> shape;
  SmallVector<int64_t, 4> loopDimToPhysicalDim;
  SmallVector<int64_t, 4> physicalDimToLoopDim;
};

/// Analyze one `sde.su_iterate` nest and recover the structural facts
/// needed by higher SDE passes. Returns nullopt when the loop is not a
/// supported perfectly nested memref-based structured loop.
std::optional<StructuredLoopSummary> analyzeStructuredLoop(SdeSuIterateOp op);

/// Recover a runtime-neutral structured neighborhood summary directly from SDE
/// loop-access analysis.
std::optional<StructuredNeighborhoodInfo>
extractNeighborhoodSummary(const StructuredLoopSummary &summary);

/// Recover a static write-backed output layout with a loop-dim to physical-dim
/// map. Returns nullopt when external writes disagree or the layout is not
/// statically shapeable.
std::optional<StructuredOutputLayoutPlan>
findCompatibleOutputLayoutPlan(const StructuredLoopSummary &summary);

/// Convenience wrapper around analyzeStructuredLoop + output layout recovery.
std::optional<StructuredOutputLayoutPlan>
findCompatibleOutputLayoutPlan(SdeSuIterateOp op);

/// Return true when a one-dimensional apparent reduction is only reducing
/// within the owner-local output slice. These loops can use elementwise
/// pipeline planning because no cross-owner reduction carrier is needed.
bool isOwnerLocalPipelineReduction(SdeSuIterateOp op);

/// Return true when a matmul-classified scheduling unit has separate external
/// roots for the logical lhs and rhs input windows. Current owner-slice
/// materialization has one physical dependency view per root, so self-Gram
/// shapes such as correlation must stay on host-whole views until the boundary
/// can represent both windows independently.
bool hasDistinctExternalMatmulInputRoots(SdeSuIterateOp op);

//===----------------------------------------------------------------------===//
// Shared affine decomposition utilities
//===----------------------------------------------------------------------===//

/// Affine expression normalized to one loop dim plus a constant offset.
struct AffineDimOffset {
  std::optional<unsigned> dim;
  int64_t offset = 0;
};

/// Extract a single-dim + constant form from an affine expression.
/// Recursively decomposes through Add expressions.
std::optional<AffineDimOffset> extractDimOffset(AffineExpr expr);

/// Check whether an indexing map contains any non-zero constant stencil
/// offsets of the form `dim + c` where c != 0.
bool hasConstantOffsets(AffineMap map);

} // namespace mlir::carts::sde

#endif // ARTS_DIALECT_SDE_ANALYSIS_STRUCTUREDMETHODANALYSIS_H
