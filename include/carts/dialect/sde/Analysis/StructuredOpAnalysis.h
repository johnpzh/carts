///==========================================================================///
/// File: StructuredOpAnalysis.h
///
/// Reusable structural loop analysis for SDE scheduling-unit loops.
///
/// This analysis stays on the SDE side of the pipeline so semantic pattern
/// classification can be shared by SDE-owned distribution and tiling planning
/// without depending on target runtime IR.
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
