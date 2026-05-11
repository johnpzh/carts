///==========================================================================///
/// File: AnalysisDependencies.h
///
/// Lightweight analysis dependency declarations for CARTS passes.
/// Each pass can declare which analyses it reads and which it invalidates.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_ANALYSISDEPENDENCIES_H
#define ARTS_DIALECT_CORE_ANALYSIS_ANALYSISDEPENDENCIES_H

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace arts {

/// Analysis types that can be depended on.
enum class AnalysisKind : uint8_t {
  DbAnalysis,
  EdtAnalysis,
  EpochAnalysis,
  LoopAnalysis,
  StringAnalysis,
  DbHeuristics,
  EdtHeuristics,
  RuntimeConfig,
};

/// Declares which analyses a pass reads and invalidates.
/// Declarative registry used by AnalysisManager for selective invalidation.
/// Invalidation is active; reads coverage is good for mutating passes but
/// is not yet comprehensive for every pass-local declaration.
struct AnalysisDependencyInfo {
  /// Analyses that this pass reads (queries via AM).
  llvm::ArrayRef<AnalysisKind> reads;

  /// Analyses that this pass invalidates.
  llvm::ArrayRef<AnalysisKind> invalidates;

  /// Whether this pass uses AnalysisManager at all.
  bool usesAnalysisManager() const {
    return !reads.empty() || !invalidates.empty();
  }

  /// Whether this pass is safe to run without AM (no dependencies).
  bool isSafeWithoutAM() const { return reads.empty() && invalidates.empty(); }
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_ANALYSISDEPENDENCIES_H
