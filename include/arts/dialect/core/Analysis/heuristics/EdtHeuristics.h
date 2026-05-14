///==========================================================================///
/// File: EdtHeuristics.h
///
/// EDT-specific heuristic policy and analysis façade for distribution
/// decisions.
///
/// Pass-facing policy queries should go through this interface instead of
/// calling DistributionHeuristics directly. This keeps machine/loop/DB-backed
/// distribution analysis centralized behind AnalysisManager.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_EDTHEURISTICS_H
#define ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_EDTHEURISTICS_H

#include "arts/dialect/core/Analysis/Analysis.h"
#include "arts/dialect/core/Analysis/heuristics/DistributionHeuristics.h"
#include <optional>
#include <string>

namespace mlir {
namespace arts {

class EdtHeuristics : public ArtsAnalysis {
public:
  explicit EdtHeuristics(AnalysisManager &manager) : ArtsAnalysis(manager) {}

  void invalidate() override {}

  DistributionStrategy chooseStrategy(EdtConcurrency concurrency) const;
  EdtDistributionKind chooseKind(const DistributionStrategy &strategy,
                                 EdtDistributionPattern pattern) const;
  ParallelismDecision resolveParallelism() const;
  std::optional<WorkerConfig> resolveWorkerConfig(EdtOp edt) const;

private:
  const RuntimeConfig &getRuntimeConfig() const;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_EDTHEURISTICS_H
