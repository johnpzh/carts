///==========================================================================///
/// File: EdtHeuristics.cpp
///
/// EDT heuristic policy: distribution strategy and kind selection.
///==========================================================================///

#include "carts/dialect/arts/Analysis/heuristics/EdtHeuristics.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

const RuntimeConfig &EdtHeuristics::getRuntimeConfig() const {
  return getAnalysisManager().getRuntimeConfig();
}

DistributionStrategy
EdtHeuristics::chooseStrategy(EdtConcurrency concurrency) const {
  return DistributionHeuristics::analyzeStrategy(concurrency,
                                                 &getRuntimeConfig());
}

EdtDistributionKind
EdtHeuristics::chooseKind(const DistributionStrategy &strategy,
                          EdtDistributionPattern pattern) const {
  return DistributionHeuristics::selectDistributionKind(strategy, pattern);
}

ParallelismDecision EdtHeuristics::resolveParallelism() const {
  return DistributionHeuristics::resolveParallelismFromMachine(
      &getRuntimeConfig());
}

std::optional<WorkerConfig>
EdtHeuristics::resolveWorkerConfig(EdtOp edt) const {
  return DistributionHeuristics::resolveWorkerConfig(edt,
                                                     &getRuntimeConfig());
}
