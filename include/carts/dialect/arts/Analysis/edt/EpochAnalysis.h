///==========================================================================///
/// File: EpochAnalysis.h
///
/// Pass-facing facade for epoch scheduling/structural policy queries.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_EDT_EPOCHANALYSIS_H
#define ARTS_DIALECT_CORE_ANALYSIS_EDT_EPOCHANALYSIS_H

#include "carts/dialect/arts/Analysis/Analysis.h"
#include "carts/dialect/arts/Analysis/heuristics/EpochHeuristics.h"

namespace mlir {
namespace arts {

class EpochAnalysis : public ArtsAnalysis {
public:
  explicit EpochAnalysis(AnalysisManager &manager) : ArtsAnalysis(manager) {}

  void invalidate() override {}

  EpochAccessSummary summarizeEpochAccess(EpochOp epoch) const;
  EpochFusionDecision
  evaluateEpochFusion(EpochOp first, EpochOp second,
                      bool continuationEnabled = true,
                      const EpochAccessSummary *firstSummary = nullptr,
                      const EpochAccessSummary *secondSummary = nullptr) const;
  EpochContinuationDecision
  evaluateContinuation(EpochOp epoch, EpochOp previousEpoch = nullptr,
                       bool continuationEnabled = true,
                       const EpochAccessSummary *previousSummary = nullptr,
                       const EpochAccessSummary *epochSummary = nullptr) const;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_EDT_EPOCHANALYSIS_H
