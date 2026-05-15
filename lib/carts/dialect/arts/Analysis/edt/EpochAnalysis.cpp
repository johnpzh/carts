///==========================================================================///
/// File: EpochAnalysis.cpp
///
/// Pass-facing epoch analysis facade.
///==========================================================================///

#include "carts/dialect/arts/Analysis/edt/EpochAnalysis.h"

using namespace mlir;
using namespace mlir::arts;

EpochAccessSummary EpochAnalysis::summarizeEpochAccess(EpochOp epoch) const {
  return EpochHeuristics::summarizeEpochAccess(epoch);
}

EpochFusionDecision EpochAnalysis::evaluateEpochFusion(
    EpochOp first, EpochOp second, bool continuationEnabled,
    const EpochAccessSummary *firstSummary,
    const EpochAccessSummary *secondSummary) const {
  return EpochHeuristics::evaluateEpochFusion(
      first, second, continuationEnabled, firstSummary, secondSummary);
}

EpochContinuationDecision EpochAnalysis::evaluateContinuation(
    EpochOp epoch, EpochOp previousEpoch, bool continuationEnabled,
    const EpochAccessSummary *previousSummary,
    const EpochAccessSummary *epochSummary) const {
  return EpochHeuristics::evaluateContinuation(
      epoch, previousEpoch, continuationEnabled, previousSummary, epochSummary);
}
