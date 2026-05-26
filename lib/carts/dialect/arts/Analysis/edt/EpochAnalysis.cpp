///==========================================================================///
/// File: EpochAnalysis.cpp
///
/// Pass-facing epoch analysis facade.
///==========================================================================///

#include "carts/dialect/arts/Analysis/edt/EpochAnalysis.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

EpochAccessSummary EpochAnalysis::summarizeEpochAccess(EpochOp epoch) const {
  return EpochHeuristics::summarizeEpochAccess(epoch);
}

EpochFusionDecision EpochAnalysis::evaluateEpochFusion(
    EpochOp first, EpochOp second, const EpochAccessSummary *firstSummary,
    const EpochAccessSummary *secondSummary) const {
  return EpochHeuristics::evaluateEpochFusion(first, second, firstSummary,
                                              secondSummary);
}
