///==========================================================================///
/// File: DbHeuristics.h
///
/// DB-specific heuristic policy for partitioning decisions.
///
/// Responsibility split:
///   - DbGraph / DbNode own canonical facts
///   - DbAnalysis exposes those facts
///   - DbHeuristics consumes prepared DB contexts/facts and returns policy
///     decisions only
///   - DB materialization/refinement passes apply those decisions
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_DBHEURISTICS_H
#define ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_DBHEURISTICS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {

class Operation;

namespace carts::arts {

struct HeuristicDecision {
  std::string heuristic;
  bool applied;
  std::string rationale;
  int64_t affectedArtsId;
  int64_t affectedAllocId;
  llvm::SmallVector<int64_t> affectedDbIds;
  std::string sourceLocation;
  llvm::StringMap<int64_t> costModelInputs;
};

class DbHeuristics {
public:
  DbHeuristics() = default;

  void recordDecision(llvm::StringRef heuristic, bool applied,
                      llvm::StringRef rationale, Operation *op,
                      const llvm::StringMap<int64_t> &inputs = {});
  llvm::ArrayRef<HeuristicDecision> getDecisions() const;
  void clearDecisions() { decisions.clear(); }

private:
  static constexpr int64_t kMaxAffectedDbIds = 1024;

  llvm::SmallVector<HeuristicDecision> decisions;
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_DBHEURISTICS_H
