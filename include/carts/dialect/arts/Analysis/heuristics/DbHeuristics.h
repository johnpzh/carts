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
///   - DbPartitioning remains the controller that applies those decisions
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_DBHEURISTICS_H
#define ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_DBHEURISTICS_H

#include "carts/dialect/arts/Utils/RuntimeConfig.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace mlir {

class Operation;

namespace arts {

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
  explicit DbHeuristics(const mlir::arts::RuntimeConfig &machine);

  bool isSingleNode() const;
  bool isValid() const;

  static constexpr int64_t kMaxOuterDBs = 1024;
  static constexpr int64_t kMaxDepsPerEDT = 8;
  static constexpr int64_t kMinInnerBytes = 64;

  void setMaxOuterDBs(int64_t value) { maxOuterDBsOverride = value; }
  void setMaxDepsPerEDT(int64_t value) { maxDepsPerEDTOverride = value; }
  void setMinInnerBytes(int64_t value) { minInnerBytesOverride = value; }

  int64_t getMaxOuterDBs() const { return maxOuterDBsOverride; }
  int64_t getMaxDepsPerEDT() const { return maxDepsPerEDTOverride; }
  int64_t getMinInnerBytes() const { return minInnerBytesOverride; }

  void recordDecision(llvm::StringRef heuristic, bool applied,
                      llvm::StringRef rationale, Operation *op,
                      const llvm::StringMap<int64_t> &inputs = {});
  llvm::ArrayRef<HeuristicDecision> getDecisions() const;
  void exportDecisionsToJson(llvm::json::OStream &J) const;
  void clearDecisions() { decisions.clear(); }

private:
  const mlir::arts::RuntimeConfig &machine;
  llvm::SmallVector<HeuristicDecision> decisions;
  int64_t maxOuterDBsOverride = kMaxOuterDBs;
  int64_t maxDepsPerEDTOverride = kMaxDepsPerEDT;
  int64_t minInnerBytesOverride = kMinInnerBytes;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_DBHEURISTICS_H
