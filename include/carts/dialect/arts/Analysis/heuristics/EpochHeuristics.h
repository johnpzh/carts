#ifndef ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_EPOCHHEURISTICS_H
#define ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_EPOCHHEURISTICS_H

#include "carts/Dialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <string>

namespace mlir {
namespace carts::arts {

/// Effective epoch-local memory use after ignoring structural acquires that are
/// never consumed by loads/stores or equivalent side-effecting users.
enum class EpochAccessMode : uint8_t {
  None = 0,
  Read = 1,
  Write = 2,
  ReadWrite = 3
};

inline EpochAccessMode combineEpochAccessModes(EpochAccessMode lhs,
                                               EpochAccessMode rhs) {
  return static_cast<EpochAccessMode>(static_cast<uint8_t>(lhs) |
                                      static_cast<uint8_t>(rhs));
}

inline bool epochAccessModeHasWrite(EpochAccessMode mode) {
  return (static_cast<uint8_t>(mode) &
          static_cast<uint8_t>(EpochAccessMode::Write)) != 0;
}

struct EpochAccessSummary {
  llvm::DenseMap<Operation *, EpochAccessMode> allocModes;
  llvm::DenseSet<Operation *> acquireAllocs;

  EpochAccessMode lookup(Operation *alloc) const {
    auto it = allocModes.find(alloc);
    return it == allocModes.end() ? EpochAccessMode::None : it->second;
  }

  void record(Operation *alloc, EpochAccessMode mode) {
    if (!alloc || mode == EpochAccessMode::None)
      return;
    allocModes[alloc] = combineEpochAccessModes(lookup(alloc), mode);
  }

  void mergeFrom(const EpochAccessSummary &other) {
    for (const auto &entry : other.allocModes)
      record(entry.first, entry.second);
    acquireAllocs.insert(other.acquireAllocs.begin(),
                         other.acquireAllocs.end());
  }

  bool empty() const { return allocModes.empty(); }
};

struct EpochFusionDecision {
  bool shouldFuse = false;
  std::string rationale;
};

class EpochHeuristics {
public:
  /// Summarize the effective alloc-level accesses inside one epoch. Unused
  /// structural acquires are intentionally ignored so fusion policy can reason
  /// about the accesses that actually execute.
  static EpochAccessSummary summarizeEpochAccess(EpochOp epoch);

  /// Decide whether two consecutive epochs may be fused.
  static EpochFusionDecision
  evaluateEpochFusion(EpochOp first, EpochOp second,
                      const EpochAccessSummary *firstSummary = nullptr,
                      const EpochAccessSummary *secondSummary = nullptr);
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_EPOCHHEURISTICS_H
