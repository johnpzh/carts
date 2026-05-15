///==========================================================================///
/// File: EdtAnalysis.h
///
/// This pass performs a comprehensive analysis on arts.edt operations.
/// It also provides helpers to analyze regions and find dependencies,
/// parameters, and constants.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTANALYSIS_H
#define ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTANALYSIS_H

#include "carts/dialect/arts/Analysis/Analysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include "carts/dialect/arts/Analysis/edt/EdtInfo.h"
#include "carts/dialect/arts/Analysis/graphs/edt/EdtGraph.h"

#include <atomic>
#include <optional>
#include <shared_mutex>

namespace mlir {
namespace arts {

class LoopAnalysis;

///==========================================================================///
/// ReductionOp: recognized reduction operation kinds
///==========================================================================///

enum class ReductionOp {
  ADD,
  MIN,
  MAX,
  OR,
  AND,
  XOR,
};

///==========================================================================///
/// EdtAnalysis: per-EDT summaries and pairwise affinity metrics
///==========================================================================///

class EdtAnalysis : public ArtsAnalysis {
public:
  EdtAnalysis(AnalysisManager &AM);

  /// Build summaries for all EDTs in the module.
  void analyze();

  /// Debug printing
  void print(func::FuncOp func, llvm::raw_ostream &os);
  void toJson(func::FuncOp func, llvm::raw_ostream &os);

  /// Get EDT-level summary distribution pattern.
  std::optional<EdtDistributionPattern> getEdtDistributionPattern(EdtOp edt);

  /// Get DB access pattern inferred from EDT-level loop access analysis.
  /// Returns std::nullopt when the allocation was not observed in analyzed
  /// loops.
  std::optional<DbAccessPattern> getAllocAccessPattern(Operation *allocOp);

  /// Iterate all analyzed DB allocation access patterns.
  void forEachAllocAccessPattern(
      llvm::function_ref<void(Operation *, DbAccessPattern)> fn);

  /// Graph accessor
  EdtGraph &getOrCreateEdtGraph(func::FuncOp func);

  /// Invalidate EDT graph for a specific function
  bool invalidateGraph(func::FuncOp func);

  /// Invalidate internal caches
  void invalidate() override;

  /// Convenience: get EDT node by op (derives parent func internally).
  EdtNode *getEdtNode(EdtOp op);

  /// Expose sub-analyses so that EdtNode / EdtGraph can reach them
  /// without storing a raw AnalysisManager pointer.
  LoopAnalysis &getLoopAnalysis();

  //===--------------------------------------------------------------------===//
  /// Reduction pattern analysis.
  //===--------------------------------------------------------------------===//

  /// Try to classify a binary arithmetic operation as a reduction operation.
  static std::optional<ReductionOp> classifyReductionOp(Operation *op);

  /// Return a human-readable string name for a ReductionOp kind.
  static StringRef reductionOpName(ReductionOp op);

  /// Check whether a reduction operation is suitable for atomic lowering.
  static bool isAtomicCapable(ReductionOp op);

  /// Match a load-modify-store reduction pattern on a DbRefOp result within
  /// an EDT body.
  static std::optional<ReductionOp> matchLoadModifyStore(Value dbRefResult,
                                                         Region &edtBody);

  using ArtsAnalysis::getAnalysisManager;

private:
  /// Order index per EDT in a function
  DenseMap<EdtOp, unsigned> edtOrderIndex;
  DenseMap<Operation *, EdtDistributionPattern> edtPatternByOp;
  DenseMap<Operation *, DbAccessPattern> allocPatternByOp;
  std::atomic<bool> analyzed{false};

  /// Per-function graph caches
  llvm::DenseMap<func::FuncOp, std::unique_ptr<EdtGraph>> edtGraphs;
  mutable std::shared_mutex edtGraphMutex;

  void analyzeFunc(func::FuncOp func);
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTANALYSIS_H
