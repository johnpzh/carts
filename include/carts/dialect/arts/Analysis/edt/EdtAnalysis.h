///==========================================================================///
/// File: EdtAnalysis.h
///
/// This file defines the ARTS EDT graph analysis facade.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTANALYSIS_H
#define ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTANALYSIS_H

#include "carts/dialect/arts/Analysis/Analysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"

#include "carts/dialect/arts/Analysis/edt/EdtInfo.h"
#include "carts/dialect/arts/Analysis/graphs/edt/EdtGraph.h"

#include <memory>
#include <shared_mutex>

namespace mlir {
namespace carts::arts {

class LoopAnalysis;

///==========================================================================///
/// EdtAnalysis: per-EDT summaries and pairwise affinity metrics
///==========================================================================///

class EdtAnalysis : public ArtsAnalysis {
public:
  EdtAnalysis(AnalysisManager &AM);

  /// Graph accessor
  EdtGraph &getOrCreateEdtGraph(func::FuncOp func);

  /// Invalidate EDT graph for a specific function
  bool invalidateGraph(func::FuncOp func);

  /// Invalidate internal caches
  void invalidate() override;

  /// Expose sub-analyses so that EdtNode / EdtGraph can reach them
  /// without storing a raw AnalysisManager pointer.
  LoopAnalysis &getLoopAnalysis();

  using ArtsAnalysis::getAnalysisManager;

private:
  /// Per-function graph caches
  llvm::DenseMap<func::FuncOp, std::unique_ptr<EdtGraph>> edtGraphs;
  mutable std::shared_mutex edtGraphMutex;
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTANALYSIS_H
