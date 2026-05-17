///==========================================================================///
/// File: EdtDataFlowAnalysis.h
///
/// Dataflow-based dependency analysis that produces EDT-level edges.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTDATAFLOWANALYSIS_H
#define ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTDATAFLOWANALYSIS_H

#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbGraph.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/dialect/arts/Analysis/graphs/edt/EdtEdge.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
namespace carts::arts {

class DbAliasAnalysis;
class AnalysisManager;

struct EdtDep {
  EdtOp from;
  EdtOp to;
  SmallVector<DbEdge, 2> edges;
};

/// Performs a lightweight dataflow analysis over EDT regions to determine
/// dependencies between tasks based on their datablock acquires.
class EdtDataFlowAnalysis {
public:
  EdtDataFlowAnalysis(DbGraph *dbGraph, AnalysisManager *AM);

  SmallVector<EdtDep, 16> run(func::FuncOp func);

private:
  struct Environment {
    llvm::DenseMap<DbAllocNode *, llvm::SetVector<DbAcquireNode *>> writers;
    llvm::DenseMap<DbAllocNode *, llvm::SetVector<DbAcquireNode *>> readers;
  };

  std::pair<Environment, bool> processRegion(Region &region, Environment &env);
  std::pair<Environment, bool> processEdt(EdtOp edtOp, Environment &env);
  std::pair<Environment, bool> processEpoch(EpochOp epochOp, Environment &env);
  std::pair<Environment, bool> processIf(scf::IfOp ifOp, Environment &env);
  std::pair<Environment, bool> processFor(scf::ForOp forOp, Environment &env);

  SmallVector<DbAcquireNode *> findDefinitions(DbAcquireNode *acquire,
                                               Environment &env);
  bool unionInto(Environment &target, const Environment &source);
  std::optional<DbDepType> classifyDep(DbAcquireNode *producer,
                                       DbAcquireNode *consumer);

  void recordDep(DbAcquireNode *producer, DbAcquireNode *consumer,
                 DbDepType depType);

  DbGraph *dbGraph = nullptr;
  DbAliasAnalysis *aliasAA = nullptr;

  llvm::DenseMap<std::pair<Operation *, Operation *>,
                 llvm::SetVector<DbEdge, llvm::SmallVector<DbEdge, 2>>>
      dependencyMap;
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_EDT_EDTDATAFLOWANALYSIS_H
