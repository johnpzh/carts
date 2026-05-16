///==========================================================================///
/// File: EdtAnalysis.cpp
///==========================================================================///

#include "carts/dialect/arts/Analysis/edt/EdtAnalysis.h"
#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/graphs/edt/EdtGraph.h"
#include "carts/dialect/arts/Analysis/graphs/edt/EdtNode.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

///==========================================================================///
/// EdtAnalysis
///==========================================================================///

EdtAnalysis::EdtAnalysis(AnalysisManager &AM) : ArtsAnalysis(AM) {
  assert(AM.getModule() && "Module is required");
}

EdtGraph &EdtAnalysis::getOrCreateEdtGraph(func::FuncOp func) {
  // Fast path: shared lock for read.
  {
    std::shared_lock<std::shared_mutex> readLock(edtGraphMutex);
    auto it = edtGraphs.find(func);
    if (it != edtGraphs.end())
      return *it->second;
  }
  // Build the graph outside the lock to avoid holding edtGraphMutex while
  // calling into DbAnalysis (which uses its own graphMutex).
  DbGraph &db = getAnalysisManager().getDbAnalysis().getOrCreateGraph(func);

  // Slow path: exclusive lock for write.
  std::unique_lock<std::shared_mutex> writeLock(edtGraphMutex);
  // Re-check after acquiring write lock (another thread may have created it).
  auto &graph = edtGraphs[func];
  if (graph)
    return *graph;

  graph = std::make_unique<EdtGraph>(func, &db, this);
  graph->build();
  return *graph;
}

bool EdtAnalysis::invalidateGraph(func::FuncOp func) {
  std::unique_lock<std::shared_mutex> writeLock(edtGraphMutex);
  auto it = edtGraphs.find(func);
  if (it != edtGraphs.end()) {
    if (it->second)
      it->second->invalidate();
    edtGraphs.erase(it);
    return true;
  }
  return false;
}

void EdtAnalysis::invalidate() {
  std::unique_lock<std::shared_mutex> writeLock(edtGraphMutex);
  for (auto &kv : edtGraphs)
    if (kv.second)
      kv.second->invalidate();
  edtGraphs.clear();
}

EdtNode *EdtAnalysis::getEdtNode(EdtOp op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func)
    return nullptr;
  return getOrCreateEdtGraph(func).getEdtNode(op);
}

LoopAnalysis &EdtAnalysis::getLoopAnalysis() {
  return getAnalysisManager().getLoopAnalysis();
}
