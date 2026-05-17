///==========================================================================///
/// File: EdtGraph.cpp
/// Implementation of EdtGraph for EDT analysis and optimization.
///==========================================================================///

#include "carts/dialect/arts/Analysis/graphs/edt/EdtGraph.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/edt/EdtAnalysis.h"
#include "carts/dialect/arts/Analysis/edt/EdtDataFlowAnalysis.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/dialect/arts/Analysis/graphs/edt/EdtEdge.h"
#include "carts/dialect/arts/Analysis/graphs/edt/EdtNode.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "carts/dialect/arts/Analysis/loop/LoopNode.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/LocationMetadata.h"
#include "carts/utils/OperationAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_graph);

namespace {
void sortEdtNodesByHierId(SmallVectorImpl<EdtNode *> &nodes) {
  llvm::sort(nodes, [](EdtNode *lhs, EdtNode *rhs) {
    return lhs->getHierId() < rhs->getHierId();
  });
}
} // namespace

EdtGraph::EdtGraph(func::FuncOp func, DbGraph *dbGraph, EdtAnalysis *EA)
    : func(func), dbGraph(dbGraph), edtAnalysis(EA) {
  ARTS_INFO("Creating EDT graph for function: " << func.getName().str());
}

/// Build task graph nodes and DB-induced dependency edges.
void EdtGraph::build() {
  if (isBuilt.load(std::memory_order_acquire) &&
      !needsRebuild.load(std::memory_order_acquire))
    return;
  ARTS_INFO("Building EDT graph");
  invalidate();
  assert(dbGraph && "DbGraph is required to build EdtGraph");
  collectNodes();
  linkEdtsToLoops();
  buildDependencies();
  isBuilt.store(true, std::memory_order_release);
  needsRebuild.store(false, std::memory_order_release);
}

void EdtGraph::invalidate() {
  edtNodes.clear();
  edges.clear();
  nodes.clear();
  isBuilt.store(false, std::memory_order_release);
  needsRebuild.store(true, std::memory_order_release);
}

NodeBase *EdtGraph::getOrCreateNode(Operation *op) {
  if (auto edtOp = dyn_cast<EdtOp>(op)) {
    auto it = edtNodes.find(edtOp);
    if (it != edtNodes.end())
      return it->second.get();
    auto newNode = std::make_unique<EdtNode>(edtOp);
    newNode->setHierId("T" + std::to_string(edtNodes.size() + 1));
    NodeBase *ptr = newNode.get();
    edtNodes[edtOp] = std::move(newNode);
    nodes.push_back(ptr);
    return ptr;
  }
  return nullptr;
}

NodeBase *EdtGraph::getNode(Operation *op) const {
  if (auto edtOp = dyn_cast<EdtOp>(op)) {
    auto it = edtNodes.find(edtOp);
    return it != edtNodes.end() ? it->second.get() : nullptr;
  }
  return nullptr;
}

EdtNode *EdtGraph::getEdtNode(EdtOp edt) const {
  auto it = edtNodes.find(edt);
  return it != edtNodes.end() ? it->second.get() : nullptr;
}

void EdtGraph::forEachNode(const std::function<void(NodeBase *)> &fn) const {
  for (auto *node : nodes)
    fn(node);
}

bool EdtGraph::addEdge(NodeBase *from, NodeBase *to, EdgeBase *edge) {
  auto key = std::make_pair(from, to);
  if (edges.count(key))
    return false;
  edges[key] = std::unique_ptr<EdgeBase>(edge);
  from->addOutEdge(edge);
  to->addInEdge(edge);
  return true;
}

bool EdtGraph::isEdtReachable(EdtOp fromOp, EdtOp toOp) {
  EdtNode *from = static_cast<EdtNode *>(getNode(fromOp.getOperation()));
  EdtNode *to = static_cast<EdtNode *>(getNode(toOp.getOperation()));
  if (!from || !to)
    return false;

  DenseSet<NodeBase *> visited;
  SmallVector<NodeBase *, 8> worklist;
  visited.insert(from);
  worklist.push_back(from);

  while (!worklist.empty()) {
    NodeBase *current = worklist.pop_back_val();
    for (auto *edge : current->getOutEdges()) {
      NodeBase *next = edge->getTo();
      if (next == to)
        return true;
      if (visited.insert(next).second)
        worklist.push_back(next);
    }
  }
  return false;
}

void EdtGraph::getDeterministicTopologicalOrder(
    SmallVectorImpl<EdtNode *> &topoOrder,
    SmallVectorImpl<EdtNode *> &leftoverNodes) const {
  topoOrder.clear();
  leftoverNodes.clear();

  SmallVector<EdtNode *, 16> allNodes;
  forEachNode([&](NodeBase *base) {
    if (auto *edtNode = dyn_cast<EdtNode>(base))
      allNodes.push_back(edtNode);
  });

  if (allNodes.empty())
    return;

  DenseMap<EdtNode *, unsigned> inDegree;
  for (auto *node : allNodes)
    inDegree[node] = 0;

  for (auto *node : allNodes) {
    for (auto *edge : node->getInEdges()) {
      auto *fromNode = dyn_cast<EdtNode>(edge->getFrom());
      if (fromNode && inDegree.count(fromNode))
        ++inDegree[node];
    }
  }

  SmallVector<EdtNode *, 16> worklist;
  for (auto *node : allNodes) {
    if (inDegree[node] == 0)
      worklist.push_back(node);
  }
  sortEdtNodesByHierId(worklist);

  topoOrder.reserve(allNodes.size());
  while (!worklist.empty()) {
    EdtNode *current = worklist.pop_back_val();
    topoOrder.push_back(current);

    SmallVector<EdtNode *, 8> successors;
    for (auto *edge : current->getOutEdges()) {
      auto *toNode = dyn_cast<EdtNode>(edge->getTo());
      if (toNode && inDegree.count(toNode)) {
        --inDegree[toNode];
        if (inDegree[toNode] == 0)
          successors.push_back(toNode);
      }
    }

    sortEdtNodesByHierId(successors);
    worklist.append(successors.begin(), successors.end());
  }

  if (topoOrder.size() == allNodes.size())
    return;

  DenseSet<EdtNode *> visitedNodes(topoOrder.begin(), topoOrder.end());
  leftoverNodes.reserve(allNodes.size() - topoOrder.size());
  for (auto *node : allNodes) {
    if (!visitedNodes.contains(node))
      leftoverNodes.push_back(node);
  }
  sortEdtNodesByHierId(leftoverNodes);
}

bool EdtGraph::areEdtsIndependent(EdtOp a, EdtOp b) {
  if (!a || !b || a == b)
    return false;

  EdtNode *nodeA = getEdtNode(a);
  EdtNode *nodeB = getEdtNode(b);
  if (!nodeA || !nodeB)
    return false;

  Operation *opA = nodeA->getOp();
  Operation *opB = nodeB->getOp();
  if (!opA || !opB)
    return false;

  return !DbUtils::hasSharedWritableRootConflict(opA, opB);
}

llvm::json::Value EdtGraph::exportToJsonValue() const {
  using namespace llvm::json;

  /// ArtsMate-compatible EDT entities export
  Array edts;

  forEachNode([&](NodeBase *node) {
    auto *edtNode = cast<EdtNode>(node);
    const EdtInfo &info = edtNode->getInfo();
    EdtOp edtOp = edtNode->getEdtOp();

    Object edt;

    /// ARTS ID
    int64_t artsId = getArtsId(edtNode->getOp());
    if (artsId != 0)
      edt["arts_id"] = artsId;

    /// EDT type (task for now)
    edt["type"] = "task";

    /// Concurrency scope
    edt["concurrency"] = "intranode";

    /// Associated loops (EDT ↔ Loop bidirectional relationship)
    if (!edtNode->getAssociatedLoops().empty()) {
      Array loopIds;
      for (LoopNode *loop : edtNode->getAssociatedLoops()) {
        int64_t loopId = getArtsId(loop->getOp());
        if (loopId != 0)
          loopIds.push_back(loopId);
      }
      if (!loopIds.empty())
        edt["loop_ids"] = std::move(loopIds);
    }

    /// Loop metadata (always false — metadata infrastructure removed)
    edt["in_loop"] = false;

    /// Source location
    auto loc = LocationMetadata::fromLocation(edtOp->getLoc());
    if (loc.isValid())
      edt["loc"] = loc.file + ":" + std::to_string(loc.line);
    else
      edt["loc"] = "unknown";

    /// Static analysis data from EdtInfo
    Object staticAnalysis;
    staticAnalysis["max_loop_depth"] = (int64_t)info.maxLoopDepth;
    staticAnalysis["order_index"] = (int64_t)info.orderIndex;
    edt["static_analysis"] = std::move(staticAnalysis);

    edts.push_back(std::move(edt));
  });

  return llvm::json::Value(std::move(edts));
}

/// Collect all arts.edt operations in the function and create task nodes.
void EdtGraph::collectNodes() {
  ARTS_INFO("Phase 1 - Collecting EDT nodes");

  func.walk([&](EdtOp edtOp) { getOrCreateNode(edtOp.getOperation()); });

  ARTS_INFO("Collected " << edtNodes.size() << " tasks");
}

/// Link EDTs to their enclosing loops (if any)
void EdtGraph::linkEdtsToLoops() {
  ARTS_INFO("Phase 1.5 - Linking EDTs to enclosing loops");

  if (!dbGraph) {
    ARTS_WARN("  No DbGraph available, skipping loop linking");
    return;
  }

  /// Get LoopAnalysis through DbGraph -> DbAnalysis
  auto *dbAnalysis = dbGraph->getAnalysis();
  if (!dbAnalysis) {
    ARTS_WARN("  No DbAnalysis available, skipping loop linking");
    return;
  }

  auto *loopAnalysis = dbAnalysis->getLoopAnalysis();
  if (!loopAnalysis) {
    ARTS_WARN("  No LoopAnalysis available, skipping loop linking");
    return;
  }

  size_t linkedCount = 0;
  for (auto &[edtOp, edtNode] : edtNodes) {
    Operation *op = edtOp.getOperation();

    SmallVector<LoopNode *, 4> enclosingLoops;
    loopAnalysis->collectEnclosingLoops(op, enclosingLoops);
    if (!enclosingLoops.empty()) {
      edtNode->setAssociatedLoops(enclosingLoops);
      linkedCount += enclosingLoops.size();
      ARTS_DEBUG("  Linked " << edtNode->getHierId() << " to "
                             << enclosingLoops.size() << " enclosing loops");
    }
  }

  ARTS_INFO("Linked " << linkedCount << " loop references to EDTs");
}

/// Build inter-EDT edges by connecting release sites in one EDT to acquire
/// sites in another EDT for the same root DbAlloc.
void EdtGraph::buildDependencies() {
  ARTS_INFO("Phase 2 - Building EDT dependencies");
  assert(dbGraph && "DbGraph is required to build EDT dependencies");
  assert(edtAnalysis && "EdtAnalysis is required to build EDT dependencies");
  EdtDataFlowAnalysis dataflow(dbGraph, &edtAnalysis->getAnalysisManager());
  auto deps = dataflow.run(func);

  unsigned created = 0;
  for (auto &dep : deps) {
    if (!dep.from || !dep.to || dep.edges.empty())
      continue;
    EdtNode *fromNode = getEdtNode(dep.from);
    EdtNode *toNode = getEdtNode(dep.to);
    if (!fromNode || !toNode)
      continue;

    auto *edge = new EdtDepEdge(fromNode, toNode, dep.edges.front());
    if (addEdge(fromNode, toNode, edge)) {
      ++created;
    } else {
      delete edge;
    }
  }

  ARTS_INFO("Created " << created << " EDT dependency edges");
}
