///==========================================================================///
/// File: EdtAnalysis.cpp
///==========================================================================///

#include "arts/dialect/core/Analysis/edt/EdtAnalysis.h"
#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/db/DbAnalysis.h"
#include "arts/dialect/core/Analysis/graphs/edt/EdtGraph.h"
#include "arts/dialect/core/Analysis/graphs/edt/EdtNode.h"
#include "arts/dialect/core/Analysis/loop/LoopAnalysis.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::arts;

///==========================================================================///
/// EdtAnalysis
///==========================================================================///

EdtAnalysis::EdtAnalysis(AnalysisManager &AM) : ArtsAnalysis(AM) {
  assert(AM.getModule() && "Module is required");
}

void EdtAnalysis::analyze() {
  edtPatternByOp.clear();
  allocPatternByOp.clear();
  edtOrderIndex.clear();

  auto &AM = getAnalysisManager();
  for (auto func : AM.getModule().getOps<func::FuncOp>())
    analyzeFunc(func);
  analyzed.store(true, std::memory_order_release);
}

void EdtAnalysis::analyzeFunc(func::FuncOp func) {
  unsigned edtIndex = 0;
  auto &analysisManager = getAnalysisManager();
  EdtGraph &edtGraph = getOrCreateEdtGraph(func);
  func.walk([&](EdtOp edt) {
    auto edtNode = edtGraph.getEdtNode(edt);
    if (!edtNode)
      return;
    auto &info = edtNode->getInfo();
    info.orderIndex = edtIndex;
    edtOrderIndex[edt] = edtIndex++;

    /// Compute max loop depth within the EDT body.
    unsigned maxDepth = 0;
    edt.getBody().walk([&](Operation *op) {
      if (!isa<scf::ForOp, affine::AffineForOp, scf::ParallelOp, arts::ForOp>(
              op))
        return;
      unsigned depth = 0;
      for (Operation *parent = op->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (parent == edt.getOperation())
          break;
        if (isa<scf::ForOp, affine::AffineForOp, scf::ParallelOp, arts::ForOp>(
                parent))
          ++depth;
      }
      maxDepth = std::max(maxDepth, depth + 1);
    });
    info.maxLoopDepth = maxDepth;

    info.loopDistributionPatterns.clear();
    info.dominantDistributionPattern = EdtDistributionPattern::unknown;

    bool sawLoop = false;
    bool mixed = false;
    EdtDistributionPattern summary = EdtDistributionPattern::unknown;

    edt.walk([&](arts::ForOp forOp) {
      if (forOp->getParentOfType<EdtOp>() != edt)
        return;
      DbAnalysis::LoopDbAccessSummary loopAccessSummary;
      if (auto summary =
              analysisManager.getLoopDbAccessSummary(forOp.getOperation()))
        loopAccessSummary = *summary;
      for (auto &[allocOp, pattern] : loopAccessSummary.allocPatterns) {
        auto it = allocPatternByOp.find(allocOp);
        if (it == allocPatternByOp.end()) {
          allocPatternByOp[allocOp] = pattern;
        } else {
          it->second = DbUtils::mergeAccessPattern(it->second, pattern);
        }
      }

      EdtDistributionPattern pattern = loopAccessSummary.distributionPattern;
      info.loopDistributionPatterns[forOp.getOperation()] = pattern;
      if (!sawLoop) {
        summary = pattern;
        sawLoop = true;
      } else if (summary != pattern) {
        mixed = true;
      }
    });

    if (sawLoop && !mixed)
      info.dominantDistributionPattern = summary;
    edtPatternByOp[edt.getOperation()] = info.dominantDistributionPattern;
  });
}

void EdtAnalysis::print(func::FuncOp func, llvm::raw_ostream &os) {
  os << "EdtAnalysis for function: " << func.getName() << "\n";

  EdtGraph &edtGraph = getOrCreateEdtGraph(func);
  func.walk([&](EdtOp edt) {
    auto edtNode = edtGraph.getEdtNode(edt);
    if (!edtNode)
      return;
    const EdtInfo &info = edtNode->getInfo();
    os << "  EDT #" << info.orderIndex << ":\n";
    os << "    Max loop depth: " << info.maxLoopDepth << "\n";
  });
}

void EdtAnalysis::toJson(func::FuncOp func, llvm::raw_ostream &os) {
  os << "{\n";
  os << "  \"function\": \"" << func.getName() << "\",\n";
  os << "  \"edts\": [\n";

  bool first = true;
  EdtGraph &edtGraph = getOrCreateEdtGraph(func);
  func.walk([&](EdtOp edt) {
    auto edtNode = edtGraph.getEdtNode(edt);
    if (!edtNode)
      return;
    const EdtInfo &info = edtNode->getInfo();

    if (!first)
      os << ",\n";
    first = false;

    os << "    {\n";
    os << "      \"edtId\": " << info.orderIndex << ",\n";
    os << "      \"maxLoopDepth\": " << info.maxLoopDepth << "\n";
    os << "    }";
  });

  os << "\n  ]\n";
  os << "}\n";
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

std::optional<EdtDistributionPattern>
EdtAnalysis::getEdtDistributionPattern(EdtOp edt) {
  if (!edt)
    return std::nullopt;
  if (!analyzed.load(std::memory_order_acquire))
    analyze();
  auto it = edtPatternByOp.find(edt.getOperation());
  if (it == edtPatternByOp.end())
    return std::nullopt;
  return it->second;
}

std::optional<DbAccessPattern>
EdtAnalysis::getAllocAccessPattern(Operation *allocOp) {
  if (!allocOp)
    return std::nullopt;
  if (!analyzed.load(std::memory_order_acquire))
    analyze();
  auto it = allocPatternByOp.find(allocOp);
  if (it == allocPatternByOp.end())
    return std::nullopt;
  return it->second;
}

void EdtAnalysis::forEachAllocAccessPattern(
    llvm::function_ref<void(Operation *, DbAccessPattern)> fn) {
  if (!analyzed.load(std::memory_order_acquire))
    analyze();
  for (const auto &entry : allocPatternByOp)
    fn(entry.first, entry.second);
}

bool EdtAnalysis::invalidateGraph(func::FuncOp func) {
  std::unique_lock<std::shared_mutex> writeLock(edtGraphMutex);
  auto it = edtGraphs.find(func);
  if (it != edtGraphs.end()) {
    if (it->second)
      it->second->invalidate();
    edtGraphs.erase(it);
    analyzed.store(false, std::memory_order_release);
    return true;
  }
  return false;
}

bool EdtAnalysis::isParallelEdtFusable(EdtOp edt) {
  if (edt.getType() != arts::EdtType::parallel)
    return false;
  if (!edt.getDependencies().empty())
    return false;
  Block &body = edt.getBody().front();
  for (Operation &op : body) {
    if (isa<arts::ForOp>(&op))
      continue;
    if (isa<arts::YieldOp>(&op))
      continue;
    return false;
  }
  return true;
}

void EdtAnalysis::invalidate() {
  std::unique_lock<std::shared_mutex> writeLock(edtGraphMutex);
  for (auto &kv : edtGraphs)
    if (kv.second)
      kv.second->invalidate();
  edtGraphs.clear();
  edtPatternByOp.clear();
  allocPatternByOp.clear();
  edtOrderIndex.clear();
  analyzed.store(false, std::memory_order_release);
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

///==========================================================================///
/// Reduction pattern analysis.
///==========================================================================///

std::optional<ReductionOp> EdtAnalysis::classifyReductionOp(Operation *op) {
  if (!op)
    return std::nullopt;

  if (isa<arith::AddIOp>(op) || isa<arith::AddFOp>(op))
    return ReductionOp::ADD;

  if (isa<arith::MinSIOp>(op) || isa<arith::MinUIOp>(op) ||
      isa<arith::MinimumFOp>(op) || isa<arith::MinNumFOp>(op))
    return ReductionOp::MIN;

  if (isa<arith::MaxSIOp>(op) || isa<arith::MaxUIOp>(op) ||
      isa<arith::MaximumFOp>(op) || isa<arith::MaxNumFOp>(op))
    return ReductionOp::MAX;

  if (isa<arith::OrIOp>(op))
    return ReductionOp::OR;
  if (isa<arith::AndIOp>(op))
    return ReductionOp::AND;
  if (isa<arith::XOrIOp>(op))
    return ReductionOp::XOR;

  return std::nullopt;
}

StringRef EdtAnalysis::reductionOpName(ReductionOp op) {
  switch (op) {
  case ReductionOp::ADD:
    return "add";
  case ReductionOp::MIN:
    return "min";
  case ReductionOp::MAX:
    return "max";
  case ReductionOp::OR:
    return "or";
  case ReductionOp::AND:
    return "and";
  case ReductionOp::XOR:
    return "xor";
  }
  return "unknown";
}

bool EdtAnalysis::isAtomicCapable(ReductionOp op) {
  return op == ReductionOp::ADD || op == ReductionOp::MIN ||
         op == ReductionOp::MAX;
}

std::optional<ReductionOp> EdtAnalysis::matchLoadModifyStore(Value dbRefResult,
                                                             Region &edtBody) {
  SmallVector<Operation *, 4> loads;
  SmallVector<Operation *, 4> stores;

  DbUtils::forEachReachableMemoryAccess(
      dbRefResult,
      [&](const DbUtils::MemoryAccessInfo &access) {
        if (access.isRead())
          loads.push_back(access.op);
        else if (access.isWrite())
          stores.push_back(access.op);
        return WalkResult::advance();
      },
      &edtBody);

  if (loads.size() != 1 || stores.size() != 1)
    return std::nullopt;

  Operation *loadOp = loads.front();
  Operation *storeOp = stores.front();

  Value storedVal;
  if (auto store = dyn_cast<memref::StoreOp>(storeOp))
    storedVal = store.getValueToStore();
  else if (auto store = dyn_cast<affine::AffineStoreOp>(storeOp))
    storedVal = store.getValueToStore();
  else
    return std::nullopt;

  if (!storedVal || !storedVal.getDefiningOp())
    return std::nullopt;

  Operation *modifyOp = storedVal.getDefiningOp();
  auto reductionKind = classifyReductionOp(modifyOp);
  if (!reductionKind)
    return std::nullopt;

  Value loadResult = loadOp->getResult(0);
  bool usesLoad = false;
  for (Value operand : modifyOp->getOperands()) {
    if (operand == loadResult) {
      usesLoad = true;
      break;
    }
  }

  if (!usesLoad)
    return std::nullopt;

  return reductionKind;
}
