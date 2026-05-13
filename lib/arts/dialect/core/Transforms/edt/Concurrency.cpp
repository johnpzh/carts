///==========================================================================///
/// File: Concurrency.cpp
/// Defines concurrency-driven optimizations using ARTS analysis managers.
/// The Concurrency pass make high-level decisions about task (EDT) placement
/// and data (DB) affinity - it advices on where to execute EDTs or
/// nodes—essentially, deciding locality (e.g., which worker thread, node, or
/// memory tier) based on access patterns, dependencies, and the abstract
/// machine model.
///
/// Example:
///   Before:
///     arts.edt <parallel> { arts.for ... }
///
///   After:
///     arts.edt <parallel> { arts.for ... }
///     // with updated workers/concurrency/route attrs from machine + analysis
///==========================================================================///

#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/graphs/edt/EdtNode.h"
#define GEN_PASS_DEF_CONCURRENCY
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/DbUtils.h"
#include "arts/utils/LoopUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/machine/RuntimeConfig.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include "llvm/Support/MathExtras.h"

#include "arts/utils/Debug.h"
#include "llvm/ADT/Statistic.h"
ARTS_DEBUG_SETUP(concurrency)

using namespace mlir::arts;
using namespace mlir;

namespace {

static std::optional<int64_t> getPlannedOwnerBlockSize(ForOp forOp) {
  if (!forOp)
    return std::nullopt;

  auto pattern = getEffectiveDepPattern(forOp.getOperation());
  if (!pattern || !isStencilHaloDepPattern(*pattern))
    return std::nullopt;

  auto ownerDims = readI64ArrayAttr(getPlanOwnerDimsAttr(forOp.getOperation()));
  auto blockShape =
      readI64ArrayAttr(getPlanPhysicalBlockShapeAttr(forOp.getOperation()));
  if (!blockShape || blockShape->empty())
    blockShape =
        readI64ArrayAttr(getPlanLogicalWorkerSliceAttr(forOp.getOperation()));
  if (!ownerDims || ownerDims->empty() || !blockShape || blockShape->empty())
    return std::nullopt;

  int64_t ownerDim = ownerDims->front();
  size_t blockIdx = 0;
  if (ownerDim >= 0 && static_cast<size_t>(ownerDim) < blockShape->size())
    blockIdx = static_cast<size_t>(ownerDim);
  int64_t blockSize = (*blockShape)[blockIdx];
  if (blockSize <= 1)
    return std::nullopt;
  return blockSize;
}

static std::optional<int64_t>
getPlannedStencilWorkerCount(EdtOp edtOp, int64_t hardwareWorkers) {
  if (!edtOp || hardwareWorkers <= 0)
    return std::nullopt;

  std::optional<int64_t> best;
  Block &body = edtOp.getBody().front();
  for (Operation &op : body.without_terminator()) {
    auto forOp = dyn_cast<ForOp>(&op);
    if (!forOp)
      continue;

    auto blockSize = getPlannedOwnerBlockSize(forOp);
    auto tripCount = getStaticTripCount(forOp.getOperation());
    if (!blockSize || !tripCount || *tripCount <= 0)
      continue;

    int64_t logicalWorkers =
        llvm::divideCeil(*tripCount, std::max<int64_t>(1, *blockSize));
    if (logicalWorkers <= hardwareWorkers)
      continue;
    if (logicalWorkers > hardwareWorkers * 4)
      logicalWorkers = hardwareWorkers * 4;
    best = best ? std::max(*best, logicalWorkers) : logicalWorkers;
  }
  return best;
}

} // namespace

static llvm::Statistic numEdtsAssignedInternode{
    "concurrency", "NumEdtsAssignedInternode",
    "Number of EDTs assigned internode concurrency"};
static llvm::Statistic numEdtsAssignedIntranode{
    "concurrency", "NumEdtsAssignedIntranode",
    "Number of EDTs assigned intranode concurrency"};
static llvm::Statistic numEdtsKeptIntranodeForTasks{
    "concurrency", "NumEdtsKeptIntranodeForTasks",
    "Number of EDTs kept intranode due to nested task EDTs"};
static llvm::Statistic numRuntimeQueriesRemapped{
    "concurrency", "NumRuntimeQueriesRemapped",
    "Number of runtime query ops remapped for concurrency model"};
static llvm::Statistic numDbModesAdjusted{
    "concurrency", "NumDbModesAdjusted",
    "Number of DB modes adjusted for concurrency"};

using namespace mlir;

///==========================================================================///
/// ConcurrencyPass
///==========================================================================///
struct ConcurrencyPass : public impl::ConcurrencyBase<ConcurrencyPass> {
  explicit ConcurrencyPass(mlir::arts::AnalysisManager *AM) : AM(AM) {
    assert(AM && "AnalysisManager must be provided externally");
  }
  void runOnOperation() override;

private:
  void applyEdtParallelismStrategy(EdtOp edtOp);
  void adjustDbModes();
  void updateRuntimeQueryOperations();

  RuntimeConfig *runtimeConfig = nullptr;
  mlir::arts::AnalysisManager *AM = nullptr;
  ModuleOp module;
};

void ConcurrencyPass::runOnOperation() {
  module = getOperation();

  /// Get runtimeConfig from AnalysisManager
  runtimeConfig = &AM->getRuntimeConfig();
  /// Rebuild DB graph from current IR so concurrency decisions consume
  /// up-to-date DB read/write users.
  AM->getDbAnalysis().invalidate();

  ARTS_INFO_HEADER(Concurrency);
  ARTS_DEBUG_REGION(module.dump(););
  ARTS_INFO("Machine valid=" << (runtimeConfig->isValid() ? 1 : 0)
                             << ", nodes=" << runtimeConfig->getNodeCount()
                             << ", threads=" << runtimeConfig->getThreads());

  /// Apply EDT parallelism strategy to all EDT operations
  module.walk([&](EdtOp edtOp) { applyEdtParallelismStrategy(edtOp); });

  /// Adjust datablock modes based on EDT concurrency
  adjustDbModes();

  /// Update runtime query operations based on EDT concurrency
  updateRuntimeQueryOperations();

  ARTS_INFO_FOOTER(Concurrency);
  ARTS_DEBUG_REGION(module.dump(););
}

void ConcurrencyPass::updateRuntimeQueryOperations() {
  ARTS_DEBUG("Updating runtime query operations");

  /// Map of query kind swaps for concurrency remapping:
  /// internode EDTs: currentWorker -> currentNode, totalWorkers -> totalNodes
  /// intranode EDTs: currentNode -> currentWorker, totalNodes -> totalWorkers
  SmallVector<std::pair<RuntimeQueryOp, RuntimeQueryKind>> toReplace;

  /// Phase 1: Collect operations to replace
  module.walk([&](EdtOp edtOp) {
    bool isInterNode = (edtOp.getConcurrency() == EdtConcurrency::internode);

    edtOp.walk([&](RuntimeQueryOp queryOp) {
      auto kind = queryOp.getKind();
      if (isInterNode) {
        if (kind == RuntimeQueryKind::currentWorker)
          toReplace.emplace_back(queryOp, RuntimeQueryKind::currentNode);
        else if (kind == RuntimeQueryKind::totalWorkers)
          toReplace.emplace_back(queryOp, RuntimeQueryKind::totalNodes);
      } else {
        if (kind == RuntimeQueryKind::currentNode)
          toReplace.emplace_back(queryOp, RuntimeQueryKind::currentWorker);
        else if (kind == RuntimeQueryKind::totalNodes)
          toReplace.emplace_back(queryOp, RuntimeQueryKind::totalWorkers);
      }
    });
  });

  /// Phase 2: Replace collected operations (safe - no iteration happening)
  numRuntimeQueriesRemapped += toReplace.size();
  for (auto &[op, newKind] : toReplace) {
    ARTS_INFO("Replacing runtime_query "
              << stringifyRuntimeQueryKind(op.getKind()) << " with "
              << stringifyRuntimeQueryKind(newKind));
    OpBuilder builder(op);
    auto newOp = RuntimeQueryOp::create(builder, op.getLoc(), newKind);
    op.replaceAllUsesWith(newOp.getResult());
    op.erase();
  }
}

void ConcurrencyPass::adjustDbModes() {
  /// For each EDT that has parallelism strategy set, adjust datablock modes
  module.walk([&](EdtOp edtOp) {
    /// No parallelism strategy set
    if (!edtOp.getWorkers().has_value())
      return;

    /// Set Datablock mode based on access mode
    std::function<void(Operation *)> adjustDbMode = [&](Operation *op) {
      if (auto dbAlloc = dyn_cast<DbAllocOp>(op)) {
        ArtsMode accessMode = dbAlloc.getMode();
        DbMode newDbMode = DbUtils::convertArtsModeToDbMode(accessMode);

        /// Update the dbMode attribute
        OpBuilder builder(dbAlloc.getContext());
        dbAlloc.setDbModeAttr(DbModeAttr::get(builder.getContext(), newDbMode));
        ++numDbModesAdjusted;
        ARTS_INFO("Adjusted datablock mode to "
                  << (newDbMode == DbMode::read ? "read" : "write"));
      }

      /// Recursively check operands
      for (Value operand : op->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp())
          adjustDbMode(defOp);
      }
    };

    /// Start traversal from EDT dependencies
    for (Value dep : edtOp.getDependencies()) {
      if (Operation *defOp = dep.getDefiningOp())
        adjustDbMode(defOp);
    }
  });
}

void ConcurrencyPass::applyEdtParallelismStrategy(EdtOp edtOp) {
  /// Only apply to parallel EDTs that don't already have workers attribute set
  if (edtOp.getType() != EdtType::parallel)
    return;

  /// Already has parallelism strategy set
  if (edtOp.getWorkers().has_value())
    return;

  /// If no config file is present, avoid forcing a compile-time worker count.
  /// Let the runtime query decide (arts.runtime_query<total_workers>).
  if (!runtimeConfig->hasConfigFile()) {
    ARTS_WARN("No arts.cfg; using runtime worker count for parallel EDT");
    edtOp.setConcurrency(EdtConcurrency::intranode);
    return;
  }

  /// Check for nested task EDTs - these must remain intranode.
  /// Task parallelism is designed for local execution within a single node.
  /// Distributing a parallel EDT containing tasks across nodes would cause
  /// incorrect work distribution (each node would only process its portion
  /// while tasks expect to see the full iteration space).
  auto func = edtOp->getParentOfType<func::FuncOp>();
  auto *edtNode = func ? AM->getEdtAnalysis().getEdtNode(edtOp) : nullptr;
  bool hasNestedTasks = edtNode ? edtNode->hasNestedTaskEdts() : false;
  auto &heuristics = AM->getEdtHeuristics();
  ParallelismDecision machineDecision = heuristics.resolveParallelism();
  if (hasNestedTasks) {
    int64_t workers = std::max<int64_t>(1, machineDecision.workersPerNode);
    if (auto plannedWorkers =
            getPlannedStencilWorkerCount(edtOp, workers))
      workers = std::max(workers, *plannedWorkers);
    ARTS_INFO("Parallel EDT contains nested tasks - keeping intranode with "
              << workers << " workers");
    ++numEdtsKeptIntranodeForTasks;
    edtOp.setConcurrency(EdtConcurrency::intranode);
    arts::setWorkers(edtOp.getOperation(), workers);
    arts::setWorkersPerNode(edtOp.getOperation(), 0);
    return;
  }

  if (AM->getDbAnalysis().hasNonInternodeConsumerForWrittenDb(edtOp)) {
    int64_t workers = std::max<int64_t>(1, machineDecision.workersPerNode);
    if (auto plannedWorkers =
            getPlannedStencilWorkerCount(edtOp, workers))
      workers = std::max(workers, *plannedWorkers);
    ARTS_INFO("Parallel EDT writes DB consumed outside internode EDT flow - "
              "keeping intranode with "
              << workers << " workers");
    edtOp.setConcurrency(EdtConcurrency::intranode);
    arts::setWorkers(edtOp.getOperation(), workers);
    arts::setWorkersPerNode(edtOp.getOperation(), 0);
    return;
  }

  /// Determine parallelism strategy based on abstract machine
  int numWorkers = 0;
  EdtConcurrency concurrencyType = EdtConcurrency::intranode;

  if (!runtimeConfig->isValid()) {
    ARTS_WARN(
        "Abstract machine validation failed; using available configuration "
        "data when setting parallelism");
  }

  int nodeCount =
      runtimeConfig->hasValidNodeCount() ? runtimeConfig->getNodeCount() : 0;
  int workerThreads = static_cast<int>(machineDecision.workersPerNode);
  numWorkers = static_cast<int>(machineDecision.totalWorkers);
  if (auto plannedWorkers = getPlannedStencilWorkerCount(
          edtOp, std::max<int64_t>(1, machineDecision.workersPerNode))) {
    numWorkers =
        static_cast<int>(std::max<int64_t>(numWorkers, *plannedWorkers));
  }
  concurrencyType = machineDecision.concurrency;

  if (concurrencyType == EdtConcurrency::internode) {
    ARTS_INFO("Setting EDT parallelism: inter-node across "
              << nodeCount << " nodes x " << workerThreads << " workers");
  } else if (numWorkers > 1) {
    ARTS_INFO("Setting EDT parallelism: intra-node across " << numWorkers
                                                            << " workers");
  } else if (numWorkers == 1 && nodeCount == 1) {
    ARTS_INFO("Setting EDT parallelism: single-node execution with 1 worker");
  } else {
    /// Unknown configuration - default to intranode with runtime-determined
    /// workers The EDT will query worker count at runtime via RuntimeQueryOp
    numWorkers = 0;
    concurrencyType = EdtConcurrency::intranode;
    ARTS_INFO("Unknown configuration; defaulting to intranode with runtime "
              "worker count");
  }

  /// Set the attributes on the EDT operation
  if (concurrencyType == EdtConcurrency::internode)
    ++numEdtsAssignedInternode;
  else
    ++numEdtsAssignedIntranode;
  edtOp.setConcurrency(concurrencyType);
  arts::setWorkers(edtOp.getOperation(), numWorkers);

  if (concurrencyType == EdtConcurrency::internode && workerThreads > 0) {
    arts::setWorkersPerNode(edtOp.getOperation(),
                            static_cast<int64_t>(workerThreads));
  } else {
    arts::setWorkersPerNode(edtOp.getOperation(), 0);
  }
  ARTS_INFO("Set EDT concurrency="
            << (concurrencyType == EdtConcurrency::internode ? "internode"
                                                             : "intranode")
            << ", workers=" << numWorkers);
}

namespace mlir {
namespace arts {

std::unique_ptr<Pass> createConcurrencyPass(mlir::arts::AnalysisManager *AM) {
  return std::make_unique<ConcurrencyPass>(AM);
}

} // namespace arts
} // namespace mlir
