///==========================================================================///
/// File: DistributionHeuristics.h
///
/// Machine-aware work-distribution policy heuristics for the CARTS compiler.
///
/// This module owns H2 policy selection: machine topology inspection,
/// distribution-strategy selection, worker-topology defaults, and loop
/// coarsening hints.
///
/// Lowering-time SSA/runtime materialization for those policies is performed
/// across the SDE-to-CODIR and CODIR-to-ARTS materialization boundary.
///
///===----------------------------------------------------------------------===///
/// Strategies
///===----------------------------------------------------------------------===///
///
/// Flat (single-level):
///   Used for intranode (single-node) execution.
///   All W workers divide N iterations equally using balanced distribution:
///     base = N / W,  rem = N % W
///     worker i gets: start = i*base + min(i, rem), count = base + (i < rem)
///
///   DB acquire and iteration bounds are identical per worker.
///
///   Example: N=100 iterations, W=4 workers
///     Worker 0: iter=[0, 25)    acquire=[0, 25)
///     Worker 1: iter=[25, 50)   acquire=[25, 50)
///     Worker 2: iter=[50, 75)   acquire=[50, 75)
///     Worker 3: iter=[75, 100)  acquire=[75, 100)
///
/// TwoLevel (inter-node + intra-node):
///   Used for internode (multi-node) execution.
///   Distributes work in two hierarchical levels:
///
///   Level 1 - Inter-node (DB block assignment):
///     Assigns iteration chunks to NODES. Each node gets one or more
///     contiguous DB blocks. All worker threads on the same node share the
///     same DB acquire range (reducing remote data transfers).
///
///     blockSize = ceil(arrayDim / numNodes)     [matches DbPartitioning]
///     totalChunks = ceil(totalIterations / blockSize)
///     (nodeStart, nodeChunks) = balancedDistribute(totalChunks, numNodes,
///                                                  nodeId)
///     nodeRows = nodeChunks * blockSize  (clamped to remaining iterations)
///
///   Level 2 - Intra-node (thread sub-division):
///     Within each node's block, further divides rows across local worker
///     threads. This ensures ALL threads on ALL nodes have work.
///
///     subBlock = ceil(nodeRows / workersPerNode)
///     localStart = localThreadId * subBlock
///     localCount = min(subBlock, nodeRows - localStart)
///
///   DB acquire uses Level 1 (node-level) bounds.
///   Inner loop uses Level 2 (thread-level) bounds.
///
///   Example: N=255 iterations, 6 nodes, 2 workers/node = 12 total workers
///            blockSize = ceil(256/6) = 43, totalChunks = 6
///
///     Node 0 (workers 0,1):  acquire=[0, 43)
///       Thread 0: iter=[0, 22)     Thread 1: iter=[22, 43)
///     Node 1 (workers 2,3):  acquire=[43, 86)
///       Thread 0: iter=[43, 65)    Thread 1: iter=[65, 86)
///     Node 2 (workers 4,5):  acquire=[86, 129)
///       Thread 0: iter=[86, 108)   Thread 1: iter=[108, 129)
///     Node 3 (workers 6,7):  acquire=[129, 172)
///       Thread 0: iter=[129, 151)  Thread 1: iter=[151, 172)
///     Node 4 (workers 8,9):  acquire=[172, 215)
///       Thread 0: iter=[172, 194)  Thread 1: iter=[194, 215)
///     Node 5 (workers 10,11): acquire=[215, 255)
///       Thread 0: iter=[215, 235)  Thread 1: iter=[235, 255)
///
///     All 6 nodes active, all 12 workers busy.
///
///   Why TwoLevel applies to ALL internode arrays (not just stencil):
///     - Stencil arrays: Fixes idle workers. Halo logic requires workers on
///       the same node to map to the same DB block.
///     - Block arrays: All workers on a node acquire the SAME block ->
///       reduces remote acquire count, improves cache sharing.
///     - Unified code path: simpler than separate stencil/block paths.
///
///===----------------------------------------------------------------------===///
/// Key invariants
///===----------------------------------------------------------------------===///
///
  ///   1. Worker ID mapping:
  ///        nodeId        = workerId / workersPerNode
  ///        localThreadId = workerId % workersPerNode
  ///      This must match Core EDT routing during materialization/lowering.
///
///   2. DB alignment:
///        blockSize = ceil(arrayDim / numNodes)
  ///      Must match DbPartitioning's block size so task chunk boundaries align
  ///      with DB block boundaries.
///
///   3. Stencil safety:
///        All workers on the same node have baseOffset within the same
///        DB block, so blockIdx = baseOffset / planBlockSize is identical
///        for all threads on a node.
///
///   4. Balanced distribution:
///        Uses floor+remainder pattern to avoid idle tail workers:
///          base = total / n,  rem = total % n
///          participant i: start = i*base + min(i, rem)
///                         count = base + (i < rem ? 1 : 0)
///
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_DISTRIBUTIONHEURISTICS_H
#define ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_DISTRIBUTIONHEURISTICS_H

#include "carts/Dialect.h"
#include "carts/dialect/arts/Utils/RuntimeConfig.h"
#include <optional>

namespace mlir {
namespace arts {

class AnalysisManager;
class LoopAnalysis;

/// H2: Distribution strategy kinds
enum class DistributionKind {
  Flat,        ///  Single-level: all workers divide iterations equally
  TwoLevel,    ///  Two-level: nodes get DB blocks, threads subdivide within
  BlockCyclic, ///  Cyclic chunks: chunk k -> worker (k % totalWorkers)
  Tiling2D,    ///  Matmul-oriented 2D worker grid (row ownership + column
               ///  striping)
  Replicated ///  Full replication: each node holds a complete copy (read-only)
};

/// Machine topology analysis result (compile-time, no IR)
struct DistributionStrategy {
  DistributionKind kind = DistributionKind::Flat;
  int64_t numNodes = 1;        ///  Total nodes
  int64_t workersPerNode = 1;  ///  Worker threads per node
  int64_t totalWorkers = 1;    ///  numNodes * workersPerNode
  bool useDbAlignment = false; ///  blockSize > 1 for DB boundary alignment
};

/// Resolved worker topology for an EDT.
struct WorkerConfig {
  int64_t totalWorkers = 0;
  int64_t workersPerNode = 0;
  bool internode = false;
};

/// Machine-derived EDT topology defaults.
struct ParallelismDecision {
  EdtConcurrency concurrency = EdtConcurrency::intranode;
  int64_t totalWorkers = 0;
  int64_t workersPerNode = 0;
};

class DistributionHeuristics {
public:
  /// H2.1: Analyze machine topology -> distribution strategy
  /// Pure analysis, no IR emission.
  static DistributionStrategy
  analyzeStrategy(EdtConcurrency concurrency,
                  const RuntimeConfig *machine = nullptr);

  /// Select IR distribution kind from machine strategy + detected loop pattern.
  static EdtDistributionKind
  selectDistributionKind(const DistributionStrategy &strategy,
                         EdtDistributionPattern pattern);

  /// Resolve default EDT parallelism from machine topology.
  /// Used by passes to avoid duplicating node/worker resolution logic.
  static ParallelismDecision
  resolveParallelismFromMachine(const RuntimeConfig *machine);

  /// Resolve compile-time worker topology for an EDT from attrs + machine.
  /// Returns nullopt when worker count is not computable.
  static std::optional<WorkerConfig>
  resolveWorkerConfig(EdtOp edt, const RuntimeConfig *machine = nullptr);

};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_HEURISTICS_DISTRIBUTIONHEURISTICS_H
