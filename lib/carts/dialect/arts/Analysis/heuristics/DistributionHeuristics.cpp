///==========================================================================///
/// File: DistributionHeuristics.cpp
///
/// H2 distribution policy heuristics: machine-aware strategy, pattern, and
/// worker-topology selection.
///==========================================================================///

#include "carts/dialect/arts/Analysis/heuristics/DistributionHeuristics.h"
#include "carts/utils/OperationAttributes.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::arts;

ParallelismDecision DistributionHeuristics::resolveParallelismFromMachine(
    const RuntimeConfig *machine) {
  ParallelismDecision decision;
  if (!machine)
    return decision;

  int64_t nodeCount = machine->getNodeCount();
  int64_t workerThreads = machine->getRuntimeWorkersPerNode();

  if (nodeCount > 1) {
    decision.concurrency = EdtConcurrency::internode;
    decision.workersPerNode = workerThreads > 0 ? workerThreads : 1;
    decision.totalWorkers = machine->getRuntimeTotalWorkers();
    if (decision.totalWorkers <= 0)
      decision.totalWorkers = nodeCount * decision.workersPerNode;
    return decision;
  }

  decision.concurrency = EdtConcurrency::intranode;
  if (workerThreads > 0) {
    decision.totalWorkers = workerThreads;
    decision.workersPerNode = workerThreads;
    return decision;
  }

  if (nodeCount == 1) {
    decision.totalWorkers = 1;
    decision.workersPerNode = 1;
  }

  return decision;
}

std::optional<WorkerConfig>
DistributionHeuristics::resolveWorkerConfig(EdtOp edt,
                                            const RuntimeConfig *machine) {
  WorkerConfig cfg;
  cfg.internode = edt.getConcurrency() == EdtConcurrency::internode;

  if (auto workers = edt ? arts::getWorkers(edt.getOperation()) : std::nullopt) {
    cfg.totalWorkers = *workers;
    if (cfg.totalWorkers <= 0)
      return std::nullopt;

    if (cfg.internode) {
      if (auto workersPerNode =
              edt ? arts::getWorkersPerNode(edt.getOperation())
                  : std::nullopt) {
        cfg.workersPerNode = *workersPerNode;
      } else if (machine && machine->getNodeCount() > 0) {
        int64_t nc = machine->getNodeCount();
        cfg.workersPerNode = std::max<int64_t>(1, cfg.totalWorkers / nc);
      }
      if (cfg.workersPerNode <= 0)
        cfg.workersPerNode = 1;
    } else {
      cfg.workersPerNode = cfg.totalWorkers;
    }
    return cfg;
  }

  if (auto module = edt->getParentOfType<ModuleOp>()) {
    if (auto runtimeTotalWorkers = getRuntimeTotalWorkers(module);
        runtimeTotalWorkers && *runtimeTotalWorkers > 0) {
      cfg.totalWorkers = *runtimeTotalWorkers;
      if (!cfg.internode) {
        cfg.workersPerNode = cfg.totalWorkers;
      } else if (auto runtimeTotalNodes = getRuntimeTotalNodes(module);
                 runtimeTotalNodes && *runtimeTotalNodes > 0) {
        cfg.workersPerNode =
            std::max<int64_t>(1, cfg.totalWorkers / *runtimeTotalNodes);
      } else {
        cfg.workersPerNode = cfg.totalWorkers;
      }
      return cfg;
    }
  }

  if (!machine)
    return std::nullopt;

  int64_t runtimeWorkersPerNode = machine->getRuntimeWorkersPerNode();
  if (runtimeWorkersPerNode <= 0)
    return std::nullopt;

  if (!cfg.internode) {
    cfg.workersPerNode = runtimeWorkersPerNode;
    cfg.totalWorkers = cfg.workersPerNode;
    return cfg;
  }

  if (machine->getNodeCount() <= 0)
    return std::nullopt;

  cfg.workersPerNode = runtimeWorkersPerNode;
  cfg.totalWorkers = machine->getRuntimeTotalWorkers();
  if (cfg.totalWorkers <= 0)
    return std::nullopt;

  return cfg;
}

DistributionStrategy
DistributionHeuristics::analyzeStrategy(EdtConcurrency concurrency,
                                        const RuntimeConfig *machine) {
  DistributionStrategy strategy;

  if (concurrency == EdtConcurrency::intranode) {
    strategy.kind = DistributionKind::Flat;
    strategy.numNodes = 1;
    if (machine) {
      strategy.workersPerNode = machine->getRuntimeWorkersPerNode();
      strategy.totalWorkers = strategy.workersPerNode;
    }
    strategy.useDbAlignment = false;
    return strategy;
  }

  strategy.kind = DistributionKind::TwoLevel;
  if (machine) {
    strategy.numNodes = machine->getNodeCount();
    strategy.workersPerNode = machine->getRuntimeWorkersPerNode();
    strategy.totalWorkers = machine->getRuntimeTotalWorkers();
  }
  strategy.useDbAlignment = true;
  return strategy;
}

EdtDistributionKind DistributionHeuristics::selectDistributionKind(
    const DistributionStrategy &strategy, EdtDistributionPattern pattern) {
  /// Matmul override: prefer tiling_2d for TwoLevel, block otherwise.
  if (pattern == EdtDistributionPattern::matmul) {
    if (strategy.kind == DistributionKind::TwoLevel)
      return EdtDistributionKind::tiling_2d;
    return EdtDistributionKind::block;
  }

  /// Strategy-based dispatch (general case).
  switch (strategy.kind) {
  case DistributionKind::TwoLevel:
    return EdtDistributionKind::two_level;
  case DistributionKind::BlockCyclic:
    return EdtDistributionKind::block_cyclic;
  case DistributionKind::Tiling2D:
    return EdtDistributionKind::tiling_2d;
  case DistributionKind::Replicated:
    return EdtDistributionKind::replicated;
  case DistributionKind::Flat:
    break;
  }

  /// Fallback to pattern-specific defaults.
  switch (pattern) {
  case EdtDistributionPattern::triangular:
    /// The block-cyclic triangular bounds path can collapse the inner work
    /// range to zero for row-dependent lower bounds. Keep triangular kernels on
    /// block decomposition until that bounds analysis is fully modeled.
    return EdtDistributionKind::block;
  case EdtDistributionPattern::stencil:
  case EdtDistributionPattern::uniform:
  case EdtDistributionPattern::matmul:
  case EdtDistributionPattern::unknown:
    return EdtDistributionKind::block;
  }
  return EdtDistributionKind::block;
}
