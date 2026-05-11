///==========================================================================///
/// File: DistributionHeuristics.cpp
///
/// H2 distribution policy heuristics: machine-aware strategy, pattern, and
/// worker-topology selection.
///==========================================================================///

#include "arts/dialect/core/Analysis/heuristics/DistributionHeuristics.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/heuristics/HeuristicUtils.h"
#include "arts/dialect/core/Analysis/loop/LoopAnalysis.h"
#include "arts/utils/LoopUtils.h"
#include "arts/utils/LoweringContractUtils.h"
#include "arts/utils/OperationAttributes.h"
#include <algorithm>
#include <cmath>
#include <limits>

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
DistributionHeuristics::resolveWorkerConfig(EdtOp parallelEdt,
                                            const RuntimeConfig *machine) {
  WorkerConfig cfg;
  cfg.internode = parallelEdt.getConcurrency() == EdtConcurrency::internode;

  if (auto workers = parallelEdt ? arts::getWorkers(parallelEdt.getOperation())
                                 : std::nullopt) {
    cfg.totalWorkers = *workers;
    if (cfg.totalWorkers <= 0)
      return std::nullopt;

    if (cfg.internode) {
      if (auto workersPerNode =
              parallelEdt ? arts::getWorkersPerNode(parallelEdt.getOperation())
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

  if (auto module = parallelEdt->getParentOfType<ModuleOp>()) {
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

LoopCoarseningDecision DistributionHeuristics::computeLoopCoarseningDecision(
    ForOp forOp, LoopAnalysis &loopAnalysis, const WorkerConfig &workerCfg) {
  LoopCoarseningDecision decision;
  decision.desiredWorkers = std::max<int64_t>(1, workerCfg.totalWorkers);

  if (workerCfg.totalWorkers <= 0)
    return decision;

  auto tripOpt = loopAnalysis.getStaticTripCount(forOp.getOperation());
  if (!tripOpt)
    return decision;

  int64_t tripCount = *tripOpt;
  if (tripCount <= 0)
    return decision;

  int64_t blockSize = ceilDivPositive(tripCount, workerCfg.totalWorkers);
  if (blockSize > 1)
    decision.blockSize = blockSize;
  return decision;
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

DistributionStrategy
DistributionHeuristics::resolveLoweringStrategy(EdtOp originalParallel,
                                                ForOp forOp) {
  DistributionStrategy strategy =
      analyzeStrategy(originalParallel.getConcurrency(), /*machine=*/nullptr);
  if (auto selectedKind = getEdtDistributionKind(forOp.getOperation())) {
    switch (*selectedKind) {
    case EdtDistributionKind::block:
      strategy.kind = DistributionKind::Flat;
      break;
    case EdtDistributionKind::two_level:
      strategy.kind = DistributionKind::TwoLevel;
      break;
    case EdtDistributionKind::block_cyclic:
      strategy.kind = DistributionKind::BlockCyclic;
      break;
    case EdtDistributionKind::tiling_2d:
      strategy.kind = DistributionKind::Tiling2D;
      break;
    case EdtDistributionKind::replicated:
      strategy.kind = DistributionKind::Replicated;
      break;
    }
  }
  return strategy;
}

std::optional<EdtDistributionPattern>
DistributionHeuristics::resolveDistributionPattern(AnalysisManager *AM,
                                                   ForOp forOp,
                                                   EdtOp originalParallel) {
  if (auto pattern = getEdtDistributionPattern(forOp.getOperation()))
    return pattern;
  if (AM) {
    if (auto pattern = AM->getLoopDistributionPattern(forOp.getOperation());
        pattern && *pattern != EdtDistributionPattern::unknown) {
      return pattern;
    }
  }
  return getEdtDistributionPattern(originalParallel.getOperation());
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
