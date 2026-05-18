///==========================================================================///
/// File: SDECostModel.h
///
/// Runtime-agnostic planning model for SDE optimization decisions.
/// The dialect boundary provides a concrete implementation. SDE passes see
/// ONLY this interface, never target-runtime types.
///
/// All methods use SDE-level concepts: tasks, barriers, data movement.
/// Target object terminology belongs at the dialect boundary.
///==========================================================================///

#ifndef CARTS_DIALECT_SDE_UTILS_SDECOSTMODEL_H
#define CARTS_DIALECT_SDE_UTILS_SDECOSTMODEL_H

#include "carts/dialect/sde/IR/SdeDialect.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mlir::carts::sde {

class SDECostModel {
public:
  virtual ~SDECostModel() = default;

  // --- Task lifecycle costs (normalized cycles) ---
  virtual double getTaskCreationCost() const = 0;
  virtual double getTaskSyncCost() const = 0;

  // --- Reduction costs ---
  virtual double getReductionCost(int64_t workerCount) const = 0;
  virtual double getAtomicUpdateCost() const = 0;

  // --- Generic memory-access cost ---
  virtual double getDataAccessCost() const = 0;

  // --- Scheduling costs ---
  virtual double getSchedulingOverhead(SdeScheduleKind kind,
                                       int64_t tripCount) const = 0;

  // --- Abstract execution capacity ---
  virtual int getLogicalWorkerCapacity() const = 0;

  // --- Abstract execution topology ---
  // SDE may use these runtime-neutral locality groups to choose enough source
  // tasks for both intra-group and inter-group parallelism. The model is
  // intentionally abstract: target dialects map locality groups to concrete
  // concepts such as nodes, sockets, or accelerator islands, while SDE only
  // reasons about source-level work availability.
  virtual int getWorkerLocalityGroupCount() const { return 1; }
  virtual int getWorkersPerLocalityGroup() const {
    return std::max(1, getLogicalWorkerCapacity());
  }

  // --- Hardware parameters ---
  virtual int getVectorWidth() const = 0;
  virtual int64_t getL2CacheSize() const = 0;

  // --- Vector planning policy ---
  virtual int getVectorWidthForElementBits(unsigned elementBits) const {
    int baseWidth = std::max(1, getVectorWidth());
    if (elementBits == 0)
      return baseWidth;

    constexpr unsigned referenceElementBits =
        std::numeric_limits<uint64_t>::digits;
    unsigned scale = std::max<unsigned>(1, referenceElementBits / elementBits);
    return baseWidth * static_cast<int>(scale);
  }

  virtual int getVectorUnrollFactor(bool reuseHeavy) const {
    int baseWidth = std::max(1, getVectorWidth());
    return reuseHeavy ? baseWidth * baseWidth : baseWidth;
  }

  virtual int getVectorInterleaveCount() const {
    int baseWidth = std::max(1, getVectorWidth());
    return baseWidth * baseWidth;
  }

  // --- Derived thresholds (computed, not hardcoded) ---
  virtual int64_t getMinIterationsPerWorker() const {
    return std::max<int64_t>(
        1, static_cast<int64_t>(getTaskCreationCost() /
                                (getDataAccessCost() + 1.0)));
  }

  // Owner-local pipeline codelets execute multiple local stages before exposing
  // completion. Keep their owner slices large enough to amortize task lifecycle
  // and synchronization, while scaling the floor sublinearly with logical
  // capacity so large runs do not create one tiny pipeline task per worker.
  virtual int64_t getMinPipelineOwnerIterationsPerTask() const {
    int64_t lifecycleIterations = std::max<int64_t>(
        1, static_cast<int64_t>(
               std::ceil((getTaskCreationCost() + getTaskSyncCost()) /
                         (getDataAccessCost() + 1.0))));
    int64_t capacityIterations = std::max<int64_t>(
        1, static_cast<int64_t>(std::ceil(std::log2(
               static_cast<double>(std::max(2, getLogicalWorkerCapacity()))))));
    return std::max({getMinIterationsPerWorker(), lifecycleIterations,
                     capacityIterations});
  }

  virtual int64_t getInterLocalityTaskWaves() const {
    int64_t localityGroups =
        std::max<int64_t>(1, getWorkerLocalityGroupCount());
    if (localityGroups <= 1)
      return 1;
    // Cross-locality launches need at least one spare wave so every locality
    // can receive work while earlier tasks are paying runtime/communication
    // startup costs. Scale sublinearly to avoid exploding tiny kernels.
    return std::max<int64_t>(
        2, 1 + static_cast<int64_t>(std::ceil(std::log2(static_cast<double>(
                   std::max<int64_t>(2, localityGroups))))));
  }

  virtual int64_t getOwnerLocalPipelineTargetTaskWaves() const {
    int64_t baseIterations = std::max<int64_t>(1, getMinIterationsPerWorker());
    int64_t pipelineIterations =
        std::max<int64_t>(baseIterations,
                          getMinPipelineOwnerIterationsPerTask());
    int64_t amortizationWaves = std::max<int64_t>(
        1, (pipelineIterations + baseIterations - 1) / baseIterations);
    return std::max(amortizationWaves, getInterLocalityTaskWaves());
  }
};

} // namespace mlir::carts::sde

#endif // CARTS_DIALECT_SDE_UTILS_SDECOSTMODEL_H
