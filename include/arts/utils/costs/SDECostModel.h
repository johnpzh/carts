///==========================================================================///
/// File: SDECostModel.h
///
/// Runtime-agnostic cost model for SDE optimization decisions.
/// Every target runtime (ARTS, Legion, StarPU, GPU) provides a concrete
/// implementation. SDE passes see ONLY this interface — never ARTS types.
///
/// All methods use SDE-level concepts: tasks, barriers, data movement.
/// No EDT, DB, epoch, CDAG, or GUID terminology.
///==========================================================================///

#ifndef ARTS_UTILS_COSTS_SDECOSTMODEL_H
#define ARTS_UTILS_COSTS_SDECOSTMODEL_H

#include "arts/dialect/sde/IR/SdeDialect.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mlir::arts::sde {

class SDECostModel {
public:
  virtual ~SDECostModel() = default;

  // --- Task lifecycle costs (normalized cycles) ---
  virtual double getTaskCreationCost() const = 0;
  virtual double getTaskSyncCost() const = 0;

  // --- Reduction costs ---
  virtual double getReductionCost(int64_t workerCount) const = 0;
  virtual double getAtomicUpdateCost() const = 0;

  // --- Data movement costs ---
  virtual double getLocalDataAccessCost() const = 0;
  virtual double getRemoteDataAccessCost() const = 0;
  virtual double getHaloExchangeCostPerByte() const = 0;

  // --- Scheduling costs ---
  virtual double getSchedulingOverhead(SdeScheduleKind kind,
                                       int64_t tripCount) const = 0;

  // --- Machine topology (abstract) ---
  virtual int getWorkerCount() const = 0;
  virtual int getNodeCount() const = 0;
  virtual bool isDistributed() const { return getNodeCount() > 1; }

  // --- Hardware parameters ---
  virtual int getVectorWidth() const = 0;
  virtual int64_t getL2CacheSize() const = 0;

  // --- Derived thresholds (computed, not hardcoded) ---
  virtual int64_t getMinIterationsPerWorker() const {
    return std::max<int64_t>(
        1, static_cast<int64_t>(getTaskCreationCost() /
                                (getLocalDataAccessCost() + 1.0)));
  }
};

} // namespace mlir::arts::sde

#endif // ARTS_UTILS_COSTS_SDECOSTMODEL_H
