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

#ifndef ARTS_UTILS_COSTS_SDECOSTMODEL_H
#define ARTS_UTILS_COSTS_SDECOSTMODEL_H

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

  // --- Data movement costs ---
  virtual double getLocalDataAccessCost() const = 0;
  virtual double getRemoteDataAccessCost() const = 0;
  virtual double getHaloExchangeCostPerByte() const = 0;

  // --- Scheduling costs ---
  virtual double getSchedulingOverhead(SdeScheduleKind kind,
                                       int64_t tripCount) const = 0;

  // --- Abstract execution capacity ---
  virtual int getLogicalWorkerCapacity() const = 0;
  virtual int getLogicalNodeCapacity() const = 0;
  virtual bool isDistributed() const { return getLogicalNodeCapacity() > 1; }

  // Compatibility shims for non-SDE callers that have not adopted the
  // runtime-neutral names yet.
  int getWorkerCount() const { return getLogicalWorkerCapacity(); }
  int getNodeCount() const { return getLogicalNodeCapacity(); }

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
                                (getLocalDataAccessCost() + 1.0)));
  }
};

} // namespace mlir::carts::sde

#endif // ARTS_UTILS_COSTS_SDECOSTMODEL_H
