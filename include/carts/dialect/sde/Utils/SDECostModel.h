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
};

} // namespace mlir::carts::sde

#endif // CARTS_DIALECT_SDE_UTILS_SDECOSTMODEL_H
