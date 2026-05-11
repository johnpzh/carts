///==========================================================================///
/// File: ARTSCostModel.h
///
/// ARTS-specific implementation of SDECostModel. Costs change based on
/// RuntimeConfig topology (local vs distributed):
///   - Halo exchange: 0.01/byte local vs 0.5/byte distributed (50x)
///   - Task sync: 3000 local vs 5000 distributed (1.7x)
///   - Task creation: 1800 local vs 2500 distributed (1.4x)
///==========================================================================///

#ifndef ARTS_UTILS_COSTS_ARTSCOSTMODEL_H
#define ARTS_UTILS_COSTS_ARTSCOSTMODEL_H

#include "arts/utils/costs/SDECostModel.h"
#include "arts/utils/machine/RuntimeConfig.h"

namespace mlir::arts {

class ARTSCostModel : public sde::SDECostModel {
  const RuntimeConfig &machine;

public:
  explicit ARTSCostModel(const RuntimeConfig &am) : machine(am) {}

  // --- Task lifecycle (maps SDE tasks -> ARTS EDTs) ---
  double getTaskCreationCost() const override {
    return machine.isDistributed() ? 2500.0 : 1800.0;
  }
  double getTaskSyncCost() const override {
    return machine.isDistributed() ? 5000.0 : 3000.0;
  }

  // --- Reduction (maps SDE reductions -> ARTS atomics/trees) ---
  double getReductionCost(int64_t workerCount) const override {
    double treeCost = std::log2(workerCount) * getTaskSyncCost();
    double linearCost = workerCount * getAtomicUpdateCost();
    return std::min(treeCost, linearCost);
  }
  double getAtomicUpdateCost() const override {
    return machine.isDistributed() ? 500.0 : 100.0;
  }

  // --- Data movement (maps SDE data access -> ARTS DB acquire) ---
  double getLocalDataAccessCost() const override { return 500.0; }
  double getRemoteDataAccessCost() const override { return 5000.0; }
  double getHaloExchangeCostPerByte() const override {
    return machine.isDistributed() ? 0.5 : 0.01;
  }
  double getAllocationCost() const override {
    return machine.isDistributed() ? 2000.0 : 1500.0;
  }

  // --- Scheduling ---
  double getSchedulingOverhead(sde::SdeScheduleKind kind,
                               int64_t tripCount) const override {
    switch (kind) {
    case sde::SdeScheduleKind::static_:
      return 0.0;
    case sde::SdeScheduleKind::dynamic:
      return getTaskCreationCost() * 0.1;
    case sde::SdeScheduleKind::guided:
      return getTaskCreationCost() * 0.05;
    default:
      return 0.0;
    }
  }

  // --- Hardware parameters ---
  int getVectorWidth() const override { return 4; } // 256-bit SIMD / 64-bit
  int getCacheLineSize() const override { return 64; }
  int64_t getL1CacheSize() const override { return 32768; }  // 32KB
  int64_t getL2CacheSize() const override { return 262144; } // 256KB

  // --- Reduction splitting ---
  int64_t getReductionSplitFactor(int64_t tripCount) const override {
    return std::max<int64_t>(
        1, std::min<int64_t>(tripCount,
                             static_cast<int64_t>(getWorkerCount()) / 4));
  }

  // --- Topology ---
  int getWorkerCount() const override {
    return machine.getRuntimeTotalWorkers();
  }
  int getNodeCount() const override { return machine.getNodeCount(); }
};

} // namespace mlir::arts

#endif // ARTS_UTILS_COSTS_ARTSCOSTMODEL_H
