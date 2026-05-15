///==========================================================================///
/// File: ScheduleRefinement.cpp
///
/// Cost-model-backed SDE schedule refinement. The current implementation keeps
/// the legality gate deliberately narrow: it only rewrites one-dimensional
/// loops whose source schedule remained abstract (`auto` or `runtime`), and
/// selects the cheapest concrete SDE schedule with the active cost model
/// before chunk synthesis runs. Constant-trip loops are refined directly;
/// symbolic-trip loops refine only when the same schedule wins across a small
/// set of SDE-level probe trip counts derived from the active cost model.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_SCHEDULEREFINEMENT
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/LoopUtils.h"
#include "arts/utils/ValueAnalysis.h"
#include "arts/utils/costs/SDECostModel.h"

#include "mlir/IR/BuiltinOps.h"

#include <array>
#include <limits>

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::carts;

namespace {

static bool isScheduleRefinementCandidate(sde::SdeScheduleKindAttr schedule) {
  if (!schedule)
    return false;
  return schedule.getValue() == sde::SdeScheduleKind::auto_ ||
         schedule.getValue() == sde::SdeScheduleKind::runtime;
}

static bool isOneDimensionalLoop(sde::SdeSuIterateOp op) {
  return op.getLowerBounds().size() == 1 && op.getUpperBounds().size() == 1 &&
         op.getSteps().size() == 1;
}

static bool hasPositiveConstantStep(sde::SdeSuIterateOp op) {
  if (!isOneDimensionalLoop(op))
    return false;

  int64_t step = 0;
  return arts::ValueAnalysis::getConstantIndex(op.getSteps().front(), step) &&
         step > 0;
}

static sde::SdeScheduleKind selectSchedule(sde::SDECostModel &costModel,
                                           int64_t tripCount) {
  static constexpr std::array<sde::SdeScheduleKind, 3> kCandidateSchedules = {
      sde::SdeScheduleKind::static_, sde::SdeScheduleKind::guided,
      sde::SdeScheduleKind::dynamic};

  sde::SdeScheduleKind bestSchedule = kCandidateSchedules.front();
  double bestCost = costModel.getSchedulingOverhead(bestSchedule, tripCount);
  for (sde::SdeScheduleKind candidate :
       ArrayRef(kCandidateSchedules).drop_front()) {
    double candidateCost =
        costModel.getSchedulingOverhead(candidate, tripCount);
    if (candidateCost < bestCost) {
      bestSchedule = candidate;
      bestCost = candidateCost;
    }
  }
  return bestSchedule;
}

static int64_t saturatingMultiply(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0)
    return 1;
  if (lhs > std::numeric_limits<int64_t>::max() / rhs)
    return std::numeric_limits<int64_t>::max();
  return lhs * rhs;
}

static std::optional<sde::SdeScheduleKind>
selectSymbolicSchedule(sde::SDECostModel &costModel) {
  int64_t workerCount =
      std::max<int64_t>(1, costModel.getLogicalWorkerCapacity());
  int64_t minIterations =
      std::max<int64_t>(1, costModel.getMinIterationsPerWorker());
  std::array<int64_t, 4> probeTripCounts = {
      1, minIterations, workerCount,
      saturatingMultiply(workerCount, minIterations)};

  std::optional<sde::SdeScheduleKind> selectedSchedule;
  for (int64_t probeTripCount : probeTripCounts) {
    sde::SdeScheduleKind candidate = selectSchedule(costModel, probeTripCount);
    if (!selectedSchedule) {
      selectedSchedule = candidate;
      continue;
    }
    if (*selectedSchedule != candidate)
      return std::nullopt;
  }

  return selectedSchedule;
}

struct ScheduleRefinementPass
    : public arts::impl::ScheduleRefinementBase<ScheduleRefinementPass> {
  explicit ScheduleRefinementPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    if (!costModel)
      return;

    getOperation().walk([&](sde::SdeSuIterateOp op) {
      if (op.getChunkSize() ||
          !isScheduleRefinementCandidate(op.getScheduleAttr()) ||
          !isOneDimensionalLoop(op))
        return;

      std::optional<int64_t> tripCount = getStaticTripCount(op.getOperation());
      std::optional<sde::SdeScheduleKind> refinedScheduleKind;
      if (tripCount) {
        refinedScheduleKind = selectSchedule(*costModel, *tripCount);
      } else if (hasPositiveConstantStep(op)) {
        refinedScheduleKind = selectSymbolicSchedule(*costModel);
      }
      if (!refinedScheduleKind)
        return;

      auto refinedSchedule =
          sde::SdeScheduleKindAttr::get(&getContext(), *refinedScheduleKind);
      if (refinedSchedule == op.getScheduleAttr())
        return;
      op.setScheduleAttr(refinedSchedule);
    });
  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass>
createScheduleRefinementPass(sde::SDECostModel *costModel) {
  return std::make_unique<ScheduleRefinementPass>(costModel);
}

} // namespace mlir::carts::sde
