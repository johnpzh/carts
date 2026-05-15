///==========================================================================///
/// File: ReductionStrategy.cpp
///
/// Cost-model-backed SDE reduction strategy selection. The current
/// implementation stays loop-uniform: it annotates eligible reduction-bearing
/// sde.su_iterate ops with a single SDE-owned recommendation and leaves
/// lowering behavior unchanged.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_REDUCTIONSTRATEGY
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "carts/dialect/sde/Utils/SDECostModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::carts;

namespace {

static bool isAtomicCapable(sde::SdeReductionKind kind, Type elementType) {
  return kind == sde::SdeReductionKind::add && !isa<FloatType>(elementType);
}

static Type getAccumulatorElementType(Value accumulator) {
  if (auto shapedType = dyn_cast<ShapedType>(accumulator.getType()))
    return shapedType.getElementType();
  return accumulator.getType();
}

static bool hasNestedSequentialLoop(sde::SdeSuIterateOp op) {
  bool found = false;
  op->walk([&](Operation *nested) {
    if (isa<scf::ForOp, affine::AffineForOp>(nested)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static std::optional<SmallVector<sde::SdeReductionKind>>
getReductionKinds(sde::SdeSuIterateOp op) {
  if (op.getReductionAccumulators().empty())
    return std::nullopt;

  auto kinds = op.getReductionKindsAttr();
  if (!kinds || kinds.size() != op.getReductionAccumulators().size())
    return std::nullopt;

  SmallVector<sde::SdeReductionKind> parsedKinds;
  parsedKinds.reserve(kinds.size());
  for (Attribute attr : kinds) {
    auto kindAttr = dyn_cast<sde::SdeReductionKindAttr>(attr);
    if (!kindAttr || kindAttr.getValue() == sde::SdeReductionKind::custom)
      return std::nullopt;
    parsedKinds.push_back(kindAttr.getValue());
  }

  return parsedKinds;
}

struct ReductionStrategyPass
    : public arts::impl::ReductionStrategyBase<ReductionStrategyPass> {
  explicit ReductionStrategyPass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    if (!costModel)
      return;

    int64_t workerCount =
        std::max<int64_t>(1, costModel->getLogicalWorkerCapacity());
    if (workerCount <= 1)
      return;

    getOperation().walk([&](sde::SdeSuIterateOp op) {
      if (op.getReductionStrategyAttr())
        return;

      std::optional<SmallVector<sde::SdeReductionKind>> reductionKinds =
          getReductionKinds(op);
      if (!reductionKinds)
        return;

      auto strategy = sde::SdeReductionStrategy::tree;
      if (hasNestedSequentialLoop(op)) {
        strategy = sde::SdeReductionStrategy::local_accumulate;
      } else if (llvm::all_of(
                     llvm::zip(*reductionKinds, op.getReductionAccumulators()),
                     [](auto pair) {
                       auto [kind, accumulator] = pair;
                       return isAtomicCapable(
                           kind, getAccumulatorElementType(accumulator));
                     })) {
        double atomicCost =
            static_cast<double>(workerCount) * costModel->getAtomicUpdateCost();
        double collectiveCost = costModel->getReductionCost(workerCount);
        if (atomicCost <= collectiveCost)
          strategy = sde::SdeReductionStrategy::atomic;
      }

      op.setReductionStrategyAttr(
          sde::SdeReductionStrategyAttr::get(&getContext(), strategy));
    });
  }

private:
  sde::SDECostModel *costModel = nullptr;
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass>
createReductionStrategyPass(sde::SDECostModel *costModel) {
  return std::make_unique<ReductionStrategyPass>(costModel);
}

} // namespace mlir::carts::sde
