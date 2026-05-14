///==========================================================================///
/// File: EpochOptScheduling.cpp
///
/// Scheduling-oriented epoch transforms used by EpochOpt.
///==========================================================================///

#include "arts/dialect/core/Transforms/EpochOptInternal.h"
#include "arts/utils/Debug.h"

ARTS_DEBUG_SETUP(epoch_opt);

using namespace mlir;
using namespace mlir::arts;

namespace mlir::arts::epoch_opt {

LogicalResult
transformToContinuation(EpochOp epochOp,
                        const EpochContinuationDecision &decision) {
  OpBuilder builder(epochOp->getContext());
  Location loc = epochOp.getLoc();

  ARTS_DEBUG(
      "  DB acquires captured: " << decision.capturedDbAcquireValues.size());

  SmallVector<Value> deps(decision.capturedDbAcquireValues.begin(),
                          decision.capturedDbAcquireValues.end());

  builder.setInsertionPointAfter(epochOp);
  auto edtOp = EdtOp::create(builder, loc, EdtType::task,
                             EdtConcurrency::intranode, deps);
  edtOp->setAttr(ControlDep, builder.getIntegerAttr(builder.getI32Type(), 1));
  edtOp->setAttr(ContinuationForEpoch, builder.getUnitAttr());

  Block &edtBlock = edtOp.getBody().front();
  for (Value dep : deps)
    edtBlock.addArgument(dep.getType(), loc);
  builder.setInsertionPointToStart(&edtBlock);

  IRMapping valueMapping;
  for (auto [oldDep, blockArg] : llvm::zip(deps, edtBlock.getArguments()))
    valueMapping.map(oldDep, blockArg);

  for (Operation *op : decision.tailOps) {
    Operation *cloned = builder.clone(*op, valueMapping);
    for (auto [oldRes, newRes] :
         llvm::zip(op->getResults(), cloned->getResults()))
      valueMapping.map(oldRes, newRes);
  }

  if (edtBlock.empty() || !edtBlock.back().hasTrait<OpTrait::IsTerminator>())
    YieldOp::create(builder, loc);

  for (Operation *op : llvm::reverse(decision.tailOps))
    op->erase();

  epochOp->setAttr(ContinuationForEpoch, builder.getUnitAttr());

  ARTS_INFO("  Created continuation EDT with " << deps.size()
                                               << " DB deps + 1 control dep");
  return success();
}

} // namespace mlir::arts::epoch_opt
