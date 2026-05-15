///==========================================================================///
/// File: BlockLoopStripMining.cpp
///
/// Strip-mine innermost loops that access block-partitioned datablocks.
/// This is a late DB-aware cleanup that must run after DbPartitioning.
/// It removes per-iteration div/rem + db_ref overhead by introducing a
/// block loop and local index loop, reusing db_ref per block.
///
/// Transform implementations live in BlockLoopStripMiningSupport.cpp.
///==========================================================================///

#include "arts/dialect/core/Transforms/BlockLoopStripMiningInternal.h"
#define GEN_PASS_DEF_BLOCKLOOPSTRIPMINING
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

ARTS_DEBUG_SETUP(block_loop_strip_mining);

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::arts::block_loop_strip_mining;

namespace {

struct BlockLoopStripMiningPass
    : public impl::BlockLoopStripMiningBase<BlockLoopStripMiningPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    bool changed = false;
    do {
      changed = false;
      DominanceInfo domInfo(func);

      /// Re-run a post-order walk after each rewrite instead of holding a
      /// pre-collected loop list. Neighborhood strip-mining can rewrite
      /// enclosing loops, so fixed-point traversal avoids stale handles to
      /// nested loops while still keeping the transform generic.
      (void)func.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
        if (!loop->getBlock() || loop.getRegion().empty())
          return WalkResult::advance();
        if (isGeneratedByStripMining(loop))
          return WalkResult::advance();

        /// Neighborhood-aware strip-mining can safely rewrite enclosing loops
        /// that carry block-neighborhood div/rem state into nested stencil
        /// loops. Restrict only the single-dimension block strip-miner to
        /// innermost loops; otherwise patterns like Seidel keep their expensive
        /// row-boundary carry math because the relevant loop is not the
        /// innermost point loop.
        if (auto neighborhoodInfo = analyzeNeighborhoodLoop(loop, domInfo)) {
          if (stripMineNeighborhoodLoop(loop, *neighborhoodInfo)) {
            ARTS_DEBUG("Applied neighborhood-aware block strip-mining to loop");
            changed = true;
            return WalkResult::interrupt();
          }
        }

        if (!isInnermostLoop(loop))
          return WalkResult::advance();

        auto info = analyzeLegacyLoop(loop, domInfo);
        if (!info)
          return WalkResult::advance();

        if (stripMineLegacyLoop(loop, *info)) {
          ARTS_DEBUG("Applied block strip-mining to loop");
          changed = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    } while (changed);

    clearGeneratedStripMiningMarks(func);
  }
};

} // namespace

namespace mlir {
namespace arts {
/// Create the block loop strip-mining pass used to reduce per-iteration
/// div/rem + db_ref overhead for block-partitioned access patterns.
std::unique_ptr<Pass> createBlockLoopStripMiningPass() {
  return std::make_unique<BlockLoopStripMiningPass>();
}
/// End block loop strip-mining pass creation.
} // namespace arts
} // namespace mlir
