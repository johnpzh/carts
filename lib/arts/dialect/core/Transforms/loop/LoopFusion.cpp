///==========================================================================///
/// File: LoopFusion.cpp
///
/// Fuses independent arts.for loops within same parallel EDT.
/// Uses AnalysisManager and LoopAnalysis for bounds compatibility while
/// proving independence from per-loop memory hazards.
///
/// Before (2 separate loops with implicit barrier between them):
///   arts.edt <parallel> {
///     arts.for(%c0) to(%N) {  /// Stage 1: E = A * B
///       %E_view = arts.db_acquire ...
///       /// compute E[i]
///       arts.db_release %E_view
///     }
///     /// implicit barrier here
///     arts.for(%c0) to(%N) {  /// Stage 2: F = C * D (independent!)
///       %F_view = arts.db_acquire ...
///       /// compute F[i]
///       arts.db_release %F_view
///     }
///   }
///
/// After (independent loops fused, no barrier):
///   arts.edt <parallel> {
///     arts.for(%c0) to(%N) {  /// Fused: E & F computed together
///       %E_view = arts.db_acquire ...
///       /// compute E[i]
///       arts.db_release %E_view
///       %F_view = arts.db_acquire ...
///       /// compute F[i]
///       arts.db_release %F_view
///     }
///   }
///
/// This reduces EDT overhead and synchronization costs when loops have no
/// cross-loop write hazards. Shared read-only inputs are allowed.
///==========================================================================///

#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/loop/LoopAnalysis.h"
#include "arts/passes/Passes.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/LoopUtils.h"
#include "arts/utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "arts/utils/Debug.h"

using namespace mlir;
using namespace mlir::arts;

#define GEN_PASS_DEF_LOOPFUSION
#include "arts/passes/Passes.h.inc"

ARTS_DEBUG_SETUP(loop_fusion);

namespace {
struct LoopFusionPass : public ::impl::LoopFusionBase<LoopFusionPass> {
  LoopFusionPass(mlir::arts::AnalysisManager *AM)
      : AM(AM), numAdjacentLoopPairsExamined(
                    this, "num-adjacent-loop-pairs-examined",
                    "Number of adjacent arts.for pairs examined for fusion"),
        numPairsBlockedByBarrier(
            this, "num-loop-pairs-blocked-by-barrier",
            "Number of adjacent arts.for pairs skipped because a barrier "
            "separated them"),
        numPairsRejectedByBounds(
            this, "num-loop-pairs-rejected-by-bounds",
            "Number of adjacent arts.for pairs rejected due to incompatible "
            "bounds"),
        numPairsRejectedByDependence(
            this, "num-loop-pairs-rejected-by-dependence",
            "Number of adjacent arts.for pairs rejected by dependence checks"),
        numLoopsFused(this, "num-loops-fused",
                      "Number of arts.for loops fused") {
    assert(AM && "AnalysisManager must be provided externally");
  }
  LoopFusionPass(const LoopFusionPass &other)
      : ::impl::LoopFusionBase<LoopFusionPass>(other), AM(other.AM),
        numAdjacentLoopPairsExamined(
            this, "num-adjacent-loop-pairs-examined",
            "Number of adjacent arts.for pairs examined for fusion"),
        numPairsBlockedByBarrier(
            this, "num-loop-pairs-blocked-by-barrier",
            "Number of adjacent arts.for pairs skipped because a barrier "
            "separated them"),
        numPairsRejectedByBounds(
            this, "num-loop-pairs-rejected-by-bounds",
            "Number of adjacent arts.for pairs rejected due to incompatible "
            "bounds"),
        numPairsRejectedByDependence(
            this, "num-loop-pairs-rejected-by-dependence",
            "Number of adjacent arts.for pairs rejected by dependence checks"),
        numLoopsFused(this, "num-loops-fused",
                      "Number of arts.for loops fused") {}

  void runOnOperation() override {
    ModuleOp module = getOperation();

    /// Get LoopAnalysis from AnalysisManager
    LoopAnalysis &loopAnalysis = AM->getLoopAnalysis();

    module.walk([&](EdtOp parallelEdt) {
      if (parallelEdt.getType() != EdtType::parallel)
        return;

      SmallVector<ForOp, 8> forOps;
      for (Operation &op : parallelEdt.getBody().front()) {
        if (auto forOp = dyn_cast<ForOp>(&op))
          forOps.push_back(forOp);
      }

      bool changed = true;
      while (changed) {
        changed = false;
        for (size_t i = 0; i + 1 < forOps.size(); ++i) {
          ForOp a = forOps[i];
          ForOp b = forOps[i + 1];
          ++numAdjacentLoopPairsExamined;

          /// Skip if barrier between them
          Operation *next = a->getNextNode();
          while (next && next != b.getOperation()) {
            if (isa<BarrierOp>(next))
              break;
            next = next->getNextNode();
          }
          if (next != b.getOperation()) {
            ++numPairsBlockedByBarrier;
            continue;
          }

          if (!haveCompatibleBounds(a, b)) {
            ++numPairsRejectedByBounds;
            continue;
          }

          if (!areIndependent(a, b, loopAnalysis)) {
            ++numPairsRejectedByDependence;
            continue;
          }

          {
            ARTS_INFO("Fusing independent for loops");
            fuseForOps(a, b);
            ++numLoopsFused;
            forOps.erase(forOps.begin() + i + 1);
            changed = true;
            break;
          }
        }
      }
    });
  }

private:
  mlir::arts::AnalysisManager *AM = nullptr;
  Statistic numAdjacentLoopPairsExamined;
  Statistic numPairsBlockedByBarrier;
  Statistic numPairsRejectedByBounds;
  Statistic numPairsRejectedByDependence;
  Statistic numLoopsFused;

  bool haveCompatibleBounds(ForOp a, ForOp b);
  bool areIndependent(ForOp a, ForOp b, LoopAnalysis &loopAnalysis);
  void fuseForOps(ForOp first, ForOp second);
};
} // namespace

bool LoopFusionPass::haveCompatibleBounds(ForOp a, ForOp b) {
  auto &loopAnalysis = AM->getLoopAnalysis();
  LoopNode *nodeA = loopAnalysis.getOrCreateLoopNode(a.getOperation());
  LoopNode *nodeB = loopAnalysis.getOrCreateLoopNode(b.getOperation());
  return LoopNode::haveCompatibleBounds(nodeA, nodeB);
}

bool LoopFusionPass::areIndependent(ForOp a, ForOp b,
                                    LoopAnalysis &loopAnalysis) {
  auto loopAClass = classifyPointwiseLoopCompute(a);
  auto loopBClass = classifyPointwiseLoopCompute(b);
  if (loopAClass != loopBClass) {
    ARTS_DEBUG("Loops have different pointwise compute classes - keep "
               "separate");
    return false;
  }

  /// First check: no shared writable accesses between loops.
  ARTS_INFO("Checking independence between two ForOps");
  if (DbUtils::hasSharedWritableRootConflict(a.getOperation(),
                                             b.getOperation())) {
    ARTS_DEBUG("Loops share a writable memory root - not independent");
    return false;
  }

  /// Second check: consult loop metadata for diagnostics only. Internal
  /// loop-carried dependences do not, by themselves, block fusion because the
  /// fused loop preserves each iteration's program order. Only cross-loop
  /// hazards matter here.
  LoopNode *loopNodeA = loopAnalysis.getLoopNode(a.getOperation());
  LoopNode *loopNodeB = loopAnalysis.getLoopNode(b.getOperation());

  (void)loopNodeA;
  (void)loopNodeB;

  ARTS_DEBUG("Loops are independent - can fuse");
  return true; /// No shared memrefs - loops are independent
}

void LoopFusionPass::fuseForOps(ForOp first, ForOp second) {
  OpBuilder builder(second);
  Block &firstBody = first.getRegion().front();
  Block &secondBody = second.getRegion().front();

  IRMapping mapper;
  if (secondBody.getNumArguments() > 0 && firstBody.getNumArguments() > 0)
    mapper.map(secondBody.getArgument(0), firstBody.getArgument(0));

  builder.setInsertionPoint(firstBody.getTerminator());
  for (Operation &op : secondBody.without_terminator())
    builder.clone(op, mapper);

  second.erase();
}

namespace mlir {
namespace arts {
std::unique_ptr<Pass> createLoopFusionPass(mlir::arts::AnalysisManager *AM) {
  return std::make_unique<LoopFusionPass>(AM);
}
} // namespace arts
} // namespace mlir
