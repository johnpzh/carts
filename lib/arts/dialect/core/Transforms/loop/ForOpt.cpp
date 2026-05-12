///==========================================================================///
/// File: ForOpt.cpp
///
/// Applies pre-lowering optimization hints to arts.for loops.
///
/// Example:
///   Before:
///     arts.for ... { ... }
///
///   After:
///     arts.for ... { ... }  // with tuning attrs/hints for downstream lowering
///==========================================================================///

#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#define GEN_PASS_DEF_FOROPT
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/OperationAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "arts/utils/Debug.h"
#include "llvm/ADT/Statistic.h"
ARTS_DEBUG_SETUP(for_opt);

using namespace mlir::arts;

static llvm::Statistic numLoopsAnnotatedWithHints{
    "for_opt", "NumLoopsAnnotatedWithHints",
    "Number of arts.for loops annotated with partition hints"};
static llvm::Statistic numAccessPatternsAnnotated{
    "for_opt", "NumAccessPatternsAnnotated",
    "Number of DB allocs annotated with access patterns"};
static llvm::Statistic numLoopsSkippedExistingHint{
    "for_opt", "NumLoopsSkippedExistingHint",
    "Number of loops skipped because they already had a hint"};

using namespace mlir;

namespace {

static void annotateAccessPatterns(ModuleOp module,
                                   mlir::arts::AnalysisManager *AM) {
  if (!AM)
    return;

  /// Rebuild DB analysis from current IR before querying alloc access patterns.
  auto &dbAnalysis = AM->getDbAnalysis();
  dbAnalysis.invalidate();

  module.walk([&](DbAllocOp allocOp) {
    if (allocOp.getElementSizes().empty())
      return;

    DbAccessPattern pattern = DbAccessPattern::uniform;
    if (auto analyzed = dbAnalysis.getAllocAccessPattern(allocOp))
      pattern = *analyzed;
    if (pattern == DbAccessPattern::unknown)
      pattern = DbAccessPattern::uniform;

    setDbAccessPattern(allocOp.getOperation(), pattern);
    ++numAccessPatternsAnnotated;
    ARTS_DEBUG("Annotated alloc " << allocOp << " with access_pattern="
                                  << stringifyDbAccessPattern(pattern));
  });
}

struct ForOptPass : public impl::ForOptBase<ForOptPass> {
  explicit ForOptPass(mlir::arts::AnalysisManager *AM) : AM(AM) {
    assert(AM && "AnalysisManager must be provided externally");
  }

  void runOnOperation() override {
    module = getOperation();

    ARTS_INFO_HEADER(ForOpt);
    ARTS_DEBUG_REGION(module.dump(););

    annotateAccessPatterns(module, AM);

    module.walk([&](EdtOp parallelEdt) {
      if (parallelEdt.getType() != EdtType::parallel)
        return;

      auto &heuristics = AM->getEdtHeuristics();
      auto workerCfg = heuristics.resolveWorkerConfig(parallelEdt);
      if (!workerCfg) {
        ARTS_DEBUG("Skipping arts.for optimization (unknown worker count)");
        return;
      }

      parallelEdt.walk([&](ForOp forOp) {
        if (forOp->getParentOfType<EdtOp>() != parallelEdt)
          return;

        /// Respect explicit user/compiler hints already present.
        if (getPartitioningHint(forOp.getOperation())) {
          ++numLoopsSkippedExistingHint;
          return;
        }

        auto decision =
            heuristics.computeLoopCoarseningDecision(forOp, *workerCfg);
        if (!decision.blockSize)
          return;

        setPartitioningHint(forOp.getOperation(),
                            PartitioningHint::block(*decision.blockSize));
        ++numLoopsAnnotatedWithHints;

        ARTS_INFO("Set arts.partition_hint blockSize="
                  << *decision.blockSize << " for arts.for ("
                  << stringifyEdtConcurrency(parallelEdt.getConcurrency())
                  << ", workers=" << workerCfg->totalWorkers
                  << ", desired_workers=" << decision.desiredWorkers << ")");
      });
    });

    ARTS_INFO_FOOTER(ForOpt);
    ARTS_DEBUG_REGION(module.dump(););
  }

private:
  mlir::arts::AnalysisManager *AM = nullptr;
  ModuleOp module;
};

} // namespace

namespace mlir {
namespace arts {
std::unique_ptr<Pass> createForOptPass(mlir::arts::AnalysisManager *AM) {
  return std::make_unique<ForOptPass>(AM);
}
} // namespace arts
} // namespace mlir
