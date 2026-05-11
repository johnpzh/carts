///==========================================================================///
/// File: EdtDistribution.cpp
///
/// Annotates arts.for loops with distribution strategy decisions while keeping
/// loop semantics unchanged.
///
/// Transformation (annotation only):
///   BEFORE:
///     arts.for (...) {
///       ...
///     }
///
///   AFTER:
///     arts.for (...) {
///       ...
///     } {distribution_kind = ..., distribution_pattern = ...}
///
/// This pass does not create/remove dependencies; it only writes typed attrs
///
/// Example:
///   matmul loop in internode EDT -> annotation picks
///   `distribution_kind=tiling_2d`
/// consumed later by lowering.
///==========================================================================///

#define GEN_PASS_DEF_EDTDISTRIBUTION
#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisDependencies.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/KernelFamily.h"
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/StencilAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <cassert>

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_distribution);

using namespace mlir::arts;

static const AnalysisKind kEdtDistribution_reads[] = {
    AnalysisKind::DbAnalysis, AnalysisKind::EdtHeuristics,
    AnalysisKind::RuntimeConfig};
static const AnalysisKind kEdtDistribution_invalidates[] = {
    AnalysisKind::DbAnalysis};
[[maybe_unused]] static const AnalysisDependencyInfo kEdtDistribution_deps = {
    kEdtDistribution_reads, kEdtDistribution_invalidates};

#include "llvm/ADT/Statistic.h"
static llvm::Statistic numLoopsAnnotated{
    "edt_distribution", "NumLoopsAnnotated",
    "Number of arts.for loops annotated with distribution strategy"};
static llvm::Statistic numBlockStrategiesSelected{
    "edt_distribution", "NumBlockStrategiesSelected",
    "Number of loops assigned block distribution kind"};
static llvm::Statistic numCyclicStrategiesSelected{
    "edt_distribution", "NumCyclicStrategiesSelected",
    "Number of loops assigned cyclic distribution kind"};
static llvm::Statistic numPlanDrivenAnnotations{
    "edt_distribution", "NumPlanDrivenAnnotations",
    "Number of loops annotated using structured kernel plan"};

using namespace mlir;

namespace {

struct EdtDistributionPass
    : public impl::EdtDistributionBase<EdtDistributionPass> {
  explicit EdtDistributionPass(mlir::arts::AnalysisManager *AM) : AM(AM) {
    assert(AM && "AnalysisManager must be provided externally");
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *machine = &AM->getRuntimeConfig();
    if (!machine->hasConfigFile() || !machine->hasValidNodeCount() ||
        !machine->hasValidThreads()) {
      module.emitError(
          "invalid ARTS machine configuration for distribution planning");
      signalPassFailure();
      return;
    }

    /// Query distribution patterns from DB analysis built from the current IR.
    AM->getDbAnalysis().invalidate();

    module.walk([&](EdtOp edt) {
      if (edt.getType() != EdtType::parallel && edt.getType() != EdtType::task)
        return;

      auto &heuristics = AM->getEdtHeuristics();
      DistributionStrategy strategy =
          heuristics.chooseStrategy(edt.getConcurrency());

      edt.walk([&](ForOp forOp) {
        if (forOp->getParentOfType<EdtOp>() != edt)
          return;

        auto existingKind = getEdtDistributionKind(forOp.getOperation());
        auto existingPattern = getEdtDistributionPattern(forOp.getOperation());
        if (existingKind && existingPattern) {
          if (!forOp->hasAttr(AttrNames::Operation::DistributionVersion)) {
            forOp->setAttr(
                AttrNames::Operation::DistributionVersion,
                IntegerAttr::get(IntegerType::get(forOp.getContext(), 32), 1));
          }
          return;
        }

        /// Plan-driven path: if the structured kernel plan stamped a family,
        /// use it to inform the distribution pattern selection.
        EdtDistributionPattern pattern =
            existingPattern.value_or(EdtDistributionPattern::unknown);
        if (pattern == EdtDistributionPattern::unknown)
          if (auto familyAttr = forOp->getAttrOfType<StringAttr>(
                  AttrNames::Operation::Plan::KernelFamily)) {
            auto family = parseKernelFamily(familyAttr.getValue());
            if (family) {
              switch (*family) {
              case KernelFamily::Stencil:
              case KernelFamily::Wavefront:
              case KernelFamily::TimestepChain:
                pattern = EdtDistributionPattern::stencil;
                break;
              case KernelFamily::Uniform:
                pattern = EdtDistributionPattern::uniform;
                break;
              case KernelFamily::ReductionMixed:
                pattern = EdtDistributionPattern::matmul;
                break;
              case KernelFamily::Unknown:
                break;
              }
              ++numPlanDrivenAnnotations;
              ARTS_DEBUG("Using plan-driven distribution pattern for family="
                         << familyAttr.getValue());
            }
          }
        /// Fallback: use heuristic analysis when no plan is present.
        if (pattern == EdtDistributionPattern::unknown) {
          if (auto analyzedPattern =
                  heuristics.resolveDistributionPattern(forOp, edt))
            pattern = *analyzedPattern;
        }
        EdtDistributionKind kind =
            existingKind.value_or(heuristics.chooseKind(strategy, pattern));
        /// Keep distribution_kind focused on execution decomposition.
        /// 2-D stencil owner semantics already flow through the
        /// stencil/lowering contract attrs (owner dims, halo, block shape).
        /// Overloading `tiling_2d` here forces stencil loops such as Seidel
        /// through the matmul-oriented Tile2DTaskLoopLowering path even when
        /// they only need block-style task distribution with N-D ownership
        /// preserved.
        if (!existingKind)
          setEdtDistributionKind(forOp.getOperation(), kind);
        if (!existingPattern && pattern != EdtDistributionPattern::unknown)
          setEdtDistributionPattern(forOp.getOperation(), pattern);
        ++numLoopsAnnotated;
        if (kind == EdtDistributionKind::block)
          ++numBlockStrategiesSelected;
        else if (kind == EdtDistributionKind::block_cyclic)
          ++numCyclicStrategiesSelected;
        forOp->setAttr(
            AttrNames::Operation::DistributionVersion,
            IntegerAttr::get(IntegerType::get(forOp.getContext(), 32), 1));

        ARTS_DEBUG("Annotated arts.for with kind="
                   << static_cast<int>(kind)
                   << " pattern=" << static_cast<int>(pattern));
      });
    });
  }

private:
  mlir::arts::AnalysisManager *AM = nullptr;
};

} // namespace

namespace mlir {
namespace arts {
std::unique_ptr<Pass>
createEdtDistributionPass(mlir::arts::AnalysisManager *AM) {
  return std::make_unique<EdtDistributionPass>(AM);
}
} // namespace arts
} // namespace mlir
