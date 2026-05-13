///==========================================================================///
/// File: EdtDistribution.cpp
///
/// Completes ARTS distribution contracts while keeping loop semantics
/// unchanged.
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
///   SDE-classified loop carrying depPattern=<matmul> and distribution_kind
///   keeps that SDE-authored decomposition and completes the matching
///   distribution_pattern=<matmul> contract for Core lowering.
///==========================================================================///

#define GEN_PASS_DEF_EDTDISTRIBUTION
#include "arts/Dialect.h"
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/OperationAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <cassert>

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_distribution);

using namespace mlir::arts;

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

using namespace mlir;

namespace {

static void globalizeChunkLocalInnerLoopBounds(ModuleOp module) {
  // When arts.for has block distribution but the acquire is coarse (no
  // per-task slice geometry), the SDE-generated body can use chunk-local
  // indexing inside an inner scf.for. Globalize the inner loop bounds so each
  // task writes to its own slice. Conservative guards keep this surgical: only
  // the SDE chunk-loop signature is rewritten.
  module.walk([](ForOp artsFor) {
    auto kind = getEdtDistributionKind(artsFor);
    if (!kind || (*kind != EdtDistributionKind::block &&
                  *kind != EdtDistributionKind::block_cyclic))
      return;

    Block *body = artsFor.getBody();
    if (!body || body->getNumArguments() == 0)
      return;
    Value artsForIV = body->getArgument(0);

    scf::ForOp inner = nullptr;
    for (Operation &op : body->getOperations()) {
      if (auto sf = dyn_cast<scf::ForOp>(&op)) {
        if (inner)
          return;
        inner = sf;
      }
    }
    if (!inner)
      return;

    auto lbConst =
        inner.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    if (!lbConst || lbConst.value() != 0)
      return;

    auto minOp = inner.getUpperBound().getDefiningOp<arith::MinUIOp>();
    if (!minOp)
      return;
    auto subOp = minOp.getRhs().getDefiningOp<arith::SubIOp>();
    if (!subOp)
      subOp = minOp.getLhs().getDefiningOp<arith::SubIOp>();
    if (!subOp || subOp.getRhs() != artsForIV)
      return;

    OpBuilder builder(inner);
    Value newUpper = arith::AddIOp::create(builder, inner.getLoc(), artsForIV,
                                           inner.getUpperBound());
    inner.getLowerBoundMutable().assign(artsForIV);
    inner.getUpperBoundMutable().assign(newUpper);
  });
}

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
          if (!getDistributionVersionAttr(forOp.getOperation()))
            setDistributionVersion(forOp.getOperation(), 1);
          return;
        }

        std::optional<EdtDistributionPattern> parentPattern;
        if (!existingPattern) {
          parentPattern = getEdtDistributionPattern(edt.getOperation());
          if (parentPattern && *parentPattern == EdtDistributionPattern::unknown)
            parentPattern = std::nullopt;
        }

        EdtDistributionPattern pattern =
            existingPattern.value_or(parentPattern.value_or(
                EdtDistributionPattern::unknown));
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
        if (!existingPattern && parentPattern)
          setEdtDistributionPattern(forOp.getOperation(), *parentPattern);
        ++numLoopsAnnotated;
        if (kind == EdtDistributionKind::block)
          ++numBlockStrategiesSelected;
        else if (kind == EdtDistributionKind::block_cyclic)
          ++numCyclicStrategiesSelected;
        setDistributionVersion(forOp.getOperation(), 1);

        ARTS_DEBUG("Annotated arts.for with kind="
                   << static_cast<int>(kind)
                   << " pattern=" << static_cast<int>(pattern));
      });
    });

    globalizeChunkLocalInnerLoopBounds(module);
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
