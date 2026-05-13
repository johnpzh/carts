///==========================================================================///
/// File: Passes.h
///
/// This file declares pass creation APIs and pass registration hooks
/// for the ARTS dialect.
///==========================================================================///

#ifndef ARTS_PASSES_PASSES_H
#define ARTS_PASSES_PASSES_H

#include "arts/Dialect.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Dialect.h"

namespace mlir {
class PatternRewriter;
class RewritePatternSet;
class DominanceInfo;

namespace arts {
class AnalysisManager;
class RuntimeConfig;

/// General IR cleanup and canonicalization passes.
std::unique_ptr<Pass> createArtsInlinerPass();
/// Raise nested pointer allocations to N-dimensional memrefs.
std::unique_ptr<Pass> createMemrefNormalizationPass();
/// Convert residual OMP task dependencies to arts.omp_dep.
std::unique_ptr<Pass> createHandleDepsPass();
/// Eliminate dead ARTS operations and dead helper IR.
std::unique_ptr<Pass> createDCEPass();

/// EDT and loop-structure transformation passes.
std::unique_ptr<Pass> createEdtStructuralOptPass(AnalysisManager *AM,
                                                 bool runAnalysis);
std::unique_ptr<Pass> createConcurrencyPass(AnalysisManager *AM);
std::unique_ptr<Pass> createCreateDbsPass(AnalysisManager *AM);
std::unique_ptr<Pass> createDbModeTighteningPass(AnalysisManager *AM,
                                                 bool forceInout = false);
std::unique_ptr<Pass> createDbScratchEliminationPass();
std::unique_ptr<Pass> createDbDistributedOwnershipPass(AnalysisManager *AM);
std::unique_ptr<Pass> createDbTransformsPass(AnalysisManager *AM);
std::unique_ptr<Pass> createCreateEpochsPass();

/// Lower ARTS dialect operations into LLVM-ready IR.
std::unique_ptr<Pass> createConvertArtsToLLVMPass();
std::unique_ptr<Pass> createConvertArtsToLLVMPass(bool debug,
                                                  bool distributedInitPerWorker,
                                                  const RuntimeConfig *machine);

/// EDT-local cleanup and codegen-preparation passes.
std::unique_ptr<Pass> createEdtICMPass();
std::unique_ptr<Pass> createEdtAllocaSinkingPass();
std::unique_ptr<Pass> createDataPtrHoistingPass();
std::unique_ptr<Pass> createGuidRangCallOptPass();
std::unique_ptr<Pass> createRuntimeCallOptPass();
std::unique_ptr<Pass> createEdtPtrRematerializationPass();
std::unique_ptr<Pass> createDbLoweringPass(uint64_t idStride = 1000);
std::unique_ptr<Pass> createEpochLoweringPass();
std::unique_ptr<Pass> createParallelEdtLoweringPass();
std::unique_ptr<Pass> createEdtLoweringPass(uint64_t idStride = 1000);
std::unique_ptr<Pass> createEdtLoweringPass(AnalysisManager *AM,
                                            uint64_t idStride = 1000);
std::unique_ptr<Pass> createLoweringContractCleanupPass();

/// High-level scheduling and distribution passes.
std::unique_ptr<Pass> createForOptPass(AnalysisManager *AM);
std::unique_ptr<Pass>
createDistributedHostLoopOutliningPass(AnalysisManager *AM);
std::unique_ptr<Pass> createEdtDistributionPass(AnalysisManager *AM);
std::unique_ptr<Pass> createEdtOrchestrationOptPass();
std::unique_ptr<Pass> createForLoweringPass();
std::unique_ptr<Pass> createForLoweringPass(AnalysisManager *AM);
std::unique_ptr<Pass> createLoopFusionPass(AnalysisManager *AM);
std::unique_ptr<Pass> createEpochOptPass();
std::unique_ptr<Pass> createEpochOptPass(AnalysisManager *AM);
/// Create EpochOpt with explicit scheduling flags.
/// Structural opts (narrowing/fusion/loop-fusion) are always enabled.
std::unique_ptr<Pass> createEpochOptPass(AnalysisManager *AM,
                                         bool enableAmortization,
                                         bool enableContinuation,
                                         bool enableCPSDriver,
                                         bool enableCPSChain = false);
/// Create EpochOpt with scheduling-only flags (structural opts disabled).
std::unique_ptr<Pass> createEpochOptSchedulingPass(AnalysisManager *AM,
                                                   bool enableAmortization,
                                                   bool enableContinuation,
                                                   bool enableCPSDriver,
                                                   bool enableCPSChain = false);
std::unique_ptr<Pass> createHoistingPass();
std::unique_ptr<Pass> createBlockLoopStripMiningPass();

std::unique_ptr<Pass> createEdtTransformsPass(AnalysisManager *AM);

/// Validation passes for lowering contracts.
std::unique_ptr<Pass> createContractValidationPass(bool failOnError = false);

/// Verification passes at lowering boundaries.
std::unique_ptr<Pass> createVerifyEdtCreatedPass();
std::unique_ptr<Pass> createVerifyEpochCreatedPass();
std::unique_ptr<Pass> createVerifyForLoweredPass();
std::unique_ptr<Pass> createVerifyCoreObjectsOnlyPass(bool reportOnly = false);
std::unique_ptr<Pass> createVerifyPreLoweredPass();
std::unique_ptr<Pass> createVerifyLoweredPass();
std::unique_ptr<Pass> createVerifyEdtLoweredPass();
std::unique_ptr<Pass> createVerifyDbLoweredPass();
std::unique_ptr<Pass> createVerifyEpochLoweredPass();
} // namespace arts
} // namespace mlir

namespace mlir {
/// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace omp {
class OpenMPDialect;
} // namespace omp

namespace memref {
class MemRefDialect;
} // namespace memref

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

namespace func {
class FuncDialect;
} // namespace func

namespace arith {
class ArithDialect;
} // namespace arith

namespace polygeist {
class PolygeistDialect;
} // namespace polygeist

// Core ARTS passes (generated from include/arts/passes/Passes.td)
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "arts/passes/Passes.h.inc"

} // namespace mlir

// Per-dialect pass declarations (self-contained headers with dialect deps)
#include "arts/dialect/rt/Transforms/Passes.h"
#include "arts/dialect/sde/Transforms/Passes.h"

// Per-dialect pass registrations (need create functions visible from above)
#define GEN_PASS_REGISTRATION
#include "arts/dialect/rt/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "arts/dialect/sde/Transforms/Passes.h.inc"

#endif /// ARTS_PASSES_PASSES_H
