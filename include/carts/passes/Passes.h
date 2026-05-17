///==========================================================================///
/// File: Passes.h
///
/// This file declares pass creation APIs and pass registration hooks
/// for the ARTS dialect.
///==========================================================================///

#ifndef ARTS_PASSES_PASSES_H
#define ARTS_PASSES_PASSES_H

#include "carts/Dialect.h"
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

namespace carts::arts {
class AnalysisManager;
class RuntimeConfig;

/// Eliminate dead ARTS operations and dead helper IR.
std::unique_ptr<Pass> createDCEPass();

/// EDT and loop-structure transformation passes.
std::unique_ptr<Pass> createEdtStructuralOptPass(AnalysisManager *AM,
                                                 bool runAnalysis);
std::unique_ptr<Pass> createCreateDbsPass(AnalysisManager *AM);
std::unique_ptr<Pass> createDbModeTighteningPass(AnalysisManager *AM,
                                                 bool forceInout = false);
std::unique_ptr<Pass> createDbScratchEliminationPass();
std::unique_ptr<Pass> createDbDistributedOwnershipPass(AnalysisManager *AM);
std::unique_ptr<Pass> createDbTransformsPass(AnalysisManager *AM);
std::unique_ptr<Pass> createCreateEpochsPass();

/// EDT-local cleanup and ARTS object refinement passes.
std::unique_ptr<Pass> createEdtAllocaSinkingPass();
std::unique_ptr<Pass> createEdtPtrRematerializationPass();
std::unique_ptr<Pass> createLoweringContractCleanupPass();

/// High-level epoch scheduling passes.
std::unique_ptr<Pass> createEpochOptPass();
std::unique_ptr<Pass> createEpochOptPass(AnalysisManager *AM);
/// Create EpochOpt with explicit scheduling flags.
/// Structural opts (narrowing/fusion) are always enabled.
std::unique_ptr<Pass> createEpochOptPass(AnalysisManager *AM,
                                         bool enableAmortization);
std::unique_ptr<Pass> createHoistingPass();
std::unique_ptr<Pass> createBlockLoopStripMiningPass();

std::unique_ptr<Pass> createEdtTransformsPass(AnalysisManager *AM);

/// Validation passes for lowering contracts.
std::unique_ptr<Pass> createContractValidationPass(bool failOnError = false);

/// Verification passes at lowering boundaries.
std::unique_ptr<Pass> createVerifyEdtCreatedPass();
std::unique_ptr<Pass> createVerifyEpochCreatedPass();
std::unique_ptr<Pass> createVerifyArtsObjectsOnlyPass();
std::unique_ptr<Pass> createVerifyPreLoweredPass();
std::unique_ptr<Pass> createVerifyEdtLoweredPass();
std::unique_ptr<Pass> createVerifyDbLoweredPass();
std::unique_ptr<Pass> createVerifyEpochLoweredPass();
} // namespace carts::arts
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

// Core ARTS passes (generated from include/carts/passes/Passes.td)
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "carts/passes/Passes.h.inc"

} // namespace mlir

// Per-dialect pass declarations (self-contained headers with dialect deps)
#include "carts/dialect/arts-rt/Transforms/Passes.h"
#include "carts/dialect/sde/Transforms/Passes.h"

// Per-dialect pass registrations (need create functions visible from above)
#define GEN_PASS_REGISTRATION
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"

namespace mlir::carts::sde {
#define GEN_PASS_REGISTRATION
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::sde

#endif /// ARTS_PASSES_PASSES_H
