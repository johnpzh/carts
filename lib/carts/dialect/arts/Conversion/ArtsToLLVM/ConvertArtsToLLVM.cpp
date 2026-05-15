///==========================================================================///
/// File: ConvertArtsToLLVM.cpp
///
/// This file implements a pass to convert ARTS dialect operations into
/// LLVM dialect operations.
///
/// Example:
///   Before:
///     %g = arts.edt_create ...
///     arts.record_dep %g, ...
///     arts.wait_on_epoch %e
///
///   After:
///     %g = call @arts_edt_create(...)
///     call @arts_add_dependence(...)
///     call @arts_wait_on_handle(...)
///==========================================================================///

#include "carts/dialect/arts/Conversion/ArtsToLLVM/ConvertArtsToLLVMInternal.h"

#include "carts/Dialect.h"
#include "carts/dialect/arts/Conversion/ArtsToLLVM/CodegenSupport.h"
#define GEN_PASS_DEF_CONVERTARTSTOLLVM
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "carts/dialect/arts/Utils/RuntimeConfig.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(convert_arts_to_llvm);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

using namespace mlir::carts::arts::convert_arts_to_llvm;

///===----------------------------------------------------------------------===///
/// Pass Implementation
///===----------------------------------------------------------------------===///
namespace {
struct ConvertArtsToLLVMPass
    : public impl::ConvertArtsToLLVMBase<ConvertArtsToLLVMPass> {

  explicit ConvertArtsToLLVMPass(bool debug = false,
                                 bool distributedInitPerWorker = false,
                                 const RuntimeConfig *machine = nullptr)
      : debugMode(debug), distributedInitPerWorker(distributedInitPerWorker),
        machine(machine) {}

  void runOnOperation() override;

private:
  ArtsCodegen *AC = nullptr;
  bool debugMode = false;
  bool distributedInitPerWorker = false;
  const RuntimeConfig *machine = nullptr;
};
} // namespace

///===----------------------------------------------------------------------===///
/// Core Pass Implementation
///===----------------------------------------------------------------------===///

void ConvertArtsToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  ARTS_INFO_HEADER(ConvertArtsToLLVMPass);
  ARTS_DEBUG_REGION(module.dump(););

  //// Initialize codegen infrastructure
  auto ownedAC = std::make_unique<ArtsCodegen>(module, debugMode);
  AC = ownedAC.get();
  AC->setDistributedInitInWorkers(distributedInitPerWorker);
  AC->setRuntimeConfig(machine);
  ARTS_DEBUG_TYPE("ArtsCodegen initialized successfully");

  //// Apply patterns with greedy rewriter (four runs)
  GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive);

  /// Run 1: runtime patterns (core arts ops: RuntimeQuery, Barrier, AtomicAdd)
  {
    ARTS_INFO("Running runtime patterns");
    RewritePatternSet runtimePatterns(context);
    convert_arts_to_llvm::populateRuntimePatterns(runtimePatterns, AC);
    if (failed(applyPatternsGreedily(module, std::move(runtimePatterns),
                                     config))) {
      ARTS_ERROR("Failed to apply runtime ARTS to LLVM conversion patterns");
      return signalPassFailure();
    }
  }

  /// Run 2: arts_rt patterns (EDT create, deps, state, epochs)
  {
    ARTS_INFO("Running arts_rt to LLVM patterns");
    RewritePatternSet rtPatterns(context);
    convert_arts_to_llvm::populateRtToLLVMPatterns(rtPatterns, AC);
    if (failed(applyPatternsGreedily(module, std::move(rtPatterns), config))) {
      ARTS_ERROR("Failed to apply arts_rt to LLVM conversion patterns");
      return signalPassFailure();
    }
  }

  /// Run 3: Db Patterns
  {
    ARTS_INFO("Running db patterns");
    RewritePatternSet dbPatterns(context);
    convert_arts_to_llvm::populateDbPatterns(dbPatterns, AC);
    if (failed(applyPatternsGreedily(module, std::move(dbPatterns), config))) {
      ARTS_ERROR("Failed to apply DbAcquire/DbRelease conversion patterns");
      return signalPassFailure();
    }
  }

  /// Run 4: Other Patterns
  {
    ARTS_INFO("Running other patterns");
    RewritePatternSet otherPatterns(context);
    convert_arts_to_llvm::populateOtherPatterns(otherPatterns, AC);
    if (failed(
            applyPatternsGreedily(module, std::move(otherPatterns), config))) {
      ARTS_ERROR("Failed to apply other conversion patterns");
      return signalPassFailure();
    }
  }

  /// Run 5: Cleanup — re-apply arts_rt patterns for ops created in Runs 3-4
  /// (e.g., DbAcquirePattern in Run 3 creates DbGepOp that Run 2 can't see).
  {
    ARTS_INFO("Running arts_rt cleanup patterns");
    RewritePatternSet cleanupPatterns(context);
    convert_arts_to_llvm::populateRtToLLVMPatterns(cleanupPatterns, AC);
    if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns),
                                     config))) {
      ARTS_ERROR("Failed to apply arts_rt cleanup patterns");
      return signalPassFailure();
    }
  }
  //// Initialize runtime
  AC->initRT(AC->getUnknownLoc());

  //// Cleanup
  AC = nullptr;

  ARTS_INFO_FOOTER(ConvertArtsToLLVMPass);
  ARTS_DEBUG_REGION(module.dump(););
}

///===----------------------------------------------------------------------===///
/// Pass Functions
///===----------------------------------------------------------------------===///
namespace mlir {
namespace carts::arts {
std::unique_ptr<Pass> createConvertArtsToLLVMPass() {
  return std::make_unique<ConvertArtsToLLVMPass>();
}

std::unique_ptr<Pass>
createConvertArtsToLLVMPass(bool debug, bool distributedInitPerWorker,
                            const RuntimeConfig *machine) {
  return std::make_unique<ConvertArtsToLLVMPass>(
      debug, distributedInitPerWorker, machine);
}
} // namespace carts::arts
} // namespace mlir
