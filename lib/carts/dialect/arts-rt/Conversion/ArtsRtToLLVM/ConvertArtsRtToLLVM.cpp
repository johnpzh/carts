///==========================================================================///
/// File: ConvertArtsRtToLLVM.cpp
///
/// This file implements the ARTS-RT-owned pass that lowers runtime-shaped
/// ARTS-RT operations into LLVM dialect runtime calls. A small residual set of
/// ARTS ops is still handled here until the ARTS-to-RT boundary is complete.
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

#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/ConvertArtsRtToLLVMInternal.h"

#include "carts/Dialect.h"
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/CodegenSupport.h"
#include "carts/dialect/arts-rt/Transforms/Passes.h"
namespace mlir::carts::arts_rt {
#define GEN_PASS_DEF_CONVERTARTSRTTOLLVM
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts_rt
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(convert_arts_rt_to_llvm);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts_rt;
using mlir::carts::debugStream;

namespace arts_rt_to_llvm = mlir::carts::arts_rt::convert_arts_rt_to_llvm;

///===----------------------------------------------------------------------===///
/// Pass Implementation
///===----------------------------------------------------------------------===///
namespace {
struct ConvertArtsRtToLLVMPass
    : public mlir::carts::arts_rt::impl::ConvertArtsRtToLLVMBase<
          ConvertArtsRtToLLVMPass> {

  explicit ConvertArtsRtToLLVMPass(bool debug = false,
                                   bool distributedInitPerWorker = false,
                                   const arts::RuntimeConfig *machine = nullptr)
      : debugMode(debug), distributedInitPerWorker(distributedInitPerWorker),
        machine(machine) {}

  void runOnOperation() override;

private:
  ArtsCodegen *AC = nullptr;
  bool debugMode = false;
  bool distributedInitPerWorker = false;
  const arts::RuntimeConfig *machine = nullptr;
};
} // namespace

///===----------------------------------------------------------------------===///
/// Core Pass Implementation
///===----------------------------------------------------------------------===///

void ConvertArtsRtToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  ARTS_INFO_HEADER(ConvertArtsRtToLLVMPass);
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
    arts_rt_to_llvm::populateRuntimePatterns(runtimePatterns, AC);
    if (failed(applyPatternsGreedily(module, std::move(runtimePatterns),
                                     config))) {
      ARTS_ERROR("Failed to apply runtime-to-LLVM conversion patterns");
      return signalPassFailure();
    }
  }

  /// Run 2: arts_rt op patterns (EDT create, deps, state, epochs)
  {
    ARTS_INFO("Running arts_rt op to LLVM patterns");
    RewritePatternSet rtPatterns(context);
    arts_rt_to_llvm::populateArtsRtOpToLLVMPatterns(rtPatterns, AC);
    if (failed(applyPatternsGreedily(module, std::move(rtPatterns), config))) {
      ARTS_ERROR("Failed to apply arts_rt-to-LLVM conversion patterns");
      return signalPassFailure();
    }
  }

  /// Run 3: Db Patterns
  {
    ARTS_INFO("Running db patterns");
    RewritePatternSet dbPatterns(context);
    arts_rt_to_llvm::populateDbPatterns(dbPatterns, AC);
    if (failed(applyPatternsGreedily(module, std::move(dbPatterns), config))) {
      ARTS_ERROR("Failed to apply DbAcquire/DbRelease conversion patterns");
      return signalPassFailure();
    }
  }

  /// Run 4: Other Patterns
  {
    ARTS_INFO("Running other patterns");
    RewritePatternSet otherPatterns(context);
    arts_rt_to_llvm::populateOtherPatterns(otherPatterns, AC);
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
    arts_rt_to_llvm::populateArtsRtOpToLLVMPatterns(cleanupPatterns, AC);
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

  ARTS_INFO_FOOTER(ConvertArtsRtToLLVMPass);
  ARTS_DEBUG_REGION(module.dump(););
}

///===----------------------------------------------------------------------===///
/// Pass Functions
///===----------------------------------------------------------------------===///
namespace mlir {
namespace carts::arts_rt {
std::unique_ptr<Pass> createConvertArtsRtToLLVMPass() {
  return std::make_unique<ConvertArtsRtToLLVMPass>();
}

std::unique_ptr<Pass>
createConvertArtsRtToLLVMPass(bool debug, bool distributedInitPerWorker,
                              const arts::RuntimeConfig *machine) {
  return std::make_unique<ConvertArtsRtToLLVMPass>(
      debug, distributedInitPerWorker, machine);
}
} // namespace carts::arts_rt
} // namespace mlir
