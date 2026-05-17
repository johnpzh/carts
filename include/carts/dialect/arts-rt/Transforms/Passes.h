///==========================================================================///
/// File: Passes.h
///
/// Pass declarations for the ARTS Runtime (arts_rt) dialect.
///
/// This header is self-contained: it includes all dialect headers required
/// by dependentDialects in RT Passes.td so that .h.inc can be safely
/// included without manual per-file dependency management.
///==========================================================================///

#ifndef ARTS_DIALECT_RT_TRANSFORMS_PASSES_H
#define ARTS_DIALECT_RT_TRANSFORMS_PASSES_H

#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Dialect.h"

namespace mlir::carts::arts {

class RuntimeConfig;

} // namespace mlir::carts::arts

namespace mlir::carts::arts_rt {

#define GEN_PASS_DECL
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"

/// Lower ARTS DB handles into runtime ABI shape.
std::unique_ptr<::mlir::Pass> createDbLoweringPass(uint64_t idStride = 1000);
/// Lower outlined ARTS EDT objects into ARTS-RT launch/dependency ops.
std::unique_ptr<::mlir::Pass> createEdtLoweringPass(uint64_t idStride = 1000);
/// Lower ARTS epochs into ARTS-RT epoch ops.
std::unique_ptr<::mlir::Pass> createEpochLoweringPass();
/// Verify no arts.edt ops survive EDT lowering.
std::unique_ptr<::mlir::Pass> createVerifyEdtLoweredPass();
/// Verify no arts.epoch ops survive epoch lowering.
std::unique_ptr<::mlir::Pass> createVerifyEpochLoweredPass();
/// Verify no high-level scheduler ops survive pre-lowering.
std::unique_ptr<::mlir::Pass> createVerifyPreLoweredPass();
/// Lower runtime-shaped ARTS-RT ops to LLVM dialect runtime calls.
std::unique_ptr<::mlir::Pass> createConvertArtsRtToLLVMPass();
std::unique_ptr<::mlir::Pass>
createConvertArtsRtToLLVMPass(bool debug, bool distributedInitPerWorker,
                              const arts::RuntimeConfig *machine);
/// Erase abstract ARTS lowering-contract metadata after ARTS-RT lowering.
std::unique_ptr<::mlir::Pass> createLoweringContractCleanupPass();
/// Hoist dependency/data pointer loads after runtime ABI lowering.
std::unique_ptr<::mlir::Pass> createDataPtrHoistingPass();
/// Rewrite scalar GUID reservation loops to range reservation calls.
std::unique_ptr<::mlir::Pass> createGuidRangeCallOptPass();
/// Hoist and deduplicate pure ARTS runtime calls.
std::unique_ptr<::mlir::Pass> createRuntimeCallOptPass();
/// Generate LLVM alias scope metadata for ARTS data pointers.
std::unique_ptr<::mlir::Pass> createAliasScopeGenPass();
/// Attach LLVM loop vectorization hints to EDT function loops.
std::unique_ptr<::mlir::Pass> createLoopVectorizationHintsPass();
/// Transform memory-based reductions to register-based iter_args.
std::unique_ptr<::mlir::Pass> createScalarReplacementPass();
/// Verify no ARTS DB ops survive DB lowering.
std::unique_ptr<::mlir::Pass> createVerifyDbLoweredPass();
/// Verify no ARTS/ARTS-RT ops survive after runtime-to-LLVM lowering.
std::unique_ptr<::mlir::Pass> createVerifyLoweredPass();

} // namespace mlir::carts::arts_rt

#endif // ARTS_DIALECT_RT_TRANSFORMS_PASSES_H
