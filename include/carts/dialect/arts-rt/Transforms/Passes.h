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

#include "carts/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Dialect.h"

namespace mlir::carts::arts {

#define GEN_PASS_DECL
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"

/// Generate LLVM alias scope metadata for ARTS data pointers.
std::unique_ptr<::mlir::Pass> createAliasScopeGenPass();
/// Attach LLVM loop vectorization hints to EDT function loops.
std::unique_ptr<::mlir::Pass> createLoopVectorizationHintsPass();
/// Transform memory-based reductions to register-based iter_args.
std::unique_ptr<::mlir::Pass> createScalarReplacementPass();

} // namespace mlir::carts::arts

#endif // ARTS_DIALECT_RT_TRANSFORMS_PASSES_H
