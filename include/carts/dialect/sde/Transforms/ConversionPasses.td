///==========================================================================///
/// Conversion passes - dialect boundary crossings (OMP to SDE, SDE boundary)
///==========================================================================///
#ifndef ARTS_DIALECT_SDE_TRANSFORMS_CONVERSION_PASSES
#define ARTS_DIALECT_SDE_TRANSFORMS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def SdeInputInliner : Pass<"sde-input-inliner", "mlir::ModuleOp"> {
  let summary = "Inline helper calls needed for SDE source normalization";
  let description = [{
    Inlines helper functions only when exposing the callee body is necessary
    for SDE source normalization: OpenMP regions, nested memref wrappers, and
    memref helper calls whose boundary would otherwise freeze a stale storage
    layout before SDE planning.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::func::FuncDialect",
    "::mlir::carts::sde::CartsSdeDialect",
    "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::carts::sde::createSdeInputInlinerPass()";
}

def SdeMemrefNormalization
    : Pass<"sde-memref-normalization", "mlir::ModuleOp"> {
  let summary = "Raise pointer-shaped source storage to SDE memref views";
  let description = [{
    Raises nested array-of-pointer allocations to multi-dimensional memrefs,
    rewrites pointer-based accesses to direct memref indexed accesses, and
    normalizes OpenMP task dependencies into memref values consumed by
    ConvertOpenMPToSde.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::func::FuncDialect",
    "::mlir::carts::sde::CartsSdeDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::carts::sde::createSdeMemrefNormalizationPass()";
}

def SdeHandleDeps : Pass<"sde-handle-deps", "mlir::ModuleOp"> {
  let summary = "Normalize residual OpenMP dependency operands for SDE";
  let description = [{
    Converts remaining OpenMP task dependency variables that were not handled
    by SdeMemrefNormalization into memref values that ConvertOpenMPToSde can
    turn into sde.mu_dep operations. Handles static arrays, token containers,
    subview chunks, direct allocations, and block arguments.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect",
    "::mlir::func::FuncDialect",
    "::mlir::carts::sde::CartsSdeDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::carts::sde::createSdeHandleDepsPass()";
}

def ConvertOpenMPToSde : Pass<"convert-openmp-to-sde", "mlir::ModuleOp"> {
  let summary = "Convert OpenMP operations to SDE operations";
  let description = [{
    Converts OpenMP ops into SDE ops, preserving information that a direct
    target conversion would otherwise lose:
    - Reduction combiner kind + identity
    - Nowait semantics
    - Schedule + chunk size
    - Task completion tokens
  }];
  let dependentDialects = ["::mlir::memref::MemRefDialect",
                           "::mlir::omp::OpenMPDialect",
                           "::mlir::carts::sde::CartsSdeDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::scf::SCFDialect"];
  let constructor = "mlir::carts::sde::createConvertOpenMPToSdePass()";
}

#endif // ARTS_DIALECT_SDE_TRANSFORMS_CONVERSION_PASSES
