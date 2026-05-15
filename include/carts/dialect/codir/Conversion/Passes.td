///==========================================================================///
/// File: Passes.td
///
/// Pass declarations for the SDE -> CODIR -> ARTS conversion boundaries.
///==========================================================================///
#ifndef CARTS_DIALECT_CODIR_CONVERSION_PASSES
#define CARTS_DIALECT_CODIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertSdeToCodir : Pass<"convert-sde-to-codir", "mlir::ModuleOp"> {
  let summary = "Materialize SDE planning intent as CODIR codelet IR";
  let description = [{
    Boundary pass for the SDE-to-CODIR split. It consumes SDE codelet plans and
    materializes isolated CODIR codelets with explicit dependencies, scalar
    parameters, access modes, and token-local memref views.
  }];
  let dependentDialects = [
    "::mlir::carts::sde::CartsSdeDialect",
    "::mlir::carts::codir::CartsCodirDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
  let constructor = "mlir::carts::codir::createConvertSdeToCodirPass()";
}

def ConvertCodirToArts : Pass<"convert-codir-to-arts", "mlir::ModuleOp"> {
  let summary = "Materialize CODIR codelets as abstract ARTS objects";
  let description = [{
    Boundary pass for the CODIR-to-ARTS split. It lowers CODIR codelet deps,
    params, and bodies to abstract ARTS DB acquire and EDT objects.
  }];
  let dependentDialects = [
    "::mlir::carts::codir::CartsCodirDialect",
    "::mlir::arts::ArtsDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
  let constructor = "mlir::carts::codir::createConvertCodirToArtsPass()";
}

#endif // CARTS_DIALECT_CODIR_CONVERSION_PASSES
