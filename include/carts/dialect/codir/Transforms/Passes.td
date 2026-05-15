///==========================================================================///
/// File: Passes.td
///
/// Pass declarations for the CODIR (Codelet IR) dialect.
///==========================================================================///
#ifndef CARTS_DIALECT_CODIR_TRANSFORMS_PASSES
#define CARTS_DIALECT_CODIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ConvertSdeToCodir : Pass<"convert-sde-to-codir", "mlir::ModuleOp"> {
  let summary = "Materialize SDE planning intent as CODIR codelet IR";
  let description = [{
    Placeholder boundary pass for the target SDE-to-CODIR split. It currently
    preserves IR unchanged so the compiler can expose and test the boundary
    before moving behavior out of the direct SDE-to-ARTS compatibility path.
  }];
  let dependentDialects = [
    "::mlir::carts::sde::CartsSdeDialect",
    "::mlir::carts::codir::CartsCodirDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
  let constructor = "mlir::carts::codir::createConvertSdeToCodirPass()";
}

def VerifyCodir : Pass<"verify-codir", "mlir::ModuleOp"> {
  let summary = "Verify CODIR codelet boundary invariants";
  let description = [{
    Verification barrier for CODIR. It keeps memory dependencies and scalar
    parameters in separate operand groups, requires body block arguments to
    mirror deps then params, and rejects non-scalar yielded values until CODIR
    grows an explicit memory-result ABI. The active MLIR verifier checks
    operation traits such as codelet isolation before and after the pass.
  }];
  let dependentDialects = ["::mlir::carts::codir::CartsCodirDialect"];
  let constructor = "mlir::carts::codir::createVerifyCodirPass()";
}

def ConvertCodirToArts : Pass<"convert-codir-to-arts", "mlir::ModuleOp"> {
  let summary = "Materialize CODIR codelets as abstract ARTS objects";
  let description = [{
    Placeholder boundary pass for the target CODIR-to-ARTS split. It currently
    preserves IR unchanged so the registered pass surface can be tested without
    changing the active direct SDE-to-ARTS compatibility pipeline.
  }];
  let dependentDialects = [
    "::mlir::carts::codir::CartsCodirDialect",
    "::mlir::arts::ArtsDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
  let constructor = "mlir::carts::codir::createConvertCodirToArtsPass()";
}

#endif // CARTS_DIALECT_CODIR_TRANSFORMS_PASSES
