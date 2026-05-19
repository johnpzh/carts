///==========================================================================///
/// File: Passes.td
///
/// Pass declarations for the CODIR (Codelet IR) dialect.
///==========================================================================///
#ifndef CARTS_DIALECT_CODIR_TRANSFORMS_PASSES
#define CARTS_DIALECT_CODIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

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

def CodirCodeletOpt : Pass<"codir-codelet-opt", "mlir::ModuleOp"> {
  let summary = "Clean up isolated CODIR codelet bodies";
  let description = [{
    CODIR-local optimization pass. It removes dead side-effect-free operations
    inside isolated codir.codelet bodies after SDE-to-CODIR token-local rewrites
    and before CODIR verification / ARTS materialization. This keeps codelet
    rematerialization cleanup owned by CODIR instead of relying on later ARTS
    cleanup passes.
  }];
  let dependentDialects = [
    "::mlir::carts::codir::CartsCodirDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect"
  ];
  let constructor = "mlir::carts::codir::createCodirCodeletOptPass()";
}

def StoragePlanning : Pass<"storage-planning", "mlir::ModuleOp"> {
  let summary = "Normalize CODIR dependency storage plans before ARTS lowering";
  let description = [{
    CODIR-owned storage planning pass. It turns high-level dependency storage
    requests into explicit per-dependency contracts that ARTS materialization can
    consume directly: owner-local compute blocks, phase redistribution bridges,
    or host-whole local fallback. This keeps bridge selection and cost policy out
    of the CODIR-to-ARTS conversion.
  }];
  let dependentDialects = [
    "::mlir::carts::codir::CartsCodirDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect"
  ];
  let constructor = "mlir::carts::codir::createStoragePlanningPass()";
}

#endif // CARTS_DIALECT_CODIR_TRANSFORMS_PASSES
