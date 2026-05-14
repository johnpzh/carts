///==========================================================================///
/// Conversion passes - dialect boundary crossings (OMP to SDE, SDE boundary)
///==========================================================================///
#ifndef ARTS_DIALECT_SDE_TRANSFORMS_CONVERSION_PASSES
#define ARTS_DIALECT_SDE_TRANSFORMS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

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
                           "::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::arts::ArtsDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::scf::SCFDialect"];
  let constructor = "mlir::arts::sde::createConvertOpenMPToSdePass()";
}

def ConvertSdeToArts : Pass<"convert-sde-to-arts", "mlir::ModuleOp"> {
  let summary = "Materialize SDE operations at the target boundary";
  let description = [{
    Materializes SDE dialect ops into downstream Core operations. SDE-owned
    memref MU/token/codelet plans and explicit structured-loop contracts are
    consumed at this boundary; target-specific storage, access, task, and
    synchronization objects are created outside the SDE dialect contract.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::arts::ArtsDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::arith::ArithDialect"];
  let constructor = "mlir::arts::sde::createConvertSdeToArtsPass()";
}

#endif // ARTS_DIALECT_SDE_TRANSFORMS_CONVERSION_PASSES
