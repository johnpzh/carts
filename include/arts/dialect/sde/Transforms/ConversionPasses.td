///==========================================================================///
/// Conversion passes — dialect boundary crossings (OMP→SDE, SDE→ARTS)
///==========================================================================///
#ifndef ARTS_DIALECT_SDE_TRANSFORMS_CONVERSION_PASSES
#define ARTS_DIALECT_SDE_TRANSFORMS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertOpenMPToSde : Pass<"convert-openmp-to-sde", "mlir::ModuleOp"> {
  let summary = "Convert OpenMP operations to SDE operations";
  let description = [{
    Converts OpenMP ops into SDE ops, preserving information that the
    direct OMP-to-ARTS conversion loses:
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
  let summary = "Convert SDE operations to ARTS operations";
  let description = [{
    Converts SDE dialect ops into ARTS dialect ops, producing the same IR
    that ConvertOpenMPToArts currently generates. When RaiseToLinalg
    materializes transient linalg.generic carriers, this pass derives the
    matching ARTS contract from them, copies semantic contract attributes, and
    erases the carriers so downstream passes continue on loop/memref IR. When
    no transient carrier is authoritative, it prefers any explicit SDE-owned
    semantic contract already stamped on `sde.su_iterate`, then falls
    back to the structural SDE classification attribute.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::arts::ArtsDialect",
                           "::mlir::bufferization::BufferizationDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::tensor::TensorDialect"];
  let constructor = "mlir::arts::sde::createConvertSdeToArtsPass()";
}

#endif // ARTS_DIALECT_SDE_TRANSFORMS_CONVERSION_PASSES
