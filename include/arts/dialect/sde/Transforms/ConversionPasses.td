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
    that ConvertOpenMPToArts currently generates. It lowers SDE-owned memref
    MU/token/codelet plans directly to ARTS DB/acquire/EDT operations and
    translates explicit SDE structured-loop contracts onto Core operations.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::arts::ArtsDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::arith::ArithDialect"];
  let constructor = "mlir::arts::sde::createConvertSdeToArtsPass()";
}

#endif // ARTS_DIALECT_SDE_TRANSFORMS_CONVERSION_PASSES
