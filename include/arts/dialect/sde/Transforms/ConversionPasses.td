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
                           "::mlir::carts::sde::CartsSdeDialect",
                           "::mlir::arts::ArtsDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::scf::SCFDialect"];
  let constructor = "mlir::carts::sde::createConvertOpenMPToSdePass()";
}

#endif // ARTS_DIALECT_SDE_TRANSFORMS_CONVERSION_PASSES
