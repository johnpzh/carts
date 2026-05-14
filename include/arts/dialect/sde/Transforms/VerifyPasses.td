///==========================================================================///
/// Verify passes — post-lowering verification barriers
///==========================================================================///
#ifndef ARTS_DIALECT_SDE_TRANSFORMS_VERIFY_PASSES
#define ARTS_DIALECT_SDE_TRANSFORMS_VERIFY_PASSES

include "mlir/Pass/PassBase.td"

def VerifySdeLowered : Pass<"verify-sde-lowered", "mlir::ModuleOp"> {
  let summary = "Verify no SDE operations survive past SDE-to-ARTS conversion";
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect"];
  let constructor = "mlir::arts::sde::createVerifySdeLoweredPass()";
}

def VerifySdeCpsPlan : Pass<"verify-sde-cps-plan", "mlir::ModuleOp"> {
  let summary = "Verify SDE CPS stage groups before Core conversion";
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect"];
  let constructor = "mlir::arts::sde::createVerifySdeCpsPlanPass()";
}

#endif // ARTS_DIALECT_SDE_TRANSFORMS_VERIFY_PASSES
