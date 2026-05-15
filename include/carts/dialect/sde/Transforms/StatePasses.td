///==========================================================================///
/// State passes — memref-state cleanup before SDE planning.
///==========================================================================///
#ifndef ARTS_DIALECT_SDE_TRANSFORMS_STATE_PASSES
#define ARTS_DIALECT_SDE_TRANSFORMS_STATE_PASSES

include "mlir/Pass/PassBase.td"

def ScalarForwarding : Pass<"scalar-forwarding", "mlir::ModuleOp"> {
  let summary = "Forward constant-initialized scalar memrefs across region boundaries";
  let description = [{
    Detects rank-0 memref.alloca ops initialized with a constant value and
    replaces loads inside nested regions (omp.parallel, cu_region, etc.) with
    the constant when all stores inside that region write the same value.

    This eliminates Polygeist's continuation-boolean pattern where a memref<i1>
    is used as a structured return guard, turning always-true scf.if conditions
    into foldable constants.
  }];
  let dependentDialects = ["::mlir::memref::MemRefDialect",
                           "::mlir::arith::ArithDialect"];
  let constructor = "mlir::carts::sde::createScalarForwardingPass()";
}

def MemoryUnitMaterialization
    : Pass<"sde-memory-unit-materialization", "mlir::ModuleOp"> {
  let summary = "Materialize SDE shared memrefs as memory units";
  let description = [{
    Rewrites shared memref allocations used by SDE scheduling units into
    `sde.mu_alloc` storage before the SDE/CODIR/ARTS boundary. This pass is
    intentionally SDE-local: it creates target-neutral MU storage declarations
    and leaves conversion to ARTS DB objects to CODIR-to-ARTS lowering.

    Physical layout and owner-slice facts stay on the scheduling unit and are
    consumed at boundary conversion to form acquire windows for supported
    elementwise/matmul owner slices. Stencil, reduction, halo, and other
    complex blocked plans remain on the raw bridge until their SDE-local view
    rewrite exists. The pass does not rediscover DB layouts or perform Core
    index rewriting.
  }];
  let dependentDialects = ["::mlir::carts::sde::CartsSdeDialect",
                           "::mlir::memref::MemRefDialect"];
  let constructor =
      "mlir::carts::sde::createMemoryUnitMaterializationPass()";
}

#endif // ARTS_DIALECT_SDE_TRANSFORMS_STATE_PASSES
