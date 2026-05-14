///==========================================================================///
/// Dep passes — structural transforms (loop interchange, tiling, fusion)
///==========================================================================///
#ifndef ARTS_DIALECT_SDE_TRANSFORMS_DEP_PASSES
#define ARTS_DIALECT_SDE_TRANSFORMS_DEP_PASSES

include "mlir/Pass/PassBase.td"

def LoopInterchange
    : Pass<"loop-interchange", "mlir::ModuleOp"> {
  let summary = "Cost-model-driven loop interchange using SDE memref access analysis";
  let description = [{
    Analyzes SDE loop/access maps to determine optimal loop ordering for cache
    efficiency. For matmul patterns, moves the reduction dimension outermost
    so the parallel dimension with stride-1 access on the B operand becomes
    innermost.

    This pass replaces target-level loop reordering with a more general
    solution that operates on structured SDE memref loops before boundary
    conversion.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::scf::SCFDialect"];
  let constructor = "mlir::arts::sde::createLoopInterchangePass()";
}

def Tiling : Pass<"tiling", "mlir::ModuleOp"> {
  let summary = "Apply SDE pattern-driven tiling using memref loop/access facts";
  let description = [{
    Uses SDE-owned pattern/effect facts, abstract logical capacity, and the
    memref loop/access facts to apply real loop tiling before boundary
    conversion. Symbolic capacity is represented with
    `sde.resource_query <logical_workers>` and remains target-neutral in SDE.
    The current implementation focuses on the first executable subset:
    one-dimensional elementwise loops are strip-mined into logical-capacity
    tiles and the surrounding sde.su_iterate step is rewritten so the
    transformed loop nest survives into later lowering.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect"];
  let constructor = "mlir::arts::sde::createTilingPass()";
}

def PatternAnalysis
    : Pass<"sde-pattern-analysis", "mlir::ModuleOp"> {
  let summary = "Analyze and stamp approved SDE structured patterns";
  let description = [{
    Uses SDE-owned loop/access analysis to identify approved structured pattern
    families on eligible `sde.su_iterate` operations. This pass is the first
    SDE pattern planning point after OpenMP-to-SDE conversion: later SDE
    transforms consume these facts instead of rediscovering pattern legality.

    PatternAnalysis stamps only SDE IR facts. It does not stamp target-specific
    attributes and it does not cross the dialect boundary. Later conversion may
    lower the final SDE plan, but raw pattern-analysis facts remain SDE
    ownership.

    The current implementation stamps:
    - `classification(<...>)` from the current SDE structured analysis
    - approved `pattern` values for uniform, stencil, wavefront, matmul,
      reduction, and elementwise-pipeline families
    - generic access/owner/footprint summaries for recognized ND neighborhood
      loops
    - SDE execution hints for vectorization/unrolling/interleaving
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect"];
  let constructor = "mlir::arts::sde::createPatternAnalysisPass()";
}

def ElementwiseFusion
    : Pass<"elementwise-fusion", "mlir::ModuleOp"> {
  let summary = "Fuse consecutive SDE elementwise scheduling units";
  let description = [{
    Fuses consecutive sibling `sde.su_iterate` operations that already
    carry an SDE-owned `classification(<elementwise>)`, share the same
    iteration space and scheduling policy, and write disjoint destinations.
    The fused scheduling unit is reclassified as
    `classification(<elementwise_pipeline>)`, keeping elementwise pipeline
    ownership on the SDE side of the boundary instead of inventing that family
    later in target IR.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect"];
  let constructor = "mlir::arts::sde::createElementwiseFusionPass()";
}

def IterationSpaceDecomposition
    : Pass<"iteration-space-decomposition", "mlir::ModuleOp"> {
  let summary = "Decompose scheduling-unit iteration spaces into interior and boundary regions";
  let description = [{
    Splits scheduling-unit inner loops that mix interior and boundary
    computations behind control flow into separate loops: a branch-free
    interior loop suitable for tiling and distribution, plus dedicated
    boundary loops that preserve the original edge behavior.

    The current implementation handles the neighborhood boundary pattern
    where row/column boundary checks guard stencil-like inner loops.
    Future decompositions (e.g., triangular peeling, remainder splitting)
    can be added as additional matchers within this pass.
  }];
  let dependentDialects = ["::mlir::func::FuncDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::memref::MemRefDialect"];
  let constructor = "mlir::arts::sde::createIterationSpaceDecompositionPass()";
}

#endif // ARTS_DIALECT_SDE_TRANSFORMS_DEP_PASSES
