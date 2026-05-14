///==========================================================================///
/// Dep passes — structural transforms (loop interchange, tiling, fusion)
///==========================================================================///
#ifndef ARTS_DIALECT_SDE_TRANSFORMS_DEP_PASSES
#define ARTS_DIALECT_SDE_TRANSFORMS_DEP_PASSES

include "mlir/Pass/PassBase.td"

def LoopInterchange
    : Pass<"loop-interchange", "mlir::ModuleOp"> {
  let summary = "Cost-model-driven loop interchange using SDE linalg carriers";
  let description = [{
    Analyzes the indexing maps and iterator types of transient linalg.generic
    carriers to determine optimal loop ordering for cache efficiency. For
    matmul patterns, moves the reduction dimension outermost so the parallel
    dimension with stride-1 access on the B operand becomes innermost.

    This pass replaces the ARTS-level LoopReordering pass with a more general
    solution that operates on structured SDE carriers before ARTS lowering.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::scf::SCFDialect"];
  let constructor = "mlir::arts::sde::createLoopInterchangePass()";
}

def Tiling : Pass<"tiling", "mlir::ModuleOp"> {
  let summary = "Apply SDE pattern-driven tiling using linalg carriers";
  let description = [{
    Uses SDE-owned pattern/effect facts, abstract logical capacity, and the
    transient tensor-backed linalg carriers to apply real loop tiling before
    SDE->ARTS lowering. Symbolic capacity is represented with
    `sde.resource_query <logical_workers>` and is bound to ARTS only at
    ConvertSdeToArts.
    The current implementation focuses on the first executable subset:
    one-dimensional elementwise loops are strip-mined into logical-capacity
    tiles and the surrounding sde.su_iterate step is rewritten so the
    transformed loop nest survives into later ARTS lowering without exposing
    tensor IR past the SDE boundary.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::bufferization::BufferizationDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::tensor::TensorDialect"];
  let constructor = "mlir::arts::sde::createTilingPass()";
}

def PatternAnalysis
    : Pass<"sde-pattern-analysis", "mlir::ModuleOp"> {
  let summary = "Analyze and stamp approved SDE structured patterns";
  let description = [{
    Uses SDE-owned loop/access analysis together with transient tensor/linalg
    carriers to identify approved structured pattern families on eligible
    `sde.su_iterate` operations. This pass is the first SDE pattern
    planning point after tensor/linalg raising: later SDE transforms consume
    these facts instead of rediscovering pattern legality.

    PatternAnalysis stamps only SDE IR facts. It does not stamp ARTS/Core
    attributes and it does not cross the dialect boundary. ConvertSdeToArts may
    later lower the final SDE plan, but raw pattern-analysis facts remain SDE
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
    later on ARTS IR.
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

def Vectorization
    : Pass<"sde-vectorization", "mlir::ModuleOp"> {
  let summary = "Vectorize static-shape linalg.generic carriers in su_iterate bodies";
  let description = [{
    Vectorizes carrier-authoritative `linalg.generic` ops inside `su_iterate`
    using upstream MLIR's linalg::vectorize(). Replaces eligible carriers with
    `vector.transfer_read/write` + arith-on-vectors. LLVM's backend decomposes
    large vectors into hardware-sized chunks.

    Only vectorizes carriers with all-parallel iterator types, static shapes on
    all operands and loop ranges, and tensor semantics.
  }];
  let dependentDialects = ["::mlir::linalg::LinalgDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::vector::VectorDialect"];
  let constructor = "mlir::arts::sde::createVectorizationPass()";
}

#endif // ARTS_DIALECT_SDE_TRANSFORMS_DEP_PASSES
