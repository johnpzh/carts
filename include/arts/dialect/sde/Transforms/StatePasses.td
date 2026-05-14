///==========================================================================///
/// State passes — IR form changes (raising, codelet conversion)
///==========================================================================///
#ifndef ARTS_DIALECT_SDE_TRANSFORMS_STATE_PASSES
#define ARTS_DIALECT_SDE_TRANSFORMS_STATE_PASSES

include "mlir/Pass/PassBase.td"

def RaiseMemrefToTensor : Pass<"raise-memref-to-tensor", "mlir::ModuleOp"> {
  let summary = "Lift SDE memref-fallback tasks into cu_codelet tensor form.";
  let description = [{
    See docs/compiler/raise-memref-to-tensor-rfc.md. Converts raisable
    cu_task ops whose bodies read/write shared memrefs into cu_codelet
    ops whose bodies consume !sde.token operands, threading the affected
    memrefs as tensor SSA through the enclosing SDE regions.

    v1 scope: memref<KxT> with small constant K and constant-indexed
    accesses inside task bodies only. Enabled by default: the tensor-path
    lowering fixes the cross-task dep race that the memref-fallback path
    hit on samples like `deps.c`. Gate can be flipped off with
    `--raise-memref-to-tensor=enabled=false` for A/B comparison.
  }];
  let dependentDialects = ["::mlir::tensor::TensorDialect",
                           "::mlir::bufferization::BufferizationDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::affine::AffineDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::arts::sde::ArtsSdeDialect"];
  let options = [
    Option<"enabled", "enabled", "bool", "true",
           "Enable the v1 raise transform. On by default; closes the Gap D "
           "boundary-write race on deps.c and unblocks the tensor path for "
           "the full v1 scope.">
  ];
  let constructor = "mlir::arts::sde::createRaiseMemrefToTensorPass()";
}

def RaiseToLinalg : Pass<"raise-to-linalg", "mlir::ModuleOp"> {
  let summary = "Raise supported sde.su_iterate bodies to transient linalg.generic carriers";
  let description = [{
    Analyzes loop nests inside sde.su_iterate operations, stamps structural
    classification on the source loop, and materializes transient
    linalg.generic carriers for the currently supported subset of loop bodies.
    Unsupported bodies remain in loop form for downstream passes to handle
    via independent IR analysis.

    Recognized patterns:
    - elementwise: all parallel iterators, identity/permutation indexing maps
    - matmul: triple-nested with multiply-accumulate and reduction dimension
    - stencil: parallel iterators with constant-offset indexing maps
    - reduction: has reduction iterators with accumulation pattern

    The pass also stamps an SDE-owned classification on the source loop so
    SDE->ARTS can materialize the matching Core contracts directly.

    Today, only the elementwise/matmul-style subset is materialized as a
    linalg.generic carrier. Stencil/reduction bodies keep their SDE loop shape
    and explicit plan attributes until direct Core materialization.

    ConvertSdeToArts consumes the transient linalg.generic ops, stamps pattern
    contracts on generated Core EDT/epoch/DB operations, and erases the
    carriers so downstream passes continue on Core object IR.
  }];
  let dependentDialects = ["::mlir::memref::MemRefDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::scf::SCFDialect"];
  let constructor = "mlir::arts::sde::createRaiseToLinalgPass()";
}

def RaiseToTensor : Pass<"raise-to-tensor", "mlir::ModuleOp"> {
  let summary = "Full function-scope tensor raising (tensor mem2reg)";
  let description = [{
    Walks each function body top-to-bottom, maintaining a mapping from each
    raised memref.alloca to its current tensor SSA value. Rewrites
    memref.load → tensor.extract, memref.store → tensor.insert, and threads
    tensors through scf.for, cu_region, and su_iterate via iter_args.

    Also converts memref-backed linalg.generic carriers inside su_iterate to
    tensor-backed form for downstream SDE tensor passes.

    After this pass, raisable allocas are eliminated and all data flows as
    pure tensor SSA values through the SDE region hierarchy.
  }];
  let dependentDialects = ["::mlir::bufferization::BufferizationDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::arts::sde::ArtsSdeDialect"];
  let constructor = "mlir::arts::sde::createRaiseToTensorPass()";
}

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
  let constructor = "mlir::arts::sde::createScalarForwardingPass()";
}

def LowerToMemref : Pass<"lower-to-memref", "mlir::ModuleOp"> {
  let summary = "Lower tensor SSA values back to memref.alloca form (inverse of RaiseToTensor)";
  let description = [{
    Walks each function body top-to-bottom, maintaining a mapping from each
    tensor value to its backing memref (alloca or original memref). Rewrites
    tensor.extract → memref.load, tensor.insert → memref.store, and strips
    tensor iter_args from scf.for, cu_region (parallel/master), and
    su_iterate.

    After this pass, raisable allocas are restored and all data flows as
    memref.alloca + memref.load/memref.store. Only <single> cu_region
    iter_args are preserved for ConvertToCodelet.

    Also erases dead linalg.generic carriers and remaining tensor chain ops.
  }];
  let dependentDialects = ["::mlir::bufferization::BufferizationDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::vector::VectorDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::arts::sde::ArtsSdeDialect"];
  let constructor = "mlir::arts::sde::createLowerToMemrefPass()";
}

def ConvertToCodelet : Pass<"convert-to-codelet", "mlir::ModuleOp"> {
  let summary = "Convert tensor iter_args cu_regions to mu_data/mu_token/cu_codelet";
  let description = [{
    Converts `sde.cu_region <single>` operations that carry tensor
    `iter_args` into the canonical SDE codelet form: `sde.mu_data` anchors,
    `sde.mu_token` access handles, and `sde.cu_codelet` compute units.

    This bridges the gap between the tensor-first pipeline (which threads
    tensor values through `cu_region` iter_args) and `ConvertSdeToArts`
    (which already knows how to lower `cu_codelet` + `mu_token` to ARTS EDTs
    with DB acquire/release).

    Current scope: `cu_region <single>` with tensor iter_args only. The
    `<parallel>` case inside `su_iterate` is left to future work.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::bufferization::BufferizationDialect",
                           "::mlir::tensor::TensorDialect"];
  let constructor = "mlir::arts::sde::createConvertToCodeletPass()";
}

def TensorCleanup : Pass<"tensor-cleanup", "mlir::ModuleOp"> {
  let summary = "Clean up tensor artifacts inside cu_codelet bodies.";
  let dependentDialects = ["::mlir::tensor::TensorDialect",
                           "::mlir::linalg::LinalgDialect"];
  let constructor = "mlir::arts::sde::createTensorCleanupPass()";
}

def TokenModeRefine : Pass<"token-mode-refinement", "mlir::ModuleOp"> {
  let summary = "Downgrade access modes on mu_token ops in cu_codelet bodies.";
  let dependentDialects = ["::mlir::tensor::TensorDialect"];
  let constructor = "mlir::arts::sde::createTokenModeRefinementPass()";
}

#endif // ARTS_DIALECT_SDE_TRANSFORMS_STATE_PASSES
