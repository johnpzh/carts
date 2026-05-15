///==========================================================================///
/// File: Passes.td
///
/// Pass declarations for the ARTS Runtime (arts_rt) dialect.
///==========================================================================///
#ifndef ARTS_DIALECT_RT_TRANSFORMS_PASSES
#define ARTS_DIALECT_RT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def DataPtrHoisting
    : Pass<"arts-data-ptr-hoisting", "mlir::ModuleOp"> {
  let summary =
      "Hoist invariant dep/db views and recover loop-local blocked slices";
  let description = [{
    This pass hoists invariant llvm.load operations that extract dependency or
    datablock pointers to before enclosing loops.

    It also recognizes blocked worker loops that already stay inside a single
    owned slice and rewrites them to reuse one hoisted dep view plus the local
    loop IV, instead of rebuilding dep_gep/div/rem selection on every element.

    Without hoisting, these pointer loads happen inside innermost loops
    (O(n^3) loads for 3-nested loops). With hoisting, the pointers are loaded
    once before the loop nest (O(1) loads).

    This is critical for performance because LLVM's LICM cannot hoist these loads
    without TBAA metadata proving they don't alias with data stores.

    Example transformation:
    ```mlir
    scf.for %i = ... {
      scf.for %j = ... {
        % ptr = llvm.load % dep_ptr_addr : !llvm.ptr->!llvm.ptr % val =
                    polygeist.load % ptr[% i, % j] : memref < ? x ? xf64 >
      }
    }
    ```
    becomes:
    ```mlir
    %ptr = llvm.load %dep_ptr_addr : !llvm.ptr -> !llvm.ptr
    scf.for %i = ... {
      scf.for %j = ... {
        % val = polygeist.load % ptr[% i, % j] : memref < ? x ? xf64 >
      }
    }
    ```
  }];
  let dependentDialects = [
    "::mlir::arts::ArtsDialect", "::mlir::LLVM::LLVMDialect",
    "::mlir::scf::SCFDialect"
  ];
  let constructor = "mlir::arts::createDataPtrHoistingPass()";
}

def GuidRangCallOpt
    : Pass<"arts-guid-range-call-opt", "mlir::ModuleOp"> {
  let summary = "Rewrite per-iteration GUID reserve calls into GUID-range calls";
  let description = [{
    Runs after ARTS-to-LLVM lowering and rewrites canonical loops of the form:

      scf.for %i = 0 to N step 1 {
        %g = call @arts_guid_reserve(...)
        ...
      }

    into:

      %range = call @arts_guid_reserve_range(..., N, ...)
      scf.for %i = 0 to N step 1 {
        %g = call @arts_guid_from_index(%range, %i)
        ...
      }

    This pass is conservative and requires lower-bound 0, step 1,
    and route(0) to preserve runtime semantics.

    For dynamic trip counts it emits a guarded rewrite:
    use range reservation only when upperBound > 1 and
    upperBound <= UINT32_MAX,
    otherwise preserve per-iteration reserve behavior.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect", "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect", "::mlir::LLVM::LLVMDialect"
  ];
  let constructor = "mlir::arts::createGuidRangCallOptPass()";
}

def RuntimeCallOpt
    : Pass<"arts-runtime-call-opt", "mlir::ModuleOp"> {
  let summary = "Hoist and deduplicate selected pure ARTS runtime calls";
  let description = [{
    Runs after ARTS-to-LLVM lowering and optimizes only a conservative set of
    runtime topology/identity queries that are safe to treat as loop-invariant
    within a function:

      - arts_get_total_workers
      - arts_get_total_nodes
      - arts_get_current_worker
      - arts_get_current_node

    The pass intentionally ignores stateful runtime calls (guid reserve, DB
    create, EDT/event signaling, etc.) and only rewrites zero-operand query
    calls.

    Transformations:
      1) Hoist loop-invariant calls out of nested `scf.for` loops.
      2) Deduplicate dominated duplicate calls per callee within a function.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect", "::mlir::scf::SCFDialect"
  ];
  let constructor = "mlir::arts::createRuntimeCallOptPass()";
}

def EdtLowering : Pass<"edt-lowering", "mlir::ModuleOp"> {
  let summary = "Lower ARTS EDTs to LLVM";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect",
    "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect",
    "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createEdtLoweringPass()";
}

def EpochLowering : Pass<"epoch-lowering", "mlir::ModuleOp"> {
  let summary = "Lower ARTS epochs and propagate epoch GUIDs to EDTs";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect",
    "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect",
    "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createEpochLoweringPass()";
}

def AliasScopeGen : Pass<"arts-alias-scope-gen", "mlir::ModuleOp"> {
  let summary = "Generate LLVM alias scope metadata for ARTS data pointers";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let constructor = "mlir::arts::createAliasScopeGenPass()";
}

def LoopVectorizationHints
    : Pass<"arts-loop-vectorization-hints", "mlir::ModuleOp"> {
  let summary = "Attach LLVM loop vectorization hints to EDT function loops";
  let description = [{
    This pass attaches LLVM loop optimization hints to loop backedges in EDT
    functions. It uses MLIR's native LoopAnnotationAttr, which automatically
    translates to !llvm.loop metadata during LLVM IR emission.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let constructor = "mlir::arts::createLoopVectorizationHintsPass()";
}

def ScalarReplacement
    : Pass<"arts-scalar-replacement", "mlir::ModuleOp"> {
  let summary = "Transform memory-based reductions to register-based iter_args";
  let description = [{
    This pass transforms memory-based reduction patterns in scf.for loops
    to register-based iter_args patterns to enable LLVM vectorization.
  }];
  let dependentDialects = [
    "::mlir::scf::SCFDialect", "::mlir::memref::MemRefDialect", "::mlir::arith::ArithDialect",
    "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createScalarReplacementPass()";
}

#endif // ARTS_DIALECT_RT_TRANSFORMS_PASSES
