///==========================================================================///
/// File: Passes.td
///
/// This file defines pass declarations for the ARTS dialect.
///==========================================================================///
#ifndef CARTS_PASSES
#define CARTS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def DeadCodeElimination : Pass<"arts-dce", "mlir::ModuleOp"> {
  let summary = "Remove dead ARTS objects and shared helper IR";
  let description = [{
    Performs dead code elimination for ARTS objects and shared helper IR:
    - Removes raw-dead ARTS DB, EDT, acquire, release, and undef operations
    - Runs shared dialect-neutral cleanup for dead memref helper IR
    - Removes dead private symbols

    SDE input cleanup uses the SDE-owned `sde-dead-state-cleanup` pass instead
    of invoking ARTS object cleanup before ARTS objects exist.
  }];
  let dependentDialects = ["::mlir::memref::MemRefDialect",
                           "::mlir::carts::arts::ArtsDialect"];
  let constructor = "mlir::carts::arts::createDCEPass()";
}

def DbModeTightening : Pass<"db-mode-tightening", "mlir::ModuleOp"> {
  let summary = "Tighten DB access modes and infer storage types";
  let description =
      [{Tightens the access mode (in / out / inout)
            of DB allocations and acquires based on the actual load /
        store operations performed on them, and infers storage types.}];
  let dependentDialects =
      ["::mlir::memref::MemRefDialect", "::mlir::func::FuncDialect", "::mlir::carts::arts::ArtsDialect"];
  let options = [
    Option<"forceInout", "force-inout", "bool", "false",
           "Force all stores to use inout mode (disables mode tightening to out)">
  ];
  let constructor = "mlir::carts::arts::createDbModeTighteningPass(nullptr)";
}

def DbScratchElimination : Pass<"db-scratch-elimination", "mlir::ModuleOp"> {
  let summary = "Eliminate task-private single-element scratch DBs";
  let description = [{
    Replaces task-private single-element scratch DBs with local allocas.
  }];
  let dependentDialects =
      ["::mlir::memref::MemRefDialect", "::mlir::func::FuncDialect", "::mlir::carts::arts::ArtsDialect"];
  let constructor = "mlir::carts::arts::createDbScratchEliminationPass()";
}

def DbDistributedOwnership
    : Pass<"distributed-db-ownership", "mlir::ModuleOp"> {
  let summary = "Mark eligible DB allocations for distributed ownership";
  let description = [{
    Selects host-level DB allocations that are safe to distribute across
    nodes and annotates them with the `distributed` attribute for lowering.
  }];
  let dependentDialects =
      ["::mlir::func::FuncDialect", "::mlir::carts::arts::ArtsDialect", "::mlir::memref::MemRefDialect"];
  let constructor = "mlir::carts::arts::createDbDistributedOwnershipPass(nullptr)";
}

def DbTransforms : Pass<"db-transforms", "mlir::ModuleOp"> {
  let summary = "Post-DB transforms (contracts, halo, cleanup)";
  let description = [{
    Hosts DB-level transformations that run after DB partitioning and
    distributed ownership decisions are known. Current transforms:
    - DT-1: Contract persistence -- persist refined AcquireContractSummary
      back to IR via upsertLoweringContract().
    - DT-2: Stencil halo consolidation
    - DT-6: DB lifetime shortening
    - DT-7: Dead DB elimination

    Placeholder slots:
    - DT-3: ESD annotation
    - DT-5: Acquire coalescing
  }];
  let dependentDialects =
      ["::mlir::func::FuncDialect", "::mlir::carts::arts::ArtsDialect"];
  let constructor = "mlir::carts::arts::createDbTransformsPass(nullptr)";
}

def CreateDbs : Pass<"create-dbs", "mlir::ModuleOp"> {
  let summary = "Identify Dbs in the program";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::carts::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::carts::arts::createCreateDbsPass(nullptr)";
}

def EdtStructuralOpt : Pass<"edt-structural-opt", "mlir::ModuleOp"> {
  let summary = "Optimize EDTs";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::carts::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::carts::arts::createEdtStructuralOptPass(nullptr, true)";
  let options = [Option<"runAnalysis", "run-analysis", "bool",
                        /*default=*/"true", "Enable EDT analysis.">];
}

def EdtAllocaSinking : Pass<"edt-alloca-sinking", "mlir::ModuleOp"> {
  let summary = "Clone external stack allocas into EDT regions";
  let dependentDialects =
      ["::mlir::memref::MemRefDialect", "::mlir::func::FuncDialect", "::mlir::carts::arts::ArtsDialect"];
  let constructor = "mlir::carts::arts::createEdtAllocaSinkingPass()";
}

def EdtTransforms : Pass<"edt-transforms", "mlir::ModuleOp"> {
  let summary = "Post-DB EDT transforms (granularity, affinity, critical path)";
  let description = [{
    Hosts EDT-level transformations that run after DB partitioning and
    mode decisions are known. Current placeholder slots:
    - ET-1: Task granularity control
    - ET-2: Data affinity placement
    - ET-6: Critical path analysis
    - EXT-EDT-2: Dead dependency elimination
  }];
  let dependentDialects =
      ["::mlir::func::FuncDialect", "::mlir::carts::arts::ArtsDialect"];
  let constructor = "mlir::carts::arts::createEdtTransformsPass(nullptr)";
}

def CreateEpochs : Pass<"create-epochs", "mlir::ModuleOp"> {
  let summary = "Analyzes the program and creates ARTS epochs";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::carts::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::carts::arts::createCreateEpochsPass()";
}

def EdtPtrRematerialization
    : Pass<"edt-ptr-rematerialization", "mlir::ModuleOp"> {
  let summary = "Rematerialize pointer dependencies in EDTs.";
  let description =
      [{This pass rematerializes pointer dependencies in event -
        driven tasks(EDTs) to optimize memory usage and data movement.}];
  let constructor = "mlir::carts::arts::createEdtPtrRematerializationPass()";
}

def EpochOpt : Pass<"arts-epoch-opt", "mlir::ModuleOp"> {
  let summary = "Unified epoch optimizations (structural + scheduling)";
  let description = [{
    Runs all epoch-related optimizations:

    Structural optimizations:
        1. Epoch scope narrowing -
            split large epochs into smaller independent epochs at positions
                where no EDT after the cut depends on any EDT before the cut
        2. Epoch fusion -
            merge independent consecutive epochs when their DB accesses
                are independent (no shared writes)

    Scheduling optimizations:
        3. Repetition amortization -
            amortize epoch create/wait overhead for repeated epoch loops by
            wrapping EDT bodies with repeat loops
  }];
  let options = [
    Option<"enableEpochNarrowing", "epoch-narrowing", "bool", "true",
           "Enable epoch scope narrowing">,
    Option<"enableEpochFusion", "epoch-fusion", "bool", "true",
           "Enable epoch fusion">,
    Option<"enableAmortization", "amortization", "bool", "false",
           "Enable repetition amortization for repeated epoch loops">
  ];
  let dependentDialects =
      ["::mlir::func::FuncDialect", "::mlir::carts::arts::ArtsDialect", "::mlir::scf::SCFDialect",
       "::mlir::arith::ArithDialect"];
  let constructor = "mlir::carts::arts::createEpochOptPass()";
}

def Hoisting : Pass<"arts-hoisting", "mlir::ModuleOp"> {
  let summary = "Hoist loop-invariant ARTS operations";
  let description = [{
    Unified hoisting for ARTS operations:
    - Hoists read-only db_acquire out of worker loops in epochs when operands
      do not depend on the worker induction variable
    - Hoists db_ref out of inner loops in EDTs when source and indices are
      loop-invariant
  }];
  let options = [
    Option<"enableAcquireHoisting", "acquire", "bool", "true",
           "Hoist db_acquire in epochs">,
    Option<"enableRefHoisting", "ref", "bool", "true", "Hoist db_ref in EDTs">
  ];
  let dependentDialects = [
    "::mlir::func::FuncDialect", "::mlir::carts::arts::ArtsDialect", "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect"
  ];
  let constructor = "mlir::carts::arts::createHoistingPass()";
}

def BlockLoopStripMining : Pass<"block-stripmine", "mlir::func::FuncOp"> {
  let summary = "Strip-mine block-partitioned loops to remove div/rem overhead";
  let description = [{Matches block-partitioned loop indexing patterns and
                      rewrites them into block-local loops that reuse db_ref.
                      The 1-D path handles innermost div/rem loops, while the
                      neighborhood-aware path can also rewrite
                      enclosing stencil loops that carry block-neighbor
                      indexing state.}];
  let dependentDialects = [
    "::mlir::func::FuncDialect", "::mlir::scf::SCFDialect", "::mlir::arith::ArithDialect",
    "::mlir::carts::arts::ArtsDialect"
  ];
  let constructor = "mlir::carts::arts::createBlockLoopStripMiningPass()";
}

def ContractValidation : Pass<"contract-validation", "mlir::ModuleOp"> {
  let summary = "Validate LoweringContractOp integrity across the module";
  let description = [{
    Lightweight verification pass that walks all arts.lowering_contract
    operations and checks structural invariants:
    - Target value exists and is a valid memref
    - supported_block_halo requires a stencil-family dep_pattern
    - min_offsets/max_offsets must have matching rank
    - block_shape rank > 0 if present
    - owner_dims values are non-negative

    Also warns about DbAcquireOps carrying stencil attributes with no
    corresponding LoweringContractOp on the source pointer (orphans).

    Collects per-pattern statistics and optionally fails on errors.
  }];
  let dependentDialects =
      ["::mlir::memref::MemRefDialect", "::mlir::func::FuncDialect", "::mlir::carts::arts::ArtsDialect"];
  let options = [
    Option<"failOnError", "fail-on-error", "bool", "false",
           "Fail compilation if any contract validation errors are found">,
    Option<"warnOnProofGaps", "warn-on-proof-gaps", "bool", "false",
           "Emit advisory warnings for ownership proof gaps">
  ];
  let constructor = "mlir::carts::arts::createContractValidationPass()";
}

///===----------------------------------------------------------------------===///
/// Verification Passes
///===----------------------------------------------------------------------===///

def VerifyEdtCreated : Pass<"verify-edt-created", "mlir::ModuleOp"> {
  let summary = "Verify EDT operations exist after OpenMP conversion";
  let dependentDialects = ["::mlir::carts::arts::ArtsDialect"];
  let constructor = "mlir::carts::arts::createVerifyEdtCreatedPass()";
}

def VerifyEpochCreated : Pass<"verify-epoch-created", "mlir::ModuleOp"> {
  let summary = "Verify epoch operations exist after epoch creation";
  let dependentDialects = ["::mlir::carts::arts::ArtsDialect"];
  let constructor = "mlir::carts::arts::createVerifyEpochCreatedPass()";
}

def VerifyArtsObjectsOnly : Pass<"verify-arts-objects-only", "mlir::ModuleOp"> {
  let summary = "Verify the CODIR-to-ARTS boundary contains only ARTS objects";
  let description = [{
    Rejects IR that violates the target dialect layering contract: no residual
    SDE or OpenMP operations may remain after CODIR-to-ARTS materialization.
  }];
  let dependentDialects = [
    "::mlir::carts::arts::ArtsDialect",
    "::mlir::carts::sde::CartsSdeDialect"
  ];
  let constructor = "mlir::carts::arts::createVerifyArtsObjectsOnlyPass()";
}

#endif // CARTS_PASSES
