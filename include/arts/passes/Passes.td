///==========================================================================///
/// File: Passes.td
///
/// This file defines pass declarations for the ARTS dialect.
///==========================================================================///
#ifndef CARTS_PASSES
#define CARTS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ArtsInliner : Pass<"arts-inliner", "mlir::ModuleOp"> {
  let summary = "Inline functions in the program";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createArtsInlinerPass()";
}

def MemrefNormalization : Pass<"memref-normalization", "mlir::ModuleOp"> {
  let summary = "Raise nested pointer allocations to N-dimensional memrefs";
  let description = [{
    This pass raises nested array-of-pointer allocations to multi-dimensional memrefs
    and rewrites pointer-based accesses to direct memref indexed accesses. It also
    normalizes OpenMP task dependencies into memref values consumed by
    ConvertOpenMPToSde.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect",
    "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createMemrefNormalizationPass()";
}

def HandleDeps : Pass<"handle-deps", "mlir::ModuleOp"> {
  let summary = "Normalize residual OMP dependency operands";
  let description = [{
    Converts remaining OpenMP task dependency variables that were not handled
    by MemrefNormalization into memref values that ConvertOpenMPToSde can
    turn into sde.mu_dep operations. Handles static
    arrays, token containers, subview chunks, direct allocations, and block
    arguments.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect",
    "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createHandleDepsPass()";
}

def DeadCodeElimination : Pass<"arts-dce", "mlir::ModuleOp"> {
  let summary = "Remove dead loads, stores, allocas, and symbols";
  let description = [{
    Performs dead code elimination for memref operations:
    - Removes loads whose results are unused
    - Removes stores to allocas that are never loaded
    - Removes allocas with no remaining uses
    - Runs SymbolDCE for dead functions and globals

    This is particularly useful after openmp-to-arts conversion where OMP
    dependency proxy allocas become unused.
  }];
  let dependentDialects = ["::mlir::memref::MemRefDialect"];
  let constructor = "mlir::arts::createDCEPass()";
}

def DbModeTightening : Pass<"db-mode-tightening", "mlir::ModuleOp"> {
  let summary = "Tighten DB access modes and infer storage types";
  let description =
      [{Tightens the access mode (in / out / inout)
            of DB allocations and acquires based on the actual load /
        store operations performed on them, and infers storage types.}];
  let dependentDialects =
      ["::mlir::memref::MemRefDialect", "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect"];
  let options = [
    Option<"forceInout", "force-inout", "bool", "false",
           "Force all stores to use inout mode (disables mode tightening to out)">
  ];
  let constructor = "mlir::arts::createDbModeTighteningPass(nullptr)";
}

def DbScratchElimination : Pass<"db-scratch-elimination", "mlir::ModuleOp"> {
  let summary = "Eliminate task-private single-element scratch DBs";
  let description = [{
    Replaces task-private single-element scratch DBs with local allocas.
  }];
  let dependentDialects =
      ["::mlir::memref::MemRefDialect", "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createDbScratchEliminationPass()";
}

def DbDistributedOwnership
    : Pass<"distributed-db-ownership", "mlir::ModuleOp"> {
  let summary = "Mark eligible DB allocations for distributed ownership";
  let description = [{
    Selects host-level DB allocations that are safe to distribute across
    nodes and annotates them with the `distributed` attribute for lowering.
  }];
  let dependentDialects =
      ["::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect", "::mlir::memref::MemRefDialect"];
  let constructor = "mlir::arts::createDbDistributedOwnershipPass(nullptr)";
}

def DbTransforms : Pass<"db-transforms", "mlir::ModuleOp"> {
  let summary = "Post-DB transforms (contracts, halo, GUID, cleanup)";
  let description = [{
    Hosts DB-level transformations that run after DB partitioning and
    distributed ownership decisions are known. Current transforms:
    - DT-1: Contract persistence -- persist refined AcquireContractSummary
      back to IR via upsertLoweringContract().
    - DT-2: Stencil halo consolidation
    - DT-4: GUID range allocation
    - DT-6: DB lifetime shortening
    - DT-7: Dead DB elimination

    Placeholder slots:
    - DT-3: ESD annotation
    - DT-5: Acquire coalescing
  }];
  let dependentDialects =
      ["::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createDbTransformsPass(nullptr)";
}

def CreateDbs : Pass<"create-dbs", "mlir::ModuleOp"> {
  let summary = "Identify Dbs in the program";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createCreateDbsPass(nullptr)";
}

def DbLowering : Pass<"db-lowering", "mlir::ModuleOp"> {
  let summary = "Lower ARTS DB handles to opaque pointer form";
  let description =
      [{This pass converts ARTS DataBlocks from typed memrefs to the opaque
            pointer representation required by the runtime
                .It assumes the DBs are already in canonical form.}];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createDbLoweringPass()";
}

def EdtStructuralOpt : Pass<"edt-structural-opt", "mlir::ModuleOp"> {
  let summary = "Optimize EDTs";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createEdtStructuralOptPass(nullptr, true)";
  let options = [Option<"runAnalysis", "run-analysis", "bool",
                        /*default=*/"true", "Enable EDT analysis.">];
}

def EdtAllocaSinking : Pass<"edt-alloca-sinking", "mlir::ModuleOp"> {
  let summary = "Clone external stack allocas into EDT regions";
  let dependentDialects =
      ["::mlir::memref::MemRefDialect", "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createEdtAllocaSinkingPass()";
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
      ["::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createEdtTransformsPass(nullptr)";
}

def CreateEpochs : Pass<"create-epochs", "mlir::ModuleOp"> {
  let summary = "Analyzes the program and creates ARTS epochs";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createCreateEpochsPass()";
}

def ConvertArtsToLLVM : Pass<"convert-arts-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert ARTS Dialect to LLVM Dialect";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createConvertArtsToLLVMPass()";
  let options = [
    Option<"debug", "debug", "bool", /*default=*/"false", "Enable debug mode.">,
  ];
}

def EdtICM : Pass<"arts-edt-icm", "mlir::ModuleOp"> {
  let summary = "Hoist invariant operations from arts edt regions";
  let dependentDialects = ["::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createEdtICMPass()";
}

def EdtPtrRematerialization
    : Pass<"edt-ptr-rematerialization", "mlir::ModuleOp"> {
  let summary = "Rematerialize pointer dependencies in EDTs.";
  let description =
      [{This pass rematerializes pointer dependencies in event -
        driven tasks(EDTs) to optimize memory usage and data movement.}];
  let constructor = "mlir::arts::createEdtPtrRematerializationPass()";
}

def LoweringContractCleanup : Pass<"lowering-contract-cleanup", "mlir::ModuleOp"> {
  let summary = "Erase arts.lowering_contract metadata ops before LLVM lowering";
  let description = [{
    Removes `arts.lowering_contract` operations after all DB/EDT optimization
    passes have consumed them. This keeps semantic contracts available through
    post-db-refinement and late-concurrency-cleanup while ensuring
    pre-lowering/LLVM translation does not see dialect-only metadata ops.
  }];
  let dependentDialects = ["::mlir::arts::ArtsDialect", "::mlir::func::FuncDialect"];
  let constructor = "mlir::arts::createLoweringContractCleanupPass()";
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

    Scheduling optimizations (gated by --arts-epoch-finish-continuation):
        3. Repetition amortization -
            amortize epoch create/wait overhead for repeated epoch loops by
            wrapping EDT bodies with repeat loops
        4. Finish-EDT continuation -
            outline epoch tail code into continuation EDTs that fire via
            ARTS-native finish-EDT signaling instead of blocking waits
  }];
  let options = [
    Option<"enableEpochNarrowing", "epoch-narrowing", "bool", "true",
           "Enable epoch scope narrowing">,
    Option<"enableEpochFusion", "epoch-fusion", "bool", "true",
           "Enable epoch fusion">,
    Option<"enableAmortization", "amortization", "bool", "false",
           "Enable repetition amortization for repeated epoch loops">,
    Option<"enableContinuation", "continuation", "bool", "false",
           "Enable finish-EDT continuation scheduling">
  ];
  let dependentDialects =
      ["::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect", "::mlir::scf::SCFDialect",
       "::mlir::arith::ArithDialect"];
  let constructor = "mlir::arts::createEpochOptPass()";
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
    "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect", "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect"
  ];
  let constructor = "mlir::arts::createHoistingPass()";
}

def BlockLoopStripMining : Pass<"block-stripmine", "mlir::func::FuncOp"> {
  let summary = "Strip-mine block-partitioned loops to remove div/rem overhead";
  let description = [{Matches block-partitioned loop indexing patterns and
                      rewrites them into block-local loops that reuse db_ref.
                      The legacy path handles innermost 1-D div/rem loops,
                      while the neighborhood-aware path can also rewrite
                      enclosing stencil loops that carry block-neighbor
                      indexing state.}];
  let dependentDialects = [
    "::mlir::func::FuncDialect", "::mlir::scf::SCFDialect", "::mlir::arith::ArithDialect",
    "::mlir::arts::ArtsDialect"
  ];
  let constructor = "mlir::arts::createBlockLoopStripMiningPass()";
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
      ["::mlir::memref::MemRefDialect", "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect"];
  let options = [
    Option<"failOnError", "fail-on-error", "bool", "false",
           "Fail compilation if any contract validation errors are found">,
    Option<"warnOnProofGaps", "warn-on-proof-gaps", "bool", "false",
           "Emit advisory warnings for ownership proof gaps">
  ];
  let constructor = "mlir::arts::createContractValidationPass()";
}

///===----------------------------------------------------------------------===///
/// Verification Passes
///===----------------------------------------------------------------------===///

def VerifyEdtCreated : Pass<"verify-edt-created", "mlir::ModuleOp"> {
  let summary = "Verify EDT operations exist after OpenMP conversion";
  let dependentDialects = ["::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createVerifyEdtCreatedPass()";
}

def VerifyEpochCreated : Pass<"verify-epoch-created", "mlir::ModuleOp"> {
  let summary = "Verify epoch operations exist after epoch creation";
  let dependentDialects = ["::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createVerifyEpochCreatedPass()";
}

def VerifyCoreObjectsOnly : Pass<"verify-core-objects-only", "mlir::ModuleOp"> {
  let summary = "Verify the SDE/Core boundary contains only Core ARTS objects";
  let description = [{
    Rejects IR that violates the target dialect layering contract: no residual
    SDE operations may remain at the SDE/Core boundary after direct
    materialization.
  }];
  let dependentDialects = [
    "::mlir::arts::ArtsDialect",
    "::mlir::carts::sde::CartsSdeDialect"
  ];
  let constructor = "mlir::arts::createVerifyCoreObjectsOnlyPass()";
}

def VerifyPreLowered : Pass<"verify-pre-lowered", "mlir::ModuleOp"> {
  let summary = "Verify pre-lowering removed high-level scheduler ops";
  let dependentDialects = ["::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createVerifyPreLoweredPass()";
}

def VerifyLowered : Pass<"verify-lowered", "mlir::ModuleOp"> {
  let summary = "Verify no high-level ARTS ops survive past arts-to-llvm";
  let dependentDialects = ["::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createVerifyLoweredPass()";
}

def VerifyEdtLowered : Pass<"verify-edt-lowered", "mlir::ModuleOp"> {
  let summary = "Verify all EDTs have been lowered";
  let dependentDialects = ["::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createVerifyEdtLoweredPass()";
}

def VerifyDbLowered : Pass<"verify-db-lowered", "mlir::ModuleOp"> {
  let summary = "Verify all DB operations have been lowered";
  let dependentDialects = ["::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createVerifyDbLoweredPass()";
}

def VerifyEpochLowered : Pass<"verify-epoch-lowered", "mlir::ModuleOp"> {
  let summary = "Verify all epoch operations have been lowered";
  let dependentDialects = ["::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createVerifyEpochLoweredPass()";
}

#endif // CARTS_PASSES
