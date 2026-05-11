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
    converts OpenMP task dependencies into arts.omp_dep operations.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createMemrefNormalizationPass()";
}

def HandleDeps : Pass<"handle-deps", "mlir::ModuleOp"> {
  let summary = "Convert residual OMP dependencies to arts.omp_dep";
  let description = [{
    Converts remaining OpenMP task dependency variables that were not handled
    by MemrefNormalization into arts.omp_dep operations. Handles static
    arrays, token containers, subview chunks, direct allocations, and block
    arguments.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::polygeist::PolygeistDialect"
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

def Concurrency : Pass<"concurrency", "mlir::ModuleOp"> {
  let summary =
      "Build EDT concurrency graph and apply concurrency-driven cleanups";
  let dependentDialects = ["::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createConcurrencyPass(nullptr)";
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

def ParallelEdtLowering : Pass<"parallel-edt-lowering", "mlir::ModuleOp"> {
  let summary = "Lower ARTS parallel EDTs to task graphs";
  let description = [{This pass transforms parallel EDT operations that contain
                          loop constructs into epoch -
                      wrapped task graphs
                          .It must execute before DB lowering to ensure
                              proper dependency management.}];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::scf::SCFDialect",
    "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createParallelEdtLoweringPass()";
}

def ForOpt : Pass<"arts-for-opt", "mlir::ModuleOp"> {
  let summary = "Apply pre-lowering optimization hints to arts.for";
  let description = [{
    Applies optimization hints to arts.for prior to lowering by writing explicit
    `arts.partition_hint` metadata (block mode + block size). This pass can
    coarsen work for both intra-node and inter-node execution while preserving
    distributed worker utilization constraints for internode mode.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect"
  ];
  let constructor = "mlir::arts::createForOptPass(nullptr)";
}

def DistributedHostLoopOutlining
    : Pass<"distributed-host-loop-outlining", "mlir::ModuleOp"> {
  let summary =
      "Outline eligible host producer loops into distributed ARTS EDTs";
  let description = [{
    Converts eligible host-side scf.for loops (outside existing EDT regions)
    into arts.edt<task><internode> containing arts.for. This is used by
    --distributed-db to lift host producer loops into the regular distributed
    distribution/lowering pipeline.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect", "::mlir::memref::MemRefDialect"
  ];
  let constructor =
      "mlir::arts::createDistributedHostLoopOutliningPass(nullptr)";
}

def EdtDistribution : Pass<"edt-distribution", "mlir::ModuleOp"> {
  let summary = "Select and annotate distribution strategy for arts.for loops";
  let description =
      [{Classifies loop access patterns inside parallel EDTs and selects an
            initial distribution strategy,
        recording the decision as attributes consumed by ForLowering.}];
  let dependentDialects = [
    "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect", "::mlir::memref::MemRefDialect"
  ];
  let constructor = "mlir::arts::createEdtDistributionPass(nullptr)";
}

def EdtOrchestrationOpt : Pass<"edt-orchestration-opt", "mlir::ModuleOp"> {
  let summary = "Recognize repeated sibling waves and stamp orchestration contracts";
  let description = [{
    Matches repeated multi-wave timestep loops where several sibling
    arts.edt<parallel> regions inside one enclosing scf.for represent ordered
    waves of the same repeated kernel family. The pass records a canonical
    orchestration contract on the participating EDTs/arts.for ops so
    ForLowering can preserve the group structure instead of lowering each wave
    in isolation.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect", "::mlir::memref::MemRefDialect"
  ];
  let constructor = "mlir::arts::createEdtOrchestrationOptPass()";
}

def ForLowering : Pass<"for-lowering", "mlir::ModuleOp"> {
  let summary = "Lower arts.for inside parallel EDTs to chunked task EDTs";
  let description = [{
    Lowers arts.for operations nested within arts.edt<parallel> by creating a
    symbolic worker-id placeholder, computing chunk bounds, acquiring required
    DBs, and generating a task EDT with a local scf.for over the chunk.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::LLVM::LLVMDialect", "::mlir::func::FuncDialect",
    "::mlir::arts::ArtsDialect", "::mlir::arith::ArithDialect", "::mlir::scf::SCFDialect",
    "::mlir::polygeist::PolygeistDialect"
  ];
  let constructor = "mlir::arts::createForLoweringPass()";
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

def LoopFusion : Pass<"arts-loop-fusion", "mlir::ModuleOp"> {
  let summary = "Fuse independent arts.for loops within parallel EDTs";
  let description = [{
    Identifies consecutive arts.for operations within parallel EDTs that:
    1. Have compatible iteration bounds
    2. Access disjoint sets of memory locations (no data dependencies)
    Uses LoopAnalysis for enhanced dependency checking.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect", "::mlir::func::FuncDialect", "::mlir::arts::ArtsDialect",
    "::mlir::arith::ArithDialect", "::mlir::scf::SCFDialect", "::mlir::omp::OpenMPDialect",
    "::mlir::affine::AffineDialect"
  ];
  let constructor = "mlir::arts::createLoopFusionPass(nullptr)";
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
        3. Worker loop fusion -
            merge compatible worker loops within epochs when they have identical
                bounds and no loop-carried values

    Scheduling optimizations (gated by --arts-epoch-finish-continuation):
        4. Repetition amortization -
            amortize epoch create/wait overhead for repeated epoch loops by
            wrapping EDT bodies with repeat loops
        5. Finish-EDT continuation -
            outline epoch tail code into continuation EDTs that fire via
            ARTS-native finish-EDT signaling instead of blocking waits
        6. CPS loop driver -
            wrap time-step loops containing epochs in a single outer
            epoch+driver EDT, eliminating repeated main-thread blocking
  }];
  let options = [
    Option<"enableEpochNarrowing", "epoch-narrowing", "bool", "true",
           "Enable epoch scope narrowing">,
    Option<"enableEpochFusion", "epoch-fusion", "bool", "true",
           "Enable epoch fusion">,
    Option<"enableLoopFusion", "loop-fusion", "bool", "true",
           "Enable worker loop fusion">,
    Option<"enableAmortization", "amortization", "bool", "false",
           "Enable repetition amortization for repeated epoch loops">,
    Option<"enableContinuation", "continuation", "bool", "false",
           "Enable finish-EDT continuation scheduling">,
    Option<"enableCPSDriver", "cps-driver", "bool", "false",
           "Enable CPS loop driver transform">,
    Option<"enableCPSChain", "cps-chain", "bool", "false",
           "Enable CPS continuation chain transform">
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
           "Emit advisory warnings for ownership proof gaps and CPS pack-schema gaps">
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

def VerifyForLowered : Pass<"verify-for-lowered", "mlir::ModuleOp"> {
  let summary = "Verify all arts.for ops have been lowered";
  let dependentDialects = ["::mlir::arts::ArtsDialect"];
  let constructor = "mlir::arts::createVerifyForLoweredPass()";
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
