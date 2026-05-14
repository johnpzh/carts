///==========================================================================///
/// Effect passes — scheduling decisions (scope, schedule, chunk, reduction,
///                  distribution, barrier elimination)
///==========================================================================///
#ifndef ARTS_DIALECT_SDE_TRANSFORMS_EFFECT_PASSES
#define ARTS_DIALECT_SDE_TRANSFORMS_EFFECT_PASSES

include "mlir/Pass/PassBase.td"

def ScopeSelection : Pass<"scope-selection", "mlir::ModuleOp"> {
  let summary = "Select SDE concurrency scope from the active cost model";
  let description = [{
    Uses the SDE-owned cost model to choose the concurrency scope for
    sde.cu_region <parallel> operations. The current implementation maps the
    active logical topology directly: local-only configurations become
    `scope(<local>)` and multi-placement configurations become
    `scope(<distributed>)`.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect"];
  let constructor = "mlir::arts::sde::createScopeSelectionPass()";
}

def ScheduleRefinement
    : Pass<"schedule-refinement", "mlir::ModuleOp"> {
  let summary = "Refine abstract SDE schedule hints into concrete schedules";
  let description = [{
    Uses the SDE-owned cost model to refine eligible `sde.su_iterate`
    schedule hints before later SDE optimizations run. The current
    implementation is deliberately narrow: it only handles one-dimensional
    loops with constant trip counts, only rewrites `auto` / `runtime`
    schedules, and chooses the lowest-overhead concrete policy among
    `static`, `dynamic`, and `guided`.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect"];
  let constructor = "mlir::arts::sde::createScheduleRefinementPass()";
}

def ChunkOpt : Pass<"chunk-opt", "mlir::ModuleOp"> {
  let summary = "Fill missing SDE chunk sizes from logical capacity";
  let description = [{
    Uses SDE-owned logical capacity to synthesize chunk sizes for eligible
    sde.su_iterate operations when the source program left them unspecified.
    Symbolic trip-count cases use `sde.resource_query <logical_workers>` so
    the boundary layer, not SDE, binds the expression to a concrete target
    mechanism. The current implementation handles one-dimensional
    dynamic/guided loops and preserves explicit source chunk sizes.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::arith::ArithDialect"];
  let constructor = "mlir::arts::sde::createChunkOptPass()";
}

def ReductionStrategy : Pass<"reduction-strategy", "mlir::ModuleOp"> {
  let summary = "Select an SDE reduction strategy from the active cost model";
  let description = [{
    Uses the SDE-owned cost model to annotate eligible reduction-bearing
    sde.su_iterate operations with a reduction strategy recommendation.
    The current implementation is deliberately narrow: it only handles
    single-reduction loops with preserved reduction kinds, chooses between
    `atomic` and `tree`, and leaves lowering behavior unchanged.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect"];
  let constructor = "mlir::arts::sde::createReductionStrategyPass()";
}

def DistributionPlanning
    : Pass<"distribution-planning", "mlir::ModuleOp"> {
  let summary = "Derive SDE-owned distribution intent from structured SDE loops";
  let description = [{
    Uses SDE-owned structured classification, concurrency scope, and abstract
    logical-capacity heuristics to wrap eligible `sde.su_iterate` operations in
    `sde.su_distribute`. The current implementation is deliberately
    narrow: it only authors a blocked advisory for local elementwise loops and
    an owner-compute advisory for distributed stencil loops with enough work to
    amortize distribution.

    The target boundary is responsible for materializing this advisory into
    downstream distribution metadata.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect"];
  let constructor = "mlir::arts::sde::createDistributionPlanningPass()";
}

def BarrierElimination
    : Pass<"barrier-elimination", "mlir::ModuleOp"> {
  let summary = "Eliminate redundant SDE barriers between independent scheduling units";
  let description = [{
    Analyzes read and write sets of consecutive sde.su_iterate operations
    separated by sde.su_barrier ops. When the write-set of the predecessor
    loop is provably disjoint from the read-set of the successor loop, the
    barrier is marked for elimination. The target boundary consumes that mark
    and omits the corresponding downstream synchronization object.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::memref::MemRefDialect"];
  let constructor = "mlir::arts::sde::createBarrierEliminationPass()";
}

def CpsPlanning : Pass<"cps-planning", "mlir::ModuleOp"> {
  let summary = "Mark SDE-owned CPS timestep candidates before Core conversion";
  let description = [{
    Groups timestep/wave candidates already recognized by SDE barrier analysis
    under SDE-owned CPS candidate attributes, and emits SDE completion tokens
    for barrier-delimited stage boundaries. The pass does not set
    `async_strategy = cps_chain`; that final strategy remains reserved until
    SDE has made every scalar, data, and control carry explicit. Core must only
    materialize final CPS plans from that explicit SDE graph.
  }];
  let dependentDialects = ["::mlir::arts::sde::ArtsSdeDialect",
                           "::mlir::scf::SCFDialect"];
  let constructor = "mlir::arts::sde::createCpsPlanningPass()";
}

#endif // ARTS_DIALECT_SDE_TRANSFORMS_EFFECT_PASSES
