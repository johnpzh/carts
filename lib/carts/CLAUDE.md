# lib/carts/ — Compiler Implementation

## Directory Layout

```
dialect/                     MLIR dialects (one subdirectory per CARTS dialect)
  sde/                       SDE dialect — source semantics
    Analysis/                StructuredOpAnalysis (loop/access classification)
    Conversion/
      OmpToSde/              ConvertOpenMPToSde (OMP→SDE boundary)
      PolygeistToSde/        SdeInputInliner, SdeMemrefNormalization, SdeHandleDeps
    IR/                      SdeDialect.cpp, SdeOps.cpp
    Transforms/
      state/                 PatternAnalysis, MemoryUnitMaterialization,
                              ScalarForwarding
      dep/loop/              LoopInterchange, Tiling, IterationSpaceDecomposition
      dep/fusion/            ElementwiseFusion
      effect/scheduling/     ScheduleRefinement, ChunkOpt, ReductionStrategy
      effect/distribution/   DistributionPlanning, BarrierElimination
    Utils/                   SDE-specific utilities (SDECostModel)
    Verify/                  VerifySdeLowered
  codir/                     CODIR dialect — codelet isolation
    IR/                      CodirDialect, CodirOps
    Conversion/              SdeToCodir, CodirToArts
    Transforms/              CodirCodeletOpt, VerifyCodir
    Utils/                   CodeletABIUtils
  arts/                      ARTS dialect — abstract orchestration (DB, EDT, epoch)
    Analysis/                All analysis (db, edt, graphs, heuristics, loop)
    IR/                      ARTS dialect definition
    Transforms/              All ARTS passes (db/, edt/, loop/, epoch/, verify/)
    Utils/                   ARTS-specific utilities (DbUtils, EdtUtils,
                             LoweringContractUtils, PartitionPredicates,
                             BlockedAccessUtils, MetadataEnums, ARTSCostModel,
                             LoopInvarianceUtils, RuntimeConfig,
                             LocationMetadata)
  arts-rt/                   ARTS-RT dialect — runtime ABI
    IR/                      arts_rt dialect (RtDialect.cpp, RtOps.cpp)
    Conversion/
      ArtsToRt/              DB/EDT/epoch lowering (ARTS → ARTS-RT/runtime shape)
      ArtsRtToLLVM/          Runtime ABI codegen and ARTS-RT → LLVM patterns
    Transforms/              DataPtrHoisting, RuntimeCallOpt, ScalarReplacement,
                             LoopVectorizationHints, AliasScopeGen
    Utils/                   ARTS-RT-specific utilities (IdRegistry)
passes/                      Per-dialect pass library wiring
utils/                       CARTS-shared utilities (Debug, LoopUtils,
                             OperationAttributes, PassInstrumentation,
                             RemovalUtils, StencilAttributes, Utils,
                             ValueAnalysis, benchmarks, testing)
```
