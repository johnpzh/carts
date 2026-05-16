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
      state/raising/         RaiseToLinalg, RaiseToTensor, RaiseMemrefToTensor
      state/codelet/         ConvertToCodelet, ScalarForwarding
      state/                 PatternAnalysis
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
                             IdRegistry, LocationMetadata, LoweringContractUtils,
                             PartitionPredicates, BlockedAccessUtils,
                             ARTSCostModel, RuntimeConfig)
  arts-rt/                   ARTS-RT dialect — runtime ABI
    IR/                      arts_rt dialect (RtDialect.cpp, RtOps.cpp)
    Conversion/
      ArtsToRt/              DB/EDT/epoch lowering (ARTS → ARTS-RT/runtime shape)
      ArtsRtToLLVM/            Runtime ABI codegen and residual ARTS lowering
      RtToLLVM/              arts_rt → LLVM patterns
    Transforms/              DataPtrHoisting, RuntimeCallOpt, ScalarReplacement,
                             LoopVectorizationHints, AliasScopeGen
    Utils/                   ARTS-RT-specific utilities (LoopInvarianceUtils)
passes/                      Umbrella pass library (MLIRCartsTransforms)
utils/                       CARTS-shared utilities (Debug, LoopUtils,
                             OperationAttributes, PassInstrumentation,
                             RemovalUtils, StencilAttributes, Utils,
                             ValueAnalysis, benchmarks, testing)
```
