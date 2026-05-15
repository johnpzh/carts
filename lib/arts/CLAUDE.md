# lib/arts/ — Core Compiler Implementation

## Directory Layout

```
dialect/                 MLIR dialects (IREE-style per-dialect structure)
  core/                  THE COMPILER — all passes, analysis, transforms
    Analysis/            All analysis (db, edt, graphs, heuristics, loop)
    Conversion/
      ArtsToLLVM/        ConvertArtsToLLVM pass + core LLVM patterns
      ArtsToRt/          EDT/epoch lowering (core → rt conversion)
    IR/                  Core dialect definition
    Transforms/          All passes + transform libraries (db/, edt/, loop/, kernel/, dep/)
  rt/                    THIN — dialect def + post-lowering optimizations
    IR/                  arts_rt dialect (RtDialect.cpp, RtOps.cpp)
    Conversion/
      RtToLLVM/          arts_rt → LLVM conversion patterns
    Transforms/          DataPtrHoisting, GuidRangCallOpt, RuntimeCallOpt, ScalarReplacement, LoopVectorizationHints, AliasScopeGen
  sde/                   SEMANTIC — OMP conversion, linalg raising, semantic transforms
    Analysis/            StructuredOpAnalysis (loop/access classification)
    Conversion/
      OmpToSde/          ConvertOpenMPToSde (OMP→SDE boundary)
      PolygeistToSde/    Inliner, MemrefNormalization, HandleDeps (pre-SDE Polygeist bridge)
    IR/                  sde dialect (SdeDialect.cpp, SdeOps.cpp)
    Transforms/
      state/raising/     RaiseToLinalg, RaiseToTensor, RaiseMemrefToTensor
      state/codelet/     ConvertToCodelet, ScalarForwarding
      state/             PatternAnalysis
      dep/loop/          LoopInterchange, Tiling, IterationSpaceDecomposition
      dep/fusion/        ElementwiseFusion
      effect/scheduling/ ScheduleRefinement, ChunkOpt, ReductionStrategy
      effect/distribution/ DistributionPlanning, BarrierElimination
    Verify/              VerifySdeLowered
codegen/                 Shared lowering infra (ArtsCodegen)
passes/
  verify/                Verification barrier passes
utils/                   Shared utilities (attrs, debug, loop, value, metadata)
```
