# Debug Codepath Map

High-value files by failure class.

## Shared Compiler Entry Points

- `tools/compile/Compile.cpp`
- `docs/compiler/pipeline.md`
- `docs/heuristics/partitioning.md`
- `docs/heuristics/distribution.md`

## Miscompile / Semantic Drift

- `lib/arts/dialect/sde/Transforms/CollectMetadata.cpp`
- `lib/arts/dialect/core/Conversion/SdeToArts/SdeToArtsPatterns.cpp`
- `lib/arts/dialect/core/Transforms/db/DbTransformsPass.cpp`
- `lib/arts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`
- `lib/arts/utils/LoweringContractUtils.cpp`
- `lib/arts/utils/PatternSemantics.cpp`

## Runtime / Epoch / Lowering

- `lib/arts/dialect/core/Conversion/ArtsToLLVM/ConvertArtsToLLVM.cpp`
- `lib/arts/dialect/core/Conversion/ArtsToRt/EdtLowering.cpp`
- `lib/arts/dialect/core/Conversion/ArtsToRt/EpochLowering.cpp`
- `external/arts/`

## Distributed Ownership / Multi-Node

- `lib/arts/dialect/core/Transforms/DbDistributedOwnership.cpp`
- `lib/arts/codegen/Codegen.cpp`
- `lib/arts/dialect/core/Conversion/ArtsToLLVM/ConvertArtsToLLVM.cpp`

## Analysis / Invalidation / Ordering

- `include/arts/dialect/core/Analysis/AnalysisDependencies.h`
- `include/arts/dialect/core/Analysis/AnalysisManager.h`
- `lib/arts/dialect/core/Analysis/AnalysisManager.cpp`
- `docs/compiler/phase-ordering-semantics.md`
- `docs/plans/phase-ordering-design.md`

## Benchmark Harness / Artifacts

- `tools/scripts/triage.py`
- `external/carts-benchmarks/common/carts.mk`
- `docs/benchmarks/`
