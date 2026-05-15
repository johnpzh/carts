# Debug Codepath Map

High-value files by failure class.

## Shared Compiler Entry Points

- `tools/compile/Compile.cpp`
- `docs/compiler/pipeline.md`
- `docs/heuristics/partitioning.md`
- `docs/heuristics/distribution.md`

## Miscompile / Semantic Drift

- `lib/carts/dialect/sde/Transforms/CollectMetadata.cpp`
- `lib/carts/dialect/codir/Conversion/SdeToCodir/SdeToCodir.cpp`
- `lib/carts/dialect/codir/Conversion/CodirToArts/CodirToArts.cpp`
- `lib/carts/dialect/arts/Transforms/db/DbTransformsPass.cpp`
- `lib/carts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`
- `lib/carts/utils/LoweringContractUtils.cpp`
- `lib/carts/utils/PatternSemantics.cpp`

## Runtime / Epoch / Lowering

- `lib/carts/dialect/arts/Conversion/ArtsToLLVM/ConvertArtsToLLVM.cpp`
- `lib/carts/dialect/arts/Conversion/ArtsToRt/EdtLowering.cpp`
- `lib/carts/dialect/arts/Conversion/ArtsToRt/EpochLowering.cpp`
- `external/arts/`

## Distributed Ownership / Multi-Node

- `lib/carts/dialect/arts/Transforms/DbDistributedOwnership.cpp`
- `lib/carts/codegen/Codegen.cpp`
- `lib/carts/dialect/arts/Conversion/ArtsToLLVM/ConvertArtsToLLVM.cpp`

## Analysis / Invalidation / Ordering

- `include/carts/dialect/arts/Analysis/AnalysisDependencies.h`
- `include/carts/dialect/arts/Analysis/AnalysisManager.h`
- `lib/carts/dialect/arts/Analysis/AnalysisManager.cpp`
- `docs/compiler/phase-ordering-semantics.md`
- `docs/plans/phase-ordering-design.md`

## Benchmark Harness / Artifacts

- `tools/scripts/triage.py`
- `external/carts-benchmarks/common/carts.mk`
- `docs/benchmarks/`
