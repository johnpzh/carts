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
- `lib/carts/dialect/arts/Utils/LoweringContractUtils.cpp`
- `include/carts/dialect/arts/IR/Ops.td`
- `include/carts/dialect/arts/IR/Attributes.td`
- `include/carts/utils/StencilAttributes.h`

## Runtime / Epoch / Lowering

- `lib/carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/ConvertArtsRtToLLVM.cpp`
- `lib/carts/dialect/arts-rt/Conversion/ArtsToRt/EdtLowering.cpp` (pre-lowering)
- `lib/carts/dialect/arts-rt/Conversion/ArtsToRt/EpochLowering.cpp` (pre-lowering)
- `external/arts/`

## Distributed Ownership / Multi-Node

- `lib/carts/dialect/arts/Transforms/DbDistributedOwnership.cpp`
- `lib/carts/codegen/Codegen.cpp`
- `lib/carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/ConvertArtsRtToLLVM.cpp`

## Analysis / Invalidation / Ordering

- `include/carts/dialect/arts/Analysis/AnalysisDependencies.h`
- `include/carts/dialect/arts/Analysis/AnalysisManager.h`
- `lib/carts/dialect/arts/Analysis/AnalysisManager.cpp`
- `docs/compiler/phase-ordering-semantics.md`
- `tools/compile/Compile.cpp`
- `docs/compiler/pipeline.md`

## Benchmark Harness / Artifacts

- `tools/scripts/triage.py`
- `external/carts-benchmarks/common/carts.mk`
- `docs/benchmarks/`
