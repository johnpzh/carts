# Distributed Codepaths

Primary files for distributed ownership and multi-node routing:

- `docs/heuristics/distribution.md`
- `lib/arts/dialect/core/Transforms/DbDistributedOwnership.cpp`
- `lib/arts/dialect/core/Conversion/SdeToArts/SdeToArtsPatterns.cpp`
- `lib/arts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`
- `lib/arts/dialect/core/Conversion/ArtsToLLVM/ConvertArtsToLLVM.cpp`
- `lib/arts/codegen/Codegen.cpp`

High-value grep tokens:

- `distributed`
- `--distributed-db`
- `DbDistributedOwnership`
- `DistributionPlanning`
- `distributed_db_init`
- `distributed_db_init_worker`
- `artsGetTotalNodes`
- `artsGuidGetRank`
- `route = linearIndex`
