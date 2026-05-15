# Distributed Codepaths

Primary files for distributed ownership and multi-node routing:

- `docs/heuristics/distribution.md`
- `lib/carts/dialect/arts/Transforms/DbDistributedOwnership.cpp`
- `lib/carts/dialect/codir/Conversion/SdeToCodir/SdeToCodir.cpp`
- `lib/carts/dialect/codir/Conversion/CodirToArts/CodirToArts.cpp`
- `lib/carts/dialect/sde/Transforms/effect/distribution/DistributionPlanning.cpp`
- `lib/carts/dialect/arts/Conversion/ArtsToLLVM/ConvertArtsToLLVM.cpp`
- `lib/carts/codegen/Codegen.cpp`

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
