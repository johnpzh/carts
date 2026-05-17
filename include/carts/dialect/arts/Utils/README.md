# ARTS Utils

ARTS utilities own helpers for the abstract ARTS orchestration layer: DBs,
EDTs, epochs, dependency slots, placement, distributed ownership, DB layout
plans, and ARTS-level metadata.

Keep runtime ABI details out of this directory. Packing, runtime calls, pointer
lowering, and LLVM-facing cleanup belong in `arts-rt/Utils` or an ARTS-RT
conversion helper.

Keep source-semantics and codelet-isolation helpers out as well. SDE owns
OpenMP/source planning intent, and CODIR owns codelet-local deps, params, and
token-local views.

Current utility groups:

- `Db*`, `PartitionPredicates`, and `BlockedAccessUtils` describe abstract DB
  layout, access, partitioning, and ownership decisions before runtime ABI
  lowering.
- `EdtUtils`, `ArtsOpUtils`, and `LoweringContractUtils` describe ARTS object
  structure and lowering contracts while the IR is still in the ARTS dialect.
- `LoopInvarianceUtils`, `LoopStructureUtils`, and `ValueAnalysisUtils` provide
  ARTS-aware analysis extensions over the shared `carts/utils` layer.
- `RuntimeConfig`, `RuntimeOpUtils`, `LocationMetadata`, `MetadataEnums`, and
  `ARTSCostModel` carry ARTS-level configuration, query, metadata, and cost
  decisions.
