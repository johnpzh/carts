///==========================================================================///
/// File: CodirOps.td
/// This file defines Codelet IR operations.
///==========================================================================///

#ifndef CARTS_CODIR_OPS
#define CARTS_CODIR_OPS

include "carts/dialect/codir/IR/CodirAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CartsCodir_CodeletOp
    : CartsCodir_Op<"codelet", [
        AttrSizedOperandSegments,
        IsolatedFromAbove,
        SingleBlock
      ]> {
  let summary = "isolated codelet boundary with explicit deps and params";
  let description = [{
    `codir.codelet` is the target codelet isolation surface. Dependencies and
    scalar parameters are explicit operands; dependency access modes are
    explicit metadata. The body is isolated from above so conversion cannot
    recover hidden captures later in ARTS or ARTS-RT.
  }];

  let arguments = (ins
    OptionalAttr<ArrayAttr>:$dep_modes,
    OptionalAttr<ArrayAttr>:$dep_storage_views,
    OptionalAttr<ArrayAttr>:$dep_owner_dims,
    OptionalAttr<UnitAttr>:$task_depend,
    OptionalAttr<UnitAttr>:$ordered_task_depend,
    OptionalAttr<UnitAttr>:$completion_barrier,
    OptionalAttr<PatternAttr>:$pattern,
    OptionalAttr<ReductionStrategyAttr>:$reduction_strategy,
    UnitAttr:$partial_reduction,
    OptionalAttr<ArrayAttr>:$partial_reduction_dims,
    OptionalAttr<ArrayAttr>:$partial_reduction_owner_dims,
    OptionalAttr<ArrayAttr>:$partial_reduction_dep_result_dim_maps,
    UnitAttr:$partial_reduction_split_required,
    OptionalAttr<ArrayAttr>:$partial_reduction_split_dims,
    OptionalAttr<I64Attr>:$partial_reduction_split_factor,
    OptionalAttr<I64Attr>:$partial_reduction_split_owner_task_count,
    OptionalAttr<I64Attr>:$partial_reduction_split_target_worker_count,
    OptionalAttr<DistributionKindAttr>:$distribution_kind,
    OptionalAttr<IterationTopologyAttr>:$iteration_topology,
    OptionalAttr<RepetitionStructureAttr>:$repetition_structure,
    OptionalAttr<AsyncStrategyAttr>:$async_strategy,
    OptionalAttr<ArrayAttr>:$plan_owner_dims,
    OptionalAttr<ArrayAttr>:$tile_owner_dims,
    OptionalAttr<ArrayAttr>:$tile_shape,
    OptionalAttr<ArrayAttr>:$logical_worker_slice,
    OptionalAttr<ArrayAttr>:$halo_shape,
    OptionalAttr<ArrayAttr>:$access_min_offsets,
    OptionalAttr<ArrayAttr>:$access_max_offsets,
    OptionalAttr<ArrayAttr>:$spatial_dims,
    OptionalAttr<ArrayAttr>:$write_footprint,
    UnitAttr:$in_place_safe,
    UnitAttr:$in_place_shared_state,
    Variadic<AnyType>:$deps,
    Variadic<AnyType>:$params
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    (`deps` `(` $deps^ `:` type($deps) `)`)?
    (`params` `(` $params^ `:` type($params) `)`)?
    attr-dict-with-keyword $body
  }];
}

def CartsCodir_YieldOp
    : CartsCodir_Op<"yield", [
        Pure,
        ReturnLike,
        Terminator,
        HasParent<"CodeletOp">
      ]> {
  let summary = "terminator for CODIR codelet bodies";
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "($results^ `:` type($results))? attr-dict";
}

#endif // CARTS_CODIR_OPS
