///==========================================================================///
/// File: CodirAttrs.td
/// This file defines CODIR-owned attributes.
///==========================================================================///

#ifndef CARTS_CODIR_ATTRS
#define CARTS_CODIR_ATTRS

include "carts/dialect/codir/IR/CodirDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class CartsCodir_I32Enum<string name, string description,
                         list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let cppNamespace = "::mlir::carts::codir";
  let genSpecializedAttr = 0;
}

class CartsCodir_I32EnumAttr<string mnemonic, CartsCodir_I32Enum enumInfo>
    : EnumAttr<CartsCodirDialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def CodirAccessMode
    : CartsCodir_I32Enum<"CodirAccessMode", "CODIR dependency access mode", [
      I32EnumAttrCase<"read", 0, "read">,
      I32EnumAttrCase<"write", 1, "write">,
      I32EnumAttrCase<"readwrite", 2, "readwrite">
    ]>;
def AccessModeAttr : CartsCodir_I32EnumAttr<"access_mode", CodirAccessMode>;

def CodirStorageViewKind
    : CartsCodir_I32Enum<"CodirStorageViewKind",
                         "CODIR dependency storage view contract", [
      I32EnumAttrCase<"host_whole", 0, "host_whole">,
      I32EnumAttrCase<"compute_block", 1, "compute_block">,
      I32EnumAttrCase<"replicated_read", 2, "replicated_read">,
      I32EnumAttrCase<"phase_redistributed", 3, "phase_redistributed">
    ]>;
def StorageViewKindAttr
    : CartsCodir_I32EnumAttr<"storage_view", CodirStorageViewKind>;

def CodirPattern
    : CartsCodir_I32Enum<"CodirPattern",
                         "CODIR codelet dependency and schedule family", [
      I32EnumAttrCase<"uniform", 0, "uniform">,
      I32EnumAttrCase<"stencil_tiling_nd", 1, "stencil_tiling_nd">,
      I32EnumAttrCase<"cross_dim_stencil_3d", 2, "cross_dim_stencil_3d">,
      I32EnumAttrCase<"higher_order_stencil", 3, "higher_order_stencil">,
      I32EnumAttrCase<"wavefront_2d", 4, "wavefront_2d">,
      I32EnumAttrCase<"jacobi_alternating_buffers", 5,
                      "jacobi_alternating_buffers">,
      I32EnumAttrCase<"matmul", 6, "matmul">,
      I32EnumAttrCase<"elementwise_pipeline", 7, "elementwise_pipeline">,
      I32EnumAttrCase<"reduction", 8, "reduction">
    ]>;
def PatternAttr : CartsCodir_I32EnumAttr<"pattern", CodirPattern>;

def CodirDistributionKind
    : CartsCodir_I32Enum<"CodirDistributionKind",
                         "CODIR codelet distribution advisory", [
      I32EnumAttrCase<"owner_compute", 0, "owner_compute">,
      I32EnumAttrCase<"blocked", 1, "blocked">,
      I32EnumAttrCase<"cyclic", 2, "cyclic">
    ]>;
def DistributionKindAttr
    : CartsCodir_I32EnumAttr<"distribution_kind", CodirDistributionKind>;

def CodirIterationTopology
    : CartsCodir_I32Enum<"CodirIterationTopology",
                         "CODIR logical iteration topology", [
      I32EnumAttrCase<"owner_strip", 0, "owner_strip">,
      I32EnumAttrCase<"owner_tile", 1, "owner_tile">,
      I32EnumAttrCase<"owner_tile_2d", 2, "owner_tile_2d">
    ]>;
def IterationTopologyAttr
    : CartsCodir_I32EnumAttr<"iteration_topology", CodirIterationTopology>;

def CodirRepetitionStructure
    : CartsCodir_I32Enum<"CodirRepetitionStructure",
                         "CODIR repeated-wave or timestep structure", [
      I32EnumAttrCase<"none", 0, "none">,
      I32EnumAttrCase<"pair_step", 1, "pair_step">,
      I32EnumAttrCase<"k_step", 2, "k_step">,
      I32EnumAttrCase<"full_timestep", 3, "full_timestep">
    ]>;
def RepetitionStructureAttr
    : CartsCodir_I32EnumAttr<"repetition_structure",
                             CodirRepetitionStructure>;

def CodirAsyncStrategy
    : CartsCodir_I32Enum<"CodirAsyncStrategy",
                         "CODIR async task-shape strategy", [
      I32EnumAttrCase<"blocking", 0, "blocking">,
      I32EnumAttrCase<"advance_stage", 1, "advance_stage">,
      I32EnumAttrCase<"cps_chain", 2, "cps_chain">
    ]>;
def AsyncStrategyAttr
    : CartsCodir_I32EnumAttr<"async_strategy", CodirAsyncStrategy>;

#endif // CARTS_CODIR_ATTRS
