///==========================================================================///
/// File: SdeTypes.td
/// This file declares the SDE dialect types.
///==========================================================================///

#ifndef ARTS_SDE_TYPES
#define ARTS_SDE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "SdeDialect.td"

class CartsSde_Type<string name, string typeMnemonic>
    : TypeDef<CartsSdeDialect, name> {
  let mnemonic = typeMnemonic;
}

def CompletionType : CartsSde_Type<"Completion", "completion"> {
  let summary = "typed SDE completion token";
  let description = [{
    `!sde.completion` models SDE-level completion tokens used to express
    task completion and wait surfaces without exposing runtime-specific event
    objects in SDE IR.
  }];
}

def DepType : CartsSde_Type<"Dep", "dep"> {
  let summary = "typed SDE dependency token";
  let description = [{
    `!sde.dep` models an explicit SDE dependency handle produced by
    `sde.mu_dep` and consumed by SDE task-dependence surfaces.
  }];
}

def TokenType : CartsSde_Type<"Token", "token"> {
  let summary = "Handle to a (sub)region of shared memref storage.";
  let description = [{
    `!sde.token` names a subset of an MU-backed memref with a declared access
    mode. It is the canonical SDE access handle for codelet memory operands.

    The `slice_type` parameter records the shape of the addressed region so
    codelet block arguments can be typed directly.
  }];
  let parameters = (ins "::mlir::MemRefType":$slice_type);
  let assemblyFormat = "`<` $slice_type `>`";
}

#endif // ARTS_SDE_TYPES
