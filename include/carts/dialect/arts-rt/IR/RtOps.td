///==========================================================================///
/// File: RtOps.td
/// This file defines the runtime-only operations for the ARTS runtime dialect.
///==========================================================================///

#ifndef ARTS_RT_OPS
#define ARTS_RT_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "carts/dialect/arts-rt/IR/RtDialect.td"

class ArtsRt_I32Enum<string name, string description,
                     list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let cppNamespace = "::mlir::arts::rt";
  let genSpecializedAttr = 0;
}

class ArtsRt_I32EnumAttr<string mnemonic, ArtsRt_I32Enum enumInfo>
    : EnumAttr<ArtsRtDialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def DepAccessMode
    : ArtsRt_I32Enum<"DepAccessMode", "Dependency GUID access mode", [
      I32EnumAttrCase<"direct", 0, "direct">,
      I32EnumAttrCase<"from_depv", 1, "from_depv">
    ]>;
def DepAccessModeAttr : ArtsRt_I32EnumAttr<"dep_access", DepAccessMode>;

def CreateEpochOp
    : ArtsRt_Op<"create_epoch",
                [AttrSizedOperandSegments, MemoryEffects<[MemWrite]>]> {
  let summary = "Create an ARTS epoch and return its GUID";
  let description = [{
    Creates an ARTS epoch and returns its GUID. Unlike `arts.epoch`, this is a
    flat runtime-level descriptor created during pre-lowering.

    Optionally accepts a finish EDT GUID and slot. When provided, the epoch
    signals the given dependency slot of the finish EDT on completion instead
    of requiring an explicit wait operation.
  }];
  let arguments = (ins Optional<I64>:$finishEdtGuid, Optional<I32>:$finishSlot);
  let results = (outs I64:$epochGuid);
  let assemblyFormat = [{
    (`finish` `(` $finishEdtGuid^ `:` type($finishEdtGuid) `,`
                  $finishSlot `:` type($finishSlot) `)`)? attr-dict
    `:` type($epochGuid)
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    bool hasFinishTarget() { return getFinishEdtGuid() != nullptr; }
  }];
}

def WaitOnEpochOp : ArtsRt_Op<"wait_on_epoch", [MemoryEffects<[MemWrite]>]> {
  let summary = "Wait for an ARTS epoch to complete";
  let description = [{
    Blocks until the specified epoch completes.
  }];
  let arguments = (ins I64:$epochGuid);
  let assemblyFormat = "$epochGuid `:` type($epochGuid) attr-dict";
}

def EdtCreateOp : ArtsRt_Op<"edt_create", [MemoryEffects<[MemWrite]>]> {
  let summary = "Outlined EDT launch descriptor";
  let description = [{
    Intermediate op for launching an outlined EDT function with a parameter
    memref, dependency count, route, and optional epoch GUID.
  }];
  let arguments = (ins AnyMemRef:$param_memref, I32:$depCount, I32:$route,
                       Optional<I64>:$epochGuid);
  let results = (outs I64:$guid);
  let assemblyFormat = [{
    `(` $param_memref `:` type($param_memref) `)`
    `depCount` `(` $depCount `)`
    `route` `(` $route `)`
    (`epoch` `(` $epochGuid^ `:` type($epochGuid) `)`)? attr-dict
  }];
  let builders = [
    OpBuilder<(ins "Value":$param_memref, "Value":$depCount)>,
    OpBuilder<(ins "Value":$param_memref, "Value":$depCount,
                   "Value":$route)>,
    OpBuilder<(ins "Value":$param_memref, "Value":$depCount,
                   "Value":$route, "Value":$epochGuid)>
  ];
}

def EdtParamPackOp : ArtsRt_Op<"edt_param_pack", [MemoryEffects<[MemWrite]>]> {
  let summary = "Pack parameters into memref<?xi64>";
  let description = [{
    Packs multiple parameters (cast to i64) into a memref for ARTS paramv.
  }];
  let arguments = (ins Variadic<AnyType>:$params);
  let results = (outs AnyMemRef:$memref);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $params `)` attr-dict `:` type($params) `:` type($memref)";
}

def EdtParamUnpackOp : ArtsRt_Op<"edt_param_unpack", [Pure]> {
  let summary = "Unpack parameters from memref<?xi64>";
  let description = [{
    Unpacks a parameter memref back to the original typed values.
  }];
  let arguments = (ins AnyMemRef:$memref);
  let results = (outs Variadic<AnyType>:$unpacked);
  let hasVerifier = 1;
  let assemblyFormat =
      "$memref `:` type($memref) attr-dict `:` `(` type($unpacked) `)`";
}

def RecordDepOp
    : ArtsRt_Op<"rec_dep",
                [MemoryEffects<[MemWrite]>, AttrSizedOperandSegments]> {
  let summary = "Record in-dependencies for an EDT launch";
  let description = [{
    Records datablock dependencies for the EDT.
  }];
  let arguments = (ins
    I64:$edt_guid,
    Variadic<AnyType>:$datablocks,
    Variadic<I1>:$bounds_valids,
    Variadic<Index>:$byte_offsets,
    Variadic<Index>:$byte_sizes,
    DefaultValuedAttr<DepAccessModeAttr,
                      "::mlir::arts::rt::DepAccessMode::direct">:$access_mode,
    OptionalAttr<DenseI32ArrayAttr>:$acquire_modes,
    OptionalAttr<DenseI32ArrayAttr>:$dep_flags
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $edt_guid `(` $datablocks `:` type($datablocks) `)`
    ( `bounds_valids` `(` $bounds_valids^ `)` )?
    ( `byte_offsets` `(` $byte_offsets^ `)` )?
    ( `byte_sizes` `(` $byte_sizes^ `)` )?
    attr-dict
  }];
}

def IncrementDepOp : ArtsRt_Op<"inc_dep", [MemoryEffects<[MemWrite]>]> {
  let summary = "Increment out-latches for EDT dependencies";
  let description = [{
    Increments latch count for out-dependencies of an EDT launch.
  }];
  let arguments = (ins
    I64:$edt_guid,
    Variadic<AnyType>:$datablocks,
    DefaultValuedAttr<DepAccessModeAttr,
                      "::mlir::arts::rt::DepAccessMode::direct">:$access_mode
  );
  let assemblyFormat =
      "$edt_guid `(` $datablocks `:` type($datablocks) `)` attr-dict";
}

def DepGepOp : ArtsRt_Op<"dep_gep", [Pure, AttrSizedOperandSegments]> {
  let summary = "Extract a dependency entry pointer pair";
  let arguments = (ins AnyType:$dep_struct, Index:$offset,
                       Variadic<AnyType>:$indices, Variadic<Index>:$strides);
  let results = (outs AnyType:$guid, AnyType:$ptr);
  let assemblyFormat = [{
    `(` $dep_struct `)`
    `offset` `[` $offset `:` type($offset) `]`
    ( `indices` `[` $indices^ `:` type($indices) `]` )?
    ( `strides` `[` $strides^ `]` )?
    attr-dict `:` type($dep_struct) `->` type($guid) `,` type($ptr)
  }];
}

def DepDbAcquireOp
    : ArtsRt_Op<"dep_db_acquire", [Pure, AttrSizedOperandSegments]> {
  let summary = "Acquire a datablock from a dependency structure";
  let arguments = (ins
    AnyType:$dep_struct,
    Index:$offset,
    Variadic<AnyType>:$indices,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Optional<I1>:$bounds_valid
  );
  let results = (outs AnyType:$guid, AnyType:$ptr);
  let assemblyFormat = [{
    `(` $dep_struct `)`
    `offset` `[` $offset `:` type($offset) `]`
    ( `indices` `[` $indices^ `:` type($indices) `]` )?
    ( `offsets` `[` $offsets^ `]` )?
    ( `sizes` `[` $sizes^ `]` )?
    ( `bounds_valid` `(` $bounds_valid^ `)` )?
    attr-dict `:` type($dep_struct) `->` type($guid) `,` type($ptr)
  }];
}

def DbGepOp : ArtsRt_Op<"db_gep", [Pure, AttrSizedOperandSegments]> {
  let summary = "Compute an element pointer using byte strides";
  let arguments = (ins AnyType:$base_ptr, Variadic<Index>:$indices,
                       Variadic<Index>:$strides);
  let results = (outs AnyType:$ptr);
  let assemblyFormat = [{
    `(` $base_ptr `:` type($base_ptr) `)`
    `indices` `[` $indices `]`
    `strides` `[` $strides `]`
    attr-dict `:` type($ptr)
  }];
  let builders = [OpBuilder<(ins "Type":$ptr, "Value":$base_ptr,
                                 "SmallVector<Value>":$indices,
                                 "SmallVector<Value>":$strides)>];
}

def StatePackOp : ArtsRt_Op<"state_pack", [MemoryEffects<[MemWrite]>]> {
  let summary = "Pack scalar values into a state channel";
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs MemRefOf<[I64]>:$state);
  let assemblyFormat =
      "`(` $values `)` attr-dict `:` type($values) `->` type($state)";
}

def StateUnpackOp : ArtsRt_Op<"state_unpack", [Pure]> {
  let summary = "Unpack values from a state channel";
  let arguments = (ins MemRefOf<[I64]>:$state);
  let results = (outs Variadic<AnyType>:$values);
  let assemblyFormat =
      "$state `:` type($state) attr-dict `:` `(` type($values) `)`";
}

def DepBindOp : ArtsRt_Op<"dep_bind", [MemoryEffects<[MemWrite]>]> {
  let summary = "Bind a DB GUID to a dependency slot";
  let arguments = (ins I64:$guid, I64:$mode);
  let results = (outs I64:$slot);
  let assemblyFormat = "`(` $guid `,` $mode `)` attr-dict";
  let builders = [OpBuilder<(ins "Value":$guid, "Value":$mode), [{
    $_state.addOperands({guid, mode});
    $_state.addTypes($_builder.getI64Type());
  }]>];
}

def DepForwardOp : ArtsRt_Op<"dep_forward", [Pure]> {
  let summary = "Forward a dependency slot";
  let arguments = (ins I64:$slot);
  let results = (outs I64:$forwarded);
  let assemblyFormat = "$slot attr-dict";
  let builders = [OpBuilder<(ins "Value":$slot), [{
    $_state.addOperands(slot);
    $_state.addTypes($_builder.getI64Type());
  }]>];
}

#endif // ARTS_RT_OPS
