///==========================================================================///
/// File: CodirOps.td
/// This file defines Codelet IR operations.
///==========================================================================///

#ifndef CARTS_CODIR_OPS
#define CARTS_CODIR_OPS

include "carts/dialect/codir/IR/CodirDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
