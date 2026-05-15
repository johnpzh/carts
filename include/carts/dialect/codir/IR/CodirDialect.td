///==========================================================================///
/// File: CodirDialect.td
/// This file defines the Codelet IR dialect.
///==========================================================================///

#ifndef CARTS_CODIR_DIALECT
#define CARTS_CODIR_DIALECT

include "mlir/IR/OpBase.td"

def CartsCodirDialect : Dialect {
  let name = "codir";
  let cppNamespace = "::mlir::carts::codir";

  let summary = "Codelet IR with explicit deps, params, and local views";
  let description = [{
    CODIR is the CARTS codelet boundary dialect. It owns isolated codelets,
    explicit dependency and scalar-parameter ABI shape, yielded values, and
    token-local memref views between SDE planning and ARTS object
    materialization.
  }];

  let useDefaultAttributePrinterParser = 1;
}

class CartsCodir_Op<string mnemonic, list<Trait> traits = []>
    : Op<CartsCodirDialect, mnemonic, traits>;

#endif // CARTS_CODIR_DIALECT
