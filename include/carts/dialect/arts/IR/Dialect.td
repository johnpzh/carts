///==========================================================================///
/// File: Dialect.td
/// This file defines the ARTS Dialect.
///==========================================================================///
#ifndef ARTS_DIALECT
#define ARTS_DIALECT

include "mlir/IR/OpBase.td"

///===----------------------------------------------------------------------===///
/// ARTS Dialect definition.
///===----------------------------------------------------------------------===///
def ArtsDialect : Dialect {
  let name = "arts";
  let cppNamespace = "::mlir::arts";

  let summary = "Types and operations for the ARTS dialect";
  let description =
      [{ This dialect contains operations for the ARTS Runtime. }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

///===----------------------------------------------------------------------===///
/// ARTS operation definition.
///===----------------------------------------------------------------------===///
class Arts_Op<string mnemonic, list<Trait> traits = []>
    : Op<ArtsDialect, mnemonic, traits>;

#endif // ARTS_DIALECT