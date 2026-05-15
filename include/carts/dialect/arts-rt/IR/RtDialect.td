///==========================================================================///
/// File: RtDialect.td
/// This file defines the ARTS runtime dialect.
///==========================================================================///

#ifndef ARTS_RT_DIALECT
#define ARTS_RT_DIALECT

include "mlir/IR/OpBase.td"

def ArtsRtDialect : Dialect {
  let name = "arts_rt";
  let cppNamespace = "::mlir::arts::rt";

  let summary = "Flat ARTS runtime operations";
  let description = [{
    This dialect contains flat runtime-level operations created during
    pre-lowering and consumed during LLVM conversion.
  }];

  let useDefaultAttributePrinterParser = 1;
}

class ArtsRt_Op<string mnemonic, list<Trait> traits = []>
    : Op<ArtsRtDialect, mnemonic, traits>;

#endif // ARTS_RT_DIALECT
