///==========================================================================///
/// File: SdeDialect.td
/// This file defines the Structured Decomposition Environment dialect.
///==========================================================================///

#ifndef ARTS_SDE_DIALECT
#define ARTS_SDE_DIALECT

include "mlir/IR/OpBase.td"

def CartsSdeDialect : Dialect {
  let name = "sde";
  let cppNamespace = "::mlir::carts::sde";

  let summary = "Structured Decomposition Environment -- runtime-agnostic parallel IR";
  let description = [{
    SDE captures parallel computation intent using three orthogonal op
    categories: CU (Compute Units -- what computation happens), SU (Scheduling
    Units -- how/when it runs), and MU (Memory Units -- data access and
    movement).

    SDE is target-agnostic and describes task/data structure without naming a
    concrete runtime. The primary input is OpenMP code via OMP-to-SDE
    conversion.

    SDE state and memory movement are memref-based. Scalar SSA values may be
    captured as task parameters, while MU dependencies and tokens name memref
    storage slices.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class CartsSde_Op<string mnemonic, list<Trait> traits = []>
    : Op<CartsSdeDialect, mnemonic, traits>;

#endif // ARTS_SDE_DIALECT
