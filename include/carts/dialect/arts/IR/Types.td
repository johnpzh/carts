///==========================================================================///
/// File: Types.td
/// This file declares the ARTS dialect types.
///==========================================================================///

#ifndef ARTS_TYPES
#define ARTS_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"

///===----------------------------------------------------------------------===///
/// ARTS Types
///===----------------------------------------------------------------------===///
class Arts_Type<string name, string typeMnemonic> : TypeDef<ArtsDialect, name> {
  let mnemonic = typeMnemonic;
}

///===----------------------------------------------------------------------===///
/// arts.event
///===----------------------------------------------------------------------===///
def EventType : Arts_Type<"Event", "event"> {
  let summary = "represents an event in the ARTS dialect";
  let description = [{
    `arts.event` is a type used to represent events in the ARTS dialect
          .Events are associated with EDT completion or specific dependency
              states.}];
}

#endif // ARTS_TYPES