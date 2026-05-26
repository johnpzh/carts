///==========================================================================///
/// File: ArtsOpInterfaces.td
/// Declares ARTS dialect op interfaces used to dispatch attribute family
/// queries (partition, distributed-db) without raw-string AttrNames fallbacks.
/// Implementing ops provide the ODS-generated attr-name accessor for each
/// family they own.
///==========================================================================///

#ifndef ARTS_OP_INTERFACES
#define ARTS_OP_INTERFACES

include "mlir/IR/OpBase.td"

///===----------------------------------------------------------------------===///
/// ArtsPartitionedOpInterface
///
/// Implemented by ARTS ops that carry the `partition_mode` attribute.
/// Provides typed access to the ODS-generated attr name so passes can avoid
/// raw `AttrNames::Operation::PartitionMode` lookups.
///===----------------------------------------------------------------------===///

def ArtsPartitionedOpInterface : OpInterface<"ArtsPartitionedOpInterface"> {
  let description = [{
    Op interface for ARTS operations carrying the `partition_mode` attribute.
  }];
  let cppNamespace = "::mlir::carts::arts";

  let methods = [
    InterfaceMethod<[{
        Returns the ODS-generated attr name for `partition_mode`.
      }],
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getPartitionModeAttrName",
      /*args=*/(ins),
      /*methodBody=*/[{ return $_op.getPartitionModeAttrName(); }]
    >,
  ];
}

///===----------------------------------------------------------------------===///
/// ArtsDistributedDbOpInterface
///
/// Implemented by ARTS ops that carry the `distributed` unit attribute on a
/// DB allocation. Provides typed access to the ODS-generated attr name so
/// passes can avoid raw `AttrNames::Operation::Distributed` lookups.
///===----------------------------------------------------------------------===///

def ArtsDistributedDbOpInterface : OpInterface<"ArtsDistributedDbOpInterface"> {
  let description = [{
    Op interface for ARTS DB ops carrying the `distributed` unit attribute.
  }];
  let cppNamespace = "::mlir::carts::arts";

  let methods = [
    InterfaceMethod<[{
        Returns the ODS-generated attr name for `distributed`.
      }],
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getDistributedAttrName",
      /*args=*/(ins),
      /*methodBody=*/[{ return $_op.getDistributedAttrName(); }]
    >,
  ];
}

#endif // ARTS_OP_INTERFACES
