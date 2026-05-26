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

///===----------------------------------------------------------------------===///
/// ArtsDistributedOpInterface
///
/// Implemented by ARTS ops that carry the distribution attribute family:
/// `distribution_kind`, `distribution_pattern`, `distribution_version`, and
/// `depPattern`. Provides typed access to each ODS-generated attr name so
/// passes can avoid raw `AttrNames::Operation::Distribution*`/`DepPatternAttr`
/// lookups when dispatching on a generic `Operation *`.
///===----------------------------------------------------------------------===///

def ArtsDistributedOpInterface : OpInterface<"ArtsDistributedOpInterface"> {
  let description = [{
    Op interface for ARTS ops carrying the distribution attribute family
    (`distribution_kind`, `distribution_pattern`, `distribution_version`,
    `depPattern`).
  }];
  let cppNamespace = "::mlir::carts::arts";

  let methods = [
    InterfaceMethod<[{
        Returns the ODS-generated attr name for `distribution_kind`.
      }],
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getDistributionKindAttrName",
      /*args=*/(ins),
      /*methodBody=*/[{ return $_op.getDistributionKindAttrName(); }]
    >,
    InterfaceMethod<[{
        Returns the ODS-generated attr name for `distribution_pattern`.
      }],
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getDistributionPatternAttrName",
      /*args=*/(ins),
      /*methodBody=*/[{ return $_op.getDistributionPatternAttrName(); }]
    >,
    InterfaceMethod<[{
        Returns the ODS-generated attr name for `distribution_version`.
      }],
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getDistributionVersionAttrName",
      /*args=*/(ins),
      /*methodBody=*/[{ return $_op.getDistributionVersionAttrName(); }]
    >,
    InterfaceMethod<[{
        Returns the ODS-generated attr name for `depPattern`.
      }],
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getDepPatternAttrName",
      /*args=*/(ins),
      /*methodBody=*/[{ return $_op.getDepPatternAttrName(); }]
    >,
  ];
}

#endif // ARTS_OP_INTERFACES
