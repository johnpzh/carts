///==========================================================================///
/// File: DbLayoutStrategy.h
///
/// Layout-aware element pointer computation for datablock lowering.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTSTRATEGY_H
#define ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTSTRATEGY_H

#include "carts/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace arts {

class ArtsCodegen;

struct LayoutInfo {
  DbAllocOp alloc;
  PartitionMode mode = PartitionMode::coarse;
  SmallVector<Value> sizes;
  SmallVector<Value> elementSizes;
  unsigned outerRank = 0;
  unsigned innerRank = 0;
};

/// Build layout information for a datablock handle or acquire pointer.
LayoutInfo buildLayoutInfo(Value source);

/// Compute the LLVM element pointer for a datablock access given base pointer,
/// indices, and layout info. All partition modes currently use linearized
/// addressing.
Value computeDbElementPointer(ArtsCodegen &AC, Location loc, Value base,
                              ArrayRef<Value> indices,
                              const LayoutInfo &layout);

} // namespace arts
} // namespace mlir

#endif /// ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTSTRATEGY_H
