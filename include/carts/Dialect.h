///==========================================================================///
/// File: Dialect.h
/// Defines the Arts dialect and the operations within it.
///==========================================================================///

#ifndef CARTS_DIALECT_H
#define CARTS_DIALECT_H

/// Dialects
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
/// Others
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMapInfo.h"
#include <optional>

/// Arts Dialect
#include "carts/dialect/arts/IR/OpsDialect.h.inc"

/// Arts Dialect Types
#define GET_TYPEDEF_CLASSES
#include "carts/dialect/arts/IR/OpsTypes.h.inc"

/// Arts Dialect Enums
#include "carts/dialect/arts/IR/OpsEnums.h.inc"

namespace mlir {
namespace carts::arts {

struct LoweringContractInfo;

/// Information about a single partition entry.
struct PartitionInfo {
  PartitionMode mode = PartitionMode::coarse;
  SmallVector<Value, 4> indices;            /// For fine_grained: [%i, %j]
  SmallVector<Value, 4> offsets;            /// For chunked/stencil
  SmallVector<Value, 4> sizes;              /// For chunked/stencil
  SmallVector<unsigned, 4> partitionedDims; /// Original dimension indices

  /// Number of partition dimensions
  unsigned dimCount() const {
    if (!indices.empty())
      return indices.size();
    if (!offsets.empty())
      return offsets.size();
    return 0;
  }

  bool isFineGrained() const { return mode == PartitionMode::fine_grained; }
  bool isBlock() const { return mode == PartitionMode::block; }
  bool isStencil() const { return mode == PartitionMode::stencil; }
  bool isCoarse() const { return mode == PartitionMode::coarse; }
};

} // namespace carts::arts
} // namespace mlir

/// Arts Dialect Attributes
#define GET_ATTRDEF_CLASSES
#include "carts/dialect/arts/IR/OpsAttributes.h.inc"

/// Arts Dialect Operations
#define GET_OP_CLASSES
#include "carts/dialect/arts/IR/Ops.h.inc"

#endif // CARTS_DIALECT_H
