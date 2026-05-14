///==========================================================================///
/// File: DbIndexerBase.h
///
/// Abstract base class for raw-memref DB index localizers.
///
/// This file defines the common interface and shared utilities for datablock
/// index localizers used by CreateDbs while a task body still contains raw
/// memref accesses.
///
/// These classes do not choose partitioning policy. They consume an explicit
/// SDE-authored/Core-materialized DB layout and rewrite uses mechanically.
///
/// The base class provides shared template-method implementations for:
/// - transformOps(): iterates operations, delegates to virtual hooks
/// - transformDbRefUsers(): rewrites DbRefOp users with localized indices
/// - transformUsesInParentRegion(): rewrites allocation uses in parent region
///
/// Derived classes override virtual hooks for mode-specific behavior:
/// - localize()/localizeLinearized(): index math
/// - detectLinearizedStride(): stride detection strategy
/// - handleSubIndexOp(): polygeist::SubIndexOp rewriting
/// - localizeForDbRefUser(): DbRefOp user localization strategy
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_DB_DBINDEXERBASE_H
#define ARTS_DIALECT_CORE_TRANSFORMS_DB_DBINDEXERBASE_H

#include "arts/Dialect.h"
#include "arts/utils/DbUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace arts {

/// Shared result structure for index localization.
/// Used by all indexer modes to return the transformed indices.
struct LocalizedIndices {
  SmallVector<Value> dbRefIndices;
  SmallVector<Value> memrefIndices;
};

/// Shared helper - extract indices from load/store/ref operations.
inline ValueRange getIndicesFromOp(Operation *op) {
  if (auto access = DbUtils::getMemoryAccessInfo(op))
    return access->indices;
  if (auto ref = dyn_cast<DbRefOp>(op))
    return ref.getIndices();
  return {};
}

/// Forward declaration
class ArtsCodegen;

/// Abstract base class for datablock index localizers.
///
/// Each derived class implements mode-specific localization:
/// - DbBlockIndexer: div/mod localization for block allocation
/// - DbElementWiseIndexer: direct element coordinate mapping
///
/// All indexers hold PartitionInfo as the canonical source of partition data:
/// - DbElementWiseIndexer uses partitionInfo.indices (element COORDINATES)
/// - DbBlockIndexer uses partitionInfo.offsets/sizes (range start/size)
///
/// Shared template methods (transformOps, transformDbRefUsers,
/// transformUsesInParentRegion) implement the common iteration/rewriting
/// pattern and delegate mode-specific logic to virtual hooks.
class DbIndexerBase {
protected:
  /// Canonical source of partition semantics. Indexers access the fields
  /// appropriate for their mode:
  /// - fine_grained: use partitionInfo.indices directly as db_ref indices
  /// - block/stencil: use partitionInfo.offsets/sizes for div/mod formulas
  PartitionInfo partitionInfo;
  unsigned outerRank, innerRank;

public:
  /// Constructor with PartitionInfo - the canonical way to create indexers.
  DbIndexerBase(const PartitionInfo &info, unsigned outerRank,
                unsigned innerRank)
      : partitionInfo(info), outerRank(outerRank), innerRank(innerRank) {}

  /// Accessors for partition info
  const PartitionInfo &getPartitionInfo() const { return partitionInfo; }
  PartitionMode getMode() const { return partitionInfo.mode; }

  virtual ~DbIndexerBase() = default;

  //===--------------------------------------------------------------------===//
  /// Pure virtual: mode-specific index localization
  //===--------------------------------------------------------------------===//

  /// Transform global multi-dimensional indices to local coordinates.
  virtual LocalizedIndices localize(ArrayRef<Value> globalIndices,
                                    OpBuilder &builder, Location loc) = 0;

  /// Transform a linearized global index for flattened 1D memrefs.
  virtual LocalizedIndices localizeLinearized(Value globalLinearIndex,
                                              Value stride, OpBuilder &builder,
                                              Location loc) = 0;

  //===--------------------------------------------------------------------===//
  /// Virtual hooks for template method customization
  //===--------------------------------------------------------------------===//

  /// Detect linearized stride for a set of indices and element type.
  /// Block mode uses MemRefType shape; ElementWise uses oldElementSizes.
  /// Returns {isLinearized, stride}. Default implementation checks
  /// MemRefType shape (Block-style).
  virtual std::pair<bool, Value> detectLinearizedStride(ValueRange indices,
                                                        Type elementType,
                                                        OpBuilder &builder,
                                                        Location loc);

  /// Handle a polygeist::SubIndexOp during transformOps.
  /// Returns true if handled (op was rewritten); false to skip.
  /// Default implementation handles the common pattern shared by Block
  /// and ElementWise modes.
  virtual bool handleSubIndexOp(Operation *op, Value dbPtr, Type elementType,
                                ArtsCodegen &AC,
                                llvm::SetVector<Operation *> &opsToRemove);

  /// Handle an access with empty indices during transformOps.
  /// Returns true if handled; false to skip. Default returns false.
  virtual bool handleEmptyIndices(Operation *op, Value dbPtr, Type elementType,
                                  ArtsCodegen &AC,
                                  llvm::SetVector<Operation *> &opsToRemove);

  /// Localize indices for a DbRefOp user during transformDbRefUsers.
  /// Default implementation uses localize()/localizeLinearized().
  /// ElementWise overrides to use localizeForFineGrained().
  virtual LocalizedIndices localizeForDbRefUser(ValueRange elementIndices,
                                                Type newElementType,
                                                OpBuilder &builder,
                                                Location loc);

  //===--------------------------------------------------------------------===//
  /// Shared implementations (template methods)
  //===--------------------------------------------------------------------===//

  /// Transform a DbRefOp and its load/store users to use local coordinates.
  /// Shared implementation: iterates users, detects linearized access,
  /// calls localizeForDbRefUser(), creates new DbRefOp + load/store.
  virtual void transformDbRefUsers(DbRefOp ref, Value blockArg,
                                   Type newElementType, OpBuilder &builder,
                                   llvm::SetVector<Operation *> &opsToRemove);

  /// Transform a list of operations using mode-specific localization.
  /// Shared implementation: iterates ops, handles DbRefOp/SubIndex/load/store,
  /// detects linearized access, calls localize()/localizeLinearized().
  void transformOps(ArrayRef<Operation *> ops, Value dbPtr, Type elementType,
                    ArtsCodegen &AC, llvm::SetVector<Operation *> &opsToRemove);

  /// Transform all uses of an allocation in the parent region.
  /// Shared implementation: collects users (excluding DbAllocOp and EDT uses),
  /// then calls transformOps().
  void transformUsesInParentRegion(Operation *alloc, DbAllocOp dbAlloc,
                                   ArtsCodegen &AC,
                                   llvm::SetVector<Operation *> &opsToRemove);

  unsigned getOuterRank() const { return outerRank; }
  unsigned getInnerRank() const { return innerRank; }
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_DB_DBINDEXERBASE_H
