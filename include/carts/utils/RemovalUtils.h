///==========================================================================///
/// File: RemovalUtils.h
///
/// This file defines the RemovalUtils class which manages deferred
/// operation removal with safety checks.
///==========================================================================///

#ifndef CARTS_UTILS_REMOVALUTILS_H
#define CARTS_UTILS_REMOVALUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace carts {

///===----------------------------------------------------------------------===///
/// RemovalUtils - Manages deferred operation removal
///
/// Tracks operations to be removed and provides utilities for safe removal
/// with verification. Operations are not removed immediately but queued for
/// later removal to avoid iterator invalidation during transformation passes.
///===----------------------------------------------------------------------===///
class RemovalUtils {
public:
  RemovalUtils() = default;

  /// Mark an operation for removal
  /// Automatically handles ancestor/descendant relationships:
  /// - If an ancestor is already marked, this operation is not tracked
  /// - If descendants are already marked, they are removed from tracking
  void markForRemoval(Operation *op);

  /// Check if an operation is marked for removal
  bool isMarkedForRemoval(Operation *op) const;

  /// Remove all marked operations with recursive removal support
  /// Drops all uses before erasing
  void removeAllMarked(ModuleOp module, bool recursive = false);

  /// Get the set of operations to remove (for inspection/debugging)
  const SetVector<Operation *> &getOpsToRemove() const { return opsToRemove; }

  /// Clear all marked operations without removing them
  void clear() { opsToRemove.clear(); }

  /// Get the number of operations marked for removal
  size_t size() const { return opsToRemove.size(); }

  /// Check if any operations are marked for removal
  bool empty() const { return opsToRemove.empty(); }

private:
  /// Recursively remove an operation and its dependents
  /// @param op The operation to remove
  /// @param builder Builder for creating replacement operations if needed
  /// @param seen Set of operations already visited
  /// @param recursive If true, recursively remove dependent operations
  void removeOpImpl(Operation *op, OpBuilder &builder,
                    SmallPtrSet<Operation *, 32> &seen, bool recursive);

  /// Replace remaining result uses with dialect-neutral placeholders before
  /// erasing an operation that still feeds a terminator.
  static void replaceLiveUsesWithUnrealizedCast(Operation *op,
                                                OpBuilder &builder);

  SetVector<Operation *> opsToRemove;
};

} // namespace carts
} // namespace mlir

#endif // CARTS_UTILS_REMOVALUTILS_H
