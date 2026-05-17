///==========================================================================///
/// File: RemovalUtils.cpp
/// Implements the RemovalUtils class for deferred operation removal.
///==========================================================================///

#include "carts/utils/RemovalUtils.h"
#include "carts/Dialect.h"
#include "carts/utils/Debug.h"
#include "mlir/IR/Builders.h"

ARTS_DEBUG_SETUP(removal_utils);

namespace mlir {
namespace carts {

///===----------------------------------------------------------------------===///
/// RemovalUtils Implementation
///===----------------------------------------------------------------------===///

void RemovalUtils::markForRemoval(Operation *op) {
  if (!op)
    return;
  if (isMarkedForRemoval(op))
    return;

  /// If an ancestor is already scheduled for removal, no need to track this
  /// operation - it will be removed when the ancestor is removed
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isMarkedForRemoval(parent))
      return;
  }

  /// Remove any descendants that are already scheduled; removing this op will
  /// remove them as well
  SmallVector<Operation *, 4> descendants;
  for (Operation *scheduled : opsToRemove) {
    if (op->isAncestor(scheduled) && scheduled != op)
      descendants.push_back(scheduled);
  }
  for (Operation *scheduled : descendants)
    opsToRemove.remove(scheduled);

  opsToRemove.insert(op);
}

bool RemovalUtils::isMarkedForRemoval(Operation *op) const {
  return opsToRemove.contains(op);
}

LogicalResult RemovalUtils::verifyAllUsersMarked(Value value,
                                                 StringRef allocName) {
  unsigned totalUsers = 0;
  SmallVector<Operation *> unmarkedUsers;

  for (Operation *user : value.getUsers()) {
    totalUsers++;
    if (!isMarkedForRemoval(user))
      unmarkedUsers.push_back(user);
  }

  if (!unmarkedUsers.empty()) {
    if (!allocName.empty()) {
      ARTS_DEBUG("  Warning: " << allocName << " has " << unmarkedUsers.size()
                               << " unmarked users out of " << totalUsers);
    } else {
      ARTS_DEBUG("  Warning: Value has " << unmarkedUsers.size()
                                         << " unmarked users out of "
                                         << totalUsers);
    }
    for (Operation *user : unmarkedUsers) {
      ARTS_DEBUG("    - Unmarked: " << user->getName().getStringRef());
    }
    return failure();
  }

  return success();
}

void RemovalUtils::removeAllMarked() {
  /// Track erased operations to avoid use-after-free when an operation
  /// was erased as a nested child of a previously-erased operation
  SmallPtrSet<Operation *, 32> erased;

  for (Operation *op : opsToRemove) {
    if (!op || erased.contains(op))
      continue;

    /// Check if operation is still alive (not already erased as a child)
    Block *block = op->getBlock();
    if (!block)
      continue;

    if (!op->use_empty())
      continue;

    /// Mark all nested operations as erased before erasing parent
    op->walk([&](Operation *nested) { erased.insert(nested); });
    op->erase();
  }
  opsToRemove.clear();
}

///===----------------------------------------------------------------------===///
/// Recursive Removal Implementation
///===----------------------------------------------------------------------===///

void RemovalUtils::removeOpImpl(Operation *op, OpBuilder &builder,
                                SmallPtrSet<Operation *, 32> &seen,
                                bool recursive) {
  if (!op)
    return;

  /// Check seen BEFORE any dereference to avoid use-after-free.
  if (!seen.insert(op).second)
    return;

  /// Now safe to dereference - check if operation is still linked in IR
  Block *block = op->getBlock();
  if (!block)
    return;

  Operation *parentOp = block->getParentOp();
  if (!parentOp)
    return;

  ARTS_DEBUG("   - Removing operation: " << *op);

  /// Collect unique dependents before modifying uses
  SmallPtrSet<Operation *, 8> dependentSet;
  if (recursive) {
    for (Value result : op->getResults())
      for (Operation *user : result.getUsers())
        if (!seen.contains(user))
          dependentSet.insert(user);
  }

  /// If this is a terminator, mark the parent for removal
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    if (Operation *parent = op->getParentOp()) {
      markForRemoval(parent);
      ARTS_DEBUG("   - Skipping terminator removal, marked parent instead");
    }
    return;
  }

  /// In non-recursive mode, never erase operations that still have uses
  if (!recursive && !op->use_empty())
    return;

  /// Check if any user is a terminator
  bool hasTerminatorUser = false;
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (user->hasTrait<OpTrait::IsTerminator>()) {
        hasTerminatorUser = true;
        break;
      }
    }
    if (hasTerminatorUser)
      break;
  }

  if (hasTerminatorUser) {
    replaceWithUndef(op, builder);
    ARTS_DEBUG("   - Replaced uses with undef (terminator user)");
  }
  op->dropAllUses();

  /// Mark all nested operations as seen before erasing to prevent
  /// use-after-free if they're also in the removal list
  for (Region &region : op->getRegions())
    region.walk([&](Operation *nested) { seen.insert(nested); });

  op->erase();

  if (!recursive)
    return;

  /// Remove dependents recursively
  for (Operation *userOp : dependentSet)
    removeOpImpl(userOp, builder, seen, recursive);
}

void RemovalUtils::removeAllMarked(ModuleOp module, bool recursive) {
  if (opsToRemove.empty())
    return;

  OpBuilder builder(module.getContext());
  SmallPtrSet<Operation *, 32> seen;

  /// Remove operations, processing newly discovered dead parents
  ARTS_DEBUG(" - Removing " << opsToRemove.size() << " marked operations");

  /// Filter out operations that may have been removed
  SetVector<Operation *> validOpsToRemove;
  for (Operation *op : opsToRemove) {
    if (op && op->getBlock() && op->getBlock()->getParentOp())
      validOpsToRemove.insert(op);
  }

  /// Process operations
  for (Operation *op : validOpsToRemove) {
    removeOpImpl(op, builder, seen, recursive);
  }

  opsToRemove.clear();
  ARTS_DEBUG(" - Removed " << seen.size() << " operations total");
}

///===----------------------------------------------------------------------===///
/// Static Utility Methods
///===----------------------------------------------------------------------===///

void RemovalUtils::removeUndefOps(ModuleOp module) {
  RemovalUtils mgr;
  module.walk([&](arts::UndefOp op) { mgr.markForRemoval(op); });
  ARTS_DEBUG(" - Removing " << mgr.size() << " undef operations");
  mgr.removeAllMarked(module, /*recursive=*/true);
}

void RemovalUtils::replaceWithUndef(Operation *op, OpBuilder &builder) {
  if (!op || op->getNumResults() == 0)
    return;

  OpBuilder::InsertionGuard IG(builder);
  builder.setInsertionPoint(op);
  for (Value result : op->getResults()) {
    if (!result.use_empty()) {
      auto undef =
          arts::UndefOp::create(builder, op->getLoc(), result.getType());
      result.replaceAllUsesWith(undef.getResult());
    }
  }
}

} // namespace carts
} // namespace mlir
