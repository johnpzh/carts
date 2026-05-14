///==========================================================================///
/// File: DbLayoutPlanUtils.h
///
/// Helpers for materializing DB layouts from an already-authored plan.
///
/// These utilities do not choose tensor partitioning policy. SDE owns owner
/// dimensions, physical block shape, and halo policy; this layer only converts
/// those attrs into the physical DbPhysicalLayoutPlan consumed by CreateDbs and
/// DB indexers.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTPLANUTILS_H
#define ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTPLANUTILS_H

#include "arts/dialect/core/Transforms/db/DbLayoutPlan.h"
#include "arts/utils/LoweringContractUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace arts {

/// Returns true when an operation carries the minimum SDE-authored physical
/// layout plan needed to create a block DB immediately.
bool hasPhysicalDbLayoutPlan(Operation *op);

/// Resolve structured plan attrs on `planSource` into a concrete DB rewrite
/// plan for an allocation with the provided logical element extents.
FailureOr<DbPhysicalLayoutPlan> resolvePhysicalDbLayoutPlan(Operation *planSource,
                                                     ValueRange elementSizes,
                                                     OpBuilder &builder,
                                                     Location loc);

/// Return true when a read task's owner layout names the same source owner
/// dimensions but uses a different physical block shape. This is a layout
/// translation case: Core should keep worker-local dependency windows and
/// convert logical element windows into source DB block coordinates.
bool hasReadOnlySourceLayoutMismatch(DbAllocOp sourceAlloc,
                                     Operation *taskPlanSource,
                                     const LoweringContractInfo &contract);

/// Project the source allocation's physical block shape onto the contract's
/// owner dimensions. Returns nullopt when source layout attrs cannot prove a
/// compatible owner mapping.
std::optional<SmallVector<int64_t, 4>>
getSourceOwnerBlockShape(DbAllocOp sourceAlloc,
                         const LoweringContractInfo &contract);

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTPLANUTILS_H
