///==========================================================================///
/// File: DbLayoutPlanUtils.h
///
/// Helpers for materializing DB layouts from an already-authored plan.
///
/// These utilities do not choose tensor partitioning policy. SDE owns owner
/// dimensions, physical block shape, and halo policy; this layer only converts
/// those attrs into the physical DbRewritePlan consumed by CreateDbs and DB
/// indexers.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTPLANUTILS_H
#define ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTPLANUTILS_H

#include "arts/dialect/core/Transforms/db/DbRewriter.h"
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
FailureOr<DbRewritePlan> resolvePhysicalDbLayoutPlan(Operation *planSource,
                                                     ValueRange elementSizes,
                                                     OpBuilder &builder,
                                                     Location loc);

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_DB_DBLAYOUTPLANUTILS_H
