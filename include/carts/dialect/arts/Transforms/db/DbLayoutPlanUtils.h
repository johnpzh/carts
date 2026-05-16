///==========================================================================///
/// File: DbLayoutPlanUtils.h
///
/// Helpers for materializing DB layouts from an already-authored plan.
///
/// These utilities do not choose tensor partitioning policy. SDE owns owner
/// dimensions, physical block shape, and halo policy. ARTS utilities may
/// inspect those attrs for diagnostics or ARTS object materialization, but
/// block-local access rewriting belongs to SDE MU/token lowering.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_TRANSFORMS_DB_DBLAYOUTPLANUTILS_H
#define CARTS_DIALECT_ARTS_TRANSFORMS_DB_DBLAYOUTPLANUTILS_H

#include "carts/dialect/arts/Transforms/db/DbLayoutPlan.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace carts::arts {

/// Returns true when an operation carries the minimum SDE-authored physical
/// layout plan needed to create a block DB immediately.
bool hasPhysicalDbLayoutPlan(Operation *op);

/// Resolve explicit owner-dimension and physical-block-shape attrs into a
/// concrete DB rewrite plan for an allocation with the provided logical element
/// extents. This is for dialect boundaries that have SDE/CODIR-authored plan
/// attrs before an ARTS op exists to carry the generic plan attributes.
FailureOr<DbPhysicalLayoutPlan>
resolvePhysicalDbLayoutPlan(ArrayAttr ownerDimsAttr, ArrayAttr blockShapeAttr,
                            ValueRange elementSizes, OpBuilder &builder,
                            Location loc);

} // namespace carts::arts
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_TRANSFORMS_DB_DBLAYOUTPLANUTILS_H
