///==========================================================================///
/// File: GUIDRangeDetection.h
///
/// DT-4: Detect loops with batch GUID allocation opportunities.
///
/// ARTS supports arts_guid_reserve_range(type, N, route) for O(1) batch
/// allocation, but the compiler currently emits individual
/// arts_guid_reserve() calls per DbAllocOp.  This transform walks the
/// module looking for scf::ForOp loops that contain exactly one DbAllocOp
/// in their body and marks them as candidates for range-based lowering.
///
/// Markings applied:
///   DbAllocOp  -> "guid_range_trip_count" (I64Attr)
///                  Static trip count or -1 when only known at runtime.
///   scf::ForOp -> "has_guid_range_alloc"  (UnitAttr)
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_DB_TRANSFORMS_GUIDRANGEDETECTION_H
#define ARTS_DIALECT_CORE_TRANSFORMS_DB_TRANSFORMS_GUIDRANGEDETECTION_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace arts {

/// Detect loops with batch GUID allocation opportunities.
/// Marks DbAllocOps inside qualifying loops with guid_range_candidate attr.
/// Returns true if any candidates were found.
bool detectGUIDRangeCandidates(ModuleOp module);

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_DB_TRANSFORMS_GUIDRANGEDETECTION_H
