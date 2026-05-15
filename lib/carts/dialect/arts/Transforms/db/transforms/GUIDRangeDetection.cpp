///==========================================================================///
/// File: GUIDRangeDetection.cpp
///
/// DT-4: Detect loops with batch GUID allocation opportunities.
///
/// Each DbAllocOp currently lowers to an individual arts_guid_reserve()
/// call.  When a loop creates N datablocks with identical configuration,
/// the runtime can batch-allocate all N GUIDs in O(1) via
/// arts_guid_reserve_range(type, N, route).
///
/// This transform is a lightweight marking pass.  It walks all
/// scf::ForOp loops, identifies those whose body contains exactly one
/// DbAllocOp, computes the trip count when bounds are statically
/// analyzable, and stamps the IR with attributes that a later lowering
/// phase (ConvertArtsToLLVM / DbAllocPattern) can consume.
///
/// Markings:
///   DbAllocOp  -> "guid_range_trip_count" : I64Attr
///                  Positive value = static trip count.
///                  -1             = dynamic (bounds not constant).
///   scf::ForOp -> "has_guid_range_alloc"  : UnitAttr
///
/// The transform is intentionally conservative:
///   - Only single-DbAllocOp loop bodies are matched.
///   - Nested loops with DbAllocOps are skipped (the inner loop is the
///     candidate, not the outer one).
///   - No rewriting is performed; only attributes are attached.
///
/// TODO(DT-4): This is annotation-only — attributes are stamped but
/// ConvertArtsToLLVM.cpp's DbAllocPattern does not consume
/// guid_range_trip_count. Lowering needs to: (1) check for
/// guid_range_trip_count on DbAllocOp, (2) hoist allocation outside the
/// loop, (3) replace N artsGuidReserve() with one
/// arts_guid_reserve_range(type, N, route), (4) use
/// arts_guid_from_index(rangeGuid, loopIndex) inside the loop.
///==========================================================================///

/// Dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

/// Arts
#include "carts/Dialect.h"
#include "carts/dialect/arts/Transforms/db/transforms/GUIDRangeDetection.h"
#include "carts/utils/Debug.h"
#include "carts/utils/LoopUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/ValueAnalysis.h"

ARTS_DEBUG_SETUP(guid_range_detection);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

/// Attribute names stamped by this transform (centralized in
/// OperationAttributes.h).
using AttrNames::Operation::GuidRangeTripCount;
using AttrNames::Operation::HasGuidRangeAlloc;

namespace {

/// Collect all DbAllocOps that are direct children of a ForOp body
/// (not nested inside inner ForOps).  Returns the count of DbAllocOps
/// found and stores the single candidate (when count == 1).
static unsigned collectDirectDbAllocs(scf::ForOp forOp, DbAllocOp &candidate) {
  unsigned count = 0;
  candidate = nullptr;

  forOp.getBody()->walk([&](DbAllocOp alloc) {
    /// Only count allocs whose immediately enclosing ForOp is this one.
    /// Allocs nested inside inner ForOps belong to those inner loops.
    if (alloc->getParentOfType<scf::ForOp>() != forOp)
      return;

    ++count;
    if (count == 1)
      candidate = alloc;
  });

  return count;
}

} // namespace

bool mlir::carts::arts::detectGUIDRangeCandidates(ModuleOp module) {
  ARTS_DEBUG_HEADER(GUIDRangeDetection);

  bool foundAny = false;
  MLIRContext *ctx = module.getContext();

  module.walk([&](scf::ForOp forOp) {
    /// Collect DbAllocOps that are direct children of this loop body.
    DbAllocOp allocCandidate;
    unsigned allocCount = collectDirectDbAllocs(forOp, allocCandidate);

    if (allocCount == 0) {
      ARTS_DEBUG("ForOp at " << forOp.getLoc() << ": no DbAllocOps, skipping");
      return;
    }

    if (allocCount > 1) {
      ARTS_WARN("ForOp at " << forOp.getLoc() << ": " << allocCount
                            << " DbAllocOps in body, skipping "
                            << "(only single-alloc loops supported)");
      return;
    }

    /// Single DbAllocOp -- this is a candidate.
    auto staticCount = arts::getStaticTripCount(forOp);
    int64_t tripCount = staticCount.value_or(-1);

    /// Mark the DbAllocOp with the trip count.
    auto i64Type = IntegerType::get(ctx, 64);
    allocCandidate->setAttr(GuidRangeTripCount,
                            IntegerAttr::get(i64Type, tripCount));

    /// Mark the ForOp with a unit attribute.
    forOp->setAttr(HasGuidRangeAlloc, UnitAttr::get(ctx));

    foundAny = true;

    if (tripCount > 0) {
      ARTS_INFO("ForOp at " << forOp.getLoc()
                            << ": marked DbAllocOp with static trip count "
                            << tripCount);
    } else if (tripCount == 0) {
      ARTS_INFO("ForOp at " << forOp.getLoc()
                            << ": marked DbAllocOp with zero trip count "
                            << "(loop never executes)");
    } else {
      ARTS_INFO("ForOp at " << forOp.getLoc()
                            << ": marked DbAllocOp with dynamic trip count "
                            << "(bounds not statically analyzable)");
    }
  });

  if (foundAny) {
    ARTS_INFO("GUID range detection: found candidates");
  } else {
    ARTS_DEBUG("GUID range detection: no candidates found");
  }

  ARTS_DEBUG_FOOTER(GUIDRangeDetection);
  return foundAny;
}
