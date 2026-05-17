///==========================================================================///
/// File: ArtsOpUtils.h
///
/// Small predicates for recognizing abstract ARTS operations.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_UTILS_ARTSOPUTILS_H
#define CARTS_DIALECT_ARTS_UTILS_ARTSOPUTILS_H

#include "carts/Dialect.h"

namespace mlir {
namespace carts::arts {

inline bool isArtsRegion(Operation *op) {
  return isa<EdtOp>(op) || isa<EpochOp>(op);
}

inline bool isArtsOp(Operation *op) {
  return isArtsRegion(op) ||
         isa<BarrierOp, AllocOp, DbAllocOp, DbAcquireOp, DbReleaseOp, DbFreeOp,
             RuntimeQueryOp>(op);
}

} // namespace carts::arts
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_UTILS_ARTSOPUTILS_H
