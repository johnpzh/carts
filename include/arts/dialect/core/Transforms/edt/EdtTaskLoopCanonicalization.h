///==========================================================================///
/// File: EdtTaskLoopCanonicalization.h
///
/// Helpers for canonicalizing generated EDT task-loop bodies after
/// ForLowering has cloned the source loop into task-local IR.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTTASKLOOPCANONICALIZATION_H
#define ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTTASKLOOPCANONICALIZATION_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace arts {

enum class DistributionKind;

/// Sink the generated task-local chunk loop under invariant middle/inner loops
/// when the chunk IV only indexes the contiguous memref dimension. Returns the
/// replacement loop anchor when a rewrite happened, otherwise nullptr.
Operation *sinkTaskLoopToContiguousInnerDim(scf::ForOp iterLoop,
                                            Value globalIdx,
                                            DistributionKind kind);

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTTASKLOOPCANONICALIZATION_H
