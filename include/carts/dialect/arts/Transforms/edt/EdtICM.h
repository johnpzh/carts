///==========================================================================///
/// File: EdtICM.h
///
/// Problem - When operations within EDT regions are invariant (depend only on
/// external values), keeping them inside the EDT:
/// - Creates unnecessary computation in potentially distributed contexts
/// - May duplicate work across multiple EDTs using the same values
///
/// Solution - This pass:
/// 1. Identifies operations in EDT regions that only depend on external values
/// 2. Verifies these operations are safe to move (memory-effect free, no side
/// effects)
/// 3. Hoists them out of the EDT region to the parent block
/// 4. Preserves correct execution order and dependencies
///
/// Example:
/// ```mlir
/// Before:
/// %base = memref.alloc() : memref<100xf32>
/// arts.edt {
///   %c1 = arith.constant 1 : index  // Invariant
///   %c2 = arith.constant 2 : index  // Invariant
///   %sum = arith.addi %c1, %c2      // Invariant
///   %val = memref.load %base[%sum]  // Not invariant (memory effect)
/// }
///
/// After:
/// %base = memref.alloc() : memref<100xf32>
/// %c1 = arith.constant 1 : index
/// %c2 = arith.constant 2 : index
/// %sum = arith.addi %c1, %c2
/// arts.edt {
///   %val = memref.load %base[%sum]
/// }
/// ```
/// This pass is also important when running the CreateDbs pass, since
/// it will create datablocks for all values used in the EDT region. If the
/// invariant code is not moved out, the db will be created for the
/// invariant code, which is not the intended behavior.
///==========================================================================///=

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTICM_H
#define ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTICM_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;
class Region;
class Value;
} // namespace mlir

namespace mlir {
namespace arts {
class EdtOp;

/// Checks whether the given op can be hoisted by checking that
/// - the op and none of its contained operations depend on values defined
///   inside the region (by means of calling definedOutside).
/// - the op is not a terminator.
bool canBeHoistedFromEdt(Region &edtRegion, Operation *op);

/// Move side-effect free EDT invariant code out of an EDT op.
/// Returns the number of operations moved.
uint64_t moveEdtInvariantCode(EdtOp edtOp);
} // namespace arts
} // end namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTICM_H
