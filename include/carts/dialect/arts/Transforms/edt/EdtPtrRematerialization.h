///==========================================================================///
/// File: EdtPointerRematerialization.h
///
/// Problem - When EDTs run on different nodes, pointer values passed as
/// parameters may be invalid since:
/// - Pointers are node-specific
/// - Memory layouts might differ
/// - Original pointer computation may be unavailable
///
/// Solution - This pass:
/// 1. Identifies pointer values used in EDT but defined outside
/// 2. Clones pointer-generating operations into the EDT
/// 3. Recreates the pointer computation chain locally
/// 4. Replaces uses of external pointers with local versions
///
/// Example:
/// ```mlir
/// Before:
/// %ptr = llvm.getelementptr %base[%idx] : !llvm.ptr
/// arts.edt (%ptr as %arg0) {
///   %val = llvm.load %arg0 : !llvm.ptr
/// }
///
/// After:
/// arts.edt (%base as %arg0, %idx as %arg1) {
///   %ptr = llvm.getelementptr %arg0[%arg1] : !llvm.ptr
///   %val = llvm.load %ptr : !llvm.ptr
/// }
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTPTRREMATERIALIZATION_H
#define ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTPTRREMATERIALIZATION_H

namespace mlir {
namespace arts {
class EdtOp;

void rematerializePointersInEdt(EdtOp edtOp);

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_EDT_EDTPTRREMATERIALIZATION_H