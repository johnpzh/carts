///==========================================================================///
/// File: EdtLoweringInternal.h
///
/// Local implementation contract for EdtLowering. This header is
/// intentionally private to the edt-lowering implementation split and
/// should not be used as shared compiler infrastructure.
///==========================================================================///

#ifndef ARTS_DIALECT_RT_CONVERSION_ARTSTORT_EDTLOWERINGINTERNAL_H
#define ARTS_DIALECT_RT_CONVERSION_ARTSTORT_EDTLOWERINGINTERNAL_H

#include "carts/Dialect.h"
#include "carts/dialect/arts/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/EdtUtils.h"
#include "carts/dialect/arts/Utils/LoweringContractUtils.h"
#include "carts/dialect/arts/Utils/PartitionPredicates.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace mlir::carts::arts::edt_lowering {

///===----------------------------------------------------------------------===//
/// Constants
///===----------------------------------------------------------------------===//

static constexpr int32_t kArtsDepFlagPreferDuplicate = 1 << 0;
static constexpr int32_t kArtsDepFlagPreserveShape = 1 << 1;

///===----------------------------------------------------------------------===//
/// Structs
///===----------------------------------------------------------------------===//

struct NormalizedElementSlice {
  SmallVector<Value, 4> offsets;
  SmallVector<Value, 4> sizes;
  Value representable;
  Value contiguous;
  Value wholeBlock;
};

struct DepSourceInfo {
  DbAcquireOp dbAcquire;
  rt::DepDbAcquireOp depDbAcquire;
};

///===----------------------------------------------------------------------===//
/// Helper Functions
///===----------------------------------------------------------------------===//

void normalizeTaskDepSlice(ArtsCodegen *AC, DbAcquireOp acquire,
                           const LoweringContractInfo &contract);

std::optional<NormalizedElementSlice>
normalizeCommonElementSlice(ArtsCodegen *AC, DbAcquireOp acquire,
                            DbAllocOp alloc);

Value loadRepresentativeGuidScalar(ArtsCodegen *AC, Location loc,
                                   Value guidStorage);

DepSourceInfo resolveDepSource(Value dep);

Operation *getCanonicalDependencySource(Value dep);

} // namespace mlir::carts::arts::edt_lowering

#endif // ARTS_DIALECT_RT_CONVERSION_ARTSTORT_EDTLOWERINGINTERNAL_H
