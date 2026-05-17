///==========================================================================///
/// File: RtDbUtils.h
///
/// Runtime-dialect datablock provenance helpers. This extends the abstract
/// ARTS DB utilities with ARTS-RT-only operations such as depv acquire handles
/// and runtime DB pointer arithmetic.
///==========================================================================///
#ifndef CARTS_DIALECT_ARTS_RT_UTILS_RTDBUTILS_H
#define CARTS_DIALECT_ARTS_RT_UTILS_RTDBUTILS_H

#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include <optional>

namespace mlir {
namespace carts::arts_rt {

class RtDbUtils {
public:
  static carts::arts::DbLoweringInfo
  extractDbLoweringInfo(carts::arts::DbAcquireOp op);
  static carts::arts::DbLoweringInfo extractDbLoweringInfo(DepDbAcquireOp op);
  static carts::arts::DbLoweringInfo
  extractDbLoweringInfo(carts::arts::DbAllocOp op);

  static Operation *getUnderlyingDb(Value value, unsigned depth = 0);
  static Operation *getUnderlyingDbAlloc(Value value);
  static carts::arts::DbAllocOp getAllocOpFromGuid(Value dbGuid);
  static Value getUnderlyingValue(Value value);
  static Operation *getUnderlyingOperation(Value value);
  static bool isDerivedFromPtr(Value value, Value source);

  static std::optional<int64_t> tryFoldConstantIndex(Value value,
                                                     unsigned depth = 0);
  static std::optional<int64_t> getConstantIndexStripped(Value value);

  static SmallVector<Value> getSizesFromDb(Operation *dbOp);
  static SmallVector<Value> getSizesFromDb(Value dbPtr);
  static SmallVector<Value> getDepSizesFromDb(Operation *dbOp);
  static SmallVector<Value> getDepSizesFromDb(Value dbPtr);
  static SmallVector<Value> getDepOffsetsFromDb(Operation *dbOp);
  static SmallVector<Value> getDepOffsetsFromDb(Value dbPtr);

  static carts::arts::DbMode convertArtsModeToDbMode(carts::arts::ArtsMode mode);

  static void convertElementSliceToBlockSlice(
      OpBuilder &builder, Location loc, ValueRange elementOffsets,
      ValueRange elementSizes, ValueRange blockSpans,
      ValueRange totalBlockCounts, SmallVectorImpl<Value> &blockOffsets,
      SmallVectorImpl<Value> &blockSizes);

  static void mergeNormalizedBlockSlice(
      OpBuilder &builder, Location loc, ValueRange existingOffsets,
      ValueRange existingSizes, ValueRange totalBlockCounts,
      ValueRange normalizedOffsets, ValueRange normalizedSizes,
      SmallVectorImpl<Value> &blockOffsets,
      SmallVectorImpl<Value> &blockSizes);
};

} // namespace carts::arts_rt
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_RT_UTILS_RTDBUTILS_H
