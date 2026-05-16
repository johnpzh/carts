///==========================================================================///
/// File: ValueAnalysis.h
///
/// Static analysis utilities for MLIR Values, constants, and casts.
///==========================================================================///

#ifndef CARTS_UTILS_VALUEANALYSIS_H
#define CARTS_UTILS_VALUEANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

namespace mlir {
namespace carts::arts {

/// Static analysis utilities for MLIR Values, constants, and casts.
class ValueAnalysis {
public:
  ///===----------------------------------------------------------------------===////
  /// Constant Value Analysis
  ///===----------------------------------------------------------------------===////

  /// Check if value is defined by a ConstantLike operation.
  static bool isValueConstant(Value val);

  /// Extract a constant integer/index value, returns false if not constant.
  static bool getConstantIndex(Value v, int64_t &out);

  /// Like getConstantIndex but returns optional.
  static std::optional<int64_t> getConstantValue(Value v);

  /// Return the SSA value stored in an OpFoldResult, or null for attributes.
  static Value getValueFromFoldResult(OpFoldResult ofr);

  /// Extract a constant integer/index from a value or integer attribute.
  static std::optional<int64_t> getConstantIndex(OpFoldResult ofr);

  /// Recursively fold constant index expressions through basic arith ops.
  static std::optional<int64_t> tryFoldConstantIndex(Value v,
                                                     unsigned depth = 0);

  static std::optional<int64_t> getConstantIndexStripped(Value v);

  static std::optional<double> getConstantFloat(Value v);
  static bool isConstantEqual(Value v, int64_t val);
  static bool isZeroConstant(Value v);
  static bool isOneConstant(Value v);
  static bool isConstantBool(Value v, bool expected);
  static bool isTrueConstant(Value v);

  /// Check if a value is structurally "one-like": either a literal 1, or
  /// an expression of the form `1 + (x - x)` or `1 + (min(x,y) - y)` that
  /// canonically reduces to one.
  static bool isOneLikeValue(Value value);

  /// Check equivalence after stripping casts (same Value or same constant).
  static bool sameValue(Value a, Value b);

  /// Exact SSA identity comparison for two value ranges.
  static bool areValueRangesIdentical(ValueRange lhs, ValueRange rhs);

  /// Exact SSA identity comparison for a value range prefix.
  static bool hasValueRangePrefix(ValueRange values, ValueRange prefix);

  /// ValueAnalysis::sameValue comparison for two value ranges.
  static bool areValueRangesEquivalent(ValueRange lhs, ValueRange rhs);

  ///===----------------------------------------------------------------------===////
  /// Value Range and Scale Comparison
  ///===----------------------------------------------------------------------===////

  /// Strip numeric casts and max(x, 1) clamping.
  static Value stripClampOne(Value v);

  /// Strip select(cmp(...), x, y) min/max clamp patterns recursively.
  /// Peels through up to `maxDepth` layers of select-based clamping.
  static Value stripSelectClamp(Value value, int maxDepth = 8);

  /// Shallow structural equivalence for index expressions.
  static bool areValuesEquivalent(Value a, Value b, int depth = 0);

  /// Check if value is a constant >= 1 (strips casts first).
  static bool isConstantAtLeastOne(Value v);

  /// Recursively prove value is non-zero (for div/rem safety).
  static bool isProvablyNonZero(Value v, unsigned depth = 0);

  ///===----------------------------------------------------------------------===////
  /// Value Type Conversion and Casting
  ///===----------------------------------------------------------------------===////

  /// Strip through index casts, sign/zero extensions, truncations.
  static Value stripNumericCasts(Value value);

  /// Cast to index type if needed. Returns value unchanged if non-integer.
  static Value castToIndex(Value value, OpBuilder &builder, Location loc);

  /// Cast to index type. Returns null Value for non-integer types.
  static Value ensureIndexType(Value value, OpBuilder &builder, Location loc);

  ///===----------------------------------------------------------------------===////
  /// Value Dependencies and Analysis
  ///===----------------------------------------------------------------------===////

  /// Check if value depends on base through arithmetic operations.
  static bool dependsOn(Value value, Value base, int depth = 0);

  /// Detect constant stride for base within an index expression.
  /// Returns 1 when idx==base, C when idx contains base*C, else nullopt.
  static std::optional<int64_t> getOffsetStride(Value idx, Value base,
                                                int depth = 0);

  /// Check if a pointer/memref value is derived from source through
  /// GEP, casts, SubView, and similar pointer-manipulating operations.
  static bool isDerivedFromPtr(Value value, Value source);

  /// Extract constant offset from expression relative to loopIV + chunkOffset.
  static std::optional<int64_t> extractConstantOffset(Value idx, Value loopIV,
                                                      Value chunkOffset);

  /// Strip constant add/sub offsets, returning base and accumulated offset.
  static Value stripConstantOffset(Value value, int64_t *outConst = nullptr);

  ///===----------------------------------------------------------------------===////
  /// Memref View-Like Op Stripping
  ///===----------------------------------------------------------------------===////

  /// Strip through memref view-like wrapper ops (CastOp, SubViewOp,
  /// ReinterpretCastOp) without crossing DB or EDT boundaries.
  /// Unlike getUnderlyingValue, this does NOT trace through DbAcquireOp,
  /// DbRefOp, DbGepOp, EdtOp block args, LLVM::GEPOp, or polygeist ops.
  static Value stripMemrefViewOps(Value value);

  ///===----------------------------------------------------------------------===////
  /// Underlying Value Tracing
  ///===----------------------------------------------------------------------===////

  /// Trace to root allocation through casts, acquires, GEPs, etc.
  static Value getUnderlyingValue(Value v);

  /// Like getUnderlyingValue but returns the defining operation.
  static Operation *getUnderlyingOperation(Value v);

  ///===----------------------------------------------------------------------===///
  /// Value Reconstruction for Dominance
  ///===----------------------------------------------------------------------===///

  /// Reconstruct a value at a dominating point by tracing through arithmetic.
  /// Returns the value directly if it already dominates, or nullptr on failure.
  static Value traceValueToDominating(Value value, Operation *insertBefore,
                                      OpBuilder &builder,
                                      DominanceInfo &domInfo, Location loc,
                                      unsigned depth = 0);

  /// Recreate a binary op at a dominating point. Callback traces each operand.
  template <typename OpType>
  static Value traceBinaryOp(OpType op, Operation *insertBefore,
                             OpBuilder &builder, Location loc,
                             llvm::function_ref<Value(Value)> traceValueFn) {
    Value lhs = traceValueFn(op.getLhs());
    Value rhs = traceValueFn(op.getRhs());
    if (!lhs || !rhs)
      return nullptr;
    builder.setInsertionPoint(insertBefore);
    return OpType::create(builder, loc, lhs, rhs);
  }

  ///===----------------------------------------------------------------------===///
  /// Value Cloning Utilities
  ///===----------------------------------------------------------------------===///

  /// Check if an operation can be safely cloned (constant-like or pure).
  static bool
  canCloneOperation(Operation *op, bool allowMemoryEffectFree = true,
                    llvm::function_ref<bool(Operation *)> extraAllowed = {});

  /// Clone external values and their dependencies into a target region.
  static bool cloneValuesIntoRegion(
      llvm::SetVector<Value> &values, Region *targetRegion, IRMapping &mapper,
      OpBuilder &builder, bool allowMemoryEffectFree = true,
      llvm::function_ref<bool(Operation *)> extraAllowed = {});
};

} // namespace carts::arts
} // namespace mlir

#endif // CARTS_UTILS_VALUEANALYSIS_H
