///==========================================================================///
/// File: DbUtils.h
///
/// Utility functions for working with ARTS DBs (DbAllocOp, DbAcquireOp).
/// This module consolidates all DB-related utilities including tracing,
/// size/offset extraction, and stride computation.
///==========================================================================///

#ifndef CARTS_UTILS_DBUTILS_H
#define CARTS_UTILS_DBUTILS_H

#include "carts/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

namespace mlir {
namespace arts {

/// Information extracted from a datablock operation for lowering.
/// Consolidates sizes, offsets, indices, and single-element detection into
/// a single struct returned by extractDbLoweringInfo().
struct DbLoweringInfo {
  SmallVector<Value> sizes;
  SmallVector<Value> offsets;
  SmallVector<Value> indices;
  bool isSingleElement = false;
};

///===----------------------------------------------------------------------===///
/// Datablock Utilities
///===----------------------------------------------------------------------===///
/// Utility class for working with ARTS datablocks (DbAllocOp, DbAcquireOp).
class DbUtils {
public:
  enum class MemoryAccessKind { Read, Write };

  struct MemoryAccessInfo {
    Operation *op = nullptr;
    Value memref;
    SmallVector<Value> indices;
    MemoryAccessKind kind = MemoryAccessKind::Read;

    bool isRead() const { return kind == MemoryAccessKind::Read; }
    bool isWrite() const { return kind == MemoryAccessKind::Write; }
  };

  ///===----------------------------------------------------------------------===////
  /// Datablock Tracing Utilities
  ///===----------------------------------------------------------------------===////
  /// Functions for tracing values back to their source datablock operations

  /// Trace a dependency back to its root DbAllocOp.
  /// Returns {guid, ptr} from the DbAllocOp, or nullopt if not found.
  static std::optional<std::pair<Value, Value>> traceToDbAlloc(Value dep);

  /// Finds the datablock-related operation (DbAllocOp or DbAcquireOp)
  /// associated with the given value. The depth parameter prevents infinite
  /// recursion in circular acquire chains.
  static Operation *getUnderlyingDb(Value v, unsigned depth = 0);

  /// Finds the DbAllocOp associated with the given value.
  /// Traces through DbAcquireOp chains to find the root allocation.
  static Operation *getUnderlyingDbAlloc(Value v);

  /// Trace a GUID value through acquire chains to find the originating
  /// DbAllocOp. Returns nullptr if the GUID does not trace to an allocation.
  static DbAllocOp getAllocOpFromGuid(Value dbGuid);

  /// Extract a compile-time constant outer shape from a DbAllocOp.
  /// Returns nullopt if the handle does not define a DbAllocOp or if any
  /// size dimension is dynamic.
  static std::optional<SmallVector<int64_t, 4>>
  getStaticDbOuterShape(Value dbHandle);

  /// Compute the total static partition count (product of all outer sizes)
  /// for a DbAllocOp. Returns -1 if any size is dynamic.
  static int64_t computeStaticPartitionCount(DbAllocOp alloc);

  /// Recover the original DbAllocOp arts.id for a rebuilt or forwarded DB
  /// handle family. Uses an explicit fallback alloc when available, otherwise
  /// traces through DB provenance and preserved root-id attrs.
  static std::optional<int64_t>
  resolveRootAllocId(Value value, DbAllocOp fallbackAlloc = nullptr);

  ///===----------------------------------------------------------------------===////
  /// Datablock Lowering Info Extraction
  ///===----------------------------------------------------------------------===////

  /// Extract lowering info (sizes, offsets, indices, isSingleElement) from
  /// a datablock operation. Handles DbAcquireOp and DepDbAcquireOp by
  /// extracting dependency sizes/offsets/indices; for other ops (e.g.
  /// DbAllocOp) falls back to op.getSizes() with empty offsets/indices.
  template <typename OpType>
  static DbLoweringInfo extractDbLoweringInfo(OpType op);

  ///===----------------------------------------------------------------------===////
  /// Datablock Size and Offset Extraction
  ///===----------------------------------------------------------------------===////
  /// Functions for extracting sizes, element sizes, and offsets from datablocks

  /// Extract sizes from a datablock operation.
  /// Supports DbAllocOp, DbAcquireOp, and DepDbAcquireOp.
  static SmallVector<Value> getSizesFromDb(Operation *dbOp);

  /// Extract sizes from a datablock pointer value.
  /// Traces back to the original DbAllocOp or DbAcquireOp that created it.
  static SmallVector<Value> getSizesFromDb(Value dbPtr);

  /// Extract the dependency iteration sizes used for EDT dependency counting
  /// and record_dep lowering.
  /// For block/stencil acquires, this prefers a contract-cached DB-space
  /// window when available; otherwise it falls back to the acquire's explicit
  /// DB-space sizes.
  static SmallVector<Value> getDepSizesFromDb(Operation *dbOp);

  /// Extract dependency iteration sizes from a datablock value.
  static SmallVector<Value> getDepSizesFromDb(Value dbPtr);

  /// Extract the dependency iteration offsets used for EDT dependency lowering.
  /// For block/stencil acquires, this prefers a contract-cached DB-space
  /// window when available; otherwise it falls back to the acquire's explicit
  /// DB-space offsets.
  static SmallVector<Value> getDepOffsetsFromDb(Operation *dbOp);

  /// Extract dependency iteration offsets from a datablock value.
  static SmallVector<Value> getDepOffsetsFromDb(Value dbPtr);

  ///===----------------------------------------------------------------------===////
  /// Datablock Stride Computation
  ///===----------------------------------------------------------------------===////
  /// Functions for computing strides for row-major indexing.
  /// For sizes = [D0, D1, D2, ...], stride = D1 * D2 * ... (TRAILING
  /// dimensions). This follows row-major linearization: index = i0 * stride +
  /// ...
  ///
  /// Examples:
  ///   [4, 16]     -> stride = 16
  ///   [8, 4, 2]   -> stride = 4 * 2 = 8
  ///   [64]        -> stride = 1 (single dimension)

  /// Compute static stride from sizes (compile-time constant).
  /// Returns std::nullopt if any trailing dimension is dynamic.
  static std::optional<int64_t> getStaticStride(ValueRange sizes);

  /// Overload for MemRefType shape.
  static std::optional<int64_t> getStaticStride(MemRefType memrefType);

  /// Get element stride from DbAllocOp (uses getElementSizes()).
  /// This is the stride for linearized access WITHIN each datablock element.
  static std::optional<int64_t> getStaticElementStride(DbAllocOp alloc);

  /// Compute stride as a Value (handles both static and dynamic dimensions).
  /// Materializes the row-major suffix product via upstream MLIR memref
  /// indexing helpers, returning a constant index when fully static.
  /// For single-dimension [N], returns constant 1.
  /// Returns nullptr if sizes is empty.
  static Value getStrideValue(OpBuilder &builder, Location loc,
                              ValueRange sizes);

  ///===----------------------------------------------------------------------===///
  /// Access Mode and Hints Analysis
  ///===----------------------------------------------------------------------===///

  /// Check if an ArtsMode is a writer mode (out or inout).
  static bool isWriterMode(ArtsMode mode);

  /// Return true for DB pointer wrappers whose element payload is i1. These
  /// are compiler control-token DBs, not user data.
  static bool isI1DbPtrType(Type type);

  /// Convert ArtsMode (in/out/inout) to DbMode (read/write).
  /// ArtsMode::in maps to DbMode::read; out and inout map to DbMode::write.
  static DbMode convertArtsModeToDbMode(ArtsMode mode);

  ///===----------------------------------------------------------------------===///
  /// Host-View Safety
  ///===----------------------------------------------------------------------===///

  /// Returns true when a DB value (or any forwarded host alias/view derived
  /// from it) escapes to a non-EDT use that requires a contiguous whole-view
  /// contract. Fragmented layouts such as block/fine-grained cannot preserve
  /// those uses without additional host-side rewriting, so callers should keep
  /// the allocation coarse.
  ///
  /// Direct host load/store users are considered partitionable because DB
  /// rewriters can localize those indices. Opaque users such as helper calls,
  /// dimension queries, raw pointer escapes, and memref copies are treated as
  /// unsafe and force a coarse layout. Pointer casts used only for null/non-
  /// null checks remain partitionable because they do not observe whole-view
  /// layout.
  static bool hasNonPartitionableHostViewUses(Value dbValue);

  ///===----------------------------------------------------------------------===///
  /// Offset Dep and Block Size Analysis
  ///===----------------------------------------------------------------------===///
  /// Functions for analyzing value dependencies on offsets and extracting
  /// base block sizes from size hints.

  /// Try to extract an offset-independent base block size from size hints.
  static Value extractBaseBlockSizeCandidate(Value offsetHint, Value sizeHint,
                                             int depth = 0);

  /// Pick a representative partition offset (prefer non-constant).
  /// Returns the chosen offset and its index via outIdx when provided.
  static Value pickRepresentativePartitionOffset(ArrayRef<Value> offsets,
                                                 unsigned *outIdx = nullptr);

  /// Pick the partition size corresponding to the chosen offset index.
  static Value pickRepresentativePartitionSize(ArrayRef<Value> sizes,
                                               unsigned idx);

  ///===----------------------------------------------------------------------===///
  /// Index Chain Utilities
  ///===----------------------------------------------------------------------===///

  /// Collect full index chain from DbRefOp indices plus memory operation
  /// indices. For accesses through view-like ops (e.g., subview), dynamic
  /// forwarding offsets are inserted before terminal memory-op indices.
  /// Returns a combined chain beginning with DbRef indices.
  static SmallVector<Value> collectFullIndexChain(DbRefOp dbRef,
                                                  Operation *memOp);

  /// Return the accessed memref value from a load/store op, or a null Value.
  /// Supports memref and affine load/store operations.
  static Value getAccessedMemref(Operation *memOp);

  /// Extract index operands from a memory access op.
  /// Supports memref/affine load/store operations.
  static SmallVector<Value> getMemoryAccessIndices(Operation *memOp);

  /// Return decoded memory-access information for a load/store op.
  /// Supports memref/affine load/store operations.
  static std::optional<MemoryAccessInfo> getMemoryAccessInfo(Operation *memOp);

  /// Return true when both scopes read the same underlying DB allocation or
  /// raw memref without either scope writing that root.
  static bool hasSharedReadonlyRoot(Operation *lhs, Operation *rhs);

  /// Return true when two scopes touch the same underlying DB allocation or
  /// raw memref and at least one side writes that root.
  static bool hasSharedWritableRootConflict(Operation *lhs, Operation *rhs);

  /// Collect load/store operations reachable from source through view-like
  /// forwarding ops (casts, subviews, etc.). Optionally restrict to a scope.
  static void collectReachableMemoryOps(Value source,
                                        llvm::SetVector<Operation *> &memOps,
                                        Region *scope = nullptr);

  /// Visit load/store operations reachable from source through view-like
  /// forwarding ops (casts, subviews, etc.). The callback may interrupt early.
  static void forEachReachableMemoryAccess(
      Value source,
      llvm::function_ref<WalkResult(const MemoryAccessInfo &)> visitor,
      Region *scope = nullptr);

  /// Return true when every reachable use of source is forwarding or cleanup
  /// plumbing. Collected operations can be erased to remove the dead DB chain.
  static bool collectCleanupOnlyUseChain(Value source,
                                         llvm::SetVector<Operation *> &ops,
                                         Region *scope = nullptr);

  /// Return true when an i1 control-token DB is only consumed by cleanup
  /// plumbing and stores of literal true. Collected operations can be erased
  /// after removing the owning EDT dependency slot.
  static bool
  collectTrueOnlyControlTokenUseChain(Value source,
                                      llvm::SetVector<Operation *> &ops,
                                      Region *scope = nullptr);

  ///===----------------------------------------------------------------------===///
  /// Element-to-Block Space Slice Conversion
  ///===----------------------------------------------------------------------===///

  /// Convert element-space offsets and sizes to block-space offsets and sizes.
  /// For each dimension, computes:
  ///   startBlock = elementOffset / blockSpan
  ///   endBlock   = (elementOffset + elementSize - 1) / blockSpan
  ///   blockCount = endBlock - startBlock + 1  (clamped to [0, totalBlocks])
  static void convertElementSliceToBlockSlice(
      OpBuilder &builder, Location loc, ValueRange elementOffsets,
      ValueRange elementSizes, ValueRange blockSpans,
      ValueRange totalBlockCounts, SmallVectorImpl<Value> &blockOffsets,
      SmallVectorImpl<Value> &blockSizes);

  /// Overlay a normalized owner-space prefix onto an existing DB-space slice
  /// without collapsing the source DB rank. This keeps untouched owner slots at
  /// their current range (or full-range fallback) while replacing the leading
  /// normalized slots produced by convertElementSliceToBlockSlice().
  static void mergeNormalizedBlockSlice(
      OpBuilder &builder, Location loc, ValueRange existingOffsets,
      ValueRange existingSizes, ValueRange totalBlockCounts,
      ValueRange normalizedOffsets, ValueRange normalizedSizes,
      SmallVectorImpl<Value> &blockOffsets,
      SmallVectorImpl<Value> &blockSizes);

  ///===----------------------------------------------------------------------===///
  /// Block Size and Malloc Pattern Extraction
  ///===----------------------------------------------------------------------===///

  /// Extract block size from an explicit partition-size hint.
  /// Handles direct constants, minui/minsi patterns, addi halo patterns,
  /// and maxui clamp patterns with recursive descent up to depth 4.
  static std::optional<int64_t> extractBlockSizeFromHint(Value sizeHint,
                                                         int depth = 0);

  /// Merge two DbAccessPattern values, keeping the higher-priority pattern.
  /// Priority: stencil > indexed > uniform > unknown.
  static DbAccessPattern mergeAccessPattern(DbAccessPattern lhs,
                                            DbAccessPattern rhs) {
    auto rank = [](DbAccessPattern p) -> unsigned {
      switch (p) {
      case DbAccessPattern::stencil:
        return 3;
      case DbAccessPattern::indexed:
        return 2;
      case DbAccessPattern::uniform:
        return 1;
      case DbAccessPattern::unknown:
        return 0;
      }
      return 0;
    };
    return rank(rhs) > rank(lhs) ? rhs : lhs;
  }
};

} // namespace arts
} // namespace mlir

#endif // CARTS_UTILS_DBUTILS_H
