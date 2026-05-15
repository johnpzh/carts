///==========================================================================///
/// File: MemoryAccessClassifier.h
///
/// Helper class extracted from DbAcquireNode that classifies memory access
/// patterns: load/store detection, indirect/direct access analysis, and
/// access operation collection.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_MEMORYACCESSCLASSIFIER_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_MEMORYACCESSCLASSIFIER_H

#include "carts/Dialect.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include <cstddef>

namespace mlir {
namespace arts {

class DbAcquireNode;

/// MemoryAccessClassifier -- analyzes memory access patterns for a
/// DbAcquireNode. Methods operate on the acquire's EDT body, walking
/// from the block argument through DbRefOps to load/store operations.
class MemoryAccessClassifier {
public:
  /// Collect DbRefOp -> memory-operation mappings reachable from the
  /// acquire's block argument inside its EDT user.
  static void
  collectAccessOperations(DbAcquireNode *node,
                          DenseMap<DbRefOp, SetVector<Operation *>> &result);

  /// Iterate over every memory access, invoking callback(op, isStore).
  static void
  forEachMemoryAccess(DbAcquireNode *node,
                      llvm::function_ref<void(Operation *, bool)> callback);

  /// Query helpers
  static bool hasLoads(DbAcquireNode *node);
  static bool hasStores(DbAcquireNode *node);
  static bool isReadOnlyAccess(DbAcquireNode *node);
  static bool isWriterAccess(DbAcquireNode *node);
  static bool hasMemoryAccesses(DbAcquireNode *node);
  static size_t countLoads(DbAcquireNode *node);
  static size_t countStores(DbAcquireNode *node);

  /// Indirect/direct access detection.
  /// An indirect access is one whose index depends on a memory load
  /// (data-dependent addressing).
  static bool hasIndirectAccess(DbAcquireNode *node);
  static bool hasDirectAccess(DbAcquireNode *node);
};

/// Detect indirect (data-dependent) indices derived from memory loads.
/// Shared between MemoryAccessClassifier and PartitionBoundsAnalyzer.
bool isIndirectIndex(Value idx, Value partitionOffset, int depth = 0);

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_MEMORYACCESSCLASSIFIER_H
