///==========================================================================///
/// File: DbBlockInfoComputer.h
///
/// Helper class extracted from DbAcquireNode that handles block offset/size
/// computation from loops, while-loops, and partition hints.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBBLOCKINFOCOMPUTER_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBBLOCKINFOCOMPUTER_H

#include "carts/dialect/arts/Analysis/loop/LoopNode.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace carts::arts {

class DbAcquireNode;

/// DbBlockInfoComputer -- computes block offset and size for partition
/// localization by analyzing for-loops, while-loops, and partition hints.
class DbBlockInfoComputer {
public:
  /// Compute block offset and size for the given acquire node.
  /// Tries hints first, then while-loops, then for-loops.
  static LogicalResult computeBlockInfo(DbAcquireNode *node, Value &blockOffset,
                                        Value &blockSize);

  /// Compute block info from an scf::WhileOp.
  static LogicalResult
  computeBlockInfoFromWhile(DbAcquireNode *node, scf::WhileOp whileOp,
                            Value &blockOffset, Value &blockSize,
                            Value *offsetForCheck = nullptr);

  /// Compute block info from partition hints (partitionOffset/partitionSize).
  static LogicalResult computeBlockInfoFromHints(DbAcquireNode *node,
                                                 Value &blockOffset,
                                                 Value &blockSize);

  /// Compute block info from for-like loops in the EDT body.
  static LogicalResult
  computeBlockInfoFromForLoop(DbAcquireNode *node, ArrayRef<LoopNode *> loops,
                              Value &blockOffset, Value &blockSize,
                              Value *offsetForCheck = nullptr);
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBBLOCKINFOCOMPUTER_H
