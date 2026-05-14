///==========================================================================///
/// File: DbBlockIndexer.h
///
/// Index localizer for a block datablock allocation.
///
/// This class does not choose a block policy. SDE owns that decision.  The
/// indexer only converts global element coordinates in a raw memref fallback
/// body into DB-space block coordinates plus in-block memref coordinates.
///
/// Example (A[100][50], blockSize=25, startBlock=1):
///   Block 1 covers rows 25..49.
///   Access A[27][10]:
///     blockIdx = 27 / 25 = 1
///     dbRefIdx = blockIdx - startBlock = 0
///     localRow = 27 % 25 = 2
///     memrefIdx = [2, 10]
///
/// Equations (div/mod localization):
///   dbRefIdx  = (globalRow / blockSize) - startBlock
///   memrefIdx = globalRow % blockSize
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_DB_BLOCK_DBBLOCKINDEXER_H
#define ARTS_DIALECT_CORE_TRANSFORMS_DB_BLOCK_DBBLOCKINDEXER_H

#include "arts/dialect/core/Transforms/db/DbIndexerBase.h"

namespace mlir {
namespace arts {

/// Index localizer for block datablock allocation.
/// Implements div/mod localization for coarse-grained parallelism.
/// Supports N-dimensional block partitioning where multiple leading dimensions
/// use div/mod localization.
///
/// Uses partitionInfo.sizes for block sizes. The startBlocks are computed
/// by the rewriter as: startBlock = partitionInfo.offsets / blockSize.
///
/// For each partitioned dimension d:
///   dbRefIdx[d]  = (globalIdx[d] / blockSize[d]) - startBlock[d]
///   memrefIdx[d] = globalIdx[d] % blockSize[d]
class DbBlockIndexer : public DbIndexerBase {
  /// Block size per partitioned dimension
  SmallVector<Value> blockSizes;
  /// Original dimension indices for each partitioned dimension
  SmallVector<unsigned> partitionedDims;
  /// Start block per partitioned dimension (computed)
  SmallVector<Value> startBlocks;
  /// True when allocation outer sizes are all constant 1
  bool allocSingleBlock = false;
  /// True when the acquired range is guaranteed to fit within a single block
  bool acquireSingleBlock = false;
  /// Optional zero constant defined in a dominating scope
  Value dominantZero;

public:
  /// Constructor with PartitionInfo - the canonical way to create indexers.
  /// Uses partitionInfo.sizes for blockSizes. The startBlocks must be passed
  /// separately as they require division which needs a builder.
  DbBlockIndexer(const PartitionInfo &info, ArrayRef<Value> startBlocks,
                 unsigned outerRank, unsigned innerRank,
                 bool allocSingleBlock = false, bool acquireSingleBlock = false,
                 Value dominantZero = Value());

  /// Number of partitioned dimensions
  unsigned numPartitionedDims() const {
    return !partitionedDims.empty() ? partitionedDims.size()
                                    : blockSizes.size();
  }

  //===--------------------------------------------------------------------===//
  /// Pure virtual overrides: mode-specific index localization
  //===--------------------------------------------------------------------===//

  /// Transform global multi-dimensional indices to local
  /// For block mode: dbRefIdx = global / blockSize - startBlock
  ///                 memrefIdx = global % blockSize
  LocalizedIndices localize(ArrayRef<Value> globalIndices, OpBuilder &builder,
                            Location loc) override;

  /// Transform a linearized global index (for flattened 1D memrefs)
  LocalizedIndices localizeLinearized(Value globalLinearIndex, Value stride,
                                      OpBuilder &builder,
                                      Location loc) override;

  //===--------------------------------------------------------------------===//
  /// Virtual hook overrides
  //===--------------------------------------------------------------------===//

  /// Block mode subindex handling: applies div/mod to the subindex index
  /// using blockSizes/startBlocks for proper block-relative addressing.
  bool handleSubIndexOp(Operation *op, Value dbPtr, Type elementType,
                        ArtsCodegen &AC,
                        llvm::SetVector<Operation *> &opsToRemove) override;

  //===--------------------------------------------------------------------===//
  /// Block-specific methods
  //===--------------------------------------------------------------------===//

  /// Mode-aware coordinate localization
  SmallVector<Value> localizeCoordinates(ArrayRef<Value> globalIndices,
                                         ArrayRef<Value> sliceOffsets,
                                         unsigned numIndexedDims,
                                         Type elementType, OpBuilder &builder,
                                         Location loc);
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_TRANSFORMS_DB_BLOCK_DBBLOCKINDEXER_H
