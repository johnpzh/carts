///==========================================================================///
/// File: DbNode.h
///
/// Db-specific nodes derived from NodeBase
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBNODE_H
#define ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBNODE_H

#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/graphs/base/NodeBase.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbAccessPattern.h"
#include "carts/utils/MetadataEnums.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>

/// Forward declarations for helper classes
namespace mlir {
namespace arts {
class PartitionBoundsAnalyzer;
class MemoryAccessClassifier;
class DbBlockInfoComputer;
} // namespace arts
} // namespace mlir

namespace mlir {
namespace arts {

/// Forward declarations
class DbAnalysis;
class DbAcquireNode;
class LoopNode;

/// Derived per-entry partition legality cache for one acquire.
///
/// This is graph-owned state. It captures what the compiler proved about one
/// depend-clause entry or inferred partition entry, not the final rewrite
/// payload and not the semantic contract boundary. DbPartitioning translates
/// these facts into pass-local summaries and then into planner/rewriter inputs
/// expressed in the canonical IR contract vocabulary.
struct DbPartitionEntryFact {
  PartitionInfo partition;
  Value representativeOffset;
  Value representativeSize;
  std::optional<unsigned> mappedDim;
  bool needsFullRange = false;
  bool preservesDistributedContract = false;
};

/// Per-dimension view of an acquire's partition behavior.
///
/// This stays intentionally small: enough for heuristics/controller logic to
/// know which dimensions are relevant, without turning DbAcquireNode into a
/// rewrite-plan object.
struct DbDimPartitionFact {
  unsigned dim = 0;
  DimAccessPatternType accessPattern = DimAccessPatternType::Unknown;
  bool isLeadingDynamic = false;
  bool selectedByPartitionEntry = false;
  bool needsFullRange = false;
};

/// Derived per-acquire partition facts cached on DbAcquireNode.
///
/// Ownership:
///   - produced and cached by graph-side analysis
///   - exposed through DbAnalysis
///   - consumed by controller passes such as DbPartitioning as supporting
///     evidence
///   - recomputable from IR/analysis state
///   - never used directly as the final rewrite contract or semantic authority
struct DbAcquirePartitionFacts {
  DbAcquireOp acquire;
  PartitionMode requestedMode = PartitionMode::coarse;
  ArtsDepPattern depPattern = ArtsDepPattern::unknown;
  AccessPattern accessPattern = AccessPattern::Unknown;
  SmallVector<DbPartitionEntryFact, 2> entries;
  SmallVector<DbDimPartitionFact, 4> dims;
  SmallVector<unsigned, 4> partitionDims;
  SmallVector<unsigned, 4> stencilOwnerDims;
  bool hasIndirectAccess = false;
  bool hasDirectAccess = false;
  bool hasDistributionContract = false;
  bool partitionDimsFromPeers = false;
  bool explicitCoarseRequest = false;
  bool hasBlockHints = false;
  bool inferredBlock = false;
  bool supportedBlockHalo = false;

  /// Query helpers over entries.
  bool hasFineGrainedEntries() const {
    return llvm::any_of(entries, [](const DbPartitionEntryFact &entry) {
      return entry.partition.isFineGrained() &&
             !entry.partition.indices.empty();
    });
  }
  bool hasUnmappedPartitionEntry() const {
    return llvm::any_of(entries, [](const DbPartitionEntryFact &entry) {
      return entry.representativeOffset && !entry.mappedDim;
    });
  }
  bool hasDistributedContractEntries() const {
    return llvm::any_of(entries, [](const DbPartitionEntryFact &entry) {
      return entry.preservesDistributedContract;
    });
  }

  /// Return a unique mapped owner dimension when derived graph facts already
  /// imply one, even if multi-entry acquires could not populate
  /// `partitionDims` directly.
  std::optional<unsigned> inferSingleMappedDim() const {
    if (partitionDims.size() == 1)
      return partitionDims.front();
    if (partitionDims.size() > 1)
      return std::nullopt;

    std::optional<unsigned> inferredDim;
    auto recordDim = [&](unsigned dim) -> bool {
      if (!inferredDim) {
        inferredDim = dim;
        return true;
      }
      return *inferredDim == dim;
    };

    for (const DbDimPartitionFact &dimFact : dims) {
      if (!dimFact.selectedByPartitionEntry)
        continue;
      if (!recordDim(dimFact.dim))
        return std::nullopt;
    }
    if (inferredDim)
      return inferredDim;

    for (const DbPartitionEntryFact &entry : entries) {
      if (!entry.mappedDim)
        continue;
      if (!recordDim(*entry.mappedDim))
        return std::nullopt;
    }
    if (inferredDim)
      return inferredDim;

    if (dims.size() == 1)
      return dims.front().dim;

    if (stencilOwnerDims.size() == 1)
      return stencilOwnerDims.front();
    return std::nullopt;
  }
};

///===----------------------------------------------------------------------===///
/// DbAllocNode
/// Represents a `arts.db.alloc` operation in the DB graph.
///===----------------------------------------------------------------------===///
class DbAllocNode : public NodeBase {
public:
  DbAllocNode(DbAllocOp op, DbAnalysis *analysis);

  StringRef getHierId() const override { return hierId; }
  void setHierId(std::string id) { hierId = id; }
  void print(llvm::raw_ostream &os) const override;
  Operation *getOp() const override { return op; }

  void addInEdge(EdgeBase *edge) override { inEdges.insert(edge); }
  void addOutEdge(EdgeBase *edge) override { outEdges.insert(edge); }
  const DenseSet<EdgeBase *> &getInEdges() const override { return inEdges; }
  const DenseSet<EdgeBase *> &getOutEdges() const override { return outEdges; }

  /// Graph structure management
  DbAcquireNode *getOrCreateAcquireNode(DbAcquireOp op);
  DbAcquireNode *findAcquireNode(DbAcquireOp op) const;
  SmallVector<DbAcquireNode *, 16> collectAllAcquireNodes();
  SmallVector<const DbAcquireNode *, 16> collectAllAcquireNodes() const;
  void forEachChildNode(const std::function<void(NodeBase *)> &fn) const;
  size_t getAcquireNodesSize() const { return acquireNodes.size(); }

  /// Operation accessors
  DbAllocOp getDbAllocOp() const { return dbAllocOp; }
  bool isStringDatablock() const { return isStringBacked; }
  NodeKind getKind() const override { return NodeKind::DbAlloc; }
  static bool classof(const NodeBase *N) {
    return N->getKind() == NodeKind::DbAlloc;
  }

  /// Utility methods
  bool isParallelFriendly() const;
  bool canBePartitioned();
  bool canProveNonOverlapping() const;

  /// Patterns summary
  AcquirePatternSummary summarizeAcquirePatterns() const;

  /// Analysis metadata
  uint64_t inCount = 0, outCount = 0;
  uint64_t beginIndex = 0, endIndex = 0;
  uint64_t maxLoopDepth = 0;
  unsigned long long totalAccessBytes = 0;
  uint64_t numAcquires = 0;
  bool isStringBacked = false;

  /// Metadata fields (previously inherited from MemrefMetadata)
  std::optional<int64_t> rank;
  std::optional<bool> allAccessesAffine, hasAffineAccesses,
      hasNonAffineAccesses;
  std::optional<int64_t> memoryFootprint;
  std::optional<bool> isFlattenedArrayFlag;
  std::optional<bool> hasUniformAccess, hasStrideOneAccess;
  std::optional<AccessPatternType> dominantAccessPattern;
  std::optional<bool> accessedInParallelLoop, hasLoopCarriedDeps;
  std::optional<TemporalLocalityLevel> temporalLocality;
  std::optional<SpatialLocalityLevel> spatialLocality;
  std::optional<int64_t> estimatedAccessBytes;
  SmallVector<DimAccessPatternType> dimAccessPatterns;

  /// Twin-diff overlap analysis
  bool hasSingleWriter() const;
  bool hasDynamicWriterOffsets() const;
  bool allAcquiresWorkerIndexed() const;

private:
  DbAllocOp dbAllocOp;
  Operation *op;
  DbAnalysis *analysis;
  std::string hierId;
  unsigned nextChildId = 1;

  /// Graph structure: children acquire nodes
  SmallVector<std::unique_ptr<DbAcquireNode>, 4> acquireNodes;
  DenseMap<DbAcquireOp, DbAcquireNode *> acquireMap;
};

///===----------------------------------------------------------------------===///
/// DbAcquireNode
/// Represents a `arts.db.acquire` operation in the DB graph.
///
/// DbAcquireNode caches per-acquire DB facts derived from IR and analysis:
///   - memory-access classification
///   - partition legality / mapped dimensions
///   - cached block/stencil helper results
///
/// Passes should prefer reaching this state through DbAnalysis instead of
/// rebuilding it ad hoc from IR. These caches support refinement and legality
/// checks, but `arts.lowering_contract` remains the semantic contract that
/// downstream lowering is expected to consume.
///===----------------------------------------------------------------------===///
class DbAcquireNode : public NodeBase {
public:
  DbAcquireNode(DbAcquireOp op, NodeBase *parent, DbAllocNode *rootAlloc,
                DbAnalysis *analysis, std::string initialHierId = "");

  StringRef getHierId() const override { return hierId; }
  void setHierId(std::string id) { hierId = id; }
  void print(llvm::raw_ostream &os) const override;
  Operation *getOp() const override { return op; }

  void addInEdge(EdgeBase *edge) override { inEdges.insert(edge); }
  void addOutEdge(EdgeBase *edge) override { outEdges.insert(edge); }
  const DenseSet<EdgeBase *> &getInEdges() const override { return inEdges; }
  const DenseSet<EdgeBase *> &getOutEdges() const override { return outEdges; }

  DbAllocNode *getParent() const { return rootAlloc; }
  NodeBase *getDirectParent() const { return parent; }
  DbAllocNode *getRootAlloc() const { return rootAlloc; }
  DbAnalysis *getAnalysis() const { return analysis; }

  NodeKind getKind() const override { return NodeKind::DbAcquire; }
  static bool classof(const NodeBase *N) {
    return N->getKind() == NodeKind::DbAcquire;
  }

  EdtOp getEdtUser() const { return EdtOp(edtUserOp); }
  Value getUseInEdt() const { return useInEdt; }

  /// Collect access operations mapping DbRefOp to its memory operations.

  void collectAccessOperations(
      DenseMap<DbRefOp, SetVector<Operation *>> &dbRefToMemOps);

  /// Helper methods for querying memory access.
  void
  forEachMemoryAccess(llvm::function_ref<void(Operation *, bool)> callback);

  bool hasLoads();
  bool hasStores();
  bool isReadOnlyAccess();
  bool isWriterAccess();
  bool hasMemoryAccesses();
  size_t countLoads();
  size_t countStores();
  bool hasIndirectAccess() const;
  bool hasDirectAccess() const;

  DbAcquireOp getDbAcquireOp() const { return dbAcquireOp; }
  DbReleaseOp getDbReleaseOp() const { return dbReleaseOp; }

  DbAcquireNode *getOrCreateAcquireNode(DbAcquireOp op);
  void forEachChildNode(const std::function<void(NodeBase *)> &fn) const;

  /// Utility methods for partitioning
  bool canBePartitioned();
  bool hasValidEdtAndAccesses();

  bool computePartitionBounds();
  bool canPartitionWithOffset(Value offset);
  /// Return the access-chain dimension that depends on the partition offset.
  /// If requireLeading is true, only accept leading dynamic index.
  std::optional<unsigned> getPartitionOffsetDim(Value offset,
                                                bool requireLeading = false);

  /// Return the lazily-computed derived partition-facts cache for this acquire.
  const DbAcquirePartitionFacts &getPartitionFacts() const;
  void invalidatePartitionFacts() const;

  LogicalResult computeBlockInfo(Value &blockOffset, Value &blockSize);
  LogicalResult computeBlockInfoFromWhile(scf::WhileOp whileOp,
                                          Value &blockOffset, Value &blockSize,
                                          Value *offsetForCheck = nullptr);
  LogicalResult computeBlockInfoFromHints(Value &blockOffset, Value &blockSize);
  LogicalResult computeBlockInfoFromForLoop(ArrayRef<LoopNode *> loopNodes,
                                            Value &blockOffset,
                                            Value &blockSize,
                                            Value *offsetForCheck = nullptr);

  /// Get stencil bounds for this acquire (computed during canBePartitioned)
  const std::optional<StencilBounds> &getStencilBounds() const {
    return stencilBounds;
  }

  /// Classify this acquire's access pattern using current analysis state.
  AccessPattern getAccessPattern() const;

  void setPartitionInfo(Value offset, Value size) {
    partitionOffset = offset;
    partitionSize = size;
    invalidatePartitionFacts();
  }

  std::pair<Value, Value> getPartitionInfo() const {
    return {partitionOffset, partitionSize};
  }

  const std::optional<std::pair<Value, Value>> &getOriginalBounds() const {
    return originalBounds;
  }

  /// Check if this acquire uses worker-indexed pattern: offsets[%workerId] with
  /// sizes[1]. This pattern is inherently disjoint across workers.
  bool isWorkerIndexedAccess() const;

  /// Check if any access index within this acquire depends on the given value.
  /// Used to validate that partition indices match actual memory accesses.
  bool accessIndexDependsOn(Value idx);

  /// Validate that element-wise partition indices match actual accesses.
  /// Returns false if this is a block-wise pattern (indices are block corners,
  /// but accesses span a range) - indicating we should fall back to block mode.
  bool validateElementWisePartitioning();

  /// Internal state accessors used by extracted helper classes.
  /// These are not part of the public API and should not be called directly.
  Operation *getEdtUserOp() const { return edtUserOp; }
  void setStencilBoundsInternal(std::optional<StencilBounds> sb) {
    stencilBounds = std::move(sb);
    invalidatePartitionFacts();
  }
  void setComputedBlockInfo(std::optional<std::pair<Value, Value>> info) {
    computedBlockInfo = std::move(info);
    invalidatePartitionFacts();
  }
  std::optional<std::pair<Value, Value>> getComputedBlockInfo() const {
    return computedBlockInfo;
  }
  void setOriginalBounds(std::optional<std::pair<Value, Value>> bounds) {
    originalBounds = std::move(bounds);
    invalidatePartitionFacts();
  }
  void setHasNonConstantOffset(bool v) {
    hasNonConstantOffset = v;
    invalidatePartitionFacts();
  }
  bool getHasNonConstantOffset() const { return hasNonConstantOffset; }

private:
  friend class PartitionBoundsAnalyzer;
  friend class MemoryAccessClassifier;
  friend class DbBlockInfoComputer;

  DbAcquireOp dbAcquireOp;
  DbReleaseOp dbReleaseOp{nullptr};
  Operation *op = nullptr;
  NodeBase *parent = nullptr;
  DbAllocNode *rootAlloc = nullptr;
  DbAnalysis *analysis = nullptr;
  std::string hierId;
  Operation *edtUserOp = nullptr;
  Value useInEdt = nullptr;
  Value partitionOffset, partitionSize;

  /// Cached adjusted block info
  std::optional<std::pair<Value, Value>> computedBlockInfo;
  std::optional<std::pair<Value, Value>> originalBounds;
  std::optional<StencilBounds> stencilBounds;
  bool hasNonConstantOffset = false;
  mutable std::optional<AccessPattern> accessPattern;
  mutable std::optional<DbAcquirePartitionFacts> partitionFacts;

public:
  uint64_t inCount = 0, outCount = 0;
  uint64_t beginIndex = 0, endIndex = 0;
  unsigned long long estimatedBytes = 0;

  /// Nested acquire storage
  unsigned nextChildId = 1;
  SmallVector<std::unique_ptr<DbAcquireNode>, 4> childAcquires;
  DenseMap<DbAcquireOp, DbAcquireNode *> childMap;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_GRAPHS_DB_DBNODE_H
