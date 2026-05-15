///==========================================================================///
/// File: DbAnalysis.h
///
/// This file defines the public DB analysis facade.
///
/// Responsibility split:
///   - `arts.lowering_contract` and typed DB IR attributes are the canonical
///     semantic contract.
///   - DbGraph / DbNode cache derived graph summaries and legality evidence
///     projected from that IR.
///   - DbAnalysis exposes pass-facing queries that reconcile the canonical IR
///     contract with those derived caches.
///   - Controller passes should ask DbAnalysis first instead of rebuilding DB
///     reasoning directly from raw IR or raw node methods.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_DB_DBANALYSIS_H
#define ARTS_DIALECT_CORE_ANALYSIS_DB_DBANALYSIS_H

#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/Analysis.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbAccessPattern.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/dialect/arts/Utils/LoweringContractUtils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>
#include <shared_mutex>

namespace mlir {
namespace arts {

class DbGraph;
class DbAllocNode;
class DbAcquireNode;
class DbAliasAnalysis;
class LoopAnalysis;

class DbAnalysis : public ArtsAnalysis {
public:
  /// Pass-facing summary used by DbPartitioning while assembling heuristic and
  /// planner inputs. This is a transient query result synthesized from the
  /// canonical IR contract plus graph-derived caches, not a second authority.
  struct AcquirePartitionSummary {
    PartitionMode mode = PartitionMode::coarse;
    SmallVector<Value> partitionOffsets;
    SmallVector<Value> partitionSizes;
    SmallVector<unsigned> partitionDims;
    bool isValid = false;
    bool hasIndirectAccess = false;
    bool hasDistributionContract = false;
    bool partitionDimsFromPeers = false;
  };

  /// Canonical pass-facing contract view for an acquire.
  /// This keeps the same contract vocabulary used by lowering, while letting
  /// DbAnalysis strengthen incomplete IR contracts with graph-derived evidence
  /// after DB creation. Pre-DB pattern passes seed the contract; post-DB
  /// analysis may refine it, but should not invent a separate contract
  /// language or ask downstream consumers to reason over graph-only facts.
  struct AcquireContractSummary {
    LoweringContractInfo contract;
    bool refinedByDbAnalysis = false;
    /// Canonical access-pattern view materialized from `contract`.
    AccessPattern accessPattern = AccessPattern::Unknown;
    /// Graph-derived classifier evidence kept separate from the canonical
    /// contract-facing pattern above.
    AccessPattern derivedAccessPattern = AccessPattern::Unknown;
    /// Materialized pass-facing facts derived from the graph cache.
    /// The semantic authority still remains `contract`.
    SmallVector<unsigned, 4> partitionDims;
    bool derivedFactEvidence = false;
    bool indirectAccess = false;
    bool directAccess = false;
    bool blockHints = false;
    bool inferredBlockCapability = false;
    bool fineGrainedEntries = false;
    bool unmappedPartitionEntry = false;
    bool distributedContractEntry = false;
    bool distributionContract = false;
    bool partitionDimsFromPeersFlag = false;

    bool hasIndirectAccess() const { return indirectAccess; }
    bool hasDirectAccess() const { return directAccess; }
    bool hasBlockHints() const { return blockHints; }
    bool inferredBlock() const { return inferredBlockCapability; }
    bool hasFineGrainedEntries() const { return fineGrainedEntries; }
    bool hasUnmappedPartitionEntry() const { return unmappedPartitionEntry; }
    bool preservesDistributedContractEntry() const {
      return distributedContractEntry;
    }
    bool hasDistributionContract() const { return distributionContract; }
    bool partitionDimsFromPeers() const { return partitionDimsFromPeersFlag; }
    AccessPattern getDerivedAccessPattern() const {
      return derivedAccessPattern;
    }

    bool empty() const {
      return contract.empty() && accessPattern == AccessPattern::Unknown &&
             partitionDims.empty() && !derivedFactEvidence && !indirectAccess &&
             !directAccess && !blockHints && !inferredBlockCapability &&
             !fineGrainedEntries && !unmappedPartitionEntry &&
             !distributedContractEntry && !distributionContract &&
             !partitionDimsFromPeersFlag;
    }
    bool usesStencilSemantics() const {
      return contract.isStencilFamily() || contract.usesStencilDistribution() ||
             contract.supportsBlockHalo();
    }
    bool hasExplicitStencilLayout() const {
      return contract.hasExplicitStencilContract();
    }
    PartitionMode preferredBlockPartitionMode() const {
      return usesStencilSemantics() ? PartitionMode::stencil
                                    : PartitionMode::block;
    }
    bool supportsContractDrivenBlockCapability() const {
      return hasBlockHints() || inferredBlock() || usesStencilSemantics();
    }
    bool prefersSemanticOwnerLayoutPreservation() const {
      return contract.prefersSemanticOwnerLayoutPreservation();
    }
    bool prefersNDBlock(unsigned requiredRank = 2) const {
      return contract.prefersNDBlock(requiredRank);
    }
    bool usesWavefrontStencilContract() const {
      return contract.isWavefrontStencilContract();
    }
    bool usesMatmulSemantics() const {
      return contract.getEffectiveKind() == ContractKind::Matmul;
    }
  };

  DbAnalysis(AnalysisManager &AM);

  ~DbAnalysis();

  /// Get or create DbGraph for a function, building if needed.
  DbGraph &getOrCreateGraph(func::FuncOp func);

  /// Invalidate graph for a function.
  bool invalidateGraph(func::FuncOp func);

  /// Invalidate all graphs
  void invalidate() override;

  /// Print analysis for a function.
  void print(func::FuncOp func);

  /// Access analyses
  DbAliasAnalysis *getAliasAnalysis() { return dbAliasAnalysis.get(); }
  LoopAnalysis *getLoopAnalysis();
  AcquirePartitionSummary
  analyzeAcquirePartition(DbAcquireOp acquire, OpBuilder &builder,
                          const AcquireContractSummary *summary = nullptr);
  /// IR-only canonical acquire contract view.
  /// This does not consult graph-derived partition facts, so it is safe to use
  /// while graph-side analyses are still constructing those facts.
  static AcquireContractSummary
  buildCanonicalAcquireContractSummary(DbAcquireOp acquire);
  std::optional<AcquireContractSummary>
  getAcquireContractSummary(DbAcquireOp acquire);
  AccessPattern resolveCanonicalAcquireAccessPattern(
      DbAcquireOp acquire, const AcquireContractSummary *summary = nullptr);
  ArtsDepPattern resolveCanonicalAcquireDepPattern(
      DbAcquireOp acquire, const AcquireContractSummary *summary = nullptr);
  bool hasCanonicalAcquireStencilSemantics(
      DbAcquireOp acquire, const AcquireContractSummary *summary = nullptr);
  ArtsMode inferEdtAccessMode(Operation *underlyingOp, EdtOp edt) const;
  static ArtsMode classifyMemrefUserAccessMode(Operation *op,
                                               Operation *underlyingOp);
  static bool opMatchesAccessMode(Operation *op, Operation *underlyingOp,
                                  ArtsMode requestedMode);
  static bool accessModeCanSeedNestedAcquire(ArtsMode availableMode,
                                             ArtsMode requestedMode);
  bool operationHasDistributedDbContract(Operation *op);
  bool operationHasPeerInferredPartitionDims(Operation *op);

  /// Return true when producerEdt writes DBs that are later consumed outside
  /// internode EDT flow.
  bool hasNonInternodeConsumerForWrittenDb(EdtOp producerEdt);

  /// Return true when operations a and b have conflicting DB accesses:
  /// both access the same underlying DB allocation and at least one is a
  /// writer. This is a semantic query about DB relationships (conflict
  /// detection).
  static bool hasDbConflict(Operation *a, Operation *b);

  /// Query DB access patterns through the DB graph interface.
  std::optional<AccessPattern> getAcquireAccessPattern(DbAcquireOp acquire);
  std::optional<DbAccessPattern> getAllocAccessPattern(DbAllocOp alloc);

  /// Convenience: get DB alloc node by op (derives parent func internally).
  DbAllocNode *getDbAllocNode(DbAllocOp alloc);
  /// Convenience: get DB acquire node by op (derives parent func internally).
  DbAcquireNode *getDbAcquireNode(DbAcquireOp acquire);

  using ArtsAnalysis::getAnalysisManager;

  ///===----------------------------------------------------------------------===///
  /// Partition / Granularity Queries (static)
  ///===----------------------------------------------------------------------===///

  /// Get partition mode from DbAcquireOp structure.
  static PartitionMode getPartitionModeFromStructure(DbAcquireOp acquire);

  /// Get partition mode from DbAllocOp structure.
  static PartitionMode getPartitionModeFromStructure(DbAllocOp alloc);

  /// Check if acquire is in block partition mode.
  static bool isBlock(DbAcquireOp acquire);

  /// Check if acquire is in element-wise (fine_grained) partition mode.
  static bool isElementWise(DbAcquireOp acquire);

  /// Check if acquire is in coarse partition mode.
  static bool isCoarse(DbAcquireOp acquire);

  /// Check if allocation is coarse-grained (all sizes == 1).
  static bool isCoarseGrained(DbAllocOp alloc);

  /// Check if allocation is fine-grained (partitioned into multiple DBs).
  static bool isFineGrained(DbAllocOp alloc);

  /// Check if a datablock operation has a single size of 1.
  static bool hasSingleSize(Operation *dbOp);

  ///===----------------------------------------------------------------------===///
  /// Semantic Queries (static)
  ///===----------------------------------------------------------------------===///

  /// Conservative memory-object identity for memref-like values.
  static bool isSameMemoryObject(Value lhsMemref, Value rhsMemref);

  /// Check if a DbAcquireOp has static (constant) offset and size hints.
  static bool hasStaticHints(DbAcquireOp acqOp);

  /// Check if a multi-entry acquire has a stencil access pattern.
  static bool hasMultiEntryStencilPattern(DbAcquireOp acquire,
                                          int64_t &minOffset,
                                          int64_t &maxOffset);

  /// Try to extract a constant offset between two index values.
  static std::optional<int64_t> getConstantOffsetBetween(Value idx, Value base);

  /// Returns true if the acquire belongs to an EDT with tiling_2d distribution.
  static bool isTiling2DTaskAcquire(DbAcquireOp acquire);

private:
  llvm::DenseMap<func::FuncOp, std::unique_ptr<DbGraph>> functionGraphMap;
  mutable std::shared_mutex graphMutex;
  std::unique_ptr<DbAliasAnalysis> dbAliasAnalysis;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_DB_DBANALYSIS_H
