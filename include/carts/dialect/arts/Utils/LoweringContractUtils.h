#ifndef CARTS_UTILS_LOWERINGCONTRACTUTILS_H
#define CARTS_UTILS_LOWERINGCONTRACTUTILS_H

#include "carts/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace arts {

/// Discriminator for the high-level computational pattern a lowering contract
/// represents.  When the field is `Unknown` on a `LoweringContractInfo`,
/// `getEffectiveKind()` will attempt to derive the kind from `depPattern`.
enum class ContractChange { Unchanged, Changed };

enum class ContractKind : int64_t {
  Unknown = 0,
  Stencil = 1,
  Elementwise = 2,
  Matmul = 3,
  Triangular = 4
};

struct SemanticPattern {
  ContractKind kind = ContractKind::Unknown;
  std::optional<ArtsDepPattern> depPattern;
  std::optional<EdtDistributionKind> distributionKind;
  std::optional<EdtDistributionPattern> distributionPattern;
  std::optional<int64_t> distributionVersion;
  std::optional<int64_t> revision;

  bool hasDistributionContract() const {
    return kind != ContractKind::Unknown || depPattern || distributionKind ||
           distributionPattern || distributionVersion || revision;
  }
};

struct SpatialLayout {
  SmallVector<int64_t, 4> ownerDims;
  SmallVector<Value, 4> blockShape;
  SmallVector<Value, 4> minOffsets;
  SmallVector<Value, 4> maxOffsets;
  SmallVector<Value, 4> writeFootprint;
  SmallVector<int64_t, 4> staticBlockShape;
  SmallVector<int64_t, 4> staticMinOffsets;
  SmallVector<int64_t, 4> staticMaxOffsets;
  SmallVector<int64_t, 4> spatialDims;
  SmallVector<int64_t, 4> stencilIndependentDims;
  std::optional<int64_t> centerOffset;
  bool supportedBlockHalo = false;

  bool empty() const {
    return ownerDims.empty() && blockShape.empty() && minOffsets.empty() &&
           maxOffsets.empty() && writeFootprint.empty() &&
           spatialDims.empty() && staticBlockShape.empty() &&
           staticMinOffsets.empty() && staticMaxOffsets.empty() &&
           !supportedBlockHalo && stencilIndependentDims.empty() &&
           !centerOffset;
  }
};

struct AnalysisRefinement {
  bool narrowableDep = false;
  bool postDbRefined = false;
  std::optional<int64_t> criticalPathDistance;

  bool empty() const { return !postDbRefined && !criticalPathDistance; }
};

struct LoweringContractInfo {
  SemanticPattern pattern;
  SpatialLayout spatial;
  AnalysisRefinement analysis;

  bool hasDistributionContract() const {
    return pattern.hasDistributionContract();
  }

  bool hasSemanticContract() const {
    return hasDistributionContract() || analysis.narrowableDep;
  }

  bool hasSpatialContract() const { return !spatial.empty(); }

  bool hasAnalysisRefinements() const { return !analysis.empty(); }

  bool empty() const {
    return !hasSemanticContract() && !hasSpatialContract() &&
           !hasAnalysisRefinements();
  }

  /// Resolve the effective ContractKind: returns `kind` when explicitly set,
  /// otherwise derives from `depPattern`.
  ContractKind getEffectiveKind() const;

  bool isStencilFamily() const {
    return getEffectiveKind() == ContractKind::Stencil;
  }

  bool hasOwnerDims() const { return !spatial.ownerDims.empty(); }
  bool hasExplicitStencilContract() const;
  bool usesStencilDistribution() const;
  bool supportsBlockHalo() const;
  std::optional<EdtDistributionPattern> getEffectiveDistributionPattern() const;
  bool isWavefrontFamily() const;
  bool allowsDbAlignedChunking() const;
  bool shouldHonorLoopBlockHintForDbAlignment() const;
  bool prefersWideTiling2DColumns() const;
  bool shouldReuseEnclosingEpoch() const;
  bool prefersSemanticOwnerLayoutPreservation() const;
  bool isWavefrontStencilContract() const;
  bool prefersNDBlock(unsigned requiredRank = 2) const;
  std::optional<SmallVector<int64_t, 4>> getStaticBlockShape() const;
  std::optional<SmallVector<int64_t, 4>> getStaticMinOffsets() const;
  std::optional<SmallVector<int64_t, 4>> getStaticMaxOffsets() const;
};

LoweringContractOp getLoweringContractOp(Value target);
std::optional<LoweringContractInfo> getLoweringContract(Value target);
std::optional<LoweringContractInfo> getSemanticContract(Operation *op);
LoweringContractInfo resolveLoopDistributionContract(Operation *op);
std::optional<LoweringContractInfo>
getLoweringContract(Operation *op, OpBuilder &builder, Location loc);
ContractChange mergeLoweringContractInfo(LoweringContractInfo &dest,
                                         const LoweringContractInfo &src);
void normalizeLoweringContractInfo(LoweringContractInfo &info);
std::optional<SmallVector<int64_t, 4>>
projectOwnerIndexedStaticValues(ArrayRef<int64_t> values,
                                ArrayRef<int64_t> sourceOwnerDims,
                                ArrayRef<int64_t> targetOwnerDims);
std::optional<unsigned>
getContractOwnerPosition(const LoweringContractInfo &info,
                         unsigned physicalDim);
SmallVector<unsigned, 4>
resolveContractOwnerDims(const LoweringContractInfo &info, unsigned rank);
bool prefersContractNDBlock(const LoweringContractInfo &info,
                            unsigned requiredRank = 2);

/// Resolve the effective lowering contract for a value, combining
/// any explicit LoweringContractOp annotations with semantic operation
/// attributes. Returns nullopt if no contract information is available.
std::optional<LoweringContractInfo> resolveEffectiveContract(Value target);
std::optional<LoweringContractInfo> resolveEffectiveContract(Operation *op);

/// Monotone ordered combine of two contracts. The dest contract is
/// updated with information from src. Returns Changed if any field
/// was modified.
ContractChange combineContracts(LoweringContractInfo &dest,
                                const LoweringContractInfo &src);

/// Patch specific fields of a contract with deliberate overrides.
/// Unlike combineContracts (which preserves existing values),
/// this unconditionally writes the provided fields.
void patchContract(LoweringContractInfo &contract,
                   ArrayRef<int64_t> ownerDims = {},
                   ArrayRef<int64_t> minOffsets = {},
                   ArrayRef<int64_t> maxOffsets = {});

/// Extract the halo window (min/max offsets) from a contract.
/// Returns nullopt if the contract has no offset information.
std::optional<std::pair<SmallVector<int64_t, 4>, SmallVector<int64_t, 4>>>
projectHaloWindow(const LoweringContractInfo &contract);

/// Check whether a contract has complete halo state (ownerDims +
/// min/max offsets), meaning it doesn't need graph-backed fallbacks.
bool hasCompleteHaloState(const LoweringContractInfo &contract);

/// Resolve the effective lowering contract for an acquire operation,
/// combining pointer contract + alloc contract + semantic annotations.
std::optional<LoweringContractInfo> resolveAcquireContract(DbAcquireOp acquire);

/// Check if an acquire should apply stencil halo extension to worker-local
/// read slices. Returns true when mode=in and contract supports block halo.
bool shouldApplyStencilHalo(const LoweringContractInfo &contract,
                            ArtsMode effectiveMode);
bool shouldApplyStencilHalo(const LoweringContractInfo &contract,
                            DbAcquireOp acquire);

/// Check if an acquire should use partition_offsets/partition_sizes as the
/// dependency window instead of offsets/sizes. Returns true for stencil-mode
/// read acquires with explicit stencil contracts or wavefront inout acquires.
bool shouldUsePartitionSliceAsDepWindow(const LoweringContractInfo &contract,
                                        ArtsMode effectiveMode,
                                        PartitionMode partitionMode);
bool shouldUsePartitionSliceAsDepWindow(const LoweringContractInfo &contract,
                                        DbAcquireOp acquire);

/// Check if an acquire should preserve the parent dependency range instead
/// of using the worker-local partition slice. Returns true for read acquires
/// without explicit stencil contracts and without narrowable_dep annotation.
bool shouldPreserveParentDepRange(const LoweringContractInfo &contract,
                                  ArtsMode effectiveMode);
bool shouldPreserveParentDepRange(const LoweringContractInfo &contract,
                                  DbAcquireOp acquire);

LoweringContractOp upsertLoweringContract(OpBuilder &builder, Location loc,
                                          Value target,
                                          const LoweringContractInfo &info);
void eraseLoweringContracts(Value target);
void transferOperationContract(Operation *source, Operation *target);
void transferLoweringContract(Operation *source, Value target,
                              OpBuilder &builder, Location loc);
void transferValueContract(Value source, Value target, OpBuilder &builder,
                           Location loc);
void moveValueContract(Value source, Value target, OpBuilder &builder,
                       Location loc);
void transferContract(Operation *sourceOp, Operation *targetOp,
                      Value sourceContractTarget, Value targetContractTarget,
                      OpBuilder &builder, Location loc);

} // namespace arts
} // namespace mlir

#endif // CARTS_UTILS_LOWERINGCONTRACTUTILS_H
