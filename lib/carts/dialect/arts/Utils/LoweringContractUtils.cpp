///==========================================================================///
/// File: LoweringContractUtils.cpp
///
/// Utilities for first-class lowering contracts attached to DB pointer values.
/// These helpers let pattern detection passes emit one contract and let
/// lowering/partitioning passes consume it without rediscovering benchmark-
/// specific semantics from scattered attrs.
///==========================================================================///

#include "carts/dialect/arts/Utils/LoweringContractUtils.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/StencilAttributes.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::arts;

/// Forward declarations for internal-only helpers.
static ContractChange
mergeLoweringContractInfo(LoweringContractInfo &dest,
                          const LoweringContractInfo &src);
static void copyLoweringContract(Value source, Value target, OpBuilder &builder,
                                 Location loc);

namespace {

static SmallVector<Value, 4> materializeIndexValues(OpBuilder &builder,
                                                    Location loc,
                                                    ArrayRef<int64_t> values) {
  SmallVector<Value, 4> result;
  result.reserve(values.size());
  for (int64_t value : values)
    result.push_back(arts::createConstantIndex(builder, loc, value));
  return result;
}

static std::optional<SmallVector<int64_t, 4>>
readConstantIndexValues(ValueRange values) {
  SmallVector<int64_t, 4> result;
  result.reserve(values.size());
  for (Value value : values) {
    int64_t constant = 0;
    if (!value || !ValueAnalysis::getConstantIndex(value, constant))
      return std::nullopt;
    result.push_back(constant);
  }
  return result;
}

static void mergePlanSpatialAttrs(Operation *op, LoweringContractInfo &info) {
  if (!op || !info.hasDistributionContract())
    return;

  if (info.spatial.ownerDims.empty())
    if (auto ownerDims = readI64ArrayAttr(getPlanOwnerDimsAttr(op)))
      info.spatial.ownerDims.assign(ownerDims->begin(), ownerDims->end());

  if (info.spatial.staticBlockShape.empty() &&
      info.spatial.blockShape.empty()) {
    if (auto blockShape = readI64ArrayAttr(getPlanPhysicalBlockShapeAttr(op)))
      info.spatial.staticBlockShape.assign(blockShape->begin(),
                                           blockShape->end());
  }
}

static bool valueDominatesInsertion(Value value, OpBuilder &builder) {
  if (!value)
    return false;

  Block *insertBlock = builder.getInsertionBlock();
  if (!insertBlock)
    return false;

  auto insertPt = builder.getInsertionPoint();
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == insertBlock)
      return true;
  } else if (Operation *defOp = value.getDefiningOp()) {
    if (defOp->getBlock() == insertBlock) {
      if (insertPt == insertBlock->end())
        return true;
      return defOp->isBeforeInBlock(&*insertPt);
    }
  }

  Operation *insertOp = insertPt != insertBlock->end()
                            ? &*insertPt
                            : insertBlock->getTerminator();
  if (!insertOp)
    return false;

  auto func = insertOp->getParentOfType<func::FuncOp>();
  if (!func)
    return false;

  DominanceInfo domInfo(func);
  return domInfo.dominates(value, insertOp);
}

static SmallVector<Value, 4>
materializeDominatingIndexValues(OpBuilder &builder, Location loc,
                                 ArrayRef<Value> values,
                                 ArrayRef<int64_t> staticValues) {
  SmallVector<Value, 4> result;
  result.reserve(values.size());

  for (Value value : values) {
    if (!value)
      return {};
    if (valueDominatesInsertion(value, builder)) {
      result.push_back(value);
      continue;
    }

    int64_t constant = 0;
    if (!ValueAnalysis::getConstantIndex(value, constant))
      return {};
    result.push_back(arts::createConstantIndex(builder, loc, constant));
  }

  if (!result.empty())
    return result;

  if (!staticValues.empty())
    return materializeIndexValues(builder, loc, staticValues);

  return {};
}

static std::optional<EdtDistributionPattern>
deriveDistributionPatternFromKind(ContractKind kind) {
  switch (kind) {
  case ContractKind::Elementwise:
    return EdtDistributionPattern::uniform;
  case ContractKind::Stencil:
    return EdtDistributionPattern::stencil;
  case ContractKind::Matmul:
    return EdtDistributionPattern::matmul;
  case ContractKind::Triangular:
    return EdtDistributionPattern::triangular;
  case ContractKind::Unknown:
    return std::nullopt;
  }
  return std::nullopt;
}

} // namespace

ContractKind LoweringContractInfo::getEffectiveKind() const {
  if (pattern.kind != ContractKind::Unknown)
    return pattern.kind;
  if (!pattern.depPattern) {
    if (usesStencilDistribution())
      return ContractKind::Stencil;
    return ContractKind::Unknown;
  }
  switch (*pattern.depPattern) {
  case ArtsDepPattern::stencil:
  case ArtsDepPattern::stencil_tiling_nd:
  case ArtsDepPattern::cross_dim_stencil_3d:
  case ArtsDepPattern::higher_order_stencil:
  case ArtsDepPattern::wavefront_2d:
  case ArtsDepPattern::jacobi_alternating_buffers:
    return ContractKind::Stencil;
  case ArtsDepPattern::uniform:
  case ArtsDepPattern::elementwise_pipeline:
    return ContractKind::Elementwise;
  case ArtsDepPattern::matmul:
    return ContractKind::Matmul;
  case ArtsDepPattern::triangular:
    return ContractKind::Triangular;
  case ArtsDepPattern::unknown:
    return ContractKind::Unknown;
  }
  return ContractKind::Unknown;
}

bool LoweringContractInfo::usesStencilDistribution() const {
  return pattern.distributionPattern &&
         *pattern.distributionPattern == EdtDistributionPattern::stencil;
}

bool LoweringContractInfo::hasExplicitStencilContract() const {
  /// A dep pattern alone is not enough to treat the IR as carrying an
  /// authoritative stencil contract. Consumers may only rely on explicit
  /// stencil semantics once the pattern pipeline or post-DB materialization
  /// has attached concrete ownership / halo / block-shape information.
  if (!isStencilFamily())
    return false;
  return hasOwnerDims() || !spatial.stencilIndependentDims.empty() ||
         !spatial.blockShape.empty() || !spatial.minOffsets.empty() ||
         !spatial.maxOffsets.empty() || !spatial.staticBlockShape.empty() ||
         !spatial.staticMinOffsets.empty() || !spatial.staticMaxOffsets.empty();
}

bool LoweringContractInfo::supportsBlockHalo() const {
  return spatial.supportedBlockHalo && hasExplicitStencilContract();
}

std::optional<EdtDistributionPattern>
LoweringContractInfo::getEffectiveDistributionPattern() const {
  if (pattern.distributionPattern &&
      *pattern.distributionPattern != EdtDistributionPattern::unknown)
    return pattern.distributionPattern;
  if (pattern.depPattern)
    if (auto derived = getDistributionPatternForDepPattern(*pattern.depPattern))
      return derived;
  return deriveDistributionPatternFromKind(getEffectiveKind());
}

bool LoweringContractInfo::isWavefrontFamily() const {
  return pattern.depPattern &&
         *pattern.depPattern == ArtsDepPattern::wavefront_2d;
}

bool LoweringContractInfo::allowsDbAlignedChunking() const {
  return !isWavefrontFamily();
}

bool LoweringContractInfo::shouldHonorLoopBlockHintForDbAlignment() const {
  auto pattern = getEffectiveDistributionPattern();
  return pattern && *pattern == EdtDistributionPattern::stencil &&
         allowsDbAlignedChunking();
}

bool LoweringContractInfo::prefersWideTiling2DColumns() const {
  return isWavefrontFamily();
}

bool LoweringContractInfo::shouldReuseEnclosingEpoch() const {
  return isWavefrontFamily();
}

bool LoweringContractInfo::prefersSemanticOwnerLayoutPreservation() const {
  return hasOwnerDims() && supportsBlockHalo();
}

bool LoweringContractInfo::isWavefrontStencilContract() const {
  return isWavefrontFamily() && hasExplicitStencilContract() &&
         supportsBlockHalo();
}

bool LoweringContractInfo::prefersNDBlock(unsigned requiredRank) const {
  if (!supportsBlockHalo())
    return false;
  auto staticShape = getStaticBlockShape();
  if (spatial.ownerDims.size() < requiredRank ||
      ((!staticShape || staticShape->size() < requiredRank) &&
       spatial.blockShape.size() < requiredRank))
    return false;
  return true;
}

std::optional<SmallVector<int64_t, 4>>
LoweringContractInfo::getStaticBlockShape() const {
  if (auto dynamic = readConstantIndexValues(spatial.blockShape))
    return dynamic;
  if (!spatial.staticBlockShape.empty())
    return spatial.staticBlockShape;
  return std::nullopt;
}

std::optional<SmallVector<int64_t, 4>>
LoweringContractInfo::getStaticMinOffsets() const {
  if (auto dynamic = readConstantIndexValues(spatial.minOffsets))
    return dynamic;
  if (!spatial.staticMinOffsets.empty())
    return spatial.staticMinOffsets;
  return std::nullopt;
}

std::optional<SmallVector<int64_t, 4>>
LoweringContractInfo::getStaticMaxOffsets() const {
  if (auto dynamic = readConstantIndexValues(spatial.maxOffsets))
    return dynamic;
  if (!spatial.staticMaxOffsets.empty())
    return spatial.staticMaxOffsets;
  return std::nullopt;
}

static SmallVector<LoweringContractOp>
collectLoweringContractOps(Value target) {
  SmallVector<LoweringContractOp> result;
  if (!target)
    return result;
  for (auto *user : target.getUsers()) {
    if (auto contract = dyn_cast<LoweringContractOp>(user)) {
      if (contract.getTarget() == target)
        result.push_back(contract);
    }
  }
  return result;
}

LoweringContractOp mlir::arts::getLoweringContractOp(Value target) {
  auto contracts = collectLoweringContractOps(target);
  return contracts.empty() ? nullptr : contracts.front();
}

std::optional<LoweringContractInfo>
mlir::arts::getLoweringContract(Value target) {
  Value originalTarget = target;
  auto resolveContractTarget = [&](Value value) -> Value {
    if (!value)
      return Value();
    if (auto contract = getLoweringContractOp(value))
      return contract.getTarget();

    if (auto acquire = value.getDefiningOp<DbAcquireOp>()) {
      if (auto sourcePtr = acquire.getSourcePtr()) {
        if (getLoweringContractOp(sourcePtr))
          return sourcePtr;
      }
    }

    if (Operation *allocOp = DbUtils::getUnderlyingDbAlloc(value)) {
      if (auto alloc = dyn_cast<DbAllocOp>(allocOp)) {
        if (auto allocPtr = alloc.getPtr()) {
          if (getLoweringContractOp(allocPtr))
            return allocPtr;
        }
      }
    }

    return value;
  };

  auto contract = getLoweringContractOp(resolveContractTarget(target));
  if (!contract)
    return [&]() -> std::optional<LoweringContractInfo> {
      if (auto *defOp = originalTarget.getDefiningOp()) {
        if (auto semantic = getSemanticContract(defOp))
          return *semantic;
      }
      return std::nullopt;
    }();

  LoweringContractInfo info;
  if (auto depPattern = contract.getDepPattern())
    info.pattern.depPattern = *depPattern;
  if (auto distributionKind = contract.getDistributionKind())
    info.pattern.distributionKind = *distributionKind;
  if (auto distributionPattern = contract.getDistributionPattern())
    info.pattern.distributionPattern = *distributionPattern;
  if (auto version = contract.getDistributionVersion())
    info.pattern.distributionVersion = static_cast<int64_t>(*version);
  if (auto revision = contract.getPatternRevision())
    info.pattern.revision = static_cast<int64_t>(*revision);
  info.analysis.narrowableDep = contract.getNarrowableDep().value_or(false);
  if (auto ownerDims = contract.getOwnerDims())
    info.spatial.ownerDims = SmallVector<int64_t, 4>(*ownerDims);
  if (auto centerOffset = contract.getCenterOffset())
    info.spatial.centerOffset = *centerOffset;
  info.spatial.blockShape.assign(contract.getBlockShape().begin(),
                                 contract.getBlockShape().end());
  info.spatial.minOffsets.assign(contract.getMinOffsets().begin(),
                                 contract.getMinOffsets().end());
  info.spatial.maxOffsets.assign(contract.getMaxOffsets().begin(),
                                 contract.getMaxOffsets().end());
  info.spatial.writeFootprint.assign(contract.getWriteFootprint().begin(),
                                     contract.getWriteFootprint().end());
  info.spatial.supportedBlockHalo =
      contract.getSupportedBlockHalo().value_or(false);
  if (auto spatialDims = contract.getSpatialDims())
    info.spatial.spatialDims = SmallVector<int64_t, 4>(*spatialDims);
  if (auto stencilIndependentDims = contract.getStencilIndependentDims())
    info.spatial.stencilIndependentDims =
        SmallVector<int64_t, 4>(*stencilIndependentDims);
  info.analysis.postDbRefined = contract.getPostDbRefined().value_or(false);
  if (auto criticalPathDistance = contract.getCriticalPathDistance())
    info.analysis.criticalPathDistance =
        static_cast<int64_t>(*criticalPathDistance);
  if (auto contractKind = contract.getContractKind())
    info.pattern.kind = static_cast<ContractKind>(*contractKind);

  if (auto acquire = originalTarget.getDefiningOp<DbAcquireOp>()) {
    if (Value sourcePtr = acquire.getSourcePtr();
        sourcePtr && contract.getTarget() == originalTarget) {
      /// Acquire-result contracts often persist only local refinements such as
      /// critical-path distance. Merge the source-pointer contract underneath
      /// them so consumers still see the authoritative spatial/distribution
      /// shape without having to special-case acquire results.
      if (auto inherited = getLoweringContract(sourcePtr))
        mergeLoweringContractInfo(info, *inherited);
    }
  }

  if (auto *defOp = originalTarget.getDefiningOp()) {
    if (auto semantic = getSemanticContract(defOp))
      mergeLoweringContractInfo(info, *semantic);
  }
  return info;
}

std::optional<LoweringContractInfo>
mlir::arts::getSemanticContract(Operation *op) {
  if (!op)
    return std::nullopt;

  LoweringContractInfo info;
  info.pattern.depPattern = getDepPattern(op);
  info.pattern.distributionKind = getEdtDistributionKind(op);
  info.pattern.distributionPattern = getEdtDistributionPattern(op);
  if (auto version = getDistributionVersionAttr(op))
    info.pattern.distributionVersion = version.getInt();
  info.pattern.revision = getPatternRevision(op);
  if (auto ownerDims = getStencilOwnerDims(op))
    info.spatial.ownerDims.assign(ownerDims->begin(), ownerDims->end());
  if (auto blockShape = getStencilBlockShape(op))
    info.spatial.staticBlockShape.assign(blockShape->begin(),
                                         blockShape->end());
  if (auto minOffsets = getStencilMinOffsets(op))
    info.spatial.staticMinOffsets.assign(minOffsets->begin(),
                                         minOffsets->end());
  if (auto maxOffsets = getStencilMaxOffsets(op))
    info.spatial.staticMaxOffsets.assign(maxOffsets->begin(),
                                         maxOffsets->end());
  if (auto spatialDims = getStencilSpatialDims(op))
    info.spatial.spatialDims.assign(spatialDims->begin(), spatialDims->end());
  info.spatial.centerOffset = getStencilCenterOffset(op);
  // Only propagate supportedBlockHalo when the depPattern is stencil-family;
  // otherwise an op that carries mixed attrs (e.g. uniform depPattern +
  // residual stencil spatial attrs from a sibling loop) would produce an
  // inconsistent lowering contract that fails verification.
  info.spatial.supportedBlockHalo =
      hasSupportedBlockHalo(op) && info.pattern.depPattern &&
      isStencilFamilyDepPattern(*info.pattern.depPattern);
  info.analysis.narrowableDep =
      op->hasAttr(AttrNames::Operation::Contract::NarrowableDep);
  if (auto contractKind = op->getAttrOfType<IntegerAttr>(
          AttrNames::Operation::Contract::ContractKindKey))
    info.pattern.kind = static_cast<ContractKind>(contractKind.getInt());
  mergePlanSpatialAttrs(op, info);
  if (info.empty())
    return std::nullopt;
  return info;
}

LoweringContractInfo
mlir::arts::resolveLoopDistributionContract(Operation *op) {
  if (!op)
    return LoweringContractInfo{};

  LoweringContractInfo info =
      getSemanticContract(op).value_or(LoweringContractInfo{});

  if (info.pattern.depPattern &&
      *info.pattern.depPattern == ArtsDepPattern::unknown)
    info.pattern.depPattern = std::nullopt;
  if (info.pattern.distributionPattern &&
      *info.pattern.distributionPattern == EdtDistributionPattern::unknown) {
    info.pattern.distributionPattern = std::nullopt;
  }

  for (Operation *current = op; current; current = current->getParentOp()) {
    if (!info.pattern.depPattern) {
      if (auto depPattern = getDepPattern(current);
          depPattern && *depPattern != ArtsDepPattern::unknown) {
        info.pattern.depPattern = *depPattern;
      }
    }

    if (!info.pattern.distributionKind)
      if (auto distKind = getEdtDistributionKind(current))
        info.pattern.distributionKind = *distKind;

    if (!info.pattern.distributionPattern) {
      if (auto distPattern = getEdtDistributionPattern(current);
          distPattern && *distPattern != EdtDistributionPattern::unknown) {
        info.pattern.distributionPattern = *distPattern;
      }
    }

    if (!info.pattern.distributionVersion) {
      if (auto version = getDistributionVersionAttr(current))
        info.pattern.distributionVersion = version.getInt();
    }

    if (info.pattern.kind == ContractKind::Unknown) {
      if (auto contractKind = current->getAttrOfType<IntegerAttr>(
              AttrNames::Operation::Contract::ContractKindKey)) {
        int64_t rawKind = contractKind.getInt();
        if (rawKind != static_cast<int64_t>(ContractKind::Unknown))
          info.pattern.kind = static_cast<ContractKind>(rawKind);
      }
    }
  }

  normalizeLoweringContractInfo(info);
  return info;
}

std::optional<LoweringContractInfo>
mlir::arts::getLoweringContract(Operation *op, OpBuilder &builder,
                                Location loc) {
  if (!op)
    return std::nullopt;

  LoweringContractInfo info =
      getSemanticContract(op).value_or(LoweringContractInfo{});
  if (auto blockShape = getStencilBlockShape(op))
    info.spatial.blockShape = materializeIndexValues(builder, loc, *blockShape);
  if (auto minOffsets = getStencilMinOffsets(op))
    info.spatial.minOffsets = materializeIndexValues(builder, loc, *minOffsets);
  if (auto maxOffsets = getStencilMaxOffsets(op))
    info.spatial.maxOffsets = materializeIndexValues(builder, loc, *maxOffsets);
  if (auto writeFootprint = getStencilWriteFootprint(op))
    info.spatial.writeFootprint =
        materializeIndexValues(builder, loc, *writeFootprint);

  if (info.empty())
    return std::nullopt;
  return info;
}

ContractChange
mlir::arts::mergeLoweringContractInfo(LoweringContractInfo &dest,
                                      const LoweringContractInfo &src) {
  bool changed = false;

  if (dest.pattern.kind == ContractKind::Unknown &&
      src.pattern.kind != ContractKind::Unknown) {
    dest.pattern.kind = src.pattern.kind;
    changed = true;
  }
  if (!dest.pattern.depPattern && src.pattern.depPattern) {
    dest.pattern.depPattern = src.pattern.depPattern;
    changed = true;
  }
  if (!dest.pattern.distributionKind && src.pattern.distributionKind) {
    dest.pattern.distributionKind = src.pattern.distributionKind;
    changed = true;
  }
  if (!dest.pattern.distributionPattern && src.pattern.distributionPattern) {
    dest.pattern.distributionPattern = src.pattern.distributionPattern;
    changed = true;
  }
  if (!dest.pattern.distributionVersion && src.pattern.distributionVersion) {
    dest.pattern.distributionVersion = src.pattern.distributionVersion;
    changed = true;
  }
  if (!dest.pattern.revision && src.pattern.revision) {
    dest.pattern.revision = src.pattern.revision;
    changed = true;
  }
  if (!dest.analysis.narrowableDep && src.analysis.narrowableDep) {
    dest.analysis.narrowableDep = true;
    changed = true;
  }

  auto shouldTakeHigherRank = [](size_t current, size_t incoming) -> bool {
    if (incoming == 0)
      return false;
    return current == 0 || incoming > current;
  };

  if (shouldTakeHigherRank(dest.spatial.ownerDims.size(),
                           src.spatial.ownerDims.size())) {
    dest.spatial.ownerDims.assign(src.spatial.ownerDims.begin(),
                                  src.spatial.ownerDims.end());
    changed = true;
  }
  if (!dest.spatial.centerOffset && src.spatial.centerOffset) {
    dest.spatial.centerOffset = src.spatial.centerOffset;
    changed = true;
  }

  if (shouldTakeHigherRank(dest.spatial.blockShape.size(),
                           src.spatial.blockShape.size())) {
    dest.spatial.blockShape.assign(src.spatial.blockShape.begin(),
                                   src.spatial.blockShape.end());
    changed = true;
  }
  if (shouldTakeHigherRank(dest.spatial.minOffsets.size(),
                           src.spatial.minOffsets.size())) {
    dest.spatial.minOffsets.assign(src.spatial.minOffsets.begin(),
                                   src.spatial.minOffsets.end());
    changed = true;
  }
  if (shouldTakeHigherRank(dest.spatial.maxOffsets.size(),
                           src.spatial.maxOffsets.size())) {
    dest.spatial.maxOffsets.assign(src.spatial.maxOffsets.begin(),
                                   src.spatial.maxOffsets.end());
    changed = true;
  }
  if (shouldTakeHigherRank(dest.spatial.writeFootprint.size(),
                           src.spatial.writeFootprint.size())) {
    dest.spatial.writeFootprint.assign(src.spatial.writeFootprint.begin(),
                                       src.spatial.writeFootprint.end());
    changed = true;
  }

  if (shouldTakeHigherRank(dest.spatial.staticBlockShape.size(),
                           src.spatial.staticBlockShape.size())) {
    dest.spatial.staticBlockShape.assign(src.spatial.staticBlockShape.begin(),
                                         src.spatial.staticBlockShape.end());
    changed = true;
  }
  if (shouldTakeHigherRank(dest.spatial.staticMinOffsets.size(),
                           src.spatial.staticMinOffsets.size())) {
    dest.spatial.staticMinOffsets.assign(src.spatial.staticMinOffsets.begin(),
                                         src.spatial.staticMinOffsets.end());
    changed = true;
  }
  if (shouldTakeHigherRank(dest.spatial.staticMaxOffsets.size(),
                           src.spatial.staticMaxOffsets.size())) {
    dest.spatial.staticMaxOffsets.assign(src.spatial.staticMaxOffsets.begin(),
                                         src.spatial.staticMaxOffsets.end());
    changed = true;
  }

  // Only merge supportedBlockHalo when the effective depPattern is
  // stencil-family (either already set in dest or coming from src).
  auto effectiveDepPattern = dest.pattern.depPattern ? dest.pattern.depPattern
                                                     : src.pattern.depPattern;
  if (!dest.spatial.supportedBlockHalo && src.spatial.supportedBlockHalo &&
      effectiveDepPattern && isStencilFamilyDepPattern(*effectiveDepPattern)) {
    dest.spatial.supportedBlockHalo = true;
    changed = true;
  }

  if (dest.spatial.spatialDims.empty() && !src.spatial.spatialDims.empty()) {
    dest.spatial.spatialDims.assign(src.spatial.spatialDims.begin(),
                                    src.spatial.spatialDims.end());
    changed = true;
  }

  if (dest.spatial.stencilIndependentDims.empty() &&
      !src.spatial.stencilIndependentDims.empty()) {
    dest.spatial.stencilIndependentDims.assign(
        src.spatial.stencilIndependentDims.begin(),
        src.spatial.stencilIndependentDims.end());
    changed = true;
  }

  if (!dest.analysis.postDbRefined && src.analysis.postDbRefined) {
    dest.analysis.postDbRefined = true;
    changed = true;
  }
  if (!dest.analysis.criticalPathDistance &&
      src.analysis.criticalPathDistance) {
    dest.analysis.criticalPathDistance = src.analysis.criticalPathDistance;
    changed = true;
  }

  normalizeLoweringContractInfo(dest);
  return changed ? ContractChange::Changed : ContractChange::Unchanged;
}

void mlir::arts::normalizeLoweringContractInfo(LoweringContractInfo &info) {
  const size_t expectedRank = info.spatial.ownerDims.size();
  if (expectedRank == 0)
    return;

  auto clearMismatchedDynamic = [&](SmallVector<Value, 4> &values) {
    if (!values.empty() && values.size() != expectedRank)
      values.clear();
  };
  auto clearMismatchedStatic = [&](SmallVector<int64_t, 4> &values) {
    if (!values.empty() && values.size() != expectedRank)
      values.clear();
  };

  clearMismatchedDynamic(info.spatial.blockShape);
  clearMismatchedDynamic(info.spatial.minOffsets);
  clearMismatchedDynamic(info.spatial.maxOffsets);
  clearMismatchedDynamic(info.spatial.writeFootprint);

  clearMismatchedStatic(info.spatial.staticBlockShape);
  clearMismatchedStatic(info.spatial.staticMinOffsets);
  clearMismatchedStatic(info.spatial.staticMaxOffsets);
}

std::optional<SmallVector<int64_t, 4>>
mlir::arts::projectOwnerIndexedStaticValues(ArrayRef<int64_t> values,
                                            ArrayRef<int64_t> sourceOwnerDims,
                                            ArrayRef<int64_t> targetOwnerDims) {
  if (targetOwnerDims.empty())
    return SmallVector<int64_t, 4>{values.begin(), values.end()};
  if (values.empty())
    return SmallVector<int64_t, 4>{};
  if (values.size() != sourceOwnerDims.size())
    return std::nullopt;

  SmallVector<int64_t, 4> result;
  result.reserve(targetOwnerDims.size());
  for (int64_t targetDim : targetOwnerDims) {
    std::optional<unsigned> ownerPosition;
    for (unsigned idx = 0, e = sourceOwnerDims.size(); idx < e; ++idx) {
      if (sourceOwnerDims[idx] == targetDim) {
        ownerPosition = static_cast<unsigned>(idx);
        break;
      }
    }
    if (!ownerPosition)
      return std::nullopt;
    result.push_back(values[*ownerPosition]);
  }
  return result;
}

std::optional<unsigned>
mlir::arts::getContractOwnerPosition(const LoweringContractInfo &info,
                                     unsigned physicalDim) {
  if (info.spatial.ownerDims.empty())
    return physicalDim;

  for (unsigned idx = 0; idx < info.spatial.ownerDims.size(); ++idx) {
    int64_t ownerDim = info.spatial.ownerDims[idx];
    if (ownerDim < 0)
      continue;
    if (static_cast<unsigned>(ownerDim) == physicalDim)
      return idx;
  }

  return std::nullopt;
}

SmallVector<unsigned, 4>
mlir::arts::resolveContractOwnerDims(const LoweringContractInfo &info,
                                     unsigned rank) {
  SmallVector<unsigned, 4> dims;
  dims.reserve(rank);
  for (int64_t dim : info.spatial.ownerDims) {
    if (dim < 0)
      continue;
    /// `rank` is the number of owner dimensions the caller wants to resolve,
    /// not an upper bound on the physical memref dimension number. Preserving
    /// the recorded memref dims is critical for N-D block partitions such as
    /// k-owned 3-D/4-D arrays, where ownerDims may legitimately be [2] or [3].
    dims.push_back(static_cast<unsigned>(dim));
    if (dims.size() == rank)
      break;
  }
  if (dims.empty()) {
    for (unsigned dim = 0; dim < rank; ++dim)
      dims.push_back(dim);
  }
  return dims;
}

bool mlir::arts::prefersContractNDBlock(const LoweringContractInfo &info,
                                        unsigned requiredRank) {
  return info.prefersNDBlock(requiredRank);
}

std::optional<LoweringContractInfo>
mlir::arts::resolveEffectiveContract(Value target) {
  if (!target)
    return std::nullopt;
  auto ptrContract = getLoweringContract(target);
  if (auto *defOp = target.getDefiningOp()) {
    if (auto semContract = getSemanticContract(defOp)) {
      if (!ptrContract)
        return semContract;
      mergeLoweringContractInfo(*ptrContract, *semContract);
      return ptrContract;
    }
  }
  return ptrContract;
}

std::optional<LoweringContractInfo>
mlir::arts::resolveEffectiveContract(Operation *op) {
  if (!op)
    return std::nullopt;
  auto semContract = getSemanticContract(op);
  if (!semContract || semContract->empty())
    return std::nullopt;
  return semContract;
}

ContractChange mlir::arts::combineContracts(LoweringContractInfo &dest,
                                            const LoweringContractInfo &src) {
  return mergeLoweringContractInfo(dest, src);
}

void mlir::arts::patchContract(LoweringContractInfo &contract,
                               ArrayRef<int64_t> ownerDims,
                               ArrayRef<int64_t> minOffsets,
                               ArrayRef<int64_t> maxOffsets) {
  if (!ownerDims.empty())
    contract.spatial.ownerDims.assign(ownerDims.begin(), ownerDims.end());
  if (!minOffsets.empty())
    contract.spatial.staticMinOffsets.assign(minOffsets.begin(),
                                             minOffsets.end());
  if (!maxOffsets.empty())
    contract.spatial.staticMaxOffsets.assign(maxOffsets.begin(),
                                             maxOffsets.end());
}

std::optional<std::pair<SmallVector<int64_t, 4>, SmallVector<int64_t, 4>>>
mlir::arts::projectHaloWindow(const LoweringContractInfo &contract) {
  auto minOffsets = contract.getStaticMinOffsets();
  auto maxOffsets = contract.getStaticMaxOffsets();
  if (!minOffsets || !maxOffsets)
    return std::nullopt;
  return std::make_pair(*minOffsets, *maxOffsets);
}

bool mlir::arts::hasCompleteHaloState(const LoweringContractInfo &contract) {
  return !contract.spatial.ownerDims.empty() &&
         (contract.getStaticMinOffsets().has_value()) &&
         (contract.getStaticMaxOffsets().has_value());
}

std::optional<LoweringContractInfo>
mlir::arts::resolveAcquireContract(DbAcquireOp acquire) {
  if (!acquire)
    return std::nullopt;

  auto info = getLoweringContract(acquire.getPtr());
  if (auto alloc = dyn_cast_or_null<DbAllocOp>(
          DbUtils::getUnderlyingDbAlloc(acquire.getSourcePtr())))
    if (auto allocInfo = getLoweringContract(alloc.getPtr())) {
      if (!info)
        info = *allocInfo;
      else
        mergeLoweringContractInfo(*info, *allocInfo);
    }
  return info;
}

bool mlir::arts::shouldApplyStencilHalo(const LoweringContractInfo &contract,
                                        ArtsMode effectiveMode) {
  if (effectiveMode != ArtsMode::in)
    return false;
  return contract.supportsBlockHalo();
}

bool mlir::arts::shouldApplyStencilHalo(const LoweringContractInfo &contract,
                                        DbAcquireOp acquire) {
  if (!acquire)
    return false;
  return shouldApplyStencilHalo(contract, acquire.getMode());
}

bool mlir::arts::shouldUsePartitionSliceAsDepWindow(
    const LoweringContractInfo &contract, ArtsMode effectiveMode,
    PartitionMode partitionMode) {
  const bool hasExplicitStencilContract = contract.hasExplicitStencilContract();
  const bool applyStencilHalo = shouldApplyStencilHalo(contract, effectiveMode);

  if (effectiveMode == ArtsMode::in && hasExplicitStencilContract &&
      !applyStencilHalo && partitionMode == PartitionMode::stencil)
    return true;

  if (effectiveMode == ArtsMode::inout && contract.isWavefrontStencilContract())
    return true;

  return false;
}

bool mlir::arts::shouldUsePartitionSliceAsDepWindow(
    const LoweringContractInfo &contract, DbAcquireOp acquire) {
  if (!acquire)
    return false;
  return shouldUsePartitionSliceAsDepWindow(
      contract, acquire.getMode(),
      acquire.getPartitionMode().value_or(PartitionMode::coarse));
}

bool mlir::arts::shouldPreserveParentDepRange(
    const LoweringContractInfo &contract, ArtsMode effectiveMode) {
  if (effectiveMode != ArtsMode::in)
    return false;

  const bool applyStencilHalo = shouldApplyStencilHalo(contract, effectiveMode);
  const bool hasExplicitStencilContract = contract.hasExplicitStencilContract();
  const bool prefersWorkerLocalReadSlice = contract.analysis.narrowableDep;

  return !applyStencilHalo && !hasExplicitStencilContract &&
         !prefersWorkerLocalReadSlice;
}

bool mlir::arts::shouldPreserveParentDepRange(
    const LoweringContractInfo &contract, DbAcquireOp acquire) {
  if (!acquire)
    return false;
  return shouldPreserveParentDepRange(contract, acquire.getMode());
}

LoweringContractOp
mlir::arts::upsertLoweringContract(OpBuilder &builder, Location loc,
                                   Value target,
                                   const LoweringContractInfo &info) {
  if (!target || info.empty())
    return nullptr;

  LoweringContractInfo normalizedInfo = info;
  normalizeLoweringContractInfo(normalizedInfo);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(target);

  auto staleContracts = collectLoweringContractOps(target);

  SmallVector<Value> blockShapeVals = materializeDominatingIndexValues(
      builder, loc, normalizedInfo.spatial.blockShape,
      normalizedInfo.spatial.staticBlockShape);
  SmallVector<Value> minOffsetVals = materializeDominatingIndexValues(
      builder, loc, normalizedInfo.spatial.minOffsets,
      normalizedInfo.spatial.staticMinOffsets);
  SmallVector<Value> maxOffsetVals = materializeDominatingIndexValues(
      builder, loc, normalizedInfo.spatial.maxOffsets,
      normalizedInfo.spatial.staticMaxOffsets);
  SmallVector<Value> writeFootprintVals = materializeDominatingIndexValues(
      builder, loc, normalizedInfo.spatial.writeFootprint, {});

  LoweringContractInfo materializedInfo = normalizedInfo;
  materializedInfo.spatial.blockShape = std::move(blockShapeVals);
  materializedInfo.spatial.minOffsets = std::move(minOffsetVals);
  materializedInfo.spatial.maxOffsets = std::move(maxOffsetVals);
  materializedInfo.spatial.writeFootprint = std::move(writeFootprintVals);

  auto contract =
      LoweringContractOp::create(builder, loc, target, materializedInfo);

  for (auto stale : staleContracts)
    if (stale.getOperation() != contract.getOperation())
      stale->erase();

  return contract;
}

void mlir::arts::eraseLoweringContracts(Value target) {
  for (auto contract : collectLoweringContractOps(target))
    contract->erase();
}

void mlir::arts::transferOperationContract(Operation *source,
                                           Operation *target) {
  if (!source || !target)
    return;
  copySemanticContractAttrs(source, target);
}

void mlir::arts::transferLoweringContract(Operation *source, Value target,
                                          OpBuilder &builder, Location loc) {
  if (!source || !target)
    return;

  if (auto info = getLoweringContract(source, builder, loc))
    upsertLoweringContract(builder, loc, target, *info);
}

static void copyLoweringContract(Value source, Value target, OpBuilder &builder,
                                 Location loc) {
  if (!source || !target || source == target)
    return;
  auto info = getLoweringContract(source);
  if (!info)
    return;
  upsertLoweringContract(builder, loc, target, *info);
}

void mlir::arts::transferValueContract(Value source, Value target,
                                       OpBuilder &builder, Location loc) {
  copyLoweringContract(source, target, builder, loc);
}

void mlir::arts::moveValueContract(Value source, Value target,
                                   OpBuilder &builder, Location loc) {
  if (!source || !target || source == target)
    return;
  copyLoweringContract(source, target, builder, loc);
  eraseLoweringContracts(source);
}

void mlir::arts::transferContract(Operation *sourceOp, Operation *targetOp,
                                  Value sourceContractTarget,
                                  Value targetContractTarget,
                                  OpBuilder &builder, Location loc) {
  transferOperationContract(sourceOp, targetOp);
  transferValueContract(sourceContractTarget, targetContractTarget, builder,
                        loc);
}
