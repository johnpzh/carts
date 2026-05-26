///==========================================================================///
/// File: ConversionUtils.h
///
/// Shared implementation helpers for the explicit SDE -> CODIR -> ARTS
/// conversion passes. Keep pass entry points in SdeToCodir.cpp and
/// CodirToArts.cpp so the source layout mirrors the compiler pipeline.
///==========================================================================///
#ifndef CARTS_DIALECT_CODIR_CONVERSION_CONVERSIONUTILS_H
#define CARTS_DIALECT_CODIR_CONVERSION_CONVERSIONUTILS_H
#include "SdeToCodir/SdeToCodirMetadataUtils.h"
#include "SdeToCodir/TaskDepSliceUtils.h"
#include "carts/dialect/codir/Utils/CodeletABIUtils.h"
#include "carts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "carts/utils/ArrayAttrUtils.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include <functional>
using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::codir;

namespace {

using ::mlir::carts::codir::sde_to_codir::CodirCodeletMetadata;
using ::mlir::carts::codir::sde_to_codir::convertAccessMode;
using ::mlir::carts::codir::sde_to_codir::convertStorageViewKind;
using ::mlir::carts::codir::sde_to_codir::getCodirMetadataFromSchedulingUnit;
using ::mlir::carts::codir::sde_to_codir::getCodirMetadataFromTask;

static inline codir::CodirAccessMode
mergeAccessMode(codir::CodirAccessMode lhs, codir::CodirAccessMode rhs) {
  if (lhs == rhs)
    return lhs;
  return codir::CodirAccessMode::readwrite;
}

static inline codir::CodirStorageViewKind
mergeStorageViewKind(codir::CodirStorageViewKind lhs,
                     codir::CodirStorageViewKind rhs) {
  if (lhs == rhs)
    return lhs;
  return codir::CodirStorageViewKind::phase_redistributed;
}

static inline SmallVector<Attribute>
buildCodirAccessModeAttrs(MLIRContext *ctx,
                          ArrayRef<codir::CodirAccessMode> modes) {
  SmallVector<Attribute> attrs;
  attrs.reserve(modes.size());
  for (codir::CodirAccessMode mode : modes)
    attrs.push_back(codir::CodirAccessModeAttr::get(ctx, mode));
  return attrs;
}

static inline SmallVector<Attribute>
buildCodirStorageViewAttrs(MLIRContext *ctx,
                           ArrayRef<codir::CodirStorageViewKind> storageViews) {
  SmallVector<Attribute> attrs;
  attrs.reserve(storageViews.size());
  for (codir::CodirStorageViewKind storageView : storageViews)
    attrs.push_back(codir::CodirStorageViewKindAttr::get(ctx, storageView));
  return attrs;
}

static inline codir::CodeletOp
createCodirCodelet(OpBuilder &builder, Location loc, ArrayAttr depModes,
                   ArrayAttr depStorageViews, ValueRange deps,
                   ValueRange params, const CodirCodeletMetadata &metadata = {},
                   UnitAttr taskDepend = {}, UnitAttr orderedTaskDepend = {},
                   UnitAttr completionBarrier = {}) {
  return codir::CodeletOp::create(
      builder, loc, depModes, depStorageViews, ArrayAttr{}, taskDepend,
      orderedTaskDepend, completionBarrier, metadata.pattern,
      metadata.reductionStrategy, metadata.partialReduction,
      metadata.partialReductionDims, metadata.partialReductionOwnerDims,
      metadata.partialReductionDepResultDimMaps, UnitAttr{}, ArrayAttr{},
      IntegerAttr{}, IntegerAttr{}, IntegerAttr{}, metadata.distributionKind,
      metadata.iterationTopology, metadata.repetitionStructure,
      metadata.asyncStrategy, metadata.planOwnerDims, metadata.tileOwnerDims,
      metadata.tileShape, metadata.logicalWorkerSlice, metadata.haloShape,
      metadata.accessMinOffsets, metadata.accessMaxOffsets,
      metadata.spatialDims, metadata.writeFootprint, metadata.inPlaceSafe,
      metadata.inPlaceSharedState, deps, params);
}

static inline Value materializeIndexFoldResult(OpBuilder &builder, Location loc,
                                               OpFoldResult value) {
  if (auto attr = dyn_cast<Attribute>(value)) {
    auto intAttr = cast<IntegerAttr>(attr);
    return createConstantIndex(builder, loc, intAttr.getInt());
  }
  return cast<Value>(value);
}

static inline std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();

  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  }

  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)))
    return std::nullopt;
  return constant.getSExtValue();
}

static inline bool
isKnownZeroIndex(Value value,
                 const DenseMap<Value, Value> &sourceByBlockArgument) {
  if (std::optional<int64_t> constant = getConstantIndexValue(value))
    return *constant == 0;

  auto it = sourceByBlockArgument.find(value);
  if (it == sourceByBlockArgument.end())
    return false;

  if (std::optional<int64_t> constant = getConstantIndexValue(it->second))
    return *constant == 0;
  return false;
}

static inline OpFoldResult
getStaticOrDynamicIndex(OpBuilder &builder, Value value, bool preferStatic) {
  if (preferStatic) {
    if (std::optional<int64_t> constant = getConstantIndexValue(value))
      return builder.getIndexAttr(*constant);
  }
  return value;
}

static inline SmallVector<Value>
materializeIndexFoldResults(OpBuilder &builder, Location loc,
                            ArrayRef<OpFoldResult> values) {
  SmallVector<Value> result;
  result.reserve(values.size());
  for (OpFoldResult value : values)
    result.push_back(materializeIndexFoldResult(builder, loc, value));
  return result;
}

static inline OpFoldResult
remapIndexFoldResult(OpBuilder &builder, Location loc, OpFoldResult value,
                     const DenseMap<Value, Value> &mapping) {
  if (auto attr = dyn_cast<Attribute>(value))
    return attr;
  Value oldValue = cast<Value>(value);
  auto it = mapping.find(oldValue);
  if (it == mapping.end()) {
    if (std::optional<int64_t> constant = getConstantIndexValue(oldValue))
      return createConstantIndex(builder, loc, *constant);
    Operation *defOp = oldValue.getDefiningOp();
    if (defOp && defOp->hasTrait<OpTrait::ConstantLike>() &&
        defOp->getNumResults() == 1)
      return builder.clone(*defOp)->getResult(0);
  }
  if (it == mapping.end())
    return oldValue;
  return it->second;
}

static inline SmallVector<OpFoldResult>
remapIndexFoldResults(OpBuilder &builder, Location loc,
                      ArrayRef<OpFoldResult> values,
                      const DenseMap<Value, Value> &mapping) {
  SmallVector<OpFoldResult> remapped;
  remapped.reserve(values.size());
  for (OpFoldResult value : values)
    remapped.push_back(remapIndexFoldResult(builder, loc, value, mapping));
  return remapped;
}

static inline std::optional<sde::SdeStructuredClassification>
getSdeClassification(sde::SdeSuIterateOp op) {
  if (auto classification = op.getStructuredClassification())
    return *classification;
  return std::nullopt;
}

static inline bool hasSdePhysicalOwnerSlicePlan(sde::SdeSuIterateOp op) {
  return op && op.getPhysicalBlockShapeAttr() && op.getPhysicalOwnerDimsAttr();
}

static inline bool canUseOwnerSliceBoundaryPlan(sde::SdeSuIterateOp source) {
  if (!source || source.getLowerBounds().size() != 1 ||
      source.getUpperBounds().size() != 1 || source.getSteps().size() != 1)
    return false;

  auto classification = getSdeClassification(source);
  if (!classification)
    return false;

  switch (*classification) {
  case sde::SdeStructuredClassification::matmul:
  case sde::SdeStructuredClassification::elementwise:
  case sde::SdeStructuredClassification::elementwise_pipeline:
    return true;
  case sde::SdeStructuredClassification::stencil:
  case sde::SdeStructuredClassification::reduction:
    return false;
  }
  return false;
}

static inline bool indexSelectsOwnerSlice(Value index, Value ownerIv,
                                          llvm::SmallPtrSetImpl<Value> &seen) {
  if (!index || !ownerIv)
    return false;
  if (index == ownerIv)
    return true;
  if (::mlir::carts::ValueAnalysis::dependsOn(index, ownerIv))
    return true;
  if (!seen.insert(index).second)
    return false;

  auto blockArg = dyn_cast<BlockArgument>(index);
  if (blockArg) {
    auto loop =
        dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!loop || loop.getInductionVar() != index)
      return false;

    llvm::SmallPtrSet<Value, 8> lowerSeen;
    llvm::SmallPtrSet<Value, 8> upperSeen;
    for (Value value : seen) {
      lowerSeen.insert(value);
      upperSeen.insert(value);
    }
    return indexSelectsOwnerSlice(loop.getLowerBound(), ownerIv, lowerSeen) &&
           indexSelectsOwnerSlice(loop.getUpperBound(), ownerIv, upperSeen);
  }

  Operation *def = index.getDefiningOp();
  if (!isa_and_nonnull<
          arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
          arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::IndexCastOp,
          arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
          arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp>(def))
    return false;

  for (Value operand : def->getOperands()) {
    llvm::SmallPtrSet<Value, 8> operandSeen;
    for (Value value : seen)
      operandSeen.insert(value);
    if (indexSelectsOwnerSlice(operand, ownerIv, operandSeen))
      return true;
  }
  return false;
}

static inline bool indexSelectsOwnerSlice(Value index, Value ownerIv) {
  llvm::SmallPtrSet<Value, 8> seen;
  return indexSelectsOwnerSlice(index, ownerIv, seen);
}

static inline std::optional<unsigned>
inferSingleOwnerAccessDim(sde::SdeSuIterateOp source, Value root) {
  if (!source || !root || source.getBody().empty())
    return std::nullopt;

  auto rootType = dyn_cast<MemRefType>(root.getType());
  if (!rootType || rootType.getRank() == 0)
    return std::nullopt;

  Block &body = source.getBody().front();
  if (body.getNumArguments() == 0)
    return std::nullopt;
  Value ownerIv = body.getArgument(0);

  bool sawRootAccess = false;
  bool rejected = false;
  std::optional<unsigned> selectedDim;
  auto checkAccess = [&](Value memref, OperandRange indices) {
    if (::mlir::carts::ValueAnalysis::stripMemrefViewOps(memref) != root)
      return;
    sawRootAccess = true;
    std::optional<unsigned> accessDim;
    for (auto [dim, index] : llvm::enumerate(indices)) {
      if (!indexSelectsOwnerSlice(index, ownerIv))
        continue;
      if (accessDim && *accessDim != dim) {
        rejected = true;
        return;
      }
      accessDim = static_cast<unsigned>(dim);
    }
    if (!accessDim) {
      rejected = true;
      return;
    }
    if (selectedDim && *selectedDim != *accessDim) {
      rejected = true;
      return;
    }
    selectedDim = *accessDim;
  };

  source.getBody().walk([&](Operation *op) {
    if (rejected)
      return WalkResult::interrupt();
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (isa<MemRefType>(load.getResult().getType()))
        return WalkResult::advance();
      checkAccess(load.getMemref(), load.getIndices());
      return rejected ? WalkResult::interrupt() : WalkResult::advance();
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (isa<MemRefType>(store.getValueToStore().getType()))
        return WalkResult::advance();
      checkAccess(store.getMemref(), store.getIndices());
      return rejected ? WalkResult::interrupt() : WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (!sawRootAccess || rejected)
    return std::nullopt;
  return selectedDim;
}

static inline bool allRootAccessesUseOwnerFirstDim(sde::SdeSuIterateOp source,
                                                   Value root) {
  std::optional<unsigned> ownerDim = inferSingleOwnerAccessDim(source, root);
  return ownerDim && *ownerDim == 0;
}

static inline bool hasSamePhysicalLayoutPlan(sde::SdeSuIterateOp lhs,
                                             sde::SdeSuIterateOp rhs) {
  if (!lhs || !rhs)
    return false;
  return lhs.getPhysicalOwnerDimsAttr() == rhs.getPhysicalOwnerDimsAttr() &&
         lhs.getPhysicalBlockShapeAttr() == rhs.getPhysicalBlockShapeAttr() &&
         lhs.getLogicalWorkerSliceAttr() == rhs.getLogicalWorkerSliceAttr() &&
         lhs.getPhysicalHaloShapeAttr() == rhs.getPhysicalHaloShapeAttr() &&
         lhs.getIterationTopologyAttr() == rhs.getIterationTopologyAttr();
}

static inline bool canAccessRootWithPlan(sde::SdeSuIterateOp source, Value root,
                                         sde::SdeSuIterateOp selected) {
  if (!source || !root || !selected)
    return false;
  if (!hasSdePhysicalOwnerSlicePlan(source) ||
      !canUseOwnerSliceBoundaryPlan(source) ||
      !hasSamePhysicalLayoutPlan(selected, source))
    return false;
  return allRootAccessesUseOwnerFirstDim(source, root);
}

static inline bool isMemrefForwardingOp(Operation *op) {
  if (!op || op->getNumRegions() != 0)
    return false;
  return llvm::any_of(op->getResults(), [](Value result) {
    return isa<MemRefType>(result.getType());
  });
}

static inline sde::SdeSuIterateOp getEnclosingSuIterate(Operation *op) {
  for (Operation *cur = op ? op->getParentOp() : nullptr; cur;
       cur = cur->getParentOp())
    if (auto iterate = dyn_cast<sde::SdeSuIterateOp>(cur))
      return iterate;
  return {};
}

struct CodirMemoryAccessInfo {
  Operation *op = nullptr;
  Value memref;
  SmallVector<Value> indices;
};

static inline std::optional<CodirMemoryAccessInfo>
getCodirMemoryAccessInfo(Operation *op) {
  if (!op)
    return std::nullopt;

  if (auto load = dyn_cast<memref::LoadOp>(op))
    return CodirMemoryAccessInfo{
        op, load.getMemRef(),
        SmallVector<Value>(load.getIndices().begin(), load.getIndices().end())};
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return CodirMemoryAccessInfo{op, store.getMemRef(),
                                 SmallVector<Value>(store.getIndices().begin(),
                                                    store.getIndices().end())};
  if (auto load = dyn_cast<polygeist::DynLoadOp>(op))
    return CodirMemoryAccessInfo{
        op, load.getMemref(),
        SmallVector<Value>(load.getIndices().begin(), load.getIndices().end())};
  if (auto store = dyn_cast<polygeist::DynStoreOp>(op))
    return CodirMemoryAccessInfo{op, store.getMemref(),
                                 SmallVector<Value>(store.getIndices().begin(),
                                                    store.getIndices().end())};
  if (auto load = dyn_cast<affine::AffineLoadOp>(op))
    return CodirMemoryAccessInfo{
        op, load.getMemRef(),
        SmallVector<Value>(load.getMapOperands().begin(),
                           load.getMapOperands().end())};
  if (auto store = dyn_cast<affine::AffineStoreOp>(op))
    return CodirMemoryAccessInfo{
        op, store.getMemRef(),
        SmallVector<Value>(store.getMapOperands().begin(),
                           store.getMapOperands().end())};
  return std::nullopt;
}

static inline bool hasHostMemrefAccessOutsideSchedulingUnit(Value root) {
  if (!root)
    return false;

  SmallVector<Value, 8> worklist{root};
  llvm::SmallPtrSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    for (Operation *user : llvm::make_early_inc_range(current.getUsers())) {
      if (!user || user->getParentOfType<codir::CodeletOp>() ||
          user->getParentOfType<sde::SdeSuIterateOp>())
        continue;
      if (isa<memref::DeallocOp, memref::DimOp>(user))
        continue;
      if (isa<memref::LoadOp, memref::StoreOp>(user))
        return true;
      if (isMemrefForwardingOp(user))
        for (Value result : user->getResults())
          if (isa<MemRefType>(result.getType()))
            worklist.push_back(result);
    }
  }

  return false;
}

static inline FailureOr<sde::SdeSuIterateOp>
selectMuAllocWritePlan(sde::SdeMuAllocOp op) {
  if (!op)
    return sde::SdeSuIterateOp{};

  Value root = op.getMemref();
  if (hasHostMemrefAccessOutsideSchedulingUnit(root))
    return sde::SdeSuIterateOp{};

  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module)
    return sde::SdeSuIterateOp{};

  sde::SdeSuIterateOp selected;
  WalkResult result = module.walk([&](memref::StoreOp store) {
    if (::mlir::carts::ValueAnalysis::stripMemrefViewOps(store.getMemref()) !=
        root)
      return WalkResult::advance();

    sde::SdeSuIterateOp source = getEnclosingSuIterate(store);
    if (!source || !hasSdePhysicalOwnerSlicePlan(source) ||
        !canUseOwnerSliceBoundaryPlan(source) ||
        !allRootAccessesUseOwnerFirstDim(source, root))
      return WalkResult::advance();

    if (!selected) {
      selected = source;
      return WalkResult::advance();
    }

    if (!hasSamePhysicalLayoutPlan(selected, source)) {
      store.emitError("cannot choose a physical DB layout for SDE MU "
                      "allocation with conflicting write-owner plans");
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();
  if (!selected)
    return selected;

  result = module.walk([&](Operation *nested) {
    auto access = getCodirMemoryAccessInfo(nested);
    if (!access)
      return WalkResult::advance();
    if (::mlir::carts::ValueAnalysis::stripMemrefViewOps(access->memref) !=
        root)
      return WalkResult::advance();

    sde::SdeSuIterateOp source = getEnclosingSuIterate(nested);
    if (!source)
      return WalkResult::advance();

    // Cross-phase intermediates need an explicit M3 token-local phase plan
    // before they can be block-backed safely. For now, keep row-strip DB
    // layout only when every access stays inside the selected scheduling unit.
    if (source != selected || !canAccessRootWithPlan(source, root, selected)) {
      selected = {};
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return selected;
}

static inline bool
isSdeMuAllocMaterializedWithPlan(Value dep, sde::SdeSuIterateOp source) {
  Value root = ::mlir::carts::ValueAnalysis::stripMemrefViewOps(dep);
  auto muAlloc = root ? root.getDefiningOp<sde::SdeMuAllocOp>() : nullptr;
  if (!muAlloc)
    return true;

  FailureOr<sde::SdeSuIterateOp> selected = selectMuAllocWritePlan(muAlloc);
  if (failed(selected) || !*selected)
    return false;

  return *selected == source && hasSamePhysicalLayoutPlan(*selected, source);
}

static inline bool
canBridgeSdeMuAllocHostWholeToComputeBlock(Value dep,
                                           sde::SdeSuIterateOp source) {
  Value root = ::mlir::carts::ValueAnalysis::stripMemrefViewOps(dep);
  auto muAlloc = root ? root.getDefiningOp<sde::SdeMuAllocOp>() : nullptr;
  if (!muAlloc || !source)
    return false;
  if (!hasHostMemrefAccessOutsideSchedulingUnit(root))
    return false;
  return hasSdePhysicalOwnerSlicePlan(source) &&
         canUseOwnerSliceBoundaryPlan(source) &&
         allRootAccessesUseOwnerFirstDim(source, root);
}

static inline void buildCodirSubviewMixedOperands(
    MemRefType resultType, ValueRange offsetValues, ValueRange sizeValues,
    OpBuilder &builder, Location loc, SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides) {
  offsets.clear();
  offsets.reserve(offsetValues.size());
  sizes.clear();
  sizes.reserve(sizeValues.size());
  strides.clear();
  strides.reserve(offsetValues.size());

  SmallVector<int64_t> resultStrides;
  int64_t resultOffset = ShapedType::kDynamic;
  bool resultLayoutKnown =
      succeeded(resultType.getStridesAndOffset(resultStrides, resultOffset));
  bool preferStaticOffsets =
      resultLayoutKnown && !ShapedType::isDynamic(resultOffset);

  for (Value offset : offsetValues)
    offsets.push_back(
        getStaticOrDynamicIndex(builder, offset, preferStaticOffsets));

  for (auto [idx, size] : llvm::enumerate(sizeValues)) {
    bool preferStaticSize =
        idx < static_cast<size_t>(resultType.getRank()) &&
        !resultType.isDynamicDim(static_cast<unsigned>(idx));
    sizes.push_back(getStaticOrDynamicIndex(builder, size, preferStaticSize));
  }

  Value one = createOneIndex(builder, loc);
  for (size_t idx = 0, end = offsetValues.size(); idx < end; ++idx) {
    bool preferStaticStride = !resultLayoutKnown ||
                              idx >= resultStrides.size() ||
                              !ShapedType::isDynamic(resultStrides[idx]);
    strides.push_back(preferStaticStride ? OpFoldResult(builder.getIndexAttr(1))
                                         : OpFoldResult(one));
  }
}

struct SlicedTokenLocalIndexRewrite {
  unsigned depIndex = 0;
  SmallVector<Value> sourceOffsets;
  SmallVector<Value> sourceSizes;
};

static inline bool containsValue(ValueRange values, Value value) {
  return llvm::any_of(values,
                      [&](Value existing) { return existing == value; });
}

static inline void
appendSlicedTokenOffsetParams(ArrayRef<SlicedTokenLocalIndexRewrite> rewrites,
                              SmallVectorImpl<Value> &params) {
  for (const SlicedTokenLocalIndexRewrite &rewrite : rewrites) {
    for (Value offset : rewrite.sourceOffsets) {
      if (!offset || getConstantIndexValue(offset))
        continue;
      if (!containsValue(params, offset))
        params.push_back(offset);
    }
  }
}

static inline void
appendDynamicIndexFoldResultParams(ArrayRef<OpFoldResult> values,
                                   SmallVectorImpl<Value> &params) {
  for (OpFoldResult valueOrAttr : values) {
    auto value = dyn_cast<Value>(valueOrAttr);
    if (!value || !isCodirScalarParamType(value.getType()) ||
        getConstantIndexValue(value) || containsValue(params, value))
      continue;
    params.push_back(value);
  }
}

static inline void
appendDynamicCodirDepSliceParams(ArrayRef<Value> deps,
                                 SmallVectorImpl<Value> &params) {
  for (Value dep : deps) {
    Operation *def = dep ? dep.getDefiningOp() : nullptr;
    if (auto subview = dyn_cast_or_null<memref::SubViewOp>(def)) {
      appendDynamicIndexFoldResultParams(subview.getMixedOffsets(), params);
      appendDynamicIndexFoldResultParams(subview.getMixedSizes(), params);
      appendDynamicIndexFoldResultParams(subview.getMixedStrides(), params);
      continue;
    }
    if (auto subindex = dyn_cast_or_null<polygeist::SubIndexOp>(def)) {
      Value index = subindex.getIndex();
      if (isCodirScalarParamType(index.getType()) &&
          !getConstantIndexValue(index) && !containsValue(params, index))
        params.push_back(index);
      for (Value size : subindex.getSizes()) {
        if (!isCodirScalarParamType(size.getType()) ||
            getConstantIndexValue(size) || containsValue(params, size))
          continue;
        params.push_back(size);
      }
    }
  }
}

static inline Value getCodeletParamBlockArgument(codir::CodeletOp codelet,
                                                 Value param) {
  if (!param || codelet.getBody().empty())
    return {};

  Block &body = codelet.getBody().front();
  unsigned depCount = codelet.getDeps().size();
  if (body.getNumArguments() < depCount + codelet.getParams().size())
    return {};

  for (auto [idx, operand] : llvm::enumerate(codelet.getParams()))
    if (operand == param)
      return body.getArgument(depCount + static_cast<unsigned>(idx));
  return {};
}

static inline FailureOr<Value>
materializeCodeletOffset(codir::CodeletOp codelet, Value sourceOffset) {
  if (!sourceOffset)
    return failure();

  if (std::optional<int64_t> constant = getConstantIndexValue(sourceOffset)) {
    Block &body = codelet.getBody().front();
    OpBuilder builder(&body, body.begin());
    return createConstantIndex(builder, sourceOffset.getLoc(), *constant);
  }

  if (Value arg = getCodeletParamBlockArgument(codelet, sourceOffset))
    return arg;

  return failure();
}

static inline bool isCodeletParamDerivedIndex(Value index,
                                              codir::CodeletOp codelet,
                                              ValueRange ignoredParams) {
  if (!index || !isa<IndexType>(index.getType()) || codelet.getBody().empty())
    return false;

  Block &body = codelet.getBody().front();
  unsigned depCount = codelet.getDeps().size();
  for (unsigned argIdx = depCount, e = body.getNumArguments(); argIdx < e;
       ++argIdx) {
    unsigned paramIdx = argIdx - depCount;
    if (paramIdx < codelet.getParams().size() &&
        containsValue(ignoredParams, codelet.getParams()[paramIdx]))
      continue;
    BlockArgument arg = body.getArgument(argIdx);
    if (isa<IndexType>(arg.getType()) &&
        ::mlir::carts::ValueAnalysis::dependsOn(index, arg))
      return true;

    auto indexArg = dyn_cast<BlockArgument>(index);
    if (!indexArg)
      continue;
    auto loop =
        dyn_cast_or_null<scf::ForOp>(indexArg.getOwner()->getParentOp());
    if (!loop || loop.getInductionVar() != index)
      continue;
    if (::mlir::carts::ValueAnalysis::dependsOn(loop.getLowerBound(), arg) ||
        ::mlir::carts::ValueAnalysis::dependsOn(loop.getUpperBound(), arg) ||
        ::mlir::carts::ValueAnalysis::dependsOn(loop.getStep(), arg))
      return true;
  }
  return false;
}

static inline LogicalResult
rewriteTokenLocalAccesses(codir::CodeletOp codelet,
                          ArrayRef<SlicedTokenLocalIndexRewrite> rewrites) {
  if (rewrites.empty())
    return success();
  if (codelet.getBody().empty())
    return codelet.emitOpError()
           << "requires a body before token-local index rewrite";

  Block &body = codelet.getBody().front();
  DenseMap<unsigned, SmallVector<Value>> materializedOffsets;

  auto getOffset = [&](const SlicedTokenLocalIndexRewrite &rewrite)
      -> FailureOr<SmallVector<Value>> {
    auto existing = materializedOffsets.find(rewrite.depIndex);
    if (existing != materializedOffsets.end())
      return existing->second;

    SmallVector<Value> offsets;
    offsets.reserve(rewrite.sourceOffsets.size());
    for (Value sourceOffset : rewrite.sourceOffsets) {
      FailureOr<Value> offset = materializeCodeletOffset(codelet, sourceOffset);
      if (failed(offset))
        return failure();
      offsets.push_back(*offset);
    }

    materializedOffsets[rewrite.depIndex] = offsets;
    return offsets;
  };

  auto rewriteIndexValue = [&](Operation *op, Value index, Value sourceOffset,
                               Value offset,
                               ValueRange ignoredParams) -> FailureOr<Value> {
    if (std::optional<int64_t> constant = getConstantIndexValue(sourceOffset);
        constant && *constant == 0)
      return index;

    OpBuilder builder(op);
    std::optional<int64_t> indexConstant = getConstantIndexValue(index);
    std::optional<int64_t> offsetConstant = getConstantIndexValue(sourceOffset);
    if (indexConstant && offsetConstant)
      return createConstantIndex(builder, op->getLoc(),
                                 *indexConstant - *offsetConstant);

    if (::mlir::carts::ValueAnalysis::sameValue(index, offset))
      return createZeroIndex(builder, op->getLoc());

    if (!isCodeletParamDerivedIndex(index, codelet, ignoredParams))
      return index;

    return arith::SubIOp::create(builder, op->getLoc(), index, offset)
        .getResult();
  };

  auto rewriteIndices = [&](Operation *op, Value memref,
                            MutableOperandRange indices) -> LogicalResult {
    if (indices.empty())
      return success();

    for (const SlicedTokenLocalIndexRewrite &rewrite : rewrites) {
      if (rewrite.depIndex >= codelet.getDeps().size() ||
          rewrite.depIndex >= body.getNumArguments())
        return codelet.emitOpError()
               << "has malformed sliced-token dependency metadata";

      if (memref != body.getArgument(rewrite.depIndex))
        continue;

      if (indices.size() != rewrite.sourceOffsets.size())
        return success();

      FailureOr<SmallVector<Value>> offsets = getOffset(rewrite);
      if (failed(offsets))
        return codelet.emitOpError()
               << "cannot materialize sliced token offset inside isolated "
                  "CODIR codelet";

      SmallVector<Value> rewrittenIndices;
      rewrittenIndices.reserve(indices.size());
      for (auto [indexOperand, sourceOffset, offset] :
           llvm::zip(indices, rewrite.sourceOffsets, *offsets)) {
        Value index = indexOperand.get();
        FailureOr<Value> rewritten = rewriteIndexValue(
            op, index, sourceOffset, offset, rewrite.sourceSizes);
        if (failed(rewritten))
          return failure();
        rewrittenIndices.push_back(*rewritten);
      }

      indices.assign(rewrittenIndices);
      return success();
    }

    return success();
  };

  auto rewriteSubview = [&](memref::SubViewOp subview) -> LogicalResult {
    for (const SlicedTokenLocalIndexRewrite &rewrite : rewrites) {
      if (rewrite.depIndex >= codelet.getDeps().size() ||
          rewrite.depIndex >= body.getNumArguments())
        return codelet.emitOpError()
               << "has malformed sliced-token dependency metadata";
      if (subview.getSource() != body.getArgument(rewrite.depIndex))
        continue;
      if (subview.getMixedOffsets().size() != rewrite.sourceOffsets.size())
        return success();

      FailureOr<SmallVector<Value>> offsets = getOffset(rewrite);
      if (failed(offsets))
        return codelet.emitOpError()
               << "cannot materialize sliced token offset inside isolated "
                  "CODIR codelet";

      SmallVector<OpFoldResult> rewrittenOffsets;
      rewrittenOffsets.reserve(subview.getMixedOffsets().size());
      bool changed = false;
      for (auto [mixedOffset, sourceOffset, offset] : llvm::zip(
               subview.getMixedOffsets(), rewrite.sourceOffsets, *offsets)) {
        if (auto attr = dyn_cast<Attribute>(mixedOffset)) {
          auto integerAttr = dyn_cast<IntegerAttr>(attr);
          std::optional<int64_t> sourceConstant =
              getConstantIndexValue(sourceOffset);
          if (integerAttr && sourceConstant && *sourceConstant != 0) {
            rewrittenOffsets.push_back(
                OpFoldResult(OpBuilder(subview).getIndexAttr(
                    integerAttr.getInt() - *sourceConstant)));
            changed = true;
            continue;
          }
          rewrittenOffsets.push_back(mixedOffset);
          continue;
        }

        Value offsetValue = cast<Value>(mixedOffset);
        FailureOr<Value> rewritten =
            rewriteIndexValue(subview.getOperation(), offsetValue, sourceOffset,
                              offset, rewrite.sourceSizes);
        if (failed(rewritten))
          return failure();
        changed |= *rewritten != offsetValue;
        rewrittenOffsets.push_back(*rewritten);
      }

      if (!changed)
        return success();

      OpBuilder builder(subview);
      auto resultType = memref::SubViewOp::inferResultType(
          cast<MemRefType>(subview.getSource().getType()), rewrittenOffsets,
          subview.getMixedSizes(), subview.getMixedStrides());
      auto replacement = memref::SubViewOp::create(
          builder, subview.getLoc(), resultType, subview.getSource(),
          rewrittenOffsets, subview.getMixedSizes(), subview.getMixedStrides());
      subview.getResult().replaceAllUsesWith(replacement.getResult());
      subview.erase();
      return success();
    }
    return success();
  };

  auto rewriteSubindex = [&](polygeist::SubIndexOp subindex) -> LogicalResult {
    for (const SlicedTokenLocalIndexRewrite &rewrite : rewrites) {
      if (rewrite.depIndex >= codelet.getDeps().size() ||
          rewrite.depIndex >= body.getNumArguments())
        return codelet.emitOpError()
               << "has malformed sliced-token dependency metadata";
      if (subindex.getSource() != body.getArgument(rewrite.depIndex))
        continue;
      if (rewrite.sourceOffsets.empty())
        return success();

      FailureOr<SmallVector<Value>> offsets = getOffset(rewrite);
      if (failed(offsets))
        return codelet.emitOpError()
               << "cannot materialize sliced token offset inside isolated "
                  "CODIR codelet";

      Value index = subindex.getIndex();
      FailureOr<Value> rewritten = rewriteIndexValue(
          subindex.getOperation(), index, rewrite.sourceOffsets.front(),
          offsets->front(), rewrite.sourceSizes);
      if (failed(rewritten))
        return failure();
      if (*rewritten == index)
        return success();

      OpBuilder builder(subindex);
      auto replacement = polygeist::SubIndexOp::create(
          builder, subindex.getLoc(), subindex.getType(), subindex.getSource(),
          *rewritten);
      subindex.getResult().replaceAllUsesWith(replacement.getResult());
      subindex.erase();
      return success();
    }
    return success();
  };

  SmallVector<memref::SubViewOp> subviews;
  SmallVector<polygeist::SubIndexOp> subindices;
  body.walk([&](memref::SubViewOp op) { subviews.push_back(op); });
  body.walk([&](polygeist::SubIndexOp op) { subindices.push_back(op); });
  for (memref::SubViewOp subview : subviews)
    if (failed(rewriteSubview(subview)))
      return failure();
  for (polygeist::SubIndexOp subindex : subindices)
    if (failed(rewriteSubindex(subindex)))
      return failure();

  WalkResult result = body.walk([&](Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op))
      return failed(
                 rewriteIndices(op, load.getMemref(), load.getIndicesMutable()))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    if (auto store = dyn_cast<memref::StoreOp>(op))
      return failed(rewriteIndices(op, store.getMemref(),
                                   store.getIndicesMutable()))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}

static inline Value stripCodirViewOps(Value value) {
  for (;;) {
    Operation *def = value ? value.getDefiningOp() : nullptr;
    if (auto subview = dyn_cast_or_null<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = dyn_cast_or_null<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto subindex = dyn_cast_or_null<polygeist::SubIndexOp>(def)) {
      value = subindex.getSource();
      continue;
    }
    return value;
  }
}

static inline bool isCodirViewDep(Value value) {
  Operation *def = value ? value.getDefiningOp() : nullptr;
  return isa_and_nonnull<memref::SubViewOp, polygeist::SubIndexOp>(def);
}

static inline bool
hasConservativeReadWriteConflict(ArrayRef<Value> deps,
                                 ArrayRef<codir::CodirAccessMode> modes) {
  if (deps.size() != modes.size())
    return true;

  DenseMap<Value, codir::CodirAccessMode> rootModes;
  for (auto [dep, mode] : llvm::zip(deps, modes)) {
    if (mode == codir::CodirAccessMode::readwrite)
      return true;

    Value root = stripCodirViewOps(dep);
    auto [it, inserted] = rootModes.try_emplace(root, mode);
    if (!inserted) {
      codir::CodirAccessMode merged = mergeAccessMode(it->second, mode);
      if (merged == codir::CodirAccessMode::readwrite)
        return true;
      it->second = merged;
    }
  }

  return false;
}

static inline LogicalResult materializeCodirDeps(
    sde::SdeCuCodeletOp sdeCodelet, OpBuilder &builder,
    SmallVectorImpl<Value> &deps, SmallVectorImpl<Attribute> &modes,
    SmallVectorImpl<Attribute> &storageViews,
    SmallVectorImpl<SlicedTokenLocalIndexRewrite> &localIndexRewrites) {
  deps.clear();
  deps.reserve(sdeCodelet.getTokens().size());
  modes.clear();
  modes.reserve(sdeCodelet.getTokens().size());
  storageViews.clear();
  storageViews.reserve(sdeCodelet.getTokens().size());
  localIndexRewrites.clear();

  for (auto [index, token] : llvm::enumerate(sdeCodelet.getTokens())) {
    auto tokenOp = token.getDefiningOp<sde::SdeMuTokenOp>();
    if (!tokenOp)
      return sdeCodelet.emitOpError()
             << "convert-sde-to-codir requires token operand #" << index
             << " to be defined by sde.mu_token";

    auto tokenType = dyn_cast<sde::TokenType>(token.getType());
    if (!tokenType)
      return sdeCodelet.emitOpError()
             << "convert-sde-to-codir requires token operand #" << index
             << " to have !sde.token type";

    if (tokenOp.getOffsets().empty() && tokenOp.getSizes().empty()) {
      if (tokenType.getSliceType() != tokenOp.getSource().getType()) {
        return sdeCodelet.emitOpError()
               << "convert-sde-to-codir requires whole-storage sde.mu_token "
                  "operand #"
               << index << " to have source type matching token slice type";
      }
      deps.push_back(tokenOp.getSource());
      modes.push_back(codir::CodirAccessModeAttr::get(
          builder.getContext(), convertAccessMode(tokenOp.getMode())));
      codir::CodirStorageViewKind view =
          tokenOp.getStorageViewAttr()
              ? convertStorageViewKind(tokenOp.getStorageViewAttr().getValue())
              : codir::CodirStorageViewKind::host_whole;
      storageViews.push_back(
          codir::CodirStorageViewKindAttr::get(builder.getContext(), view));
      continue;
    }

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    auto sliceType = cast<MemRefType>(tokenType.getSliceType());
    buildCodirSubviewMixedOperands(sliceType, tokenOp.getOffsets(),
                                   tokenOp.getSizes(), builder,
                                   tokenOp.getLoc(), offsets, sizes, strides);
    auto subview = memref::SubViewOp::create(
        builder, tokenOp.getLoc(), tokenType.getSliceType(),
        tokenOp.getSource(), offsets, sizes, strides);
    deps.push_back(subview.getResult());
    modes.push_back(codir::CodirAccessModeAttr::get(
        builder.getContext(), convertAccessMode(tokenOp.getMode())));
    codir::CodirStorageViewKind view =
        tokenOp.getStorageViewAttr()
            ? convertStorageViewKind(tokenOp.getStorageViewAttr().getValue())
            : codir::CodirStorageViewKind::compute_block;
    storageViews.push_back(
        codir::CodirStorageViewKindAttr::get(builder.getContext(), view));

    if (sliceType.getRank() > 0 &&
        tokenOp.getOffsets().size() ==
            static_cast<size_t>(sliceType.getRank()) &&
        tokenOp.getSizes().size() == static_cast<size_t>(sliceType.getRank()))
      localIndexRewrites.push_back(
          {static_cast<unsigned>(index),
           SmallVector<Value>(tokenOp.getOffsets().begin(),
                              tokenOp.getOffsets().end()),
           SmallVector<Value>(tokenOp.getSizes().begin(),
                              tokenOp.getSizes().end())});
  }

  return success();
}

struct CodirTaskPlan {
  SmallVector<Value> deps;
  SmallVector<codir::CodirAccessMode> depModes;
  SmallVector<codir::CodirStorageViewKind> depStorageViews;
  SmallVector<Value> params;
  SmallVector<Value> scalarMemrefParams;
  SmallVector<SlicedTokenLocalIndexRewrite> localIndexRewrites;
  llvm::SetVector<Value> cloneCaptures;
  DenseMap<Value, unsigned> depIndex;
  DenseMap<Value, unsigned> paramIndex;
  DenseMap<Value, unsigned> scalarMemrefParamIndex;
  DenseMap<Value, unsigned> slicedSourceDepIndex;
  DenseMap<Value, unsigned> exactSubviewDepIndex;
  DenseMap<Value, unsigned> exactSubindexDepIndex;
  DenseMap<Operation *, unsigned> exactRootAccessDepIndex;
};

static inline bool isDefinedInside(Value value, Region &region) {
  Region *valueRegion = value.getParentRegion();
  return valueRegion && region.isAncestor(valueRegion);
}

static inline LogicalResult addTaskDep(Value dep, codir::CodirAccessMode mode,
                                       codir::CodirStorageViewKind storageView,
                                       CodirTaskPlan &plan) {
  if (!isCodirDependencyType(dep.getType()))
    return failure();

  auto [it, inserted] = plan.depIndex.try_emplace(dep, plan.deps.size());
  if (inserted) {
    plan.deps.push_back(dep);
    plan.depModes.push_back(mode);
    plan.depStorageViews.push_back(storageView);
    return success();
  }

  unsigned index = it->second;
  plan.depModes[index] = mergeAccessMode(plan.depModes[index], mode);
  plan.depStorageViews[index] =
      mergeStorageViewKind(plan.depStorageViews[index], storageView);
  return success();
}

static inline LogicalResult addTaskDep(Value dep, codir::CodirAccessMode mode,
                                       CodirTaskPlan &plan) {
  return addTaskDep(dep, mode, codir::CodirStorageViewKind::host_whole, plan);
}

static inline LogicalResult addTaskParam(Value param, CodirTaskPlan &plan) {
  if (!isCodirScalarParamType(param.getType()))
    return failure();
  if (!plan.paramIndex.try_emplace(param, plan.params.size()).second)
    return success();
  plan.params.push_back(param);
  return success();
}

static inline bool isCloneableTaskCaptureDef(Operation *op) {
  if (!op || op->getNumRegions() != 0)
    return false;

  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;

  StringRef name = op->getName().getStringRef();
  return name == "llvm.mlir.addressof" || name == "llvm.getelementptr" ||
         name == "polygeist.undef";
}

static inline bool isCloneableSuScalarCaptureDef(Operation *op) {
  if (isCloneableTaskCaptureDef(op))
    return true;
  if (!op || op->getNumRegions() != 0 || !isSideEffectFreeArithmeticLikeOp(op))
    return false;
  return llvm::all_of(op->getResults(), [](Value result) {
    return isCodirScalarParamType(result.getType());
  });
}

static inline bool isScalarMemrefCaptureType(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType)
    return false;
  if (memrefType.getRank() == 0)
    return isCodirScalarParamType(memrefType.getElementType());
  if (memrefType.getRank() != 1 || memrefType.isDynamicDim(0) ||
      memrefType.getDimSize(0) != 1)
    return false;
  return isCodirScalarParamType(memrefType.getElementType());
}

static inline bool hasOnlyScalarLoadsInTask(Value memref, Region &taskRegion) {
  if (!isScalarMemrefCaptureType(memref.getType()))
    return false;

  for (Operation *user : memref.getUsers()) {
    if (!taskRegion.isAncestor(user->getParentRegion()))
      continue;

    auto load = dyn_cast<memref::LoadOp>(user);
    if (!load || load.getMemref() != memref)
      return false;

    auto memrefType = cast<MemRefType>(memref.getType());
    if (memrefType.getRank() == 0) {
      if (!load.getIndices().empty())
        return false;
      continue;
    }

    if (load.getIndices().size() != 1)
      return false;
    std::optional<int64_t> index = getConstantIndexValue(load.getIndices()[0]);
    if (!index || *index != 0)
      return false;
  }

  return true;
}

static inline LogicalResult addScalarMemrefParam(Value memref,
                                                 CodirTaskPlan &plan) {
  if (!isScalarMemrefCaptureType(memref.getType()))
    return failure();
  if (!plan.scalarMemrefParamIndex
           .try_emplace(memref, plan.scalarMemrefParams.size())
           .second)
    return success();
  plan.scalarMemrefParams.push_back(memref);
  return success();
}

static inline bool hasOnlyLocalizableMemrefUsesInTask(Value memref,
                                                      Region &taskRegion) {
  bool sawAccess = false;
  for (Operation *user : memref.getUsers()) {
    if (!taskRegion.isAncestor(user->getParentRegion()))
      continue;

    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      if (load.getMemref() != memref)
        return false;
      sawAccess = true;
      continue;
    }

    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (store.getMemref() != memref)
        return false;
      sawAccess = true;
      continue;
    }

    if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
      if (subview.getSource() != memref ||
          !hasOnlyDirectLoadStoreUsersInTask(subview.getResult(), taskRegion))
        return false;
      sawAccess = true;
      continue;
    }

    if (auto subindex = dyn_cast<polygeist::SubIndexOp>(user)) {
      if (subindex.getSource() != memref ||
          !hasOnlyDirectLoadStoreUsersInTask(subindex.getResult(), taskRegion))
        return false;
      sawAccess = true;
      continue;
    }

    return false;
  }

  return sawAccess;
}

static inline FailureOr<Value> materializeTaskDepView(sde::SdeMuDepOp muDep,
                                                      OpBuilder &builder) {
  auto sourceType = dyn_cast<MemRefType>(muDep.getSource().getType());
  if (!sourceType || !hasCompleteMuDepSlice(muDep))
    return failure();

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  offsets.reserve(sourceType.getRank());
  sizes.reserve(sourceType.getRank());
  strides.reserve(sourceType.getRank());

  for (Value offset : muDep.getOffsets())
    offsets.push_back(offset);
  for (Value size : muDep.getSizes())
    sizes.push_back(size);
  for (int64_t dim = 0, rank = sourceType.getRank(); dim < rank; ++dim)
    strides.push_back(builder.getIndexAttr(1));

  return memref::SubViewOp::create(builder, muDep.getLoc(), muDep.getSource(),
                                   offsets, sizes, strides)
      .getResult();
}

static inline bool canCloneTaskCapture(Value value, Region &taskRegion,
                                       llvm::DenseSet<Value> &visited) {
  if (!value || isDefinedInside(value, taskRegion))
    return true;

  if (!visited.insert(value).second)
    return true;

  Operation *def = value.getDefiningOp();
  if (!isCloneableTaskCaptureDef(def))
    return false;

  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return canCloneTaskCapture(operand, taskRegion, visited);
  });
}

static inline bool canCloneTaskCapture(Value value, Region &taskRegion) {
  llvm::DenseSet<Value> visited;
  return canCloneTaskCapture(value, taskRegion, visited);
}

static inline void
collectTaskCaptureClone(Value value, Region &taskRegion,
                        llvm::SetVector<Value> &cloneCaptures) {
  if (!value || isDefinedInside(value, taskRegion))
    return;

  Operation *def = value.getDefiningOp();
  if (!isCloneableTaskCaptureDef(def))
    return;

  for (Value operand : def->getOperands())
    collectTaskCaptureClone(operand, taskRegion, cloneCaptures);
  cloneCaptures.insert(value);
}

static inline LogicalResult buildCodirTaskPlan(sde::SdeCuTaskOp task,
                                               OpBuilder &builder,
                                               CodirTaskPlan &plan) {
  DenseMap<Value, unsigned> depSourceCounts;
  DenseMap<Value, sde::SdeMuDepOp> depSourceRepresentative;
  DenseMap<Value, SmallVector<sde::SdeMuDepOp>> depSourceOps;
  DenseSet<Value> depSourcesWithMixedSlices;
  for (Value dep : task.getDeps()) {
    if (auto muDep = dep.getDefiningOp<sde::SdeMuDepOp>()) {
      ++depSourceCounts[muDep.getSource()];
      depSourceOps[muDep.getSource()].push_back(muDep);
      auto [it, inserted] =
          depSourceRepresentative.try_emplace(muDep.getSource(), muDep);
      if (!inserted && !haveSameMuDepSlice(it->second, muDep))
        depSourcesWithMixedSlices.insert(muDep.getSource());
    }
  }

  DenseMap<Value, bool> depSourceHasPartitionedAccessProof;
  for (auto &[source, sourceDeps] : depSourceOps) {
    if (!depSourcesWithMixedSlices.contains(source))
      continue;
    depSourceHasPartitionedAccessProof[source] =
        hasPartitionedExactAccessProofInTask(source, sourceDeps,
                                             task.getBody());
  }

  SmallVector<std::pair<sde::SdeMuDepOp, Value>, 4> slicedDepViews;
  auto findSlicedDepView = [&](sde::SdeMuDepOp muDep) -> Value {
    for (auto &[existingDep, view] : slicedDepViews)
      if (haveSameMuDepSlice(existingDep, muDep))
        return view;
    return {};
  };

  for (Value dep : task.getDeps()) {
    auto muDep = dep.getDefiningOp<sde::SdeMuDepOp>();
    if (!muDep)
      return task.emitOpError()
             << "convert-sde-to-codir requires cu_task dependency to be "
                "defined by sde.mu_dep, got "
             << dep.getType();

    Value codirDep = muDep.getSource();
    bool sourceHasMixedSlices =
        depSourcesWithMixedSlices.contains(muDep.getSource());
    bool duplicateSourceHasSameSlice =
        depSourceCounts.lookup(muDep.getSource()) > 1 && !sourceHasMixedSlices;
    bool hasStaticLocalizableSlice =
        hasCodirTaskDepSliceBoundsSupport(muDep) &&
        hasOnlyLocalizableMemrefUsesInTask(muDep.getSource(), task.getBody());
    bool hasExactDynamicSubviewProof =
        hasCompleteMuDepSlice(muDep) && !hasOnlyStaticMuDepSliceBounds(muDep) &&
        hasExactSubviewAccessProofInTask(muDep, task.getBody());
    bool hasExactDynamicRootAccessProof =
        hasCompleteMuDepSlice(muDep) && !hasOnlyStaticMuDepSliceBounds(muDep) &&
        hasExactRootAccessProofInTask(muDep, task.getBody());
    bool hasPartitionedAccessProof =
        depSourceHasPartitionedAccessProof.lookup(muDep.getSource());
    bool useTokenLocalView =
        ((!sourceHasMixedSlices &&
          (depSourceCounts.lookup(muDep.getSource()) == 1 ||
           duplicateSourceHasSameSlice) &&
          (hasStaticLocalizableSlice || hasExactDynamicSubviewProof ||
           hasExactDynamicRootAccessProof)) ||
         (sourceHasMixedSlices && hasCompleteMuDepSlice(muDep) &&
          hasPartitionedAccessProof));
    if (useTokenLocalView) {
      if (Value existing = findSlicedDepView(muDep)) {
        codirDep = existing;
      } else {
        FailureOr<Value> view = materializeTaskDepView(muDep, builder);
        if (failed(view))
          return muDep.emitOpError()
                 << "failed to materialize CODIR task dependency view";
        codirDep = *view;
        slicedDepViews.push_back({muDep, codirDep});
      }
    }

    codir::CodirStorageViewKind storageView =
        muDep.getStorageViewAttr()
            ? convertStorageViewKind(muDep.getStorageViewAttr().getValue())
            : (useTokenLocalView ? codir::CodirStorageViewKind::compute_block
                                 : codir::CodirStorageViewKind::host_whole);
    if (failed(addTaskDep(codirDep, convertAccessMode(muDep.getMode()),
                          storageView, plan)))
      return muDep.emitOpError()
             << "source must be a memref for CODIR task dependency";

    if (useTokenLocalView) {
      unsigned depIndex = plan.depIndex.lookup(codirDep);
      if (sourceHasMixedSlices && hasPartitionedAccessProof) {
        SmallVector<memref::SubViewOp, 4> matchingSubviews;
        collectExactSubviewAccessProofsInTask(muDep, task.getBody(),
                                              matchingSubviews);
        for (memref::SubViewOp subview : matchingSubviews)
          plan.exactSubviewDepIndex.try_emplace(subview.getResult(), depIndex);
        SmallVector<polygeist::SubIndexOp, 4> matchingSubindices;
        collectExactSubindexAccessProofsInTask(muDep, task.getBody(),
                                               matchingSubindices);
        for (polygeist::SubIndexOp subindex : matchingSubindices)
          plan.exactSubindexDepIndex.try_emplace(subindex.getResult(),
                                                 depIndex);
        SmallVector<Operation *, 4> matchingRootAccesses;
        collectExactRootAccessProofsInTask(muDep, task.getBody(),
                                           matchingRootAccesses);
        for (Operation *access : matchingRootAccesses)
          plan.exactRootAccessDepIndex.try_emplace(access, depIndex);
      } else if (plan.slicedSourceDepIndex
                     .try_emplace(muDep.getSource(), depIndex)
                     .second) {
        plan.localIndexRewrites.push_back(
            {depIndex,
             SmallVector<Value>(muDep.getOffsets().begin(),
                                muDep.getOffsets().end()),
             SmallVector<Value>(muDep.getSizes().begin(),
                                muDep.getSizes().end())});
      }
    }
  }

  Region &taskRegion = task.getBody();
  WalkResult walkResult = taskRegion.walk([&](Operation *nested) {
    if (isa<sde::SdeYieldOp>(nested))
      return WalkResult::advance();
    if (auto subview = dyn_cast<memref::SubViewOp>(nested);
        subview && plan.exactSubviewDepIndex.contains(subview.getResult()))
      return WalkResult::advance();
    if (auto subindex = dyn_cast<polygeist::SubIndexOp>(nested);
        subindex && plan.exactSubindexDepIndex.contains(subindex.getResult()))
      return WalkResult::advance();
    bool exactRootAccess = plan.exactRootAccessDepIndex.contains(nested);
    if (exactRootAccess && isa<memref::LoadOp>(nested))
      return WalkResult::advance();
    for (Value operand : nested->getOperands()) {
      if (exactRootAccess) {
        if (auto store = dyn_cast<memref::StoreOp>(nested)) {
          if (operand == store.getMemref() ||
              llvm::is_contained(store.getIndices(), operand))
            continue;
        }
      }
      if (isDefinedInside(operand, taskRegion))
        continue;
      if (plan.depIndex.contains(operand) || plan.paramIndex.contains(operand))
        continue;
      if (plan.slicedSourceDepIndex.contains(operand)) {
        if (auto load = dyn_cast<memref::LoadOp>(nested);
            load && load.getMemref() == operand)
          continue;
        if (auto store = dyn_cast<memref::StoreOp>(nested);
            store && store.getMemref() == operand)
          continue;
        if (auto subview = dyn_cast<memref::SubViewOp>(nested);
            subview && subview.getSource() == operand)
          continue;
        if (auto subindex = dyn_cast<polygeist::SubIndexOp>(nested);
            subindex && subindex.getSource() == operand)
          continue;
      }

      if (isCodirDependencyType(operand.getType())) {
        if (hasOnlyScalarLoadsInTask(operand, taskRegion)) {
          if (failed(addScalarMemrefParam(operand, plan)))
            return WalkResult::interrupt();
          continue;
        }
        if (failed(
                addTaskDep(operand, codir::CodirAccessMode::readwrite, plan)))
          return WalkResult::interrupt();
        continue;
      }

      if (isCodirScalarParamType(operand.getType())) {
        if (failed(addTaskParam(operand, plan)))
          return WalkResult::interrupt();
        continue;
      }

      if (canCloneTaskCapture(operand, taskRegion)) {
        collectTaskCaptureClone(operand, taskRegion, plan.cloneCaptures);
        continue;
      }

      nested->emitOpError()
          << "operand cannot be captured by CODIR task-depend conversion; "
             "expected memref dependency or scalar param, got "
          << operand.getType();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  return success();
}

static inline LogicalResult convertCuTaskToCodir(sde::SdeCuTaskOp task) {
  if (task.getBody().empty())
    return task.emitOpError() << "expects a body before CODIR conversion";

  OpBuilder builder(task);
  CodirTaskPlan plan;
  if (failed(buildCodirTaskPlan(task, builder, plan)))
    return failure();

  DenseMap<Value, Value> scalarMemrefLoads;
  for (Value memref : plan.scalarMemrefParams) {
    auto memrefType = cast<MemRefType>(memref.getType());
    SmallVector<Value> indices;
    if (memrefType.getRank() == 1)
      indices.push_back(createZeroIndex(builder, task.getLoc()));
    Value scalar =
        memref::LoadOp::create(builder, task.getLoc(), memref, indices);
    scalarMemrefLoads[memref] = scalar;
    if (failed(addTaskParam(scalar, plan)))
      return task.emitOpError()
             << "failed to materialize scalar memref capture parameter";
  }
  appendSlicedTokenOffsetParams(plan.localIndexRewrites, plan.params);
  appendDynamicCodirDepSliceParams(plan.deps, plan.params);

  SmallVector<Attribute> depModeAttrs =
      buildCodirAccessModeAttrs(task.getContext(), plan.depModes);
  SmallVector<Attribute> depStorageViewAttrs =
      buildCodirStorageViewAttrs(task.getContext(), plan.depStorageViews);

  auto codelet = createCodirCodelet(
      builder, task.getLoc(), builder.getArrayAttr(depModeAttrs),
      builder.getArrayAttr(depStorageViewAttrs), plan.deps, plan.params,
      getCodirMetadataFromTask(task));
  if (!task.getDeps().empty())
    codelet.setTaskDependAttr(builder.getUnitAttr());
  bool hasSlicedTaskDepend =
      llvm::any_of(plan.deps, [](Value dep) { return isCodirViewDep(dep); });
  if (hasSlicedTaskDepend ||
      hasConservativeReadWriteConflict(plan.deps, plan.depModes))
    codelet.setOrderedTaskDependAttr(builder.getUnitAttr());

  Block *body = new Block();
  codelet.getBody().push_back(body);
  for (Value dep : plan.deps)
    body->addArgument(dep.getType(), task.getLoc());
  for (Value param : plan.params)
    body->addArgument(param.getType(), task.getLoc());

  IRMapping mapper;
  for (auto [index, dep] : llvm::enumerate(plan.deps))
    mapper.map(dep, body->getArgument(index));
  for (auto [source, depIndex] : plan.slicedSourceDepIndex)
    mapper.map(source, body->getArgument(depIndex));
  for (auto [subview, depIndex] : plan.exactSubviewDepIndex)
    mapper.map(subview, body->getArgument(depIndex));
  unsigned paramOffset = plan.deps.size();
  for (auto [index, param] : llvm::enumerate(plan.params))
    mapper.map(param, body->getArgument(paramOffset + index));

  DenseMap<Value, Value> scalarMemrefParamArgs;
  for (auto [memref, scalar] : scalarMemrefLoads) {
    Value scalarArg = mapper.lookupOrNull(scalar);
    if (!scalarArg)
      return task.emitOpError()
             << "failed to map scalar memref capture parameter";
    scalarMemrefParamArgs[memref] = scalarArg;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  std::function<LogicalResult(Value)> cloneCapture = [&](Value value) {
    if (!value || mapper.contains(value) ||
        isDefinedInside(value, task.getBody()))
      return success();

    Operation *def = value.getDefiningOp();
    if (!isCloneableTaskCaptureDef(def))
      return failure();

    for (Value operand : def->getOperands())
      if (failed(cloneCapture(operand)))
        return failure();

    Operation *cloned = builder.clone(*def, mapper);
    for (auto [oldResult, newResult] :
         llvm::zip(def->getResults(), cloned->getResults()))
      mapper.map(oldResult, newResult);
    return success();
  };
  for (Value capture : plan.cloneCaptures)
    if (failed(cloneCapture(capture)))
      return task.emitOpError()
             << "failed to clone CODIR task capture " << capture;

  auto rewriteScalarMemrefLoads = [&](Operation *cloned) -> LogicalResult {
    SmallVector<memref::LoadOp> loadsToErase;
    cloned->walk([&](memref::LoadOp load) {
      auto it = scalarMemrefParamArgs.find(load.getMemref());
      if (it == scalarMemrefParamArgs.end())
        return;
      load.getResult().replaceAllUsesWith(it->second);
      loadsToErase.push_back(load);
    });
    for (memref::LoadOp load : loadsToErase)
      load.erase();
    return success();
  };

  auto materializeZeroIndicesForDep = [&](Value dep) {
    SmallVector<Value> indices;
    auto depType = cast<MemRefType>(dep.getType());
    indices.reserve(depType.getRank());
    for (int64_t dim = 0, rank = depType.getRank(); dim < rank; ++dim)
      indices.push_back(createZeroIndex(builder, task.getLoc()));
    return indices;
  };

  for (Operation &nested : task.getBody().front()) {
    if (isa<sde::SdeYieldOp>(&nested))
      continue;
    if (auto subview = dyn_cast<memref::SubViewOp>(&nested);
        subview && plan.exactSubviewDepIndex.contains(subview.getResult()))
      continue;
    if (auto subindex = dyn_cast<polygeist::SubIndexOp>(&nested)) {
      auto exactSubindex =
          plan.exactSubindexDepIndex.find(subindex.getResult());
      if (exactSubindex != plan.exactSubindexDepIndex.end()) {
        Value depArg = body->getArgument(exactSubindex->second);
        Value zero = createZeroIndex(builder, subindex.getLoc());
        SmallVector<Value> sizes;
        sizes.reserve(subindex.getSizes().size());
        for (Value size : subindex.getSizes())
          sizes.push_back(mapper.lookupOrDefault(size));
        auto localSubindex = polygeist::SubIndexOp::create(
            builder, subindex.getLoc(), subindex.getType(), depArg, zero,
            sizes);
        mapper.map(subindex.getResult(), localSubindex.getResult());
        continue;
      }
    }
    if (auto load = dyn_cast<memref::LoadOp>(nested)) {
      auto exactAccess = plan.exactRootAccessDepIndex.find(&nested);
      if (exactAccess != plan.exactRootAccessDepIndex.end()) {
        Value depArg = body->getArgument(exactAccess->second);
        auto localLoad =
            memref::LoadOp::create(builder, load.getLoc(), depArg,
                                   materializeZeroIndicesForDep(depArg));
        mapper.map(load.getResult(), localLoad.getResult());
        continue;
      }
    }
    if (auto store = dyn_cast<memref::StoreOp>(nested)) {
      auto exactAccess = plan.exactRootAccessDepIndex.find(&nested);
      if (exactAccess != plan.exactRootAccessDepIndex.end()) {
        Value value = mapper.lookupOrDefault(store.getValue());
        Value depArg = body->getArgument(exactAccess->second);
        memref::StoreOp::create(builder, store.getLoc(), value, depArg,
                                materializeZeroIndicesForDep(depArg));
        continue;
      }
    }
    if (auto load = dyn_cast<memref::LoadOp>(nested)) {
      Value memref = load.getMemref();
      auto it = scalarMemrefLoads.find(memref);
      if (it != scalarMemrefLoads.end()) {
        mapper.map(load.getResult(), mapper.lookup(it->second));
        continue;
      }
    }
    Operation *cloned = builder.insert(nested.clone(mapper));
    if (failed(rewriteScalarMemrefLoads(cloned)))
      return task.emitOpError()
             << "failed to rewrite scalar memref captures in CODIR task body";
  }
  codir::YieldOp::create(builder, task.getLoc(), ValueRange{});
  if (failed(rewriteTokenLocalAccesses(codelet, plan.localIndexRewrites)))
    return task.emitOpError()
           << "failed to rewrite sliced task dependency accesses";

  SmallVector<sde::SdeMuDepOp> muDeps;
  for (Value dep : task.getDeps())
    if (auto muDep = dep.getDefiningOp<sde::SdeMuDepOp>())
      muDeps.push_back(muDep);

  task.erase();
  for (sde::SdeMuDepOp muDep : muDeps)
    if (muDep->use_empty())
      muDep.erase();
  return success();
}

static inline std::optional<int64_t> getPositiveI64(ArrayAttr attr,
                                                    unsigned index) {
  if (!attr || index >= attr.size())
    return std::nullopt;
  auto integer = dyn_cast<IntegerAttr>(attr[index]);
  if (!integer || integer.getInt() <= 0)
    return std::nullopt;
  return integer.getInt();
}

static inline std::optional<int64_t> getFirstPositiveI64(ArrayAttr attr) {
  return getPositiveI64(attr, 0);
}

static inline bool hasOwnerStripTopology(sde::SdeSuIterateOp source) {
  auto topology = source.getIterationTopology();
  return topology && *topology == sde::SdeIterationTopology::owner_strip;
}

static inline bool hasOwnerTileTopology(sde::SdeSuIterateOp source) {
  auto topology = source.getIterationTopology();
  return topology && (*topology == sde::SdeIterationTopology::owner_tile ||
                      *topology == sde::SdeIterationTopology::owner_tile_2d);
}

static inline Value buildSuDispatchStepFromExtent(sde::SdeSuIterateOp source,
                                                  Value step, int64_t extent,
                                                  OpBuilder &builder) {
  if (extent == 1)
    return step;
  Location loc = source.getLoc();
  std::optional<int64_t> stepConst = getConstantIndexValue(step);
  if (!stepConst)
    stepConst = ::mlir::carts::ValueAnalysis::tryFoldConstantIndex(step);
  if (stepConst) {
    if (*stepConst >= extent)
      return step;
    int64_t dispatch = ((*stepConst + extent - 1) / *stepConst) * *stepConst;
    return createConstantIndex(builder, loc, dispatch);
  }
  Value extentValue = createConstantIndex(builder, loc, extent);
  Value chunks = arith::CeilDivUIOp::create(builder, loc, extentValue, step);
  return arith::MulIOp::create(builder, loc, chunks, step);
}

static inline std::optional<unsigned>
inferSingleIvPhysicalAccessDim(sde::SdeSuIterateOp source) {
  if (!source || source.getBody().empty() ||
      source.getLowerBounds().size() != 1 ||
      source.getBody().front().getNumArguments() != 1)
    return std::nullopt;

  Value ownerIv = source.getBody().front().getArgument(0);
  std::optional<unsigned> selectedDim;
  bool rejected = false;

  auto observeAccess = [&](Value memref, OperandRange indices) {
    if (!memref || indices.empty())
      return;
    Value root = ::mlir::carts::ValueAnalysis::stripMemrefViewOps(memref);
    if (!root || isDefinedInside(root, source.getBody()))
      return;
    auto rootType = dyn_cast<MemRefType>(root.getType());
    if (!rootType || rootType.getRank() == 0 ||
        indices.size() != static_cast<size_t>(rootType.getRank()))
      return;

    for (auto [dim, index] : llvm::enumerate(indices)) {
      if (!indexSelectsOwnerSlice(index, ownerIv))
        continue;
      unsigned physicalDim = static_cast<unsigned>(dim);
      if (!selectedDim) {
        selectedDim = physicalDim;
      } else if (*selectedDim != physicalDim) {
        rejected = true;
      }
    }
  };

  source.getBody().walk([&](Operation *op) {
    if (rejected)
      return WalkResult::interrupt();
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (!isa<MemRefType>(load.getResult().getType()))
        observeAccess(load.getMemref(), load.getIndices());
      return rejected ? WalkResult::interrupt() : WalkResult::advance();
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (!isa<MemRefType>(store.getValueToStore().getType()))
        observeAccess(store.getMemref(), store.getIndices());
      return rejected ? WalkResult::interrupt() : WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (rejected)
    return std::nullopt;
  return selectedDim;
}

static inline std::optional<int64_t>
getOwnerStripDispatchExtent(sde::SdeSuIterateOp source, ArrayAttr extents) {
  if (!source || !extents)
    return std::nullopt;

  if (std::optional<SmallVector<int64_t, 4>> physicalOwnerDims =
          ::mlir::carts::readI64ArrayAttr(source.getPhysicalOwnerDimsAttr())) {
    if (physicalOwnerDims->size() == 1 && (*physicalOwnerDims)[0] >= 0) {
      unsigned physicalDim = static_cast<unsigned>((*physicalOwnerDims)[0]);
      if (std::optional<int64_t> extent = getPositiveI64(extents, physicalDim))
        return extent;
    }
  }

  if (std::optional<unsigned> physicalDim =
          inferSingleIvPhysicalAccessDim(source)) {
    if (std::optional<int64_t> extent = getPositiveI64(extents, *physicalDim))
      return extent;
  }

  return std::nullopt;
}

static inline Value buildSuDispatchStep(sde::SdeSuIterateOp source,
                                        OpBuilder &builder) {
  Value step = source.getSteps().front();
  if (hasOwnerStripTopology(source)) {
    if (std::optional<int64_t> block = getOwnerStripDispatchExtent(
            source, source.getPhysicalBlockShapeAttr()))
      return buildSuDispatchStepFromExtent(source, step, *block, builder);
    if (std::optional<int64_t> slice = getOwnerStripDispatchExtent(
            source, source.getLogicalWorkerSliceAttr()))
      return buildSuDispatchStepFromExtent(source, step, *slice, builder);
  }

  if (std::optional<int64_t> block =
          getFirstPositiveI64(source.getPhysicalBlockShapeAttr())) {
    return buildSuDispatchStepFromExtent(source, step, *block, builder);
  }

  if (hasOwnerStripTopology(source)) {
    if (std::optional<int64_t> slice =
            getFirstPositiveI64(source.getLogicalWorkerSliceAttr())) {
      return buildSuDispatchStepFromExtent(source, step, *slice, builder);
    }
  }

  if (Value chunk = source.getChunkSize()) {
    Location loc = source.getLoc();
    return arith::MulIOp::create(builder, loc, chunk, step);
  }

  return step;
}

static inline FailureOr<Value>
buildSuOwnerTileDispatchStep(sde::SdeSuIterateOp source, unsigned dim,
                             OpBuilder &builder) {
  if (dim >= source.getSteps().size())
    return failure();
  std::optional<int64_t> slice =
      getPositiveI64(source.getLogicalWorkerSliceAttr(), dim);
  if (!slice)
    return failure();
  return buildSuDispatchStepFromExtent(source, source.getSteps()[dim], *slice,
                                       builder);
}

struct SuCodeletPlan {
  SmallVector<Value> deps;
  SmallVector<codir::CodirAccessMode> depModes;
  SmallVector<codir::CodirStorageViewKind> depStorageViews;
  SmallVector<Value> params;
  llvm::SetVector<Value> cloneCaptures;
  llvm::SetVector<Value> localAllocaCaptures;
  SmallVector<SlicedTokenLocalIndexRewrite> localIndexRewrites;
  DenseMap<Value, unsigned> depIndex;
  DenseMap<Value, unsigned> paramIndex;
};

struct SuBarrierTokenDepPlan {
  DenseMap<Operation *, SmallVector<Value, 2>> tokenLocalRootsBySu;
  DenseSet<Operation *> suppressCompletionBarrierForSu;

  bool hasTokenLocalRoot(sde::SdeSuIterateOp source, Value root) const {
    auto it = tokenLocalRootsBySu.find(source.getOperation());
    if (it == tokenLocalRootsBySu.end())
      return false;
    return llvm::is_contained(it->second, root);
  }

  bool suppressCompletionBarrier(sde::SdeSuIterateOp source) const {
    return suppressCompletionBarrierForSu.contains(source.getOperation());
  }

  void addTokenLocalRoot(sde::SdeSuIterateOp source, Value root) {
    SmallVector<Value, 2> &roots = tokenLocalRootsBySu[source.getOperation()];
    if (!llvm::is_contained(roots, root))
      roots.push_back(root);
  }
};

static inline bool sameIndexValue(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  return ::mlir::carts::ValueAnalysis::sameValue(lhs, rhs);
}

static inline bool sameIndexValueRange(ValueRange lhs, ValueRange rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [lhsValue, rhsValue] : llvm::zip(lhs, rhs))
    if (!sameIndexValue(lhsValue, rhsValue))
      return false;
  return true;
}

static inline bool
hasEliminatedRequiredMemoryReason(sde::SdeSuBarrierOp barrier) {
  if (!barrier || !barrier.getBarrierEliminatedAttr())
    return false;
  if (!barrier.getTokens().empty())
    return false;
  auto reason = barrier.getBarrierReason();
  return reason && *reason == sde::SdeBarrierReason::required_memory;
}

static inline sde::SdeSuIterateOp
findSingleSuIterateInContainer(Operation *op) {
  if (!op)
    return {};
  if (auto iterate = dyn_cast<sde::SdeSuIterateOp>(op))
    return iterate;

  auto distribute = dyn_cast<sde::SdeSuDistributeOp>(op);
  if (!distribute || distribute.getBody().empty())
    return {};

  sde::SdeSuIterateOp selected;
  bool unsupportedWrapperOp = false;
  for (Operation &nested : distribute.getBody().front()) {
    if (nested.hasTrait<OpTrait::IsTerminator>() ||
        isa<sde::SdeYieldOp>(nested))
      continue;

    if (auto iterate = dyn_cast<sde::SdeSuIterateOp>(nested)) {
      if (selected && selected.getOperation() != iterate.getOperation()) {
        unsupportedWrapperOp = true;
        break;
      }
      selected = iterate;
      continue;
    }

    if (!isMemoryEffectFree(&nested)) {
      unsupportedWrapperOp = true;
      break;
    }
  }
  if (unsupportedWrapperOp)
    return {};
  return selected;
}

static inline sde::SdeSuIterateOp
findPreviousSiblingSuIterate(sde::SdeSuBarrierOp barrier) {
  Block *block = barrier ? barrier->getBlock() : nullptr;
  if (!block)
    return {};

  for (auto it = Block::reverse_iterator(barrier->getIterator());
       it != block->rend(); ++it) {
    if (auto iterate = findSingleSuIterateInContainer(&*it))
      return iterate;
    if (!isMemoryEffectFree(&*it))
      return {};
  }
  return {};
}

static inline sde::SdeSuIterateOp
findNextSiblingSuIterate(sde::SdeSuBarrierOp barrier) {
  Block *block = barrier ? barrier->getBlock() : nullptr;
  if (!block)
    return {};

  for (auto it = std::next(barrier->getIterator()); it != block->end(); ++it) {
    if (auto iterate = findSingleSuIterateInContainer(&*it))
      return iterate;
    if (!isMemoryEffectFree(&*it))
      return {};
  }
  return {};
}

static inline std::optional<Value> getSingleWriteReadIntermediate(
    const sde::StructuredMemoryEffectSummary &predEffects,
    const sde::StructuredMemoryEffectSummary &succEffects) {
  if (predEffects.hasUnknownEffects || succEffects.hasUnknownEffects)
    return std::nullopt;
  if (predEffects.writes.size() != 1)
    return std::nullopt;

  Value intermediate = *predEffects.writes.begin();
  Value intermediateRoot =
      ::mlir::carts::ValueAnalysis::stripMemrefViewOps(intermediate);
  if (!intermediateRoot || !intermediateRoot.getDefiningOp<sde::SdeMuAllocOp>())
    return std::nullopt;
  if (!intermediate || !succEffects.reads.contains(intermediate))
    return std::nullopt;
  if (succEffects.writes.contains(intermediate))
    return std::nullopt;

  for (Value read : predEffects.reads)
    if (succEffects.writes.contains(read))
      return std::nullopt;

  for (Value written : predEffects.writes)
    if (succEffects.writes.contains(written))
      return std::nullopt;

  return intermediate;
}

static inline bool
hasMatchingOneDimensionalIterationSpace(sde::SdeSuIterateOp predecessor,
                                        sde::SdeSuIterateOp successor) {
  if (!predecessor || !successor)
    return false;
  if (predecessor.getLowerBounds().size() != 1 ||
      predecessor.getUpperBounds().size() != 1 ||
      predecessor.getSteps().size() != 1 ||
      successor.getLowerBounds().size() != 1 ||
      successor.getUpperBounds().size() != 1 ||
      successor.getSteps().size() != 1)
    return false;

  return sameIndexValueRange(predecessor.getLowerBounds(),
                             successor.getLowerBounds()) &&
         sameIndexValueRange(predecessor.getUpperBounds(),
                             successor.getUpperBounds()) &&
         sameIndexValueRange(predecessor.getSteps(), successor.getSteps());
}

static inline bool hasSingleLeadingPhysicalOwnerDim(sde::SdeSuIterateOp op) {
  std::optional<SmallVector<int64_t, 4>> ownerDims =
      ::mlir::carts::readI64ArrayAttr(op.getPhysicalOwnerDimsAttr());
  return ownerDims && ownerDims->size() == 1 && (*ownerDims)[0] == 0;
}

static inline bool
hasAlignedBarrierTokenAccessWindow(sde::SdeSuIterateOp predecessor,
                                   sde::SdeSuIterateOp successor,
                                   Value intermediateRoot) {
  if (!intermediateRoot)
    return false;
  if (!hasMatchingOneDimensionalIterationSpace(predecessor, successor))
    return false;
  if (!hasSdePhysicalOwnerSlicePlan(predecessor) ||
      !hasSdePhysicalOwnerSlicePlan(successor))
    return false;
  if (!hasSingleLeadingPhysicalOwnerDim(predecessor) ||
      !hasSingleLeadingPhysicalOwnerDim(successor))
    return false;
  if (!canUseOwnerSliceBoundaryPlan(predecessor) ||
      !canUseOwnerSliceBoundaryPlan(successor))
    return false;
  if (!hasSamePhysicalLayoutPlan(predecessor, successor))
    return false;

  return canAccessRootWithPlan(predecessor, intermediateRoot, predecessor) &&
         canAccessRootWithPlan(successor, intermediateRoot, predecessor);
}

static inline void collectSuBarrierTokenDepPlans(ModuleOp module,
                                                 SuBarrierTokenDepPlan &plan) {
  SmallVector<sde::SdeSuBarrierOp> barriers;
  module.walk([&](sde::SdeSuBarrierOp barrier) {
    if (hasEliminatedRequiredMemoryReason(barrier))
      barriers.push_back(barrier);
  });

  for (sde::SdeSuBarrierOp barrier : barriers) {
    auto tryCollect = [&]() -> bool {
      sde::SdeSuIterateOp predecessor = findPreviousSiblingSuIterate(barrier);
      sde::SdeSuIterateOp successor = findNextSiblingSuIterate(barrier);
      if (!predecessor || !successor)
        return false;
      if (!predecessor.getStructuredClassificationAttr() ||
          !successor.getStructuredClassificationAttr())
        return false;

      auto predEffects =
          sde::collectStructuredMemoryEffects(predecessor.getOperation());
      auto succEffects =
          sde::collectStructuredMemoryEffects(successor.getOperation());
      std::optional<Value> intermediate =
          getSingleWriteReadIntermediate(predEffects, succEffects);
      if (!intermediate)
        return false;
      if (!hasAlignedBarrierTokenAccessWindow(predecessor, successor,
                                              *intermediate))
        return false;

      plan.addTokenLocalRoot(predecessor, *intermediate);
      plan.addTokenLocalRoot(successor, *intermediate);
      plan.suppressCompletionBarrierForSu.insert(predecessor.getOperation());
      return true;
    };

    if (!tryCollect())
      barrier->removeAttr(barrier.getBarrierEliminatedAttrName());
  }
}

static inline LogicalResult addSuDep(Value dep, codir::CodirAccessMode mode,
                                     codir::CodirStorageViewKind storageView,
                                     SuCodeletPlan &plan) {
  CodirTaskPlan taskPlan;
  taskPlan.deps = plan.deps;
  taskPlan.depModes = plan.depModes;
  taskPlan.depStorageViews = plan.depStorageViews;
  taskPlan.depIndex = plan.depIndex;
  if (failed(addTaskDep(dep, mode, storageView, taskPlan)))
    return failure();
  plan.deps = std::move(taskPlan.deps);
  plan.depModes = std::move(taskPlan.depModes);
  plan.depStorageViews = std::move(taskPlan.depStorageViews);
  plan.depIndex = std::move(taskPlan.depIndex);
  return success();
}

static inline LogicalResult addSuDep(Value dep, codir::CodirAccessMode mode,
                                     SuCodeletPlan &plan) {
  return addSuDep(dep, mode, codir::CodirStorageViewKind::host_whole, plan);
}

static inline bool canCloneSuScalarCapture(Value value,
                                           sde::SdeSuIterateOp source,
                                           llvm::DenseSet<Value> &visited) {
  if (!value || isDefinedInside(value, source.getBody()))
    return true;
  if (!visited.insert(value).second)
    return true;

  Operation *def = value.getDefiningOp();
  if (!isCloneableSuScalarCaptureDef(def))
    return false;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return canCloneSuScalarCapture(operand, source, visited);
  });
}

static inline bool canCloneSuScalarCapture(Value value,
                                           sde::SdeSuIterateOp source) {
  if (!source)
    return false;
  llvm::DenseSet<Value> visited;
  return canCloneSuScalarCapture(value, source, visited);
}

static inline LogicalResult addSuParam(Value param, SuCodeletPlan &plan,
                                       sde::SdeSuIterateOp source = nullptr) {
  if (!isCodirScalarParamType(param.getType()))
    return failure();
  if (Operation *def = param.getDefiningOp();
      def && def->hasTrait<OpTrait::ConstantLike>()) {
    plan.cloneCaptures.insert(param);
    return success();
  }
  if (!plan.paramIndex.try_emplace(param, plan.params.size()).second)
    return success();
  plan.params.push_back(param);
  return success();
}

static inline LogicalResult cloneSuCaptures(sde::SdeSuIterateOp source,
                                            SuCodeletPlan &plan,
                                            IRMapping &mapper,
                                            OpBuilder &builder) {
  std::function<LogicalResult(Value)> cloneCapture = [&](Value value) {
    if (!value || mapper.contains(value) ||
        isDefinedInside(value, source.getBody()))
      return success();

    Operation *def = value.getDefiningOp();
    if (!isCloneableSuScalarCaptureDef(def))
      return failure();

    for (Value operand : def->getOperands())
      if (failed(cloneCapture(operand)))
        return failure();

    Operation *cloned = builder.clone(*def, mapper);
    for (auto [oldResult, newResult] :
         llvm::zip(def->getResults(), cloned->getResults()))
      mapper.map(oldResult, newResult);
    return success();
  };

  for (Value capture : plan.cloneCaptures)
    if (failed(cloneCapture(capture)))
      return failure();
  return success();
}

static inline bool canLocalizeSuAllocaCapture(Value value,
                                              sde::SdeSuIterateOp source) {
  auto alloca = value.getDefiningOp<memref::AllocaOp>();
  if (!alloca || isDefinedInside(value, source.getBody()))
    return false;
  if (!alloca.getDynamicSizes().empty())
    return false;
  for (Operation *user : value.getUsers())
    if (!source.getBody().isAncestor(user->getParentRegion()))
      return false;
  return true;
}

static inline LogicalResult cloneSuLocalAllocaCaptures(SuCodeletPlan &plan,
                                                       IRMapping &mapper,
                                                       OpBuilder &builder) {
  for (Value capture : plan.localAllocaCaptures) {
    if (!capture || mapper.contains(capture))
      continue;
    auto alloca = capture.getDefiningOp<memref::AllocaOp>();
    if (!alloca)
      return failure();
    Operation *cloned = builder.clone(*alloca.getOperation(), mapper);
    mapper.map(capture, cloned->getResult(0));
  }
  return success();
}

static inline bool isSuBodyBlockArgument(Value value,
                                         sde::SdeSuIterateOp source) {
  auto arg = dyn_cast<BlockArgument>(value);
  return arg && arg.getOwner() == &source.getBody().front();
}

static inline LogicalResult collectSuOperand(Value operand, Operation *owner,
                                             sde::SdeSuIterateOp source,
                                             SuCodeletPlan &plan) {
  if (!operand || isDefinedInside(operand, source.getBody()) ||
      isSuBodyBlockArgument(operand, source))
    return success();

  if (canLocalizeSuAllocaCapture(operand, source)) {
    plan.localAllocaCaptures.insert(operand);
    return success();
  }

  if (isCodirDependencyType(operand.getType())) {
    codir::CodirAccessMode mode = codir::CodirAccessMode::readwrite;
    if (auto load = dyn_cast<memref::LoadOp>(owner);
        load && load.getMemref() == operand)
      mode = codir::CodirAccessMode::read;
    if (auto store = dyn_cast<memref::StoreOp>(owner);
        store && store.getMemref() == operand)
      mode = codir::CodirAccessMode::write;
    return addSuDep(operand, mode, plan);
  }

  if (isCodirScalarParamType(operand.getType()))
    return addSuParam(operand, plan, source);

  owner->emitOpError()
      << "operand cannot be captured by SDE scheduling-unit CODIR "
         "conversion; expected memref dependency or scalar param, got "
      << operand.getType();
  return failure();
}

static inline LogicalResult buildSuCodeletPlan(sde::SdeSuIterateOp source,
                                               Value dispatchStep,
                                               SuCodeletPlan &plan) {
  if (source.getBody().empty())
    return source.emitOpError() << "expects body before CODIR conversion";
  if (!source.getResults().empty())
    return source.emitOpError()
           << "result-producing scheduling units require explicit SDE "
              "materialization before CODIR conversion";
  if (source.getLowerBounds().empty() || source.getUpperBounds().empty() ||
      source.getSteps().empty())
    return source.emitOpError()
           << "requires at least one loop dimension before CODIR conversion";

  if (failed(addSuParam(source.getUpperBounds().front(), plan, source)) ||
      failed(addSuParam(source.getSteps().front(), plan, source)) ||
      failed(addSuParam(dispatchStep, plan, source)))
    return source.emitOpError()
           << "failed to materialize scheduling-unit loop params";

  for (unsigned dim = 1, e = source.getLowerBounds().size(); dim < e; ++dim) {
    if (failed(addSuParam(source.getLowerBounds()[dim], plan, source)) ||
        failed(addSuParam(source.getUpperBounds()[dim], plan, source)) ||
        failed(addSuParam(source.getSteps()[dim], plan, source)))
      return source.emitOpError()
             << "failed to materialize multidimensional scheduling-unit "
                "loop params";
  }

  WalkResult result = source.getBody().walk([&](Operation *nested) {
    if (isa<sde::SdeYieldOp>(nested))
      return WalkResult::advance();
    for (Value operand : nested->getOperands())
      if (failed(collectSuOperand(operand, nested, source, plan)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static inline void appendSuOwnerSliceLocalRewrites(
    sde::SdeSuIterateOp source, Value dispatchBase, OpBuilder &builder,
    SuCodeletPlan &plan,
    const SuBarrierTokenDepPlan *barrierTokenDepPlan = nullptr) {
  if (!hasSdePhysicalOwnerSlicePlan(source))
    return;

  std::optional<SmallVector<int64_t, 4>> ownerDims =
      ::mlir::carts::readI64ArrayAttr(source.getPhysicalOwnerDimsAttr());
  if (!ownerDims || ownerDims->size() != 1 || (*ownerDims)[0] < 0)
    return;
  unsigned plannedOwnerDim = static_cast<unsigned>((*ownerDims)[0]);
  bool isStencil = source.getStructuredClassification() ==
                   sde::SdeStructuredClassification::stencil;

  for (auto [depIndex, dep] : llvm::enumerate(plan.deps)) {
    auto depType = dyn_cast<MemRefType>(dep.getType());
    if (!depType || depType.getRank() == 0)
      continue;

    Value root = ::mlir::carts::ValueAnalysis::stripMemrefViewOps(dep);
    std::optional<unsigned> accessOwnerDim =
        inferSingleOwnerAccessDim(source, root);
    if (!accessOwnerDim && isStencil &&
        plannedOwnerDim < static_cast<unsigned>(depType.getRank())) {
      // Stencil read indices often carry halo offsets through casts and
      // integer arithmetic. The physical stencil plan already selected the
      // only storage owner dimension; let CODIR storage planning prove or
      // reject the concrete dependency access window from here.
      accessOwnerDim = plannedOwnerDim;
    }
    if (!accessOwnerDim || *accessOwnerDim >= depType.getRank())
      continue;

    bool hasBarrierTokenDepPlan =
        barrierTokenDepPlan &&
        barrierTokenDepPlan->hasTokenLocalRoot(source, root);
    bool needsHostWholeBridge =
        isa_and_nonnull<BlockArgument>(root) ||
        canBridgeSdeMuAllocHostWholeToComputeBlock(dep, source);
    bool matchesMaterializedPlan = *accessOwnerDim == plannedOwnerDim;

    if (!needsHostWholeBridge && matchesMaterializedPlan &&
        (hasBarrierTokenDepPlan ||
         isSdeMuAllocMaterializedWithPlan(dep, source))) {
      SmallVector<Value> offsets;
      offsets.reserve(depType.getRank());
      for (int64_t dim = 0, rank = depType.getRank(); dim < rank; ++dim) {
        if (static_cast<unsigned>(dim) == *accessOwnerDim)
          offsets.push_back(dispatchBase);
        else
          offsets.push_back(createZeroIndex(builder, source.getLoc()));
      }

      plan.localIndexRewrites.push_back(
          {static_cast<unsigned>(depIndex), std::move(offsets), {}});
      if (depIndex < plan.depStorageViews.size())
        plan.depStorageViews[depIndex] =
            codir::CodirStorageViewKind::compute_block;
      continue;
    }

    if (depIndex < plan.depStorageViews.size())
      plan.depStorageViews[depIndex] =
          codir::CodirStorageViewKind::compute_block;
  }
}

static inline Value lookupMappedParam(Value value, IRMapping &mapper) {
  if (Value mapped = mapper.lookupOrNull(value))
    return mapped;
  return value;
}

static inline void
mapSuEquivalentConstantIndexParams(sde::SdeSuIterateOp source,
                                   SuCodeletPlan &plan, IRMapping &mapper) {
  SmallVector<std::pair<int64_t, Value>> foldedParams;
  for (Value param : plan.params) {
    Operation *def = param.getDefiningOp();
    if (!def || def->hasTrait<OpTrait::ConstantLike>())
      continue;
    if (!param.getType().isIndex())
      continue;
    std::optional<int64_t> folded =
        ::mlir::carts::ValueAnalysis::tryFoldConstantIndex(param);
    if (!folded)
      continue;
    if (Value mapped = mapper.lookupOrNull(param))
      foldedParams.push_back({*folded, mapped});
  }

  if (foldedParams.empty())
    return;

  source.getBody().walk([&](Operation *op) {
    if (!op || op->getNumRegions() != 0 ||
        !isSideEffectFreeArithmeticLikeOp(op))
      return;
    for (Value result : op->getResults()) {
      if (mapper.contains(result) || !result.getType().isIndex())
        continue;
      std::optional<int64_t> folded =
          ::mlir::carts::ValueAnalysis::tryFoldConstantIndex(result);
      if (!folded)
        continue;
      auto it = llvm::find_if(
          foldedParams, [&](auto entry) { return entry.first == *folded; });
      if (it != foldedParams.end())
        mapper.map(result, it->second);
    }
  });
}

static inline bool areAllResultsMapped(Operation &op, IRMapping &mapper) {
  if (op.getNumResults() == 0)
    return false;
  return llvm::all_of(op.getResults(),
                      [&](Value result) { return mapper.contains(result); });
}

static inline LogicalResult cloneSuBodyFromDim(sde::SdeSuIterateOp source,
                                               unsigned dim, IRMapping &mapper,
                                               OpBuilder &builder,
                                               Block *computeBlock) {
  if (dim >= source.getLowerBounds().size()) {
    for (Operation &nested : computeBlock->without_terminator()) {
      if (areAllResultsMapped(nested, mapper))
        continue;
      builder.clone(nested, mapper);
    }
    return success();
  }

  Value lb = lookupMappedParam(source.getLowerBounds()[dim], mapper);
  Value ub = lookupMappedParam(source.getUpperBounds()[dim], mapper);
  Value step = lookupMappedParam(source.getSteps()[dim], mapper);
  auto loop = scf::ForOp::create(builder, source.getLoc(), lb, ub, step);
  if (source.getBody().front().getNumArguments() > dim)
    mapper.map(source.getBody().front().getArgument(dim),
               loop.getInductionVar());

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  return cloneSuBodyFromDim(source, dim + 1, mapper, builder, computeBlock);
}

static inline bool hasSuOwnerTile2dDispatchPlan(sde::SdeSuIterateOp source) {
  return hasOwnerTileTopology(source) && source.getLowerBounds().size() >= 2 &&
         source.getUpperBounds().size() >= 2 && source.getSteps().size() >= 2 &&
         getPositiveI64(source.getLogicalWorkerSliceAttr(), 0).has_value() &&
         getPositiveI64(source.getLogicalWorkerSliceAttr(), 1).has_value();
}

static inline bool requiresSuCompletionBarrier(
    sde::SdeSuIterateOp source,
    const SuBarrierTokenDepPlan *barrierTokenDepPlan = nullptr) {
  return !source.getNowaitAttr() &&
         !(barrierTokenDepPlan &&
           barrierTokenDepPlan->suppressCompletionBarrier(source));
}

static inline LogicalResult convertSuOwnerTile2dToCodir(
    sde::SdeSuIterateOp source,
    const SuBarrierTokenDepPlan *barrierTokenDepPlan = nullptr) {
  OpBuilder builder(source);
  FailureOr<Value> dispatchStep0 =
      buildSuOwnerTileDispatchStep(source, 0, builder);
  FailureOr<Value> dispatchStep1 =
      buildSuOwnerTileDispatchStep(source, 1, builder);
  if (failed(dispatchStep0) || failed(dispatchStep1))
    return source.emitOpError()
           << "owner-tile scheduling-unit conversion requires two positive "
              "logical worker slice dimensions";

  SuCodeletPlan plan;
  if (failed(buildSuCodeletPlan(source, *dispatchStep0, plan)) ||
      failed(addSuParam(*dispatchStep1, plan, source)))
    return failure();

  Location loc = source.getLoc();
  auto dispatchLoop0 =
      scf::ForOp::create(builder, loc, source.getLowerBounds()[0],
                         source.getUpperBounds()[0], *dispatchStep0);

  OpBuilder::InsertionGuard outerGuard(builder);
  builder.setInsertionPointToStart(dispatchLoop0.getBody());
  auto dispatchLoop1 =
      scf::ForOp::create(builder, loc, source.getLowerBounds()[1],
                         source.getUpperBounds()[1], *dispatchStep1);

  builder.setInsertionPointToStart(dispatchLoop1.getBody());
  if (failed(addSuParam(dispatchLoop0.getInductionVar(), plan)) ||
      failed(addSuParam(dispatchLoop1.getInductionVar(), plan)))
    return source.emitOpError()
           << "failed to materialize owner-tile scheduling-unit base params";

  SmallVector<Attribute> depModeAttrs =
      buildCodirAccessModeAttrs(source.getContext(), plan.depModes);
  SmallVector<Attribute> depStorageViewAttrs =
      buildCodirStorageViewAttrs(source.getContext(), plan.depStorageViews);

  auto codelet = createCodirCodelet(
      builder, loc, builder.getArrayAttr(depModeAttrs),
      builder.getArrayAttr(depStorageViewAttrs), plan.deps, plan.params,
      getCodirMetadataFromSchedulingUnit(source));
  if (requiresSuCompletionBarrier(source, barrierTokenDepPlan))
    codelet.setCompletionBarrierAttr(builder.getUnitAttr());

  Block *body = new Block();
  codelet.getBody().push_back(body);
  for (Value dep : plan.deps)
    body->addArgument(dep.getType(), loc);
  for (Value param : plan.params)
    body->addArgument(param.getType(), loc);

  IRMapping mapper;
  for (auto [idx, dep] : llvm::enumerate(plan.deps))
    mapper.map(dep, body->getArgument(idx));
  unsigned paramOffset = plan.deps.size();
  for (auto [idx, param] : llvm::enumerate(plan.params))
    mapper.map(param, body->getArgument(paramOffset + idx));

  builder.setInsertionPointToStart(body);
  if (failed(cloneSuCaptures(source, plan, mapper, builder)))
    return source.emitOpError()
           << "failed to clone scheduling-unit codelet capture";
  if (failed(cloneSuLocalAllocaCaptures(plan, mapper, builder)))
    return source.emitOpError()
           << "failed to localize scheduling-unit alloca capture";
  mapSuEquivalentConstantIndexParams(source, plan, mapper);

  Value base0 = mapper.lookup(dispatchLoop0.getInductionVar());
  Value upper0 = mapper.lookup(source.getUpperBounds()[0]);
  Value step0 = mapper.lookup(source.getSteps()[0]);
  Value span0 = mapper.lookup(*dispatchStep0);
  Value rawEnd0 = arith::AddIOp::create(builder, loc, base0, span0);
  Value localEnd0 = arith::MinUIOp::create(builder, loc, rawEnd0, upper0);
  auto localLoop0 = scf::ForOp::create(builder, loc, base0, localEnd0, step0);
  if (source.getBody().front().getNumArguments() > 0)
    mapper.map(source.getBody().front().getArgument(0),
               localLoop0.getInductionVar());

  OpBuilder::InsertionGuard localGuard0(builder);
  builder.setInsertionPointToStart(localLoop0.getBody());
  Value base1 = mapper.lookup(dispatchLoop1.getInductionVar());
  Value upper1 = mapper.lookup(source.getUpperBounds()[1]);
  Value step1 = mapper.lookup(source.getSteps()[1]);
  Value span1 = mapper.lookup(*dispatchStep1);
  Value rawEnd1 = arith::AddIOp::create(builder, loc, base1, span1);
  Value localEnd1 = arith::MinUIOp::create(builder, loc, rawEnd1, upper1);
  auto localLoop1 = scf::ForOp::create(builder, loc, base1, localEnd1, step1);
  if (source.getBody().front().getNumArguments() > 1)
    mapper.map(source.getBody().front().getArgument(1),
               localLoop1.getInductionVar());

  OpBuilder::InsertionGuard localGuard1(builder);
  builder.setInsertionPointToStart(localLoop1.getBody());
  Block *computeBlock = sde::getSuIterateComputeBlock(source);
  if (!computeBlock)
    return source.emitOpError()
           << "expects a computable body before CODIR conversion";
  if (failed(
          cloneSuBodyFromDim(source, /*dim=*/2, mapper, builder, computeBlock)))
    return failure();

  builder.setInsertionPointToEnd(body);
  codir::YieldOp::create(builder, loc, ValueRange{});
  if (failed(rewriteTokenLocalAccesses(codelet, plan.localIndexRewrites)))
    return source.emitOpError()
           << "failed to rewrite scheduling-unit owner-tile accesses";

  source.erase();
  return success();
}

static inline LogicalResult convertSuIterateToCodir(
    sde::SdeSuIterateOp source,
    const SuBarrierTokenDepPlan *barrierTokenDepPlan = nullptr) {
  if (hasSuOwnerTile2dDispatchPlan(source))
    return convertSuOwnerTile2dToCodir(source, barrierTokenDepPlan);

  OpBuilder builder(source);
  Value dispatchStep = buildSuDispatchStep(source, builder);

  SuCodeletPlan plan;
  if (failed(buildSuCodeletPlan(source, dispatchStep, plan)))
    return failure();

  Location loc = source.getLoc();
  auto dispatchLoop =
      scf::ForOp::create(builder, loc, source.getLowerBounds().front(),
                         source.getUpperBounds().front(), dispatchStep);

  OpBuilder::InsertionGuard outerGuard(builder);
  builder.setInsertionPointToStart(dispatchLoop.getBody());

  if (failed(addSuParam(dispatchLoop.getInductionVar(), plan)))
    return source.emitOpError()
           << "failed to materialize scheduling-unit base param";
  appendSuOwnerSliceLocalRewrites(source, dispatchLoop.getInductionVar(),
                                  builder, plan, barrierTokenDepPlan);

  SmallVector<Attribute> depModeAttrs =
      buildCodirAccessModeAttrs(source.getContext(), plan.depModes);
  SmallVector<Attribute> depStorageViewAttrs =
      buildCodirStorageViewAttrs(source.getContext(), plan.depStorageViews);

  auto codelet = createCodirCodelet(
      builder, loc, builder.getArrayAttr(depModeAttrs),
      builder.getArrayAttr(depStorageViewAttrs), plan.deps, plan.params,
      getCodirMetadataFromSchedulingUnit(source));
  if (requiresSuCompletionBarrier(source, barrierTokenDepPlan))
    codelet.setCompletionBarrierAttr(builder.getUnitAttr());

  Block *body = new Block();
  codelet.getBody().push_back(body);
  for (Value dep : plan.deps)
    body->addArgument(dep.getType(), loc);
  for (Value param : plan.params)
    body->addArgument(param.getType(), loc);

  IRMapping mapper;
  for (auto [idx, dep] : llvm::enumerate(plan.deps))
    mapper.map(dep, body->getArgument(idx));
  unsigned paramOffset = plan.deps.size();
  for (auto [idx, param] : llvm::enumerate(plan.params))
    mapper.map(param, body->getArgument(paramOffset + idx));

  builder.setInsertionPointToStart(body);
  if (failed(cloneSuCaptures(source, plan, mapper, builder)))
    return source.emitOpError()
           << "failed to clone scheduling-unit codelet capture";
  if (failed(cloneSuLocalAllocaCaptures(plan, mapper, builder)))
    return source.emitOpError()
           << "failed to localize scheduling-unit alloca capture";
  mapSuEquivalentConstantIndexParams(source, plan, mapper);

  Value base = mapper.lookup(dispatchLoop.getInductionVar());
  Value upper = mapper.lookup(source.getUpperBounds().front());
  Value step = mapper.lookup(source.getSteps().front());
  Value span = mapper.lookup(dispatchStep);
  Value rawEnd = arith::AddIOp::create(builder, loc, base, span);
  Value localEnd = arith::MinUIOp::create(builder, loc, rawEnd, upper);
  auto localLoop = scf::ForOp::create(builder, loc, base, localEnd, step);
  if (!source.getBody().front().getArguments().empty())
    mapper.map(source.getBody().front().getArgument(0),
               localLoop.getInductionVar());

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(localLoop.getBody());
  Block *computeBlock = sde::getSuIterateComputeBlock(source);
  if (!computeBlock)
    return source.emitOpError()
           << "expects a computable body before CODIR conversion";
  if (failed(
          cloneSuBodyFromDim(source, /*dim=*/1, mapper, builder, computeBlock)))
    return failure();

  builder.setInsertionPointToEnd(body);
  codir::YieldOp::create(builder, loc, ValueRange{});
  if (failed(rewriteTokenLocalAccesses(codelet, plan.localIndexRewrites)))
    return source.emitOpError()
           << "failed to rewrite scheduling-unit owner-slice accesses";

  source.erase();
  return success();
}

static inline LogicalResult inlineSdeSuDistribute(sde::SdeSuDistributeOp op) {
  if (op.getBody().empty() || op.getBody().front().getNumArguments() != 0)
    return op.emitOpError()
           << "must have an argument-free body before SDE-to-CODIR cleanup";
  Block &body = op.getBody().front();
  if (!body.empty())
    if (auto yield = dyn_cast<sde::SdeYieldOp>(&body.back()))
      yield.erase();
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  op->getBlock()->getOperations().splice(Block::iterator(op.getOperation()),
                                         body.getOperations());
  op.erase();
  return success();
}

static inline LogicalResult inlineSdeCuRegion(sde::SdeCuRegionOp op) {
  if (op.getNumResults() != 0 || !op.getIterArgs().empty() ||
      op.getBody().empty() || op.getBody().front().getNumArguments() != 0)
    return op.emitOpError()
           << "must be materialized to CODIR before SDE-to-CODIR cleanup";
  Block &body = op.getBody().front();
  if (!body.empty())
    if (auto yield = dyn_cast<sde::SdeYieldOp>(&body.back()))
      yield.erase();
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  op->getBlock()->getOperations().splice(Block::iterator(op.getOperation()),
                                         body.getOperations());
  op.erase();
  return success();
}

static inline void replaceSdeYieldWithCodirYield(codir::CodeletOp codelet) {
  Block &body = codelet.getBody().front();
  auto sdeYield = dyn_cast_or_null<sde::SdeYieldOp>(body.getTerminator());
  if (!sdeYield)
    return;

  OpBuilder builder(sdeYield);
  codir::YieldOp::create(builder, sdeYield.getLoc(), sdeYield.getValues());
  sdeYield.erase();
}

} // namespace
#endif // CARTS_DIALECT_CODIR_CONVERSION_CONVERSIONUTILS_H
