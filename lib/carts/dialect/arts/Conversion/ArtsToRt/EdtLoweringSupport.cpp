///==========================================================================///
/// File: EdtLoweringSupport.cpp
///
/// Helper functions extracted from EdtLowering.cpp to reduce file size.
/// All functions live in the mlir::arts::edt_lowering namespace and are
/// declared in EdtLoweringInternal.h.
///==========================================================================///

#include "carts/dialect/arts/Conversion/ArtsToRt/EdtLoweringInternal.h"

#include "carts/dialect/arts/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/LoweringContractUtils.h"
#include "carts/dialect/arts/Utils/PartitionPredicates.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mlir::arts;

namespace mlir::arts::edt_lowering {

///===----------------------------------------------------------------------===///
/// normalizeTaskDepSlice
///===----------------------------------------------------------------------===///

void normalizeTaskDepSlice(ArtsCodegen *AC, DbAcquireOp acquire,
                           const LoweringContractInfo &contract) {
  if (!AC || !acquire)
    return;

  auto mode = acquire.getPartitionMode();
  if (!mode || !usesBlockLayout(*mode))
    return;
  /// For contract-backed block/stencil acquires, partition_* is the
  /// authoritative task-local dependency window in element space. Lowering
  /// must convert that window into DB-space consistently for both read and
  /// write-capable acquires; falling back to offsets/sizes risks
  /// double-normalizing slices that upstream passes already rewrote into
  /// block space.
  auto offsetRange = acquire.getPartitionOffsets();
  auto sizeRange = acquire.getPartitionSizes();
  if (offsetRange.empty() || sizeRange.empty())
    return;

  unsigned rank = std::min<unsigned>(offsetRange.size(), sizeRange.size());
  if (rank == 0)
    return;

  /// Read-only acquires that intentionally preserved the parent DB-space
  /// range must keep that full-range contract through pre-lowering. Their
  /// partition_* hints still describe the worker-local element slice, but
  /// reinterpreting those hints as the authoritative DB window narrows
  /// correctness-preserving full-range fallbacks back to one block.
  if (shouldPreserveParentDepRange(contract, acquire) &&
      !acquire.getOffsets().empty() && !acquire.getSizes().empty())
    return;

  bool applyStencilHalo = shouldApplyStencilHalo(contract, acquire);
  auto haloMinOffsets = contract.getStaticMinOffsets();
  auto haloMaxOffsets = contract.getStaticMaxOffsets();
  bool hasContractHalo = applyStencilHalo || haloMinOffsets || haloMaxOffsets;
  bool needsStructuralNormalization = rank > 1 || usesBlockLayout(*mode);
  if (!hasContractHalo && !needsStructuralNormalization)
    return;

  auto alloc = dyn_cast_or_null<DbAllocOp>(
      DbUtils::getUnderlyingDbAlloc(acquire.getSourcePtr()));
  if (!alloc)
    return;

  auto outerSizes = alloc.getSizes();
  auto elementSizes = alloc.getElementSizes();
  SmallVector<unsigned, 4> dims = resolveContractOwnerDims(contract, rank);

  if (dims.size() > outerSizes.size() || dims.size() > elementSizes.size())
    return;

  Location loc = acquire.getLoc();
  OpBuilder::InsertionGuard guard(AC->getBuilder());
  AC->setInsertionPoint(acquire);

  SmallVector<Value, 4> dimElementOffsets, dimElementSizes;
  SmallVector<Value, 4> dimBlockSpans, dimTotalBlocks;
  for (unsigned i = 0; i < dims.size(); ++i) {
    unsigned ownerDim = dims[i];
    dimElementOffsets.push_back(offsetRange[i]);
    dimElementSizes.push_back(sizeRange[i]);
    dimBlockSpans.push_back(elementSizes[ownerDim]);
    /// DbAllocOp sizes are stored in owner-dimension order, not physical
    /// memref-dimension order. For a k-owned 4-D array, sizes has one entry
    /// (the k block count), so indexing it with ownerDim=3 is invalid.
    dimTotalBlocks.push_back(outerSizes[i]);
  }

  SmallVector<Value, 4> offsets, sizes;
  DbUtils::convertElementSliceToBlockSlice(AC->getBuilder(), loc, dimElementOffsets,
                                  dimElementSizes, dimBlockSpans,
                                  dimTotalBlocks, offsets, sizes);
  SmallVector<Value, 4> mergedOffsets, mergedSizes;
  /// normalizeTaskDepSlice may refine only a leading owner-space prefix from
  /// partition_* metadata. Preserve the remaining owner slots so N-D acquires
  /// keep the same DB rank that DbPartitioning established.
  DbUtils::mergeNormalizedBlockSlice(AC->getBuilder(), loc, acquire.getOffsets(),
                            acquire.getSizes(), outerSizes, offsets, sizes,
                            mergedOffsets, mergedSizes);

  acquire.getOffsetsMutable().assign(mergedOffsets);
  acquire.getSizesMutable().assign(mergedSizes);
}

///===----------------------------------------------------------------------===///
/// normalizeCommonElementSlice
///===----------------------------------------------------------------------===///

std::optional<NormalizedElementSlice>
normalizeCommonElementSlice(ArtsCodegen *AC, DbAcquireOp acquire,
                            DbAllocOp alloc) {
  if (!AC || !acquire || !alloc)
    return std::nullopt;

  auto elemOffsets = acquire.getElementOffsets();
  auto elemSizes = acquire.getElementSizes();
  auto depOffsets = acquire.getOffsets();
  auto depSizes = acquire.getSizes();
  auto blockSpans = alloc.getElementSizes();
  unsigned depRank = std::min<unsigned>(depOffsets.size(), depSizes.size());
  unsigned ownerRank = depRank;
  unsigned physicalRank = blockSpans.size();
  if (ownerRank == 0 || physicalRank == 0 ||
      elemOffsets.size() != elemSizes.size())
    return std::nullopt;

  Location loc = acquire.getLoc();
  Value zero = AC->createIndexConstant(0, loc);
  Value one = AC->createIndexConstant(1, loc);
  Value trueI1 = AC->create<arith::ConstantIntOp>(loc, 1, 1);

  auto contract = resolveAcquireContract(acquire);
  if (!contract)
    return std::nullopt;
  if (!contract->spatial.ownerDims.empty())
    ownerRank =
        std::min<unsigned>(ownerRank, contract->spatial.ownerDims.size());
  SmallVector<unsigned, 4> ownerDims =
      resolveContractOwnerDims(*contract, ownerRank);
  if (ownerDims.size() != ownerRank)
    return std::nullopt;

  DenseMap<unsigned, unsigned> ownerSlotForDim;
  for (unsigned ownerSlot = 0; ownerSlot < ownerDims.size(); ++ownerSlot) {
    unsigned physicalDim = ownerDims[ownerSlot];
    if (physicalDim >= physicalRank)
      return std::nullopt;
    ownerSlotForDim[physicalDim] = ownerSlot;
  }

  SmallVector<Value, 4> physicalElemOffsets;
  SmallVector<Value, 4> physicalElemSizes;
  if (elemOffsets.size() == physicalRank) {
    physicalElemOffsets.assign(elemOffsets.begin(), elemOffsets.end());
    physicalElemSizes.assign(elemSizes.begin(), elemSizes.end());
  } else if (elemOffsets.size() == ownerRank) {
    physicalElemOffsets.assign(physicalRank, zero);
    physicalElemSizes.assign(blockSpans.begin(), blockSpans.end());
    for (unsigned ownerSlot = 0; ownerSlot < ownerRank; ++ownerSlot) {
      unsigned physicalDim = ownerDims[ownerSlot];
      physicalElemOffsets[physicalDim] = elemOffsets[ownerSlot];
      physicalElemSizes[physicalDim] = elemSizes[ownerSlot];
    }
  } else {
    return std::nullopt;
  }

  NormalizedElementSlice slice;
  slice.representable = trueI1;
  slice.contiguous = AC->create<arith::ConstantIntOp>(loc, 0, 1);
  slice.wholeBlock = trueI1;
  slice.offsets.reserve(physicalRank);
  slice.sizes.reserve(physicalRank);

  auto clampToBlock = [&](Value absolute, Value blockStart, Value blockSpan) {
    Value inRange = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                              absolute, blockStart);
    Value shifted = AC->create<arith::SubIOp>(loc, absolute, blockStart);
    Value nonNegative =
        AC->create<arith::SelectOp>(loc, inRange, shifted, zero);
    return AC->create<arith::MinUIOp>(loc, nonNegative, blockSpan);
  };

  for (unsigned i = 0; i < physicalRank; ++i) {
    Value globalStart = AC->castToIndex(physicalElemOffsets[i], loc);
    Value globalSize = AC->castToIndex(physicalElemSizes[i], loc);
    Value globalEnd = AC->create<arith::AddIOp>(loc, globalStart, globalSize);

    Value blockSpan = AC->castToIndex(blockSpans[i], loc);
    Value depOffset = zero;
    Value depBlockCount = one;
    if (auto ownerIt = ownerSlotForDim.find(i);
        ownerIt != ownerSlotForDim.end()) {
      depOffset = AC->castToIndex(depOffsets[ownerIt->second], loc);
      depBlockCount = AC->castToIndex(depSizes[ownerIt->second], loc);
    }

    Value blockStart = AC->create<arith::MulIOp>(loc, depOffset, blockSpan);
    Value localEncodedStart = AC->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, globalStart, blockSpan);
    Value localEncodedEnd = AC->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ule, globalEnd, blockSpan);
    Value sliceAlreadyLocal =
        AC->create<arith::AndIOp>(loc, localEncodedStart, localEncodedEnd);

    Value singleBlock = AC->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ule, depBlockCount, one);
    Value localStartSingle = clampToBlock(globalStart, blockStart, blockSpan);
    Value localEndSingle = clampToBlock(globalEnd, blockStart, blockSpan);
    Value endAfterStart = AC->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, localEndSingle, localStartSingle);
    Value rawSingleSize =
        AC->create<arith::SubIOp>(loc, localEndSingle, localStartSingle);
    Value localSizeSingle =
        AC->create<arith::SelectOp>(loc, endAfterStart, rawSingleSize, zero);

    Value blockCountSpan =
        AC->create<arith::MulIOp>(loc, depBlockCount, blockSpan);
    Value coveredEnd =
        AC->create<arith::AddIOp>(loc, blockStart, blockCountSpan);
    Value coversStart = AC->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ule, globalStart, blockStart);
    Value coversEnd = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                                globalEnd, coveredEnd);
    Value fullCover = AC->create<arith::AndIOp>(loc, coversStart, coversEnd);

    Value globalRepresentable =
        AC->create<arith::OrIOp>(loc, singleBlock, fullCover);
    Value dimRepresentable =
        AC->create<arith::OrIOp>(loc, sliceAlreadyLocal, globalRepresentable);

    Value normalizedOffsetGlobal =
        AC->create<arith::SelectOp>(loc, singleBlock, localStartSingle, zero);
    Value normalizedSizeGlobal = AC->create<arith::SelectOp>(
        loc, singleBlock, localSizeSingle, blockSpan);

    Value normalizedOffset = AC->create<arith::SelectOp>(
        loc, sliceAlreadyLocal, globalStart, normalizedOffsetGlobal);
    Value normalizedSize = AC->create<arith::SelectOp>(
        loc, sliceAlreadyLocal, globalSize, normalizedSizeGlobal);

    slice.offsets.push_back(normalizedOffset);
    slice.sizes.push_back(normalizedSize);
    slice.representable =
        AC->create<arith::AndIOp>(loc, slice.representable, dimRepresentable);

    Value blockSpanIdx = AC->castToIndex(blockSpans[i], loc);
    Value zeroOffset = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 normalizedOffset, zero);
    Value fullExtent = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 normalizedSize, blockSpanIdx);
    Value dimWholeBlock =
        AC->create<arith::AndIOp>(loc, zeroOffset, fullExtent);
    slice.wholeBlock =
        AC->create<arith::AndIOp>(loc, slice.wholeBlock, dimWholeBlock);
  }

  for (unsigned pivot = 0; pivot < physicalRank; ++pivot) {
    Value pivotContiguous = trueI1;
    for (unsigned i = 0; i < physicalRank; ++i) {
      if (i < pivot) {
        Value fixedOne = AC->create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, slice.sizes[i], one);
        pivotContiguous =
            AC->create<arith::AndIOp>(loc, pivotContiguous, fixedOne);
        continue;
      }
      if (i > pivot) {
        Value zeroOffset = AC->create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, slice.offsets[i], zero);
        Value fullExtent = AC->create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, AC->castToIndex(blockSpans[i], loc),
            slice.sizes[i]);
        Value trailingFull =
            AC->create<arith::AndIOp>(loc, zeroOffset, fullExtent);
        pivotContiguous =
            AC->create<arith::AndIOp>(loc, pivotContiguous, trailingFull);
      }
    }
    slice.contiguous =
        AC->create<arith::OrIOp>(loc, slice.contiguous, pivotContiguous);
  }

  return slice;
}

///===----------------------------------------------------------------------===///
/// loadRepresentativeGuidScalar
///===----------------------------------------------------------------------===///

Value loadRepresentativeGuidScalar(ArtsCodegen *AC, Location loc,
                                   Value guidStorage) {
  if (!AC || !guidStorage)
    return {};

  auto guidType = dyn_cast<MemRefType>(guidStorage.getType());
  if (!guidType)
    return {};

  SmallVector<Value, 4> zeroIndices;
  zeroIndices.reserve(guidType.getRank());
  for (int64_t i = 0; i < guidType.getRank(); ++i)
    zeroIndices.push_back(AC->createIndexConstant(0, loc));

  return AC->create<memref::LoadOp>(loc, guidStorage, zeroIndices);
}

///===----------------------------------------------------------------------===///
/// resolveDepSource
///===----------------------------------------------------------------------===///

DepSourceInfo resolveDepSource(Value dep) {
  Operation *underlyingDb = DbUtils::getUnderlyingDb(dep);
  DepSourceInfo info;
  info.dbAcquire = dyn_cast_or_null<DbAcquireOp>(underlyingDb);
  info.depDbAcquire = dyn_cast_or_null<DepDbAcquireOp>(underlyingDb);
  return info;
}

///===----------------------------------------------------------------------===///
/// getCanonicalDependencySource
///===----------------------------------------------------------------------===///

Operation *getCanonicalDependencySource(Value dep) {
  DepSourceInfo info = resolveDepSource(dep);
  if (info.dbAcquire)
    return info.dbAcquire.getOperation();
  if (info.depDbAcquire)
    return info.depDbAcquire.getOperation();
  return nullptr;
}

} // namespace mlir::arts::edt_lowering
