///==========================================================================///
/// File: DbStencilRewriter.cpp
///
/// Stencil rewriter: 3-buffer halo for stencil patterns (e.g., jacobi).
///
/// Example transformation (N=100 rows, blockSize=25, halo=1):
///
///   BEFORE:
///     %db = arts.db_alloc memref<100x10xf64>
///     arts.db_acquire %db [0] [100]
///     %ref = arts.db_ref %db[%i]
///     memref.load %ref[%j-1], %ref[%j], %ref[%j+1]         /// stencil access
///
///   AFTER (worker 1, rows 25-49):
///     %db = arts.db_alloc memref<4x25x10xf64>
///     %owned = arts.db_acquire %db [1] [1]                 /// chunk 1
///     %left  = arts.db_acquire %db [0] [1] elem_offset=24  /// row 24
///     %right = arts.db_acquire %db [2] [1] elem_offset=0   /// row 50
///
///     /// Load selects buffer based on localRow:
///     ///   localRow < 0        -> left halo
///     ///   0 <= localRow < 25  -> owned
///     ///   localRow >= 25      -> right halo
///
/// Index localization: 3-way select based on (globalRow - baseOffset)
///==========================================================================///

#include "arts/dialect/core/Transforms/db/stencil/DbStencilRewriter.h"
#include "arts/dialect/core/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "arts/dialect/core/Transforms/db/DbPartitionWindowUtils.h"
#include "arts/dialect/core/Transforms/db/stencil/DbStencilIndexer.h"
#include "arts/utils/BlockedAccessUtils.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/Debug.h"
#include "arts/utils/EdtUtils.h"
#include "arts/utils/LoweringContractUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/RemovalUtils.h"
#include "arts/utils/StencilAttributes.h"
#include "arts/utils/Utils.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "db_transforms"

ARTS_DEBUG_SETUP(db_transforms);

using namespace mlir;
using namespace mlir::arts;

void DbStencilRewriter::transformAcquire(const DbRewriteAcquire &info,
                                         DbAllocOp newAlloc,
                                         OpBuilder &builder) {
  DbAcquireOp acquire = info.acquire;
  /// Stencil rewriter needs both:
  ///   - element offset (for local row rebasing),
  ///   - chunk index (for owned/halo DB selection).
  /// Prefer explicit offsets for element base. If present, indices carry a
  /// chunk-index hint from ForLowering (non-uniform worker chunks).
  Value elemOffset =
      info.getOffsets().empty()
          ? (info.getIndices().empty() ? Value() : info.getIndices().front())
          : info.getOffsets().front();
  Value chunkIndexHint =
      info.getIndices().empty() ? Value() : info.getIndices().front();

  ARTS_DEBUG("DbStencilRewriter::transformAcquire");
  ARTS_DEBUG("  acquire mode="
             << static_cast<int>(acquire.getMode())
             << ", partition mode=" << static_cast<int>(info.partitionInfo.mode)
             << ", offsets=" << info.partitionInfo.offsets.size()
             << ", sizes=" << info.partitionInfo.sizes.size()
             << ", indices=" << info.partitionInfo.indices.size());

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(acquire);
  Location loc = acquire.getLoc();

  /// Update source to new allocation only for host-side acquires.
  /// Acquires inside EDT regions must keep their in-scope source chain
  /// (block-arg/parent-acquire) to satisfy EDT dependency verification.
  bool inEdtRegion = static_cast<bool>(acquire->getParentOfType<EdtOp>());
  if (!inEdtRegion) {
    acquire.getSourceGuidMutable().assign(newAlloc.getGuid());
    acquire.getSourcePtrMutable().assign(newAlloc.getPtr());
  }

  /// Stencil mode uses offsets/sizes only; clear fine-grained indices and
  /// partition hints from the original acquire.
  acquire.getIndicesMutable().clear();
  acquire.clearPartitionHints();
  setPartitionMode(acquire.getOperation(), PartitionMode::stencil);

  /// Update acquire's ptr result type to match new source
  MemRefType newPtrType = cast<MemRefType>(acquire.getSourcePtr().getType());
  Type oldAcqPtrType = acquire.getPtr().getType();
  if (oldAcqPtrType != newPtrType) {
    acquire.getPtr().setType(newPtrType);

    /// Also update EDT block argument type if this acquire feeds an EDT
    auto [edt, blockArg] = getEdtBlockArgumentForAcquire(acquire);
    if (blockArg && blockArg.getType() != newPtrType)
      blockArg.setType(newPtrType);
  }

  Value zero = arts::createZeroIndex(builder, loc);
  Value one = arts::createOneIndex(builder, loc);

  /// Base-offset semantics: elemOffset is the explicit dependency-window base.
  Value baseOffset = elemOffset ? elemOffset : zero;
  if (baseOffset && !baseOffset.getType().isIndex())
    baseOffset = arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), baseOffset);
  /// `baseOffset` is the explicit element-space start of the dependency
  /// window. It stays in halo-expanded coordinates until we remap it through
  /// the block plan below.
  Value blockBaseOffset = baseOffset;
  if (chunkIndexHint && !chunkIndexHint.getType().isIndex())
    chunkIndexHint = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), chunkIndexHint);

  /// Uniform allocation block size — all blocks are allocated with this size.
  /// Use this instead of computeChunkRows to avoid non-uniform chunk sizes
  /// that disagree with the allocation and cause stencil indexing bugs.
  Value planBlockSize = plan.getBlockSize(0) ? plan.getBlockSize(0) : one;
  planBlockSize = arith::MaxUIOp::create(builder, loc, planBlockSize, one);
  SmallVector<Value> ownerBlockShape;
  Value ownerBlockSpan;
  std::optional<LoweringContractInfo> acquireContract =
      getLoweringContract(acquire.getPtr());
  if (!acquireContract)
    acquireContract = getSemanticContract(acquire.getOperation());
  if (acquireContract) {
    if (auto staticShape = acquireContract->getStaticBlockShape();
        staticShape && !staticShape->empty()) {
      for (int64_t dim : *staticShape)
        ownerBlockShape.push_back(arts::createConstantIndex(builder, loc, dim));
    } else if (!acquireContract->spatial.blockShape.empty()) {
      ownerBlockShape.assign(acquireContract->spatial.blockShape.begin(),
                             acquireContract->spatial.blockShape.end());
    }
  }
  if (!ownerBlockShape.empty())
    ownerBlockSpan = ownerBlockShape.front();
  if (ownerBlockSpan && !ownerBlockSpan.getType().isIndex())
    ownerBlockSpan = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), ownerBlockSpan);
  if (ownerBlockSpan)
    ownerBlockSpan = arith::MaxUIOp::create(builder, loc, ownerBlockSpan, one);
  ARTS_DEBUG("  baseOffset=" << baseOffset << ", blockBaseOffset="
                             << blockBaseOffset << ", elemOffset=" << elemOffset
                             << ", planBlockSize=" << planBlockSize
                             << ", ownerBlockSpan=" << ownerBlockSpan);

  /// Mixed stencil mode: only read-only stencil acquires should use ESD
  /// 3-buffer lowering. Non-stencil or write-capable acquires are lowered via
  /// regular block semantics over the same block-partitioned allocation.
  bool useStencilEsd = false;
  int64_t haloLeftVal = plan.stencilInfo ? plan.stencilInfo->haloLeft : 0;
  int64_t haloRightVal = plan.stencilInfo ? plan.stencilInfo->haloRight : 0;
  ARTS_DEBUG("  initial ESD eligibility="
             << useStencilEsd << " (hasHalo="
             << (plan.stencilInfo ? plan.stencilInfo->hasHalo() : false)
             << ")");

  /// Map the halo-expanded element base back to the owning DB block.
  /// Boundary tasks clamp the left halo to zero, so `baseOffset + haloLeft`
  /// is not itself guaranteed to be aligned. Recover the owner block index
  /// first, then rebuild the aligned owner base from the block plan.
  Value blockIdxBase = blockBaseOffset;
  if (haloLeftVal > 0) {
    Value haloLeftConst = arts::createConstantIndex(builder, loc, haloLeftVal);
    blockIdxBase =
        arith::AddIOp::create(builder, loc, blockIdxBase, haloLeftConst);
  }
  Value blockIdx =
      arith::DivUIOp::create(builder, loc, blockIdxBase, planBlockSize);
  Value ownerBaseOffset =
      arith::MulIOp::create(builder, loc, blockIdx, planBlockSize);

  if (useStencilEsd) {
    /// ESD 3-buffer lowering currently supports one owned DB block per task.
    /// Prefer the worker-local owner block shape carried by the lowering
    /// contract. Halo-expanded element windows are clamped at domain
    /// boundaries and do not reliably encode the owned block span there.
    Value elemSize = one;
    if (!info.partitionInfo.sizes.empty() && info.partitionInfo.sizes.front())
      elemSize = info.partitionInfo.sizes.front();

    Value ownedBaseForAlignment = ownerBaseOffset;
    Value ownedSpan = ownerBlockSpan ? ownerBlockSpan : elemSize;
    int64_t totalHalo = haloLeftVal + haloRightVal;
    if (!ownerBlockSpan && totalHalo > 0) {
      ownedSpan = elemSize;
      Value haloConst = arts::createConstantIndex(builder, loc, totalHalo);
      ownedSpan = arith::SubIOp::create(builder, loc, ownedSpan, haloConst);
    }

    bool singleBlock = false;
    auto constElemSzUpper = arts::extractBlockSizeFromHint(ownedSpan);
    auto constPlanBs = arts::extractBlockSizeFromHint(planBlockSize);
    bool isAligned =
        arts::isAlignedToBlock(ownedBaseForAlignment, planBlockSize);
    if (constElemSzUpper && constPlanBs && *constElemSzUpper <= *constPlanBs &&
        isAligned)
      singleBlock = true;
    if (!singleBlock &&
        arts::isValueBoundedByBlockSpan(ownedSpan, planBlockSize) && isAligned)
      singleBlock = true;
    bool hasExplicitBlockHaloContract =
        acquireContract && acquireContract->supportsBlockHalo();
    if (!singleBlock &&
        arts::isValueBoundedByBlockSpan(ownedSpan, planBlockSize) &&
        acquire.getMode() == ArtsMode::in && hasExplicitBlockHaloContract) {
      /// Explicit block-halo contracts already describe one owner block plus
      /// neighbor halos. When the owned span stays within one plan block, use
      /// that contract even if the alignment proof is hidden behind clamp /
      /// select normalization for boundary tasks.
      singleBlock = true;
    }

    ARTS_DEBUG("  ESD single-block check: singleBlock="
               << singleBlock << ", ownedSpan=" << ownedSpan
               << ", aligned=" << isAligned);
    useStencilEsd = singleBlock;
  }

  ARTS_DEBUG("  final ESD decision=" << useStencilEsd);
  if (!useStencilEsd) {
    transformAcquireAsBlock(info, acquire, newAlloc, builder);
    return;
  }

  /// In 3-buffer mode the owned acquire must describe the owner tile, not the
  /// original halo-expanded dependency window. The halo rows travel on their
  /// dedicated left/right acquires and the stencil indexer expects its
  /// base-offset contract to start at the owner tile.
  SmallVector<Value> ownedElementOffsets;
  SmallVector<Value> ownedElementSizes;
  unsigned ownerRank =
      std::max<size_t>(plan.innerSizes.size(), ownerBlockShape.size());
  ownedElementOffsets.reserve(ownerRank);
  ownedElementSizes.reserve(ownerRank);
  for (unsigned d = 0; d < ownerRank; ++d) {
    ownedElementOffsets.push_back(zero);
    Value dimSize = d < ownerBlockShape.size()
                        ? ownerBlockShape[d]
                        : (d < plan.innerSizes.size() && plan.innerSizes[d]
                               ? plan.innerSizes[d]
                               : one);
    if (!dimSize.getType().isIndex())
      dimSize = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                           dimSize);
    ownedElementSizes.push_back(
        arith::MaxUIOp::create(builder, loc, dimSize, one));
  }
  acquire.getElementOffsetsMutable().assign(ownedElementOffsets);
  acquire.getElementSizesMutable().assign(ownedElementSizes);

  /// STENCIL MODE with ESD: 3-acquire pattern
  /// Each worker acquires:
  ///   1. Owned chunk (full chunk)
  ///   2. Left halo (partial slice from previous chunk)
  ///   3. Right halo (partial slice from next chunk)
  Value chunkCount = one;

  ARTS_DEBUG("  Stencil ESD: blockIdx=" << blockIdx
                                        << ", haloLeft=" << haloLeftVal
                                        << ", haloRight=" << haloRightVal);

  /// Get inner dimension size for partial acquisition
  auto buildHaloOffsets = [&](Value rowOffset) {
    SmallVector<Value> offsets;
    offsets.push_back(rowOffset);
    for (size_t i = 1; i < plan.innerSizes.size(); ++i)
      offsets.push_back(zero);
    return offsets;
  };

  auto buildHaloSizes = [&](int64_t haloRows) {
    SmallVector<Value> sizes;
    sizes.push_back(arts::createConstantIndex(builder, loc, haloRows));
    for (size_t i = 1; i < plan.innerSizes.size(); ++i)
      sizes.push_back(plan.innerSizes[i]);
    return sizes;
  };

  /// Get number of chunks for boundary checking
  Value numBlocks = plan.outerSizes.empty() ? arts::createOneIndex(builder, loc)
                                            : plan.outerSizes[0];
  Value lastChunkIdx = arith::SubIOp::create(builder, loc, numBlocks, one);

  /// Create LEFT HALO acquire (partial: last haloLeft rows of prev chunk)
  unsigned leftHaloArgIdx = ~0u;
  if (haloLeftVal > 0) {
    Value leftChunkIdx = arith::SubIOp::create(builder, loc, blockIdx, one);

    /// element_offsets: start at (planBlockSize - haloLeft) within prev chunk
    Value leftChunkRows = planBlockSize;
    Value haloLeftConst = arts::createConstantIndex(builder, loc, haloLeftVal);
    Value leftElemOffset =
        arith::SubIOp::create(builder, loc, leftChunkRows, haloLeftConst);

    /// bounds_valid = (blockIdx != 0): valid only if not first block
    Value leftValid = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ne, blockIdx, zero);

    auto leftHalo = DbAcquireOp::create(
        builder, loc, ArtsMode::in, acquire.getSourceGuid(),
        acquire.getSourcePtr(), PartitionMode::stencil,
        /*indices=*/SmallVector<Value>{},
        /*offsets=*/SmallVector<Value>{leftChunkIdx},
        /*sizes=*/SmallVector<Value>{one},
        /*partition_indices=*/SmallVector<Value>{},
        /*partition_offsets=*/SmallVector<Value>{},
        /*partition_sizes=*/SmallVector<Value>{},
        /*bounds_valid=*/leftValid,
        /*element_offsets=*/buildHaloOffsets(leftElemOffset),
        /*element_sizes=*/buildHaloSizes(haloLeftVal));

    addHaloAcquireToEdt(acquire, leftHalo, builder);
    /// Track left halo block arg index (it's the last one added)
    auto [edt, blockArg] = getEdtBlockArgumentForAcquire(acquire);
    if (edt)
      leftHaloArgIdx = edt.getBody().front().getNumArguments() - 1;
    ARTS_DEBUG("  Created left halo acquire, argIdx=" << leftHaloArgIdx);
  }

  /// Create RIGHT HALO acquire (partial: first haloRight rows of next chunk)
  unsigned rightHaloArgIdx = ~0u;
  if (haloRightVal > 0) {
    Value rightChunkIdx = arith::AddIOp::create(builder, loc, blockIdx, one);

    /// bounds_valid = (blockIdx != lastBlock): valid only if not last block
    Value rightValid = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ne, blockIdx, lastChunkIdx);

    auto rightHalo =
        DbAcquireOp::create(builder, loc, ArtsMode::in, acquire.getSourceGuid(),
                            acquire.getSourcePtr(), PartitionMode::stencil,
                            /*indices=*/SmallVector<Value>{},
                            /*offsets=*/SmallVector<Value>{rightChunkIdx},
                            /*sizes=*/SmallVector<Value>{one},
                            /*partition_indices=*/SmallVector<Value>{},
                            /*partition_offsets=*/SmallVector<Value>{},
                            /*partition_sizes=*/SmallVector<Value>{},
                            /*bounds_valid=*/rightValid,
                            /*element_offsets=*/buildHaloOffsets(zero),
                            /*element_sizes=*/buildHaloSizes(haloRightVal));

    addHaloAcquireToEdt(acquire, rightHalo, builder);
    /// Track right halo block arg index (it's the last one added)
    auto [edt, blockArg] = getEdtBlockArgumentForAcquire(acquire);
    if (edt)
      rightHaloArgIdx = edt.getBody().front().getNumArguments() - 1;
    ARTS_DEBUG("  Created right halo acquire, argIdx=" << rightHaloArgIdx);
  }

  /// Store halo block arg indices for the 3-buffer selection approach
  auto [edt, ownedBlockArg] = getEdtBlockArgumentForAcquire(acquire);
  if (edt && (haloLeftVal > 0 || haloRightVal > 0)) {
    unsigned ownedArgIdx = ownedBlockArg.getArgNumber();

    ARTS_DEBUG("  3-buffer selection: owned=" << ownedArgIdx
                                              << ", left=" << leftHaloArgIdx
                                              << ", right=" << rightHaloArgIdx);

    /// Store halo arg indices as acquire attributes for rebaseEdtUsers
    setLeftHaloArgIndex(acquire.getOperation(), leftHaloArgIdx);
    setRightHaloArgIndex(acquire.getOperation(), rightHaloArgIdx);
  }

  /// Set acquire offsets/sizes for stencil mode (use blockIdx, not worker
  /// index)
  SmallVector<Value> newOffsets, newSizes;
  newOffsets.push_back(blockIdx);
  newSizes.push_back(chunkCount);
  acquire.getOffsetsMutable().assign(newOffsets);
  acquire.getSizesMutable().assign(newSizes);

  /// Rebase the EDT against the owner tile, not the halo-expanded dependency
  /// window. This keeps local-row selection consistent with the owned tile
  /// shape carried on the central acquire above.
  rebaseEdtUsers(acquire, builder, ownerBaseOffset);
}

void DbStencilRewriter::transformAcquireAsBlock(const DbRewriteAcquire &info,
                                                DbAcquireOp acquire,
                                                DbAllocOp newAlloc,
                                                OpBuilder &builder) {
  Location loc = acquire.getLoc();
  Value zero = arts::createZeroIndex(builder, loc);
  Value one = arts::createOneIndex(builder, loc);

  acquire.clearPartitionHints();
  setPartitionMode(acquire.getOperation(), PartitionMode::block);

  SmallVector<Value> newOffsets;
  SmallVector<Value> newSizes;
  unsigned nPartDims = plan.numPartitionedDims();
  if (nPartDims == 0)
    nPartDims = 1;
  SmallVector<Value> updatedElementOffsets(acquire.getElementOffsets().begin(),
                                           acquire.getElementOffsets().end());
  SmallVector<Value> updatedElementSizes(acquire.getElementSizes().begin(),
                                         acquire.getElementSizes().end());
  bool canUpdateElementSlice =
      acquire.getMode() == ArtsMode::in &&
      updatedElementOffsets.size() == updatedElementSizes.size() &&
      !updatedElementOffsets.empty();
  bool elementSliceChanged = false;

  bool hasPartitionSlices =
      !info.partitionInfo.offsets.empty() || !info.partitionInfo.sizes.empty();
  bool forceFullRangeStencilWriter =
      acquire.getMode() != ArtsMode::in && !hasPartitionSlices;
  bool forceFullRangeTinyStencilRead = false;
  if (acquire.getMode() == ArtsMode::in && nPartDims == 1 &&
      plan.innerRank() <= 1 && !plan.outerSizes.empty()) {
    if (auto outerBlocks =
            ValueAnalysis::tryFoldConstantIndex(plan.outerSizes.front())) {
      forceFullRangeTinyStencilRead = *outerBlocks > 0 && *outerBlocks <= 8;
    }
  }

  if (info.isFullRange || forceFullRangeStencilWriter ||
      forceFullRangeTinyStencilRead) {
    if (forceFullRangeTinyStencilRead) {
      ARTS_DEBUG("  Forcing full-range acquire for tiny 1-D stencil read "
                 "to keep coefficient indexing stable");
    }
    for (unsigned d = 0; d < nPartDims; ++d) {
      newOffsets.push_back(zero);
      Value blockCount = (d < plan.outerSizes.size() && plan.outerSizes[d])
                             ? plan.outerSizes[d]
                             : one;
      newSizes.push_back(blockCount);
    }
  } else {
    /// For block-mode fallback acquires on stencil allocations, convert
    /// element-space partition hints into DB-space block slices.
    auto contract = getLoweringContract(acquire.getPtr());
    if (!contract)
      contract = getSemanticContract(acquire.getOperation());
    auto staticMinOffsets =
        contract ? contract->getStaticMinOffsets() : std::nullopt;
    SmallVector<unsigned, 4> ownerDims =
        (contract && !contract->spatial.ownerDims.empty())
            ? resolveContractOwnerDims(*contract, nPartDims)
            : SmallVector<unsigned, 4>{};
    auto resolveOwnerDim = [&](unsigned d) -> unsigned {
      if (d < ownerDims.size())
        return ownerDims[d];
      if (d < plan.partitionedDims.size())
        return plan.partitionedDims[d];
      return d;
    };

    const auto &offsets = info.partitionInfo.offsets;
    const auto &sizes = info.partitionInfo.sizes;
    for (unsigned d = 0; d < nPartDims; ++d) {
      Value blockSize = plan.getBlockSize(d);
      if (!blockSize)
        blockSize = one;
      blockSize = arith::MaxUIOp::create(builder, loc, blockSize, one);
      Value totalExtent = (d < plan.outerSizes.size() && plan.outerSizes[d])
                              ? arith::MulIOp::create(
                                    builder, loc, plan.outerSizes[d], blockSize)
                              : Value();

      Value elemOff = (d < offsets.size() && offsets[d]) ? offsets[d] : zero;
      Value elemSz = (d < sizes.size() && sizes[d]) ? sizes[d] : one;
      elemSz = arith::MaxUIOp::create(builder, loc, elemSz, one);

      /// Keep DB-space block windows and element slices conceptually separate.
      /// Even when the acquire already carries an explicit halo element slice,
      /// partitionInfo.offsets/sizes still describe the owner slice selected by
      /// distribution/lowering. Re-expand that owner slice here before
      /// converting it to DB-space blocks, otherwise boundary tasks can miss a
      /// halo block even though the element slice metadata looks complete.
      ///
      /// Write-capable stencil acquires also need this DB-space halo expansion:
      /// they cannot use ARTS read-only byte-slice transport for writable
      /// state, but they can conservatively depend on neighboring physical DB
      /// blocks. Only read acquires update element_offsets/element_sizes for
      /// ESD slices.
      if (contract && contract->supportsBlockHalo()) {
        unsigned ownerDim = resolveOwnerDim(d);
        std::optional<unsigned> ownerPosition =
            contract ? getContractOwnerPosition(*contract, ownerDim)
                     : std::nullopt;
        auto applyHaloWindow = [&](int64_t minOffset, int64_t maxOffset) {
          ExpandedElementWindow expanded = expandElementWindowWithHalo(
              builder, loc, elemOff, elemSz, totalExtent, minOffset, maxOffset);
          elemOff = expanded.offset;
          elemSz = expanded.size;
          if (acquire.getMode() == ArtsMode::in && canUpdateElementSlice &&
              ownerDim < updatedElementOffsets.size() &&
              ownerDim < updatedElementSizes.size()) {
            updatedElementOffsets[ownerDim] = elemOff;
            updatedElementSizes[ownerDim] = elemSz;
            elementSliceChanged = true;
          }
        };

        auto staticMaxOffsets =
            contract ? contract->getStaticMaxOffsets() : std::nullopt;
        if (staticMinOffsets && staticMaxOffsets && ownerPosition &&
            *ownerPosition < staticMinOffsets->size() &&
            *ownerPosition < staticMaxOffsets->size()) {
          applyHaloWindow((*staticMinOffsets)[*ownerPosition],
                          (*staticMaxOffsets)[*ownerPosition]);
        }
      }

      Value startBlock =
          arith::DivUIOp::create(builder, loc, elemOff, blockSize);
      Value endElem = arith::AddIOp::create(builder, loc, elemOff, elemSz);
      endElem = arith::SubIOp::create(builder, loc, endElem, one);
      Value endBlock = arith::DivUIOp::create(builder, loc, endElem, blockSize);

      Value startAboveMax;
      if (d < plan.outerSizes.size()) {
        Value maxBlock =
            arith::SubIOp::create(builder, loc, plan.outerSizes[d], one);
        startAboveMax = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::ugt, startBlock, maxBlock);
        Value clampedEnd =
            arith::MinUIOp::create(builder, loc, endBlock, maxBlock);
        endBlock = arith::SelectOp::create(builder, loc, startAboveMax,
                                           endBlock, clampedEnd);
      }

      Value blockCount =
          arith::SubIOp::create(builder, loc, endBlock, startBlock);
      blockCount = arith::AddIOp::create(builder, loc, blockCount, one);

      if (startAboveMax) {
        startBlock = arith::SelectOp::create(builder, loc, startAboveMax, zero,
                                             startBlock);
        blockCount = arith::SelectOp::create(builder, loc, startAboveMax, zero,
                                             blockCount);
      }

      newOffsets.push_back(startBlock);
      newSizes.push_back(blockCount);
    }
  }

  if (newOffsets.empty())
    newOffsets.push_back(zero);
  if (newSizes.empty()) {
    Value fallbackSize = (!plan.outerSizes.empty() && plan.outerSizes.front())
                             ? plan.outerSizes.front()
                             : one;
    newSizes.push_back(fallbackSize);
  }

  acquire.getOffsetsMutable().assign(newOffsets);
  acquire.getSizesMutable().assign(newSizes);
  if (elementSliceChanged) {
    acquire.getElementOffsetsMutable().assign(updatedElementOffsets);
    acquire.getElementSizesMutable().assign(updatedElementSizes);
  }

  /// Block fallback must always rebase EDT-side db_ref users into local
  /// block coordinates. Stencil-origin acquires can still carry global-row
  /// indexing; skipping rebasing leaves invalid local block accesses.
  bool rebaseAsBlock = !info.skipRebase;
  if (rebaseAsBlock)
    rebaseEdtUsersAsBlock(acquire, builder);
}

bool DbStencilRewriter::rebaseEdtUsersAsBlock(DbAcquireOp acquire,
                                              OpBuilder &builder) {
  auto [edt, blockArg] = getEdtBlockArgumentForAcquire(acquire);
  if (!blockArg)
    return false;

  Type targetType;
  for (Operation *user : blockArg.getUsers()) {
    if (auto ref = dyn_cast<DbRefOp>(user)) {
      targetType = ref.getResult().getType();
      break;
    }
  }

  if (!targetType) {
    targetType = blockArg.getType();
    if (auto outer = dyn_cast<MemRefType>(targetType))
      if (auto inner = dyn_cast<MemRefType>(outer.getElementType()))
        targetType = inner;
  }
  if (!targetType || !isa<MemRefType>(targetType))
    return false;

  Type derivedType = targetType;
  if (auto dbAlloc =
          dyn_cast_or_null<DbAllocOp>(DbUtils::getUnderlyingDbAlloc(blockArg)))
    derivedType = dbAlloc.getAllocatedElementType();

  OpBuilder::InsertionGuard ig(builder);
  builder.setInsertionPointToStart(&edt.getBody().front());
  Location loc = acquire.getLoc();
  Value one = arts::createOneIndex(builder, loc);
  Value zero = arts::createZeroIndex(builder, loc);

  unsigned nPartDims = plan.numPartitionedDims();
  if (nPartDims == 0)
    nPartDims = 1;

  SmallVector<Value> effectiveBlockSizes;
  SmallVector<Value> effectiveStartBlocks;
  effectiveBlockSizes.reserve(nPartDims);
  effectiveStartBlocks.reserve(nPartDims);

  Region &edtRegion = edt.getBody();
  IRMapping cloneMapping;
  llvm::SetVector<Value> valuesToClone;

  for (unsigned d = 0; d < nPartDims; ++d) {
    Value bs = plan.getBlockSize(d);
    if (!bs)
      bs = one;
    bs = arith::MaxUIOp::create(builder, loc, bs, one);
    effectiveBlockSizes.push_back(bs);

    Value startBlock =
        d < acquire.getOffsets().size() ? acquire.getOffsets()[d] : zero;
    if (startBlock && !startBlock.getType().isIndex())
      startBlock = arith::IndexCastOp::create(
          builder, loc, builder.getIndexType(), startBlock);
    if (!startBlock)
      startBlock = zero;
    if (!arts::isDefinedInRegion(startBlock, &edtRegion) &&
        startBlock.getDefiningOp())
      valuesToClone.insert(startBlock);
    effectiveStartBlocks.push_back(startBlock);
  }

  if (!valuesToClone.empty()) {
    if (!ValueAnalysis::cloneValuesIntoRegion(
            valuesToClone, &edtRegion, cloneMapping, builder,
            /*allowMemoryEffectFree=*/false, arts::isStartBlockArithmeticOp))
      return false;
  }

  for (Value &startBlock : effectiveStartBlocks) {
    if (!startBlock)
      continue;
    if (cloneMapping.contains(startBlock))
      startBlock = cloneMapping.lookup(startBlock);
    if (!arts::isDefinedInRegion(startBlock, &edtRegion))
      startBlock = zero;
  }

  bool allocSingleBlock =
      !plan.outerSizes.empty() && llvm::all_of(plan.outerSizes, [](Value v) {
        auto val = ValueAnalysis::tryFoldConstantIndex(v);
        return val && *val == 1;
      });
  bool acquireSingleBlock = !acquire.getSizes().empty() &&
                            llvm::all_of(acquire.getSizes(), [](Value v) {
                              auto val = ValueAnalysis::tryFoldConstantIndex(v);
                              return val && *val == 1;
                            });

  auto indexer = createBlockIndexer(effectiveBlockSizes, effectiveStartBlocks,
                                    plan.outerRank(), plan.innerRank(),
                                    plan.partitionedDims, allocSingleBlock,
                                    acquireSingleBlock, zero);
  if (!indexer)
    return false;

  SmallVector<Operation *> users(blockArg.getUsers().begin(),
                                 blockArg.getUsers().end());
  llvm::SetVector<Operation *> opsToRemove;
  bool rewritten = false;

  for (Operation *user : users) {
    if (auto ref = dyn_cast<DbRefOp>(user)) {
      indexer->transformDbRefUsers(ref, blockArg, derivedType, builder,
                                   opsToRemove);
      rewritten = true;
    }
  }

  RemovalUtils removal;
  for (Operation *op : opsToRemove)
    removal.markForRemoval(op);
  removal.removeAllMarked();
  return rewritten;
}

void DbStencilRewriter::transformDbRef(DbRefOp ref, DbAllocOp newAlloc,
                                       OpBuilder &builder) {
  Location loc = ref.getLoc();
  Type newElementType = newAlloc.getAllocatedElementType();
  Value newSource = (ref.getSource() == oldAlloc.getPtr()) ? newAlloc.getPtr()
                                                           : ref.getSource();
  bool allocSingleBlock =
      !plan.outerSizes.empty() && llvm::all_of(plan.outerSizes, [](Value v) {
        auto val = ValueAnalysis::tryFoldConstantIndex(v);
        return val && *val == 1;
      });

  /// Collect load/store users of this db_ref
  SmallVector<Operation *> refUsers(ref.getResult().getUsers().begin(),
                                    ref.getResult().getUsers().end());
  RemovalUtils removal;

  unsigned nPartDims = plan.numPartitionedDims();
  if (nPartDims == 0)
    nPartDims = 1;
  Value zero = arts::createZeroIndex(builder, loc);
  SmallVector<Value> startBlocks;
  startBlocks.reserve(nPartDims);
  for (unsigned d = 0; d < nPartDims; ++d)
    startBlocks.push_back(zero);

  auto indexer = createBlockIndexer(plan.blockSizes, startBlocks,
                                    plan.outerRank(), plan.innerRank(),
                                    plan.partitionedDims, allocSingleBlock);

  bool hasLoadStoreUsers = false;
  for (Operation *user : refUsers) {
    auto access = DbUtils::getMemoryAccessInfo(user);
    if (!access)
      continue;

    hasLoadStoreUsers = true;
    OpBuilder::InsertionGuard ig(builder);
    builder.setInsertionPoint(user);
    Location userLoc = user->getLoc();

    ValueRange loadStoreIndices = access->indices;
    SmallVector<Value> globalIndices;
    if (!loadStoreIndices.empty()) {
      if (loadStoreIndices.size() < nPartDims && !ref.getIndices().empty()) {
        globalIndices.assign(ref.getIndices().begin(), ref.getIndices().end());
        globalIndices.append(loadStoreIndices.begin(), loadStoreIndices.end());
      } else {
        globalIndices.assign(loadStoreIndices.begin(), loadStoreIndices.end());
      }
    }

    LocalizedIndices localized =
        globalIndices.empty()
            ? LocalizedIndices()
            : indexer->localize(globalIndices, builder, userLoc);
    if (localized.dbRefIndices.empty())
      localized.dbRefIndices.push_back(arts::createZeroIndex(builder, userLoc));
    if (localized.memrefIndices.empty())
      localized.memrefIndices.push_back(
          arts::createZeroIndex(builder, userLoc));

    ARTS_DEBUG("  Outside-EDT stencil remap uses N-D block indexer");

    auto newRef = DbRefOp::create(builder, userLoc, newElementType, newSource,
                                  localized.dbRefIndices);

    /// Create new load/store with transformed inner indices
    if (access->isRead()) {
      auto load = cast<memref::LoadOp>(user);
      auto newLoad = memref::LoadOp::create(builder, userLoc, newRef,
                                            localized.memrefIndices);
      load.replaceAllUsesWith(newLoad.getResult());
    } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
      memref::StoreOp::create(builder, userLoc, store.getValue(), newRef,
                              localized.memrefIndices);
    }
    removal.markForRemoval(user);
  }

  /// For db_refs without load/store users, just update the db_ref type
  if (!hasLoadStoreUsers && !ref.getResult().use_empty()) {
    OpBuilder::InsertionGuard ig(builder);
    builder.setInsertionPoint(ref);
    SmallVector<Value> indices(ref.getIndices().begin(),
                               ref.getIndices().end());
    LocalizedIndices localized = indices.empty()
                                     ? LocalizedIndices()
                                     : indexer->localize(indices, builder, loc);
    if (localized.dbRefIndices.empty())
      localized.dbRefIndices.push_back(zero);
    auto newRef = DbRefOp::create(builder, loc, newElementType, newSource,
                                  localized.dbRefIndices);
    ref.replaceAllUsesWith(newRef.getResult());
  }

  /// Remove old load/store operations
  removal.removeAllMarked();
}

bool DbStencilRewriter::rebaseEdtUsers(DbAcquireOp acquire, OpBuilder &builder,
                                       Value baseOffsetArg,
                                       bool /*isSingleChunk*/) {
  ARTS_DEBUG("DbStencilRewriter::rebaseEdtUsers (3-buffer mode)");

  auto [edt, blockArg] = getEdtBlockArgumentForAcquire(acquire);
  Value localView = blockArg ? Value(blockArg) : acquire.getPtr();
  if (!edt)
    edt = acquire->getParentOfType<EdtOp>();
  if (!localView || !edt)
    return false;

  /// Determine element type from users or allocation
  Type targetType;
  for (Operation *user : localView.getUsers()) {
    if (auto ref = dyn_cast<DbRefOp>(user)) {
      targetType = ref.getResult().getType();
      break;
    }
  }
  if (!targetType) {
    targetType = localView.getType();
    if (auto outer = dyn_cast<MemRefType>(targetType))
      if (auto inner = dyn_cast<MemRefType>(outer.getElementType()))
        targetType = inner;
  }
  if (!targetType || !isa<MemRefType>(targetType))
    return false;

  Type derivedType = targetType;
  if (auto dbAlloc =
          dyn_cast_or_null<DbAllocOp>(DbUtils::getUnderlyingDbAlloc(localView)))
    derivedType = dbAlloc.getAllocatedElementType();

  /// Derive BASE offset for stencil localization.
  /// The incoming value is the acquire's true element base offset.
  /// The indexer uses base-offset semantics where:
  ///   - localRow < 0 -> left halo
  ///   - 0 <= localRow < blockSize -> owned
  ///   - localRow >= blockSize -> right halo
  Location loc = acquire.getLoc();
  Value zero = arts::createZeroIndex(builder, loc);
  Value one = arts::createOneIndex(builder, loc);
  Value haloLeft =
      arts::createConstantIndex(builder, loc, plan.stencilInfo->haloLeft);
  Value haloRight =
      arts::createConstantIndex(builder, loc, plan.stencilInfo->haloRight);

  Value blockSize = acquire.getElementSizes().empty()
                        ? (plan.getBlockSize(0) ? plan.getBlockSize(0) : one)
                        : acquire.getElementSizes().front();
  if (!blockSize.getType().isIndex())
    blockSize = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                           blockSize);
  blockSize = arith::MaxUIOp::create(builder, loc, blockSize, one);

  Value baseOffset = baseOffsetArg ? baseOffsetArg : zero;
  if (!baseOffset.getType().isIndex())
    baseOffset = arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), baseOffset);

  /// For stencil mode, get the 3 block args (owned, left halo, right halo)
  Value leftHaloArg, rightHaloArg;
  Value leftHaloAvail, rightHaloAvail;
  Block &edtBody = edt.getBody().front();
  auto deps = edt.getDependencies();

  /// Read halo arg indices from acquire attributes
  if (auto leftIdxOpt = getLeftHaloArgIndex(acquire.getOperation())) {
    unsigned leftIdx = *leftIdxOpt;
    if (leftIdx < edtBody.getNumArguments()) {
      leftHaloArg = edtBody.getArgument(leftIdx);
      if (leftIdx < deps.size())
        if (auto leftHaloAcquire = deps[leftIdx].getDefiningOp<DbAcquireOp>())
          leftHaloAvail = leftHaloAcquire.getBoundsValid();
    }
  }
  if (auto rightIdxOpt = getRightHaloArgIndex(acquire.getOperation())) {
    unsigned rightIdx = *rightIdxOpt;
    if (rightIdx < edtBody.getNumArguments()) {
      rightHaloArg = edtBody.getArgument(rightIdx);
      if (rightIdx < deps.size())
        if (auto rightHaloAcquire = deps[rightIdx].getDefiningOp<DbAcquireOp>())
          rightHaloAvail = rightHaloAcquire.getBoundsValid();
    }
  }

  ARTS_DEBUG("  Stencil 3-buffer: owned="
             << blockArg << ", left=" << (leftHaloArg ? "valid" : "null")
             << ", right=" << (rightHaloArg ? "valid" : "null"));

  /// Build PartitionInfo with BASE offset semantics
  PartitionInfo info;
  info.mode = PartitionMode::stencil;
  info.offsets.push_back(baseOffset);
  info.sizes.push_back(blockSize);

  /// Create stencil indexer with PartitionInfo (base-offset semantics)
  auto indexer = std::make_unique<DbStencilIndexer>(
      info, haloLeft, haloRight, plan.outerRank(), plan.innerRank(),
      leftHaloAvail, rightHaloAvail, localView, leftHaloArg, rightHaloArg);

  SmallVector<Operation *> users(localView.getUsers().begin(),
                                 localView.getUsers().end());
  llvm::SetVector<Operation *> opsToRemove;
  bool rewritten = false;

  for (Operation *user : users) {
    if (auto ref = dyn_cast<DbRefOp>(user)) {
      indexer->transformDbRefUsers(ref, localView, derivedType, builder,
                                   opsToRemove);
      rewritten = true;
    }
  }

  for (Operation *op : opsToRemove)
    if (op->use_empty())
      op->erase();

  return rewritten;
}

void DbStencilRewriter::addHaloAcquireToEdt(DbAcquireOp originalAcq,
                                            DbAcquireOp haloAcq,
                                            OpBuilder &builder) {
  ARTS_DEBUG("DbStencilRewriter::addHaloAcquireToEdt");

  /// Find the EDT that uses the original acquire
  auto [edt, blockArg] = getEdtBlockArgumentForAcquire(originalAcq);
  if (!edt) {
    ARTS_DEBUG("  No EDT found for acquire - skipping halo addition");
    return;
  }

  /// Add the halo acquire's ptr as a new EDT dependency
  /// The appendDependency method updates the EDT operands
  edt.appendDependency(haloAcq.getPtr());

  /// Add corresponding block argument to EDT body
  Block &edtBody = edt.getBody().front();
  edtBody.addArgument(haloAcq.getPtr().getType(), haloAcq.getLoc());
  Value haloArg = edtBody.getArgument(edtBody.getNumArguments() - 1);

  /// Ensure the halo acquire is released in the task body.
  /// ForLowering only inserts releases for original dependencies; halo
  /// dependencies are introduced later by this rewriter.
  bool hasRelease = false;
  for (Operation &op : edtBody.without_terminator()) {
    if (auto release = dyn_cast<DbReleaseOp>(&op)) {
      if (release.getSource() == haloArg) {
        hasRelease = true;
        break;
      }
    }
  }
  if (!hasRelease) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(edtBody.getTerminator());
    DbReleaseOp::create(builder, haloAcq.getLoc(), haloArg);
  }

  ARTS_DEBUG("  Added halo dep to EDT, now has "
             << edt.getDependencies().size() << " deps and "
             << edtBody.getNumArguments() << " block args");
}
