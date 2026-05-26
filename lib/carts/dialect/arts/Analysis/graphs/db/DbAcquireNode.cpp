///==========================================================================///
/// File: DbAcquireNode.cpp
///
/// Implementation of DbAcquire node -- slim facade that delegates to:
///   - PartitionBoundsAnalyzer (partition bound computation)
///   - MemoryAccessClassifier  (load/store/indirect/direct detection)
///   - DbBlockInfoComputer       (block offset/size computation)
///==========================================================================///

#include "carts/dialect/arts/Analysis/AccessPatternAnalysis.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/edt/EdtAnalysis.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbBlockInfoComputer.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbDimAnalyzer.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbGraph.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/dialect/arts/Analysis/graphs/db/MemoryAccessClassifier.h"
#include "carts/dialect/arts/Analysis/graphs/db/PartitionBoundsAnalyzer.h"
#include "carts/dialect/arts/Analysis/graphs/edt/EdtGraph.h"
#include "carts/dialect/arts/Analysis/graphs/edt/EdtNode.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "carts/dialect/arts/IR/ArtsDialect.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/EdtUtils.h"
#include "carts/dialect/arts/Utils/LoweringContractUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_acquire_node);

///===----------------------------------------------------------------------===///
/// DbAcquireNode
///===----------------------------------------------------------------------===///
DbAcquireNode::DbAcquireNode(DbAcquireOp op, NodeBase *parent,
                             DbAllocNode *rootAlloc, DbAnalysis *analysis,
                             std::string initialHierId)
    : dbAcquireOp(op), dbReleaseOp(nullptr), op(op.getOperation()),
      parent(parent), rootAlloc(rootAlloc), analysis(analysis) {
  if (!initialHierId.empty())
    hierId = std::move(initialHierId);

  /// Verify operation is valid
  Operation *opPtr = op.getOperation();
  if (!opPtr) {
    ARTS_ERROR("Cannot create DbAcquireNode: operation pointer is null");
    return;
  }

  /// Verify rootAlloc is valid
  if (!rootAlloc) {
    ARTS_ERROR("Cannot create DbAcquireNode: rootAlloc is null");
    return;
  }

  /// Estimate the footprint of this acquire from constant slice information.
  bool hasUnknown = false;
  unsigned long long totalElems = 1;
  for (Value v : dbAcquireOp.getSizes()) {
    int64_t cst = 0;
    if (!ValueAnalysis::getConstantIndex(v, cst)) {
      hasUnknown = true;
      break;
    }

    if (cst <= 0)
      continue;
    if (totalElems > std::numeric_limits<unsigned long long>::max() /
                         (unsigned long long)cst) {
      estimatedBytes = std::numeric_limits<unsigned long long>::max();
      hasUnknown = true;
      break;
    }
    totalElems *= (unsigned long long)cst;
  }

  if (!hasUnknown) {
    DbAllocOp rootAllocOp = rootAlloc->getDbAllocOp();
    if (Type elementType = rootAllocOp.getElementType()) {
      unsigned long long elemBytes = arts::getElementTypeByteSize(elementType);
      if (elemBytes != 0) {
        if (totalElems >
            std::numeric_limits<unsigned long long>::max() / elemBytes) {
          estimatedBytes = std::numeric_limits<unsigned long long>::max();
        } else {
          estimatedBytes = totalElems * elemBytes;
        }
      }
    }
  }

  /// Identify the EDT that consumes this acquire and the block argument used
  Value ptrResult = dbAcquireOp.getPtr();
  if (!ptrResult) {
    ARTS_ERROR("DbAcquireOp ptr result is null");
    return;
  }

  /// The ptr result should be passed directly to an EDT or another acquire
  auto users = ptrResult.getUsers();
  Operation *singleUser = users.empty() ? nullptr : *users.begin();
  if (!singleUser && !users.empty()) {
    ARTS_ERROR("DbAcquireOp ptr user is null");
    return;
  }

  /// Capture partition hints for analysis.
  AcquirePartitionHints hints = extractPartitionHints(dbAcquireOp);
  partitionOffset = hints.offset;
  partitionSize = hints.size;

  /// Initialize stencil bounds from the canonical contract when present.
  if (auto contract = getLoweringContract(dbAcquireOp.getPtr())) {
    auto minOffsets = contract->getStaticMinOffsets();
    auto maxOffsets = contract->getStaticMaxOffsets();
    if (minOffsets && maxOffsets && !minOffsets->empty() &&
        !maxOffsets->empty()) {
      stencilBounds =
          StencilBounds::create(minOffsets->front(), maxOffsets->front(),
                                /*isStencil=*/true, /*valid=*/true);
      ARTS_DEBUG("DbAcquireNode constructor: Initialized stencil bounds from "
                 "contract");
      ARTS_DEBUG("  bounds=[" << minOffsets->front() << ", "
                              << maxOffsets->front() << "]");
    } else if (contract->spatial.centerOffset) {
      int64_t centerOffset = *contract->spatial.centerOffset;
      /// Stencil pattern with center offset implies symmetric access pattern
      /// e.g., A[i-1], A[i], A[i+1] has center_offset=1
      /// Min offset = -centerOffset, Max offset = +centerOffset
      stencilBounds = StencilBounds::create(-centerOffset, centerOffset,
                                            /*isStencil=*/true, /*valid=*/true);
      ARTS_DEBUG("DbAcquireNode constructor: Initialized stencil bounds from "
                 "contract center offset");
      ARTS_DEBUG("  centerOffset=" << centerOffset << " -> bounds=["
                                   << -centerOffset << ", " << centerOffset
                                   << "]");
    } else {
      ARTS_DEBUG("DbAcquireNode constructor: No stencil bounds found in "
                 "contract");
    }
  } else {
    ARTS_DEBUG("DbAcquireNode constructor: No lowering contract found for "
               "acquire");
  }

  if (singleUser && isa<DbAcquireOp>(singleUser)) {
    /// Nested acquire; create child node lazily via getOrCreateAcquireNode
    /// The rest of this constructor computes info for this node; nested
    /// children will compute their own. For nested acquires, the EDT user
    /// is found when processing the child acquire node.
    return;
  }

  /// Use utility function to get EDT and block argument
  auto [edt, blockArg] = EdtUtils::getBlockArgumentForAcquire(dbAcquireOp);
  edtUserOp = edt.getOperation();
  useInEdt = blockArg;

  if (!edtUserOp || !useInEdt) {
    /// Host-side acquire/release chains are valid in pre-lowering IR
    /// (e.g. serial initialization), but they are not partitionable task
    /// acquires. Keep the node in the graph and treat it as non-EDT.
    ARTS_DEBUG("DbAcquireOp ptr is not consumed by an EDT; treating as "
               "non-partitionable acquire.\n"
               "  Acquire op: "
               << dbAcquireOp
               << "\n"
                  "  Single user: "
               << (singleUser ? *singleUser : *dbAcquireOp.getOperation())
               << "\n"
                  "  User type: "
               << (singleUser ? singleUser->getName().getStringRef()
                              : StringRef("<none>")));
    return;
  }

  /// Find the corresponding DbReleaseOp for this acquire
  for (Operation *user : useInEdt.getUsers()) {
    auto releaseOp = dyn_cast<DbReleaseOp>(user);
    if (!releaseOp)
      continue;
    if (releaseOp.getSource() == useInEdt) {
      dbReleaseOp = releaseOp;
      break;
    }
  }

  /// Get the in/out mode based on the memory accesses
  bool hasLoadAccesses = hasLoads();
  bool hasStoreAccesses = hasStores();
  if (hasLoadAccesses && !hasStoreAccesses) {
    inCount = 1;
    outCount = 0;
  } else if (hasStoreAccesses && !hasLoadAccesses) {
    inCount = 0;
    outCount = 1;
  } else {
    inCount = 1;
    outCount = 1;
  }
}

DbAcquireNode *DbAcquireNode::getOrCreateAcquireNode(DbAcquireOp op) {
  auto it = childMap.find(op);
  if (it != childMap.end())
    return it->second;
  std::string childId = getHierId().str() + "." + std::to_string(nextChildId++);
  auto newNode =
      std::make_unique<DbAcquireNode>(op, this, rootAlloc, analysis, childId);
  DbAcquireNode *ptr = newNode.get();
  childAcquires.push_back(std::move(newNode));
  childMap[op] = ptr;
  return ptr;
}

void DbAcquireNode::forEachChildNode(
    const std::function<void(NodeBase *)> &fn) const {
  for (const auto &n : childAcquires)
    fn(n.get());
}

///===----------------------------------------------------------------------===///
/// Memory Access Methods -- delegate to MemoryAccessClassifier
///===----------------------------------------------------------------------===///

void DbAcquireNode::collectAccessOperations(
    DenseMap<DbRefOp, SetVector<Operation *>> &dbRefToMemOps) {
  MemoryAccessClassifier::collectAccessOperations(this, dbRefToMemOps);
}

bool DbAcquireNode::hasLoads() {
  return MemoryAccessClassifier::hasLoads(this);
}

bool DbAcquireNode::hasStores() {
  return MemoryAccessClassifier::hasStores(this);
}

bool DbAcquireNode::isReadOnlyAccess() {
  return MemoryAccessClassifier::isReadOnlyAccess(this);
}

bool DbAcquireNode::hasIndirectAccess() const {
  return MemoryAccessClassifier::hasIndirectAccess(
      const_cast<DbAcquireNode *>(this));
}

bool DbAcquireNode::hasDirectAccess() const {
  return MemoryAccessClassifier::hasDirectAccess(
      const_cast<DbAcquireNode *>(this));
}

///===----------------------------------------------------------------------===///
/// Partition Bounds Methods -- delegate to PartitionBoundsAnalyzer
///===----------------------------------------------------------------------===///

bool DbAcquireNode::canBePartitioned() {
  return PartitionBoundsAnalyzer::canBePartitioned(this);
}

std::optional<unsigned>
DbAcquireNode::getPartitionOffsetDim(Value offset, bool requireLeading) {
  return PartitionBoundsAnalyzer::getPartitionOffsetDim(this, offset,
                                                        requireLeading);
}

const DbAcquirePartitionFacts &DbAcquireNode::getPartitionFacts() const {
  if (!partitionFacts)
    partitionFacts = DbDimAnalyzer::compute(const_cast<DbAcquireNode *>(this));
  return *partitionFacts;
}

void DbAcquireNode::invalidatePartitionFacts() const { partitionFacts.reset(); }

///===----------------------------------------------------------------------===///
/// Block Info Methods -- delegate to DbBlockInfoComputer
///===----------------------------------------------------------------------===///

LogicalResult DbAcquireNode::computeBlockInfo(Value &blockOffset,
                                              Value &blockSize) {
  return DbBlockInfoComputer::computeBlockInfo(this, blockOffset, blockSize);
}

///===----------------------------------------------------------------------===///
/// Access Pattern Classification
///===----------------------------------------------------------------------===///

AccessPattern DbAcquireNode::getAccessPattern() const {
  if (accessPattern) {
    ARTS_DEBUG("getAccessPattern() - using cached pattern: "
               << (*accessPattern == AccessPattern::Uniform   ? "Uniform"
                   : *accessPattern == AccessPattern::Stencil ? "Stencil"
                   : *accessPattern == AccessPattern::Indexed ? "Indexed"
                                                              : "Unknown"));
    return *accessPattern;
  }

  DbAcquireOp acqOp = const_cast<DbAcquireOp &>(dbAcquireOp);

  ARTS_DEBUG("getAccessPattern() - classifying pattern for acquire");
  ARTS_DEBUG(
      "  hasMultiplePartitionEntries: " << acqOp.hasMultiplePartitionEntries());
  ARTS_DEBUG(
      "  hasAllFineGrainedEntries: " << acqOp.hasAllFineGrainedEntries());
  ARTS_DEBUG(
      "  partitionIndices.empty: " << acqOp.getPartitionIndices().empty());
  if (acqOp.hasMultiplePartitionEntries()) {
    ARTS_DEBUG("  numPartitionEntries: " << acqOp.getNumPartitionEntries());
    for (size_t i = 0; i < acqOp.getNumPartitionEntries(); ++i) {
      ARTS_DEBUG("    entry["
                 << i << "] mode: "
                 << static_cast<int>(acqOp.getPartitionEntryMode(i)));
    }
  }

  /// Check multi-entry partition_entry_modes[] for task dependencies.
  /// If all entries are fine_grained with partition indices, return Indexed.
  if (acqOp.hasAllFineGrainedEntries() &&
      !acqOp.getPartitionIndices().empty()) {
    ARTS_DEBUG("  -> Returning Indexed (all fine-grained with indices)");
    accessPattern = AccessPattern::Indexed;
    return *accessPattern;
  }

  /// Check for explicit fine-grained (element-wise) partition hints FIRST.
  /// These come from OMP task deps and should NOT trigger stencil/ESD mode.
  /// Criteria: partition_indices non-empty AND mode is fine_grained (not
  /// block). Block-partition hints may also carry indices, while hints from
  /// OMP task deps are marked as element-wise (fine_grained).
  if (!acqOp.getPartitionIndices().empty()) {
    /// Check if this is truly fine-grained (from task deps) vs block hints
    auto mode = acqOp.getPartitionMode();
    ARTS_DEBUG("  partitionMode: " << (mode ? static_cast<int>(*mode) : -1));
    if (!mode || *mode == PartitionMode::fine_grained) {
      ARTS_DEBUG("  -> Returning Indexed (has indices, mode is fine_grained or "
                 "unset)");
      accessPattern = AccessPattern::Indexed;
      return *accessPattern;
    }
    /// If mode is block/stencil, fall through to other checks
    ARTS_DEBUG("  mode is block/stencil, falling through...");
  }

  /// Then check stencilBounds (only applies when no explicit fine-grained
  /// hints)
  if (stencilBounds && stencilBounds->valid && stencilBounds->isStencil) {
    ARTS_DEBUG("  -> Returning Stencil (stencilBounds valid and isStencil)");
    ARTS_DEBUG("     stencilBounds: [" << stencilBounds->minOffset << ", "
                                       << stencilBounds->maxOffset << "]");
    accessPattern = AccessPattern::Stencil;
    return *accessPattern;
  } else if (stencilBounds) {
    ARTS_DEBUG("  stencilBounds exists but not valid/stencil: valid="
               << stencilBounds->valid
               << ", isStencil=" << stencilBounds->isStencil);
  } else {
    ARTS_DEBUG("  No stencilBounds set");
  }

  /// Check multi-entry stencil pattern (only for acquires without partition
  /// indices). Only treat as stencil if NOT all entries are fine-grained (task
  /// deps).
  if (acqOp.hasMultiplePartitionEntries() &&
      !acqOp.hasAllFineGrainedEntries()) {
    int64_t minOffset = 0, maxOffset = 0;
    if (DbAnalysis::hasMultiEntryStencilPattern(acqOp, minOffset, maxOffset)) {
      ARTS_DEBUG("  -> Returning Stencil (multi-entry stencil pattern, not all "
                 "fine-grained)");
      ARTS_DEBUG("  -> Storing stencil bounds: min=" << minOffset
                                                     << " max=" << maxOffset);
      const_cast<DbAcquireNode *>(this)->setStencilBoundsInternal(
          StencilBounds::create(minOffset, maxOffset,
                                /*isStencil=*/true, /*valid=*/true));
      accessPattern = AccessPattern::Stencil;
      return *accessPattern;
    }
  }

  ARTS_DEBUG("  [COARSE-IR-WALK-CHECK] stencilBounds="
             << (stencilBounds ? "SET" : "null")
             << " edtUserOp=" << (edtUserOp ? "SET" : "null") << " partMode="
             << (acqOp.getPartitionMode()
                     ? static_cast<int>(*acqOp.getPartitionMode())
                     : -1)
             << " partIndicesEmpty=" << acqOp.getPartitionIndices().empty());
  bool isFineGrainedHint =
      acqOp.getPartitionMode() &&
      *acqOp.getPartitionMode() == PartitionMode::fine_grained;
  if (!stencilBounds && edtUserOp && !isFineGrainedHint &&
      acqOp.getPartitionIndices().empty()) {
    DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
    const_cast<DbAcquireNode *>(this)->collectAccessOperations(dbRefToMemOps);

    bool stencilFound = false;
    int64_t globalMinOffset = 0;
    int64_t globalMaxOffset = 0;
    EdtOp edtUser = const_cast<DbAcquireNode *>(this)->getEdtUser();

    SmallVector<AccessIndexInfo, 16> accesses;
    for (auto &[dbRef, memOps] : dbRefToMemOps) {
      for (Operation *memOp : memOps) {
        SmallVector<Value> indexChain =
            DbUtils::collectFullIndexChain(dbRef, memOp);
        if (indexChain.empty())
          continue;
        AccessIndexInfo info;
        info.dbRefPrefix = dbRef.getIndices().size();
        info.indexChain.append(indexChain.begin(), indexChain.end());
        accesses.push_back(std::move(info));
      }
    }

    bool indexedFound = false;
    if (edtUser && !accesses.empty()) {
      edtUser.walk([&](scf::ForOp nestedLoop) {
        if (stencilFound)
          return;
        Value nestedIV = nestedLoop.getInductionVar();
        if (!nestedIV)
          return;
        AccessBoundsResult bounds =
            analyzeAccessBoundsFromIndices(accesses, nestedIV, nestedIV);
        if (!bounds.valid) {
          indexedFound |= bounds.hasVariableOffset;
          return;
        }
        if (bounds.isStencil || bounds.minOffset != 0 ||
            bounds.maxOffset != 0) {
          ARTS_DEBUG("  Nested-loop stencil detected: min="
                     << bounds.minOffset << " max=" << bounds.maxOffset
                     << " stencil=" << bounds.isStencil);
          globalMinOffset = bounds.minOffset;
          globalMaxOffset = bounds.maxOffset;
          stencilFound = true;
        }
      });
    }

    if (stencilFound) {
      ARTS_DEBUG("  -> Returning Stencil (nested-loop access analysis)");
      ARTS_DEBUG("  -> Storing stencil bounds: min="
                 << globalMinOffset << " max=" << globalMaxOffset);
      const_cast<DbAcquireNode *>(this)->setStencilBoundsInternal(
          StencilBounds::create(globalMinOffset, globalMaxOffset,
                                /*isStencil=*/true, /*valid=*/true));
      accessPattern = AccessPattern::Stencil;
      return *accessPattern;
    }
    if (indexedFound) {
      ARTS_DEBUG("  -> Returning Indexed (nested-loop access analysis)");
      accessPattern = AccessPattern::Indexed;
      return *accessPattern;
    }
  }

  /// Check alloc-level metadata for pattern hints before falling back to
  /// Uniform
  if (rootAlloc && rootAlloc->dominantAccessPattern) {
    auto allocPattern = *rootAlloc->dominantAccessPattern;
    const char *patternName =
        allocPattern == AccessPatternType::Sequential      ? "Sequential"
        : allocPattern == AccessPatternType::Strided       ? "Strided"
        : allocPattern == AccessPatternType::Stencil       ? "Stencil"
        : allocPattern == AccessPatternType::GatherScatter ? "GatherScatter"
        : allocPattern == AccessPatternType::Random        ? "Random"
                                                           : "Unknown";
    ARTS_DEBUG("  alloc metadata dominantAccessPattern: "
               << patternName
               << " (enum value: " << static_cast<int>(allocPattern) << ")");
    /// MemrefMetadata uses AccessPatternType (Sequential/Strided/GatherScatter/
    /// Random/Unknown) while DbAcquireNode uses AccessPattern (Uniform/Stencil/
    /// Indexed). Map Stencil type directly.
    if (allocPattern == AccessPatternType::Stencil &&
        acqOp.getMode() == ArtsMode::in) {
      ARTS_DEBUG("  -> Using Stencil from alloc metadata (read-only acquire)");
      accessPattern = AccessPattern::Stencil;
      return *accessPattern;
    } else if (allocPattern == AccessPatternType::Stencil) {
      ARTS_DEBUG("  -> Skipping alloc Stencil for write/inout acquire");
    }
    /// Map GatherScatter/Random to Indexed for indirect patterns.
    if (allocPattern == AccessPatternType::GatherScatter ||
        allocPattern == AccessPatternType::Random) {
      ARTS_DEBUG(
          "  -> Using Indexed from alloc metadata (GatherScatter/Random)");
      accessPattern = AccessPattern::Indexed;
      return *accessPattern;
    }
  } else {
    ARTS_DEBUG("  No alloc-level dominantAccessPattern available");
  }

  ARTS_DEBUG("  -> Returning Uniform (fallback)");
  accessPattern = AccessPattern::Uniform;
  return *accessPattern;
}

///===----------------------------------------------------------------------===///
/// Remaining methods (print, worker-indexed, disjoint)
///===----------------------------------------------------------------------===///

bool DbAcquireNode::isWorkerIndexedAccess() const {
  if (!op)
    return false;

  DbAcquireOp mutableOp(op);

  Value partitionIdx =
      !mutableOp.getIndices().empty()
          ? mutableOp.getIndices().front()
          : (!mutableOp.getOffsets().empty() ? mutableOp.getOffsets().front()
                                             : nullptr);
  Value partitionSz =
      !mutableOp.getSizes().empty() ? mutableOp.getSizes().front() : nullptr;

  if (!partitionIdx || !partitionSz)
    return false;

  int64_t sz = 0;
  if (!ValueAnalysis::getConstantIndex(partitionSz, sz) || sz != 1)
    return false;

  return isa<BlockArgument>(partitionIdx);
}
