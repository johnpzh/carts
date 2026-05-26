///==========================================================================///
/// File: DbUtils.cpp
///
/// Implementation of utility functions for working with ARTS DBs.
///==========================================================================///

#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/ValueAnalysisUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_utils);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

static void getAcquireDepSlice(DbAcquireOp acquire,
                               SmallVector<Value> &sizesOut,
                               SmallVector<Value> &offsetsOut) {
  sizesOut.assign(acquire.getSizes().begin(), acquire.getSizes().end());
  offsetsOut.assign(acquire.getOffsets().begin(), acquire.getOffsets().end());
}

static bool isMemrefForwardingForSource(Operation *op, Value source) {
  if (!op || !source)
    return false;

  source = ValueAnalysis::stripMemrefViewOps(source);
  if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(op))
    return llvm::is_contained(unrealized.getInputs(), source);
  if (auto dbRef = dyn_cast<DbRefOp>(op))
    return dbRef.getSource() == source;

  for (Value result : op->getResults()) {
    if (!isa<BaseMemRefType>(result.getType()))
      continue;
    if (ValueAnalysis::stripMemrefViewOps(result) == source)
      return true;
  }

  return false;
}

static bool isCleanupTerminalOp(Operation *op, Value current) {
  if (auto release = dyn_cast<DbReleaseOp>(op))
    return release.getSource() == current;
  if (auto free = dyn_cast<DbFreeOp>(op))
    return free.getSource() == current;
  if (auto contract = dyn_cast<LoweringContractOp>(op))
    return contract.getTarget() == current;
  return false;
}

static bool isStructuralDbChainUser(Operation *op, Value current) {
  if (isMemrefForwardingForSource(op, current))
    return true;

  if (auto acquire = dyn_cast<DbAcquireOp>(op))
    return acquire.getSourcePtr() == current ||
           (acquire.getSourceGuid() && acquire.getSourceGuid() == current);

  return false;
}

static void enqueueStructuralDbChainResults(Operation *op,
                                            SmallVectorImpl<Value> &worklist) {
  if (!op)
    return;

  if (auto acquire = dyn_cast<DbAcquireOp>(op)) {
    worklist.push_back(acquire.getGuid());
    worklist.push_back(acquire.getPtr());
    return;
  }

  for (Value result : op->getResults())
    worklist.push_back(result);
}

static void appendDynamicSubviewOffsets(memref::SubViewOp subview,
                                        SmallVector<Value> &chain) {
  for (OpFoldResult off : subview.getMixedOffsets())
    if (auto v = dyn_cast<Value>(off))
      chain.push_back(v);
}

struct RootAccessSummary {
  bool reads = false;
  bool writes = false;

  void record(DbUtils::MemoryAccessKind kind) {
    if (kind == DbUtils::MemoryAccessKind::Read)
      reads = true;
    else
      writes = true;
  }

  bool isReadOnly() const { return reads && !writes; }
  bool hasWrite() const { return writes; }
};

struct MemoryRootSummary {
  DenseMap<Operation *, RootAccessSummary> dbAllocs;
  DenseMap<Value, RootAccessSummary> rawMemrefs;
};

static MemoryRootSummary summarizeMemoryRoots(Operation *scope) {
  MemoryRootSummary summary;
  if (!scope)
    return summary;

  scope->walk([&](Operation *op) {
    auto access = DbUtils::getMemoryAccessInfo(op);
    if (!access)
      return;

    if (Operation *alloc = DbUtils::getUnderlyingDbAlloc(access->memref)) {
      summary.dbAllocs[alloc].record(access->kind);
      return;
    }

    summary.rawMemrefs[access->memref].record(access->kind);
  });

  return summary;
}

} // namespace

namespace mlir::carts::arts {

uint64_t getElementTypeByteSize(Type elemTy) {
  if (!elemTy)
    return 0;

  if (auto memTy = dyn_cast<MemRefType>(elemTy)) {
    uint64_t elementBytes = getElementTypeByteSize(memTy.getElementType());
    if (elementBytes == 0)
      return 0;

    uint64_t total = elementBytes;
    for (int64_t dim : memTy.getShape()) {
      if (dim == ShapedType::kDynamic)
        return 0;
      total *= static_cast<uint64_t>(std::max<int64_t>(dim, 0));
    }
    return total;
  }
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return intTy.getWidth() / 8u;
  if (auto fTy = dyn_cast<FloatType>(elemTy))
    return fTy.getWidth() / 8u;
  return 0;
}

MemRefType getElementMemRefType(Type elementType,
                                ArrayRef<Value> elementSizes) {
  const size_t rank = elementSizes.empty() ? 1 : elementSizes.size();
  SmallVector<int64_t> elementShape(rank, ShapedType::kDynamic);
  return MemRefType::get(elementShape, elementType);
}

ArtsMode combineAccessModes(ArtsMode mode1, ArtsMode mode2) {
  if (mode1 == ArtsMode::uninitialized)
    return mode2;
  if (mode2 == ArtsMode::uninitialized)
    return mode1;

  if (mode1 == ArtsMode::inout || mode2 == ArtsMode::inout)
    return ArtsMode::inout;

  if ((mode1 == ArtsMode::in && mode2 == ArtsMode::out) ||
      (mode1 == ArtsMode::out && mode2 == ArtsMode::in))
    return ArtsMode::inout;

  if (mode1 == mode2)
    return mode1;

  return ArtsMode::inout;
}

} // namespace mlir::carts::arts

Value DbUtils::getAccessedMemref(Operation *memOp) {
  if (!memOp)
    return Value();
  if (auto load = dyn_cast<memref::LoadOp>(memOp))
    return load.getMemRef();
  if (auto store = dyn_cast<memref::StoreOp>(memOp))
    return store.getMemRef();
  if (auto load = dyn_cast<affine::AffineLoadOp>(memOp))
    return load.getMemRef();
  if (auto store = dyn_cast<affine::AffineStoreOp>(memOp))
    return store.getMemRef();
  return Value();
}

///===----------------------------------------------------------------------===///
/// Datablock Tracing Utilities
///===----------------------------------------------------------------------===///

Operation *DbUtils::getUnderlyingDb(Value v, unsigned depth) {
  if (!v)
    return nullptr;

  /// Prevent infinite recursion from circular acquire chains.
  if (depth > 20) {
    ARTS_WARN("getUnderlyingDb exceeded depth limit");
    return nullptr;
  }

  /// Case 1: Direct DbAllocOp result (either guid or ptr).
  if (auto acq = v.getDefiningOp<DbAcquireOp>())
    return acq.getOperation();
  if (auto alloc = v.getDefiningOp<DbAllocOp>())
    return alloc.getOperation();
  /// Case 2: DbAcquireOp — trace through sourceGuid or sourcePtr.
  if (auto dbLoad = v.getDefiningOp<DbRefOp>())
    return getUnderlyingDb(dbLoad.getSource(), depth + 1);

  /// Case 3: Block argument — map to parent EDT dependency and recurse.
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    Block *block = blockArg.getOwner();
    Operation *owner = block->getParentOp();
    if (auto edt = dyn_cast<EdtOp>(owner)) {
      unsigned argIndex = blockArg.getArgNumber();
      ValueRange deps = edt.getDependencies();
      if (argIndex < deps.size())
        return getUnderlyingDb(deps[argIndex], depth + 1);
    }
  }

  if (Operation *def = v.getDefiningOp()) {
    if (auto ptrToInt = dyn_cast<LLVM::PtrToIntOp>(def))
      return getUnderlyingDb(ptrToInt.getArg(), depth + 1);
    if (auto intToPtr = dyn_cast<LLVM::IntToPtrOp>(def))
      return getUnderlyingDb(intToPtr.getArg(), depth + 1);
    if (auto memrefToPtr = dyn_cast<polygeist::Memref2PointerOp>(def))
      return getUnderlyingDb(memrefToPtr.getSource(), depth + 1);
    if (auto ptrToMemref = dyn_cast<polygeist::Pointer2MemrefOp>(def))
      return getUnderlyingDb(ptrToMemref.getSource(), depth + 1);
    if (auto castOp = dyn_cast<memref::CastOp>(def))
      return getUnderlyingDb(castOp.getSource(), depth + 1);
    if (auto subview = dyn_cast<memref::SubViewOp>(def))
      return getUnderlyingDb(subview.getSource(), depth + 1);
  }

  if (Operation *root = arts::getUnderlyingOperation(v))
    if (isa<DbAcquireOp, DbAllocOp>(root))
      return root;

  return nullptr;
}

Operation *DbUtils::getUnderlyingDbAlloc(Value v) {
  Operation *underlyingDb = getUnderlyingDb(v);
  if (!underlyingDb)
    return nullptr;
  if (isa<DbAllocOp>(underlyingDb))
    return underlyingDb;
  auto dbAcquire = dyn_cast<DbAcquireOp>(underlyingDb);
  if (dbAcquire)
    return getUnderlyingDbAlloc(dbAcquire.getSourcePtr());
  return nullptr;
}

DbAllocOp DbUtils::getAllocOpFromGuid(Value dbGuid) {
  if (!dbGuid)
    return nullptr;

  if (auto dbAcquireOp = dbGuid.getDefiningOp<DbAcquireOp>())
    return dbAcquireOp.getSourceGuid()
               ? dbAcquireOp.getSourceGuid().getDefiningOp<DbAllocOp>()
               : nullptr;
  return nullptr;
}

///===----------------------------------------------------------------------===///
/// Datablock Size and Offset Extraction
///===----------------------------------------------------------------------===///

SmallVector<Value> DbUtils::getSizesFromDb(Operation *dbOp) {
  if (auto allocOp = dyn_cast<DbAllocOp>(dbOp)) {
    return SmallVector<Value>(allocOp.getSizes().begin(),
                              allocOp.getSizes().end());
  }
  if (auto acquireOp = dyn_cast<DbAcquireOp>(dbOp)) {
    return SmallVector<Value>(acquireOp.getSizes().begin(),
                              acquireOp.getSizes().end());
  }
  return {};
}

SmallVector<Value> DbUtils::getSizesFromDb(Value dbPtr) {
  /// Use getUnderlyingDb to find the original DB operation.
  Operation *underlyingDb = getUnderlyingDb(dbPtr);
  if (!underlyingDb)
    return {};

  return getSizesFromDb(underlyingDb);
}

SmallVector<Value> DbUtils::getDepSizesFromDb(Operation *dbOp) {
  if (auto allocOp = dyn_cast_or_null<DbAllocOp>(dbOp))
    return SmallVector<Value>(allocOp.getSizes().begin(),
                              allocOp.getSizes().end());

  if (auto acquireOp = dyn_cast_or_null<DbAcquireOp>(dbOp)) {
    SmallVector<Value> sizes;
    SmallVector<Value> offsets;
    getAcquireDepSlice(acquireOp, sizes, offsets);
    return sizes;
  }

  return {};
}

SmallVector<Value> DbUtils::getDepSizesFromDb(Value dbPtr) {
  Operation *underlyingDb = getUnderlyingDb(dbPtr);
  if (!underlyingDb)
    return {};
  return getDepSizesFromDb(underlyingDb);
}

SmallVector<Value> DbUtils::getDepOffsetsFromDb(Operation *dbOp) {
  if (auto acquireOp = dyn_cast_or_null<DbAcquireOp>(dbOp)) {
    SmallVector<Value> sizes;
    SmallVector<Value> offsets;
    getAcquireDepSlice(acquireOp, sizes, offsets);
    return offsets;
  }

  return {};
}

SmallVector<Value> DbUtils::getDepOffsetsFromDb(Value dbPtr) {
  Operation *underlyingDb = getUnderlyingDb(dbPtr);
  if (!underlyingDb)
    return {};
  return getDepOffsetsFromDb(underlyingDb);
}

bool DbUtils::isWriterMode(ArtsMode mode) {
  return mode == ArtsMode::out || mode == ArtsMode::inout;
}

bool DbUtils::isI1DbPtrType(Type type) {
  auto outer = dyn_cast<MemRefType>(type);
  if (!outer)
    return false;
  auto inner = dyn_cast<MemRefType>(outer.getElementType());
  return inner && inner.getElementType().isInteger(1);
}

std::optional<int64_t> DbUtils::getStaticElementCount(DbAllocOp alloc) {
  if (!alloc)
    return std::nullopt;

  int64_t total = 1;
  auto multiplyBy = [&](Value value) -> bool {
    std::optional<int64_t> folded = ValueAnalysis::tryFoldConstantIndex(
        ValueAnalysis::stripNumericCasts(value));
    if (!folded || *folded < 0)
      return false;
    if (*folded != 0 && total > std::numeric_limits<int64_t>::max() / *folded)
      return false;
    total *= *folded;
    return true;
  };

  for (Value size : alloc.getSizes())
    if (!multiplyBy(size))
      return std::nullopt;
  for (Value size : alloc.getElementSizes())
    if (!multiplyBy(size))
      return std::nullopt;

  return total;
}

bool DbUtils::isCoarseUserDataDb(DbAllocOp alloc) {
  if (!alloc)
    return false;
  if (DbUtils::isI1DbPtrType(alloc.getPtr().getType()))
    return false;
  if (alloc.getElementSizes().empty())
    return false;
  if (llvm::all_of(alloc.getElementSizes(), ValueAnalysis::isOneLikeValue))
    return false;

  PartitionMode mode =
      getPartitionMode(alloc.getOperation()).value_or(PartitionMode::coarse);
  return mode == PartitionMode::coarse;
}

bool DbUtils::isSmallCoarseUserDataDb(DbAllocOp alloc, int64_t maxElements) {
  if (maxElements <= 0 || !DbUtils::isCoarseUserDataDb(alloc))
    return false;
  std::optional<int64_t> elementCount = DbUtils::getStaticElementCount(alloc);
  return elementCount && *elementCount <= maxElements;
}

bool DbUtils::isRejectedForDistributedOwnership(DbAllocOp alloc) {
  if (!alloc || hasDistributedDbAllocation(alloc.getOperation()))
    return false;
  return alloc.getDistributedRejectReasonAttr() ||
         alloc.getLocalOnly().value_or(false);
}

bool DbUtils::isAllowedSmallReadOnlyCoarseDep(Value dep, DbAllocOp alloc) {
  auto acquire = dep.getDefiningOp<DbAcquireOp>();
  return acquire && acquire.getMode() == ArtsMode::in &&
         DbUtils::isSmallCoarseUserDataDb(alloc);
}

bool DbUtils::isAllowedReadOnlyCoarseDep(Value dep, DbAllocOp alloc) {
  auto acquire = dep.getDefiningOp<DbAcquireOp>();
  if (!acquire || acquire.getMode() != ArtsMode::in)
    return false;
  if (DbUtils::isSmallCoarseUserDataDb(alloc))
    return true;
  return DbUtils::isCoarseUserDataDb(alloc) &&
         static_cast<bool>(acquire.getReplicatedReadAttr());
}

static bool isHostWholeToComputeBlockBridgeDb(DbAllocOp alloc) {
  if (!alloc)
    return false;
  auto bridge = alloc.getStorageBridgeAttr();
  return bridge &&
         bridge.getValue() == StorageBridge::host_whole_to_compute_block;
}

bool DbUtils::isHostWholeToComputeBlockBridgeMovement(EdtOp edt) {
  if (!edt || !edt.getStorageBridgeCopyAttr() ||
      edt.getDependencies().size() != 2 || edt.getDepPatternAttr() ||
      getPlanOwnerDimsAttr(edt.getOperation()) ||
      getPlanPhysicalBlockShapeAttr(edt.getOperation()))
    return false;

  bool hasCoarseHost = false;
  bool hasBlockBridge = false;
  for (Value dep : edt.getDependencies()) {
    auto acquire = dep.getDefiningOp<DbAcquireOp>();
    if (!acquire)
      return false;
    auto alloc =
        dyn_cast_or_null<DbAllocOp>(DbUtils::getUnderlyingDbAlloc(dep));
    PartitionMode partition =
        acquire.getPartitionMode().value_or(PartitionMode::coarse);
    if (DbUtils::isCoarseUserDataDb(alloc) &&
        partition == PartitionMode::coarse) {
      hasCoarseHost = true;
      continue;
    }
    if (isHostWholeToComputeBlockBridgeDb(alloc) &&
        partition == PartitionMode::block) {
      hasBlockBridge = true;
      continue;
    }
    return false;
  }
  return hasCoarseHost && hasBlockBridge;
}

bool DbUtils::requiresLocalLaunchForDistributedDep(Value dep) {
  auto alloc = dyn_cast_or_null<DbAllocOp>(DbUtils::getUnderlyingDbAlloc(dep));
  if (!DbUtils::isRejectedForDistributedOwnership(alloc))
    return false;
  return !DbUtils::isAllowedReadOnlyCoarseDep(dep, alloc);
}

bool DbUtils::hasLocalOnlyDistributedLaunchDependency(EdtOp edt) {
  if (!edt)
    return false;
  if (DbUtils::isHostWholeToComputeBlockBridgeMovement(edt))
    return false;
  for (Value dep : edt.getDependencies())
    if (DbUtils::requiresLocalLaunchForDistributedDep(dep))
      return true;
  return false;
}

DbMode DbUtils::convertArtsModeToDbMode(ArtsMode mode) {
  return (mode == ArtsMode::in) ? DbMode::read : DbMode::write;
}

Value DbUtils::pickRepresentativePartitionOffset(ArrayRef<Value> offsets,
                                                 unsigned *outIdx) {
  if (outIdx)
    *outIdx = 0;
  for (unsigned i = 0; i < offsets.size(); ++i) {
    Value off = offsets[i];
    if (!off)
      continue;
    int64_t c = 0;
    if (!ValueAnalysis::getConstantIndex(ValueAnalysis::stripNumericCasts(off),
                                         c)) {
      if (outIdx)
        *outIdx = i;
      return off;
    }
  }
  for (unsigned i = 0; i < offsets.size(); ++i) {
    if (offsets[i]) {
      if (outIdx)
        *outIdx = i;
      return offsets[i];
    }
  }
  return Value();
}

Value DbUtils::pickRepresentativePartitionSize(ArrayRef<Value> sizes,
                                               unsigned idx) {
  if (sizes.empty())
    return Value();
  if (idx < sizes.size() && sizes[idx])
    return sizes[idx];
  return sizes.front();
}

SmallVector<Value> DbUtils::collectFullIndexChain(DbRefOp dbRef,
                                                  Operation *memOp) {
  SmallVector<Value> chain(dbRef.getIndices().begin(),
                           dbRef.getIndices().end());
  Value accessedMemref = getAccessedMemref(memOp);
  Value anchor = dbRef.getResult();

  SmallVector<Operation *, 8> forwardingOps;
  DenseSet<Value> visitedMemrefs;
  Value current = accessedMemref;
  while (current && current != anchor &&
         visitedMemrefs.insert(current).second) {
    Operation *def = current.getDefiningOp();
    if (!def)
      break;

    if (auto castOp = dyn_cast<memref::CastOp>(def)) {
      forwardingOps.push_back(def);
      current = castOp.getSource();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      forwardingOps.push_back(def);
      current = subview.getSource();
      continue;
    }
    if (auto view = dyn_cast<memref::ViewOp>(def)) {
      forwardingOps.push_back(def);
      current = view.getSource();
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      forwardingOps.push_back(def);
      current = reinterpret.getSource();
      continue;
    }
    if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (unrealized.getInputs().empty())
        break;
      forwardingOps.push_back(def);
      current = unrealized.getInputs().front();
      continue;
    }
    break;
  }

  for (Operation *op : llvm::reverse(forwardingOps))
    if (auto subview = dyn_cast<memref::SubViewOp>(op))
      appendDynamicSubviewOffsets(subview, chain);

  SmallVector<Value> memIndices = getMemoryAccessIndices(memOp);
  chain.append(memIndices.begin(), memIndices.end());
  return chain;
}

SmallVector<Value> DbUtils::getMemoryAccessIndices(Operation *memOp) {
  if (auto access = getMemoryAccessInfo(memOp))
    return access->indices;
  return {};
}

std::optional<DbUtils::MemoryAccessInfo>
DbUtils::getMemoryAccessInfo(Operation *memOp) {
  if (!memOp)
    return std::nullopt;

  if (auto load = dyn_cast<memref::LoadOp>(memOp))
    return MemoryAccessInfo{
        memOp, load.getMemRef(),
        SmallVector<Value>(load.getIndices().begin(), load.getIndices().end()),
        MemoryAccessKind::Read};
  if (auto store = dyn_cast<memref::StoreOp>(memOp))
    return MemoryAccessInfo{memOp, store.getMemRef(),
                            SmallVector<Value>(store.getIndices().begin(),
                                               store.getIndices().end()),
                            MemoryAccessKind::Write};
  if (auto load = dyn_cast<polygeist::DynLoadOp>(memOp))
    return MemoryAccessInfo{
        memOp, load.getMemref(),
        SmallVector<Value>(load.getIndices().begin(), load.getIndices().end()),
        MemoryAccessKind::Read};
  if (auto store = dyn_cast<polygeist::DynStoreOp>(memOp))
    return MemoryAccessInfo{memOp, store.getMemref(),
                            SmallVector<Value>(store.getIndices().begin(),
                                               store.getIndices().end()),
                            MemoryAccessKind::Write};
  if (auto load = dyn_cast<affine::AffineLoadOp>(memOp))
    return MemoryAccessInfo{memOp, load.getMemRef(),
                            SmallVector<Value>(load.getMapOperands().begin(),
                                               load.getMapOperands().end()),
                            MemoryAccessKind::Read};
  if (auto store = dyn_cast<affine::AffineStoreOp>(memOp))
    return MemoryAccessInfo{memOp, store.getMemRef(),
                            SmallVector<Value>(store.getMapOperands().begin(),
                                               store.getMapOperands().end()),
                            MemoryAccessKind::Write};
  return std::nullopt;
}

bool DbUtils::hasSharedWritableRootConflict(Operation *lhs, Operation *rhs) {
  MemoryRootSummary lhsSummary = summarizeMemoryRoots(lhs);
  MemoryRootSummary rhsSummary = summarizeMemoryRoots(rhs);

  for (const auto &[alloc, access] : lhsSummary.dbAllocs) {
    auto it = rhsSummary.dbAllocs.find(alloc);
    if (it != rhsSummary.dbAllocs.end() &&
        (access.hasWrite() || it->second.hasWrite())) {
      return true;
    }
  }

  for (const auto &[memref, access] : lhsSummary.rawMemrefs) {
    auto it = rhsSummary.rawMemrefs.find(memref);
    if (it != rhsSummary.rawMemrefs.end() &&
        (access.hasWrite() || it->second.hasWrite())) {
      return true;
    }
  }

  return false;
}

void DbUtils::forEachReachableMemoryAccess(
    Value source, llvm::function_ref<WalkResult(const MemoryAccessInfo &)> fn,
    Region *scope) {
  if (!source)
    return;

  SmallVector<Value, 16> worklist;
  DenseSet<Value> visited;
  worklist.push_back(source);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      if (scope && !scope->isAncestor(user->getParentRegion()))
        continue;

      if (auto access = getMemoryAccessInfo(user)) {
        if (access->memref == current) {
          if (fn(*access).wasInterrupted())
            return;
        }
        continue;
      }

      if (!isMemrefForwardingForSource(user, current))
        continue;
      if (user->getNumResults() == 0)
        continue;

      for (Value result : user->getResults())
        if (isa<MemRefType>(result.getType()))
          worklist.push_back(result);
    }
  }
}

void DbUtils::collectReachableMemoryOps(Value source,
                                        llvm::SetVector<Operation *> &memOps,
                                        Region *scope) {
  forEachReachableMemoryAccess(
      source,
      [&](const MemoryAccessInfo &access) {
        memOps.insert(access.op);
        return WalkResult::advance();
      },
      scope);
}

bool DbUtils::collectCleanupOnlyUseChain(Value source,
                                         llvm::SetVector<Operation *> &ops,
                                         Region *scope) {
  if (!source)
    return true;

  SmallVector<Value, 16> worklist;
  DenseSet<Value> visited;
  worklist.push_back(source);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      if (scope && !scope->isAncestor(user->getParentRegion()))
        continue;

      if (auto access = getMemoryAccessInfo(user)) {
        if (access->memref == current)
          return false;
        continue;
      }

      if (isa<EdtOp>(user))
        return false;

      if (isCleanupTerminalOp(user, current)) {
        ops.insert(user);
        continue;
      }

      if (!isStructuralDbChainUser(user, current))
        return false;

      ops.insert(user);
      enqueueStructuralDbChainResults(user, worklist);
    }
  }

  return true;
}

bool DbUtils::collectTrueOnlyControlTokenUseChain(
    Value source, llvm::SetVector<Operation *> &ops, Region *scope) {
  if (!source || !isI1DbPtrType(source.getType()))
    return false;

  SmallVector<Value, 8> worklist{source};
  DenseSet<Value> visited;
  bool sawTrueStore = false;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      if (scope && !scope->isAncestor(user->getParentRegion()))
        continue;

      if (isCleanupTerminalOp(user, current)) {
        ops.insert(user);
        continue;
      }

      if (auto dbRef = dyn_cast<DbRefOp>(user)) {
        if (dbRef.getSource() != current)
          return false;
        ops.insert(user);
        worklist.push_back(dbRef.getResult());
        continue;
      }

      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemRef() != current ||
            !ValueAnalysis::isTrueConstant(store.getValue()))
          return false;
        sawTrueStore = true;
        ops.insert(user);
        continue;
      }

      return false;
    }
  }

  return sawTrueStore;
}

void arts::DbUtils::convertElementSliceToBlockSlice(
    OpBuilder &builder, Location loc, ValueRange elementOffsets,
    ValueRange elementSizes, ValueRange blockSpans, ValueRange totalBlockCounts,
    SmallVectorImpl<Value> &blockOffsets, SmallVectorImpl<Value> &blockSizes) {
  unsigned rank = std::min({static_cast<unsigned>(elementOffsets.size()),
                            static_cast<unsigned>(elementSizes.size()),
                            static_cast<unsigned>(blockSpans.size()),
                            static_cast<unsigned>(totalBlockCounts.size())});
  Value zero = createZeroIndex(builder, loc);
  Value one = createOneIndex(builder, loc);
  blockOffsets.reserve(rank);
  blockSizes.reserve(rank);

  for (unsigned i = 0; i < rank; ++i) {
    Value elementOffset =
        ValueAnalysis::castToIndex(elementOffsets[i], builder, loc);
    Value elementSize =
        ValueAnalysis::castToIndex(elementSizes[i], builder, loc);
    Value blockSpan = ValueAnalysis::castToIndex(blockSpans[i], builder, loc);
    Value totalBlocks =
        ValueAnalysis::castToIndex(totalBlockCounts[i], builder, loc);

    blockSpan = arith::MaxUIOp::create(builder, loc, blockSpan, one);
    elementSize = arith::MaxUIOp::create(builder, loc, elementSize, one);

    Value startBlock =
        arith::DivUIOp::create(builder, loc, elementOffset, blockSpan);
    Value endElem =
        arith::AddIOp::create(builder, loc, elementOffset, elementSize);
    endElem = arith::SubIOp::create(builder, loc, endElem, one);
    Value endBlock = arith::DivUIOp::create(builder, loc, endElem, blockSpan);
    Value maxBlock = arith::SubIOp::create(builder, loc, totalBlocks, one);
    Value startAboveMax = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ugt, startBlock, maxBlock);
    Value clampedEnd = arith::MinUIOp::create(builder, loc, endBlock, maxBlock);
    endBlock = arith::SelectOp::create(builder, loc, startAboveMax, endBlock,
                                       clampedEnd);

    Value blockCount =
        arith::SubIOp::create(builder, loc, endBlock, startBlock);
    blockCount = arith::AddIOp::create(builder, loc, blockCount, one);
    startBlock =
        arith::SelectOp::create(builder, loc, startAboveMax, zero, startBlock);
    blockCount =
        arith::SelectOp::create(builder, loc, startAboveMax, zero, blockCount);

    blockOffsets.push_back(startBlock);
    blockSizes.push_back(blockCount);
  }
}

void arts::DbUtils::mergeNormalizedBlockSlice(
    OpBuilder &builder, Location loc, ValueRange existingOffsets,
    ValueRange existingSizes, ValueRange totalBlockCounts,
    ValueRange normalizedOffsets, ValueRange normalizedSizes,
    SmallVectorImpl<Value> &blockOffsets, SmallVectorImpl<Value> &blockSizes) {
  unsigned ownerRank = existingOffsets.size();
  ownerRank = std::max(ownerRank, static_cast<unsigned>(existingSizes.size()));
  ownerRank =
      std::max(ownerRank, static_cast<unsigned>(totalBlockCounts.size()));
  ownerRank =
      std::max(ownerRank, static_cast<unsigned>(normalizedOffsets.size()));
  ownerRank =
      std::max(ownerRank, static_cast<unsigned>(normalizedSizes.size()));
  if (ownerRank == 0) {
    blockOffsets.clear();
    blockSizes.clear();
    return;
  }

  Value zero = createZeroIndex(builder, loc);
  Value one = createOneIndex(builder, loc);
  blockOffsets.assign(ownerRank, zero);
  blockSizes.assign(ownerRank, one);

  for (unsigned i = 0; i < ownerRank; ++i) {
    if (i < totalBlockCounts.size() && totalBlockCounts[i])
      blockSizes[i] =
          ValueAnalysis::castToIndex(totalBlockCounts[i], builder, loc);
    if (i < existingOffsets.size() && existingOffsets[i])
      blockOffsets[i] =
          ValueAnalysis::castToIndex(existingOffsets[i], builder, loc);
    if (i < existingSizes.size() && existingSizes[i])
      blockSizes[i] =
          ValueAnalysis::castToIndex(existingSizes[i], builder, loc);
  }

  unsigned normalizedRank =
      std::min(static_cast<unsigned>(normalizedOffsets.size()),
               static_cast<unsigned>(normalizedSizes.size()));
  for (unsigned i = 0; i < normalizedRank; ++i) {
    blockOffsets[i] =
        ValueAnalysis::castToIndex(normalizedOffsets[i], builder, loc);
    blockSizes[i] =
        ValueAnalysis::castToIndex(normalizedSizes[i], builder, loc);
  }
}

std::optional<int64_t> arts::DbUtils::extractBlockSizeFromHint(Value sizeHint,
                                                               int depth) {
  if (!sizeHint || depth > 4)
    return std::nullopt;

  /// Case 1: Direct constant.
  if (auto folded = arts::tryFoldConstantIndex(sizeHint))
    return folded;

  /// Case 2/3: minui/minsi pattern — return the larger constant (nominal size).
  auto handleMinOp = [&](Value lhs, Value rhs) -> std::optional<int64_t> {
    auto lhsFolded = arts::tryFoldConstantIndex(lhs);
    auto rhsFolded = arts::tryFoldConstantIndex(rhs);

    if (lhsFolded && rhsFolded)
      return std::max(*lhsFolded, *rhsFolded);
    if (lhsFolded)
      return lhsFolded;
    if (rhsFolded)
      return rhsFolded;

    auto lhsExtracted = DbUtils::extractBlockSizeFromHint(lhs, depth + 1);
    auto rhsExtracted = DbUtils::extractBlockSizeFromHint(rhs, depth + 1);
    if (lhsExtracted && rhsExtracted)
      return std::max(*lhsExtracted, *rhsExtracted);
    if (lhsExtracted)
      return lhsExtracted;
    if (rhsExtracted)
      return rhsExtracted;
    return std::nullopt;
  };

  if (auto minOp = sizeHint.getDefiningOp<arith::MinUIOp>()) {
    if (auto result = handleMinOp(minOp.getLhs(), minOp.getRhs()))
      return result;
  }
  if (auto minOp = sizeHint.getDefiningOp<arith::MinSIOp>()) {
    if (auto result = handleMinOp(minOp.getLhs(), minOp.getRhs()))
      return result;
  }

  /// Case 4: addi pattern — stencil halo; extract baseBlockSize from
  /// addi(baseBlockSize, haloAdjustment) when halo is a small constant.
  if (auto addOp = sizeHint.getDefiningOp<arith::AddIOp>()) {
    auto lhsFolded = arts::tryFoldConstantIndex(addOp.getLhs());
    auto rhsFolded = arts::tryFoldConstantIndex(addOp.getRhs());

    /// If one operand is a small constant (halo), recurse on the other.
    if (rhsFolded && std::abs(*rhsFolded) <= 16) {
      return DbUtils::extractBlockSizeFromHint(addOp.getLhs(), depth + 1);
    }
    if (lhsFolded && std::abs(*lhsFolded) <= 16) {
      return DbUtils::extractBlockSizeFromHint(addOp.getRhs(), depth + 1);
    }

    /// Both constants or both large — try both sides.
    auto lhsExtracted =
        DbUtils::extractBlockSizeFromHint(addOp.getLhs(), depth + 1);
    auto rhsExtracted =
        DbUtils::extractBlockSizeFromHint(addOp.getRhs(), depth + 1);
    if (lhsExtracted)
      return lhsExtracted;
    if (rhsExtracted)
      return rhsExtracted;
  }

  /// Case 4b: subi pattern — stencil rewriters often recover the owned
  /// block span by subtracting a small halo width from an expanded slice.
  /// Keep extracting the nominal block size from the minuend.
  if (auto subOp = sizeHint.getDefiningOp<arith::SubIOp>()) {
    auto rhsFolded = arts::tryFoldConstantIndex(subOp.getRhs());
    if (rhsFolded && std::abs(*rhsFolded) <= 16)
      return DbUtils::extractBlockSizeFromHint(subOp.getLhs(), depth + 1);
  }

  /// Case 5: maxui pattern — clamp minimum; return larger constant as upper
  /// bound.
  if (auto maxOp = sizeHint.getDefiningOp<arith::MaxUIOp>()) {
    auto lhsFolded = arts::tryFoldConstantIndex(maxOp.getLhs());
    auto rhsFolded = arts::tryFoldConstantIndex(maxOp.getRhs());

    if (lhsFolded && rhsFolded)
      return std::max(*lhsFolded, *rhsFolded);
    if (lhsFolded && !rhsFolded) {
      auto rhsExtracted =
          DbUtils::extractBlockSizeFromHint(maxOp.getRhs(), depth + 1);
      if (rhsExtracted)
        return rhsExtracted;
      return lhsFolded;
    }
    if (rhsFolded && !lhsFolded) {
      auto lhsExtracted =
          DbUtils::extractBlockSizeFromHint(maxOp.getLhs(), depth + 1);
      if (lhsExtracted)
        return lhsExtracted;
      return rhsFolded;
    }

    /// Recurse for nested maxui.
    auto lhsExtracted =
        DbUtils::extractBlockSizeFromHint(maxOp.getLhs(), depth + 1);
    auto rhsExtracted =
        DbUtils::extractBlockSizeFromHint(maxOp.getRhs(), depth + 1);
    if (lhsExtracted && rhsExtracted)
      return std::max(*lhsExtracted, *rhsExtracted);
    if (lhsExtracted)
      return lhsExtracted;
    if (rhsExtracted)
      return rhsExtracted;
  }

  return std::nullopt;
}
