///==========================================================================///
/// File: DbUtils.cpp
///
/// Implementation of utility functions for working with ARTS DBs.
///==========================================================================///

#include "carts/dialect/arts/Utils/DbUtils.h"
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
#include "llvm/ADT/SmallPtrSet.h"

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_utils);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
using namespace mlir::carts::arts_rt;

namespace {

static Operation *resolveRootAllocFromAttr(Value v) {
  if (!v)
    return nullptr;

  std::optional<int64_t> rootId;
  SmallPtrSet<Operation *, 8> visited;
  for (unsigned depth = 0; depth < 16 && v; ++depth) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp || !visited.insert(defOp).second)
      break;

    if (auto candidate = getDbRootAllocId(defOp)) {
      rootId = candidate;
      break;
    }

    if (auto castOp = dyn_cast<memref::CastOp>(defOp)) {
      v = castOp.getSource();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(defOp)) {
      v = subview.getSource();
      continue;
    }
    if (auto dbRef = dyn_cast<DbRefOp>(defOp)) {
      v = dbRef.getSource();
      continue;
    }
    if (auto dbGep = dyn_cast<DbGepOp>(defOp)) {
      v = dbGep.getBasePtr();
      continue;
    }
    break;
  }

  if (!rootId || *rootId <= 0)
    return nullptr;

  auto module = v.getParentRegion()
                    ? v.getParentRegion()->getParentOfType<ModuleOp>()
                    : ModuleOp();
  if (!module)
    return nullptr;

  Operation *resolved = nullptr;
  module.walk([&](DbAllocOp alloc) {
    if (getArtsId(alloc) != *rootId)
      return WalkResult::advance();
    resolved = alloc.getOperation();
    return WalkResult::interrupt();
  });
  return resolved;
}

static void getAcquireDepSlice(DbAcquireOp acquire,
                               SmallVector<Value> &sizesOut,
                               SmallVector<Value> &offsetsOut) {
  sizesOut.assign(acquire.getSizes().begin(), acquire.getSizes().end());
  offsetsOut.assign(acquire.getOffsets().begin(), acquire.getOffsets().end());
}

static bool isMemrefForwardingOp(Operation *op, Value source) {
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

static bool isStructuralHostDbUse(Operation *op) {
  return isa<EdtOp, LoweringContractOp, DbReleaseOp, DbFreeOp>(op);
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
  if (isMemrefForwardingOp(op, current))
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

static bool isNullPointerLike(Value value) {
  if (!value)
    return false;

  if (isa<BlockArgument>(value))
    return false;

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;

  if (isa<LLVM::ZeroOp>(defOp))
    return true;

  if (auto constant = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constant.getValue()))
      return intAttr.getValue().isZero();
  }

  return false;
}

static bool isNullCheckOnlyPointerCast(polygeist::Memref2PointerOp castOp) {
  if (!castOp)
    return false;

  for (OpOperand &use : castOp.getResult().getUses()) {
    auto cmp = dyn_cast<LLVM::ICmpOp>(use.getOwner());
    if (!cmp)
      return false;

    auto pred = cmp.getPredicate();
    if (pred != LLVM::ICmpPredicate::eq && pred != LLVM::ICmpPredicate::ne)
      return false;

    Value lhs = cmp.getLhs();
    Value rhs = cmp.getRhs();
    bool lhsIsCast = lhs == castOp.getResult();
    bool rhsIsCast = rhs == castOp.getResult();
    if (!lhsIsCast && !rhsIsCast)
      return false;

    Value other = lhsIsCast ? rhs : lhs;
    if (!isNullPointerLike(other))
      return false;
  }

  return true;
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
/// Datablock Lowering Info Extraction
///===----------------------------------------------------------------------===///

/// Shared implementation: dispatches on the concrete operation type.
template <typename OpType>
DbLoweringInfo DbUtils::extractDbLoweringInfo(OpType op) {
  DbLoweringInfo info;

  if (auto acqOp = dyn_cast<DbAcquireOp>(op.getOperation())) {
    info.sizes = getDepSizesFromDb(acqOp.getOperation());
    info.offsets = getDepOffsetsFromDb(acqOp.getOperation());
    info.indices.assign(acqOp.getIndices().begin(), acqOp.getIndices().end());
  } else if (auto depAcqOp = dyn_cast<DepDbAcquireOp>(op.getOperation())) {
    info.sizes = getDepSizesFromDb(depAcqOp.getOperation());
    info.offsets = getDepOffsetsFromDb(depAcqOp.getOperation());
    info.indices.assign(depAcqOp.getIndices().begin(),
                        depAcqOp.getIndices().end());
  } else {
    info.sizes.assign(op.getSizes().begin(), op.getSizes().end());
    info.offsets.clear();
    info.indices.clear();
  }

  info.isSingleElement = false;
  if (info.sizes.empty()) {
    info.isSingleElement = true;
    return info;
  }
  if (info.sizes.size() == 1) {
    info.isSingleElement = ValueAnalysis::isOneConstant(info.sizes[0]);
  }
  return info;
}

/// Explicit template instantiations for the operation types used in lowering.
template DbLoweringInfo
DbUtils::extractDbLoweringInfo<DbAcquireOp>(DbAcquireOp op);
template DbLoweringInfo
DbUtils::extractDbLoweringInfo<DepDbAcquireOp>(DepDbAcquireOp op);
template DbLoweringInfo DbUtils::extractDbLoweringInfo<DbAllocOp>(DbAllocOp op);

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
  if (auto depAcq = v.getDefiningOp<DepDbAcquireOp>())
    return depAcq.getOperation();
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
    if (Operation *rootAlloc = resolveRootAllocFromAttr(v))
      return rootAlloc;
    if (auto gep = dyn_cast<DbGepOp>(def))
      return getUnderlyingDb(gep.getBasePtr(), depth + 1);
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

  if (Operation *rootAlloc = resolveRootAllocFromAttr(v))
    return rootAlloc;

  if (Operation *root = ValueAnalysis::getUnderlyingOperation(v))
    if (isa<DbAcquireOp, DepDbAcquireOp, DbAllocOp>(root))
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
  if (auto depDbAcquireOp = dbGuid.getDefiningOp<DepDbAcquireOp>())
    return depDbAcquireOp.getGuid()
               ? depDbAcquireOp.getGuid().getDefiningOp<DbAllocOp>()
               : nullptr;
  return nullptr;
}

std::optional<int64_t> DbUtils::resolveRootAllocId(Value value,
                                                   DbAllocOp fallbackAlloc) {
  if (fallbackAlloc) {
    int64_t allocId = getArtsId(fallbackAlloc);
    if (allocId > 0)
      return allocId;
  }

  if (Operation *rootOp = getUnderlyingDbAlloc(value))
    if (auto rootAlloc = dyn_cast<DbAllocOp>(rootOp)) {
      int64_t allocId = getArtsId(rootAlloc);
      if (allocId > 0)
        return allocId;
    }

  if (auto rootId = getDbRootAllocId(value); rootId && *rootId > 0)
    return rootId;

  return std::nullopt;
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
  if (auto depDbAcquireOp = dyn_cast<DepDbAcquireOp>(dbOp)) {
    return SmallVector<Value>(depDbAcquireOp.getSizes().begin(),
                              depDbAcquireOp.getSizes().end());
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

  if (auto depAcquireOp = dyn_cast_or_null<DepDbAcquireOp>(dbOp))
    return SmallVector<Value>(depAcquireOp.getSizes().begin(),
                              depAcquireOp.getSizes().end());

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

  if (auto depAcquireOp = dyn_cast_or_null<DepDbAcquireOp>(dbOp))
    return SmallVector<Value>(depAcquireOp.getOffsets().begin(),
                              depAcquireOp.getOffsets().end());

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

DbMode DbUtils::convertArtsModeToDbMode(ArtsMode mode) {
  return (mode == ArtsMode::in) ? DbMode::read : DbMode::write;
}

/// Custom use-chain traversal: upstream view-chain utilities
/// (skipFullyAliasingOperations, skipViewLikeOps) walk UP the def chain to
/// find the root allocation. This function walks DOWN the use chain to
/// detect non-structural host consumers that would block partitioning.
/// It also handles CARTS-specific ops (EdtOp, DbAcquireOp, DbReleaseOp,
/// LoweringContractOp, Memref2PointerOp null-checks) that have no upstream
/// equivalent, so the custom traversal is necessary.
bool DbUtils::hasNonPartitionableHostViewUses(Value dbValue) {
  if (!dbValue)
    return false;

  SmallVector<Value, 8> worklist{dbValue};
  llvm::SmallPtrSet<Value, 16> visitedValues;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visitedValues.insert(current).second)
      continue;

    for (OpOperand &use : llvm::make_early_inc_range(current.getUses())) {
      Operation *user = use.getOwner();
      if (!user || user->getParentOfType<EdtOp>())
        continue;

      if (isa<memref::DeallocOp>(user))
        continue;

      if (getMemoryAccessInfo(user))
        continue;

      if (isMemrefForwardingOp(user, current)) {
        for (Value result : user->getResults())
          if (isa<MemRefType>(result.getType()))
            worklist.push_back(result);
        continue;
      }

      /// Pointer casts used only for null/non-null checks do not require a
      /// contiguous whole-view layout. Keep treating true pointer escapes as
      /// unsafe, but allow allocation-failure guards on DB-backed views.
      if (auto memrefToPtr = dyn_cast<polygeist::Memref2PointerOp>(user)) {
        if (isNullCheckOnlyPointerCast(memrefToPtr))
          continue;
      }

      /// ARTS dependency plumbing outside an EDT is structural, not a host
      /// whole-view consumer. Follow acquire ptr results so later opaque host
      /// uses (for example helper calls on a host acquire) are still detected.
      if (auto acquire = dyn_cast<DbAcquireOp>(user)) {
        if (acquire.getSourcePtr() == current &&
            isa<MemRefType>(acquire.getPtr().getType()))
          worklist.push_back(acquire.getPtr());
        continue;
      }

      if (isStructuralHostDbUse(user))
        continue;

      ARTS_DEBUG("Host-view veto: non-partitionable host user "
                 << user->getName() << " on value type " << current.getType());
      return true;
    }
  }

  return false;
}

Value DbUtils::extractBaseBlockSizeCandidate(Value offsetHint, Value sizeHint,
                                             int depth) {
  if (!sizeHint || depth > 6)
    return Value();

  if (!offsetHint || !ValueAnalysis::dependsOn(sizeHint, offsetHint))
    return sizeHint;

  Operation *defOp = sizeHint.getDefiningOp();
  if (!defOp)
    return Value();

  if (auto minOp = dyn_cast<arith::MinUIOp>(defOp)) {
    Value lhs = minOp.getLhs();
    Value rhs = minOp.getRhs();
    bool lhsDep = ValueAnalysis::dependsOn(lhs, offsetHint);
    bool rhsDep = ValueAnalysis::dependsOn(rhs, offsetHint);
    if (lhsDep && !rhsDep)
      return extractBaseBlockSizeCandidate(offsetHint, rhs, depth + 1);
    if (rhsDep && !lhsDep)
      return extractBaseBlockSizeCandidate(offsetHint, lhs, depth + 1);
    if (Value cand = extractBaseBlockSizeCandidate(offsetHint, lhs, depth + 1))
      return cand;
    return extractBaseBlockSizeCandidate(offsetHint, rhs, depth + 1);
  }

  if (auto minOp = dyn_cast<arith::MinSIOp>(defOp)) {
    Value lhs = minOp.getLhs();
    Value rhs = minOp.getRhs();
    bool lhsDep = ValueAnalysis::dependsOn(lhs, offsetHint);
    bool rhsDep = ValueAnalysis::dependsOn(rhs, offsetHint);
    if (lhsDep && !rhsDep)
      return extractBaseBlockSizeCandidate(offsetHint, rhs, depth + 1);
    if (rhsDep && !lhsDep)
      return extractBaseBlockSizeCandidate(offsetHint, lhs, depth + 1);
    if (Value cand = extractBaseBlockSizeCandidate(offsetHint, lhs, depth + 1))
      return cand;
    return extractBaseBlockSizeCandidate(offsetHint, rhs, depth + 1);
  }

  if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    Value tVal = selectOp.getTrueValue();
    Value fVal = selectOp.getFalseValue();
    bool tDep = ValueAnalysis::dependsOn(tVal, offsetHint);
    bool fDep = ValueAnalysis::dependsOn(fVal, offsetHint);
    if (tDep && !fDep)
      return extractBaseBlockSizeCandidate(offsetHint, fVal, depth + 1);
    if (fDep && !tDep)
      return extractBaseBlockSizeCandidate(offsetHint, tVal, depth + 1);
    if (Value cand = extractBaseBlockSizeCandidate(offsetHint, tVal, depth + 1))
      return cand;
    return extractBaseBlockSizeCandidate(offsetHint, fVal, depth + 1);
  }

  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    int64_t lhsConst = 0, rhsConst = 0;
    bool lhsIsConst = ValueAnalysis::getConstantIndex(addOp.getLhs(), lhsConst);
    bool rhsIsConst = ValueAnalysis::getConstantIndex(addOp.getRhs(), rhsConst);
    if (lhsIsConst && lhsConst >= -16 && lhsConst <= 16)
      return extractBaseBlockSizeCandidate(offsetHint, addOp.getRhs(),
                                           depth + 1);
    if (rhsIsConst && rhsConst >= -16 && rhsConst <= 16)
      return extractBaseBlockSizeCandidate(offsetHint, addOp.getLhs(),
                                           depth + 1);
  }

  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    int64_t rhsConst = 0;
    if (ValueAnalysis::getConstantIndex(subOp.getRhs(), rhsConst) &&
        rhsConst >= -16 && rhsConst <= 16)
      return extractBaseBlockSizeCandidate(offsetHint, subOp.getLhs(),
                                           depth + 1);
  }

  return Value();
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

      if (!isMemrefForwardingOp(user, current))
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
  if (auto folded = ValueAnalysis::tryFoldConstantIndex(sizeHint))
    return folded;

  /// Case 2/3: minui/minsi pattern — return the larger constant (nominal size).
  auto handleMinOp = [&](Value lhs, Value rhs) -> std::optional<int64_t> {
    auto lhsFolded = ValueAnalysis::tryFoldConstantIndex(lhs);
    auto rhsFolded = ValueAnalysis::tryFoldConstantIndex(rhs);

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
    auto lhsFolded = ValueAnalysis::tryFoldConstantIndex(addOp.getLhs());
    auto rhsFolded = ValueAnalysis::tryFoldConstantIndex(addOp.getRhs());

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
    auto rhsFolded = ValueAnalysis::tryFoldConstantIndex(subOp.getRhs());
    if (rhsFolded && std::abs(*rhsFolded) <= 16)
      return DbUtils::extractBlockSizeFromHint(subOp.getLhs(), depth + 1);
  }

  /// Case 5: maxui pattern — clamp minimum; return larger constant as upper
  /// bound.
  if (auto maxOp = sizeHint.getDefiningOp<arith::MaxUIOp>()) {
    auto lhsFolded = ValueAnalysis::tryFoldConstantIndex(maxOp.getLhs());
    auto rhsFolded = ValueAnalysis::tryFoldConstantIndex(maxOp.getRhs());

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
