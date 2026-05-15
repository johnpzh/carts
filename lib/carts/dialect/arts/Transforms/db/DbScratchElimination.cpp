///==========================================================================///
/// File: DbScratchElimination.cpp
///
/// Eliminate task-private single-element scratch DBs by replacing them with
/// local allocas.  Runs after DbModeTightening / DbPartitioning have
/// established stable acquire and dependency structure.
///==========================================================================///

#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/utils/ValueAnalysis.h"
#define GEN_PASS_DEF_DBSCRATCHELIMINATION
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/EdtUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/dialect/arts/Utils/PartitionPredicates.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_scratch_elimination);

#include "llvm/ADT/Statistic.h"
static llvm::Statistic numScratchDbsEliminated{
    "db_scratch_elimination", "NumScratchDbsEliminated",
    "Number of scratch DBs fully eliminated"};
static llvm::Statistic numScratchUsesReplaced{
    "db_scratch_elimination", "NumScratchUsesReplaced",
    "Number of scratch DB uses replaced with local allocas"};
static llvm::Statistic numScratchCandidatesRejected{
    "db_scratch_elimination", "NumScratchCandidatesRejected",
    "Number of scratch DB candidates rejected"};

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

struct ScratchUse {
  DbAcquireOp acquire;
  EdtOp edt;
  BlockArgument blockArg;
  DbRefOp dbRef;
};

struct ScratchCandidate {
  DbAllocOp alloc;
  DbRefOp initRef;
  Operation *initStore = nullptr;
  SmallVector<DbFreeOp, 2> freeOps;
  SmallVector<ScratchUse, 4> uses;
};

static SmallVector<Value> getDynamicAllocaSizes(DbAllocOp alloc,
                                                MemRefType refType) {
  SmallVector<Value> dynSizes;
  ValueRange elementSizes = alloc.getElementSizes();
  if (refType.getRank() != static_cast<int64_t>(elementSizes.size()))
    return {};

  for (auto [dim, extent] : llvm::enumerate(refType.getShape()))
    if (ShapedType::isDynamic(extent))
      dynSizes.push_back(elementSizes[dim]);
  return dynSizes;
}

static bool matchInitStore(DbRefOp initRef, Operation *&initStore) {
  initStore = nullptr;
  if (!initRef)
    return true;
  if (initRef.getIndices().empty() ||
      !llvm::all_of(initRef.getIndices(), ValueAnalysis::isZeroConstant))
    return false;
  if (!initRef->hasOneUse())
    return false;

  Operation *user = *initRef->user_begin();
  auto access = DbUtils::getMemoryAccessInfo(user);
  if (!access || !access->isWrite() || access->memref != initRef.getResult())
    return false;

  initStore = user;
  return true;
}

static bool matchScratchDbRef(BlockArgument arg, DbRefOp &dbRef) {
  dbRef = nullptr;
  for (Operation *user : arg.getUsers()) {
    if (isa<DbReleaseOp>(user))
      continue;
    auto ref = dyn_cast<DbRefOp>(user);
    if (!ref || dbRef)
      return false;
    if (ref.getIndices().empty() ||
        !llvm::all_of(ref.getIndices(), ValueAnalysis::isZeroConstant))
      return false;
    dbRef = ref;
  }
  return static_cast<bool>(dbRef);
}

static bool loadsAreInitializedByTask(DbRefOp dbRef, EdtOp edt,
                                      DominanceInfo &domInfo) {
  SmallVector<Operation *, 4> stores;
  SmallVector<Operation *, 4> loads;
  DbUtils::forEachReachableMemoryAccess(
      dbRef.getResult(),
      [&](const DbUtils::MemoryAccessInfo &access) {
        if (!DbAnalysis::isSameMemoryObject(access.memref, dbRef.getResult()))
          return WalkResult::interrupt();
        if (access.isWrite())
          stores.push_back(access.op);
        else
          loads.push_back(access.op);
        return WalkResult::advance();
      },
      &edt.getRegion());

  if (stores.empty())
    return false;

  for (Operation *load : loads) {
    bool covered = llvm::any_of(stores, [&](Operation *store) {
      return domInfo.dominates(store, load);
    });
    if (!covered)
      return false;
  }

  return true;
}

static std::optional<ScratchCandidate>
matchScratchCandidate(DbAllocOp alloc, DominanceInfo &domInfo) {
  if (!alloc || alloc.getAllocType() != DbAllocType::stack)
    return std::nullopt;
  if (!DbAnalysis::isCoarseGrained(alloc) || alloc.getSizes().size() != 1 ||
      !ValueAnalysis::isOneConstant(alloc.getSizes().front()))
    return std::nullopt;

  auto ptrType = dyn_cast<MemRefType>(alloc.getPtr().getType());
  if (!ptrType || !isa<MemRefType>(ptrType.getElementType()))
    return std::nullopt;

  ScratchCandidate candidate;
  candidate.alloc = alloc;
  llvm::SmallPtrSet<Operation *, 8> seenAcquireOps;

  auto classifyPtrUser = [&](Operation *user) -> bool {
    if (auto acquire = dyn_cast<DbAcquireOp>(user)) {
      if (acquire.getSourcePtr() != alloc.getPtr())
        return false;
      if (Value srcGuid = acquire.getSourceGuid();
          srcGuid && srcGuid != alloc.getGuid())
        return false;
      if (seenAcquireOps.insert(acquire.getOperation()).second)
        candidate.uses.push_back({acquire, nullptr, nullptr, nullptr});
      return true;
    }
    if (auto ref = dyn_cast<DbRefOp>(user)) {
      if (candidate.initRef || ref->getParentOfType<EdtOp>())
        return false;
      candidate.initRef = ref;
      return true;
    }
    if (auto free = dyn_cast<DbFreeOp>(user)) {
      if (!llvm::is_contained(candidate.freeOps, free))
        candidate.freeOps.push_back(free);
      return true;
    }
    return false;
  };

  auto classifyGuidUser = [&](Operation *user) -> bool {
    if (auto acquire = dyn_cast<DbAcquireOp>(user)) {
      if (acquire.getSourcePtr() != alloc.getPtr())
        return false;
      if (Value srcGuid = acquire.getSourceGuid();
          !srcGuid || srcGuid != alloc.getGuid())
        return false;
      if (seenAcquireOps.insert(acquire.getOperation()).second)
        candidate.uses.push_back({acquire, nullptr, nullptr, nullptr});
      return true;
    }
    if (auto free = dyn_cast<DbFreeOp>(user)) {
      if (!llvm::is_contained(candidate.freeOps, free))
        candidate.freeOps.push_back(free);
      return true;
    }
    return false;
  };

  for (Operation *user : alloc.getPtr().getUsers())
    if (!classifyPtrUser(user))
      return std::nullopt;
  for (Operation *user : alloc.getGuid().getUsers())
    if (!classifyGuidUser(user))
      return std::nullopt;

  if (candidate.uses.empty() ||
      !matchInitStore(candidate.initRef, candidate.initStore))
    return std::nullopt;

  for (ScratchUse &use : candidate.uses) {
    DbAcquireOp acquire = use.acquire;
    if (!acquire || acquire.getMode() == ArtsMode::in)
      return std::nullopt;
    if (acquire.getSizes().size() != 1 ||
        !ValueAnalysis::isOneConstant(acquire.getSizes().front()) ||
        acquire.getOffsets().size() != 1 ||
        !ValueAnalysis::isZeroConstant(acquire.getOffsets().front()))
      return std::nullopt;

    auto mode = getPartitionMode(acquire.getOperation());
    if (mode && requiresWorkerBoundsPlanning(*mode))
      return std::nullopt;

    auto [edt, blockArg] = EdtUtils::getBlockArgumentForAcquire(acquire);
    if (!edt || !blockArg || acquire.getPtr().use_empty() ||
        !acquire.getPtr().hasOneUse())
      return std::nullopt;

    DbRefOp dbRef = nullptr;
    if (!matchScratchDbRef(blockArg, dbRef))
      return std::nullopt;
    if (!edt.getBody().isAncestor(dbRef->getParentRegion()))
      return std::nullopt;
    auto refType = dyn_cast<MemRefType>(dbRef.getResult().getType());
    if (!refType)
      return std::nullopt;
    if (getDynamicAllocaSizes(alloc, refType).size() !=
        static_cast<size_t>(refType.getNumDynamicDims()))
      return std::nullopt;
    if (!loadsAreInitializedByTask(dbRef, edt, domInfo))
      return std::nullopt;

    use.edt = edt;
    use.blockArg = blockArg;
    use.dbRef = dbRef;
  }

  return candidate;
}

static void eraseIfPresent(Operation *op) {
  if (op)
    op->erase();
}

struct DbScratchEliminationPass
    : public impl::DbScratchEliminationBase<DbScratchEliminationPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<ScratchCandidate, 8> candidates;

    module.walk([&](func::FuncOp func) {
      SmallVector<DbAllocOp, 8> allocs;
      func.walk([&](DbAllocOp alloc) { allocs.push_back(alloc); });
      if (allocs.empty())
        return;
      DominanceInfo domInfo(func);
      for (DbAllocOp alloc : allocs) {
        if (auto candidate = matchScratchCandidate(alloc, domInfo))
          candidates.push_back(std::move(*candidate));
        else
          ++numScratchCandidatesRejected;
      }
    });

    if (candidates.empty())
      return;

    unsigned rewrites = 0;
    for (ScratchCandidate &candidate : candidates) {
      DenseMap<Operation *, SmallVector<unsigned, 4>> argsToRemove;
      DenseMap<Operation *, SmallVector<DbAcquireOp, 4>> acquiresToErase;

      for (ScratchUse &use : candidate.uses) {
        auto refType = cast<MemRefType>(use.dbRef.getResult().getType());
        SmallVector<Value> dynamicSizes =
            getDynamicAllocaSizes(candidate.alloc, refType);
        Block &entryBlock = use.edt.getBody().front();
        OpBuilder builder(&entryBlock, entryBlock.begin());
        auto local = memref::AllocaOp::create(builder, use.dbRef.getLoc(),
                                              refType, dynamicSizes);
        use.dbRef.getResult().replaceAllUsesWith(local.getMemref());
        use.dbRef.erase();

        SmallVector<Operation *, 4> releases;
        for (Operation *user : use.blockArg.getUsers())
          if (isa<DbReleaseOp>(user))
            releases.push_back(user);
        for (Operation *release : releases)
          release->erase();

        argsToRemove[use.edt.getOperation()].push_back(
            use.blockArg.getArgNumber());
        acquiresToErase[use.edt.getOperation()].push_back(use.acquire);
        ++numScratchUsesReplaced;
        rewrites++;
      }

      for (auto &entry : argsToRemove) {
        auto edt = cast<EdtOp>(entry.first);
        Block &body = edt.getBody().front();
        ValueRange deps = edt.getDependencies();
        SmallVector<unsigned, 4> indices = entry.second;
        llvm::sort(indices);
        indices.erase(std::unique(indices.begin(), indices.end()),
                      indices.end());

        SmallVector<Value> newDeps;
        for (unsigned i = 0; i < deps.size(); ++i)
          if (!std::binary_search(indices.begin(), indices.end(), i))
            newDeps.push_back(deps[i]);

        for (unsigned idx : llvm::reverse(indices))
          body.eraseArgument(idx);
        edt.setDependencies(newDeps);
      }

      for (auto &entry : acquiresToErase) {
        for (DbAcquireOp acquire : entry.second)
          if (acquire && acquire.getGuid().use_empty() &&
              acquire.getPtr().use_empty())
            acquire.erase();
      }

      eraseIfPresent(candidate.initStore);
      eraseIfPresent(candidate.initRef.getOperation());
      for (DbFreeOp freeOp : candidate.freeOps)
        eraseIfPresent(freeOp.getOperation());
      if (candidate.alloc && candidate.alloc.getGuid().use_empty() &&
          candidate.alloc.getPtr().use_empty()) {
        candidate.alloc.erase();
        ++numScratchDbsEliminated;
      }
    }

    ARTS_INFO("DbScratchEliminationPass: removed " << rewrites
                                                   << " scratch DB uses");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::carts::arts::createDbScratchEliminationPass() {
  return std::make_unique<DbScratchEliminationPass>();
}
