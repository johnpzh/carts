///==========================================================================///
/// File: MemoryAccessClassifier.cpp
///
/// Implementation of MemoryAccessClassifier -- memory access pattern analysis
/// extracted from DbAcquireNode.
///==========================================================================///

#include "carts/dialect/arts/Analysis/graphs/db/MemoryAccessClassifier.h"
#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(memory_access_classifier);

///===----------------------------------------------------------------------===///
/// Indirect index detection helpers
///===----------------------------------------------------------------------===///

/// Detect data-dependent loop bounds for a given induction variable.
static bool loopBoundsAreIndirect(BlockArgument iv, Value partitionOffset,
                                  int depth) {
  if (depth > 8)
    return false;

  Operation *parentOp = iv.getOwner()->getParentOp();
  if (!parentOp)
    return false;

  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
    if (iv != forOp.getInductionVar())
      return false;
    if (arts::isIndirectIndex(forOp.getLowerBound(), partitionOffset,
                              depth + 1) ||
        arts::isIndirectIndex(forOp.getUpperBound(), partitionOffset,
                              depth + 1) ||
        arts::isIndirectIndex(forOp.getStep(), partitionOffset, depth + 1))
      return true;
    return false;
  }

  if (auto parOp = dyn_cast<scf::ParallelOp>(parentOp)) {
    auto ivs = parOp.getInductionVars();
    for (auto it : llvm::enumerate(ivs)) {
      if (it.value() != iv)
        continue;
      unsigned idx = it.index();
      if (arts::isIndirectIndex(parOp.getLowerBound()[idx], partitionOffset,
                                depth + 1) ||
          arts::isIndirectIndex(parOp.getUpperBound()[idx], partitionOffset,
                                depth + 1) ||
          arts::isIndirectIndex(parOp.getStep()[idx], partitionOffset,
                                depth + 1))
        return true;
      return false;
    }
  }

  return false;
}

/// Detect indirect (data-dependent) indices derived from memory loads.
/// This is the namespace-level function declared in MemoryAccessClassifier.h.
bool arts::isIndirectIndex(Value idx, Value partitionOffset, int depth) {
  if (!idx || depth > 8)
    return false;

  idx = ValueAnalysis::stripNumericCasts(idx);
  if (ValueAnalysis::isValueConstant(idx))
    return false;

  if (arts::isArtsRuntimeQuery(idx))
    return false;

  if (partitionOffset) {
    Value offsetStripped = ValueAnalysis::stripNumericCasts(partitionOffset);
    if (idx == offsetStripped)
      return false;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(idx)) {
    if (partitionOffset) {
      Value offsetStripped = ValueAnalysis::stripNumericCasts(partitionOffset);
      if (offsetStripped == blockArg ||
          ValueAnalysis::dependsOn(offsetStripped, blockArg))
        return false;
    }

    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<affine::AffineForOp>(parentOp)) {
      if (blockArg == forOp.getInductionVar())
        return loopBoundsAreIndirect(blockArg, partitionOffset, depth + 1);
    }
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      if (blockArg == forOp.getInductionVar())
        return loopBoundsAreIndirect(blockArg, partitionOffset, depth + 1);
    }
    if (auto parOp = dyn_cast<scf::ParallelOp>(parentOp)) {
      for (Value iv : parOp.getInductionVars()) {
        if (blockArg == iv)
          return loopBoundsAreIndirect(blockArg, partitionOffset, depth + 1);
      }
    }
    return true;
  }

  Operation *defOp = idx.getDefiningOp();
  if (!defOp)
    return false;

  if (auto load = dyn_cast<memref::LoadOp>(defOp)) {
    bool isDbLoad = DbUtils::getUnderlyingDb(load.getMemref()) != nullptr;
    bool hasDynamicIndex = llvm::any_of(load.getIndices(), [](Value idx) {
      int64_t constVal = 0;
      return !ValueAnalysis::getConstantIndex(idx, constVal);
    });
    return isDbLoad || hasDynamicIndex;
  }
  if (auto load = dyn_cast<LLVM::LoadOp>(defOp))
    return DbUtils::getUnderlyingDb(load.getAddr()) != nullptr;

  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
          arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::IndexCastOp,
          arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
          arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp,
          arith::SelectOp, arith::CmpIOp, arith::CmpFOp, affine::AffineApplyOp>(
          defOp)) {
    for (Value operand : defOp->getOperands()) {
      if (arts::isIndirectIndex(operand, partitionOffset, depth + 1))
        return true;
    }
    return false;
  }

  return true;
}

///===----------------------------------------------------------------------===///
/// MemoryAccessClassifier implementation
///===----------------------------------------------------------------------===///

void MemoryAccessClassifier::collectAccessOperations(
    DbAcquireNode *node,
    DenseMap<DbRefOp, SetVector<Operation *>> &dbRefToMemOps) {
  if (!node)
    return;

  Operation *edtUserOp =
      node->getEdtUser() ? node->getEdtUser().getOperation() : nullptr;
  Value useInEdt = node->getUseInEdt();
  if (!edtUserOp || !useInEdt)
    return;

  if (!edtUserOp->getParentRegion())
    return;

  EdtOp edtUser = node->getEdtUser();

  SmallVector<Value, 16> worklist;
  SetVector<Value> visited;
  worklist.push_back(useInEdt);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current))
      continue;

    for (Operation *user : current.getUsers()) {
      Region *userRegion = user->getParentRegion();
      if (!userRegion || !edtUser.getBody().isAncestor(userRegion))
        continue;

      if (auto dbRef = dyn_cast<DbRefOp>(user)) {
        dbRefToMemOps.try_emplace(dbRef);

        Value refResult = dbRef.getResult();
        worklist.push_back(refResult);
        SetVector<Operation *> memOps;
        DbUtils::collectReachableMemoryOps(refResult, memOps,
                                           &edtUser.getBody());
        for (Operation *memOp : memOps)
          dbRefToMemOps[dbRef].insert(memOp);
      }
    }
  }
}

void MemoryAccessClassifier::forEachMemoryAccess(
    DbAcquireNode *node, llvm::function_ref<void(Operation *, bool)> callback) {
  DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
  collectAccessOperations(node, dbRefToMemOps);
  for (auto &[dbRef, memOps] : dbRefToMemOps) {
    for (Operation *op : memOps) {
      auto access = DbUtils::getMemoryAccessInfo(op);
      if (!access)
        continue;
      callback(op, access->isWrite());
    }
  }
}

bool MemoryAccessClassifier::hasLoads(DbAcquireNode *node) {
  bool found = false;
  forEachMemoryAccess(node, [&](Operation *, bool isStore) {
    if (!isStore)
      found = true;
  });
  return found;
}

bool MemoryAccessClassifier::hasStores(DbAcquireNode *node) {
  bool found = false;
  forEachMemoryAccess(node, [&](Operation *, bool isStore) {
    if (isStore)
      found = true;
  });
  return found;
}

bool MemoryAccessClassifier::isReadOnlyAccess(DbAcquireNode *node) {
  bool sawLoad = false;
  bool sawStore = false;
  forEachMemoryAccess(node, [&](Operation *, bool isStore) {
    if (isStore)
      sawStore = true;
    else
      sawLoad = true;
  });

  if (sawStore)
    return false;
  if (sawLoad)
    return true;

  DbAcquireOp acqOp = node->getDbAcquireOp();
  return acqOp && acqOp.getMode() == ArtsMode::in;
}

bool MemoryAccessClassifier::isWriterAccess(DbAcquireNode *node) {
  bool sawStore = false;
  forEachMemoryAccess(node, [&](Operation *, bool isStore) {
    if (isStore)
      sawStore = true;
  });
  if (sawStore)
    return true;

  DbAcquireOp acqOp = node->getDbAcquireOp();
  return acqOp && DbUtils::isWriterMode(acqOp.getMode());
}

bool MemoryAccessClassifier::hasMemoryAccesses(DbAcquireNode *node) {
  bool found = false;
  forEachMemoryAccess(node, [&](Operation *, bool) { found = true; });
  return found;
}

size_t MemoryAccessClassifier::countLoads(DbAcquireNode *node) {
  size_t count = 0;
  forEachMemoryAccess(node, [&](Operation *, bool isStore) {
    if (!isStore)
      ++count;
  });
  return count;
}

size_t MemoryAccessClassifier::countStores(DbAcquireNode *node) {
  size_t count = 0;
  forEachMemoryAccess(node, [&](Operation *, bool isStore) {
    if (isStore)
      ++count;
  });
  return count;
}

bool MemoryAccessClassifier::hasIndirectAccess(DbAcquireNode *node) {
  DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
  collectAccessOperations(node, dbRefToMemOps);
  auto [partOffset, partSize] = node->getPartitionInfo();

  for (auto &[dbRef, memOps] : dbRefToMemOps) {
    for (Operation *memOp : memOps) {
      SmallVector<Value> fullChain =
          DbUtils::collectFullIndexChain(dbRef, memOp);
      for (Value idx : fullChain) {
        int64_t constVal;
        if (ValueAnalysis::getConstantIndex(idx, constVal))
          continue;
        if (arts::isIndirectIndex(idx, partOffset))
          return true;
      }
    }
  }

  return false;
}

bool MemoryAccessClassifier::hasDirectAccess(DbAcquireNode *node) {
  DenseMap<DbRefOp, SetVector<Operation *>> dbRefToMemOps;
  collectAccessOperations(node, dbRefToMemOps);
  auto [partOffset, partSize] = node->getPartitionInfo();

  for (auto &[dbRef, memOps] : dbRefToMemOps) {
    for (Operation *memOp : memOps) {
      SmallVector<Value> fullChain =
          DbUtils::collectFullIndexChain(dbRef, memOp);
      if (fullChain.empty())
        return true;

      bool hasIndirect = false;
      for (Value idx : fullChain) {
        int64_t constVal;
        if (ValueAnalysis::getConstantIndex(idx, constVal))
          continue;
        if (arts::isIndirectIndex(idx, partOffset)) {
          hasIndirect = true;
          break;
        }
      }

      if (!hasIndirect)
        return true;
    }
  }

  return false;
}
