///==========================================================================///
/// File: EdtUtils.cpp
///
/// Implementation of utility functions for working with ARTS EDTs.
///==========================================================================///

#include "carts/dialect/arts/Utils/EdtUtils.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::arts;

namespace mlir {
namespace arts {

std::pair<EdtOp, BlockArgument>
EdtUtils::getBlockArgumentForAcquire(DbAcquireOp acquireOp) {
  /// Find the EDT that uses this acquire's pointer result
  EdtOp edtUser = nullptr;
  Value operandValue = nullptr;

  for (auto &use : acquireOp->getUses()) {
    Operation *userOp = use.getOwner();
    if (auto edtOp = dyn_cast<EdtOp>(userOp)) {
      edtUser = edtOp;
      operandValue = edtOp->getOperand(use.getOperandNumber());
      break;
    }
  }

  if (!edtUser || !operandValue)
    return {nullptr, nullptr};

  /// The index within dependencies equals the block argument index.
  ValueRange deps = edtUser.getDependencies();
  auto depIt = std::find(deps.begin(), deps.end(), operandValue);
  if (depIt == deps.end())
    return {nullptr, nullptr};

  unsigned blockArgIdx = std::distance(deps.begin(), depIt);

  /// Get the block argument.
  Block &body = edtUser.getRegion().front();
  if (blockArgIdx >= body.getNumArguments())
    return {nullptr, nullptr};

  BlockArgument blockArg = body.getArgument(blockArgIdx);
  return {edtUser, blockArg};
}

EpochOp EdtUtils::wrapBodyInEpoch(Block &body, Location loc) {
  SmallVector<Operation *, 8> opsToMove;
  for (Operation &op : body.without_terminator())
    opsToMove.push_back(&op);

  if (opsToMove.empty())
    return nullptr;

  OpBuilder builder(body.getTerminator());
  auto epochOp = EpochOp::create(builder, loc);
  auto &epochBlock = epochOp.getRegion().emplaceBlock();

  for (Operation *op : opsToMove)
    op->moveBefore(&epochBlock, epochBlock.end());

  builder.setInsertionPointToEnd(&epochBlock);
  YieldOp::create(builder, loc);

  return epochOp;
}

std::optional<unsigned> EdtUtils::mapMemrefToArg(EdtOp edt,
                                                 Value memrefValue) {
  if (!memrefValue)
    return std::nullopt;
  Value current = ValueAnalysis::stripMemrefViewOps(memrefValue);
  if (auto dbRef = current.getDefiningOp<DbRefOp>())
    current = ValueAnalysis::stripMemrefViewOps(dbRef.getSource());

  auto blockArg = dyn_cast<BlockArgument>(current);
  if (!blockArg)
    return std::nullopt;
  Block *edtBody = &edt.getBody().front();
  if (blockArg.getOwner() != edtBody)
    return std::nullopt;
  return blockArg.getArgNumber();
}

void EdtUtils::classifyArgAccesses(EdtOp edt, SmallVectorImpl<bool> &reads,
                                   SmallVectorImpl<bool> &writes) {
  reads.assign(edt.getDependencies().size(), false);
  writes.assign(edt.getDependencies().size(), false);

  auto markRead = [&](Value memrefValue) {
    auto argIdx = EdtUtils::mapMemrefToArg(edt, memrefValue);
    if (argIdx && *argIdx < reads.size())
      reads[*argIdx] = true;
  };
  auto markWrite = [&](Value memrefValue) {
    auto argIdx = EdtUtils::mapMemrefToArg(edt, memrefValue);
    if (argIdx && *argIdx < writes.size())
      writes[*argIdx] = true;
  };

  edt.walk([&](Operation *nested) {
    if (auto load = dyn_cast<memref::LoadOp>(nested))
      markRead(load.getMemRef());
    else if (auto store = dyn_cast<memref::StoreOp>(nested))
      markWrite(store.getMemRef());
    else if (auto affineLoad = dyn_cast<affine::AffineLoadOp>(nested))
      markRead(affineLoad.getMemRef());
    else if (auto affineStore = dyn_cast<affine::AffineStoreOp>(nested))
      markWrite(affineStore.getMemRef());
  });
}

namespace {
static bool isCloneSafeStoreOperand(Value value, Value memref,
                                    llvm::DenseSet<Operation *> &visited) {
  if (!value || value == memref)
    return true;

  if (isa<BlockArgument>(value))
    return true;

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return true;

  if (defOp->hasTrait<OpTrait::ConstantLike>())
    return true;

  if (defOp->getNumRegions() != 0)
    return false;

  bool hasSideEffects = false;
  if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(defOp)) {
    hasSideEffects = memEffects.hasEffect<MemoryEffects::Write>() ||
                     memEffects.hasEffect<MemoryEffects::Allocate>() ||
                     memEffects.hasEffect<MemoryEffects::Free>();
  } else {
    hasSideEffects = !isMemoryEffectFree(defOp);
  }
  if (hasSideEffects)
    return false;

  if (!visited.insert(defOp).second)
    return true;

  return llvm::all_of(defOp->getOperands(), [&](Value operand) {
    return isCloneSafeStoreOperand(operand, memref, visited);
  });
}

} // namespace

bool EdtUtils::canCloneAllocaInitStore(memref::StoreOp store, Value memref) {
  if (!store || store.getMemRef() != memref)
    return false;

  llvm::DenseSet<Operation *> visited;
  if (!isCloneSafeStoreOperand(store.getValue(), memref, visited))
    return false;

  return llvm::all_of(store.getIndices(), [&](Value index) {
    return isCloneSafeStoreOperand(index, memref, visited);
  });
}

Value EdtUtils::traceCapturedDbHandle(Value value) {
  DenseSet<Value> visited;
  for (unsigned depth = 0; value && depth < 16; ++depth) {
    if (!visited.insert(value).second)
      break;

    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return Value();

    if (isa<DbAllocOp, DbAcquireOp, memref::AllocOp>(defOp))
      return value;

    if (auto dbRef = dyn_cast<DbRefOp>(defOp)) {
      value = dbRef.getSource();
      continue;
    }
    if (auto dbGep = dyn_cast<DbGepOp>(defOp)) {
      value = dbGep.getBasePtr();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
      value = cast.getSource();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(defOp)) {
      value = subview.getSource();
      continue;
    }
    if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
      if (unrealized.getInputs().size() == 1) {
        value = unrealized.getInputs().front();
        continue;
      }
    }

    break;
  }
  return Value();
}

void EdtUtils::classifyUserValues(ArrayRef<Value> userValues,
                                  llvm::SetVector<Value> &parameters,
                                  llvm::SetVector<Value> &constants,
                                  llvm::SetVector<Value> &dbHandles) {
  for (Value val : userValues) {
    if (Value dbHandle = EdtUtils::traceCapturedDbHandle(val)) {
      dbHandles.insert(dbHandle);
      continue;
    }

    if (auto *defOp = val.getDefiningOp()) {
      if (isa<arith::ConstantOp>(defOp)) {
        constants.insert(val);
        continue;
      }
    }

    /// Direct scalar captures are not implicit parameters. Every scalar that
    /// crosses an EDT boundary must be listed on `arts.edt params(...)` and
    /// used through the corresponding block argument.
    if (val.getType().isIntOrIndexOrFloat())
      continue;

    // Stack allocas for loop-local scratch remain clonable. Heap memref.alloc
    // handles are classified by traceCapturedDbHandle above.
  }
}

void EdtUtils::analyzeCapturedValues(
    EdtOp edt, llvm::SetVector<Value> &capturedValues,
    llvm::SetVector<Value> &parameters, llvm::SetVector<Value> &constants,
    llvm::SetVector<Value> &dbHandles) {
  if (!edt)
    return;

  getUsedValuesDefinedAbove(edt.getRegion(), capturedValues);
  /// RegionUtils also reports values defined in the EDT body when they are
  /// referenced from nested regions inside the EDT. Those are not true
  /// captures for outlining: they must remain local to the outlined function.
  auto isDefinedInsideEdt = [&](Value value) {
    if (Operation *defOp = value.getDefiningOp())
      return edt.getOperation()->isAncestor(defOp);
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      if (Operation *parentOp = blockArg.getOwner()->getParentOp())
        return edt.getOperation()->isAncestor(parentOp);
    }
    return false;
  };
  llvm::SetVector<Value> externalCaptures;
  for (Value value : capturedValues)
    if (!isDefinedInsideEdt(value))
      externalCaptures.insert(value);
  capturedValues = std::move(externalCaptures);
  EdtUtils::classifyUserValues(capturedValues.getArrayRef(), parameters,
                               constants, dbHandles);
}

SmallVector<Value> EdtUtils::collectPackedValues(EdtOp edt) {
  llvm::SetVector<Value> capturedValues;
  llvm::SetVector<Value> uniqueParameters;
  llvm::SetVector<Value> constants;
  llvm::SetVector<Value> dbHandles;
  SmallVector<Value> parameters;
  for (Value param : edt.getParams()) {
    parameters.push_back(param);
    uniqueParameters.insert(param);
  }
  EdtUtils::analyzeCapturedValues(edt, capturedValues, uniqueParameters,
                                  constants, dbHandles);

  SmallVector<Value> packedValues;
  packedValues.reserve(parameters.size());
  DenseMap<Value, unsigned> valueToPackIndex;

  for (Value parameter : parameters) {
    if (auto *defOp = parameter.getDefiningOp())
      if (isUndefLikeOp(defOp))
        continue;
    valueToPackIndex.try_emplace(parameter, packedValues.size());
    packedValues.push_back(parameter);
  }

  auto appendIfMissing = [&](Value val) {
    if (!val)
      return;
    if (val.getDefiningOp<arith::ConstantOp>())
      return;
    if (valueToPackIndex.count(val))
      return;
    valueToPackIndex[val] = packedValues.size();
    packedValues.push_back(val);
  };

  for (Value dep : edt.getDependencies()) {
    auto dbAcquireOp = dep.getDefiningOp<DbAcquireOp>();
    if (!dbAcquireOp)
      continue;

    for (Value idx : dbAcquireOp.getIndices())
      appendIfMissing(idx);
    for (Value off : dbAcquireOp.getOffsets())
      appendIfMissing(off);
    for (Value sz : dbAcquireOp.getSizes())
      appendIfMissing(sz);
    for (Value partIdx : dbAcquireOp.getPartitionIndices())
      appendIfMissing(partIdx);
    for (Value partOff : dbAcquireOp.getPartitionOffsets())
      appendIfMissing(partOff);
    for (Value partSize : dbAcquireOp.getPartitionSizes())
      appendIfMissing(partSize);

    if (auto *rawAlloc =
            DbUtils::getUnderlyingDbAlloc(dbAcquireOp.getSourcePtr()))
      if (auto alloc = dyn_cast<DbAllocOp>(rawAlloc))
        for (Value elemSz : alloc.getElementSizes())
          appendIfMissing(elemSz);
  }

  return packedValues;
}

} // namespace arts
} // namespace mlir
