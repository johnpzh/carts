#include "arts/dialect/sde/Transforms/CodeletUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::carts::sde {
namespace {

static bool isDirectCaptureType(Type type) {
  return type.isIntOrIndexOrFloat();
}

static bool isInsideMovedOps(Operation *op,
                             const DenseSet<Operation *> &movedOps) {
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (movedOps.contains(cur))
      return true;
  return false;
}

static bool canClonePureValue(Value value,
                              const DenseSet<Value> &preMappedValues,
                              const DenseSet<Value> &materializableMemrefs,
                              const DenseSet<Type> &materializableMemrefTypes,
                              const DenseSet<Operation *> &movedOps,
                              DenseSet<Value> &visiting,
                              DenseSet<Value> &memo) {
  auto memrefTy = dyn_cast<MemRefType>(value.getType());
  if (preMappedValues.contains(value) ||
      materializableMemrefs.contains(value) ||
      (memrefTy && materializableMemrefTypes.contains(memrefTy)) ||
      isValueInternalToMovedOps(value, movedOps))
    return true;
  if (memo.contains(value))
    return true;
  if (!visiting.insert(value).second)
    return true;

  Operation *defOp = value.getDefiningOp();
  if (!defOp || !isMemoryEffectFree(defOp))
    return false;

  for (Value operand : defOp->getOperands()) {
    if (!canClonePureValue(operand, preMappedValues, materializableMemrefs,
                           materializableMemrefTypes, movedOps, visiting, memo))
      return false;
  }

  visiting.erase(value);
  memo.insert(value);
  return true;
}

static void addDirectCapture(Value value, ExternalCapturePlan &plan) {
  if (plan.captureSet.insert(value).second)
    plan.captures.push_back(value);
}

static void addPureExternal(Value value, ExternalCapturePlan &plan) {
  if (plan.pureExternalSet.insert(value).second)
    plan.pureExternal.push_back(value);
}

static bool planExternalValue(Value value,
                              const DenseSet<Value> &preMappedValues,
                              const DenseSet<Value> &materializableMemrefs,
                              const DenseSet<Type> &materializableMemrefTypes,
                              const DenseSet<Operation *> &movedOps,
                              ExternalCapturePlan &plan) {
  if (preMappedValues.contains(value) ||
      isValueInternalToMovedOps(value, movedOps))
    return true;

  auto memrefTy = dyn_cast<MemRefType>(value.getType());
  if (materializableMemrefs.contains(value) ||
      (memrefTy && materializableMemrefTypes.contains(memrefTy))) {
    plan.materializedMemrefs.insert(value);
    return true;
  }

  Operation *defOp = value.getDefiningOp();
  if (defOp && isMemoryEffectFree(defOp)) {
    DenseSet<Value> visiting;
    DenseSet<Value> memo;
    if (canClonePureValue(value, preMappedValues, materializableMemrefs,
                          materializableMemrefTypes, movedOps, visiting,
                          memo)) {
      addPureExternal(value, plan);
      return true;
    }
  }

  if (isDirectCaptureType(value.getType())) {
    addDirectCapture(value, plan);
    return true;
  }

  return false;
}

} // namespace

bool hasTensorIterArgs(SdeCuRegionOp region) {
  for (Value arg : region.getIterArgs())
    if (isa<RankedTensorType>(arg.getType()))
      return true;
  return false;
}

SdeAccessMode classifyTensorAccess(BlockArgument blockArg) {
  bool hasRead = false;
  bool hasWrite = false;
  for (Operation *user : blockArg.getUsers()) {
    if (isa<tensor::ExtractOp>(user)) {
      hasRead = true;
    } else if (auto insert = dyn_cast<tensor::InsertOp>(user)) {
      if (insert.getDest() == blockArg)
        hasWrite = true;
      else
        hasRead = true;
    } else if (isa<scf::YieldOp, SdeYieldOp>(user)) {
      continue;
    } else {
      // Unknown tensor user: assume both so we preserve correctness.
      hasRead = true;
      hasWrite = true;
    }
  }
  if (hasRead && hasWrite)
    return SdeAccessMode::readwrite;
  if (hasWrite)
    return SdeAccessMode::write;
  return SdeAccessMode::read;
}

Value traceTensorBoundaryMemref(Value tensor) {
  if (auto muToTensor = tensor.getDefiningOp<SdeMuMemrefToTensorOp>())
    return muToTensor.getMemref();
  if (auto toTensor = tensor.getDefiningOp<bufferization::ToTensorOp>())
    return toTensor.getBuffer();
  if (auto blockArg = dyn_cast<BlockArgument>(tensor)) {
    auto parentRegion = blockArg.getOwner()->getParent();
    if (!parentRegion)
      return Value();
    auto parentCuRegion =
        dyn_cast_or_null<SdeCuRegionOp>(parentRegion->getParentOp());
    if (!parentCuRegion)
      return Value();
    unsigned idx = blockArg.getArgNumber();
    if (idx >= parentCuRegion.getIterArgs().size())
      return Value();
    return traceTensorBoundaryMemref(parentCuRegion.getIterArgs()[idx]);
  }
  return Value();
}

DenseSet<Operation *> collectMovedOpTree(ArrayRef<Operation *> opsToMove) {
  DenseSet<Operation *> movedOps;
  for (Operation *root : opsToMove) {
    movedOps.insert(root);
    root->walk([&](Operation *op) { movedOps.insert(op); });
  }
  return movedOps;
}

bool isValueInternalToMovedOps(Value value,
                               const DenseSet<Operation *> &movedOps) {
  if (Operation *defOp = value.getDefiningOp())
    return isInsideMovedOps(defOp, movedOps);

  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return false;
  if (Operation *parentOp = blockArg.getOwner()->getParentOp())
    return isInsideMovedOps(parentOp, movedOps);
  return false;
}

bool hasMemrefWriteInMovedOps(Value memref, ArrayRef<Operation *> opsToMove) {
  for (Operation *root : opsToMove) {
    WalkResult result = root->walk([&](Operation *op) -> WalkResult {
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (store.getMemref() == memref)
          return WalkResult::interrupt();
      } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
        if (store.getMemref() == memref)
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return true;
  }
  return false;
}

LogicalResult planExternalCaptures(
    ArrayRef<Operation *> opsToMove, const DenseSet<Value> &preMappedValues,
    const DenseSet<Value> &materializableMemrefs,
    const DenseSet<Type> &materializableMemrefTypes,
    const DenseSet<Operation *> &movedOps, ExternalCapturePlan &plan) {
  DenseSet<Value> seenExternalValues;
  for (Operation *root : opsToMove) {
    WalkResult walkResult = root->walk([&](Operation *nested) -> WalkResult {
      for (Value operand : nested->getOperands()) {
        if (preMappedValues.contains(operand) ||
            isValueInternalToMovedOps(operand, movedOps))
          continue;
        if (!seenExternalValues.insert(operand).second)
          continue;
        if (!planExternalValue(operand, preMappedValues, materializableMemrefs,
                               materializableMemrefTypes, movedOps, plan)) {
          nested->emitError("cannot capture external value for cu_codelet: ")
              << operand << " : " << operand.getType();
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();
  }
  return success();
}

} // namespace mlir::carts::sde
