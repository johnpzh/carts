///==========================================================================///
/// File: RaiseToTensor.cpp
///
/// Function-scope tensor raising (tensor mem2reg):
///   Walk each function body top-to-bottom, maintaining a
///   DenseMap<Value, Value> currentTensor mapping each raised alloca to its
///   "current tensor value."
///
///   - memref.store → tensor.insert, advances currentTensor[alloca]
///   - memref.load → tensor.extract from currentTensor[alloca]
///   - Region-holding ops (scf.for, cu_region, su_iterate) → thread current
///     tensors as iter_args, recurse into body, update currentTensor to result
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_RAISETOTENSOR
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/OperationAttributes.h"
#include "arts/utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "polygeist/Ops.h"

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(raise_to_tensor);

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::carts;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static Value stripMemrefCasts(Value v) {
  while (auto castOp = v.getDefiningOp<memref::CastOp>())
    v = castOp.getSource();
  return v;
}

/// Get the tensor type corresponding to a memref type.
static RankedTensorType getTensorType(MemRefType mrType) {
  return RankedTensorType::get(mrType.getShape(), mrType.getElementType());
}

/// Check that a subview-like value is used only for dependency metadata. Actual
/// loads/stores through a view need index composition support before they can
/// be raised safely.
static bool hasOnlyDependencyViewUsers(Value memref,
                                       Operation **blocker = nullptr) {
  for (Operation *user : memref.getUsers()) {
    if (isa<sde::SdeMuDepOp>(user))
      continue;
    if (auto castOp = dyn_cast<memref::CastOp>(user)) {
      if (!hasOnlyDependencyViewUsers(castOp.getResult(), blocker))
        return false;
      continue;
    }
    if (auto subIndexOp = dyn_cast<polygeist::SubIndexOp>(user)) {
      if (!hasOnlyDependencyViewUsers(subIndexOp.getResult(), blocker))
        return false;
      continue;
    }
    if (blocker)
      *blocker = user;
    return false;
  }
  return true;
}

/// Check if a memref value (or any of its casts) is used as an operand to
/// an op that is NOT load/store/cast/mu_dep/dependency-only subindex. This
/// catches calls, su_iterate reductionAccumulators, and any other non-raisable
/// usage.
/// If \p blocker is non-null, the first blocking operation is stored there.
static bool hasNonRaisableUser(Value memref, Operation **blocker = nullptr) {
  for (Operation *user : memref.getUsers()) {
    if (isa<memref::LoadOp, memref::StoreOp>(user))
      continue;
    if (isa<sde::SdeMuDepOp>(user))
      continue;
    // bufferization.to_tensor is a "read all" — rewritten during raising.
    if (isa<bufferization::ToTensorOp>(user))
      continue;
    if (auto castOp = dyn_cast<memref::CastOp>(user)) {
      if (hasNonRaisableUser(castOp.getResult(), blocker))
        return true;
      continue;
    }
    if (auto subIndexOp = dyn_cast<polygeist::SubIndexOp>(user)) {
      if (!hasOnlyDependencyViewUsers(subIndexOp.getResult(), blocker))
        return true;
      continue;
    }
    if (isa<polygeist::Memref2PointerOp>(user))
      continue;
    // Any other user (calls, su_iterate operands, etc.) disqualifies.
    if (blocker)
      *blocker = user;
    return true;
  }
  return false;
}

/// Return true if the operation is directly inside a supported region-holding
/// op (scf.for, cu_region, su_iterate, scf.if, scf.execute_region) or in the
/// function entry block — NOT inside scf.while, etc. that we can't thread
/// tensors through.
/// scf.execute_region is treated as transparent: it always executes, has a
/// single entry/exit, and tensor updates inside it are visible after it.
/// scf.if is supported by threading the current tensor through both branches
/// and selecting the branch result.
static bool isInSupportedParent(Operation *op, func::FuncOp func) {
  Operation *parent = op->getParentOp();
  while (parent && parent != func) {
    if (isa<scf::ForOp, sde::SdeCuRegionOp, sde::SdeSuIterateOp>(parent))
      return true;
    if (isa<func::FuncOp>(parent))
      return true;
    // scf.execute_region and scf.if are transparent for the parent check:
    // walkBlock has explicit lowering support for both.
    if (isa<scf::ExecuteRegionOp, scf::IfOp>(parent)) {
      parent = parent->getParentOp();
      continue;
    }
    // scf.while, etc. — unsupported
    if (isa<scf::WhileOp>(parent))
      return false;
    parent = parent->getParentOp();
  }
  return parent == func.getOperation();
}

static bool hasMemref2PointerUser(Value memref) {
  for (Operation *user : memref.getUsers()) {
    if (isa<polygeist::Memref2PointerOp>(user))
      return true;
    if (auto castOp = dyn_cast<memref::CastOp>(user)) {
      if (hasMemref2PointerUser(castOp.getResult()))
        return true;
    }
  }
  return false;
}

static bool hasStoreInsideSdeRegion(Value memref) {
  for (Operation *user : memref.getUsers()) {
    if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
      if (storeOp->getParentOfType<sde::SdeCuRegionOp>() ||
          storeOp->getParentOfType<sde::SdeSuIterateOp>())
        return true;
      continue;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(user)) {
      if (hasStoreInsideSdeRegion(castOp.getResult()))
        return true;
    }
  }
  return false;
}

/// Check if a memref.alloca is raisable: only used by load/store/cast/mu_dep,
/// no non-raisable users (calls, su_iterate accumulators, etc.), and all
/// load/store accesses are in supported regions.
static bool isRaisableMemref(Value memref, func::FuncOp func) {
  auto mrType = dyn_cast<MemRefType>(memref.getType());
  if (!mrType)
    return false;
  if (!TensorType::isValidElementType(mrType.getElementType()))
    return false;

  // Reject if any user (through casts) is not load/store/cast/mu_dep.
  if (hasNonRaisableUser(memref))
    return false;

  // Pointer views are tolerated only when SDE cannot mutate the backing memref.
  // Otherwise the tensor value and the original stack buffer may diverge.
  if (hasMemref2PointerUser(memref) && hasStoreInsideSdeRegion(memref))
    return false;

  // All load/store users must be in supported parent regions.
  for (Operation *user : memref.getUsers()) {
    if (isa<memref::LoadOp, memref::StoreOp>(user)) {
      if (!isInSupportedParent(user, func))
        return false;
    }
  }

  return true;
}

/// Try to resolve the canonical alloca for a memref value used in a
/// load/store. Returns null if not a raisable alloca.
static memref::AllocaOp resolveAlloca(Value memref,
                                      func::FuncOp func = nullptr) {
  Value canonical = stripMemrefCasts(memref);
  auto alloca = canonical.getDefiningOp<memref::AllocaOp>();
  if (!alloca)
    return nullptr;
  if (func && !isRaisableMemref(alloca.getResult(), func))
    return nullptr;
  if (!func) {
    // Lightweight check without func context.
    auto mrType = dyn_cast<MemRefType>(alloca.getType());
    if (!mrType)
      return nullptr;
  }
  return alloca;
}

//===----------------------------------------------------------------------===//
// Step 1: Function-scope alloca collection
//===----------------------------------------------------------------------===//

/// Return true if the alloca is accessed inside at least one SDE region
/// (cu_region or su_iterate). Allocas only accessed outside SDE regions
/// don't benefit from tensor raising.
static bool isAccessedInsideSdeRegion(memref::AllocaOp alloca) {
  for (Operation *user : alloca.getResult().getUsers()) {
    // Follow casts.
    if (auto castOp = dyn_cast<memref::CastOp>(user)) {
      for (Operation *castUser : castOp.getResult().getUsers()) {
        if (castUser->getParentOfType<sde::SdeCuRegionOp>() ||
            castUser->getParentOfType<sde::SdeSuIterateOp>())
          return true;
      }
      continue;
    }
    if (user->getParentOfType<sde::SdeCuRegionOp>() ||
        user->getParentOfType<sde::SdeSuIterateOp>())
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Verification diagnostics
//===----------------------------------------------------------------------===//

/// Return true if the alloca (or a cast of it) is used as an su_iterate
/// reduction accumulator. These allocas are managed by the su_iterate
/// reduction mechanism and do NOT need tensor raising.
static bool isManagedByReduction(memref::AllocaOp alloca) {
  auto isAccumulator = [](Value v) -> bool {
    for (Operation *user : v.getUsers()) {
      auto suIter = dyn_cast<sde::SdeSuIterateOp>(user);
      if (!suIter)
        continue;
      for (Value acc : suIter.getReductionAccumulators()) {
        if (acc == v)
          return true;
      }
    }
    return false;
  };

  if (isAccumulator(alloca.getResult()))
    return true;
  for (Operation *user : alloca.getResult().getUsers()) {
    if (auto castOp = dyn_cast<memref::CastOp>(user)) {
      if (isAccumulator(castOp.getResult()))
        return true;
    }
  }
  return false;
}

/// Find the first load/store user inside a cu_region that is nested inside
/// an unsupported control-flow op (scf.if, scf.while, etc.).
static Operation *findUnsupportedParentAccess(memref::AllocaOp alloca,
                                              func::FuncOp func) {
  for (Operation *user : alloca.getResult().getUsers()) {
    if (isa<memref::LoadOp, memref::StoreOp>(user)) {
      if (!isInSupportedParent(user, func))
        return user;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(user)) {
      for (Operation *castUser : castOp.getResult().getUsers()) {
        if (isa<memref::LoadOp, memref::StoreOp>(castUser)) {
          if (!isInSupportedParent(castUser, func))
            return castUser;
        }
      }
    }
  }
  return nullptr;
}

/// Emit a diagnostic explaining why a captured alloca was not raised.
static void diagnoseCapturedAlloca(memref::AllocaOp alloca,
                                   sde::SdeCuRegionOp cuRegion,
                                   func::FuncOp func) {
  auto diag = alloca.emitError()
              << "memref.alloca of type " << alloca.getType()
              << " captured by cu_region but was not raised to tensor";
  diag.attachNote(cuRegion.getLoc()) << "cu_region capturing this alloca";

  // Diagnose specific reason.
  Operation *blocker = nullptr;
  if (hasNonRaisableUser(alloca.getResult(), &blocker) && blocker) {
    diag.attachNote(blocker->getLoc())
        << "blocked by non-raisable user: " << blocker->getName();
  } else if (Operation *bad = findUnsupportedParentAccess(alloca, func)) {
    diag.attachNote(bad->getLoc()) << "access inside unsupported control flow ("
                                   << bad->getParentOp()->getName() << ")";
  } else if (alloca->getBlock() != &func.getBody().front()) {
    diag.attachNote() << "alloca is not in function entry block";
  } else if (!isAccessedInsideSdeRegion(alloca)) {
    diag.attachNote() << "alloca has no load/store users inside SDE regions";
  }
}

static SmallVector<memref::AllocaOp>
collectFunctionScopeAllocas(func::FuncOp func) {
  SmallVector<memref::AllocaOp> result;
  if (func.getBody().empty())
    return result;

  Block &entry = func.getBody().front();
  for (Operation &op : entry) {
    auto alloca = dyn_cast<memref::AllocaOp>(&op);
    if (!alloca)
      continue;
    if (!isRaisableMemref(alloca.getResult(), func))
      continue;
    if (!isAccessedInsideSdeRegion(alloca))
      continue;
    result.push_back(alloca);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Step 2: Initialize tensor values
//===----------------------------------------------------------------------===//

/// Create initial tensor carrier values for each alloca and populate
/// currentTensor.
static void initializeTensorValues(ArrayRef<memref::AllocaOp> allocas,
                                   DenseMap<Value, Value> &currentTensor) {
  for (memref::AllocaOp alloca : allocas) {
    auto mrType = cast<MemRefType>(alloca.getType());
    RankedTensorType tensorTy = getTensorType(mrType);
    Location loc = alloca.getLoc();

    // Create sde.mu_alloc directly after the alloca it represents. Polygeist
    // can emit stores between function-scope allocas; placing every carrier
    // after the last alloca would make those early stores rewrite to a tensor
    // value that does not dominate the rewritten store.
    OpBuilder builder(alloca);
    builder.setInsertionPointAfter(alloca);
    Value tensor = sde::SdeMuAllocOp::create(builder, loc, tensorTy,
                                             alloca.getDynamicSizes());

    currentTensor[alloca.getResult()] = tensor;
  }
}

//===----------------------------------------------------------------------===//
// Forward declarations for recursive dispatch
//===----------------------------------------------------------------------===//

static void walkBlock(Block &block, DenseMap<Value, Value> &currentTensor,
                      ArrayRef<memref::AllocaOp> allocas);

//===----------------------------------------------------------------------===//
// Step 3: Thread through scf.for
//===----------------------------------------------------------------------===//

/// Collect which raised allocas are accessed inside an scf.for body.
static SmallVector<memref::AllocaOp>
collectAccessedAllocas(Region &region, ArrayRef<memref::AllocaOp> candidates) {
  DenseSet<Value> accessed;
  region.walk([&](Operation *op) {
    Value memref;
    if (auto ld = dyn_cast<memref::LoadOp>(op))
      memref = ld.getMemref();
    else if (auto st = dyn_cast<memref::StoreOp>(op))
      memref = st.getMemref();
    else
      return;
    if (auto alloca = resolveAlloca(memref))
      accessed.insert(alloca.getResult());
  });

  SmallVector<memref::AllocaOp> result;
  for (memref::AllocaOp alloca : candidates) {
    if (accessed.contains(alloca.getResult()))
      result.push_back(alloca);
  }
  return result;
}

static void threadThroughScfFor(scf::ForOp forOp,
                                DenseMap<Value, Value> &currentTensor,
                                ArrayRef<memref::AllocaOp> allocas) {
  SmallVector<memref::AllocaOp> accessed =
      collectAccessedAllocas(forOp.getBodyRegion(), allocas);
  if (accessed.empty()) {
    // No raised allocas inside — still recurse for nested ops.
    walkBlock(*forOp.getBody(), currentTensor, allocas);
    return;
  }

  Location loc = forOp.getLoc();

  // Build extended iter_args.
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                 forOp.getInitArgs().end());
  unsigned existingIterArgs = newInitArgs.size();
  for (memref::AllocaOp alloca : accessed)
    newInitArgs.push_back(currentTensor[alloca.getResult()]);

  // Build new ForOp with empty body.
  OpBuilder builder(forOp);
  auto newFor =
      scf::ForOp::create(builder, loc, forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), newInitArgs);

  // Move old body into new ForOp (the new ForOp was created with a default body
  // including IV + iter_arg block args). We need to move ops from the old body.
  Block &oldBody = *forOp.getBody();
  Block &newBody = *newFor.getBody();

  // The new ForOp's body already has block args for the tensor iter_args
  // (created by scf::ForOp::create). Collect them.
  SmallVector<Value> tensorBlockArgs;
  for (unsigned i = 0; i < accessed.size(); ++i)
    tensorBlockArgs.push_back(
        newBody.getArgument(1 + existingIterArgs + i)); // skip IV + existing

  // Replace old block arg uses with new block args, then move ops.
  for (auto [oldArg, newArg] :
       llvm::zip(oldBody.getArguments(), newBody.getArguments()))
    oldArg.replaceAllUsesWith(newArg);

  // Erase the empty yield in the new body (if scf.for builder added one).
  if (!newBody.empty() && newBody.back().hasTrait<OpTrait::IsTerminator>())
    newBody.back().erase();

  // Move all ops from old body to new body.
  newBody.getOperations().splice(newBody.end(), oldBody.getOperations());

  // Set up currentTensor for recursion into body.
  DenseMap<Value, Value> bodyTensor(currentTensor);
  for (auto [alloca, blockArg] : llvm::zip(accessed, tensorBlockArgs))
    bodyTensor[alloca.getResult()] = blockArg;

  // Recurse into body (rewrites loads/stores/nested ops).
  // Skip the terminator — we'll update it below.
  walkBlock(newBody, bodyTensor, allocas);

  // Update yield: extend with tensor values.
  auto yield = cast<scf::YieldOp>(newBody.getTerminator());
  SmallVector<Value> yieldedValues(yield.getResults().begin(),
                                   yield.getResults().end());
  for (memref::AllocaOp alloca : accessed)
    yieldedValues.push_back(bodyTensor[alloca.getResult()]);

  OpBuilder yieldBuilder(yield);
  scf::YieldOp::create(yieldBuilder, loc, yieldedValues);
  yield.erase();

  // Replace old ForOp results.
  for (unsigned i = 0; i < existingIterArgs; ++i)
    forOp.getResult(i).replaceAllUsesWith(newFor.getResult(i));
  forOp.erase();

  // Update currentTensor with ForOp results.
  for (auto [i, alloca] : llvm::enumerate(accessed))
    currentTensor[alloca.getResult()] = newFor.getResult(existingIterArgs + i);
}

//===----------------------------------------------------------------------===//
// Step 3b: Thread through scf.if
//===----------------------------------------------------------------------===//

static SmallVector<memref::AllocaOp>
collectAccessedAllocas(scf::IfOp ifOp, ArrayRef<memref::AllocaOp> candidates) {
  DenseSet<Value> accessed;
  for (memref::AllocaOp alloca :
       collectAccessedAllocas(ifOp.getThenRegion(), candidates))
    accessed.insert(alloca.getResult());
  if (!ifOp.getElseRegion().empty()) {
    for (memref::AllocaOp alloca :
         collectAccessedAllocas(ifOp.getElseRegion(), candidates))
      accessed.insert(alloca.getResult());
  }

  SmallVector<memref::AllocaOp> result;
  for (memref::AllocaOp alloca : candidates) {
    if (accessed.contains(alloca.getResult()))
      result.push_back(alloca);
  }
  return result;
}

static void appendScfIfYield(Block &block, Location loc,
                             ArrayRef<memref::AllocaOp> accessed,
                             DenseMap<Value, Value> &branchTensor) {
  scf::YieldOp oldYield;
  if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>())
    oldYield = dyn_cast<scf::YieldOp>(&block.back());
  SmallVector<Value> yieldedValues;
  if (oldYield)
    yieldedValues.assign(oldYield.getResults().begin(),
                         oldYield.getResults().end());
  for (memref::AllocaOp alloca : accessed)
    yieldedValues.push_back(branchTensor[alloca.getResult()]);

  OpBuilder builder(block.getParent()->getContext());
  if (oldYield)
    builder.setInsertionPoint(oldYield);
  else
    builder.setInsertionPointToEnd(&block);
  scf::YieldOp::create(builder, oldYield ? oldYield.getLoc() : loc,
                       yieldedValues);
  if (oldYield)
    oldYield.erase();
}

static void threadThroughScfIf(scf::IfOp ifOp,
                               DenseMap<Value, Value> &currentTensor,
                               ArrayRef<memref::AllocaOp> allocas) {
  SmallVector<memref::AllocaOp> accessed =
      collectAccessedAllocas(ifOp, allocas);

  if (accessed.empty()) {
    DenseMap<Value, Value> thenTensor(currentTensor);
    walkBlock(ifOp.getThenRegion().front(), thenTensor, allocas);
    if (!ifOp.getElseRegion().empty()) {
      DenseMap<Value, Value> elseTensor(currentTensor);
      walkBlock(ifOp.getElseRegion().front(), elseTensor, allocas);
    }
    return;
  }

  Location loc = ifOp.getLoc();
  SmallVector<Type> newResultTypes(ifOp.getResultTypes().begin(),
                                   ifOp.getResultTypes().end());
  unsigned existingResults = newResultTypes.size();
  for (memref::AllocaOp alloca : accessed)
    newResultTypes.push_back(currentTensor[alloca.getResult()].getType());

  OpBuilder outerBuilder(ifOp);
  auto newIf = scf::IfOp::create(outerBuilder, loc, TypeRange{newResultTypes},
                                 ifOp.getCondition(),
                                 /*addThenBlock=*/false,
                                 /*addElseBlock=*/false);

  bool hadElse = !ifOp.getElseRegion().empty();
  newIf.getThenRegion().takeBody(ifOp.getThenRegion());
  if (hadElse) {
    newIf.getElseRegion().takeBody(ifOp.getElseRegion());
  } else {
    OpBuilder elseBuilder(ifOp.getContext());
    elseBuilder.createBlock(&newIf.getElseRegion());
  }

  DenseMap<Value, Value> thenTensor(currentTensor);
  walkBlock(newIf.getThenRegion().front(), thenTensor, allocas);
  appendScfIfYield(newIf.getThenRegion().front(), loc, accessed, thenTensor);

  DenseMap<Value, Value> elseTensor(currentTensor);
  if (hadElse)
    walkBlock(newIf.getElseRegion().front(), elseTensor, allocas);
  appendScfIfYield(newIf.getElseRegion().front(), loc, accessed, elseTensor);

  for (unsigned i = 0; i < existingResults; ++i)
    ifOp.getResult(i).replaceAllUsesWith(newIf.getResult(i));
  ifOp.erase();

  for (auto [i, alloca] : llvm::enumerate(accessed))
    currentTensor[alloca.getResult()] = newIf.getResult(existingResults + i);
}

//===----------------------------------------------------------------------===//
// Step 4: Thread through cu_region
//===----------------------------------------------------------------------===//

static void threadThroughCuRegion(sde::SdeCuRegionOp cuRegion,
                                  DenseMap<Value, Value> &currentTensor,
                                  ArrayRef<memref::AllocaOp> allocas) {
  SmallVector<memref::AllocaOp> accessed =
      collectAccessedAllocas(cuRegion.getBody(), allocas);

  if (accessed.empty()) {
    // Still recurse for nested ops.
    walkBlock(cuRegion.getBody().front(), currentTensor, allocas);
    return;
  }

  Location loc = cuRegion.getLoc();
  MLIRContext *ctx = cuRegion.getContext();

  // Build extended iter_args.
  SmallVector<Value> newIterArgs(cuRegion.getIterArgs().begin(),
                                 cuRegion.getIterArgs().end());
  SmallVector<Type> newResultTypes(cuRegion.getResultTypes().begin(),
                                   cuRegion.getResultTypes().end());
  unsigned existingIterArgs = newIterArgs.size();

  SmallVector<Type> tensorTypes;
  for (memref::AllocaOp alloca : accessed) {
    auto mrType = cast<MemRefType>(alloca.getType());
    RankedTensorType tensorTy = getTensorType(mrType);
    tensorTypes.push_back(tensorTy);

    // Use the current tensor value (from outer scope) if available,
    // otherwise create a snapshot from the memref.
    auto it = currentTensor.find(alloca.getResult());
    if (it != currentTensor.end()) {
      newIterArgs.push_back(it->second);
    } else {
      OpBuilder initBuilder(cuRegion);
      Value initTensor = sde::SdeMuMemrefToTensorOp::create(
          initBuilder, loc, tensorTy, alloca.getResult());
      newIterArgs.push_back(initTensor);
    }
    newResultTypes.push_back(tensorTy);
  }

  // Create new cu_region with extended iter_args.
  OpBuilder outerBuilder(cuRegion);
  auto newCuRegion = sde::SdeCuRegionOp::create(
      outerBuilder, loc, TypeRange{newResultTypes}, cuRegion.getKindAttr(),
      cuRegion.getConcurrencyScopeAttr(),
      cuRegion.getNowaitAttr() ? UnitAttr::get(ctx) : nullptr,
      ValueRange{newIterArgs});

  // Move old body into new cu_region, adding block args for tensors.
  Region &newBody = newCuRegion.getBody();
  newBody.takeBody(cuRegion.getBody());

  Block &entry = newBody.front();
  SmallVector<Value> tensorBlockArgs;
  for (Type ty : tensorTypes)
    tensorBlockArgs.push_back(entry.addArgument(ty, loc));

  // Set up body currentTensor.
  DenseMap<Value, Value> bodyTensor(currentTensor);
  for (auto [alloca, blockArg] : llvm::zip(accessed, tensorBlockArgs))
    bodyTensor[alloca.getResult()] = blockArg;

  // Recurse into body.
  walkBlock(entry, bodyTensor, allocas);

  // Update yield — add one if missing (cu_region without iter_args may lack
  // terminator).
  if (entry.empty() || !entry.back().hasTrait<OpTrait::IsTerminator>()) {
    // No terminator: create a yield with tensor values.
    OpBuilder yieldBuilder(newCuRegion.getContext());
    yieldBuilder.setInsertionPointToEnd(&entry);
    SmallVector<Value> yieldedValues;
    for (memref::AllocaOp alloca : accessed)
      yieldedValues.push_back(bodyTensor[alloca.getResult()]);
    sde::SdeYieldOp::create(yieldBuilder, loc, yieldedValues);
  } else {
    auto yield = cast<sde::SdeYieldOp>(entry.getTerminator());
    SmallVector<Value> yieldedValues(yield.getValues().begin(),
                                     yield.getValues().end());
    for (memref::AllocaOp alloca : accessed)
      yieldedValues.push_back(bodyTensor[alloca.getResult()]);

    OpBuilder yieldBuilder(yield);
    sde::SdeYieldOp::create(yieldBuilder, yield.getLoc(), yieldedValues);
    yield.erase();
  }

  // After the new cu_region: materialize results back to memrefs only for
  // allocas that were NOT already tracked in currentTensor (i.e., outermost
  // scope where we created mu_memref_to_tensor).
  OpBuilder afterBuilder(ctx);
  afterBuilder.setInsertionPointAfter(newCuRegion);
  for (auto [i, alloca] : llvm::enumerate(accessed)) {
    Value tensorResult = newCuRegion.getResult(existingIterArgs + i);
    auto it = currentTensor.find(alloca.getResult());
    if (it == currentTensor.end()) {
      // Outermost scope: write back to memref.
      sde::SdeMuTensorToMemrefOp::create(afterBuilder, loc, tensorResult,
                                         alloca.getResult());
    }
  }

  // Replace old cu_region results.
  for (unsigned i = 0; i < existingIterArgs; ++i)
    cuRegion.getResult(i).replaceAllUsesWith(newCuRegion.getResult(i));
  cuRegion.erase();

  // Update currentTensor with cu_region results.
  for (auto [i, alloca] : llvm::enumerate(accessed))
    currentTensor[alloca.getResult()] =
        newCuRegion.getResult(existingIterArgs + i);
}

//===----------------------------------------------------------------------===//
// Step 5: Thread through su_iterate
//===----------------------------------------------------------------------===//

static void threadThroughSuIterate(sde::SdeSuIterateOp suIter,
                                   DenseMap<Value, Value> &currentTensor,
                                   ArrayRef<memref::AllocaOp> allocas) {
  SmallVector<memref::AllocaOp> accessed =
      collectAccessedAllocas(suIter.getBody(), allocas);

  if (accessed.empty()) {
    // Still recurse for nested ops.
    walkBlock(suIter.getBody().front(), currentTensor, allocas);
    return;
  }

  Location loc = suIter.getLoc();
  MLIRContext *ctx = suIter.getContext();

  // Collect existing reductionAccumulators (which serve as iter_args).
  SmallVector<Value> newAccumulators(suIter.getReductionAccumulators().begin(),
                                     suIter.getReductionAccumulators().end());
  unsigned existingAccs = newAccumulators.size();

  SmallVector<Type> tensorTypes;
  for (memref::AllocaOp alloca : accessed) {
    auto mrType = cast<MemRefType>(alloca.getType());
    RankedTensorType tensorTy = getTensorType(mrType);
    tensorTypes.push_back(tensorTy);
    newAccumulators.push_back(currentTensor[alloca.getResult()]);
  }

  // Compute result types: existing results + new tensor results.
  SmallVector<Type> newResultTypes(suIter.getResultTypes().begin(),
                                   suIter.getResultTypes().end());
  newResultTypes.append(tensorTypes.begin(), tensorTypes.end());

  // Rebuild su_iterate with extended accumulators.
  OpBuilder outerBuilder(suIter);
  NamedAttrList attrs = sde::getRewrittenAttrs(suIter);

  auto newSuIter = sde::SdeSuIterateOp::create(
      outerBuilder, loc, newResultTypes, suIter.getLowerBounds(),
      suIter.getUpperBounds(), suIter.getSteps(), suIter.getScheduleAttr(),
      suIter.getChunkSize(),
      suIter.getNowaitAttr() ? UnitAttr::get(ctx) : nullptr, newAccumulators,
      suIter.getReductionKindsAttr(), suIter.getReductionStrategyAttr(),
      suIter.getStructuredClassificationAttr(),
      suIter.getPatternAttr(),
      suIter.getAccessMinOffsetsAttr(), suIter.getAccessMaxOffsetsAttr(),
      suIter.getOwnerDimsAttr(), suIter.getSpatialDimsAttr(),
      suIter.getWriteFootprintAttr(), suIter.getPhysicalOwnerDimsAttr(),
      suIter.getPhysicalBlockShapeAttr(), suIter.getLogicalWorkerSliceAttr(),
      suIter.getPhysicalHaloShapeAttr(), suIter.getIterationTopologyAttr(),
      suIter.getRepetitionStructureAttr(), suIter.getAsyncStrategyAttr(),
      suIter.getCpsGroupIdAttr(), suIter.getCpsStageIndexAttr(),
      suIter.getCpsStageCountAttr(),
      suIter.getDistributionKindAttr(), suIter.getInPlaceSafeAttr(),
      suIter.getInPlaceSharedStateAttr(), suIter.getVectorizeWidthAttr(),
      suIter.getUnrollFactorAttr(), suIter.getInterleaveCountAttr());

  // Move old body into new.
  newSuIter.getBody().takeBody(suIter.getBody());
  Block &entry = newSuIter.getBody().front();

  // Add block args for tensors (after IVs + existing iter_args).
  SmallVector<Value> tensorBlockArgs;
  for (Type ty : tensorTypes)
    tensorBlockArgs.push_back(entry.addArgument(ty, loc));

  // Set up body currentTensor.
  DenseMap<Value, Value> bodyTensor(currentTensor);
  for (auto [alloca, blockArg] : llvm::zip(accessed, tensorBlockArgs))
    bodyTensor[alloca.getResult()] = blockArg;

  // Recurse into body.
  walkBlock(entry, bodyTensor, allocas);

  // Update yield — add one if missing (su_iterate without accumulators may lack
  // terminator).
  if (entry.empty() || !entry.back().hasTrait<OpTrait::IsTerminator>()) {
    OpBuilder yieldBuilder(newSuIter.getContext());
    yieldBuilder.setInsertionPointToEnd(&entry);
    SmallVector<Value> yieldedValues;
    for (memref::AllocaOp alloca : accessed)
      yieldedValues.push_back(bodyTensor[alloca.getResult()]);
    sde::SdeYieldOp::create(yieldBuilder, loc, yieldedValues);
  } else {
    auto yield = cast<sde::SdeYieldOp>(entry.getTerminator());
    SmallVector<Value> yieldedValues(yield.getValues().begin(),
                                     yield.getValues().end());
    for (memref::AllocaOp alloca : accessed)
      yieldedValues.push_back(bodyTensor[alloca.getResult()]);

    OpBuilder yieldBuilder(yield);
    sde::SdeYieldOp::create(yieldBuilder, yield.getLoc(), yieldedValues);
    yield.erase();
  }

  // Copy attributes from old op that aren't segment sizes.
  for (auto attr : attrs)
    newSuIter->setAttr(attr.getName(), attr.getValue());

  // Replace old su_iterate results.
  for (unsigned i = 0, e = suIter.getNumResults(); i < e; ++i)
    suIter.getResult(i).replaceAllUsesWith(newSuIter.getResult(i));
  suIter.erase();

  // Update currentTensor with su_iterate results.
  for (auto [i, alloca] : llvm::enumerate(accessed))
    currentTensor[alloca.getResult()] = newSuIter.getResult(existingAccs + i);
}

//===----------------------------------------------------------------------===//
// Program-order walk
//===----------------------------------------------------------------------===//

/// Walk a block in program order, dispatching region-holding ops to the
/// appropriate threading function and rewriting flat loads/stores inline.
static void walkBlock(Block &block, DenseMap<Value, Value> &currentTensor,
                      ArrayRef<memref::AllocaOp> allocas) {
  if (allocas.empty())
    return;
  // We process ops one by one. Region-holding ops are dispatched, flat
  // loads/stores are rewritten inline.
  SmallVector<Operation *> toErase;

  for (Operation &op : llvm::make_early_inc_range(block)) {
    // Terminators are not rewritten here.
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      threadThroughScfFor(forOp, currentTensor, allocas);
      continue;
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      threadThroughScfIf(ifOp, currentTensor, allocas);
      continue;
    }

    if (auto cuRegion = dyn_cast<sde::SdeCuRegionOp>(&op)) {
      threadThroughCuRegion(cuRegion, currentTensor, allocas);
      continue;
    }

    if (auto suIter = dyn_cast<sde::SdeSuIterateOp>(&op)) {
      threadThroughSuIterate(suIter, currentTensor, allocas);
      continue;
    }

    // scf.execute_region is transparent — propagate tensor updates through.
    if (auto execRegion = dyn_cast<scf::ExecuteRegionOp>(&op)) {
      if (!execRegion.getRegion().empty())
        walkBlock(execRegion.getRegion().front(), currentTensor, allocas);
      continue;
    }

    // Other region-holding ops: recurse into their bodies with a COPY
    // of currentTensor. Values defined inside scf.if, scf.while,
    // etc. do NOT dominate ops after the region, so we must not leak
    // inner tensor updates to the outer scope.
    if (op.getNumRegions() > 0) {
      for (Region &region : op.getRegions()) {
        if (!region.empty()) {
          DenseMap<Value, Value> innerTensor(currentTensor);
          walkBlock(region.front(), innerTensor, allocas);
        }
      }
      continue;
    }

    // Flat ops: rewrite loads/stores/bufferization.to_tensor.

    // bufferization.to_tensor %alloca → currentTensor[alloca] (read-all).
    if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(&op)) {
      auto alloca = resolveAlloca(toTensorOp->getOperand(0));
      if (!alloca)
        continue;
      auto it = currentTensor.find(alloca.getResult());
      if (it == currentTensor.end())
        continue;

      Value replacement = it->second;
      // Handle type mismatch (e.g., tensor<16xi32> vs tensor<?xi32> from cast).
      if (replacement.getType() != toTensorOp.getType()) {
        OpBuilder b(toTensorOp);
        replacement = tensor::CastOp::create(b, toTensorOp.getLoc(),
                                             toTensorOp.getType(), replacement);
      }
      toTensorOp.getResult().replaceAllUsesWith(replacement);
      toErase.push_back(toTensorOp);
      continue;
    }

    if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
      auto alloca = resolveAlloca(loadOp.getMemref());
      if (!alloca)
        continue;
      auto it = currentTensor.find(alloca.getResult());
      if (it == currentTensor.end())
        continue;

      OpBuilder b(loadOp);
      Value extracted = tensor::ExtractOp::create(
          b, loadOp.getLoc(), it->second, loadOp.getIndices());
      loadOp.getResult().replaceAllUsesWith(extracted);
      toErase.push_back(loadOp);
      continue;
    }

    if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
      auto alloca = resolveAlloca(storeOp.getMemref());
      if (!alloca)
        continue;
      auto it = currentTensor.find(alloca.getResult());
      if (it == currentTensor.end())
        continue;

      OpBuilder b(storeOp);
      Value inserted = tensor::InsertOp::create(
          b, storeOp.getLoc(), storeOp.getValueToStore(), it->second,
          storeOp.getIndices());
      it->second = inserted;
      toErase.push_back(storeOp);
      continue;
    }
  }

  for (Operation *op : llvm::reverse(toErase))
    op->erase();
}

//===----------------------------------------------------------------------===//
// Function-scope raising driver
//===----------------------------------------------------------------------===//

static unsigned raiseFunctionAllocas(func::FuncOp func) {
  // Only raise in functions that contain SDE region ops.
  bool hasSdeOps = false;
  func.walk([&](Operation *op) -> WalkResult {
    if (isa<sde::SdeCuRegionOp, sde::SdeSuIterateOp>(op)) {
      hasSdeOps = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!hasSdeOps)
    return 0;

  SmallVector<memref::AllocaOp> allocas = collectFunctionScopeAllocas(func);
  if (allocas.empty())
    return 0;

  Block &entry = func.getBody().front();
  DenseMap<Value, Value> currentTensor;
  initializeTensorValues(allocas, currentTensor);

  // Walk function body in program order.
  walkBlock(entry, currentTensor, allocas);

  // Erase dead allocas and their preserve tags.
  unsigned erased = 0;
  for (memref::AllocaOp alloca : allocas) {
    alloca->removeAttr(AttrNames::Operation::Preserve);
    if (alloca.use_empty()) {
      alloca.erase();
      ++erased;
    }
  }

  ARTS_DEBUG("raised " << erased << "/" << allocas.size() << " alloca(s) in "
                       << func.getName());
  return erased;
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct RaiseToTensorPass
    : public arts::impl::RaiseToTensorBase<RaiseToTensorPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Phase 1: Function-scope tensor raising.
    unsigned totalRaised = 0;
    module.walk(
        [&](func::FuncOp func) { totalRaised += raiseFunctionAllocas(func); });

    // Phase 2: VERIFY no remaining captured allocas. If any memref.alloca
    // is captured by a cu_region but was not raised to tensor form, the pass
    // fails with a diagnostic. This ensures downstream passes never see raw
    // memref accesses inside SDE regions.
    unsigned unraised = 0;
    module.walk([&](func::FuncOp func) {
      func.walk([&](sde::SdeCuRegionOp cuRegion) {
        DenseSet<memref::AllocaOp> found;
        cuRegion.getBody().walk([&](Operation *op) {
          Value memref;
          if (auto ld = dyn_cast<memref::LoadOp>(op))
            memref = ld.getMemref();
          else if (auto st = dyn_cast<memref::StoreOp>(op))
            memref = st.getMemref();
          else
            return;
          if (auto alloca = resolveAlloca(memref)) {
            if (!cuRegion.getBody().isAncestor(alloca->getParentRegion()))
              found.insert(alloca);
          }
        });
        for (memref::AllocaOp alloca : found) {
          // Reduction accumulators are managed by su_iterate — skip.
          if (isManagedByReduction(alloca))
            continue;
          // Allocas defined inside an ancestor cu_region are local state
          // that can stay in memref form (e.g., scalars used inside scf.if).
          // Only flag truly external captures (function-scope allocas).
          if (auto parentCuRegion =
                  alloca->getParentOfType<sde::SdeCuRegionOp>())
            continue;
          diagnoseCapturedAlloca(alloca, cuRegion, func);
          ++unraised;
        }
      });
    });

    if (unraised > 0) {
      ARTS_INFO("RaiseToTensor: FAILED — "
                << unraised << " captured alloca(s) not raised to tensor");
      signalPassFailure();
      return;
    }

    ARTS_INFO("RaiseToTensor: raised "
              << totalRaised << " alloca(s), all captured memrefs eliminated");
  }
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createRaiseToTensorPass() {
  return std::make_unique<RaiseToTensorPass>();
}

} // namespace mlir::carts::sde
