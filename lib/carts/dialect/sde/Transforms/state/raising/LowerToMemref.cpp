///==========================================================================///
/// File: LowerToMemref.cpp
///
/// Inverse of RaiseToTensor's mem2reg: lowers tensor SSA values back to
/// memref.alloca + memref.load/memref.store form.
///
/// Four-phase algorithm:
///   Phase A: Walk each function body, rewriting tensor.insert → memref.store,
///            tensor.extract → memref.load, and mapping tensor values to
///            their backing memrefs. Skips <single> cu_region bodies (those
///            are handled by ConvertToCodelet).
///   Phase B: Strip tensor iter_args from scf.for, cu_region, su_iterate
///            using replaceAllUsesWith (not dropAllUses) to safely handle
///            nested ops. Tensor iter_args that feed <single> cu_regions
///            are preserved.
///   Phase C: Lower authoritative linalg carriers to scf.for loops with
///            memref.load/memref.store by converting tensor operands to
///            memrefs and calling linalg::linalgOpToLoops().
///   Phase D: Erase remaining dead linalg carriers and tensor ops.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_LOWERTOMEMREF
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "carts/utils/OperationAttributes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(lower_to_memref);

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::carts;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static MemRefType getMemRefType(RankedTensorType tensorTy) {
  return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
}

static Value getAdjacentOriginalAlloca(sde::SdeMuAllocOp muAlloc,
                                       MemRefType expectedType) {
  Operation *prev = muAlloc->getPrevNode();
  auto alloca = dyn_cast_or_null<memref::AllocaOp>(prev);
  if (!alloca || alloca.getType() != expectedType)
    return Value();
  alloca->setAttr(AttrNames::Operation::Preserve,
                  UnitAttr::get(muAlloc.getContext()));
  return alloca.getResult();
}

static void markPreserveIfAlloca(Value memref) {
  while (auto castOp = memref.getDefiningOp<memref::CastOp>())
    memref = castOp.getSource();
  if (auto alloca = memref.getDefiningOp<memref::AllocaOp>())
    alloca->setAttr(AttrNames::Operation::Preserve,
                    UnitAttr::get(alloca.getContext()));
}

/// Trace a tensor value through tensor.insert / tensor.cast chains to find
/// the backing memref.
static Value traceBackingMemref(Value tensor,
                                DenseMap<Value, Value> &backingMemref) {
  auto it = backingMemref.find(tensor);
  if (it != backingMemref.end())
    return it->second;
  if (auto muToTensor = tensor.getDefiningOp<sde::SdeMuMemrefToTensorOp>()) {
    markPreserveIfAlloca(muToTensor.getMemref());
    return muToTensor.getMemref();
  }
  if (auto toTensor = tensor.getDefiningOp<bufferization::ToTensorOp>()) {
    markPreserveIfAlloca(toTensor.getBuffer());
    return toTensor.getBuffer();
  }
  if (auto insertOp = tensor.getDefiningOp<tensor::InsertOp>())
    return traceBackingMemref(insertOp.getDest(), backingMemref);
  if (auto castOp = tensor.getDefiningOp<tensor::CastOp>())
    return traceBackingMemref(castOp.getSource(), backingMemref);
  if (auto ifOp = tensor.getDefiningOp<scf::IfOp>()) {
    unsigned resultIdx = cast<OpResult>(tensor).getResultNumber();
    auto traceYieldedBacking = [&](Region &region) -> Value {
      if (region.empty())
        return Value();
      auto yield =
          dyn_cast_or_null<scf::YieldOp>(region.front().getTerminator());
      if (!yield || resultIdx >= yield.getResults().size())
        return Value();
      return traceBackingMemref(yield.getResults()[resultIdx], backingMemref);
    };

    Value thenBacking = traceYieldedBacking(ifOp.getThenRegion());
    Value elseBacking = traceYieldedBacking(ifOp.getElseRegion());
    if (thenBacking && (!elseBacking || thenBacking == elseBacking))
      return thenBacking;
    if (!thenBacking && elseBacking)
      return elseBacking;
  }
  if (auto forOp = tensor.getDefiningOp<scf::ForOp>()) {
    unsigned resultIdx = cast<OpResult>(tensor).getResultNumber();
    if (resultIdx < forOp.getInitArgs().size())
      return traceBackingMemref(forOp.getInitArgs()[resultIdx], backingMemref);
  }
  if (auto cuRegion = tensor.getDefiningOp<sde::SdeCuRegionOp>()) {
    unsigned resultIdx = cast<OpResult>(tensor).getResultNumber();
    if (resultIdx < cuRegion.getIterArgs().size())
      return traceBackingMemref(cuRegion.getIterArgs()[resultIdx],
                                backingMemref);
  }
  if (auto suIter = tensor.getDefiningOp<sde::SdeSuIterateOp>()) {
    unsigned resultIdx = cast<OpResult>(tensor).getResultNumber();
    if (resultIdx < suIter.getReductionAccumulators().size())
      return traceBackingMemref(suIter.getReductionAccumulators()[resultIdx],
                                backingMemref);
  }
  return Value();
}

/// Find or create a backing memref alloca for a tensor init value.
/// Traces through mu_memref_to_tensor / bufferization.to_tensor to find
/// existing allocas, or creates a new memref.alloca as fallback.
static Value getOrCreateBacking(Value initTensor, OpBuilder &builder,
                                DenseMap<Value, Value> &backingMemref) {
  if (Value backing = traceBackingMemref(initTensor, backingMemref))
    return backing;

  // Trace through known conversion ops.
  if (auto muToTensor =
          initTensor.getDefiningOp<sde::SdeMuMemrefToTensorOp>()) {
    markPreserveIfAlloca(muToTensor.getMemref());
    backingMemref[initTensor] = muToTensor.getMemref();
    return muToTensor.getMemref();
  }
  if (auto toTensor = initTensor.getDefiningOp<bufferization::ToTensorOp>()) {
    markPreserveIfAlloca(toTensor.getBuffer());
    backingMemref[initTensor] = toTensor.getBuffer();
    return toTensor.getBuffer();
  }

  // Follow tensor block arguments back to their region iter_args. This keeps
  // nested loops inside <single> regions tied to the parent tensor backing.
  if (auto blockArg = dyn_cast<BlockArgument>(initTensor)) {
    Operation *owner = blockArg.getOwner()->getParentOp();
    unsigned argNo = blockArg.getArgNumber();
    Value backing;

    if (auto forOp = dyn_cast_or_null<scf::ForOp>(owner)) {
      if (argNo > 0) {
        unsigned initIdx = argNo - 1;
        if (initIdx < forOp.getInitArgs().size())
          backing = getOrCreateBacking(forOp.getInitArgs()[initIdx], builder,
                                       backingMemref);
      }
    } else if (auto cuRegion = dyn_cast_or_null<sde::SdeCuRegionOp>(owner)) {
      if (argNo < cuRegion.getIterArgs().size())
        backing = getOrCreateBacking(cuRegion.getIterArgs()[argNo], builder,
                                     backingMemref);
    } else if (auto suIter = dyn_cast_or_null<sde::SdeSuIterateOp>(owner)) {
      unsigned numIVs = suIter.getLowerBounds().size();
      if (argNo >= numIVs) {
        unsigned accIdx = argNo - numIVs;
        if (accIdx < suIter.getReductionAccumulators().size())
          backing =
              getOrCreateBacking(suIter.getReductionAccumulators()[accIdx],
                                 builder, backingMemref);
      }
    }

    if (backing) {
      backingMemref[initTensor] = backing;
      return backing;
    }
  }

  // Trace through tensor.insert chains.
  if (auto insertOp = initTensor.getDefiningOp<tensor::InsertOp>()) {
    Value backing =
        getOrCreateBacking(insertOp.getDest(), builder, backingMemref);
    backingMemref[initTensor] = backing;
    return backing;
  }
  if (auto castOp = initTensor.getDefiningOp<tensor::CastOp>()) {
    Value backing =
        getOrCreateBacking(castOp.getSource(), builder, backingMemref);
    backingMemref[initTensor] = backing;
    return backing;
  }

  if (auto ifOp = initTensor.getDefiningOp<scf::IfOp>()) {
    unsigned resultIdx = cast<OpResult>(initTensor).getResultNumber();
    auto getYieldedBacking = [&](Region &region) -> Value {
      if (region.empty())
        return Value();
      auto yield =
          dyn_cast_or_null<scf::YieldOp>(region.front().getTerminator());
      if (!yield || resultIdx >= yield.getResults().size())
        return Value();
      return getOrCreateBacking(yield.getResults()[resultIdx], builder,
                                backingMemref);
    };

    Value thenBacking = getYieldedBacking(ifOp.getThenRegion());
    Value elseBacking = getYieldedBacking(ifOp.getElseRegion());
    if (thenBacking && (!elseBacking || thenBacking == elseBacking)) {
      backingMemref[initTensor] = thenBacking;
      return thenBacking;
    }
    if (!thenBacking && elseBacking) {
      backingMemref[initTensor] = elseBacking;
      return elseBacking;
    }
  }

  if (auto forOp = initTensor.getDefiningOp<scf::ForOp>()) {
    unsigned resultIdx = cast<OpResult>(initTensor).getResultNumber();
    if (resultIdx < forOp.getInitArgs().size()) {
      Value backing = getOrCreateBacking(forOp.getInitArgs()[resultIdx],
                                         builder, backingMemref);
      backingMemref[initTensor] = backing;
      return backing;
    }
  }

  if (auto cuRegion = initTensor.getDefiningOp<sde::SdeCuRegionOp>()) {
    unsigned resultIdx = cast<OpResult>(initTensor).getResultNumber();
    if (resultIdx < cuRegion.getIterArgs().size()) {
      Value backing = getOrCreateBacking(cuRegion.getIterArgs()[resultIdx],
                                         builder, backingMemref);
      backingMemref[initTensor] = backing;
      return backing;
    }
  }

  if (auto suIter = initTensor.getDefiningOp<sde::SdeSuIterateOp>()) {
    unsigned resultIdx = cast<OpResult>(initTensor).getResultNumber();
    if (resultIdx < suIter.getReductionAccumulators().size()) {
      Value backing = getOrCreateBacking(
          suIter.getReductionAccumulators()[resultIdx], builder, backingMemref);
      backingMemref[initTensor] = backing;
      return backing;
    }
  }

  // Fallback: create a temporary alloca.
  auto tensorTy = cast<RankedTensorType>(initTensor.getType());
  Value alloca = memref::AllocaOp::create(builder, initTensor.getLoc(),
                                          getMemRefType(tensorTy));
  backingMemref[initTensor] = alloca;
  return alloca;
}

//===----------------------------------------------------------------------===//
// Phase A: Walk block, rewrite tensor ops to memref, map backing memrefs
//===----------------------------------------------------------------------===//

static void walkBlock(Block &block, DenseMap<Value, Value> &backingMemref);

static void processRegionOp(sde::SdeCuRegionOp cuRegion,
                            DenseMap<Value, Value> &backingMemref) {
  // Do NOT walk into <single> bodies — ConvertToCodelet handles them.
  if (cuRegion.getKind() == sde::SdeCuKind::single)
    return;

  Block &body = cuRegion.getBody().front();

  // Map block args for tensor iter_args.
  for (auto [i, arg] : llvm::enumerate(cuRegion.getIterArgs())) {
    if (!isa<RankedTensorType>(arg.getType()))
      continue;
    Value backing = traceBackingMemref(arg, backingMemref);
    if (backing)
      backingMemref[body.getArgument(i)] = backing;
  }
  walkBlock(body, backingMemref);
  // Map results to backing memrefs.
  for (auto [i, arg] : llvm::enumerate(cuRegion.getIterArgs())) {
    if (!isa<RankedTensorType>(arg.getType()))
      continue;
    Value backing = traceBackingMemref(arg, backingMemref);
    if (backing && i < cuRegion.getNumResults())
      backingMemref[cuRegion.getResult(i)] = backing;
  }
}

static void processRegionOp(sde::SdeSuIterateOp suIter,
                            DenseMap<Value, Value> &backingMemref) {
  unsigned numIVs = suIter.getLowerBounds().size();
  Block &body = suIter.getBody().front();
  for (auto [i, acc] : llvm::enumerate(suIter.getReductionAccumulators())) {
    if (!isa<RankedTensorType>(acc.getType()))
      continue;
    Value backing = traceBackingMemref(acc, backingMemref);
    unsigned blockArgIdx = numIVs + i;
    if (backing && blockArgIdx < body.getNumArguments())
      backingMemref[body.getArgument(blockArgIdx)] = backing;
  }
  walkBlock(body, backingMemref);
  for (auto [i, acc] : llvm::enumerate(suIter.getReductionAccumulators())) {
    if (!isa<RankedTensorType>(acc.getType()))
      continue;
    Value backing = traceBackingMemref(acc, backingMemref);
    if (backing && i < suIter.getNumResults())
      backingMemref[suIter.getResult(i)] = backing;
  }
}

static void processRegionOp(scf::ForOp forOp,
                            DenseMap<Value, Value> &backingMemref) {
  for (auto [i, init] : llvm::enumerate(forOp.getInitArgs())) {
    if (!isa<RankedTensorType>(init.getType()))
      continue;
    Value backing = traceBackingMemref(init, backingMemref);
    if (backing)
      backingMemref[forOp.getBody()->getArgument(1 + i)] = backing;
  }
  walkBlock(*forOp.getBody(), backingMemref);
  for (auto [i, init] : llvm::enumerate(forOp.getInitArgs())) {
    if (!isa<RankedTensorType>(init.getType()))
      continue;
    Value backing = traceBackingMemref(init, backingMemref);
    if (backing && i < forOp.getNumResults())
      backingMemref[forOp.getResult(i)] = backing;
  }
}

static void walkBlock(Block &block, DenseMap<Value, Value> &backingMemref) {
  SmallVector<Operation *> toErase;

  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    if (auto muAlloc = dyn_cast<sde::SdeMuAllocOp>(&op)) {
      auto tensorTy = cast<RankedTensorType>(muAlloc.getTensor().getType());
      MemRefType memrefTy = getMemRefType(tensorTy);
      OpBuilder b(muAlloc);
      Value alloca = getAdjacentOriginalAlloca(muAlloc, memrefTy);
      if (!alloca)
        alloca = memref::AllocaOp::create(b, muAlloc.getLoc(), memrefTy,
                                          muAlloc.getDynamicSizes());
      backingMemref[muAlloc.getTensor()] = alloca;
      toErase.push_back(muAlloc);
      continue;
    }

    if (auto muToTensor = dyn_cast<sde::SdeMuMemrefToTensorOp>(&op)) {
      markPreserveIfAlloca(muToTensor.getMemref());
      backingMemref[muToTensor.getTensor()] = muToTensor.getMemref();
      toErase.push_back(muToTensor);
      continue;
    }

    if (auto muToMemref = dyn_cast<sde::SdeMuTensorToMemrefOp>(&op)) {
      toErase.push_back(muToMemref);
      continue;
    }

    if (auto insertOp = dyn_cast<tensor::InsertOp>(&op)) {
      Value backing = traceBackingMemref(insertOp.getDest(), backingMemref);
      if (backing) {
        OpBuilder b(insertOp);
        memref::StoreOp::create(b, insertOp.getLoc(), insertOp.getScalar(),
                                backing, insertOp.getIndices());
        backingMemref[insertOp.getResult()] = backing;
        toErase.push_back(insertOp);
        continue;
      }
    }

    if (auto extractOp = dyn_cast<tensor::ExtractOp>(&op)) {
      Value backing = traceBackingMemref(extractOp.getTensor(), backingMemref);
      if (backing) {
        OpBuilder b(extractOp);
        Value loaded = memref::LoadOp::create(b, extractOp.getLoc(), backing,
                                              extractOp.getIndices());
        extractOp.getResult().replaceAllUsesWith(loaded);
        toErase.push_back(extractOp);
        continue;
      }
    }

    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(&op)) {
      markPreserveIfAlloca(toTensor.getBuffer());
      backingMemref[toTensor.getResult()] = toTensor.getBuffer();
      toErase.push_back(toTensor);
      continue;
    }

    if (auto castOp = dyn_cast<tensor::CastOp>(&op)) {
      Value backing = traceBackingMemref(castOp.getSource(), backingMemref);
      if (backing) {
        backingMemref[castOp.getResult()] = backing;
        toErase.push_back(castOp);
        continue;
      }
    }

    // Vector transfer ops on tensors (from carrier vectorization).
    if (auto readOp = dyn_cast<vector::TransferReadOp>(&op)) {
      if (isa<RankedTensorType>(readOp.getBase().getType())) {
        Value backing = traceBackingMemref(readOp.getBase(), backingMemref);
        if (backing) {
          OpBuilder b(readOp);
          auto newRead = vector::TransferReadOp::create(
              b, readOp.getLoc(), readOp.getVectorType(), backing,
              readOp.getIndices(), readOp.getPermutationMapAttr(),
              readOp.getPadding(), readOp.getMask(), readOp.getInBoundsAttr());
          readOp.getResult().replaceAllUsesWith(newRead.getResult());
          toErase.push_back(readOp);
          continue;
        }
      }
    }
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(&op)) {
      if (isa<RankedTensorType>(writeOp.getBase().getType())) {
        Value backing = traceBackingMemref(writeOp.getBase(), backingMemref);
        if (backing) {
          OpBuilder b(writeOp);
          vector::TransferWriteOp::create(
              b, writeOp.getLoc(), writeOp.getVector(), backing,
              writeOp.getIndices(), writeOp.getPermutationMapAttr(),
              writeOp.getMask(), writeOp.getInBoundsAttr());
          if (writeOp.getNumResults() > 0)
            backingMemref[writeOp.getResult()] = backing;
          toErase.push_back(writeOp);
          continue;
        }
      }
    }

    // Region-holding ops: set up backing map and recurse.
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      processRegionOp(forOp, backingMemref);
      continue;
    }
    if (auto cuRegion = dyn_cast<sde::SdeCuRegionOp>(&op)) {
      processRegionOp(cuRegion, backingMemref);
      continue;
    }
    if (auto suIter = dyn_cast<sde::SdeSuIterateOp>(&op)) {
      processRegionOp(suIter, backingMemref);
      continue;
    }

    // scf.execute_region is transparent — propagate backing map through.
    if (auto execRegion = dyn_cast<scf::ExecuteRegionOp>(&op)) {
      if (!execRegion.getRegion().empty())
        walkBlock(execRegion.getRegion().front(), backingMemref);
      continue;
    }

    // Other region-holding ops.
    if (op.getNumRegions() > 0) {
      for (Region &region : op.getRegions()) {
        if (!region.empty()) {
          DenseMap<Value, Value> innerBacking(backingMemref);
          walkBlock(region.front(), innerBacking);
        }
      }
      continue;
    }
  }

  for (Operation *op : llvm::reverse(toErase)) {
    if (op->use_empty())
      op->erase();
  }
}

//===----------------------------------------------------------------------===//
// Phase B: Strip tensor iter_args using replaceAllUsesWith
//
// For each region-holding op with tensor iter_args:
//   1. Replace tensor block arg uses with mu_memref_to_tensor from backing
//   2. Add mu_tensor_to_memref before yields (store back to backing)
//   3. Replace tensor results with mu_memref_to_tensor after the op
//   4. Rebuild the op without tensor iter_args
//===----------------------------------------------------------------------===//

/// Strip tensor iter_args from scf.for ops.
static void stripScfForTensorArgs(ModuleOp module,
                                  DenseMap<Value, Value> &backingMemref) {
  SmallVector<scf::ForOp> forOps;
  module.walk([&](scf::ForOp op) {
    for (Value init : op.getInitArgs()) {
      if (isa<RankedTensorType>(init.getType())) {
        forOps.push_back(op);
        break;
      }
    }
  });

  for (scf::ForOp forOp : forOps) {
    SmallVector<unsigned> tensorArgIndices;
    SmallVector<unsigned> keptIndices;
    for (unsigned i = 0, e = forOp.getInitArgs().size(); i < e; ++i) {
      if (isa<RankedTensorType>(forOp.getInitArgs()[i].getType()))
        tensorArgIndices.push_back(i);
      else
        keptIndices.push_back(i);
    }
    if (tensorArgIndices.empty())
      continue;

    MLIRContext *ctx = forOp.getContext();
    Location loc = forOp.getLoc();
    Block &body = *forOp.getBody();
    OpBuilder outerBuilder(forOp);

    // Find backing allocas for tensor init values.
    SmallVector<Value> backingAllocas;
    for (unsigned idx : tensorArgIndices) {
      Value backing = getOrCreateBacking(forOp.getInitArgs()[idx], outerBuilder,
                                         backingMemref);
      backingAllocas.push_back(backing);
      backingMemref[body.getArgument(1 + idx)] = backing;
    }

    // Replace tensor block arg uses with mu_memref_to_tensor from alloca.
    for (auto [i, idx] : llvm::enumerate(tensorArgIndices)) {
      BlockArgument blockArg = body.getArgument(1 + idx);
      auto tensorTy = cast<RankedTensorType>(blockArg.getType());
      OpBuilder bodyStart(ctx);
      bodyStart.setInsertionPointToStart(&body);
      Value tensorFromAlloca = sde::SdeMuMemrefToTensorOp::create(
          bodyStart, loc, tensorTy, backingAllocas[i]);
      backingMemref[tensorFromAlloca] = backingAllocas[i];
      blockArg.replaceAllUsesWith(tensorFromAlloca);
    }

    // Before yield: store yielded tensors back to alloca.
    auto yield = cast<scf::YieldOp>(body.getTerminator());
    for (auto [i, idx] : llvm::enumerate(tensorArgIndices)) {
      if (idx < yield.getResults().size()) {
        OpBuilder preYield(yield);
        sde::SdeMuTensorToMemrefOp::create(
            preYield, loc, yield.getResults()[idx], backingAllocas[i]);
      }
    }

    // After for: replace tensor results with reads from alloca.
    OpBuilder afterBuilder(ctx);
    afterBuilder.setInsertionPointAfter(forOp);
    for (auto [i, idx] : llvm::enumerate(tensorArgIndices)) {
      if (idx >= forOp.getNumResults())
        continue;
      auto tensorTy = cast<RankedTensorType>(forOp.getResult(idx).getType());
      Value tensorAfter = sde::SdeMuMemrefToTensorOp::create(
          afterBuilder, loc, tensorTy, backingAllocas[i]);
      backingMemref[tensorAfter] = backingAllocas[i];
      forOp.getResult(idx).replaceAllUsesWith(tensorAfter);
    }

    // Rebuild scf.for without tensor iter_args.
    SmallVector<Value> keptInitArgs;
    for (unsigned idx : keptIndices)
      keptInitArgs.push_back(forOp.getInitArgs()[idx]);

    auto newFor = scf::ForOp::create(outerBuilder, loc, forOp.getLowerBound(),
                                     forOp.getUpperBound(), forOp.getStep(),
                                     keptInitArgs);
    Block &newBody = *newFor.getBody();

    // Map IV.
    body.getArgument(0).replaceAllUsesWith(newBody.getArgument(0));
    // Map kept iter_arg block args.
    for (auto [newIdx, oldIdx] : llvm::enumerate(keptIndices))
      body.getArgument(1 + oldIdx)
          .replaceAllUsesWith(newBody.getArgument(1 + newIdx));

    // Move ops from old body to new. Erase default yield first.
    if (!newBody.empty() && newBody.back().hasTrait<OpTrait::IsTerminator>())
      newBody.back().erase();
    newBody.getOperations().splice(newBody.end(), body.getOperations());

    // Fix yield: keep only non-tensor values.
    auto newYield = cast<scf::YieldOp>(newBody.getTerminator());
    SmallVector<Value> newYieldValues;
    for (unsigned idx : keptIndices)
      newYieldValues.push_back(newYield.getResults()[idx]);
    OpBuilder yieldBuilder(newYield);
    scf::YieldOp::create(yieldBuilder, loc, newYieldValues);
    newYield.erase();

    // Replace kept results.
    for (auto [newIdx, oldIdx] : llvm::enumerate(keptIndices))
      forOp.getResult(oldIdx).replaceAllUsesWith(newFor.getResult(newIdx));

    forOp.erase();
  }
}

/// Strip tensor iter_args from cu_region ops back to their backing memrefs.
/// This includes <single> regions: scalar local state should stay tied to the
/// user-visible memref instead of being copied into isolated codelet-local
/// state that later host reads cannot observe.
static void stripCuRegionTensorArgs(ModuleOp module,
                                    DenseMap<Value, Value> &backingMemref) {
  SmallVector<sde::SdeCuRegionOp> cuRegions;
  module.walk([&](sde::SdeCuRegionOp op) {
    for (Value arg : op.getIterArgs()) {
      if (isa<RankedTensorType>(arg.getType())) {
        cuRegions.push_back(op);
        break;
      }
    }
  });

  for (sde::SdeCuRegionOp cuRegion : cuRegions) {
    Block &body = cuRegion.getBody().front();
    SmallVector<unsigned> tensorArgIndices; // to strip
    SmallVector<unsigned> keptIndices;      // non-tensor
    for (unsigned i = 0, e = cuRegion.getIterArgs().size(); i < e; ++i) {
      if (isa<RankedTensorType>(cuRegion.getIterArgs()[i].getType()))
        tensorArgIndices.push_back(i);
      else
        keptIndices.push_back(i);
    }
    if (tensorArgIndices.empty())
      continue;

    MLIRContext *ctx = cuRegion.getContext();
    Location loc = cuRegion.getLoc();
    OpBuilder outerBuilder(cuRegion);

    // Find backing allocas for tensor init values.
    SmallVector<Value> backingAllocas;
    for (unsigned idx : tensorArgIndices) {
      Value backing = getOrCreateBacking(cuRegion.getIterArgs()[idx],
                                         outerBuilder, backingMemref);
      backingAllocas.push_back(backing);
      backingMemref[body.getArgument(idx)] = backing;
    }

    // Replace tensor block arg uses with mu_memref_to_tensor from alloca.
    for (auto [i, idx] : llvm::enumerate(tensorArgIndices)) {
      BlockArgument blockArg = body.getArgument(idx);
      auto tensorTy = cast<RankedTensorType>(blockArg.getType());
      OpBuilder bodyStart(ctx);
      bodyStart.setInsertionPointToStart(&body);
      Value tensorFromAlloca = sde::SdeMuMemrefToTensorOp::create(
          bodyStart, loc, tensorTy, backingAllocas[i]);
      backingMemref[tensorFromAlloca] = backingAllocas[i];
      blockArg.replaceAllUsesWith(tensorFromAlloca);
    }

    // Before yield: store yielded tensors back to alloca.
    auto yield = cast<sde::SdeYieldOp>(body.getTerminator());
    for (auto [i, idx] : llvm::enumerate(tensorArgIndices)) {
      if (idx < yield.getValues().size()) {
        OpBuilder preYield(yield);
        sde::SdeMuTensorToMemrefOp::create(
            preYield, loc, yield.getValues()[idx], backingAllocas[i]);
      }
    }

    // After cu_region: replace tensor results with reads from alloca.
    OpBuilder afterBuilder(ctx);
    afterBuilder.setInsertionPointAfter(cuRegion);
    for (auto [i, idx] : llvm::enumerate(tensorArgIndices)) {
      if (idx >= cuRegion.getNumResults())
        continue;
      auto tensorTy = cast<RankedTensorType>(cuRegion.getResult(idx).getType());
      Value tensorAfter = sde::SdeMuMemrefToTensorOp::create(
          afterBuilder, loc, tensorTy, backingAllocas[i]);
      backingMemref[tensorAfter] = backingAllocas[i];
      cuRegion.getResult(idx).replaceAllUsesWith(tensorAfter);
    }

    // Rebuild cu_region without stripped tensor iter_args.
    SmallVector<Value> keptIterArgs;
    SmallVector<Type> keptResultTypes;
    for (unsigned idx : keptIndices) {
      keptIterArgs.push_back(cuRegion.getIterArgs()[idx]);
      keptResultTypes.push_back(cuRegion.getResult(idx).getType());
    }

    OpBuilder rebuilder(cuRegion);
    auto newRegion = sde::SdeCuRegionOp::create(
        rebuilder, loc, TypeRange{keptResultTypes}, cuRegion.getKindAttr(),
        cuRegion.getConcurrencyScopeAttr(),
        cuRegion.getNowaitAttr() ? UnitAttr::get(ctx) : nullptr,
        ValueRange{keptIterArgs});

    newRegion.getBody().takeBody(cuRegion.getBody());
    Block &newBody = newRegion.getBody().front();

    // Erase stripped tensor block args (reverse order to keep indices valid).
    for (auto it = tensorArgIndices.rbegin(); it != tensorArgIndices.rend();
         ++it)
      newBody.eraseArgument(*it);

    // Fix yield: keep only non-stripped values.
    if (auto oldYield =
            dyn_cast_or_null<sde::SdeYieldOp>(newBody.getTerminator())) {
      SmallVector<Value> newYieldValues;
      for (unsigned idx : keptIndices) {
        if (idx < oldYield.getValues().size())
          newYieldValues.push_back(oldYield.getValues()[idx]);
      }
      OpBuilder yieldBuilder(oldYield);
      sde::SdeYieldOp::create(yieldBuilder, loc, newYieldValues);
      oldYield.erase();
    }

    // Replace kept results.
    for (auto [newIdx, oldIdx] : llvm::enumerate(keptIndices))
      cuRegion.getResult(oldIdx).replaceAllUsesWith(
          newRegion.getResult(newIdx));

    cuRegion.erase();
  }
}

/// Strip tensor accumulators from su_iterate ops.
static void stripSuIterateTensorArgs(ModuleOp module,
                                     DenseMap<Value, Value> &backingMemref) {
  SmallVector<sde::SdeSuIterateOp> suIters;
  module.walk([&](sde::SdeSuIterateOp op) {
    for (Value acc : op.getReductionAccumulators()) {
      if (isa<RankedTensorType>(acc.getType())) {
        suIters.push_back(op);
        break;
      }
    }
  });

  for (sde::SdeSuIterateOp suIter : suIters) {
    ValueRange allAccumulators = suIter.getReductionAccumulators();
    SmallVector<unsigned> tensorAccIndices;
    SmallVector<unsigned> realAccIndices;
    for (auto [i, acc] : llvm::enumerate(allAccumulators)) {
      if (isa<RankedTensorType>(acc.getType()))
        tensorAccIndices.push_back(i);
      else
        realAccIndices.push_back(i);
    }
    if (tensorAccIndices.empty())
      continue;

    unsigned numIVs = suIter.getLowerBounds().size();
    MLIRContext *ctx = suIter.getContext();
    Location loc = suIter.getLoc();
    Block &body = suIter.getBody().front();
    OpBuilder outerBuilder(suIter);

    // Find backing allocas for tensor accumulator init values.
    SmallVector<Value> backingAllocas;
    for (unsigned accIdx : tensorAccIndices) {
      Value backing = getOrCreateBacking(allAccumulators[accIdx], outerBuilder,
                                         backingMemref);
      backingAllocas.push_back(backing);
      unsigned blockArgIdx = numIVs + accIdx;
      if (blockArgIdx < body.getNumArguments())
        backingMemref[body.getArgument(blockArgIdx)] = backing;
    }

    // Replace tensor block arg uses with mu_memref_to_tensor from alloca.
    for (auto [i, accIdx] : llvm::enumerate(tensorAccIndices)) {
      unsigned blockArgIdx = numIVs + accIdx;
      if (blockArgIdx >= body.getNumArguments())
        continue;
      BlockArgument blockArg = body.getArgument(blockArgIdx);
      auto tensorTy = cast<RankedTensorType>(blockArg.getType());
      OpBuilder bodyStart(ctx);
      bodyStart.setInsertionPointToStart(&body);
      Value tensorFromAlloca = sde::SdeMuMemrefToTensorOp::create(
          bodyStart, loc, tensorTy, backingAllocas[i]);
      backingMemref[tensorFromAlloca] = backingAllocas[i];
      blockArg.replaceAllUsesWith(tensorFromAlloca);
    }

    // Before yield: store yielded tensors back to alloca.
    if (auto yield = dyn_cast_or_null<sde::SdeYieldOp>(body.getTerminator())) {
      for (auto [i, accIdx] : llvm::enumerate(tensorAccIndices)) {
        if (accIdx < yield.getValues().size()) {
          OpBuilder preYield(yield);
          sde::SdeMuTensorToMemrefOp::create(
              preYield, loc, yield.getValues()[accIdx], backingAllocas[i]);
        }
      }
    }

    // After su_iterate: replace tensor results with reads from alloca.
    OpBuilder afterBuilder(ctx);
    afterBuilder.setInsertionPointAfter(suIter);
    for (auto [i, accIdx] : llvm::enumerate(tensorAccIndices)) {
      if (accIdx >= suIter.getNumResults())
        continue;
      auto tensorTy =
          cast<RankedTensorType>(suIter.getResult(accIdx).getType());
      Value tensorAfter = sde::SdeMuMemrefToTensorOp::create(
          afterBuilder, loc, tensorTy, backingAllocas[i]);
      backingMemref[tensorAfter] = backingAllocas[i];
      suIter.getResult(accIdx).replaceAllUsesWith(tensorAfter);
    }

    // Rebuild su_iterate with only real (memref) reduction accumulators.
    NamedAttrList attrs = sde::getRewrittenAttrs(suIter);

    SmallVector<Value> realAccValues;
    SmallVector<Type> newResultTypes;
    for (unsigned idx : realAccIndices) {
      realAccValues.push_back(allAccumulators[idx]);
      if (idx < suIter.getNumResults())
        newResultTypes.push_back(suIter.getResult(idx).getType());
    }

    ArrayAttr newReductionKinds;
    if (auto kindsAttr = suIter.getReductionKindsAttr()) {
      SmallVector<Attribute> filteredKinds;
      for (unsigned idx : realAccIndices) {
        if (idx < kindsAttr.size())
          filteredKinds.push_back(kindsAttr[idx]);
      }
      if (!filteredKinds.empty())
        newReductionKinds = ArrayAttr::get(ctx, filteredKinds);
    }

    OpBuilder rebuilder(suIter);
    auto newSuIter = sde::SdeSuIterateOp::create(
        rebuilder, loc, newResultTypes, suIter.getLowerBounds(),
        suIter.getUpperBounds(), suIter.getSteps(), suIter.getScheduleAttr(),
        suIter.getChunkSize(),
        suIter.getNowaitAttr() ? UnitAttr::get(ctx) : nullptr, realAccValues,
        newReductionKinds, suIter.getReductionStrategyAttr(),
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

    for (auto attr : attrs)
      newSuIter->setAttr(attr.getName(), attr.getValue());

    newSuIter.getBody().takeBody(suIter.getBody());
    Block &newBody = newSuIter.getBody().front();

    // Erase tensor block args (reverse order to keep indices valid).
    for (auto it = tensorAccIndices.rbegin(); it != tensorAccIndices.rend();
         ++it) {
      unsigned argIdx = numIVs + *it;
      if (argIdx < newBody.getNumArguments())
        newBody.eraseArgument(argIdx);
    }

    // Fix yield: keep only real accumulator values.
    if (auto yield =
            dyn_cast_or_null<sde::SdeYieldOp>(newBody.getTerminator())) {
      SmallVector<Value> newYieldValues;
      for (unsigned idx : realAccIndices) {
        if (idx < yield.getValues().size())
          newYieldValues.push_back(yield.getValues()[idx]);
      }
      OpBuilder yieldBuilder(yield);
      sde::SdeYieldOp::create(yieldBuilder, loc, newYieldValues);
      yield.erase();
    }

    // Replace kept results.
    for (auto [newIdx, oldIdx] : llvm::enumerate(realAccIndices)) {
      if (oldIdx < suIter.getNumResults() && newIdx < newSuIter.getNumResults())
        suIter.getResult(oldIdx).replaceAllUsesWith(
            newSuIter.getResult(newIdx));
    }

    suIter.erase();
  }
}

/// After tensor iter_args are stripped, <single> regions contain explicit
/// mu_memref_to_tensor anchors for their token-backed state. Revisit those
/// bodies so tensor.extract/tensor.insert become concrete memref.load/store
/// side effects before cleanup, including rank-0 scalar carriers.
static void lowerSingleRegionTensorOps(ModuleOp module,
                                       DenseMap<Value, Value> &backingMemref) {
  SmallVector<sde::SdeCuRegionOp> singleRegions;
  module.walk([&](sde::SdeCuRegionOp cuRegion) {
    if (cuRegion.getKind() == sde::SdeCuKind::single)
      singleRegions.push_back(cuRegion);
  });

  for (sde::SdeCuRegionOp cuRegion : singleRegions) {
    if (cuRegion.getBody().empty())
      continue;

    Block &body = cuRegion.getBody().front();
    OpBuilder builder(cuRegion);
    bool hasTensorState = false;
    for (auto [i, arg] : llvm::enumerate(cuRegion.getIterArgs())) {
      auto tensorTy = dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorTy)
        continue;
      if (i >= body.getNumArguments())
        continue;
      Value backing = getOrCreateBacking(arg, builder, backingMemref);
      backingMemref[body.getArgument(i)] = backing;
      hasTensorState = true;
    }

    body.walk([&](sde::SdeMuMemrefToTensorOp op) {
      auto tensorTy = dyn_cast<RankedTensorType>(op.getTensor().getType());
      if (!tensorTy)
        return;
      markPreserveIfAlloca(op.getMemref());
      backingMemref[op.getTensor()] = op.getMemref();
      hasTensorState = true;
    });

    if (hasTensorState)
      walkBlock(body, backingMemref);
  }
}

//===----------------------------------------------------------------------===//
// Phase C: Lower authoritative carriers to scf.for loops
//===----------------------------------------------------------------------===//

/// Trace a tensor value back to a memref. Walks through
/// bufferization.to_tensor, mu_memref_to_tensor, tensor.cast,
/// tensor.extract_slice, and tensor.empty chains. For extract_slice, creates a
/// memref.subview. For tensor.empty, creates a memref.alloca as backing
/// storage.
static Value traceToMemref(Value tensor, OpBuilder &builder) {
  if (auto toTensor = tensor.getDefiningOp<bufferization::ToTensorOp>())
    return toTensor.getBuffer();
  if (auto muToTensor = tensor.getDefiningOp<sde::SdeMuMemrefToTensorOp>())
    return muToTensor.getMemref();
  if (auto castOp = tensor.getDefiningOp<tensor::CastOp>())
    return traceToMemref(castOp.getSource(), builder);
  if (auto sliceOp = tensor.getDefiningOp<tensor::ExtractSliceOp>()) {
    Value srcMemref = traceToMemref(sliceOp.getSource(), builder);
    if (!srcMemref)
      return Value();
    // Create a memref.subview matching the extract_slice. Let MLIR infer
    // the correct strided result type from the source memref and offsets.
    auto srcMemrefTy = cast<MemRefType>(srcMemref.getType());
    auto resultMemrefTy = memref::SubViewOp::inferResultType(
        srcMemrefTy, sliceOp.getStaticOffsets(), sliceOp.getStaticSizes(),
        sliceOp.getStaticStrides());
    return memref::SubViewOp::create(
        builder, sliceOp.getLoc(), cast<MemRefType>(resultMemrefTy), srcMemref,
        sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
        sliceOp.getMixedStrides());
  }
  if (auto emptyOp = tensor.getDefiningOp<tensor::EmptyOp>()) {
    auto tensorTy = cast<RankedTensorType>(emptyOp.getType());
    auto memrefTy = getMemRefType(tensorTy);
    return memref::AllocaOp::create(builder, emptyOp.getLoc(), memrefTy,
                                    emptyOp.getDynamicSizes());
  }
  return Value();
}

/// Lower a single tensor-backed linalg.generic carrier to scf.for loops
/// with memref.load/memref.store. Also erases associated tensor.extract_slice
/// and tensor.insert_slice ops from tiled carriers. Returns true on success.
static bool lowerCarrierToLoops(linalg::GenericOp carrier) {
  // Only handle tensor-backed carriers.
  if (carrier.hasPureBufferSemantics())
    return false;
  if (!llvm::all_of(carrier.getDpsInputs(),
                    [](Value v) { return isa<TensorType>(v.getType()); }))
    return false;
  if (!llvm::all_of(carrier.getDpsInits(),
                    [](Value v) { return isa<TensorType>(v.getType()); }))
    return false;

  OpBuilder builder(carrier);
  Location loc = carrier.getLoc();

  // Trace all operands back to memrefs (handles extract_slice → subview).
  SmallVector<Value> memrefInputs;
  for (Value input : carrier.getDpsInputs()) {
    Value memref = traceToMemref(input, builder);
    if (!memref) {
      ARTS_DEBUG("lowerCarrierToLoops: cannot trace input to memref");
      return false;
    }
    memrefInputs.push_back(memref);
  }

  SmallVector<Value> memrefOutputs;
  for (Value output : carrier.getDpsInits()) {
    Value memref = traceToMemref(output, builder);
    if (!memref) {
      ARTS_DEBUG("lowerCarrierToLoops: cannot trace output to memref");
      return false;
    }
    memrefOutputs.push_back(memref);
  }

  // Create an equivalent memref-backed linalg.generic.
  auto indexingMaps = carrier.getIndexingMapsArray();
  auto iterTypes = carrier.getIteratorTypesArray();

  auto memrefGeneric = linalg::GenericOp::create(
      builder, loc, /*resultTypes=*/TypeRange{}, memrefInputs, memrefOutputs,
      indexingMaps, iterTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        // Clone the carrier's body, mapping old block args to new ones.
        IRMapping mapper;
        Block &carrierBody = carrier.getRegion().front();
        for (auto [oldArg, newArg] :
             llvm::zip(carrierBody.getArguments(), blockArgs))
          mapper.map(oldArg, newArg);

        for (Operation &op : carrierBody.without_terminator())
          nestedBuilder.clone(op, mapper);

        // Map yield values.
        auto carrierYield = cast<linalg::YieldOp>(carrierBody.getTerminator());
        SmallVector<Value> yieldValues;
        for (Value v : carrierYield.getValues())
          yieldValues.push_back(mapper.lookupOrDefault(v));
        linalg::YieldOp::create(nestedBuilder, nestedLoc, yieldValues);
      });

  // Lower the memref-backed generic to scf.for loops.
  IRRewriter rewriter(carrier.getContext());
  rewriter.setInsertionPoint(memrefGeneric);
  auto result = linalg::linalgOpToLoops(rewriter, memrefGeneric);
  if (failed(result)) {
    ARTS_DEBUG("lowerCarrierToLoops: linalgOpToLoops failed");
    memrefGeneric.erase();
    return false;
  }

  // Erase tensor.insert_slice ops that use the carrier's results.
  for (Value carrierResult : carrier.getResults()) {
    for (Operation *user :
         llvm::make_early_inc_range(carrierResult.getUsers())) {
      if (isa<tensor::InsertSliceOp>(user))
        user->erase();
    }
  }

  // linalgOpToLoops erases the op on success, no need to erase memrefGeneric.
  ARTS_DEBUG("lowerCarrierToLoops: lowered carrier with "
             << carrier.getNumDpsInputs() << " inputs, "
             << carrier.getNumDpsInits() << " outputs");
  return true;
}

/// Lower all authoritative linalg.generic carriers inside SDE regions.
/// A carrier is "authoritative" if its parent su_iterate/cu_region body
/// contains NO scalar executable ops (memref.load/memref.store) alongside
/// the carrier — meaning the carrier IS the computation.
static unsigned lowerAuthoritativeCarriers(ModuleOp module) {
  SmallVector<linalg::GenericOp> carriers;
  module.walk([&](linalg::GenericOp generic) {
    // Only consider carriers inside SDE regions.
    if (!generic->getParentOfType<sde::SdeSuIterateOp>() &&
        !generic->getParentOfType<sde::SdeCuRegionOp>())
      return;
    // Only consider tensor-backed carriers.
    if (generic.hasPureBufferSemantics())
      return;
    // Check if authoritative: parent block has no scalar body ops
    // (memref.load/store). Carrier support ops (mu_memref_to_tensor,
    // extract_slice, insert_slice, arith for tile math) are allowed.
    Block *parent = generic->getBlock();
    bool hasScalarBody = false;
    for (Operation &op : parent->without_terminator()) {
      if (isa<linalg::GenericOp, bufferization::ToTensorOp,
              sde::SdeMuMemrefToTensorOp, tensor::EmptyOp, tensor::CastOp,
              tensor::ExtractSliceOp, tensor::InsertSliceOp>(&op))
        continue;
      if (isa<memref::LoadOp, memref::StoreOp>(&op)) {
        hasScalarBody = true;
        break;
      }
      // Allow pure ops (arith for tile size computation, etc.)
      if (isMemoryEffectFree(&op))
        continue;
      // Unknown side-effecting op — not authoritative.
      hasScalarBody = true;
      break;
    }
    // Only lower authoritative carriers (no coexisting scalar body).
    if (!hasScalarBody)
      carriers.push_back(generic);
  });

  unsigned lowered = 0;
  for (linalg::GenericOp carrier : carriers) {
    if (lowerCarrierToLoops(carrier)) {
      carrier.erase();
      ++lowered;
    }
  }
  return lowered;
}

//===----------------------------------------------------------------------===//
// Phase D: Erase dead carriers
//===----------------------------------------------------------------------===//

static void eraseDeadCarriers(ModuleOp module) {
  // Iteratively erase dead tensor/linalg/sde-mu ops.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    module.walk([&](Operation *op) {
      if (isa<linalg::GenericOp, bufferization::ToTensorOp, tensor::EmptyOp,
              tensor::CastOp, tensor::ExtractOp, tensor::InsertOp,
              tensor::ExtractSliceOp, tensor::InsertSliceOp,
              vector::TransferReadOp, vector::TransferWriteOp,
              sde::SdeMuAllocOp, sde::SdeMuMemrefToTensorOp,
              sde::SdeMuTensorToMemrefOp>(op)) {
        if (op->use_empty())
          dead.push_back(op);
      }
    });
    for (Operation *op : llvm::reverse(dead)) {
      op->erase();
      changed = true;
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct LowerToMemrefPass
    : public arts::impl::LowerToMemrefBase<LowerToMemrefPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Phase A: Walk each function, rewrite tensor ops to memref ops,
    //          build backing memref map.
    unsigned totalLowered = 0;
    DenseMap<Value, Value> backingMemref;
    module.walk([&](func::FuncOp func) {
      if (func.getBody().empty())
        return;
      unsigned before = backingMemref.size();
      walkBlock(func.getBody().front(), backingMemref);
      totalLowered += backingMemref.size() - before;
    });

    // Phase B: Strip tensor iter_args.
    // Order: cu_regions first (innermost via post-order walk), then
    // su_iterates, then scf.for (outermost).
    stripCuRegionTensorArgs(module, backingMemref);
    stripSuIterateTensorArgs(module, backingMemref);
    stripScfForTensorArgs(module, backingMemref);
    lowerSingleRegionTensorOps(module, backingMemref);

    // Phase C: Lower authoritative carriers to scf.for loops.
    unsigned carriersLowered = lowerAuthoritativeCarriers(module);

    // Phase D: Erase remaining dead carriers.
    eraseDeadCarriers(module);

    ARTS_INFO("LowerToMemref: lowered "
              << totalLowered << " tensor value(s) to memref, "
              << carriersLowered << " carrier(s) to loops");
  }
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createLowerToMemrefPass() {
  return std::make_unique<LowerToMemrefPass>();
}

} // namespace mlir::carts::sde
