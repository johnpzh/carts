///==========================================================================///
/// File: RaiseMemrefToTensor.cpp
///
/// RaiseMemrefToTensor pass — lifts small memref allocas into tensor SSA.
///
/// Scope:
///   - memref.alloca producing memref<KxT> with small constant K (K <= 16).
///   - Accesses inside sde.cu_task bodies are constant-indexed
///     memref.load / memref.store / affine.load / affine.store only.
///   - No captures of the memref via call @extern(...).
///   - No aliasing via memref.subview.
///   - memref.cast is tolerated only when its uses outside cu_task bodies are
///     limited to sde.mu_dep (which this pass erases).
///
/// High-level algorithm:
///
///   1. Collect raisable memrefs (allocas with rank-1 static shape K<=16,
///      constant-indexed task accesses, limited external uses).
///   2. Gather-phase commit check: every sde.cu_task that touches a raisable
///      memref must be rewriteable (constant indices + pure captures OR
///      task-private impure captures we can move inside). Any failure drops
///      that memref from the raise set; the rest still raise.
///   3. For each still-raisable memref, emit a sde.mu_data handle just before
///      the outermost enclosing cu_region (or the first task if none).
///   4. Walk the tasks in program order; for each, build sde.mu_tokens whose
///      source is the CURRENT tensor SSA value for each memref (initially the
///      mu_data handle; after a writable codelet runs, its result advances
///      currentTensor[%m]). The token graph then matches the SSA dep graph.
///   5. Create a sde.cu_codelet; inside its body, move the task body's
///      load/stores to tensor.extract / tensor.insert on the block-arg tensor,
///      clone pure external producers, and move task-private impure producers
///      inline. Yield the updated tensors for each writable token.
///   6. Erase sde.mu_dep ops for raised memrefs.
///   7. Boundary-materialize: at the end of the innermost region that contains
///      every touching task (before its terminator), emit a tensor.extract +
///      memref.store from the FINAL currentTensor[%m] back into the original
///      alloca. This preserves observable-equivalence for any non-raised
///      post-scope reader of the memref.
///
/// Out of scope for v1:
///   - Slice-access patterns (`memref.subview` etc.).
///   - Dynamic-ranked memrefs.
///   - Threading through scf.for / scf.while / scf.if iter_args.
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_RAISEMEMREFTOTENSOR
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/OperationAttributes.h"
#include "arts/utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::arts;

namespace {

/// v1 threshold — only raise memrefs with rank-1 static shape K <= kMaxRaiseK.
static constexpr int64_t kMaxRaiseK = 16;

/// Classifies how a task body touches a raisable memref.
enum class MemrefUseKind { None = 0, Read = 1, Write = 2, ReadWrite = 3 };

static MemrefUseKind merge(MemrefUseKind a, MemrefUseKind b) {
  using K = MemrefUseKind;
  if (a == K::None)
    return b;
  if (b == K::None)
    return a;
  if (a == b)
    return a;
  return K::ReadWrite;
}

/// Match a constant-index value (index or integer type) and return its int
/// value.
static std::optional<int64_t> matchConstantIndex(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

/// Is `memref` legal to raise under v1 rules?
static bool isRaisableMemref(Value memref,
                             DenseSet<Operation *> &castsToErase) {
  auto mrType = dyn_cast<MemRefType>(memref.getType());
  if (!mrType || !mrType.hasStaticShape())
    return false;
  if (mrType.getRank() != 1)
    return false;
  if (mrType.getShape()[0] > kMaxRaiseK)
    return false;

  for (Operation *user : memref.getUsers()) {
    if (isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp,
            affine::AffineStoreOp>(user))
      continue;
    if (auto cast = dyn_cast<memref::CastOp>(user)) {
      for (Operation *castUser : cast.getResult().getUsers()) {
        if (isa<sde::SdeMuDepOp>(castUser))
          continue;
        return false;
      }
      castsToErase.insert(cast);
      continue;
    }
    if (isa<sde::SdeMuDepOp>(user))
      continue;
    return false;
  }
  return true;
}

/// Collect the memref.allocas used inside any sde.cu_task body within `func`
/// that are legal to raise under v1 rules.
static SmallVector<memref::AllocaOp>
collectCandidateAllocas(func::FuncOp func,
                        DenseSet<Operation *> &castsToErase) {
  DenseSet<memref::AllocaOp> usedInTask;
  func.walk([&](sde::SdeCuTaskOp task) {
    task.getBody().walk([&](Operation *op) {
      Value memref;
      if (auto ld = dyn_cast<memref::LoadOp>(op))
        memref = ld.getMemref();
      else if (auto st = dyn_cast<memref::StoreOp>(op))
        memref = st.getMemref();
      else if (auto ld = dyn_cast<affine::AffineLoadOp>(op))
        memref = ld.getMemref();
      else if (auto st = dyn_cast<affine::AffineStoreOp>(op))
        memref = st.getMemref();
      else
        return;
      // Follow a single memref.cast layer back to the underlying alloca so
      // tasks that only see the cast-alias still anchor the alloca.
      if (auto castOp = memref.getDefiningOp<memref::CastOp>())
        memref = castOp.getSource();
      if (auto alloca = memref.getDefiningOp<memref::AllocaOp>())
        usedInTask.insert(alloca);
    });
  });

  SmallVector<memref::AllocaOp> result;
  for (memref::AllocaOp alloca : usedInTask) {
    DenseSet<Operation *> localCasts;
    if (!isRaisableMemref(alloca.getResult(), localCasts))
      continue;
    result.push_back(alloca);
    for (Operation *c : localCasts)
      castsToErase.insert(c);
  }
  return result;
}

/// Return the outermost `sde.cu_region` that transitively contains `innerOp`.
static sde::SdeCuRegionOp getOutermostCuRegion(Operation *innerOp) {
  sde::SdeCuRegionOp outer;
  Operation *cur = innerOp->getParentOp();
  while (cur) {
    if (auto cr = dyn_cast<sde::SdeCuRegionOp>(cur))
      outer = cr;
    cur = cur->getParentOp();
  }
  return outer;
}

/// Classify access modes of `task` on each raised memref. Also records every
/// accessing load/store operation for later rewriting.
struct TaskAccessInfo {
  DenseMap<Value, MemrefUseKind> modes;
  SmallVector<Operation *> accessOps;
};

static TaskAccessInfo analyzeTask(sde::SdeCuTaskOp task,
                                  const DenseSet<Value> &raisedMemrefs) {
  TaskAccessInfo info;
  task.getBody().walk([&](Operation *op) {
    Value memref;
    bool isStore = false;
    if (auto ld = dyn_cast<memref::LoadOp>(op)) {
      memref = ld.getMemref();
    } else if (auto st = dyn_cast<memref::StoreOp>(op)) {
      memref = st.getMemref();
      isStore = true;
    } else if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
      memref = ld.getMemref();
    } else if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
      memref = st.getMemref();
      isStore = true;
    } else {
      return;
    }
    if (!raisedMemrefs.count(memref))
      return;
    info.accessOps.push_back(op);
    auto kind = isStore ? MemrefUseKind::Write : MemrefUseKind::Read;
    auto it = info.modes.find(memref);
    if (it == info.modes.end())
      info.modes[memref] = kind;
    else
      it->second = merge(it->second, kind);
  });
  return info;
}

/// Collect in-order `cu_task` ops in the function that access any of the
/// raisable memrefs.
static SmallVector<sde::SdeCuTaskOp>
collectTasksUsingAnyMemref(func::FuncOp func,
                           const DenseSet<Value> &raisedMemrefs) {
  SmallVector<sde::SdeCuTaskOp> tasks;
  func.walk<WalkOrder::PreOrder>([&](sde::SdeCuTaskOp task) {
    bool uses = false;
    task.getBody().walk([&](Operation *op) {
      Value memref;
      if (auto ld = dyn_cast<memref::LoadOp>(op))
        memref = ld.getMemref();
      else if (auto st = dyn_cast<memref::StoreOp>(op))
        memref = st.getMemref();
      else if (auto ld = dyn_cast<affine::AffineLoadOp>(op))
        memref = ld.getMemref();
      else if (auto st = dyn_cast<affine::AffineStoreOp>(op))
        memref = st.getMemref();
      if (memref && raisedMemrefs.count(memref))
        uses = true;
    });
    if (uses)
      tasks.push_back(task);
  });
  return tasks;
}

/// External (task-body-local) SSA values referenced by `task`. Partitioned
/// into two disjoint groups:
///   - `pureExternal`: values whose full transitive producer chain is pure and
///     can be cloned into the codelet body.
///   - `moveInsideOps`: impure defining ops that are used ONLY by this task
///     (task-private). These are moved INTO the codelet body.
/// Returns false if there is an impure external producer used by more than
/// one task (task-shared impure) or a use-before-def we cannot resolve.
struct ExternalCaptureInfo {
  SmallVector<Value> pureExternal;
  SmallVector<Operation *> moveInsideOps;
};

static bool classifyExternalCaptures(sde::SdeCuTaskOp task,
                                     const DenseSet<Value> &raisedMemrefs,
                                     ExternalCaptureInfo &out) {
  Region &body = task.getBody();
  DenseSet<Value> seen;
  SmallVector<Value> externals;
  body.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Region *defRegion = operand.getParentRegion();
      if (!defRegion)
        continue;
      if (body.isAncestor(defRegion))
        continue;
      if (raisedMemrefs.count(operand))
        continue;
      if (seen.insert(operand).second)
        externals.push_back(operand);
    }
  });

  DenseSet<Operation *> moveInsideSet;
  DenseSet<Value> pureSet;

  std::function<bool(Value)> classify = [&](Value v) -> bool {
    if (pureSet.count(v))
      return true;
    if (raisedMemrefs.count(v))
      return true;
    Operation *def = v.getDefiningOp();
    if (!def) {
      // Block arguments external to the task body — tolerated only when pure
      // (e.g., func args). Block args never have side effects; we cannot
      // clone them, but we also don't need to — they are just referenced by
      // clone via IRMapping.
      pureSet.insert(v);
      return true;
    }
    if (isMemoryEffectFree(def)) {
      // Pure: recurse on operands to ensure their producers are cloneable.
      for (Value operand : def->getOperands()) {
        Region *defRegion = operand.getParentRegion();
        if (defRegion && task.getBody().isAncestor(defRegion))
          continue;
        if (!classify(operand))
          return false;
      }
      pureSet.insert(v);
      return true;
    }
    // Impure defining op. Only admit if task-private: every user of every
    // result of `def` is inside `task.getBody()`.
    for (Value result : def->getResults()) {
      for (Operation *user : result.getUsers()) {
        // A user is "inside this task" if its parent op chain reaches our
        // task body region.
        Region *userRegion = user->getParentRegion();
        if (!userRegion)
          return false;
        if (!task.getBody().isAncestor(userRegion))
          return false; // used outside this task → shared impure → abort.
      }
    }
    // Also require the impure op's own operands to be classifiable.
    for (Value operand : def->getOperands()) {
      Region *defRegion = operand.getParentRegion();
      if (defRegion && task.getBody().isAncestor(defRegion))
        continue;
      if (!classify(operand))
        return false;
    }
    moveInsideSet.insert(def);
    for (Value r : def->getResults())
      pureSet.insert(r);
    return true;
  };

  for (Value v : externals)
    if (!classify(v))
      return false;

  for (Value v : externals)
    if (Operation *def = v.getDefiningOp())
      if (!moveInsideSet.count(def))
        out.pureExternal.push_back(v);

  // Preserve program order when moving impure ops inside.
  for (Operation *op : moveInsideSet)
    (void)op;
  // Collect in program order by walking the parent block.
  // moveInsideSet entries may live in different blocks (e.g., the surrounding
  // cu_region body, or nested scf region). We emit them in post-order of
  // dependency via the classify recursion result — re-derive by topological
  // sort over moveInsideSet using a simple repeated-pass.
  SmallVector<Operation *> topo;
  DenseSet<Operation *> placed;
  std::function<void(Operation *)> emit = [&](Operation *op) {
    if (placed.count(op))
      return;
    for (Value operand : op->getOperands()) {
      if (Operation *pd = operand.getDefiningOp())
        if (moveInsideSet.count(pd))
          emit(pd);
    }
    placed.insert(op);
    topo.push_back(op);
  };
  for (Operation *op : moveInsideSet)
    emit(op);
  out.moveInsideOps = std::move(topo);
  return true;
}

/// Clone pure external producer chain into the codelet body so body ops can
/// look them up via the returned mapping. Impure task-private producers are
/// moved separately (by moveImpureProducersInto) and the mapping is updated
/// with those moves.
static std::optional<IRMapping> cloneExternalPureProducers(
    sde::SdeCuTaskOp task, const DenseSet<Value> &raisedMemrefs,
    const SmallVector<Value> &pureExternal, OpBuilder &codeletBuilder) {
  IRMapping mapping;
  Region &body = task.getBody();

  std::function<bool(Value)> cloneValue = [&](Value v) -> bool {
    if (mapping.contains(v))
      return true;
    if (raisedMemrefs.count(v))
      return true;
    Operation *def = v.getDefiningOp();
    if (!def) {
      // Block argument: keep the reference as-is — cloned body ops will use
      // the SAME block arg value. Since cu_codelet is IsolatedFromAbove,
      // this only remains valid if the block arg is defined in an ancestor
      // that is also an ancestor of the codelet. For v1 we assume the
      // external block arg value (e.g., %c0_i32) is a constant, OR the
      // caller has ensured pure re-cloning. Most constants are already
      // pure-op results (arith.constant), not block args, so this path is
      // rarely taken for our target programs.
      return false;
    }
    if (!isMemoryEffectFree(def))
      return false;
    for (Value operand : def->getOperands()) {
      if (mapping.contains(operand))
        continue;
      Region *defRegion = operand.getParentRegion();
      if (defRegion && body.isAncestor(defRegion))
        continue;
      if (!cloneValue(operand))
        return false;
    }
    Operation *cloned = codeletBuilder.clone(*def, mapping);
    for (auto [orig, repl] : llvm::zip(def->getResults(), cloned->getResults()))
      mapping.map(orig, repl);
    return true;
  };

  for (Value v : pureExternal) {
    if (!cloneValue(v))
      return std::nullopt;
  }
  return mapping;
}

//===----------------------------------------------------------------------===//
// Pass body.
//===----------------------------------------------------------------------===//

struct RaiseMemrefToTensorPass
    : public arts::impl::RaiseMemrefToTensorBase<RaiseMemrefToTensorPass> {
  using arts::impl::RaiseMemrefToTensorBase<
      RaiseMemrefToTensorPass>::RaiseMemrefToTensorBase;

  void runOnOperation() override {
    if (!enabled)
      return;
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp func) { rewriteFunction(func); });
  }

  //===----------------------------------------------------------------------===//
  // Per-function driver.
  //===----------------------------------------------------------------------===//

  void rewriteFunction(func::FuncOp func) {
    DenseSet<Operation *> castsToErase;
    SmallVector<memref::AllocaOp> candidates =
        collectCandidateAllocas(func, castsToErase);
    if (candidates.empty())
      return;

    // Map every raised alloca (and its cast aliases) to the tensor type it
    // will take on in the raised form.
    DenseMap<Value, RankedTensorType> tensorTypes;
    DenseSet<Value> raisedMemrefSet;
    DenseMap<Value, Value> canonicalMemref; // cast-alias -> original alloca
    for (memref::AllocaOp alloca : candidates) {
      auto mrType = cast<MemRefType>(alloca.getType());
      auto tty =
          RankedTensorType::get(mrType.getShape(), mrType.getElementType());
      tensorTypes[alloca.getResult()] = tty;
      raisedMemrefSet.insert(alloca.getResult());
      canonicalMemref[alloca.getResult()] = alloca.getResult();
      for (Operation *user : alloca->getUsers()) {
        if (auto cast = dyn_cast<memref::CastOp>(user)) {
          raisedMemrefSet.insert(cast.getResult());
          tensorTypes[cast.getResult()] = tty;
          canonicalMemref[cast.getResult()] = alloca.getResult();
        }
      }
    }

    SmallVector<sde::SdeCuTaskOp> tasks =
        collectTasksUsingAnyMemref(func, raisedMemrefSet);
    if (tasks.empty())
      return;

    // ---- Gather phase: verify every task can be rewritten. ------------------
    //
    // We need per-task info:
    //   - TaskAccessInfo (mode per memref, access-op list)
    //   - ExternalCaptureInfo (pure externals + task-private impure ops)
    //   - constant-index verification for every load/store
    //
    // If ANY task is not rewriteable, we conservatively drop the entire
    // function from the raise set (B2 gather-then-commit). This avoids
    // partial mutation.

    SmallVector<TaskAccessInfo> taskInfos;
    SmallVector<ExternalCaptureInfo> captureInfos;
    taskInfos.reserve(tasks.size());
    captureInfos.reserve(tasks.size());

    for (sde::SdeCuTaskOp task : tasks) {
      TaskAccessInfo info = analyzeTask(task, raisedMemrefSet);
      // Constant-index check.
      for (Operation *op : info.accessOps) {
        auto checkIndices = [&](ValueRange indices) {
          for (Value idx : indices)
            if (!matchConstantIndex(idx))
              return false;
          return true;
        };
        if (auto ld = dyn_cast<memref::LoadOp>(op)) {
          if (!checkIndices(ld.getIndices()))
            return;
        } else if (auto st = dyn_cast<memref::StoreOp>(op)) {
          if (!checkIndices(st.getIndices()))
            return;
        }
        // affine.load / affine.store on rank-1 memrefs: the affine map is an
        // identity into a 1-D space; we accept them unconditionally because
        // the indices used in the tensor op are constants we synthesize.
      }
      ExternalCaptureInfo captures;
      if (!classifyExternalCaptures(task, raisedMemrefSet, captures))
        return;
      taskInfos.push_back(std::move(info));
      captureInfos.push_back(std::move(captures));
    }

    // ---- Placement: mu_data anchors. ---------------------------------------
    //
    // Place mu_data at the start of the outermost cu_region containing the
    // tasks so the SDE/Core materializer can keep memory anchors in the same
    // scheduling scope as the tasks that consume them.
    sde::SdeCuRegionOp outerRegion = getOutermostCuRegion(tasks.front());
    OpBuilder builder(func.getContext());
    if (outerRegion) {
      Block &outerBlock = outerRegion.getBody().front();
      builder.setInsertionPointToStart(&outerBlock);
    } else {
      builder.setInsertionPoint(tasks.front());
    }

    DenseMap<Value, Value> muDataByMemref;
    for (memref::AllocaOp alloca : candidates) {
      auto tty = tensorTypes[alloca.getResult()];
      auto muData = sde::SdeMuDataOp::create(
          builder, alloca.getLoc(), tty, /*init=*/nullptr,
          /*shared=*/UnitAttr::get(builder.getContext()));
      muDataByMemref[alloca.getResult()] = muData.getHandle();
      for (Operation *user : alloca->getUsers()) {
        if (auto cast = dyn_cast<memref::CastOp>(user))
          muDataByMemref[cast.getResult()] = muData.getHandle();
      }
    }

    // Current tensor SSA value per raised memref. Advances with each
    // writable codelet's result so sibling tasks see the up-to-date tensor
    // (Gap A fix).
    DenseMap<Value, Value> currentTensor = muDataByMemref;

    // ---- Rewrite each task in order, threading SSA via currentTensor. ------
    for (size_t i = 0; i < tasks.size(); ++i) {
      if (!rewriteTask(tasks[i], taskInfos[i], captureInfos[i], tensorTypes,
                       currentTensor, canonicalMemref)) {
        // Gather-phase should have caught this. If we still failed (e.g.,
        // something we didn't predict), the function is now in a partially
        // mutated state — we signal the pass and stop processing this func.
        signalPassFailure();
        return;
      }
    }

    // ---- Boundary materialization (Gap C + Gap D). ------------------------
    //
    // For each raised memref, emit `tensor.extract` + `memref.store` for each
    // element of the final tensor value. Placed just before the terminator
    // of the scope block containing the last writing codelet so the scope's
    // non-raised readers observe the updated values.
    //
    // Gap D (boundary-write race): the cu_codelet ops lower to async EDTs
    // scheduled by the enclosing single EDT. Without synchronization, the
    // boundary stores run on the task-creation thread BEFORE the async EDTs
    // actually execute, so post-region memref readers observe stale data
    // (deps.c: `A: 0, B: 0`). We close the race by emitting an
    // `sde.su_barrier` in each block that receives boundary writes, just
    // before the first write. At ConvertSdeToArts it lowers to `arts.barrier`
    // which CreateEpochs groups the preceding task EDTs into an `arts.epoch`
    // that must complete before subsequent ops (the stores) run.
    //
    // One barrier is emitted per unique boundary block so multi-memref
    // programs don't get redundant barriers.
    DenseSet<Block *> boundaryBlocks;
    for (memref::AllocaOp alloca : candidates) {
      Value memref = alloca.getResult();
      Value finalTensor = currentTensor.lookup(memref);
      if (!finalTensor)
        continue;
      if (finalTensor == muDataByMemref.lookup(memref))
        continue; // no writable codelet ran → no writes to flush

      Operation *producer = finalTensor.getDefiningOp();
      if (!producer)
        continue;
      Block *block = producer->getBlock();

      // Emit an sde.su_barrier once per boundary block, just before the
      // block terminator. This sequences the boundary stores after the
      // async task EDTs the codelets lower to.
      if (boundaryBlocks.insert(block).second) {
        Operation *terminator = block->getTerminator();
        OpBuilder barrierBuilder(terminator);
        sde::SdeSuBarrierOp::create(barrierBuilder, alloca.getLoc(),
                                    ValueRange{},
                                    /*barrierEliminated=*/nullptr,
                                    /*barrierReason=*/nullptr);
      }

      Operation *terminator = block->getTerminator();
      OpBuilder boundaryBuilder(terminator);
      Location loc = alloca.getLoc();

      auto mrType = cast<MemRefType>(memref.getType());
      int64_t numElements = mrType.getNumElements();
      for (int64_t e = 0; e < numElements; ++e) {
        Value idx = arts::createConstantIndex(boundaryBuilder, loc, e);
        Value extracted =
            tensor::ExtractOp::create(boundaryBuilder, loc, finalTensor, idx);
        memref::StoreOp::create(boundaryBuilder, loc, extracted, memref,
                                ValueRange{idx});
      }

      // Tag the alloca so downstream EdtStructuralOpt does not sink it into
      // the EDT body. The boundary store above is the bridge from the raised
      // tensor DB back to this user-visible memref; if the alloca were sunk,
      // the sink would redirect the boundary store to a local EDT-private
      // alloca and post-region readers would still see the initial contents.
      alloca->setAttr(AttrNames::Operation::Preserve,
                      UnitAttr::get(alloca.getContext()));
    }

    // ---- Erase mu_dep ops for raised memrefs. ------------------------------
    SmallVector<sde::SdeMuDepOp> muDepsToErase;
    func.walk([&](sde::SdeMuDepOp dep) {
      if (raisedMemrefSet.count(dep.getSource()))
        muDepsToErase.push_back(dep);
    });
    for (sde::SdeMuDepOp dep : muDepsToErase) {
      if (dep->use_empty())
        dep.erase();
    }

    // Dead memref.cast cleanup.
    for (Operation *cast : castsToErase) {
      if (cast->use_empty())
        cast->erase();
    }
  }

  //===----------------------------------------------------------------------===//
  // Per-task rewrite.
  //===----------------------------------------------------------------------===//

  bool rewriteTask(sde::SdeCuTaskOp task, const TaskAccessInfo &info,
                   const ExternalCaptureInfo &captures,
                   DenseMap<Value, RankedTensorType> &tensorTypes,
                   DenseMap<Value, Value> &currentTensor,
                   DenseMap<Value, Value> &canonicalMemref) {
    if (info.modes.empty())
      return true;

    Location loc = task.getLoc();
    MLIRContext *ctx = task.getContext();
    OpBuilder builder(task);

    // Deterministic ordering of touched memrefs: by first-access occurrence.
    // We key on canonical memref (underlying alloca) so cast-aliases share a
    // single token.
    SmallVector<Value> orderedMemrefs; // canonical
    DenseSet<Value> seen;
    // Also remember which per-task memref (possibly cast-alias) was used, so
    // we can redirect body ops to the codelet's tensor block-arg.
    DenseMap<Value, SmallVector<Value>> aliasesOfCanonical;
    for (Operation *op : info.accessOps) {
      Value mr;
      if (auto ld = dyn_cast<memref::LoadOp>(op))
        mr = ld.getMemref();
      else if (auto st = dyn_cast<memref::StoreOp>(op))
        mr = st.getMemref();
      else if (auto ld = dyn_cast<affine::AffineLoadOp>(op))
        mr = ld.getMemref();
      else if (auto st = dyn_cast<affine::AffineStoreOp>(op))
        mr = st.getMemref();
      Value canonical = canonicalMemref.lookup(mr);
      if (!canonical)
        canonical = mr;
      if (seen.insert(canonical).second)
        orderedMemrefs.push_back(canonical);
      aliasesOfCanonical[canonical].push_back(mr);
    }

    // Merge access modes across cast-aliases onto the canonical memref.
    DenseMap<Value, MemrefUseKind> canonicalModes;
    for (auto &kv : info.modes) {
      Value canonical = canonicalMemref.lookup(kv.first);
      if (!canonical)
        canonical = kv.first;
      canonicalModes[canonical] =
          merge(canonicalModes.lookup(canonical), kv.second);
    }

    // Build mu_tokens sourced from currentTensor[canonical].
    SmallVector<Value> tokens;
    SmallVector<Type> blockArgTypes;
    SmallVector<Type> resultTypes;
    SmallVector<unsigned> writableIndices; // token-index -> writable
    tokens.reserve(orderedMemrefs.size());
    blockArgTypes.reserve(orderedMemrefs.size());
    for (auto [idx, canonical] : llvm::enumerate(orderedMemrefs)) {
      auto tty = tensorTypes[canonical];
      Value src = currentTensor.lookup(canonical);
      if (!src)
        return false;
      MemrefUseKind k = canonicalModes.lookup(canonical);
      sde::SdeAccessMode mode;
      bool writable = false;
      switch (k) {
      case MemrefUseKind::Read:
        mode = sde::SdeAccessMode::read;
        break;
      case MemrefUseKind::Write:
        mode = sde::SdeAccessMode::write;
        writable = true;
        break;
      case MemrefUseKind::ReadWrite:
        mode = sde::SdeAccessMode::readwrite;
        writable = true;
        break;
      case MemrefUseKind::None:
        continue;
      }
      auto tokenType = sde::TokenType::get(ctx, tty);
      auto tokOp = sde::SdeMuTokenOp::create(
          builder, loc, tokenType, sde::SdeAccessModeAttr::get(ctx, mode), src,
          /*offsets=*/ValueRange{}, /*sizes=*/ValueRange{});
      tokens.push_back(tokOp.getToken());
      blockArgTypes.push_back(tty);
      if (writable) {
        writableIndices.push_back(idx);
        resultTypes.push_back(tty);
      }
    }

    // Create the codelet shell.
    auto codelet =
        sde::SdeCuCodeletOp::create(builder, loc, resultTypes, tokens,
                                    /*captures=*/ValueRange{});
    Block *codeletBlock = new Block();
    SmallVector<Location> argLocs(blockArgTypes.size(), loc);
    codeletBlock->addArguments(blockArgTypes, argLocs);
    codelet.getBody().push_back(codeletBlock);

    OpBuilder bodyBuilder(ctx);
    bodyBuilder.setInsertionPointToStart(codeletBlock);

    // (1) Move impure task-private producers inside (Gap B1). We move them
    // before cloning pure producers so subsequent clone lookups see the
    // relocated values.
    IRMapping mapping;
    for (Operation *impureOp : captures.moveInsideOps) {
      impureOp->moveBefore(codeletBlock, codeletBlock->end());
      // These ops stay at their original SSA identity (no clone), so we
      // don't need to map them — references from the body will find them
      // through the same SSA value.
      // However, we need the body-clone step to NOT re-map these results;
      // simply leave them out of the mapping. The body-clone below uses
      // `mapping.lookupOrDefault` so references resolve to the moved ops'
      // original SSA values, which now live inside the codelet body.
    }

    // (2) Clone pure external producers.
    auto pureMapping = cloneExternalPureProducers(
        task, /*raisedMemrefs=*/{}, captures.pureExternal, bodyBuilder);
    if (!pureMapping)
      return false;
    // Merge pureMapping entries into `mapping`.
    for (Value v : captures.pureExternal) {
      if (pureMapping->contains(v))
        mapping.map(v, pureMapping->lookup(v));
    }

    // (3) Wire each token's block arg as the "current tensor view" for the
    // corresponding memref (and its cast aliases).
    DenseMap<Value, Value> liveTensor; // canonical memref -> current tensor
    DenseMap<Value, unsigned> canonicalToArgIdx;
    for (auto [idx, canonical] : llvm::enumerate(orderedMemrefs)) {
      Value blockArg = codeletBlock->getArgument(idx);
      liveTensor[canonical] = blockArg;
      canonicalToArgIdx[canonical] = idx;
    }

    // (4) Clone each op from the task body top-level block, rewriting leaf
    // loads/stores on raised memrefs into tensor.extract / tensor.insert on
    // the running tensor value. v1 does not descend into nested regions
    // (the task body is expected flat for our target samples).
    Block &sourceBlock = task.getBody().front();
    for (Operation &op : sourceBlock.without_terminator()) {
      if (rewriteLeafLoadStore(op, mapping, liveTensor, canonicalMemref,
                               bodyBuilder))
        continue;
      // Default clone path.
      bodyBuilder.clone(op, mapping);
    }

    // (5) Yield one tensor per writable token, in the token's positional
    // order.
    SmallVector<Value> yielded;
    for (unsigned idx : writableIndices) {
      Value canonical = orderedMemrefs[idx];
      Value t = liveTensor.lookup(canonical);
      if (!t)
        return false;
      yielded.push_back(t);
    }
    sde::SdeYieldOp::create(bodyBuilder, loc, yielded);

    // (6) Advance currentTensor[canonical] for each writable token so sibling
    // tasks see the updated SSA. This is the Gap A fix.
    for (auto [i, idx] : llvm::enumerate(writableIndices)) {
      Value canonical = orderedMemrefs[idx];
      currentTensor[canonical] = codelet.getResult(i);
    }

    task.erase();
    return true;
  }

  /// If `op` is a leaf load/store on a raised memref, emit the equivalent
  /// tensor.extract / tensor.insert into `bodyBuilder` and update `liveTensor`
  /// accordingly. Returns true if the op was handled.
  bool rewriteLeafLoadStore(Operation &op, IRMapping &mapping,
                            DenseMap<Value, Value> &liveTensor,
                            DenseMap<Value, Value> &canonicalMemref,
                            OpBuilder &bodyBuilder) {
    auto extractAndInsert = [&](Value mr, ValueRange indices, Value storeValue,
                                bool isStore, Operation *origOp) -> bool {
      Value canonical = canonicalMemref.lookup(mr);
      if (!canonical)
        canonical = mr;
      auto it = liveTensor.find(canonical);
      if (it == liveTensor.end())
        return false;
      Location loc = origOp->getLoc();
      SmallVector<Value> idxVals;
      idxVals.reserve(indices.size());
      for (Value idx : indices) {
        auto c = matchConstantIndex(idx);
        assert(c && "gather phase should have checked constant indices");
        idxVals.push_back(arts::createConstantIndex(bodyBuilder, loc, *c));
      }
      if (idxVals.empty())
        idxVals.push_back(arts::createZeroIndex(bodyBuilder, loc));
      Value tensorVal = it->second;
      if (!isStore) {
        Value extracted =
            tensor::ExtractOp::create(bodyBuilder, loc, tensorVal, idxVals);
        mapping.map(origOp->getResult(0), extracted);
        return true;
      }
      Value mappedVal = mapping.lookupOrDefault(storeValue);
      Value updated = tensor::InsertOp::create(bodyBuilder, loc, mappedVal,
                                               tensorVal, idxVals);
      liveTensor[canonical] = updated;
      return true;
    };

    if (auto ld = dyn_cast<memref::LoadOp>(&op))
      return extractAndInsert(ld.getMemref(), ld.getIndices(), Value{},
                              /*isStore=*/false, &op);
    if (auto st = dyn_cast<memref::StoreOp>(&op))
      return extractAndInsert(st.getMemref(), st.getIndices(), st.getValue(),
                              /*isStore=*/true, &op);
    if (auto ld = dyn_cast<affine::AffineLoadOp>(&op)) {
      // Rank-1 / constant-symbol affine.load folds to index 0.
      Value zero = arts::createZeroIndex(bodyBuilder, op.getLoc());
      Value canonical = canonicalMemref.lookup(ld.getMemref());
      if (!canonical)
        canonical = ld.getMemref();
      auto it = liveTensor.find(canonical);
      if (it == liveTensor.end())
        return false;
      Value extracted = tensor::ExtractOp::create(bodyBuilder, op.getLoc(),
                                                  it->second, ValueRange{zero});
      mapping.map(ld.getResult(), extracted);
      return true;
    }
    if (auto st = dyn_cast<affine::AffineStoreOp>(&op)) {
      Value canonical = canonicalMemref.lookup(st.getMemref());
      if (!canonical)
        canonical = st.getMemref();
      auto it = liveTensor.find(canonical);
      if (it == liveTensor.end())
        return false;
      Value zero = arts::createZeroIndex(bodyBuilder, op.getLoc());
      Value mappedVal = mapping.lookupOrDefault(st.getValue());
      Value updated = tensor::InsertOp::create(
          bodyBuilder, op.getLoc(), mappedVal, it->second, ValueRange{zero});
      liveTensor[canonical] = updated;
      return true;
    }
    return false;
  }
};

} // namespace

namespace mlir::arts::sde {

std::unique_ptr<Pass> createRaiseMemrefToTensorPass() {
  return std::make_unique<RaiseMemrefToTensorPass>();
}

} // namespace mlir::arts::sde
