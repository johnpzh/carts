///==========================================================================///
/// File: CreateEpochs.cpp
///
/// Creates explicit arts.epoch regions around groups of EDT launches that must
/// complete before control proceeds.
///
/// Example:
///   Before:
///     arts.edt_create %a
///     arts.edt_create %b
///     ... continuation ...
///
///   After:
///     arts.epoch {
///       arts.edt_create %a
///       arts.edt_create %b
///     }
///     ... continuation ...
///==========================================================================///

/// Dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
/// Arts
#include "carts/dialect/arts/IR/ArtsDialect.h"
#define GEN_PASS_DEF_CREATEEPOCHS
#include "carts/passes/Passes.h"
#include "carts/passes/Passes.h.inc"
#include "carts/dialect/arts/Utils/EdtUtils.h"
#include "carts/utils/Utils.h"
#include "mlir/Pass/Pass.h"
/// Other
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#include "carts/utils/Debug.h"
#include "llvm/ADT/Statistic.h"
ARTS_DEBUG_SETUP(create_epochs);

static llvm::Statistic numSyncEdtsWrapped{
    "create_epochs", "NumSyncEdtsWrapped",
    "Number of sync EDTs wrapped in epoch regions"};
static llvm::Statistic numBarriersProcessed{
    "create_epochs", "NumBarriersProcessed",
    "Number of barrier operations converted to epoch regions"};
static llvm::Statistic numEpochsCreated{
    "create_epochs", "NumEpochsCreated",
    "Total number of epoch regions created"};

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

static void clearIsSyncAttr(EdtOp op) {
  auto newTypeAttr = EdtTypeAttr::get(op.getContext(), EdtType::single);
  op.setTypeAttr(newTypeAttr);
}

static void setIsTaskAttr(EdtOp op) {
  auto newTypeAttr = EdtTypeAttr::get(op.getContext(), EdtType::task);
  op.setTypeAttr(newTypeAttr);
}

static void wrapEdtInEpoch(EdtOp op, bool demoteToTask) {
  if (EdtUtils::isInsideEpoch(op))
    return;
  auto loc = op.getLoc();
  OpBuilder builder(op);
  auto epochOp = EpochOp::create(builder, loc);
  auto &epochBlock = epochOp.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&epochBlock);
  YieldOp::create(builder, loc);
  op->moveBefore(&epochBlock, --epochBlock.end());

  if (demoteToTask) {
    clearIsSyncAttr(op);
    setIsTaskAttr(op);
  }
}

/// Helper function to process synchronous EDT ops.
static void processSyncEdtOp(EdtOp op) {
  /// Only process EDT ops with sync attribute.
  if (op.getTypeAttr().getValue() != EdtType::sync)
    return;
  ARTS_DEBUG("Processing Sync EDT Op: " << op);
  wrapEdtInEpoch(op, /*demoteToTask=*/true);
  ++numSyncEdtsWrapped;
  ++numEpochsCreated;
}

static bool containsEdtLaunch(Operation *op) {
  if (isa<EdtOp>(op))
    return true;

  bool found = false;
  op->walk([&](EdtOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

static bool isMovableBarrierSegmentOp(Operation *op) {
  if (isa<DbAcquireOp, LoweringContractOp>(op))
    return true;

  // Scalar SSA helpers that feed only moved EDTs must move with the EDTs when
  // the barrier segment is wrapped, otherwise the new epoch can be inserted
  // before their definitions and break dominance. Keep this limited to
  // regionless, side-effect-free ops; memory reads/writes and nested control
  // remain outside unless they are already part of an EDT launch container.
  return op->getNumRegions() == 0 && isMemoryEffectFree(op);
}

static Operation *getAncestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

static Operation *getInsertionAfter(Operation *op, Operation *barrier) {
  if (!op || !barrier || op == barrier)
    return barrier;
  auto next = std::next(Block::iterator(op));
  if (next == op->getBlock()->end() || &*next == barrier)
    return barrier;
  return &*next;
}

static Operation *adjustEpochInsertionForDominance(
    Operation *insertionOp, Operation *barrier,
    ArrayRef<Operation *> opsToMove,
    const llvm::SmallDenseSet<Operation *, 8> &opsToMoveSet) {
  if (!insertionOp || !barrier)
    return insertionOp;

  Block *block = barrier->getBlock();
  if (!block)
    return insertionOp;

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *moved : opsToMove) {
      WalkResult result = moved->walk([&](Operation *nested) {
        for (Value operand : nested->getOperands()) {
          Operation *def = operand.getDefiningOp();
          Operation *top = getAncestorInBlock(def, block);
          if (!top || top == barrier || opsToMoveSet.count(top))
            continue;
          if (!top->isBeforeInBlock(barrier))
            continue;
          if (top == insertionOp || insertionOp->isBeforeInBlock(top)) {
            insertionOp = getInsertionAfter(top, barrier);
            changed = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        break;
    }
  }

  return insertionOp;
}

static void processBarrierOp(BarrierOp barrier) {
  ARTS_DEBUG("Processing BarrierOp");
  Block *block = barrier->getBlock();
  if (!block)
    return;

  SmallVector<Operation *, 8> opsToMove;
  llvm::SmallDenseSet<Operation *, 8> opsToMoveSet;
  Operation *epochInsertionOp = nullptr;

  auto barrierIt = Block::iterator(barrier.getOperation());
  auto segmentBegin = barrierIt;
  while (segmentBegin != block->begin()) {
    auto prev = std::prev(segmentBegin);
    if (isa<BarrierOp, EpochOp>(*prev))
      break;
    segmentBegin = prev;
  }

  // First pass: collect ops containing an EDT launch (the primary movers).
  for (auto it = segmentBegin; it != barrierIt; ++it) {
    Operation *op = &*it;
    if (!containsEdtLaunch(op))
      continue;
    if (!isa<EdtOp>(op) && op->getNumResults() != 0)
      continue;
    if (!epochInsertionOp)
      epochInsertionOp = op;
    opsToMove.push_back(op);
    opsToMoveSet.insert(op);
  }

  if (opsToMove.empty()) {
    barrier.erase();
    return;
  }

  // Second pass: also move interleaved ops in the segment whose every user
  // is either another segment op being moved or a descendant of one. This
  // handles patterns like `arts.db_acquire` sitting between task EDTs that
  // consume its ptr — leaving them behind would break SSA dominance when
  // the EDTs move into the epoch. We iterate to a fixed point so chains of
  // such ops are all moved.
  //
  // Conservative condition: an op is movable only if ALL of its results'
  // users live inside an op we're already moving (or inside the segment op
  // itself). Ops with side-effects outside the SSA world stay put.
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto it = segmentBegin; it != barrierIt; ++it) {
      Operation *op = &*it;
      if (opsToMoveSet.count(op))
        continue;
      if (containsEdtLaunch(op))
        continue; // already handled
      if (epochInsertionOp && op->isBeforeInBlock(epochInsertionOp))
        continue;
      // Skip ops with side effects that we cannot safely reorder. db_acquire
      // and lowering contracts are dep-tracking SSA producers, and pure
      // regionless scalar ops are safe to move alongside the EDTs that consume
      // them.
      if (!isMovableBarrierSegmentOp(op))
        continue;
      bool allUsersInside = true;
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          // Walk up until we find an op in the current block.
          Operation *cur = user;
          while (cur && cur->getBlock() != block)
            cur = cur->getParentOp();
          if (!cur || !opsToMoveSet.count(cur)) {
            allUsersInside = false;
            break;
          }
        }
        if (!allUsersInside)
          break;
      }
      if (!allUsersInside)
        continue;
      opsToMove.push_back(op);
      opsToMoveSet.insert(op);
      changed = true;
    }
  }

  // Re-sort opsToMove into program order so the moveBefore sequence in the
  // epoch block preserves source order.
  llvm::sort(opsToMove,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  epochInsertionOp = adjustEpochInsertionForDominance(
      epochInsertionOp, barrier.getOperation(), opsToMove, opsToMoveSet);

  /// Create epoch and move operations into it
  Location loc = barrier.getLoc();
  OpBuilder builder(block, Block::iterator(epochInsertionOp));
  auto epochOp = EpochOp::create(builder, loc);
  auto &epochRegion = epochOp.getRegion();
  if (epochRegion.empty())
    epochRegion.push_back(new Block());
  Block *newBlock = &epochRegion.front();

  for (Operation *op : opsToMove) {
    if (!op->getBlock())
      continue;
    ARTS_INFO("Moving operation: " << *op);
    op->moveBefore(newBlock, newBlock->end());
  }

  builder.setInsertionPointToEnd(newBlock);
  YieldOp::create(builder, loc);

  ++numBarriersProcessed;
  ++numEpochsCreated;
  barrier.erase();
}

///==========================================================================///
/// Pass Implementation
///==========================================================================///
namespace {
struct CreateEpochsPass : public impl::CreateEpochsBase<CreateEpochsPass> {
  void runOnOperation() override;
};
} // end namespace

void CreateEpochsPass::runOnOperation() {
  ModuleOp module = getOperation();
  ARTS_INFO_HEADER(CreateEpochsPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// Process Sync EDT Ops: for each EDT op that is sync, create an epoch op
  /// and move the EDT op inside the epoch op.
  ARTS_DEBUG_HEADER(ProcessSyncEdtOp);
  SmallVector<EdtOp> syncEdts;
  module.walk([&](EdtOp op) {
    if (op.getTypeAttr().getValue() == EdtType::sync)
      syncEdts.push_back(op);
  });
  for (EdtOp op : syncEdts) {
    if (op->getBlock())
      processSyncEdtOp(op);
  }
  ARTS_DEBUG_FOOTER(ProcessSyncEdtOp);

  /// Process Barrier Ops: for each barrier, collect all EDTs that are affected
  /// by the barrier and embed them in a new epoch op.
  ARTS_DEBUG_HEADER(ProcessBarrierOp);
  SmallVector<BarrierOp> barriers;
  module.walk([&](BarrierOp barrier) { barriers.push_back(barrier); });
  for (BarrierOp barrier : barriers) {
    if (barrier->getBlock())
      processBarrierOp(barrier);
  }
  ARTS_DEBUG_FOOTER(ProcessBarrierOp);

  ARTS_INFO_FOOTER(CreateEpochsPass);
  ARTS_DEBUG_REGION(module.dump(););
}

///==========================================================================///
/// Pass creation
///==========================================================================///
namespace mlir {
namespace carts::arts {
std::unique_ptr<Pass> createCreateEpochsPass() {
  return std::make_unique<CreateEpochsPass>();
}
} // namespace carts::arts
} // namespace mlir
