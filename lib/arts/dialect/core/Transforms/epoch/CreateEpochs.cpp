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
#include "arts/Dialect.h"
#define GEN_PASS_DEF_CREATEEPOCHS
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/EdtUtils.h"
#include "arts/utils/Utils.h"
#include "mlir/Pass/Pass.h"
/// Other
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "arts/utils/Debug.h"
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
using namespace mlir::arts;

static void clearIsSyncAttr(EdtOp op) {
  auto newTypeAttr = EdtTypeAttr::get(op.getContext(), EdtType::single);
  op.setTypeAttr(newTypeAttr);
}

static void setIsTaskAttr(EdtOp op) {
  auto newTypeAttr = EdtTypeAttr::get(op.getContext(), EdtType::task);
  op.setTypeAttr(newTypeAttr);
}

static void wrapEdtInEpoch(EdtOp op, bool demoteToTask) {
  if (isInsideEpoch(op))
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

static void processBarrierOp(BarrierOp barrier) {
  ARTS_DEBUG("Processing BarrierOp");
  Block *block = barrier->getBlock();
  if (!block)
    return;

  SmallVector<Operation *, 8> opsToMove;
  llvm::SmallDenseSet<Operation *, 8> opsToMoveSet;

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
      // Skip ops with side effects that we cannot safely reorder. db_acquire
      // is a dep-tracking SSA producer and is safe to move alongside the
      // EDTs that consume it; restrict the hoist to that specific op for
      // now to avoid perturbing other patterns.
      if (!isa<DbAcquireOp, LoweringContractOp>(op))
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

  /// Create epoch and move operations into it
  Location loc = barrier.getLoc();
  Operation *insertionOp = opsToMove.front();
  OpBuilder builder(block, Block::iterator(insertionOp));
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
namespace arts {
std::unique_ptr<Pass> createCreateEpochsPass() {
  return std::make_unique<CreateEpochsPass>();
}
} // namespace arts
} // namespace mlir
