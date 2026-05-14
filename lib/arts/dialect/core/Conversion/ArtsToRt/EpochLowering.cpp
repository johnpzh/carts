///==========================================================================///
/// Epoch Lowering Pass
/// Transforms arts.epoch into CreateEpochOp + WaitOnEpochOp, propagating
/// epoch GUIDs to contained EdtCreateOps.
///
/// For continuation epochs (marked by EpochOpt), wires the epoch's finish
/// target to the continuation EDT and skips WaitOnEpochOp.
///
/// Example (standard path):
///   Before:
///     arts.epoch { ... arts.edt_create ... }
///
///   After:
///     %e = arts.create_epoch
///     ... arts.edt_create(..., %e) ...
///     arts.wait_on_epoch %e
///
/// Example (continuation path):
///   Before:
///     arts.epoch { ... arts.edt_create ... } {arts.continuation_for_epoch}
///     %cont = arts.edt_create(...) {arts.continuation_for_epoch, ...}
///
///   After:
///     %cont = arts.edt_create(...)
///     %e = arts.create_epoch finish(%cont_guid, %control_slot)
///     ... arts.edt_create(..., %e) ...
///     // NO wait_on_epoch
///==========================================================================///

#include "arts/dialect/rt/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_EPOCHLOWERING
#include "arts/dialect/rt/Transforms/Passes.h.inc"
} // namespace mlir::arts
#include "arts/dialect/core/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "arts/dialect/rt/IR/RtDialect.h"
#include "arts/passes/Passes.h"
#include "arts/utils/OperationAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(epoch_lowering);

#include "llvm/ADT/Statistic.h"
static llvm::Statistic numEpochsLowered{
    "epoch_lowering", "NumEpochsLowered",
    "Number of epoch operations lowered to CreateEpochOp + WaitOnEpochOp"};
static llvm::Statistic numContinuationEpochsLowered{
    "epoch_lowering", "NumContinuationEpochsLowered",
    "Number of continuation epochs lowered without WaitOnEpochOp"};
static llvm::Statistic numEmptyEpochsElided{
    "epoch_lowering", "NumEmptyEpochsElided", "Number of empty epochs elided"};
static llvm::Statistic numEdtCreatesUpdatedWithEpoch{
    "epoch_lowering", "NumEdtCreatesUpdatedWithEpoch",
    "Number of EdtCreateOps updated with epoch GUID"};

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arts;
using AttrNames::Operation::ContinuationForEpoch;
using AttrNames::Operation::ControlDep;
using namespace mlir::arts::rt;

///===----------------------------------------------------------------------===///
/// Epoch Lowering Pass Implementation
///===----------------------------------------------------------------------===///
struct EpochLoweringPass
    : public arts::impl::EpochLoweringBase<EpochLoweringPass> {
  explicit EpochLoweringPass(bool debug = false) : debugMode(debug) {}

  void runOnOperation() override;

private:
  /// State
  ModuleOp module;
  ArtsCodegen *AC = nullptr;
  bool debugMode = false;
};

///===----------------------------------------------------------------------===///
/// Pass Implementation
///===----------------------------------------------------------------------===///

void EpochLoweringPass::runOnOperation() {
  module = getOperation();
  auto ownedAC = std::make_unique<ArtsCodegen>(module, debugMode);
  AC = ownedAC.get();

  ARTS_INFO_HEADER(EpochLoweringPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// Collect all epoch operations bottom-to-top (post-order) so inner epochs
  /// are lowered before their parents.
  SmallVector<EpochOp> epochOps;
  module.walk<WalkOrder::PostOrder>(
      [&](EpochOp epochOp) { epochOps.push_back(epochOp); });

  ARTS_INFO("Found " << epochOps.size() << " epoch operations to lower");

  for (EpochOp epochOp : epochOps) {
    ARTS_INFO("Lowering Epoch Op " << epochOp);

    /// Elide empty epochs.
    auto &epochRegion = epochOp.getRegion();
    if (epochRegion.empty() ||
        epochRegion.front().without_terminator().empty()) {
      ++numEmptyEpochsElided;
      epochOp.erase();
      continue;
    }

    /// Check if this epoch is marked for persistent structured region lowering.
    /// When set, worker EDTs within this epoch are expected to maintain stable
    /// owner-local slices across multiple logical timesteps.
    bool isPersistent = false;
    if (auto persistAttr = epochOp->getAttrOfType<BoolAttr>(
            AttrNames::Operation::PersistentRegion))
      isPersistent = persistAttr.getValue();
    if (isPersistent) {
      ARTS_INFO("  Persistent structured region: epoch will use stable "
                "owner-strip execution");
    }

    /// Check if this epoch has a continuation EDT (set by EpochOpt).
    bool hasContinuation = epochOp->hasAttr(ContinuationForEpoch);
    Value finishGuid;
    Value finishSlot;

    if (hasContinuation) {
      /// Find the continuation EdtCreateOp placed after this epoch by EpochOpt
      /// (already lowered from arts.edt by EdtLowering).
      EdtCreateOp contEdtCreate = nullptr;
      for (Operation *op = epochOp->getNextNode(); op; op = op->getNextNode()) {
        if (auto edt = dyn_cast<EdtCreateOp>(op)) {
          if (edt->hasAttr(ContinuationForEpoch)) {
            contEdtCreate = edt;
            break;
          }
        }
      }

      if (contEdtCreate) {
        /// Recursively move the continuation EdtCreateOp and ALL transitive
        /// operand-defining ops before the epoch. A shallow move misses
        /// values that feed into edt_param_pack but are not direct operands of
        /// EdtCreateOp.
        DenseSet<Operation *> moved;
        std::function<void(Operation *)> moveWithDeps = [&](Operation *op) {
          if (!op || moved.contains(op))
            return;
          if (op->getBlock() != epochOp->getBlock())
            return;
          if (!epochOp->isBeforeInBlock(op))
            return;
          moved.insert(op);
          for (Value operand : op->getOperands())
            if (auto *defOp = operand.getDefiningOp())
              moveWithDeps(defOp);
          op->moveBefore(epochOp);
        };
        moveWithDeps(contEdtCreate.getOperation());

        finishGuid = contEdtCreate.getGuid();
        /// Control slot = depCount - 1 (EdtLowering added +1 for the control
        /// dep).
        AC->setInsertionPointAfter(contEdtCreate);
        Value depCount = contEdtCreate.getDepCount();
        Value one = AC->createIntConstant(1, AC->Int32, epochOp.getLoc());
        finishSlot = AC->create<arith::SubIOp>(epochOp.getLoc(), depCount, one);
        ARTS_INFO("  Continuation path: finish GUID from "
                  << contEdtCreate << ", control slot = depCount - 1");
      } else {
        ARTS_WARN("  Epoch marked for continuation but no continuation "
                  "EdtCreateOp found; falling back to wait path");
        hasContinuation = false;
      }
    }

    /// Create the CreateEpochOp.
    AC->setInsertionPoint(epochOp);
    auto createEpochOp = AC->create<CreateEpochOp>(
        epochOp.getLoc(), IntegerType::get(AC->getContext(), 64),
        hasContinuation ? finishGuid : Value(),
        hasContinuation ? finishSlot : Value());
    auto currentEpoch = createEpochOp.getEpochGuid();
    bool needsWait = !hasContinuation;

    /// Propagate persistent region flag to the lowered CreateEpochOp.
    if (isPersistent)
      createEpochOp->setAttr(AttrNames::Operation::PersistentRegion,
                             BoolAttr::get(createEpochOp.getContext(), true));

    /// Collect EdtCreateOps that need the epoch GUID.
    SmallVector<EdtCreateOp, 8> edtCreatesToUpdate;
    epochOp.walk([&](EdtCreateOp edtCreateOp) {
      if (!edtCreateOp.getEpochGuid())
        edtCreatesToUpdate.push_back(edtCreateOp);
    });

    ARTS_INFO("Updating " << edtCreatesToUpdate.size()
                          << " EdtCreateOps with epoch GUID");

    numEdtCreatesUpdatedWithEpoch += edtCreatesToUpdate.size();
    for (EdtCreateOp edtCreateOp : edtCreatesToUpdate) {
      AC->setInsertionPoint(edtCreateOp);
      auto newEdtCreateOp = AC->create<EdtCreateOp>(
          edtCreateOp.getLoc(), edtCreateOp.getParamMemref(),
          edtCreateOp.getDepCount(), edtCreateOp.getRoute(), currentEpoch);
      for (auto attr : edtCreateOp->getAttrs())
        newEdtCreateOp->setAttr(attr.getName(), attr.getValue());
      edtCreateOp->replaceAllUsesWith(newEdtCreateOp);
      edtCreateOp->erase();
    }

    /// Move operations out of the epoch region, tracking where to insert
    /// the wait afterward.
    auto &epochRegionForMove = epochOp.getRegion();
    Operation *insertionAfter = epochOp.getOperation();
    if (!epochRegionForMove.empty()) {
      auto &epochBlock = epochRegionForMove.front();
      SmallVector<Operation *> opsToMove;
      for (auto &innerOp : epochBlock.without_terminator()) {
        if (!isa<EpochOp>(innerOp))
          opsToMove.push_back(&innerOp);
      }
      for (auto *opToMove : opsToMove) {
        opToMove->moveBefore(epochOp);
        insertionAfter = opToMove;
      }
    }

    /// Determine whether to emit WaitOnEpochOp.
    ///
    /// Non-continuation epochs always get a blocking wait.
    ///
    /// Regular continuation epochs skip the wait: the creator EDT terminates
    /// promptly after spawning workers, and arts_increment_finished_epoch_list
    /// releases the creator's guard active count naturally.
    if (hasContinuation) {
      ARTS_INFO("  Skipping WaitOnEpochOp (continuation path)");
    }
    if (needsWait) {
      AC->setInsertionPointAfter(insertionAfter);
      AC->create<WaitOnEpochOp>(epochOp.getLoc(), currentEpoch);
    }

    if (hasContinuation)
      ++numContinuationEpochsLowered;
    else
      ++numEpochsLowered;

    /// Replace the epoch op with the epoch GUID.
    epochOp.replaceAllUsesWith(currentEpoch);
    epochOp.erase();
  }

  SmallVector<EdtCreateOp> strayControlDepCreates;
  module.walk([&](EdtCreateOp edt) {
    if (edt->hasAttr(ControlDep))
      strayControlDepCreates.push_back(edt);
  });

  for (EdtCreateOp edt : strayControlDepCreates) {
    bool hasFinishTarget = false;
    for (Operation *user : edt.getGuid().getUsers()) {
      if (auto createEpoch = dyn_cast<CreateEpochOp>(user))
        if (createEpoch.getFinishEdtGuid() == edt.getGuid()) {
          hasFinishTarget = true;
          break;
        }
    }
    if (hasFinishTarget)
      continue;

    OpBuilder::InsertionGuard guard(AC->getBuilder());
    AC->setInsertionPoint(edt);
    Value one = AC->createIntConstant(1, AC->Int32, edt.getLoc());
    Value depCount = edt.getDepCount();
    Value adjustedDepCount =
        AC->create<arith::SubIOp>(edt.getLoc(), depCount, one);
    edt->setOperand(1, adjustedDepCount);
    edt->removeAttr(ControlDep);
    ARTS_INFO("Removed stray control dep from continuation kickoff "
              << edt.getOperationName());
  }

  SmallVector<Operation *> duplicateReleases;
  module.walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        llvm::SmallDenseSet<Value, 8> releasedValues;
        for (Operation &nested : block) {
          auto release = dyn_cast<DbReleaseOp>(&nested);
          if (!release)
            continue;
          if (!releasedValues.insert(release.getSource()).second)
            duplicateReleases.push_back(release);
        }
      }
    }
  });
  for (Operation *release : duplicateReleases) {
    ARTS_DEBUG("Removing duplicate db_release introduced during epoch "
               "lowering: "
               << *release);
    release->erase();
  }

  ARTS_INFO_FOOTER(EpochLoweringPass);
  AC = nullptr;
  ARTS_DEBUG_REGION(module.dump(););
}

///===----------------------------------------------------------------------===///
/// Pass Registration
///===----------------------------------------------------------------------===///

namespace mlir {
namespace arts {

std::unique_ptr<Pass> createEpochLoweringPass() {
  return std::make_unique<EpochLoweringPass>();
}

} // namespace arts
} // namespace mlir
