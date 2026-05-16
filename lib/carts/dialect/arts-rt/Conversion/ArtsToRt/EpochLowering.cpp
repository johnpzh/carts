///==========================================================================///
/// Epoch Lowering Pass
/// Transforms arts.epoch into CreateEpochOp + WaitOnEpochOp, propagating
/// epoch GUIDs to contained EdtCreateOps.
///
/// Example (standard path):
///   Before:
///     arts.epoch { ... arts.edt_create ... }
///
///   After:
///     %e = arts.create_epoch
///     ... arts.edt_create(..., %e) ...
///     arts.wait_on_epoch %e
///==========================================================================///

#include "carts/dialect/arts-rt/Transforms/Passes.h"
namespace mlir::carts::arts {
#define GEN_PASS_DEF_EPOCHLOWERING
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts
#include "carts/dialect/arts-rt/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "carts/passes/Passes.h"
#include "carts/utils/OperationAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(epoch_lowering);

#include "llvm/ADT/Statistic.h"
static llvm::Statistic numEpochsLowered{
    "epoch_lowering", "NumEpochsLowered",
    "Number of epoch operations lowered to CreateEpochOp + WaitOnEpochOp"};
static llvm::Statistic numEmptyEpochsElided{
    "epoch_lowering", "NumEmptyEpochsElided", "Number of empty epochs elided"};
static llvm::Statistic numEdtCreatesUpdatedWithEpoch{
    "epoch_lowering", "NumEdtCreatesUpdatedWithEpoch",
    "Number of EdtCreateOps updated with epoch GUID"};

using namespace mlir;
using namespace mlir::func;
using namespace mlir::carts;
using namespace mlir::carts::arts;
using namespace mlir::carts::arts_rt;

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

    /// Create the CreateEpochOp.
    AC->setInsertionPoint(epochOp);
    auto createEpochOp = AC->create<CreateEpochOp>(
        epochOp.getLoc(), IntegerType::get(AC->getContext(), 64),
        /*finishEdtGuid=*/Value(), /*finishSlot=*/Value());
    auto currentEpoch = createEpochOp.getEpochGuid();

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

    AC->setInsertionPointAfter(insertionAfter);
    AC->create<WaitOnEpochOp>(epochOp.getLoc(), currentEpoch);
    ++numEpochsLowered;

    /// Replace the epoch op with the epoch GUID.
    epochOp.replaceAllUsesWith(currentEpoch);
    epochOp.erase();
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
namespace carts::arts {

std::unique_ptr<Pass> createEpochLoweringPass() {
  return std::make_unique<EpochLoweringPass>();
}

} // namespace carts::arts
} // namespace mlir
