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
#include "arts/utils/DbUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "polygeist/Ops.h"
#include <limits>

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
static llvm::Statistic numCpsAdvancesResolved{
    "epoch_lowering", "NumCpsAdvancesResolved",
    "Number of CPS advance placeholders resolved"};
static llvm::Statistic numEdtCreatesUpdatedWithEpoch{
    "epoch_lowering", "NumEdtCreatesUpdatedWithEpoch",
    "Number of EdtCreateOps updated with epoch GUID"};

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arts;
using AttrNames::Operation::ContinuationForEpoch;
using AttrNames::Operation::ControlDep;
using AttrNames::Operation::CPSAdditiveParams;
using AttrNames::Operation::CPSAdvanceHasIvArg;
using AttrNames::Operation::CPSChainId;
using AttrNames::Operation::CPSDepRouting;
using AttrNames::Operation::CPSDirectRecreate;
using AttrNames::Operation::CPSForwardDeps;
using AttrNames::Operation::CPSInitIter;
using AttrNames::Operation::CPSIterCounterParamIdx;
using AttrNames::Operation::CPSNumCarry;
using AttrNames::Operation::CPSOuterEpochParamIdx;
using AttrNames::Operation::CPSParamPerm;
using AttrNames::Operation::CPSPreserveCarryAbi;
using namespace mlir::arts::rt;

///===----------------------------------------------------------------------===///
/// Epoch Lowering Pass Implementation
///===----------------------------------------------------------------------===///
struct EpochLoweringPass
    : public arts::impl::EpochLoweringBase<EpochLoweringPass> {
  explicit EpochLoweringPass(bool debug = false) : debugMode(debug) {}

  void runOnOperation() override;

private:
  LogicalResult compactContinuationParamAbi();

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

  auto setCpsSchemaAttrs = [&](Operation *op, ArrayRef<int64_t> perm,
                               unsigned iterIdx, unsigned epochIdx) {
    if (!op)
      return;
    if (perm.empty())
      op->removeAttr(CPSParamPerm);
    else
      op->setAttr(CPSParamPerm, AC->getBuilder().getDenseI64ArrayAttr(perm));
    op->setAttr(CPSIterCounterParamIdx,
                AC->getBuilder().getI64IntegerAttr(iterIdx));
    op->setAttr(CPSOuterEpochParamIdx,
                AC->getBuilder().getI64IntegerAttr(epochIdx));
  };

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
    bool isCPSChainInnerEpoch = false;
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
        isCPSChainInnerEpoch = contEdtCreate->hasAttr(CPSChainId);

        /// Recursively move the continuation EdtCreateOp and ALL transitive
        /// operand-defining ops before the epoch. A shallow move misses ops
        /// like DB handle extractions (memref.load, llvm.ptrtoint) that feed
        /// into edt_param_pack but are not direct operands of EdtCreateOp.
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
    bool needsWait = !hasContinuation || isCPSChainInnerEpoch;

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

    if (needsWait) {
      SmallVector<EdtCreateOp, 8> localAccumulateCreates;
      epochOp.walk([&](EdtCreateOp edtCreateOp) {
        if (edtCreateOp.getEpochGuid() != currentEpoch)
          return;
        auto strategy = edtCreateOp->getAttrOfType<StringAttr>(
            AttrNames::Operation::Contract::ReductionStrategy);
        if (strategy && strategy.getValue() ==
                            AttrNames::Operation::Contract::
                                ReductionStrategyValue::LocalAccumulate)
          localAccumulateCreates.push_back(edtCreateOp);
      });

      for (EdtCreateOp edtCreateOp : localAccumulateCreates) {
        Operation *insertAfter = edtCreateOp.getOperation();
        while (Operation *next = insertAfter->getNextNode()) {
          auto recordDep = dyn_cast<RecordDepOp>(next);
          if (!recordDep || recordDep.getEdtGuid() != edtCreateOp.getGuid())
            break;
          insertAfter = next;
        }
        AC->setInsertionPointAfter(insertAfter);
        AC->create<WaitOnEpochOp>(edtCreateOp.getLoc(), currentEpoch);
      }
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
    ///
    /// CPS chain inner epochs are the exception: the creator (mainBody) is
    /// blocked waiting on the outer CPS epoch and cannot terminate until the
    /// entire chain completes. Without an explicit wait, the creator's guard
    /// active count in the inner epoch is never released, creating a circular
    /// dependency → deadlock. Emitting WaitOnEpochOp releases the guard and
    /// blocks only until the first iteration's workers finish.
    if (hasContinuation && !isCPSChainInnerEpoch) {
      ARTS_INFO("  Skipping WaitOnEpochOp (continuation path)");
    }
    if (needsWait) {
      AC->setInsertionPointAfter(insertionAfter);
      AC->create<WaitOnEpochOp>(epochOp.getLoc(), currentEpoch);
    }

    /// Check CPS outer epoch attributes BEFORE erasing the EpochOp.
    bool isCPSOuterEpoch =
        epochOp->hasAttr(AttrNames::Operation::CPSOuterEpoch);
    int64_t cpsInitIter = 0;
    if (auto initIterAttr = epochOp->getAttrOfType<IntegerAttr>(CPSInitIter))
      cpsInitIter = initIterAttr.getInt();

    if (hasContinuation)
      ++numContinuationEpochsLowered;
    else
      ++numEpochsLowered;

    /// Replace the epoch op with the epoch GUID.
    epochOp.replaceAllUsesWith(currentEpoch);
    epochOp.erase();

    /// For CPS outer epochs, inject the iteration counter AND outer epoch
    /// GUID into all continuation EdtCreateOps' param_packs so they
    /// propagate through the chain.
    if (isCPSOuterEpoch) {
      func::FuncOp parentFunc = createEpochOp->getParentOfType<func::FuncOp>();
      ARTS_INFO("CPS outer epoch: injecting iter counter + outer GUID "
                "into chain EDTs");
      if (parentFunc) {
        parentFunc.walk([&](EdtCreateOp edt) {
          if (!edt->hasAttr(CPSChainId)) {
            return;
          }
          ARTS_INFO("  Found chain EDT: " << edt);
          auto packOp = edt.getParamMemref().getDefiningOp<EdtParamPackOp>();
          if (!packOp)
            return;
          /// Build a new param_pack with iter counter + outer epoch GUID
          /// appended. Iter counter first so its index is stable.
          SmallVector<Value> packParams(packOp->getOperands());
          AC->setInsertionPoint(packOp);
          Value initIterVal =
              AC->createIntConstant(cpsInitIter, AC->Int64, packOp->getLoc());
          packParams.push_back(initIterVal);
          unsigned iterIdx = packParams.size() - 1;
          packParams.push_back(currentEpoch);
          unsigned epochIdx = packParams.size() - 1;
          auto packType = MemRefType::get(
              {static_cast<int64_t>(packParams.size())}, AC->Int64);
          auto newPack = AC->create<EdtParamPackOp>(packOp->getLoc(), packType,
                                                    packParams);
          packOp->getResult(0).replaceAllUsesWith(newPack.getMemref());
          packOp->erase();
          SmallVector<int64_t> identityPerm;
          identityPerm.reserve(iterIdx);
          for (unsigned slot = 0; slot < iterIdx; ++slot)
            identityPerm.push_back(slot);
          setCpsSchemaAttrs(edt, identityPerm, iterIdx, epochIdx);
          if (auto funcAttr = edt->getAttrOfType<StringAttr>(
                  AttrNames::Operation::OutlinedFunc)) {
            if (auto outlined =
                    module.lookupSymbol<func::FuncOp>(funcAttr.getValue())) {
              setCpsSchemaAttrs(outlined, identityPerm, iterIdx, epochIdx);
            }
          }
          ARTS_INFO("  Injected iter counter at idx "
                    << iterIdx << ", outer epoch at idx " << epochIdx);
        });
      }
    }
  }

  /// Propagate iter counter + outer epoch GUID through the chain. The
  /// initial injection (above) only added them to chain EDTs in the main
  /// function. Outlined chain functions (e.g. __arts_edt_5 containing
  /// chain_1) also need to pass them to their child chain EDTs.
  /// This MUST run before CPSAdvance resolution because it computes param
  /// permutations that the CPSAdvance resolver needs.
  {
    struct ChainFuncInfo {
      StringAttr funcName;
      int64_t iterIdx;
      int64_t epochIdx;
    };
    SmallVector<ChainFuncInfo> worklist;
    DenseSet<StringAttr> queuedFuncs;

    auto enqueueChainFunc = [&](StringAttr funcName, int64_t iterIdx,
                                int64_t epochIdx) {
      if (!funcName || !queuedFuncs.insert(funcName).second)
        return;
      worklist.push_back({funcName, iterIdx, epochIdx});
    };

    module.walk([&](EdtCreateOp edt) {
      auto iterAttr = edt->getAttrOfType<IntegerAttr>(CPSIterCounterParamIdx);
      auto epochAttr = edt->getAttrOfType<IntegerAttr>(CPSOuterEpochParamIdx);
      auto funcAttr =
          edt->getAttrOfType<StringAttr>(AttrNames::Operation::OutlinedFunc);
      if (iterAttr && epochAttr && funcAttr)
        enqueueChainFunc(funcAttr, iterAttr.getInt(), epochAttr.getInt());
    });

    while (!worklist.empty()) {
      ChainFuncInfo info = worklist.pop_back_val();
      auto func = module.lookupSymbol<func::FuncOp>(info.funcName.getValue());
      if (!func)
        continue;

      SmallVector<EdtCreateOp> childChainEdts;
      func.walk([&](EdtCreateOp edt) {
        if (edt->hasAttr(CPSChainId) && !edt->hasAttr(CPSOuterEpochParamIdx))
          childChainEdts.push_back(edt);
      });

      SmallVector<EdtCreateOp> directRecreateContinuations;
      func.walk([&](EdtCreateOp edt) {
        if (!edt->hasAttr(ContinuationForEpoch))
          return;
        auto childFuncAttr =
            edt->getAttrOfType<StringAttr>(AttrNames::Operation::OutlinedFunc);
        if (!childFuncAttr)
          return;
        func::FuncOp childFunc =
            module.lookupSymbol<func::FuncOp>(childFuncAttr.getValue());
        if (!childFunc)
          return;
        bool childUsesDirectRecreate = false;
        childFunc.walk([&](CPSAdvanceOp advanceOp) {
          if (advanceOp->hasAttr(CPSDirectRecreate))
            childUsesDirectRecreate = true;
        });
        if (childUsesDirectRecreate)
          directRecreateContinuations.push_back(edt);
      });

      if (childChainEdts.empty() && directRecreateContinuations.empty())
        continue;

      EdtParamUnpackOp unpackOp = nullptr;
      func.walk([&](EdtParamUnpackOp op) {
        if (!unpackOp)
          unpackOp = op;
      });

      Location loc = func.getLoc();
      Value paramArrayMemref;
      if (unpackOp) {
        AC->setInsertionPointAfter(unpackOp);
        paramArrayMemref = unpackOp->getOperand(0);
      } else if (func.getNumArguments() > 1) {
        Value candidate = func.getArgument(1);
        if (auto memrefType = dyn_cast<MemRefType>(candidate.getType());
            memrefType && memrefType.getElementType().isInteger(64)) {
          AC->setInsertionPointToStart(&func.front());
          paramArrayMemref = candidate;
        }
      }
      if (!paramArrayMemref)
        continue;

      /// Load iter counter from parent's param array.
      Value iterIdxVal = AC->createIndexConstant(info.iterIdx, loc);
      Value iterCounter = AC->create<memref::LoadOp>(loc, paramArrayMemref,
                                                     ValueRange{iterIdxVal});
      /// Load outer epoch GUID from parent's param array.
      Value epochIdxVal = AC->createIndexConstant(info.epochIdx, loc);
      Value outerEpochGuid = AC->create<memref::LoadOp>(
          loc, paramArrayMemref, ValueRange{epochIdxVal});

      ARTS_INFO("CPS propagation: " << info.funcName.getValue()
                                    << " loading iter[" << info.iterIdx
                                    << "], epoch[" << info.epochIdx << "]");

      auto attachToOuterEpoch = [&](EdtCreateOp edt) {
        if (!edt || edt.getEpochGuid() || !outerEpochGuid)
          return;
        edt.getEpochGuidMutable().assign(outerEpochGuid);
        ARTS_DEBUG("CPS propagation: attached child continuation EDT to "
                   "outer epoch");
      };

      /// Build a map from parent function's unpack results to their indices.
      DenseMap<Value, int64_t> unpackResultToIdx;
      if (unpackOp)
        for (unsigned i = 0; i < unpackOp->getNumResults(); ++i)
          unpackResultToIdx[unpackOp->getResult(i)] = i;

      /// Compute the parent param index a value derives from. We check two
      /// paths:
      /// 1. SSA identity with EdtParamUnpackOp results (pre-CSE)
      /// 2. memref.load from the param array with a constant index (post-CSE)
      /// DB handles go through conversions (inttoptr, ptrtoint) so we trace
      /// the def chain up to 10 levels deep.
      auto traceToParamIdx = [&](Value val) -> int64_t {
        for (int depth = 0; depth < 10; ++depth) {
          auto it = unpackResultToIdx.find(val);
          if (it != unpackResultToIdx.end())
            return it->second;
          auto *defOp = val.getDefiningOp();
          if (!defOp)
            return -1;
          if (auto loadOp = dyn_cast<memref::LoadOp>(defOp)) {
            if (loadOp.getMemRef() == paramArrayMemref &&
                loadOp.getIndices().size() == 1) {
              if (auto idxOp = loadOp.getIndices()[0]
                                   .getDefiningOp<arith::ConstantIndexOp>())
                return idxOp.value();
            }
          }
          if (defOp->getNumOperands() == 0)
            return -1;
          val = defOp->getOperand(0);
        }
        return -1;
      };

      auto buildExpandedDirectRecreateCarry = [&](EdtParamPackOp packOp,
                                                  unsigned carryCount)
          -> std::pair<SmallVector<Value>, SmallVector<int64_t>> {
        SmallVector<Value> newCarryValues;
        SmallVector<int64_t> newPerm;
        DenseSet<int64_t> representedCanonicalSlots;

        for (Value operand : packOp.getOperands()) {
          int64_t canonicalIdx = traceToParamIdx(operand);
          if (canonicalIdx < 0 ||
              canonicalIdx >= static_cast<int64_t>(carryCount))
            continue;
          newCarryValues.push_back(operand);
          newPerm.push_back(canonicalIdx);
          representedCanonicalSlots.insert(canonicalIdx);
        }

        for (unsigned canonicalSlot = 0; canonicalSlot < carryCount;
             ++canonicalSlot) {
          if (representedCanonicalSlots.contains(canonicalSlot))
            continue;
          Value slotValue;
          if (unpackOp && canonicalSlot < unpackOp.getNumResults()) {
            slotValue = unpackOp.getResult(canonicalSlot);
          } else {
            Value slotIdx = AC->createIndexConstant(canonicalSlot, loc);
            slotValue = AC->create<memref::LoadOp>(loc, paramArrayMemref,
                                                   ValueRange{slotIdx});
          }
          newCarryValues.push_back(slotValue);
          newPerm.push_back(canonicalSlot);
        }

        return {std::move(newCarryValues), std::move(newPerm)};
      };

      for (EdtCreateOp edt : childChainEdts) {
        auto packOp = edt.getParamMemref().getDefiningOp<EdtParamPackOp>();
        if (!packOp)
          continue;

        auto childFuncAttr =
            edt->getAttrOfType<StringAttr>(AttrNames::Operation::OutlinedFunc);
        func::FuncOp childFunc =
            childFuncAttr
                ? module.lookupSymbol<func::FuncOp>(childFuncAttr.getValue())
                : func::FuncOp();
        bool childUsesDirectRecreate = false;
        if (childFunc) {
          childFunc.walk([&](CPSAdvanceOp advanceOp) {
            if (advanceOp->hasAttr(CPSDirectRecreate))
              childUsesDirectRecreate = true;
          });
        }
        if (childUsesDirectRecreate) {
          ARTS_INFO("CPS propagation: direct-recreate child "
                    << childFunc.getName() << " under " << info.funcName
                    << " preserving carry prefix up to iter slot "
                    << info.iterIdx);
        }

        SmallVector<int64_t> perm;
        for (Value operand : packOp->getOperands())
          perm.push_back(traceToParamIdx(operand));

        SmallVector<Value> packParams;
        if (childUsesDirectRecreate) {
          unsigned carryCount = std::max<int64_t>(info.iterIdx, 0);
          ARTS_INFO("  rebuilding child pack with carryCount="
                    << carryCount << ", iter=" << info.iterIdx
                    << ", epoch=" << info.epochIdx);
          auto expandedCarry =
              buildExpandedDirectRecreateCarry(packOp, carryCount);
          packParams = std::move(expandedCarry.first);
          perm = std::move(expandedCarry.second);
          if (childFunc) {
            auto existingPreserve =
                childFunc->getAttrOfType<IntegerAttr>(CPSPreserveCarryAbi);
            if (!existingPreserve || existingPreserve.getInt() < carryCount) {
              childFunc->setAttr(
                  CPSPreserveCarryAbi,
                  AC->getBuilder().getI64IntegerAttr(carryCount));
            }
          }
        } else {
          packParams.assign(packOp->getOperands().begin(),
                            packOp->getOperands().end());
        }
        packParams.push_back(iterCounter);
        unsigned newIterIdx = packParams.size() - 1;
        packParams.push_back(outerEpochGuid);
        unsigned newEpochIdx = packParams.size() - 1;
        AC->setInsertionPoint(packOp);
        auto packType = MemRefType::get(
            {static_cast<int64_t>(packParams.size())}, AC->Int64);
        auto newPack =
            AC->create<EdtParamPackOp>(packOp->getLoc(), packType, packParams);
        packOp->getResult(0).replaceAllUsesWith(newPack.getMemref());
        packOp->erase();
        setCpsSchemaAttrs(edt, perm, newIterIdx, newEpochIdx);
        attachToOuterEpoch(edt);
        ARTS_INFO("  Injected iter[" << newIterIdx << "], epoch[" << newEpochIdx
                                     << "] for child chain EDT");

        if (auto childFuncAttr = edt->getAttrOfType<StringAttr>(
                AttrNames::Operation::OutlinedFunc)) {
          if (auto outlined =
                  module.lookupSymbol<func::FuncOp>(childFuncAttr.getValue()))
            setCpsSchemaAttrs(outlined, perm, newIterIdx, newEpochIdx);
          enqueueChainFunc(childFuncAttr, newIterIdx, newEpochIdx);
        }
      }

      for (EdtCreateOp edt : directRecreateContinuations) {
        auto packOp = edt.getParamMemref().getDefiningOp<EdtParamPackOp>();
        if (!packOp)
          continue;

        SmallVector<int64_t> perm;
        perm.reserve(packOp.getOperands().size());
        for (Value operand : packOp.getOperands())
          perm.push_back(traceToParamIdx(operand));

        unsigned carryCount = std::max<int64_t>(info.iterIdx, 0);
        ARTS_INFO(
            "CPS propagation: rewriting local direct-recreate continuation "
            << edt << " with carryCount=" << carryCount);
        auto expandedCarry =
            buildExpandedDirectRecreateCarry(packOp, carryCount);
        SmallVector<Value> packParams = std::move(expandedCarry.first);
        perm = std::move(expandedCarry.second);
        packParams.push_back(iterCounter);
        unsigned newIterIdx = packParams.size() - 1;
        packParams.push_back(outerEpochGuid);
        unsigned newEpochIdx = packParams.size() - 1;

        AC->setInsertionPoint(packOp);
        auto packType = MemRefType::get(
            {static_cast<int64_t>(packParams.size())}, AC->Int64);
        auto newPack =
            AC->create<EdtParamPackOp>(packOp->getLoc(), packType, packParams);
        packOp->getResult(0).replaceAllUsesWith(newPack.getMemref());
        packOp->erase();

        setCpsSchemaAttrs(edt, perm, newIterIdx, newEpochIdx);
        attachToOuterEpoch(edt);

        auto childFuncAttr =
            edt->getAttrOfType<StringAttr>(AttrNames::Operation::OutlinedFunc);
        func::FuncOp childFunc =
            childFuncAttr
                ? module.lookupSymbol<func::FuncOp>(childFuncAttr.getValue())
                : func::FuncOp();
        if (childFunc) {
          setCpsSchemaAttrs(childFunc, perm, newIterIdx, newEpochIdx);
          auto existingPreserve =
              childFunc->getAttrOfType<IntegerAttr>(CPSPreserveCarryAbi);
          if (!existingPreserve || existingPreserve.getInt() < carryCount) {
            childFunc->setAttr(CPSPreserveCarryAbi,
                               AC->getBuilder().getI64IntegerAttr(carryCount));
          }
        }
      }
    }
  }

  /// Resolve CPSAdvanceOp placeholders left by the CPS chain transform.
  SmallVector<CPSAdvanceOp> cpsAdvanceOps;
  module.walk([&](CPSAdvanceOp op) { cpsAdvanceOps.push_back(op); });

  if (!cpsAdvanceOps.empty())
    ARTS_INFO("Resolving " << cpsAdvanceOps.size()
                           << " CPS advance placeholder(s)");

  for (CPSAdvanceOp advanceOp : cpsAdvanceOps) {
    StringRef targetChainId = advanceOp.getTargetChainId();

    auto scoreContinuationCandidate = [&](EdtCreateOp edt) -> int {
      int score = 0;
      if (edt->hasAttr(AttrNames::Operation::ControlDep))
        score += 4;
      if (edt->hasAttr(ContinuationForEpoch))
        score += 2;
      for (Operation *user : edt.getGuid().getUsers()) {
        if (auto recordDep = dyn_cast<RecordDepOp>(user))
          if (recordDep.getEdtGuid() == edt.getGuid()) {
            score += 1;
            break;
          }
      }
      return score;
    };

    auto considerContinuationCandidate =
        [&](EdtCreateOp edt, EdtCreateOp &bestEdt, int &bestScore) {
          auto chainAttr = edt->getAttrOfType<StringAttr>(CPSChainId);
          if (!chainAttr || chainAttr.getValue() != targetChainId)
            return;
          int score = scoreContinuationCandidate(edt);
          if (!bestEdt || score > bestScore) {
            bestEdt = edt;
            bestScore = score;
          }
        };

    /// Find the best EdtCreateOp with matching cps_chain_id in the same
    /// function. Prefer the canonical continuation kickoff site that still
    /// carries the control-dep / rec_dep contract over inline self-recreate
    /// sites that only share the same chain id.
    func::FuncOp parentFunc = advanceOp->getParentOfType<func::FuncOp>();
    EdtCreateOp contEdtCreate = nullptr;
    int contEdtScore = std::numeric_limits<int>::min();
    if (parentFunc) {
      parentFunc.walk([&](EdtCreateOp edt) {
        considerContinuationCandidate(edt, contEdtCreate, contEdtScore);
      });
    }

    /// Fall back to module-wide search (target may be in a different
    /// function), but keep the highest-scoring candidate overall.
    module.walk([&](EdtCreateOp edt) {
      considerContinuationCandidate(edt, contEdtCreate, contEdtScore);
    });

    if (!contEdtCreate) {
      advanceOp.emitError("CPS advance: no EdtCreateOp with chain_id '")
          << targetChainId << "' found";
      return signalPassFailure();
    }

    auto outlinedFunc = contEdtCreate->getAttrOfType<StringAttr>(
        AttrNames::Operation::OutlinedFunc);
    if (!outlinedFunc) {
      advanceOp.emitError("CPS advance: continuation has no outlined_func");
      return signalPassFailure();
    }

    ARTS_INFO("CPS advance: resolving chain '" << targetChainId << "' → "
                                               << outlinedFunc.getValue());

    Location loc = advanceOp.getLoc();
    AC->setInsertionPoint(advanceOp);

    /// Re-materialize depCount and route as local constants to avoid
    /// cross-function SSA references.
    Value origDepCount = contEdtCreate.getDepCount();
    int64_t depCountVal = 1; // default: 1 dep (control)
    if (auto cst = origDepCount.getDefiningOp<arith::ConstantIntOp>())
      depCountVal = cst.value();
    else if (auto cst = origDepCount.getDefiningOp<arith::ConstantOp>())
      if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
        depCountVal = intAttr.getInt();
    Value localDepCount = AC->createIntConstant(depCountVal, AC->Int32, loc);

    Value origRoute = contEdtCreate.getRoute();
    int64_t routeVal = 0;
    if (auto cst = origRoute.getDefiningOp<arith::ConstantIntOp>())
      routeVal = cst.value();
    else if (auto cst = origRoute.getDefiningOp<arith::ConstantOp>())
      if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
        routeVal = intAttr.getInt();
    Value localRoute = AC->createIntConstant(routeVal, AC->Int32, loc);

    SmallVector<Value> nextParams(advanceOp.getNextIterParams());

    /// Find the local edt_param_unpack to get this function's params.
    EdtParamUnpackOp localUnpack = nullptr;
    parentFunc.walk([&](EdtParamUnpackOp op) {
      if (!localUnpack)
        localUnpack = op;
    });

    /// Read extra parameters (iter counter + outer epoch GUID) from the raw
    /// parameter memref. These were injected by the CPS propagation above.
    /// IMPORTANT: Read CPS indices from the LOCAL function's own EdtCreateOp,
    /// not from the target chain's EdtCreateOp. Each chain function has its
    /// own CPS param indices based on its own param count.
    Value outerEpochGuid;
    Value curIter;
    IntegerAttr localOuterEpochIdxAttr;
    IntegerAttr localIterCounterIdxAttr;
    DenseI64ArrayAttr localParamPermAttr;
    EdtParamPackOp localSchemaPackOp = nullptr;
    Value localParamArrayMemref;
    if (parentFunc) {
      localIterCounterIdxAttr =
          parentFunc->getAttrOfType<IntegerAttr>(CPSIterCounterParamIdx);
      localOuterEpochIdxAttr =
          parentFunc->getAttrOfType<IntegerAttr>(CPSOuterEpochParamIdx);
      localParamPermAttr =
          parentFunc->getAttrOfType<DenseI64ArrayAttr>(CPSParamPerm);

      EdtCreateOp bestLocalSchemaEdt = nullptr;
      int bestSchemaScore = -1;
      module.walk([&](EdtCreateOp edt) {
        auto funcAttr =
            edt->getAttrOfType<StringAttr>(AttrNames::Operation::OutlinedFunc);
        if (!funcAttr || funcAttr.getValue() != parentFunc.getName())
          return;

        int score = 0;
        if (auto perm = edt->getAttrOfType<DenseI64ArrayAttr>(CPSParamPerm))
          score += 100 + static_cast<int>(perm.size());
        if (edt->getAttrOfType<IntegerAttr>(CPSIterCounterParamIdx))
          score += 10;
        if (edt->getAttrOfType<IntegerAttr>(CPSOuterEpochParamIdx))
          score += 10;
        if (edt->hasAttr(ContinuationForEpoch))
          score += 1;
        if (score <= bestSchemaScore)
          return;

        bestSchemaScore = score;
        bestLocalSchemaEdt = edt;
      });

      if (bestLocalSchemaEdt) {
        if (!localSchemaPackOp)
          localSchemaPackOp = bestLocalSchemaEdt.getParamMemref()
                                  .getDefiningOp<EdtParamPackOp>();
        if (!localIterCounterIdxAttr)
          localIterCounterIdxAttr =
              bestLocalSchemaEdt->getAttrOfType<IntegerAttr>(
                  CPSIterCounterParamIdx);
        if (!localOuterEpochIdxAttr)
          localOuterEpochIdxAttr =
              bestLocalSchemaEdt->getAttrOfType<IntegerAttr>(
                  CPSOuterEpochParamIdx);
        if (!localParamPermAttr)
          localParamPermAttr =
              bestLocalSchemaEdt->getAttrOfType<DenseI64ArrayAttr>(
                  CPSParamPerm);
      }
    }
    IntegerAttr targetIterCounterIdxAttr =
        contEdtCreate->getAttrOfType<IntegerAttr>(CPSIterCounterParamIdx);
    IntegerAttr targetOuterEpochIdxAttr =
        contEdtCreate->getAttrOfType<IntegerAttr>(CPSOuterEpochParamIdx);
    DenseI64ArrayAttr targetParamPermAttr =
        contEdtCreate->getAttrOfType<DenseI64ArrayAttr>(CPSParamPerm);
    if (outlinedFunc) {
      if (auto targetFunc =
              module.lookupSymbol<func::FuncOp>(outlinedFunc.getValue())) {
        if (!targetIterCounterIdxAttr)
          targetIterCounterIdxAttr =
              targetFunc->getAttrOfType<IntegerAttr>(CPSIterCounterParamIdx);
        if (!targetOuterEpochIdxAttr)
          targetOuterEpochIdxAttr =
              targetFunc->getAttrOfType<IntegerAttr>(CPSOuterEpochParamIdx);
        if (!targetParamPermAttr)
          targetParamPermAttr =
              targetFunc->getAttrOfType<DenseI64ArrayAttr>(CPSParamPerm);
      }
    }
    if (!localIterCounterIdxAttr)
      localIterCounterIdxAttr = targetIterCounterIdxAttr;
    if (!localOuterEpochIdxAttr)
      localOuterEpochIdxAttr = targetOuterEpochIdxAttr;
    if (localUnpack) {
      localParamArrayMemref = localUnpack->getOperand(0);
    } else if (parentFunc && parentFunc.getNumArguments() > 1) {
      Value candidate = parentFunc.getArgument(1);
      if (auto memrefType = dyn_cast<MemRefType>(candidate.getType());
          memrefType && memrefType.getElementType().isInteger(64)) {
        localParamArrayMemref = candidate;
      }
    }
    if (localParamArrayMemref) {
      if (localIterCounterIdxAttr) {
        unsigned idx = localIterCounterIdxAttr.getInt();
        Value idxVal = AC->createIndexConstant(idx, loc);
        curIter = AC->create<memref::LoadOp>(loc, localParamArrayMemref,
                                             ValueRange{idxVal});
      }
      if (localOuterEpochIdxAttr) {
        unsigned idx = localOuterEpochIdxAttr.getInt();
        Value idxVal = AC->createIndexConstant(idx, loc);
        outerEpochGuid = AC->create<memref::LoadOp>(loc, localParamArrayMemref,
                                                    ValueRange{idxVal});
      }
    }

    bool isAdditive = advanceOp->hasAttr(CPSAdditiveParams);

    if (!outerEpochGuid) {
      auto outerEpochGuidOp = AC->create<GetEdtEpochGuidOp>(
          loc, IntegerType::get(AC->getContext(), 64));
      outerEpochGuid = outerEpochGuidOp.getEpochGuid();
    }

    bool directRecreate = advanceOp->hasAttr(CPSDirectRecreate);
    int64_t recreatedDepCountVal = depCountVal;
    if (directRecreate &&
        contEdtCreate->hasAttr(AttrNames::Operation::ControlDep) &&
        recreatedDepCountVal > 0) {
      /// Direct recreate re-emits only the recorded DB deps. The epoch-finish
      /// control slot is valid only on the continuation-epoch path that
      /// recreates the finish-target epoch.
      --recreatedDepCountVal;
    }
    Value tNext;
    Value continueCond;
    if (isAdditive && curIter && nextParams.size() >= 2) {
      Value step = nextParams[0];
      Value ub = nextParams[1];
      if (step.getType() != curIter.getType())
        step = AC->create<arith::IndexCastOp>(loc, curIter.getType(), step);
      if (ub.getType() != curIter.getType())
        ub = AC->create<arith::IndexCastOp>(loc, curIter.getType(), ub);
      tNext = AC->create<arith::AddIOp>(loc, curIter, step);
      continueCond =
          AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tNext, ub);
    }

    auto materializeLoopBackParam = [&](Value value) -> Value {
      if (!isa<MemRefType>(value.getType()))
        return value;
      /// Direct-recreate continuations may need stable handle scaffolding
      /// (including DB-family handle arrays) to relaunch nested work. Keep the
      /// existing param ABI for memref-like values and let explicit dep slots
      /// carry the actual runtime dependency edges.
      Value rawPtr = AC->castToLLVMPtr(value, loc);
      return AC->create<LLVM::PtrToIntOp>(loc, AC->Int64, rawPtr);
    };

    bool hasCanonicalLoopBackContract = advanceOp->hasAttr(CPSNumCarry);
    auto targetOriginalPack =
        contEdtCreate.getParamMemref().getDefiningOp<EdtParamPackOp>();

    auto findOriginalRecDep = [&](EdtCreateOp edt) -> RecordDepOp {
      for (Operation *user : edt.getGuid().getUsers())
        if (auto recordDep = dyn_cast<RecordDepOp>(user))
          if (recordDep.getEdtGuid() == edt.getGuid())
            return recordDep;
      return nullptr;
    };
    RecordDepOp targetOriginalRecDep = findOriginalRecDep(contEdtCreate);

    /// Build the self-recreation param pack using the canonical carry
    /// params from CPSAdvanceOp. EpochOpt step CPS-8 ensures carry params
    /// are in the EXACT same order as EdtLowering::packParams, so we can
    /// use them directly. materializeLoopBackParam converts memref
    /// dbHandles to raw i64, matching packParams' conversion.
    SmallVector<Value> canonicalLoopBackParams;
    if (auto numCarryAttr =
            advanceOp->getAttrOfType<IntegerAttr>(CPSNumCarry)) {
      unsigned paramStart = (isAdditive && nextParams.size() >= 2) ? 2 : 0;
      unsigned available =
          nextParams.size() > paramStart ? nextParams.size() - paramStart : 0;
      unsigned carryCount =
          std::min<unsigned>(numCarryAttr.getInt(), available);
      canonicalLoopBackParams.reserve(carryCount);
      for (unsigned i = 0; i < carryCount; ++i)
        canonicalLoopBackParams.push_back(
            materializeLoopBackParam(nextParams[paramStart + i]));
    }

    auto rebuildCpsPackToTargetSchema = [&]() -> SmallVector<Value> {
      SmallVector<Value> newPackParams;
      if (!hasCanonicalLoopBackContract)
        return newPackParams;

      /// Plan-aware path: when the new launch state schema is present,
      /// use it to interpret the carry param layout structurally instead
      /// of relying on positional CPSParamPerm archaeology.
      int64_t schemaNumScalars = 0;
      int64_t schemaNumDataGuids = 0;
      int64_t schemaNumDataPtrs = 0;
      if (auto stateSchema = contEdtCreate->getAttrOfType<DenseI64ArrayAttr>(
              AttrNames::Operation::LaunchState::StateSchema)) {
        ArrayRef<int64_t> schema = stateSchema.asArrayRef();
        /// schema format: [numScalars, numTimingDbs, numDataGuids, numDataPtrs]
        schemaNumScalars = schema.size() > 0 ? schema[0] : 0;
        schemaNumDataGuids = schema.size() > 2 ? schema[2] : 0;
        schemaNumDataPtrs = schema.size() > 3 ? schema[3] : 0;
        int64_t numTotal = 0;
        for (int64_t v : schema)
          numTotal += v;
        /// Use canonical loop-back params directly; the schema just
        /// documents the layout for debugging and downstream consumers.
        ARTS_DEBUG("CPS advance: using launch state schema — "
                   << schemaNumScalars << " scalars, " << numTotal
                   << " total");
        /// Fall through to canonical path below — the schema is advisory.
      }
      int64_t schemaHasScratchDep = 0;
      int64_t schemaNumTimingDepSlots = 0;
      if (auto depSchema = contEdtCreate->getAttrOfType<DenseI64ArrayAttr>(
              AttrNames::Operation::LaunchState::DepSchema)) {
        ArrayRef<int64_t> schema = depSchema.asArrayRef();
        schemaNumTimingDepSlots = schema.size() > 0 ? schema[0] : 0;
        schemaHasScratchDep = schema.size() > 1 ? schema[1] : 0;
      }

      auto isTargetSlotTypeCompatible = [&](unsigned slot, Value candidate) {
        if (!candidate)
          return false;
        if (!targetOriginalPack || slot >= targetOriginalPack.getNumOperands())
          return true;
        Type expectedType = targetOriginalPack.getOperand(slot).getType();
        return candidate.getType() == expectedType;
      };

      if (parentFunc && localUnpack &&
          outlinedFunc.getValue() == parentFunc.getName()) {
        auto unpackedValues = localUnpack.getResults();
        unsigned localCarryCount = unpackedValues.size();
        if (localIterCounterIdxAttr)
          localCarryCount = std::min<unsigned>(
              localCarryCount, localIterCounterIdxAttr.getInt());
        else if (localOuterEpochIdxAttr)
          localCarryCount = std::min<unsigned>(localCarryCount,
                                               localOuterEpochIdxAttr.getInt());

        unsigned totalPackSlots = localCarryCount;
        if (targetIterCounterIdxAttr)
          totalPackSlots = std::max<unsigned>(
              totalPackSlots, targetIterCounterIdxAttr.getInt() + 1);
        if (targetOuterEpochIdxAttr)
          totalPackSlots = std::max<unsigned>(
              totalPackSlots, targetOuterEpochIdxAttr.getInt() + 1);

        Value zeroI64 = AC->createIntConstant(0, AC->Int64, loc);
        newPackParams.assign(totalPackSlots, zeroI64);
        for (unsigned i = 0; i < localCarryCount && i < totalPackSlots; ++i) {
          bool unsafeCarry = false;
          if (localSchemaPackOp && i < localSchemaPackOp.getNumOperands()) {
            // Only values that trace through a DbAcquireOp are unsafe to
            // carry — they are data pointers that may be invalidated by
            // release/reacquire across iterations. Values that trace
            // directly to DbAllocOp results (guid/ptr handle arrays for
            // block-partitioned DBs) are stable malloc'd arrays and safe
            // to carry across CPS iterations.
            Operation *underlyingDb =
                DbUtils::getUnderlyingDb(localSchemaPackOp.getOperand(i));
            unsafeCarry = underlyingDb && isa<DbAcquireOp>(underlyingDb);
          }
          if (!unsafeCarry)
            newPackParams[i] = unpackedValues[i];
        }

        Value iterValue = tNext ? tNext : curIter;
        if (targetIterCounterIdxAttr && iterValue) {
          unsigned iterIdx = targetIterCounterIdxAttr.getInt();
          if (iterIdx < newPackParams.size())
            newPackParams[iterIdx] = iterValue;
        }
        if (targetOuterEpochIdxAttr && outerEpochGuid) {
          unsigned epochIdx = targetOuterEpochIdxAttr.getInt();
          if (epochIdx < newPackParams.size())
            newPackParams[epochIdx] = outerEpochGuid;
        }
        return newPackParams;
      }

      unsigned targetCarryCount = canonicalLoopBackParams.size();
      if (targetParamPermAttr)
        targetCarryCount = targetParamPermAttr.size();
      else if (targetIterCounterIdxAttr)
        targetCarryCount = targetIterCounterIdxAttr.getInt();
      else if (targetOuterEpochIdxAttr)
        targetCarryCount = targetOuterEpochIdxAttr.getInt();

      unsigned totalPackSlots = targetCarryCount;
      if (targetIterCounterIdxAttr)
        totalPackSlots = std::max<unsigned>(
            totalPackSlots, targetIterCounterIdxAttr.getInt() + 1);
      if (targetOuterEpochIdxAttr)
        totalPackSlots = std::max<unsigned>(
            totalPackSlots, targetOuterEpochIdxAttr.getInt() + 1);
      if (totalPackSlots == 0)
        return newPackParams;

      auto resolveCanonicalIndex = [](ArrayRef<int64_t> perm,
                                      unsigned slot) -> int64_t {
        if (slot < perm.size() && perm[slot] >= 0)
          return perm[slot];
        return slot;
      };

      int64_t canonicalWidth = canonicalLoopBackParams.size();
      auto growCanonicalWidth = [&](ArrayRef<int64_t> perm) {
        for (int64_t idx : perm) {
          if (idx >= 0)
            canonicalWidth = std::max<int64_t>(canonicalWidth, idx + 1);
        }
      };
      if (localParamPermAttr)
        growCanonicalWidth(localParamPermAttr.asArrayRef());
      if (targetParamPermAttr)
        growCanonicalWidth(targetParamPermAttr.asArrayRef());

      SmallVector<Value> canonicalValues(std::max<int64_t>(canonicalWidth, 0));
      ArrayRef<int64_t> localPerm = localParamPermAttr
                                        ? localParamPermAttr.asArrayRef()
                                        : ArrayRef<int64_t>();
      ArrayRef<int64_t> targetPerm = targetParamPermAttr
                                         ? targetParamPermAttr.asArrayRef()
                                         : ArrayRef<int64_t>();
      auto sameCanonicalValue = [](Value lhs, Value rhs) {
        if (!lhs || !rhs)
          return false;
        if (lhs == rhs)
          return true;
        return ValueAnalysis::stripNumericCasts(lhs) ==
               ValueAnalysis::stripNumericCasts(rhs);
      };
      for (unsigned i = 0; i < canonicalLoopBackParams.size(); ++i) {
        int64_t canonicalIdx = resolveCanonicalIndex(localPerm, i);
        if (canonicalIdx < 0)
          continue;
        if (static_cast<size_t>(canonicalIdx) >= canonicalValues.size())
          canonicalValues.resize(canonicalIdx + 1);
        canonicalValues[canonicalIdx] = canonicalLoopBackParams[i];
      }

      IntegerAttr preserveCarryAbiAttr =
          parentFunc
              ? parentFunc->getAttrOfType<IntegerAttr>(CPSPreserveCarryAbi)
              : IntegerAttr();
      bool preserveIncomingCarryAbi = localParamArrayMemref &&
                                      preserveCarryAbiAttr &&
                                      preserveCarryAbiAttr.getInt() > 0;

      SmallVector<Value> preservedIncomingCarry(
          std::max<int64_t>(canonicalWidth, 0));
      if (preserveIncomingCarryAbi) {
        unsigned preservedCarryCount =
            std::max<int64_t>(preserveCarryAbiAttr.getInt(), 0);
        if (localParamPermAttr)
          preservedCarryCount = std::max<unsigned>(preservedCarryCount,
                                                   localParamPermAttr.size());
        for (unsigned i = 0; i < preservedCarryCount; ++i) {
          int64_t canonicalIdx = resolveCanonicalIndex(localPerm, i);
          if (canonicalIdx < 0)
            continue;
          if (static_cast<size_t>(canonicalIdx) >=
              preservedIncomingCarry.size())
            preservedIncomingCarry.resize(canonicalIdx + 1);
          Value slotIdx = AC->createIndexConstant(i, loc);
          preservedIncomingCarry[canonicalIdx] = AC->create<memref::LoadOp>(
              loc, localParamArrayMemref, ValueRange{slotIdx});
        }
      }

      auto computeLocalExplicitCarryCount = [&]() -> unsigned {
        if (localUnpack) {
          unsigned localCarryCount = localUnpack.getNumResults();
          if (hasCanonicalLoopBackContract)
            localCarryCount = std::min<unsigned>(
                localCarryCount, canonicalLoopBackParams.size());
          if (localParamPermAttr)
            localCarryCount =
                std::min<unsigned>(localCarryCount, localParamPermAttr.size());
          else if (localIterCounterIdxAttr)
            localCarryCount = std::min<unsigned>(
                localCarryCount, localIterCounterIdxAttr.getInt());
          else if (localOuterEpochIdxAttr)
            localCarryCount = std::min<unsigned>(
                localCarryCount, localOuterEpochIdxAttr.getInt());
          return localCarryCount;
        }

        if (localParamArrayMemref) {
          unsigned localCarryCount = canonicalLoopBackParams.size();
          if (localParamPermAttr)
            localCarryCount =
                std::min<unsigned>(localCarryCount, localParamPermAttr.size());
          else if (localIterCounterIdxAttr)
            localCarryCount = std::min<unsigned>(
                localCarryCount, localIterCounterIdxAttr.getInt());
          else if (localOuterEpochIdxAttr)
            localCarryCount = std::min<unsigned>(
                localCarryCount, localOuterEpochIdxAttr.getInt());
          return localCarryCount;
        }

        return 0;
      };

      unsigned localExplicitCarryCount = computeLocalExplicitCarryCount();

      SmallVector<Value> localCarryFallback(
          std::max<int64_t>(canonicalWidth, 0));
      if (localUnpack) {
        auto unpackedValues = localUnpack.getResults();
        for (unsigned i = 0; i < localExplicitCarryCount; ++i) {
          int64_t canonicalIdx = resolveCanonicalIndex(localPerm, i);
          if (canonicalIdx < 0)
            continue;
          if (static_cast<size_t>(canonicalIdx) >= localCarryFallback.size())
            localCarryFallback.resize(canonicalIdx + 1);
          localCarryFallback[canonicalIdx] =
              materializeLoopBackParam(unpackedValues[i]);
        }
      } else if (localParamArrayMemref) {
        for (unsigned i = 0; i < localExplicitCarryCount; ++i) {
          int64_t canonicalIdx = resolveCanonicalIndex(localPerm, i);
          if (canonicalIdx < 0)
            continue;
          if (static_cast<size_t>(canonicalIdx) >= localCarryFallback.size())
            localCarryFallback.resize(canonicalIdx + 1);
          Value slotIdx = AC->createIndexConstant(i, loc);
          localCarryFallback[canonicalIdx] = AC->create<memref::LoadOp>(
              loc, localParamArrayMemref, ValueRange{slotIdx});
        }
      }

      DenseMap<Operation *, unsigned> targetDepRootCounts;
      if (targetOriginalRecDep) {
        for (Value depGuid : targetOriginalRecDep.getDatablocks())
          if (Operation *root = DbUtils::getUnderlyingDbAlloc(depGuid))
            ++targetDepRootCounts[root];
      }

      auto resolveTargetDepSlotForPackSlot =
          [&](unsigned packSlot) -> std::optional<unsigned> {
        if (!targetOriginalPack || !targetOriginalRecDep ||
            packSlot >= targetOriginalPack.getNumOperands())
          return std::nullopt;
        Operation *packRoot = DbUtils::getUnderlyingDbAlloc(
            targetOriginalPack.getOperand(packSlot));
        if (!packRoot || targetDepRootCounts.lookup(packRoot) != 1)
          return std::nullopt;

        for (auto [depIdx, depGuid] :
             llvm::enumerate(targetOriginalRecDep.getDatablocks())) {
          if (DbUtils::getUnderlyingDbAlloc(depGuid) == packRoot)
            return static_cast<unsigned>(depIdx);
        }
        return std::nullopt;
      };

      auto resolveLocalDepSlot = [&](Value sourcePtr) -> std::optional<unsigned> {
        sourcePtr = ValueAnalysis::stripMemrefViewOps(sourcePtr);
        if (auto depAcquire = sourcePtr.getDefiningOp<DepDbAcquireOp>()) {
          if (auto offset =
                  ValueAnalysis::getConstantIndexStripped(depAcquire.getOffset()))
            return static_cast<unsigned>(*offset);
        }
        if (auto blockArg = dyn_cast<BlockArgument>(sourcePtr))
          return blockArg.getArgNumber();
        return std::nullopt;
      };

      struct LocalDbGuidCarryCandidate {
        Value value;
        std::optional<unsigned> depSlot;
      };
      SmallVector<LocalDbGuidCarryCandidate, 4> localDbGuidCarryCandidates;
      advanceOp.walk([&](DbAcquireOp acquire) {
        Value sourceGuid = acquire.getSourceGuid();
        if (!sourceGuid)
          return;
        auto sourceType = dyn_cast<MemRefType>(sourceGuid.getType());
        if (!sourceType || !sourceType.getElementType().isInteger(64))
          return;

        Value materialized = materializeLoopBackParam(sourceGuid);
        Value stripped = ValueAnalysis::stripNumericCasts(materialized);
        std::optional<unsigned> depSlot =
            resolveLocalDepSlot(acquire.getSourcePtr());
        bool seen = llvm::any_of(localDbGuidCarryCandidates, [&](auto candidate) {
          return ValueAnalysis::stripNumericCasts(candidate.value) == stripped &&
                 candidate.depSlot == depSlot;
        });
        if (!seen)
          localDbGuidCarryCandidates.push_back({materialized, depSlot});
      });

      struct TargetDataGuidSlot {
        unsigned packSlot = 0;
        int64_t partitionCount = 0;
        int64_t scratchOffset = 0;
        DbAllocOp alloc;
      };
      SmallVector<TargetDataGuidSlot, 4> targetDataGuidSlots;
      auto stripPackedDbHandle = [&](Value value) {
        for (unsigned depth = 0; depth < 16 && value; ++depth) {
          if (auto ptrToInt = value.getDefiningOp<LLVM::PtrToIntOp>()) {
            value = ptrToInt.getArg();
            continue;
          }
          if (auto intToPtr = value.getDefiningOp<LLVM::IntToPtrOp>()) {
            value = intToPtr.getArg();
            continue;
          }
          if (auto memrefToPtr =
                  value.getDefiningOp<polygeist::Memref2PointerOp>()) {
            value = memrefToPtr.getSource();
            continue;
          }
          if (auto ptrToMemref =
                  value.getDefiningOp<polygeist::Pointer2MemrefOp>()) {
            value = ptrToMemref.getSource();
            continue;
          }
          if (auto castOp = value.getDefiningOp<memref::CastOp>()) {
            value = castOp.getSource();
            continue;
          }
          break;
        }
        return ValueAnalysis::stripMemrefViewOps(value);
      };

      int64_t nextScratchOffset = 0;
      if (targetOriginalPack && schemaNumDataGuids > 0) {
        for (auto [slot, operand] :
             llvm::enumerate(targetOriginalPack.getOperands())) {
          Value handle = stripPackedDbHandle(operand);
          auto handleType = handle ? dyn_cast<MemRefType>(handle.getType())
                                   : MemRefType();
          if (!handleType || !handleType.getElementType().isInteger(64))
            continue;

          DbAllocOp alloc =
              dyn_cast_or_null<DbAllocOp>(DbUtils::getUnderlyingDbAlloc(handle));
          if (!alloc)
            continue;
          Value allocGuid = ValueAnalysis::stripMemrefViewOps(alloc.getGuid());
          if (handle != allocGuid)
            continue;

          int64_t partitionCount = DbUtils::computeStaticPartitionCount(alloc);
          if (partitionCount <= 1)
            continue;

          targetDataGuidSlots.push_back(
              {static_cast<unsigned>(slot), partitionCount, nextScratchOffset,
               alloc});
          nextScratchOffset += partitionCount;
          if (static_cast<int64_t>(targetDataGuidSlots.size()) >=
              schemaNumDataGuids)
            break;
        }
      }

      SmallVector<Value, 4> synthesizedScratchGuidCarries(
          targetDataGuidSlots.size());
      Value synthesizedScratchCarry;
      Value scratchDepPtr;
      auto getScratchDepPtr = [&]() -> Value {
        if (scratchDepPtr || !parentFunc || parentFunc.getNumArguments() <= 3 ||
            schemaHasScratchDep <= 0)
          return scratchDepPtr;

        unsigned scratchSlot = static_cast<unsigned>(
            std::max<int64_t>(schemaNumTimingDepSlots, 0) +
            std::max<int64_t>(schemaNumDataPtrs, 0));
        Value depv = parentFunc.getArgument(3);
        Value scratchSlotVal = AC->createIndexConstant(scratchSlot, loc);
        auto i64MemrefTy =
            MemRefType::get({ShapedType::kDynamic}, AC->Int64);
        auto scratchAcquire = AC->create<DepDbAcquireOp>(
            loc, i64MemrefTy, i64MemrefTy, depv, scratchSlotVal,
            SmallVector<Value>{}, SmallVector<Value>{}, SmallVector<Value>{},
            Value());
        scratchDepPtr = scratchAcquire.getPtr();
        return scratchDepPtr;
      };

      auto synthesizeScratchCarry = [&]() -> Value {
        if (synthesizedScratchCarry)
          return synthesizedScratchCarry;
        if (schemaHasScratchDep <= 0 || nextScratchOffset <= 0)
          return Value();

        Value scratchPtr = getScratchDepPtr();
        if (!scratchPtr)
          return Value();

        auto localType = MemRefType::get({nextScratchOffset}, AC->Int64);
        auto localScratch = AC->create<memref::AllocOp>(loc, localType);
        setDbStaticOuterShape(localScratch.getOperation(), {nextScratchOffset});

        Value zero = AC->createIndexConstant(0, loc);
        Value one = AC->createIndexConstant(1, loc);
        Value count = AC->createIndexConstant(nextScratchOffset, loc);
        auto copyLoop = AC->create<scf::ForOp>(loc, zero, count, one);
        {
          OpBuilder::InsertionGuard guard(AC->getBuilder());
          Block *loopBody = copyLoop.getBody();
          AC->getBuilder().setInsertionPoint(loopBody->getTerminator());
          Value i = copyLoop.getInductionVar();
          Value guid = AC->create<memref::LoadOp>(loc, scratchPtr,
                                                  ValueRange{i});
          AC->create<memref::StoreOp>(loc, guid, localScratch.getResult(),
                                      ValueRange{i});
        }

        synthesizedScratchCarry =
            materializeLoopBackParam(localScratch.getResult());
        return synthesizedScratchCarry;
      };

      auto synthesizeGuidCarryFromScratch = [&](unsigned dataIdx) -> Value {
        if (dataIdx >= targetDataGuidSlots.size())
          return Value();
        if (synthesizedScratchGuidCarries[dataIdx])
          return synthesizedScratchGuidCarries[dataIdx];

        Value scratchPtr = getScratchDepPtr();
        if (!scratchPtr)
          return Value();

        TargetDataGuidSlot slot = targetDataGuidSlots[dataIdx];
        if (slot.partitionCount <= 0)
          return Value();

        auto localType = MemRefType::get({slot.partitionCount}, AC->Int64);
        auto localGuid = AC->create<memref::AllocOp>(loc, localType);
        if (auto rootId = DbUtils::resolveRootAllocId(slot.alloc.getGuid(),
                                                      slot.alloc))
          setDbRootAllocId(localGuid.getOperation(), *rootId);
        setDbStaticOuterShape(localGuid.getOperation(), {slot.partitionCount});

        Value zero = AC->createIndexConstant(0, loc);
        Value one = AC->createIndexConstant(1, loc);
        Value count = AC->createIndexConstant(slot.partitionCount, loc);
        Value base = AC->createIndexConstant(slot.scratchOffset, loc);
        auto copyLoop = AC->create<scf::ForOp>(loc, zero, count, one);
        {
          OpBuilder::InsertionGuard guard(AC->getBuilder());
          Block *loopBody = copyLoop.getBody();
          AC->getBuilder().setInsertionPoint(loopBody->getTerminator());
          Value i = copyLoop.getInductionVar();
          Value scratchIdx = AC->create<arith::AddIOp>(loc, base, i);
          Value guid =
              AC->create<memref::LoadOp>(loc, scratchPtr,
                                         ValueRange{scratchIdx});
          AC->create<memref::StoreOp>(loc, guid, localGuid.getResult(),
                                      ValueRange{i});
        }

        Value replacement = localGuid.getResult();
        auto dynGuidTy = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
        if (replacement.getType() != dynGuidTy) {
          auto castOp =
              AC->create<memref::CastOp>(loc, dynGuidTy, replacement);
          if (auto rootId = DbUtils::resolveRootAllocId(slot.alloc.getGuid(),
                                                        slot.alloc))
            setDbRootAllocId(castOp.getOperation(), *rootId);
          setDbStaticOuterShape(castOp.getOperation(),
                                {slot.partitionCount});
          replacement = castOp.getResult();
        }

        Value materialized = materializeLoopBackParam(replacement);
        synthesizedScratchGuidCarries[dataIdx] = materialized;
        return materialized;
      };

      auto findScratchGuidForTargetSlot = [&](unsigned slot) -> Value {
        for (auto [dataIdx, dataSlot] :
             llvm::enumerate(targetDataGuidSlots)) {
          if (dataSlot.packSlot == slot)
            return synthesizeGuidCarryFromScratch(dataIdx);
        }
        return Value();
      };

      auto findLocalDbGuidForTargetSlot = [&](unsigned slot) -> Value {
        if (auto targetDepSlot = resolveTargetDepSlotForPackSlot(slot)) {
          for (const auto &candidate : localDbGuidCarryCandidates)
            if (candidate.depSlot && *candidate.depSlot == *targetDepSlot)
              return candidate.value;
        }
        if (Value scratchGuid = findScratchGuidForTargetSlot(slot))
          return scratchGuid;
        if (slot < localDbGuidCarryCandidates.size())
          return localDbGuidCarryCandidates[slot].value;
        return Value();
      };

      Value zeroI64 = AC->createIntConstant(0, AC->Int64, loc);
      newPackParams.assign(totalPackSlots, zeroI64);

      unsigned missingCarrySlots = 0;
      for (unsigned slot = 0; slot < targetCarryCount; ++slot) {
        int64_t canonicalIdx = resolveCanonicalIndex(targetPerm, slot);
        bool preserveStructuralCarrySlot =
            directRecreate && preserveIncomingCarryAbi && canonicalIdx >= 0 &&
            static_cast<unsigned>(canonicalIdx) >= localExplicitCarryCount &&
            static_cast<size_t>(canonicalIdx) < preservedIncomingCarry.size() &&
            preservedIncomingCarry[canonicalIdx] &&
            isTargetSlotTypeCompatible(slot,
                                       preservedIncomingCarry[canonicalIdx]);

        /// Direct-recreate continuations can compact away kickoff-only carry
        /// slots locally while still needing the original kickoff ABI on
        /// relaunch. When that happens, treat the preserved incoming param
        /// slots as the source of truth for those structural slots instead of
        /// recomputing them from current dep state.
        if (preserveStructuralCarrySlot) {
          newPackParams[slot] = preservedIncomingCarry[canonicalIdx];
          continue;
        }

        if (targetOriginalPack && slot < targetOriginalPack.getNumOperands()) {
          Value originalOperand = targetOriginalPack.getOperand(slot);
          bool matchedOriginalSlot = false;
          for (unsigned canonicalSlot = 0;
               canonicalSlot < canonicalLoopBackParams.size();
               ++canonicalSlot) {
            Value canonicalValue = canonicalLoopBackParams[canonicalSlot];
            if (!sameCanonicalValue(originalOperand, canonicalValue))
              continue;
            newPackParams[slot] = canonicalValue;
            matchedOriginalSlot = true;
            break;
          }
          if (matchedOriginalSlot)
            continue;
        }

        if (schemaHasScratchDep > 0 && canonicalIdx == 0) {
          if (Value scratchCarry = synthesizeScratchCarry();
              scratchCarry && isTargetSlotTypeCompatible(slot, scratchCarry)) {
            newPackParams[slot] = scratchCarry;
            continue;
          }
        }

        if (canonicalIdx >= 0 &&
            static_cast<size_t>(canonicalIdx) < canonicalValues.size() &&
            canonicalValues[canonicalIdx] &&
            isTargetSlotTypeCompatible(slot, canonicalValues[canonicalIdx])) {
          newPackParams[slot] = canonicalValues[canonicalIdx];
          continue;
        }
        if (!targetParamPermAttr && slot < canonicalLoopBackParams.size()) {
          if (canonicalLoopBackParams[slot] &&
              isTargetSlotTypeCompatible(slot, canonicalLoopBackParams[slot])) {
            newPackParams[slot] = canonicalLoopBackParams[slot];
            continue;
          }
        }
        if (canonicalIdx >= 0 &&
            static_cast<size_t>(canonicalIdx) < localCarryFallback.size() &&
            localCarryFallback[canonicalIdx] &&
            isTargetSlotTypeCompatible(slot,
                                       localCarryFallback[canonicalIdx])) {
          newPackParams[slot] = localCarryFallback[canonicalIdx];
          continue;
        }
        if (preserveIncomingCarryAbi && canonicalIdx >= 0 &&
            static_cast<size_t>(canonicalIdx) < preservedIncomingCarry.size() &&
            preservedIncomingCarry[canonicalIdx] &&
            isTargetSlotTypeCompatible(slot,
                                       preservedIncomingCarry[canonicalIdx])) {
          /// Preserved incoming ABI slots are a structural fallback for target
          /// schema holes. Fresh loop-back values from the current iteration
          /// must win whenever they are available.
          newPackParams[slot] = preservedIncomingCarry[canonicalIdx];
          continue;
        }
        if (Value localDbGuidCandidate = findLocalDbGuidForTargetSlot(slot);
            localDbGuidCandidate &&
            isTargetSlotTypeCompatible(slot, localDbGuidCandidate)) {
          /// Some CPS stencil continuations rebuild full GUID tables locally
          /// from a compact table dependency, then relaunch the sibling chain
          /// whose kickoff ABI still has those table handles as structural
          /// slots. Use the rebuilt table whose dependency slot matches the
          /// target ABI instead of relying on acquire walk order.
          newPackParams[slot] = localDbGuidCandidate;
          continue;
        }
        ++missingCarrySlots;
      }

      Value iterValue = tNext ? tNext : curIter;
      if (targetIterCounterIdxAttr && iterValue) {
        unsigned iterIdx = targetIterCounterIdxAttr.getInt();
        if (iterIdx < newPackParams.size())
          newPackParams[iterIdx] = iterValue;
      }
      if (targetOuterEpochIdxAttr && outerEpochGuid) {
        unsigned epochIdx = targetOuterEpochIdxAttr.getInt();
        if (epochIdx < newPackParams.size())
          newPackParams[epochIdx] = outerEpochGuid;
      }

      if (missingCarrySlots > 0) {
        ARTS_WARN("CPS advance: rebuilt continuation pack for "
                  << outlinedFunc.getValue() << " with " << missingCarrySlots
                  << " schema hole(s); preserving slot count with zero fill"
                  << " (targetCarryCount=" << targetCarryCount
                  << ", canonicalCarry=" << canonicalLoopBackParams.size()
                  << ")");
      }

      return newPackParams;
    };

    struct DeferredDepEmit {
      enum class Kind { WholeDb, ForwardedSlot };
      Kind kind;
      Value depGuid;
      unsigned fromSlot = 0;
      unsigned toSlot = 0;
      int32_t mode = 0;
      std::optional<int32_t> depFlags;
    };

    struct ContinuationEpochPlan {
      Value epochGuid;
      Value contGuid;
      SmallVector<DeferredDepEmit, 8> deferredDeps;
    };

    auto emitWholeDbDep = [&](Value targetGuid, Value depGuid, unsigned slotIdx,
                              int32_t mode, std::optional<int32_t> depFlags) {
      if (!targetGuid || !depGuid)
        return;
      depGuid = AC->ensureI64(depGuid, loc);
      Value slot = AC->createIntConstant(slotIdx, AC->Int32, loc);
      Value modeVal = AC->createIntConstant(mode, AC->Int32, loc);
      ArtsCodegen::RuntimeCallBuilder RCB(*AC, loc);
      if (depFlags && *depFlags != 0) {
        Value flagsVal = AC->createIntConstant(*depFlags, AC->Int32, loc);
        RCB.callVoid(types::ARTSRTL_arts_add_dependence_ex,
                     {depGuid, targetGuid, slot, modeVal, flagsVal});
      } else {
        RCB.callVoid(types::ARTSRTL_arts_add_dependence,
                     {depGuid, targetGuid, slot, modeVal});
      }
    };

    auto emitForwardedDepSlot = [&](Value targetGuid, unsigned fromSlot,
                                    unsigned toSlot, int32_t mode,
                                    std::optional<int32_t> depFlags) {
      if (!targetGuid || !parentFunc || parentFunc.getNumArguments() <= 3)
        return;
      Value depv = parentFunc.getArgument(3);
      Value fromSlotIdx = AC->createIndexConstant(fromSlot, loc);
      SmallVector<Value> emptyArgs;
      auto depGep = AC->create<DepGepOp>(loc, AC->llvmPtr, AC->llvmPtr, depv,
                                         fromSlotIdx, emptyArgs, emptyArgs);
      Value guidPtr = depGep.getGuid();
      Value depGuid = AC->create<LLVM::LoadOp>(loc, AC->Int64, guidPtr);
      emitWholeDbDep(targetGuid, depGuid, toSlot, mode, depFlags);
    };

    auto createContinuationLaunch = [&]() -> ContinuationEpochPlan {
      ContinuationEpochPlan plan;
      Value paramMemref;
      auto findOriginalRecDep = [&](EdtCreateOp edt) -> RecordDepOp {
        for (Operation *user : edt.getGuid().getUsers())
          if (auto recordDep = dyn_cast<RecordDepOp>(user))
            if (recordDep.getEdtGuid() == edt.getGuid())
              return recordDep;
        return nullptr;
      };
      // Direct relaunch still has to recreate the target kickoff ABI, not the
      // compact local continuation ABI. Otherwise relaunched params shift when
      // the local continuation dropped target-only carry slots.
      SmallVector<Value> schemaPackParams = rebuildCpsPackToTargetSchema();
      if (!schemaPackParams.empty()) {
        auto packType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
        paramMemref =
            AC->create<EdtParamPackOp>(loc, packType, schemaPackParams)
                .getMemref();
      } else if (localUnpack) {
        auto unpackResults = localUnpack.getResults();
        SmallVector<Value> newPackParams;
        newPackParams.reserve(unpackResults.size() + 2);
        for (Value unpacked : unpackResults)
          newPackParams.push_back(unpacked);
        if (tNext)
          newPackParams.push_back(tNext);
        else if (curIter)
          newPackParams.push_back(curIter);
        if (outerEpochGuid)
          newPackParams.push_back(outerEpochGuid);
        auto packType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
        paramMemref = AC->create<EdtParamPackOp>(loc, packType, newPackParams)
                          .getMemref();
      } else {
        SmallVector<Value> packVals;
        if (tNext)
          packVals.push_back(tNext);
        else if (curIter)
          packVals.push_back(curIter);
        if (outerEpochGuid)
          packVals.push_back(outerEpochGuid);
        auto packType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
        paramMemref =
            AC->create<EdtParamPackOp>(loc, packType, packVals).getMemref();
      }

      auto originalRecDep = findOriginalRecDep(contEdtCreate);
      unsigned originalDepCount =
          originalRecDep ? originalRecDep.getDatablocks().size() : 0;
      Value recreatedDepCount = localDepCount;
      if (directRecreate) {
        int64_t directDepCountVal = originalRecDep
                                        ? static_cast<int64_t>(originalDepCount)
                                        : recreatedDepCountVal;
        recreatedDepCount =
            AC->createIntConstant(directDepCountVal, AC->Int32, loc);
      }

      auto newContEdt = AC->create<EdtCreateOp>(
          loc, paramMemref, recreatedDepCount, localRoute, outerEpochGuid);
      setOutlinedFunc(newContEdt, outlinedFunc.getValue());
      if (!directRecreate)
        newContEdt->setAttr(ContinuationForEpoch,
                            AC->getBuilder().getUnitAttr());
      if (contEdtCreate->hasAttr(CPSForwardDeps))
        newContEdt->setAttr(CPSForwardDeps, AC->getBuilder().getUnitAttr());
      if (auto chainIdAttr =
              contEdtCreate->getAttrOfType<StringAttr>(CPSChainId))
        newContEdt->setAttr(CPSChainId, chainIdAttr);
      if (targetIterCounterIdxAttr)
        newContEdt->setAttr(CPSIterCounterParamIdx, targetIterCounterIdxAttr);
      if (targetOuterEpochIdxAttr)
        newContEdt->setAttr(CPSOuterEpochParamIdx, targetOuterEpochIdxAttr);
      if (targetParamPermAttr)
        newContEdt->setAttr(CPSParamPerm, targetParamPermAttr);

      Value newContGuid = newContEdt.getGuid();
      plan.contGuid = newContGuid;

      auto getLoopBackValueForPackIndex = [&](unsigned packIdx) -> Value {
        if (packIdx < canonicalLoopBackParams.size())
          return canonicalLoopBackParams[packIdx];
        if (localUnpack && packIdx < localUnpack.getNumResults())
          return localUnpack.getResult(packIdx);
        return Value();
      };

      auto findPackIndexForSafeWholeDbDep =
          [&](EdtParamPackOp packOp, Value depGuid) -> std::optional<unsigned> {
        if (!packOp || !depGuid)
          return std::nullopt;

        SmallVector<std::pair<Value, Value>> candidateSources;
        candidateSources.push_back({depGuid, Value()});

        if (auto depAcquire = depGuid.getDefiningOp<DbAcquireOp>()) {
          if (depAcquire.getSizes().size() == 1 &&
              depAcquire.getOffsets().size() == 1) {
            if (auto foldedSize = ValueAnalysis::tryFoldConstantIndex(
                    depAcquire.getSizes()[0]);
                foldedSize && *foldedSize == 1) {
              candidateSources.push_back(
                  {depAcquire.getSourceGuid(), depAcquire.getOffsets()[0]});
            }
          }
        }

        auto sameIndex = [](Value lhs, Value rhs) {
          return ValueAnalysis::stripNumericCasts(lhs) ==
                 ValueAnalysis::stripNumericCasts(rhs);
        };

        for (auto [idx, operand] : llvm::enumerate(packOp.getOperands())) {
          for (auto [sourceGuid, offset] : candidateSources) {
            if (operand == sourceGuid && !offset)
              return idx;
            auto loadOp = operand.getDefiningOp<memref::LoadOp>();
            if (!loadOp || loadOp.getIndices().size() != 1)
              continue;
            if (loadOp.getMemRef() != sourceGuid)
              continue;
            if (!offset) {
              if (ValueAnalysis::isZeroConstant(
                      ValueAnalysis::stripNumericCasts(loadOp.getIndices()[0])))
                return idx;
              continue;
            }
            if (sameIndex(loadOp.getIndices()[0], offset))
              return idx;
          }
        }

        return std::nullopt;
      };

      auto originalPack =
          contEdtCreate.getParamMemref().getDefiningOp<EdtParamPackOp>();
      ArrayRef<int32_t> acquireModes =
          originalRecDep && originalRecDep.getAcquireModes()
              ? ArrayRef<int32_t>(*originalRecDep.getAcquireModes())
              : ArrayRef<int32_t>{};
      ArrayRef<int32_t> depFlags =
          originalRecDep && originalRecDep.getDepFlags()
              ? ArrayRef<int32_t>(*originalRecDep.getDepFlags())
              : ArrayRef<int32_t>{};
      SmallVector<bool> satisfiedOriginalDepSlots(originalDepCount, false);

      auto getOriginalDepMode = [&](unsigned depIdx) -> int32_t {
        return depIdx < acquireModes.size() ? acquireModes[depIdx] : 2;
      };

      auto getOriginalDepFlags =
          [&](unsigned depIdx) -> std::optional<int32_t> {
        if (depIdx < depFlags.size())
          return depFlags[depIdx];
        return std::nullopt;
      };

      auto resolveOriginalDepSlotForCarryIndex =
          [&](int64_t carryIdx) -> std::optional<unsigned> {
        if (carryIdx < 0 || !originalRecDep || !originalPack)
          return std::nullopt;

        for (auto [depIdx, depGuid] :
             llvm::enumerate(originalRecDep.getDatablocks())) {
          auto packIdx = findPackIndexForSafeWholeDbDep(originalPack, depGuid);
          if (!packIdx)
            continue;
          if (*packIdx == static_cast<unsigned>(carryIdx))
            return depIdx;
        }
        return std::nullopt;
      };

      auto emitMappedCarryDep = [&](int64_t carryIdx, Value depGuid,
                                    std::optional<unsigned> fallbackSlot,
                                    int32_t fallbackMode,
                                    std::optional<int32_t> fallbackFlags) {
        if (!depGuid)
          return;

        if (auto mappedSlot = resolveOriginalDepSlotForCarryIndex(carryIdx)) {
          plan.deferredDeps.push_back({DeferredDepEmit::Kind::WholeDb, depGuid,
                                       0, *mappedSlot,
                                       getOriginalDepMode(*mappedSlot),
                                       getOriginalDepFlags(*mappedSlot)});
          satisfiedOriginalDepSlots[*mappedSlot] = true;
          return;
        }

        if (fallbackSlot) {
          plan.deferredDeps.push_back({DeferredDepEmit::Kind::WholeDb, depGuid,
                                       0, *fallbackSlot, fallbackMode,
                                       fallbackFlags});
          if (*fallbackSlot < satisfiedOriginalDepSlots.size())
            satisfiedOriginalDepSlots[*fallbackSlot] = true;
        }
      };

      /// Contract-aware path: structured continuations carry an explicit
      /// launch schema stamped by EpochOpt. That schema proves the relaunch ABI
      /// for this continuation, so preserve original dependency slots with
      /// dep_forward instead of reconstructing them positionally.
      bool hasStructuredLaunchContract =
          contEdtCreate->hasAttr(
              AttrNames::Operation::LaunchState::DepSchema) ||
          contEdtCreate->hasAttr(
              AttrNames::Operation::LaunchState::StateSchema);
      if (hasStructuredLaunchContract && originalRecDep) {
        ARTS_DEBUG("CPS advance: contract-aware dep_forward path");
        for (unsigned depIdx = 0; depIdx < originalDepCount; ++depIdx) {
          plan.deferredDeps.push_back(
              {DeferredDepEmit::Kind::ForwardedSlot, Value(), depIdx, depIdx,
               getOriginalDepMode(depIdx), getOriginalDepFlags(depIdx)});
        }
        /// Emit dep_forward ops for each forwarded slot.
        for (unsigned depIdx = 0; depIdx < originalDepCount; ++depIdx) {
          Value slotVal = AC->createIntConstant(depIdx, AC->Int64, loc);
          AC->create<DepForwardOp>(loc, slotVal);
        }
      } else if (auto depRouting = advanceOp->getAttrOfType<DenseI64ArrayAttr>(
                     CPSDepRouting)) {
        ArrayRef<int64_t> routing = depRouting.asArrayRef();
        int64_t numTimingDbs = routing.size() > 0 ? routing[0] : 0;
        int64_t hasScratch = routing.size() > 1 ? routing[1] : 0;
        unsigned paramStart = (isAdditive && nextParams.size() >= 2) ? 2 : 0;
        unsigned guidDataStart = 2;

        if (hasScratch && guidDataStart < routing.size()) {
          int64_t carryIdx = routing[guidDataStart];
          if (carryIdx >= 0) {
            size_t paramIdx =
                static_cast<size_t>(paramStart) + static_cast<size_t>(carryIdx);
            if (paramIdx < nextParams.size()) {
              Value scratchGuid = nextParams[paramIdx];
              std::optional<unsigned> fallbackSlot = std::nullopt;
              if (!originalRecDep)
                fallbackSlot = static_cast<unsigned>(numTimingDbs);
              emitMappedCarryDep(carryIdx, scratchGuid, fallbackSlot,
                                 /*fallbackMode=*/1,
                                 /*fallbackFlags=*/std::nullopt);
            }
          }
          ++guidDataStart;
        }

        for (int64_t t = 0; t < numTimingDbs; ++t) {
          unsigned routingIdx = guidDataStart + t;
          if (routingIdx >= routing.size())
            break;
          int64_t carryIdx = routing[routingIdx];
          if (carryIdx < 0)
            continue;
          size_t paramIdx =
              static_cast<size_t>(paramStart) + static_cast<size_t>(carryIdx);
          if (paramIdx >= nextParams.size())
            continue;
          Value timingGuid = nextParams[paramIdx];
          std::optional<unsigned> fallbackSlot = std::nullopt;
          if (!originalRecDep)
            fallbackSlot = static_cast<unsigned>(t);
          emitMappedCarryDep(carryIdx, timingGuid, fallbackSlot,
                             /*fallbackMode=*/2,
                             /*fallbackFlags=*/std::nullopt);
        }

        if (originalRecDep) {
          for (unsigned depIdx = 0; depIdx < originalDepCount; ++depIdx) {
            if (satisfiedOriginalDepSlots[depIdx])
              continue;
            plan.deferredDeps.push_back(
                {DeferredDepEmit::Kind::ForwardedSlot, Value(), depIdx, depIdx,
                 getOriginalDepMode(depIdx), getOriginalDepFlags(depIdx)});
          }
        }
      } else if (contEdtCreate->hasAttr(CPSForwardDeps) && originalRecDep) {
        for (unsigned depIdx = 0; depIdx < originalDepCount; ++depIdx) {
          plan.deferredDeps.push_back(
              {DeferredDepEmit::Kind::ForwardedSlot, Value(), depIdx, depIdx,
               getOriginalDepMode(depIdx), getOriginalDepFlags(depIdx)});
        }
      } else if (auto originalRecDep = findOriginalRecDep(contEdtCreate)) {
        bool hasExplicitSlices = !originalRecDep.getByteOffsets().empty() ||
                                 !originalRecDep.getByteSizes().empty();
        bool hasGuardedDeps = llvm::any_of(
            originalRecDep.getBoundsValids(), [](Value boundsValid) {
              return boundsValid &&
                     !ValueAnalysis::isOneConstant(
                         ValueAnalysis::stripNumericCasts(boundsValid));
            });

        if (!hasExplicitSlices && !hasGuardedDeps && originalPack) {
          for (auto [depIdx, depGuid] :
               llvm::enumerate(originalRecDep.getDatablocks())) {
            auto packIdx =
                findPackIndexForSafeWholeDbDep(originalPack, depGuid);
            if (!packIdx)
              continue;
            Value localDepGuid = getLoopBackValueForPackIndex(*packIdx);
            if (!localDepGuid)
              continue;
            plan.deferredDeps.push_back(
                {DeferredDepEmit::Kind::WholeDb, localDepGuid, 0,
                 static_cast<unsigned>(depIdx), getOriginalDepMode(depIdx),
                 getOriginalDepFlags(depIdx)});
          }
        }
      }

      return plan;
    };

    auto emitContinuationEpoch = [&]() -> ContinuationEpochPlan {
      ContinuationEpochPlan plan = createContinuationLaunch();
      Value newContGuid = plan.contGuid;
      Value one = AC->createIntConstant(1, AC->Int32, loc);
      Value finishSlot = AC->create<arith::SubIOp>(loc, localDepCount, one);
      auto innerEpoch = AC->create<CreateEpochOp>(
          loc, IntegerType::get(AC->getContext(), 64), newContGuid, finishSlot);
      plan.epochGuid = innerEpoch.getEpochGuid();
      return plan;
    };

    Value epochGuid;
    Value deferredContGuid;
    SmallVector<DeferredDepEmit, 8> deferredDepEmits;
    if (directRecreate && continueCond) {
      auto ifOp = AC->create<scf::IfOp>(loc, TypeRange{AC->Int64}, continueCond,
                                        /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard guard(AC->getBuilder());
        Block &thenBlock = ifOp.getThenRegion().front();
        AC->getBuilder().setInsertionPointToEnd(&thenBlock);
        auto plan = createContinuationLaunch();
        deferredDepEmits = std::move(plan.deferredDeps);
        AC->create<scf::YieldOp>(loc, ValueRange{plan.contGuid});
      }
      {
        OpBuilder::InsertionGuard guard(AC->getBuilder());
        Block &elseBlock = ifOp.getElseRegion().front();
        AC->getBuilder().setInsertionPointToEnd(&elseBlock);
        Value zeroGuid = AC->createIntConstant(0, AC->Int64, loc);
        AC->create<scf::YieldOp>(loc, ValueRange{zeroGuid});
      }
      deferredContGuid = ifOp.getResult(0);
      AC->setInsertionPointAfter(ifOp);
    } else if (directRecreate) {
      auto plan = createContinuationLaunch();
      deferredContGuid = plan.contGuid;
      deferredDepEmits = std::move(plan.deferredDeps);
    } else if (continueCond) {
      auto ifOp = AC->create<scf::IfOp>(
          loc, TypeRange{outerEpochGuid.getType(), AC->Int64}, continueCond,
          /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard guard(AC->getBuilder());
        Block &thenBlock = ifOp.getThenRegion().front();
        AC->getBuilder().setInsertionPointToEnd(&thenBlock);
        auto plan = emitContinuationEpoch();
        deferredDepEmits = std::move(plan.deferredDeps);
        AC->create<scf::YieldOp>(loc,
                                 ValueRange{plan.epochGuid, plan.contGuid});
      }
      {
        OpBuilder::InsertionGuard guard(AC->getBuilder());
        Block &elseBlock = ifOp.getElseRegion().front();
        AC->getBuilder().setInsertionPointToEnd(&elseBlock);
        Value zeroGuid = AC->createIntConstant(0, AC->Int64, loc);
        AC->create<scf::YieldOp>(loc, ValueRange{outerEpochGuid, zeroGuid});
      }
      epochGuid = ifOp.getResult(0);
      deferredContGuid = ifOp.getResult(1);
      AC->setInsertionPointAfter(ifOp);
    } else {
      auto plan = emitContinuationEpoch();
      epochGuid = plan.epochGuid;
      deferredContGuid = plan.contGuid;
      deferredDepEmits = std::move(plan.deferredDeps);
    }

    if (!directRecreate) {
      /// If the advance region has a block argument serving as the IV
      /// placeholder (from wrappingIf support), replace it with the actual
      /// runtime iteration counter (tNext or curIter).
      Block &advBody = advanceOp.getEpochBody().front();
      if (advanceOp->hasAttr(CPSAdvanceHasIvArg) &&
          advBody.getNumArguments() > 0) {
        Value ivReplacement = tNext ? tNext : curIter;
        if (ivReplacement) {
          // Cast to index if needed (block arg is index type).
          if (ivReplacement.getType() != advBody.getArgument(0).getType()) {
            AC->setInsertionPointAfter(ivReplacement.getDefiningOp()
                                           ? ivReplacement.getDefiningOp()
                                           : &advBody.front());
            ivReplacement = AC->create<arith::IndexCastOp>(
                loc, advBody.getArgument(0).getType(), ivReplacement);
          }
          advBody.getArgument(0).replaceAllUsesWith(ivReplacement);
        }
        advBody.eraseArgument(0);
      }

      /// Move worker ops from CPSAdvanceOp's epochBody AFTER the epoch
      /// creation.
      SmallVector<Operation *> workerOps;
      for (Operation &op : advBody.without_terminator())
        workerOps.push_back(&op);

      Operation *insertAfter = epochGuid.getDefiningOp();
      if (!insertAfter) {
        advanceOp.emitError("CPS advance: expected epoch-guid producer");
        signalPassFailure();
        return;
      }
      for (Operation *workerOp : workerOps) {
        workerOp->moveAfter(insertAfter);
        insertAfter = workerOp;
      }

      /// Update EdtCreateOps in moved worker ops with the epoch GUID.
      for (Operation *workerOp : workerOps) {
        auto updateEdtCreates = [&](EdtCreateOp edtCreate) {
          if (!edtCreate.getEpochGuid()) {
            AC->setInsertionPoint(edtCreate);
            auto updated = AC->create<EdtCreateOp>(
                edtCreate.getLoc(), edtCreate.getParamMemref(),
                edtCreate.getDepCount(), edtCreate.getRoute(), epochGuid);
            for (auto attr : edtCreate->getAttrs())
              updated->setAttr(attr.getName(), attr.getValue());
            edtCreate->replaceAllUsesWith(updated);
            edtCreate->erase();
          }
        };

        if (auto edtCreate = dyn_cast<EdtCreateOp>(workerOp)) {
          updateEdtCreates(edtCreate);
        } else {
          workerOp->walk(updateEdtCreates);
        }
      }

      if (!deferredDepEmits.empty()) {
        // Preserve CPS frontier order: current-iteration worker writes must be
        // registered before the next continuation's whole-DB reads/writes.
        AC->setInsertionPointAfter(insertAfter);
        if (continueCond) {
          auto depIf =
              AC->create<scf::IfOp>(loc, continueCond, /*withElse=*/false);
          OpBuilder::InsertionGuard guard(AC->getBuilder());
          Block &thenBlock = depIf.getThenRegion().front();
          AC->getBuilder().setInsertionPoint(thenBlock.getTerminator());
          for (const DeferredDepEmit &dep : deferredDepEmits) {
            switch (dep.kind) {
            case DeferredDepEmit::Kind::WholeDb:
              emitWholeDbDep(deferredContGuid, dep.depGuid, dep.toSlot,
                             dep.mode, dep.depFlags);
              break;
            case DeferredDepEmit::Kind::ForwardedSlot:
              emitForwardedDepSlot(deferredContGuid, dep.fromSlot, dep.toSlot,
                                   dep.mode, dep.depFlags);
              break;
            }
          }
        } else {
          for (const DeferredDepEmit &dep : deferredDepEmits) {
            switch (dep.kind) {
            case DeferredDepEmit::Kind::WholeDb:
              emitWholeDbDep(deferredContGuid, dep.depGuid, dep.toSlot,
                             dep.mode, dep.depFlags);
              break;
            case DeferredDepEmit::Kind::ForwardedSlot:
              emitForwardedDepSlot(deferredContGuid, dep.fromSlot, dep.toSlot,
                                   dep.mode, dep.depFlags);
              break;
            }
          }
        }
      }
    } else if (!deferredDepEmits.empty()) {
      if (continueCond) {
        auto depIf =
            AC->create<scf::IfOp>(loc, continueCond, /*withElse=*/false);
        OpBuilder::InsertionGuard guard(AC->getBuilder());
        Block &thenBlock = depIf.getThenRegion().front();
        AC->getBuilder().setInsertionPoint(thenBlock.getTerminator());
        for (const DeferredDepEmit &dep : deferredDepEmits) {
          switch (dep.kind) {
          case DeferredDepEmit::Kind::WholeDb:
            emitWholeDbDep(deferredContGuid, dep.depGuid, dep.toSlot, dep.mode,
                           dep.depFlags);
            break;
          case DeferredDepEmit::Kind::ForwardedSlot:
            emitForwardedDepSlot(deferredContGuid, dep.fromSlot, dep.toSlot,
                                 dep.mode, dep.depFlags);
            break;
          }
        }
      } else {
        for (const DeferredDepEmit &dep : deferredDepEmits) {
          switch (dep.kind) {
          case DeferredDepEmit::Kind::WholeDb:
            emitWholeDbDep(deferredContGuid, dep.depGuid, dep.toSlot, dep.mode,
                           dep.depFlags);
            break;
          case DeferredDepEmit::Kind::ForwardedSlot:
            emitForwardedDepSlot(deferredContGuid, dep.fromSlot, dep.toSlot,
                                 dep.mode, dep.depFlags);
            break;
          }
        }
      }
    }

    advanceOp.erase();
    ++numCpsAdvancesResolved;
    ARTS_INFO("CPS advance: resolved with epoch finish → "
              << outlinedFunc.getValue());
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

  if (failed(compactContinuationParamAbi()))
    return signalPassFailure();

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

LogicalResult EpochLoweringPass::compactContinuationParamAbi() {
  DenseMap<StringRef, SmallVector<EdtCreateOp>> createsByFunc;
  module.walk([&](EdtCreateOp edt) {
    if (!edt->hasAttr(ContinuationForEpoch))
      return;
    auto funcAttr =
        edt->getAttrOfType<StringAttr>(AttrNames::Operation::OutlinedFunc);
    if (!funcAttr)
      return;
    createsByFunc[funcAttr.getValue()].push_back(edt);
  });

  for (auto &[funcName, creates] : createsByFunc) {
    func::FuncOp func = module.lookupSymbol<func::FuncOp>(funcName);
    if (!func)
      continue;

    EdtParamUnpackOp unpackOp = nullptr;
    func.walk([&](EdtParamUnpackOp op) {
      if (!unpackOp)
        unpackOp = op;
    });
    if (!unpackOp)
      continue;

    Value paramMemref = unpackOp.getOperand();
    unsigned unpackSlots = unpackOp.getNumResults();
    unsigned totalSlots = unpackSlots;

    auto updateTotalSlots = [&](unsigned idx) {
      totalSlots = std::max<unsigned>(totalSlots, idx + 1);
    };
    for (EdtCreateOp edt : creates) {
      if (auto attr = edt->getAttrOfType<IntegerAttr>(CPSIterCounterParamIdx))
        updateTotalSlots(attr.getInt());
      if (auto attr = edt->getAttrOfType<IntegerAttr>(CPSOuterEpochParamIdx))
        updateTotalSlots(attr.getInt());
      if (auto packOp = edt.getParamMemref().getDefiningOp<EdtParamPackOp>())
        totalSlots = std::max<unsigned>(totalSlots, packOp->getNumOperands());
    }

    SmallVector<bool> keep(totalSlots, false);
    for (auto [idx, result] : llvm::enumerate(unpackOp.getResults()))
      if (!result.use_empty())
        keep[idx] = true;

    if (auto preserveCarryAttr =
            func->getAttrOfType<IntegerAttr>(CPSPreserveCarryAbi)) {
      unsigned preserveCount = std::max<int64_t>(preserveCarryAttr.getInt(), 0);
      if (keep.size() < preserveCount)
        keep.resize(preserveCount, false);
      for (unsigned idx = 0; idx < preserveCount; ++idx)
        keep[idx] = true;
      totalSlots = std::max<unsigned>(totalSlots, preserveCount);
    }

    SmallVector<memref::LoadOp> directParamLoads;
    func.walk([&](memref::LoadOp load) {
      if (load.getMemRef() != paramMemref || load.getIndices().size() != 1)
        return;
      auto idxOp = load.getIndices()[0].getDefiningOp<arith::ConstantIndexOp>();
      if (!idxOp)
        return;
      unsigned idx = idxOp.value();
      if (idx >= keep.size())
        keep.resize(idx + 1, false);
      updateTotalSlots(idx);
      keep[idx] = true;
      directParamLoads.push_back(load);
    });

    bool allLive = llvm::all_of(keep, [](bool live) { return live; });
    if (allLive)
      continue;

    SmallVector<int64_t> oldToNew(totalSlots, -1);
    SmallVector<unsigned> keptSlots;
    for (unsigned oldIdx = 0; oldIdx < totalSlots; ++oldIdx) {
      if (!keep[oldIdx])
        continue;
      oldToNew[oldIdx] = keptSlots.size();
      keptSlots.push_back(oldIdx);
    }
    if (keptSlots.empty())
      continue;

    OpBuilder builder(func.getContext());

    SmallVector<Type> liveTypes;
    SmallVector<unsigned> keptUnpackSlots;
    liveTypes.reserve(unpackSlots);
    keptUnpackSlots.reserve(unpackSlots);
    for (unsigned oldIdx = 0; oldIdx < unpackSlots; ++oldIdx) {
      if (!keep[oldIdx])
        continue;
      keptUnpackSlots.push_back(oldIdx);
      liveTypes.push_back(unpackOp.getResult(oldIdx).getType());
    }

    builder.setInsertionPoint(unpackOp);
    auto newUnpack = EdtParamUnpackOp::create(builder, unpackOp.getLoc(),
                                              liveTypes, paramMemref);
    for (auto [newIdx, oldIdx] : llvm::enumerate(keptUnpackSlots))
      unpackOp.getResult(oldIdx).replaceAllUsesWith(
          newUnpack.getResult(newIdx));
    unpackOp.erase();

    for (memref::LoadOp load : directParamLoads) {
      if (!load || load->getBlock() == nullptr)
        continue;
      auto idxOp = load.getIndices()[0].getDefiningOp<arith::ConstantIndexOp>();
      if (!idxOp)
        continue;
      unsigned oldIdx = idxOp.value();
      if (oldIdx >= oldToNew.size() || oldToNew[oldIdx] < 0)
        continue;
      unsigned newIdx = oldToNew[oldIdx];
      if (newIdx == oldIdx)
        continue;
      builder.setInsertionPoint(load);
      Value newIdxVal = arith::ConstantIndexOp::create(
          builder, load.getLoc(), static_cast<int64_t>(newIdx));
      auto newLoad = memref::LoadOp::create(builder, load.getLoc(), paramMemref,
                                            ValueRange{newIdxVal});
      load.getResult().replaceAllUsesWith(newLoad.getResult());
      load.erase();
    }

    auto getCarryCount = [&](EdtCreateOp edt) -> unsigned {
      if (auto perm = edt->getAttrOfType<DenseI64ArrayAttr>(CPSParamPerm))
        return perm.size();
      if (auto iterAttr =
              edt->getAttrOfType<IntegerAttr>(CPSIterCounterParamIdx))
        return iterAttr.getInt();
      if (auto epochAttr =
              edt->getAttrOfType<IntegerAttr>(CPSOuterEpochParamIdx))
        return epochAttr.getInt();
      return 0;
    };

    for (EdtCreateOp edt : creates) {
      if (auto packOp = edt.getParamMemref().getDefiningOp<EdtParamPackOp>()) {
        SmallVector<Value> operands;
        operands.reserve(keptSlots.size());
        for (unsigned oldIdx : keptSlots)
          if (oldIdx < packOp->getNumOperands())
            operands.push_back(packOp->getOperand(oldIdx));
        builder.setInsertionPoint(packOp);
        auto packType =
            MemRefType::get({ShapedType::kDynamic}, builder.getI64Type());
        auto newPack = EdtParamPackOp::create(builder, packOp.getLoc(),
                                              packType, operands);
        edt->setOperand(0, newPack.getMemref());
        if (packOp->use_empty())
          packOp.erase();
      }

      if (auto iterAttr =
              edt->getAttrOfType<IntegerAttr>(CPSIterCounterParamIdx)) {
        int64_t oldIdx = iterAttr.getInt();
        if (oldIdx >= 0 && static_cast<size_t>(oldIdx) < oldToNew.size() &&
            oldToNew[oldIdx] >= 0) {
          edt->setAttr(CPSIterCounterParamIdx,
                       builder.getI64IntegerAttr(oldToNew[oldIdx]));
        }
      }
      if (auto epochAttr =
              edt->getAttrOfType<IntegerAttr>(CPSOuterEpochParamIdx)) {
        int64_t oldIdx = epochAttr.getInt();
        if (oldIdx >= 0 && static_cast<size_t>(oldIdx) < oldToNew.size() &&
            oldToNew[oldIdx] >= 0) {
          edt->setAttr(CPSOuterEpochParamIdx,
                       builder.getI64IntegerAttr(oldToNew[oldIdx]));
        }
      }

      unsigned oldCarryCount = getCarryCount(edt);
      SmallVector<int64_t> newPerm;
      if (auto permAttr = edt->getAttrOfType<DenseI64ArrayAttr>(CPSParamPerm)) {
        auto perm = permAttr.asArrayRef();
        for (unsigned oldSlot = 0;
             oldSlot < oldCarryCount && oldSlot < keep.size() &&
             oldSlot < perm.size();
             ++oldSlot) {
          if (keep[oldSlot])
            newPerm.push_back(perm[oldSlot]);
        }
      } else {
        for (unsigned oldSlot = 0;
             oldSlot < oldCarryCount && oldSlot < keep.size(); ++oldSlot) {
          if (keep[oldSlot])
            newPerm.push_back(oldSlot);
        }
      }
      if (!newPerm.empty())
        edt->setAttr(CPSParamPerm, builder.getDenseI64ArrayAttr(newPerm));
    }
  }

  return success();
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
