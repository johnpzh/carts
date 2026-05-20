///==========================================================================///
/// File: StoragePlanning.cpp
///
/// CODIR-owned dependency storage planning before ARTS materialization.
///==========================================================================///

#include "carts/dialect/codir/Transforms/Passes.h"

#include "carts/utils/ArrayAttrUtils.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <limits>

namespace mlir::carts::codir {
#define GEN_PASS_DEF_STORAGEPLANNING
#include "carts/dialect/codir/Transforms/Passes.h.inc"
} // namespace mlir::carts::codir

using namespace mlir;
using namespace mlir::carts;

namespace {

static constexpr int64_t kMaxPhaseRedistributionBridgeElements =
    16LL * 1024LL * 1024LL;

struct MemoryAccessInfo {
  Value memref;
  SmallVector<Value> indices;
};

static bool isMemrefForwardingOp(Operation *op) {
  if (!op || op->getNumRegions() != 0)
    return false;
  return llvm::any_of(op->getResults(), [](Value result) {
    return isa<MemRefType>(result.getType());
  });
}

static Value stripStorageViews(Value value) {
  for (;;) {
    Operation *def = value ? value.getDefiningOp() : nullptr;
    if (auto subview = dyn_cast_or_null<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = dyn_cast_or_null<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    return value;
  }
}

static bool isStorageView(Value value) {
  return isa_and_nonnull<memref::SubViewOp>(value ? value.getDefiningOp()
                                                  : nullptr);
}

static std::optional<MemoryAccessInfo>
getMemoryAccessInfo(Operation *op) {
  if (auto load = dyn_cast_or_null<memref::LoadOp>(op))
    return MemoryAccessInfo{
        load.getMemRef(),
        SmallVector<Value>(load.getIndices().begin(), load.getIndices().end())};
  if (auto store = dyn_cast_or_null<memref::StoreOp>(op))
    return MemoryAccessInfo{
        store.getMemRef(),
        SmallVector<Value>(store.getIndices().begin(),
                           store.getIndices().end())};
  return std::nullopt;
}

static bool indexSelectsOwnerSlice(Value index, Value ownerIv,
                                   llvm::SmallPtrSetImpl<Value> &seen) {
  if (!index || !ownerIv || !seen.insert(index).second)
    return false;
  if (ValueAnalysis::dependsOn(index, ownerIv))
    return true;

  auto blockArg = dyn_cast<BlockArgument>(index);
  if (blockArg) {
    auto loop =
        dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!loop || loop.getInductionVar() != index)
      return false;

    return indexSelectsOwnerSlice(loop.getLowerBound(), ownerIv, seen) ||
           indexSelectsOwnerSlice(loop.getUpperBound(), ownerIv, seen);
  }

  Operation *def = index.getDefiningOp();
  if (!isa_and_nonnull<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                       arith::DivSIOp, arith::DivUIOp, arith::RemSIOp,
                       arith::RemUIOp, arith::IndexCastOp,
                       arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp,
                       arith::TruncIOp, arith::MinSIOp, arith::MinUIOp,
                       arith::MaxSIOp, arith::MaxUIOp>(def))
    return false;

  return llvm::any_of(def->getOperands(), [&](Value operand) {
    return indexSelectsOwnerSlice(operand, ownerIv, seen);
  });
}

static bool indexSelectsOwnerSlice(Value index, Value ownerIv) {
  llvm::SmallPtrSet<Value, 8> seen;
  return indexSelectsOwnerSlice(index, ownerIv, seen);
}

static bool hasTileOwnerSlicePlan(codir::CodeletOp codelet) {
  return codelet && codelet.getTileShapeAttr() &&
         codelet.getTileOwnerDimsAttr();
}

static std::optional<unsigned>
getSingleTileOwnerDim(codir::CodeletOp codelet) {
  if (!hasTileOwnerSlicePlan(codelet))
    return std::nullopt;
  std::optional<SmallVector<int64_t, 4>> ownerDims =
      readI64ArrayAttr(codelet.getTileOwnerDimsAttr());
  if (!ownerDims || ownerDims->size() != 1 || ownerDims->front() < 0)
    return std::nullopt;
  return static_cast<unsigned>(ownerDims->front());
}

static Value getOwnerBaseArgument(codir::CodeletOp codelet) {
  if (!codelet || codelet.getBody().empty() || codelet.getParams().empty())
    return {};

  Block &body = codelet.getBody().front();
  unsigned depCount = codelet.getDeps().size();
  unsigned paramCount = codelet.getParams().size();
  if (body.getNumArguments() < depCount + paramCount)
    return {};
  return body.getArgument(depCount + paramCount - 1);
}

static std::optional<unsigned>
inferDepOwnerAccessDim(codir::CodeletOp codelet, unsigned depIndex) {
  if (!codelet || codelet.getBody().empty() ||
      depIndex >= codelet.getDeps().size())
    return std::nullopt;

  Block &body = codelet.getBody().front();
  if (depIndex >= body.getNumArguments())
    return std::nullopt;

  Value depArg = body.getArgument(depIndex);
  auto depType = dyn_cast<MemRefType>(depArg.getType());
  if (!depType || depType.getRank() == 0)
    return std::nullopt;

  Value ownerBase = getOwnerBaseArgument(codelet);
  if (!ownerBase)
    return std::nullopt;

  bool sawDirectRootAccess = false;
  bool rejected = false;
  std::optional<unsigned> selectedDim;
  body.walk([&](Operation *op) {
    if (rejected)
      return WalkResult::interrupt();

    auto access = getMemoryAccessInfo(op);
    if (!access || access->memref != depArg)
      return WalkResult::advance();

    sawDirectRootAccess = true;
    std::optional<unsigned> accessDim;
    for (auto [dim, index] : llvm::enumerate(access->indices)) {
      if (!indexSelectsOwnerSlice(index, ownerBase))
        continue;
      if (accessDim && *accessDim != dim) {
        rejected = true;
        return WalkResult::interrupt();
      }
      accessDim = static_cast<unsigned>(dim);
    }
    if (!accessDim || *accessDim >= depType.getRank()) {
      rejected = true;
      return WalkResult::interrupt();
    }
    if (selectedDim && *selectedDim != *accessDim) {
      rejected = true;
      return WalkResult::interrupt();
    }
    selectedDim = *accessDim;
    return WalkResult::advance();
  });

  if (!sawDirectRootAccess || rejected)
    return std::nullopt;
  return selectedDim;
}

static std::optional<unsigned> getDepOwnerDim(codir::CodeletOp codelet,
                                              unsigned depIndex) {
  if (std::optional<unsigned> inferred =
          inferDepOwnerAccessDim(codelet, depIndex))
    return inferred;
  return getSingleTileOwnerDim(codelet);
}

static bool depAccessesStayWithinSingleOwnerSlice(codir::CodeletOp codelet,
                                                  unsigned depIndex) {
  std::optional<unsigned> ownerDim = getDepOwnerDim(codelet, depIndex);
  if (!ownerDim || !codelet || codelet.getBody().empty())
    return false;

  Block &body = codelet.getBody().front();
  if (depIndex >= codelet.getDeps().size() ||
      depIndex >= body.getNumArguments())
    return false;

  Value depArg = body.getArgument(depIndex);
  auto depType = dyn_cast<MemRefType>(depArg.getType());
  if (!depType || depType.getRank() == 0 || *ownerDim >= depType.getRank())
    return false;

  Value ownerBase = getOwnerBaseArgument(codelet);
  if (!ownerBase)
    return false;

  bool sawDirectRootAccess = false;
  bool rejected = false;
  body.walk([&](Operation *op) {
    if (rejected)
      return WalkResult::interrupt();

    auto access = getMemoryAccessInfo(op);
    if (!access || access->memref != depArg)
      return WalkResult::advance();

    sawDirectRootAccess = true;
    if (access->indices.size() <= *ownerDim ||
        !indexSelectsOwnerSlice(access->indices[*ownerDim], ownerBase)) {
      rejected = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return sawDirectRootAccess && !rejected;
}

static bool hasHostMemrefAccessOutsideCodelet(Value root) {
  if (!root)
    return false;

  SmallVector<Value, 8> worklist{root};
  llvm::SmallPtrSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    for (Operation *user : llvm::make_early_inc_range(current.getUsers())) {
      if (!user || user->getParentOfType<codir::CodeletOp>())
        continue;
      if (isa<memref::DeallocOp, memref::DimOp>(user))
        continue;
      if (isa<memref::LoadOp, memref::StoreOp>(user))
        return true;
      if (isMemrefForwardingOp(user))
        for (Value result : user->getResults())
          if (isa<MemRefType>(result.getType()))
            worklist.push_back(result);
    }
  }

  return false;
}

static std::optional<int64_t> getStaticLogicalElementCount(Value value) {
  Value root = stripStorageViews(value);
  if (!root)
    return std::nullopt;
  auto memrefType = dyn_cast<MemRefType>(root.getType());
  if (!memrefType || !memrefType.hasStaticShape())
    return std::nullopt;

  int64_t count = 1;
  for (int64_t dim : memrefType.getShape()) {
    if (dim < 0 || count > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    count *= dim;
  }
  return count;
}

static bool isPerDependencyRedistribution(codir::CodeletOp codelet,
                                          unsigned depIndex) {
  std::optional<unsigned> depOwnerDim = inferDepOwnerAccessDim(codelet, depIndex);
  std::optional<unsigned> codeletOwnerDim = getSingleTileOwnerDim(codelet);
  return depOwnerDim && codeletOwnerDim && *depOwnerDim != *codeletOwnerDim;
}

static std::optional<unsigned>
getCodeletDepOperandIndex(codir::CodeletOp codelet, OpOperand &use) {
  if (!codelet)
    return std::nullopt;
  unsigned operandIndex = use.getOperandNumber();
  if (operandIndex >= codelet.getDeps().size())
    return std::nullopt;
  return operandIndex;
}

static std::optional<codir::CodirStorageViewKind>
getDepStorageViewKind(codir::CodeletOp codelet, unsigned depIndex) {
  ArrayAttr views = codelet ? codelet.getDepStorageViewsAttr() : ArrayAttr{};
  if (!views || depIndex >= views.size())
    return std::nullopt;
  auto view = dyn_cast<codir::CodirStorageViewKindAttr>(views[depIndex]);
  if (!view)
    return std::nullopt;
  return view.getValue();
}

static bool storageViewUsesComputeBlock(codir::CodirStorageViewKind view) {
  return view == codir::CodirStorageViewKind::compute_block ||
         view == codir::CodirStorageViewKind::phase_redistributed;
}

static bool hasSameBlockStoragePlan(codir::CodeletOp lhs,
                                    unsigned lhsDepIndex,
                                    codir::CodeletOp rhs,
                                    unsigned rhsDepIndex) {
  if (!lhs || !rhs)
    return false;
  std::optional<unsigned> lhsOwnerDim = getDepOwnerDim(lhs, lhsDepIndex);
  std::optional<unsigned> rhsOwnerDim = getDepOwnerDim(rhs, rhsDepIndex);
  return lhsOwnerDim && rhsOwnerDim && *lhsOwnerDim == *rhsOwnerDim &&
         lhs.getTileShapeAttr() == rhs.getTileShapeAttr() &&
         lhs.getLogicalWorkerSliceAttr() == rhs.getLogicalWorkerSliceAttr();
}

static bool isCompatibleBlockStorageParticipant(codir::CodeletOp seed,
                                                unsigned seedDepIndex,
                                                codir::CodeletOp candidate,
                                                unsigned candidateDepIndex) {
  if (!seed || !candidate ||
      !hasSameBlockStoragePlan(seed, seedDepIndex, candidate,
                               candidateDepIndex))
    return false;
  std::optional<codir::CodirStorageViewKind> view =
      getDepStorageViewKind(candidate, candidateDepIndex);
  if (view && !storageViewUsesComputeBlock(*view))
    return false;
  if (!hasTileOwnerSlicePlan(candidate) ||
      !depAccessesStayWithinSingleOwnerSlice(candidate, candidateDepIndex))
    return false;
  return true;
}

static bool hasIncompatibleSharedStorageUse(codir::CodeletOp seed,
                                            unsigned seedDepIndex, Value root) {
  if (!seed || !root)
    return false;

  SmallVector<Value, 8> worklist{root};
  llvm::SmallPtrSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    for (OpOperand &use : llvm::make_early_inc_range(current.getUses())) {
      Operation *owner = use.getOwner();
      if (!owner)
        continue;
      if (owner->getParentOfType<codir::CodeletOp>() &&
          !isa<codir::CodeletOp>(owner))
        continue;
      if (isa<memref::DeallocOp, memref::DimOp>(owner))
        continue;

      if (auto codelet = dyn_cast<codir::CodeletOp>(owner)) {
        std::optional<unsigned> depIndex =
            getCodeletDepOperandIndex(codelet, use);
        if (!depIndex)
          return true;
        if (codelet == seed && *depIndex == seedDepIndex)
          continue;
        if (!isCompatibleBlockStorageParticipant(seed, seedDepIndex, codelet,
                                                 *depIndex))
          return true;
        continue;
      }

      if (isMemrefForwardingOp(owner)) {
        for (Value result : owner->getResults())
          if (isa<MemRefType>(result.getType()))
            worklist.push_back(result);
        continue;
      }

      if (isa<memref::LoadOp, memref::StoreOp>(owner))
        return true;
    }
  }

  return false;
}

static bool needsSharedRootRedistribution(codir::CodeletOp codelet,
                                          unsigned depIndex, Value dep) {
  Value root = stripStorageViews(dep);
  if (!root)
    return false;
  return hasIncompatibleSharedStorageUse(codelet, depIndex, root);
}

static bool shouldDeferPhaseRedistribution(codir::CodeletOp codelet,
                                           unsigned depIndex, Value dep) {
  if (!isPerDependencyRedistribution(codelet, depIndex))
    return false;

  std::optional<int64_t> elements = getStaticLogicalElementCount(dep);
  return elements && *elements > kMaxPhaseRedistributionBridgeElements;
}

static ArrayAttr buildOwnerDimsAttr(MLIRContext *ctx,
                                    std::optional<unsigned> ownerDim) {
  if (!ownerDim)
    return ArrayAttr::get(ctx, {});
  return buildI64ArrayAttr(ctx, SmallVector<int64_t, 1>{*ownerDim});
}

static codir::CodirStorageViewKind
chooseStorageView(codir::CodeletOp codelet, unsigned depIndex,
                  codir::CodirStorageViewKind requested) {
  if (requested != codir::CodirStorageViewKind::compute_block)
    return requested;
  if (!codelet || depIndex >= codelet.getDeps().size())
    return codir::CodirStorageViewKind::host_whole;

  Value dep = codelet.getDeps()[depIndex];
  if (isStorageView(dep))
    return codir::CodirStorageViewKind::compute_block;
  if (!hasTileOwnerSlicePlan(codelet) ||
      !depAccessesStayWithinSingleOwnerSlice(codelet, depIndex))
    return codir::CodirStorageViewKind::host_whole;

  Value root = stripStorageViews(dep);
  if (needsSharedRootRedistribution(codelet, depIndex, dep)) {
    if (shouldDeferPhaseRedistribution(codelet, depIndex, dep))
      return codir::CodirStorageViewKind::host_whole;
    return codir::CodirStorageViewKind::phase_redistributed;
  }

  bool needsHostBridge =
      isa_and_nonnull<BlockArgument>(root) ||
      hasHostMemrefAccessOutsideCodelet(root);
  if (!needsHostBridge)
    return codir::CodirStorageViewKind::compute_block;

  if (shouldDeferPhaseRedistribution(codelet, depIndex, dep))
    return codir::CodirStorageViewKind::host_whole;
  return codir::CodirStorageViewKind::phase_redistributed;
}

struct StoragePlanningPass
    : public codir::impl::StoragePlanningBase<StoragePlanningPass> {
  void runOnOperation() override {
    getOperation().walk([&](codir::CodeletOp codelet) {
      ArrayAttr storageViews = codelet.getDepStorageViewsAttr();

      SmallVector<Attribute> plannedViews;
      SmallVector<Attribute> plannedOwnerDims;
      plannedViews.reserve(codelet.getDeps().size());
      plannedOwnerDims.reserve(codelet.getDeps().size());
      bool changed = !storageViews ||
                     storageViews.size() != codelet.getDeps().size();
      for (unsigned index = 0, e = codelet.getDeps().size(); index < e;
           ++index) {
        codir::CodirStorageViewKind requested =
            hasTileOwnerSlicePlan(codelet)
                ? codir::CodirStorageViewKind::compute_block
                : codir::CodirStorageViewKind::host_whole;
        if (storageViews && index < storageViews.size()) {
          auto viewAttr =
              dyn_cast<codir::CodirStorageViewKindAttr>(storageViews[index]);
          if (!viewAttr) {
            plannedViews.push_back(storageViews[index]);
            plannedOwnerDims.push_back(buildOwnerDimsAttr(
                codelet.getContext(), getDepOwnerDim(codelet, index)));
            continue;
          }
          requested = viewAttr.getValue();
        }

        codir::CodirStorageViewKind planned =
            chooseStorageView(codelet, index, requested);
        if (planned != requested)
          changed = true;
        if (storageViews && index < storageViews.size()) {
          auto viewAttr =
              dyn_cast<codir::CodirStorageViewKindAttr>(storageViews[index]);
          if (viewAttr && planned != viewAttr.getValue())
            changed = true;
        }
        plannedViews.push_back(
            codir::CodirStorageViewKindAttr::get(codelet.getContext(), planned));
        plannedOwnerDims.push_back(buildOwnerDimsAttr(
            codelet.getContext(), getDepOwnerDim(codelet, index)));
      }

      if (storageViews && storageViews.size() > codelet.getDeps().size()) {
        for (unsigned index = codelet.getDeps().size(),
                      e = storageViews.size();
             index < e; ++index)
          plannedViews.push_back(storageViews[index]);
      }

      if (changed)
        codelet.setDepStorageViewsAttr(
            ArrayAttr::get(codelet.getContext(), plannedViews));
      codelet.setDepOwnerDimsAttr(
          ArrayAttr::get(codelet.getContext(), plannedOwnerDims));
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::carts::codir::createStoragePlanningPass() {
  return std::make_unique<StoragePlanningPass>();
}
