///==========================================================================///
/// File: MemoryUnitMaterialization.cpp
///
/// Materialize SDE shared memrefs as SDE memory units.
///==========================================================================///

#include "carts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "carts/dialect/sde/Analysis/StructuredOpAnalysis.h"
#include "carts/dialect/sde/Transforms/Passes.h"
#include "carts/utils/ArrayAttrUtils.h"
namespace mlir::carts::sde {
#define GEN_PASS_DEF_MEMORYUNITMATERIALIZATION
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::sde

#include "carts/utils/ValueAnalysis.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::carts;

namespace {

static bool hasNestedMemrefElement(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && isa<MemRefType>(type.getElementType());
}

static bool isMuMaterializableAllocation(Value root) {
  if (!root || !isa<MemRefType>(root.getType()) || hasNestedMemrefElement(root))
    return false;

  Operation *def = root.getDefiningOp();
  if (!def)
    return false;

  if (auto alloc = dyn_cast<memref::AllocOp>(def))
    return alloc.getSymbolOperands().empty();
  if (auto alloca = dyn_cast<memref::AllocaOp>(def))
    return alloca.getSymbolOperands().empty();
  return false;
}

static bool isPrivateAllocationForSchedulingUnit(Value root,
                                                 sde::SdeSuIterateOp op) {
  if (!root || !op)
    return false;
  Operation *def = root.getDefiningOp();
  if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(def))
    return false;

  bool sawUse = false;
  for (Operation *user : root.getUsers()) {
    if (user == def)
      continue;
    sawUse = true;
    if (!op->isAncestor(user))
      return false;
  }
  return sawUse;
}

static bool hasPhysicalOwnerSlicePlan(sde::SdeSuIterateOp op) {
  return op.getPhysicalBlockShapeAttr() || op.getPhysicalOwnerDimsAttr();
}

static bool hasSameI64Values(ArrayAttr attr, ArrayRef<int64_t> values) {
  std::optional<SmallVector<int64_t, 4>> attrValues = readI64ArrayAttr(attr);
  return attrValues && ArrayRef<int64_t>(*attrValues) == values;
}

static std::optional<sde::LoopIndexedOutputPlan>
getUnclassifiedOutputOnlyOwnerSlicePlan(sde::SdeSuIterateOp op) {
  if (!op || !hasPhysicalOwnerSlicePlan(op) ||
      op.getStructuredClassificationAttr())
    return std::nullopt;

  std::optional<sde::LoopIndexedOutputPlan> outputPlan =
      sde::findConsistentLoopIndexedOutputPlanWithOwnerDims(op);
  if (!outputPlan || outputPlan->ownerPhysicalDims.empty())
    return std::nullopt;
  if (!hasSameI64Values(op.getPhysicalOwnerDimsAttr(),
                        outputPlan->ownerPhysicalDims))
    return std::nullopt;

  sde::StructuredMemoryEffectSummary effects =
      sde::collectStructuredMemoryEffects(op.getBody());
  if (effects.hasUnknownEffects || effects.writes.empty())
    return std::nullopt;

  for (Value read : effects.reads) {
    if (sde::isDefinedInside(op.getOperation(), read))
      continue;
    if (!effects.writes.contains(read))
      return std::nullopt;
  }

  for (Value written : effects.writes) {
    if (sde::isDefinedInside(op.getOperation(), written))
      continue;
    if (effects.reads.contains(written))
      return std::nullopt;
  }

  return outputPlan;
}

static bool canMaterializePlannedOwnerSlices(sde::SdeSuIterateOp op) {
  if (!hasPhysicalOwnerSlicePlan(op))
    return true;

  auto classification = op.getStructuredClassification();
  if (!classification)
    return getUnclassifiedOutputOnlyOwnerSlicePlan(op).has_value();

  switch (*classification) {
  case sde::SdeStructuredClassification::matmul:
  case sde::SdeStructuredClassification::elementwise:
  case sde::SdeStructuredClassification::elementwise_pipeline:
  case sde::SdeStructuredClassification::stencil:
    return true;
  case sde::SdeStructuredClassification::reduction:
    return op.getReductionAccumulators().empty() &&
           sde::findLoopIndexedOutputPlan(op).has_value();
  }
  return false;
}

static void demoteUnsupportedPhysicalStoragePlan(sde::SdeSuIterateOp op) {
  if (!op || !hasPhysicalOwnerSlicePlan(op) ||
      canMaterializePlannedOwnerSlices(op))
    return;

  // Physical storage attrs are a promise that the SDE/CODIR boundary can
  // materialize token-local views. Keep logical scheduling intent, but do not
  // export an unsupported DB layout to the residual raw CreateDbs bridge.
  op.removePhysicalOwnerDimsAttr();
  op.removePhysicalBlockShapeAttr();
  op.removePhysicalHaloShapeAttr();
}

static void collectSchedulingUnitMemrefRoots(sde::SdeSuIterateOp op,
                                             SetVector<Value> &roots) {
  if (!op || !canMaterializePlannedOwnerSlices(op))
    return;

  bool collectWritesOnly =
      !op.getStructuredClassificationAttr() &&
      getUnclassifiedOutputOnlyOwnerSlicePlan(op).has_value();

  op.getBody().walk([&](Operation *nested) {
    Value memref;
    if (auto load = dyn_cast<memref::LoadOp>(nested)) {
      if (collectWritesOnly)
        return;
      if (isa<MemRefType>(load.getResult().getType()))
        return;
      memref = load.getMemref();
    } else if (auto store = dyn_cast<memref::StoreOp>(nested)) {
      if (isa<MemRefType>(store.getValueToStore().getType()))
        return;
      memref = store.getMemref();
    } else {
      return;
    }

    Value root = ::mlir::carts::ValueAnalysis::stripMemrefViewOps(memref);
    if (!root || sde::isDefinedInside(op.getOperation(), root))
      return;
    if (isPrivateAllocationForSchedulingUnit(root, op))
      return;
    if (isMuMaterializableAllocation(root))
      roots.insert(root);
  });
}

static void eraseDeallocUsers(Value root, PatternRewriter &rewriter) {
  SmallVector<memref::DeallocOp> deallocs;
  for (Operation *user : llvm::make_early_inc_range(root.getUsers()))
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
      deallocs.push_back(dealloc);

  for (memref::DeallocOp dealloc : deallocs)
    rewriter.eraseOp(dealloc);
}

static FailureOr<Value> createMuAllocForRoot(Value root,
                                             PatternRewriter &rewriter) {
  Operation *def = root.getDefiningOp();
  if (!def)
    return failure();

  SmallVector<Value> dynamicSizes;
  if (auto alloc = dyn_cast<memref::AllocOp>(def)) {
    llvm::append_range(dynamicSizes, alloc.getDynamicSizes());
  } else if (auto alloca = dyn_cast<memref::AllocaOp>(def)) {
    llvm::append_range(dynamicSizes, alloca.getDynamicSizes());
  } else {
    return failure();
  }

  auto memrefType = cast<MemRefType>(root.getType());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(def);
  auto muAlloc = sde::SdeMuAllocOp::create(rewriter, def->getLoc(), memrefType,
                                           ValueRange(dynamicSizes));

  eraseDeallocUsers(root, rewriter);
  root.replaceAllUsesWith(muAlloc.getMemref());
  if (def->use_empty())
    rewriter.eraseOp(def);
  return muAlloc.getMemref();
}

struct MemoryUnitMaterializationPass
    : public sde::impl::MemoryUnitMaterializationBase<
          MemoryUnitMaterializationPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SetVector<Value> roots;
    module.walk([&](sde::SdeSuIterateOp op) {
      collectSchedulingUnitMemrefRoots(op, roots);
      demoteUnsupportedPhysicalStoragePlan(op);
    });

    PatternRewriter rewriter(module.getContext());
    for (Value root : roots) {
      if (!isMuMaterializableAllocation(root))
        continue;
      if (failed(createMuAllocForRoot(root, rewriter))) {
        if (Operation *def = root.getDefiningOp())
          def->emitError("failed to materialize SDE memory unit");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createMemoryUnitMaterializationPass() {
  return std::make_unique<MemoryUnitMaterializationPass>();
}

} // namespace mlir::carts::sde
