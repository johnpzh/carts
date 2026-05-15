///==========================================================================///
/// File: MemoryUnitMaterialization.cpp
///
/// Materialize SDE shared memrefs as SDE memory units.
///==========================================================================///

#include "arts/dialect/sde/Analysis/SdeAnalysisUtils.h"
#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_MEMORYUNITMATERIALIZATION
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/ValueAnalysis.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::arts;
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

static bool hasPhysicalOwnerSlicePlan(sde::SdeSuIterateOp op) {
  return op.getPhysicalBlockShapeAttr() || op.getPhysicalOwnerDimsAttr();
}

static bool canMaterializePlannedOwnerSlices(sde::SdeSuIterateOp op) {
  if (!hasPhysicalOwnerSlicePlan(op))
    return true;

  auto classification = op.getStructuredClassification();
  if (!classification)
    return false;

  switch (*classification) {
  case sde::SdeStructuredClassification::matmul:
  case sde::SdeStructuredClassification::elementwise:
  case sde::SdeStructuredClassification::elementwise_pipeline:
    return true;
  case sde::SdeStructuredClassification::stencil:
  case sde::SdeStructuredClassification::reduction:
    return false;
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

  op.getBody().walk([&](Operation *nested) {
    Value memref;
    if (auto load = dyn_cast<memref::LoadOp>(nested)) {
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

    Value root = arts::ValueAnalysis::stripMemrefViewOps(memref);
    if (!root || sde::isDefinedInside(op.getOperation(), root))
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
    : public arts::impl::MemoryUnitMaterializationBase<
          MemoryUnitMaterializationPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SetVector<Value> roots;
    module.walk([&](sde::SdeSuIterateOp op) {
      collectSchedulingUnitMemrefRoots(op, roots);
    });
    module.walk([&](sde::SdeSuIterateOp op) {
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
