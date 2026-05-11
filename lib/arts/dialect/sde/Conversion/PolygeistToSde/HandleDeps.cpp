///===----------------------------------------------------------------------===///
/// File: HandleDeps.cpp
///
/// This pass converts residual OMP task dependencies to arts.omp_dep
/// operations. It handles dependencies not covered by MemrefNormalization
/// (e.g., static arrays, token containers, direct memref allocations, block
/// arguments).
///===----------------------------------------------------------------------===///

#define GEN_PASS_DEF_HANDLEDEPS
#include "arts/Dialect.h"
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/Debug.h"
#include "arts/utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

ARTS_DEBUG_SETUP(handle_deps);

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::omp;

namespace {

/// Info about an OMP dependency (local to this pass)
struct DepInfo {
  Value source;
  SmallVector<Value> indices, sizes;
  ArtsMode mode;
  omp::TaskOp taskOp;
  unsigned depVarIndex;
  SmallVector<Operation *> opsToRemove;
};

/// Check if alloca is a scalar token container (rank-0 or rank-1 with size <=
/// 1)
static bool isScalarTokenContainer(memref::AllocaOp allocaOp) {
  auto type = dyn_cast<MemRefType>(allocaOp.getType());
  if (!type)
    return false;
  return type.getRank() == 0 || (type.getRank() == 1 && type.hasStaticShape() &&
                                 type.getNumElements() <= 1);
}

/// Find the value stored to a token container alloca
static std::optional<Value> getStoredValueToToken(memref::AllocaOp allocaOp) {
  for (Operation *user : allocaOp->getUsers()) {
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (store.getMemref() == allocaOp.getResult())
        return store.getValue();
    }
  }
  return std::nullopt;
}

struct HandleDepsPass : public impl::HandleDepsBase<HandleDepsPass> {
  void runOnOperation() override;

private:
  std::optional<DepInfo> extractDepInfo(Value depVar, omp::TaskOp taskOp,
                                        unsigned depIdx,
                                        omp::ClauseTaskDepend depMode,
                                        OpBuilder &builder);
};

///===----------------------------------------------------------------------===///
/// extractDepInfo
///===----------------------------------------------------------------------===///

std::optional<DepInfo>
HandleDepsPass::extractDepInfo(Value depVar, omp::TaskOp taskOp,
                               unsigned depIdx, omp::ClauseTaskDepend depMode,
                               OpBuilder &builder) {

  /// Skip if already arts.omp_dep
  if (depVar.getDefiningOp<arts::OmpDepOp>())
    return std::nullopt;

  DepInfo info;
  info.taskOp = taskOp;
  info.depVarIndex = depIdx;
  info.mode = arts::convertOmpMode(depMode);
  Location loc = taskOp.getLoc();

  Operation *defOp = depVar.getDefiningOp();

  /// Pattern 1: Token container alloca - follow store->load chain
  if (auto allocaOp = dyn_cast_or_null<memref::AllocaOp>(defOp)) {
    if (isScalarTokenContainer(allocaOp)) {
      if (auto storedVal = getStoredValueToToken(allocaOp)) {
        /// Stored value should be from a memref.load
        if (auto loadOp = storedVal->getDefiningOp<memref::LoadOp>()) {
          info.source = loadOp.getMemref();
          info.indices.append(loadOp.getIndices().begin(),
                              loadOp.getIndices().end());
          for (size_t d = 0; d < info.indices.size(); ++d)
            info.sizes.push_back(arts::createConstantIndex(builder, loc, 1));
          info.indices =
              arts::clampDepIndices(info.source, info.indices, builder, loc);
          return info;
        }
      }
    }
    /// Non-token alloca - whole array dependency
    info.source = depVar;
    return info;
  }

  /// Pattern 2: memref.alloc - whole array dependency
  if (isa_and_nonnull<memref::AllocOp>(defOp)) {
    info.source = depVar;
    return info;
  }

  /// Pattern 2.5: memref.cast - look through to source
  /// Chunked dependencies generate: subview -> cast -> depend
  /// We need to look through the cast to find the underlying subview.
  if (auto castOp = dyn_cast_or_null<memref::CastOp>(defOp)) {
    Value castSource = castOp.getSource();
    if (auto subview = castSource.getDefiningOp<memref::SubViewOp>()) {
      info.source = subview.getSource();
      for (auto offset : subview.getMixedOffsets())
        info.indices.push_back(
            getValueOrCreateConstantIndexOp(builder, loc, offset));
      for (auto size : subview.getMixedSizes())
        info.sizes.push_back(
            getValueOrCreateConstantIndexOp(builder, loc, size));
      return info;
    }
    /// Cast of something other than subview - use cast source as-is
    info.source = castSource;
    return info;
  }

  /// Pattern 3: SubView - chunk dependency
  if (auto subview = dyn_cast_or_null<memref::SubViewOp>(defOp)) {
    info.source = subview.getSource();
    for (auto offset : subview.getMixedOffsets())
      info.indices.push_back(
          getValueOrCreateConstantIndexOp(builder, loc, offset));
    for (auto size : subview.getMixedSizes())
      info.sizes.push_back(getValueOrCreateConstantIndexOp(builder, loc, size));
    return info;
  }

  /// Pattern 4: memref.load - element/row dependency
  if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(defOp)) {
    info.source = loadOp.getMemref();
    info.indices.append(loadOp.getIndices().begin(), loadOp.getIndices().end());
    for (size_t d = 0; d < info.indices.size(); ++d)
      info.sizes.push_back(arts::createConstantIndex(builder, loc, 1));
    info.indices =
        arts::clampDepIndices(info.source, info.indices, builder, loc);
    return info;
  }

  /// Pattern 5: Block argument
  if (isa<BlockArgument>(depVar)) {
    info.source = depVar;
    return info;
  }

  return std::nullopt;
}

///===----------------------------------------------------------------------===///
/// runOnOperation
///===----------------------------------------------------------------------===///

void HandleDepsPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  ARTS_INFO_HEADER(HandleDepsPass);

  OpBuilder builder(ctx);

  ARTS_DEBUG("=== Handling Remaining OMP Dependencies ===");

  module.walk([&](omp::TaskOp task) {
    auto dependVars = task.getDependVars();
    auto dependAttrs = task.getDependKindsAttr();
    if (dependVars.empty() || !dependAttrs)
      return;

    for (unsigned i = 0; i < dependVars.size(); ++i) {
      Value depVar = dependVars[i];

      /// Skip already processed (arts.omp_dep)
      if (depVar.getDefiningOp<arts::OmpDepOp>())
        continue;

      /// Set insertion point before task - needed for creating constants in
      /// extractDepInfo
      builder.setInsertionPoint(task);

      auto depAttr = cast<omp::ClauseTaskDependAttr>(dependAttrs[i]);
      auto depInfo =
          extractDepInfo(depVar, task, i, depAttr.getValue(), builder);
      if (!depInfo)
        continue;

      /// Create arts.omp_dep using DepInfo fields (insertion point already set)
      auto ompDepOp =
          arts::OmpDepOp::create(builder, task.getLoc(),
                                 depInfo->source.getType(), /// result type
                                 depInfo->mode,             /// ArtsMode
                                 depInfo->source,           /// db
                                 depInfo->indices,          /// offset
                                 depInfo->sizes);           /// size

      task.getDependVarsMutable()[i].set(ompDepOp.getResult());
      ARTS_DEBUG("  Created arts.omp_dep for task dep "
                 << i << " with " << depInfo->indices.size() << " indices");
    }
  });

  ARTS_DEBUG("=== Dep Handling Complete ===");
  ARTS_INFO_FOOTER(HandleDepsPass);
}

} // namespace

namespace mlir {
namespace arts {
std::unique_ptr<Pass> createHandleDepsPass() {
  return std::make_unique<HandleDepsPass>();
}
} // namespace arts
} // namespace mlir
