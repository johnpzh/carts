
///==========================================================================///
/// File: Inliner.cpp
///
/// Specialized inliner policy for ARTS pipelines.
///
/// Example:
///   Before:
///     func.func @helper(...) { ... }
///     func.func @kernel(...) { call @helper(...) }
///
///   After (when profitable/legal under ARTS policy):
///     func.func @kernel(...) {
///       ... inlined body of @helper ...
///     }
///==========================================================================///

#define GEN_PASS_DEF_ARTSINLINER
#include "arts/Dialect.h"
#include "arts/passes/Passes.h"
#include "arts/passes/Passes.h.inc"
#include "arts/utils/Utils.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(inliner);

using namespace mlir;
using namespace mlir::arts;

namespace {

static bool containsSequentialLoopLikeOp(Operation *op) {
  bool found = false;
  op->walk([&](Operation *nested) {
    if (isa<scf::ForOp, scf::WhileOp, scf::ParallelOp, scf::ForallOp,
            affine::AffineForOp, affine::AffineParallelOp>(nested)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool hasNestedMemrefType(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  return memrefType && isa<MemRefType>(memrefType.getElementType());
}

static bool hasMemrefType(Type type) { return isa<MemRefType>(type); }

template <typename TypePredicate>
static bool callOrCalleeUsesType(func::CallOp call,
                                 CallableOpInterface callableOp,
                                 TypePredicate predicate) {
  for (Value operand : call.getOperands()) {
    if (predicate(operand.getType()))
      return true;
  }

  for (Type resultType : call.getResultTypes()) {
    if (predicate(resultType))
      return true;
  }

  auto funcOp = dyn_cast_or_null<func::FuncOp>(callableOp.getOperation());
  if (!funcOp)
    return false;

  for (BlockArgument arg : funcOp.getArguments()) {
    if (predicate(arg.getType()))
      return true;
  }

  for (Type resultType : funcOp.getFunctionType().getResults()) {
    if (predicate(resultType))
      return true;
  }

  return false;
}

static bool
hasLayoutSensitiveNestedMemrefSignature(func::CallOp call,
                                        CallableOpInterface callableOp) {
  return callOrCalleeUsesType(call, callableOp, hasNestedMemrefType);
}

static bool hasLayoutSensitiveMemrefSignature(func::CallOp call,
                                              CallableOpInterface callableOp) {
  return callOrCalleeUsesType(call, callableOp, hasMemrefType);
}

/// Keep early inlining focused on helpers that materially improve ARTS
/// analysis. Inlining large sequential loop helpers tends to rematerialize
/// initialized arrays into downstream kernels, which hurts benchmark codegen.
/// We still inline helpers that carry OpenMP because later passes need that
/// structure in the caller to expose ARTS parallelism and loop fusion.
///
/// Nested-memref helpers are the exception: MemrefNormalization changes
/// the storage layout of those values, so the helper body must be visible in
/// the caller before that pass runs. Otherwise the call boundary would retain
/// the stale pre-raise type/layout contract.
///
/// The same reasoning applies to plain memref helper functions that survive
/// into DB creation/partitioning. Once a caller-side allocation is promoted to
/// block/stencil DB layout, a remaining full-view helper call would otherwise
/// force coarse fallback because the callee still expects the original
/// contiguous memref contract.
static bool shouldInlineCall(func::CallOp call,
                             CallableOpInterface callableOp) {
  auto funcOp = dyn_cast_or_null<func::FuncOp>(callableOp.getOperation());
  if (!funcOp)
    return true;

  if (hasLayoutSensitiveNestedMemrefSignature(call, callableOp)) {
    ARTS_INFO("Inlining layout-sensitive nested-memref helper "
              << funcOp.getSymName());
    return true;
  }

  if (hasLayoutSensitiveMemrefSignature(call, callableOp) &&
      containsSequentialLoopLikeOp(funcOp)) {
    ARTS_INFO("Inlining layout-sensitive memref helper "
              << funcOp.getSymName());
    return true;
  }

  if (containsOmpOp(funcOp))
    return true;

  if (containsSequentialLoopLikeOp(funcOp)) {
    ARTS_INFO("Skipping sequential loop helper " << funcOp.getSymName()
                                                 << " to preserve materialized"
                                                 << " memory semantics");
    return false;
  }

  return true;
}

/// Conservative fallback for calls inside OMP regions. Generic MLIR inlining
/// can assert here because the OMP dialect does not register a
/// DialectInlinerInterface. For single-block func.func callees we can inline by
/// cloning the body and remapping the return operands directly.
static LogicalResult inlineSingleBlockFuncCallInOmp(func::CallOp call,
                                                    func::FuncOp callee) {
  Region &calleeRegion = callee.getBody();
  if (!calleeRegion.hasOneBlock())
    return failure();

  Block &calleeBlock = calleeRegion.front();
  auto returnOp = dyn_cast<func::ReturnOp>(calleeBlock.getTerminator());
  if (!returnOp)
    return failure();

  OpBuilder builder(call);
  IRMapping mapper;
  for (auto [arg, operand] :
       llvm::zip(callee.getArguments(), call.getOperands())) {
    mapper.map(arg, operand);
  }

  for (Operation &op : calleeBlock.without_terminator())
    builder.clone(op, mapper);

  SmallVector<Value> replacementValues;
  replacementValues.reserve(returnOp.getNumOperands());
  for (Value retVal : returnOp.getOperands())
    replacementValues.push_back(mapper.lookupOrDefault(retVal));

  if (call.getNumResults() != replacementValues.size())
    return failure();

  if (call.getNumResults() == 0) {
    call.erase();
    return success();
  }

  call.replaceAllUsesWith(replacementValues);
  call.erase();
  return success();
}

struct ArtsInlinerInterface : public mlir::InlinerInterface {
  using mlir::InlinerInterface::InlinerInterface;

  ///===--------------------------------------------------------------------===///
  /// Analysis Hooks
  ///===--------------------------------------------------------------------===///

  /// All call operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  ///===--------------------------------------------------------------------===///
  /// Transformation Hooks
  ///===--------------------------------------------------------------------===///

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op, mlir::Block *newDest) const final {
    /// Only "std.return" needs to be handled here.
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op);
    if (!returnOp)
      return;

    /// Replace the return with a branch to the dest.
    mlir::OpBuilder builder(op);
    mlir::cf::BranchOp::create(builder, op->getLoc(), newDest,
                               returnOp.getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op,
                        mlir::ValueRange valuesToRepl) const final {
    /// Only "std.return" needs to be handled here.
    auto returnOp = mlir::cast<mlir::func::ReturnOp>(op);

    /// Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};

struct ArtsInlinerPass : public impl::ArtsInlinerBase<ArtsInlinerPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ARTS_DEBUG_HEADER(ArtsInlinerPass);

    /// Build the inliner interface
    ArtsInlinerInterface inliner(&getContext());

    /// Use a worklist approach to handle newly created calls during inlining
    bool changed;
    unsigned iteration = 0;
    do {
      changed = false;
      iteration++;

      ARTS_INFO("Inlining iteration " << iteration);

      /// Collect all call ops in the module
      SmallVector<func::CallOp> calls;
      module.walk([&](func::CallOp call) { calls.push_back(call); });

      ARTS_INFO("Found " << calls.size() << " call operations");

      /// Process each call operation
      for (func::CallOp call : calls) {
        /// Skip already erased operations
        if (call->getParentOp() == nullptr) {
          ARTS_WARN("Skipping erased call");
          continue;
        }

        /// Resolve the callable operation
        auto callable = call.getCallableForCallee();
        CallableOpInterface callableOp;
        if (SymbolRefAttr symRef = dyn_cast<SymbolRefAttr>(callable)) {
          callableOp =
              dyn_cast<CallableOpInterface>(module.lookupSymbol(symRef));
        } else if (auto func = dyn_cast<Value>(callable)) {
          callableOp = dyn_cast<CallableOpInterface>(func.getDefiningOp());
        }

        if (!callableOp || !callableOp.getCallableRegion()) {
          ARTS_WARN("Cannot resolve callable for call");
          continue;
        }

        if (!shouldInlineCall(call, callableOp))
          continue;

        /// Generic MLIR inlining into OMP regions can assert because the OMP
        /// dialect does not register a DialectInlinerInterface. Use the
        /// conservative single-block fallback there instead.
        if (isInsideOmpRegion(call)) {
          if (auto funcOp = dyn_cast<func::FuncOp>(callableOp.getOperation())) {
            if (succeeded(inlineSingleBlockFuncCallInOmp(call, funcOp))) {
              ARTS_INFO("Successfully manually inlined call inside OMP region");
              changed = true;
              continue;
            }
          }
          ARTS_INFO("Skipping call inside OMP region");
          continue;
        }

        /// Skip if the callable region is empty
        if (callableOp.getCallableRegion()->empty()) {
          ARTS_INFO("Skipping call to empty function");
          continue;
        }

        /// Check if this is a recursive call (basic check)
        if (auto funcOp = dyn_cast<func::FuncOp>(callableOp.getOperation())) {
          auto enclosingFn = call->getParentOfType<func::FuncOp>();
          if (enclosingFn && enclosingFn.getSymName() == funcOp.getSymName()) {
            ARTS_INFO("Skipping recursive call to " << funcOp.getSymName());
            continue;
          }
        }

        ARTS_INFO("Attempting to inline call");

        /// Attempt to inline the call
        auto cloneCallback = [](OpBuilder &builder, Region *src,
                                Block *inlineBlock, Block *postInsertBlock,
                                IRMapping &mapper,
                                bool shouldCloneInlinedRegion) {
          Region *insertRegion = inlineBlock->getParent();
          if (shouldCloneInlinedRegion)
            src->cloneInto(insertRegion, postInsertBlock->getIterator(),
                           mapper);
          else
            insertRegion->getBlocks().splice(postInsertBlock->getIterator(),
                                             src->getBlocks(), src->begin(),
                                             src->end());
        };
        if (succeeded(mlir::inlineCall(inliner, cloneCallback, call, callableOp,
                                       callableOp.getCallableRegion(), true))) {
          ARTS_INFO("Successfully inlined call");
          call.erase();
          changed = true;
        } else {
          ARTS_WARN("Failed to inline call");
        }
      }
      /// Limit iterations to prevent infinite loops
    } while (changed && iteration < 100);

    if (iteration >= 100)
      ARTS_WARN("Hit maximum iteration limit");

    ARTS_INFO("Completed aggressive inlining after " << iteration
                                                     << " iterations");

    /// Clean up unused functions
    cleanupUnusedFunctions(module);

    ARTS_DEBUG_FOOTER(ArtsInlinerPass);
    ARTS_DEBUG_REGION(module.dump(););
  }

private:
  void cleanupUnusedFunctions(ModuleOp module) {
    ARTS_INFO("Cleaning up unused functions");

    /// Collect all function operations
    SmallVector<func::FuncOp> functions;
    module.walk([&](func::FuncOp func) { functions.push_back(func); });

    /// Remove functions that are no longer used
    for (func::FuncOp func : functions) {
      if (func.isPrivate() && func.getSymbolUses(module)->empty()) {
        ARTS_INFO("Removing unused function: " << func.getSymName());
        func.erase();
      }
    }
  }
};

} // namespace

namespace mlir {
namespace arts {
std::unique_ptr<Pass> createArtsInlinerPass() {
  return std::make_unique<ArtsInlinerPass>();
}
} // namespace arts
} // namespace mlir
