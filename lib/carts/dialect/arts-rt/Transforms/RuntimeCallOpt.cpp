///==========================================================================///
/// File: RuntimeCallOpt.cpp
///
/// Post-ARTS-RT-to-LLVM optimization for selected pure runtime query calls.
///
/// Scope is intentionally conservative:
/// - optimize only known topology/identity queries with no operands
/// - never rewrite stateful runtime calls (guid reserve/create, DB create,
///   EDT/event signaling, etc.)
///
/// Transformations:
///  1) Hoist loop-invariant calls out of nested scf.for loops
///  2) Deduplicate dominated duplicate calls per callee
///==========================================================================///

#include "carts/dialect/arts-rt/Transforms/Passes.h"
namespace mlir::carts::arts_rt {
#define GEN_PASS_DEF_RUNTIMECALLOPT
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts_rt
#include "carts/dialect/arts-rt/Utils/RuntimeCallUtils.h"
#include "carts/dialect/arts/Utils/LoopInvarianceUtils.h"
#include "carts/passes/Passes.h"
#include "carts/utils/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"

ARTS_DEBUG_SETUP(runtime_call_opt);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts_rt;
using mlir::carts::debugStream;
using mlir::carts::arts::findHoistTarget;

namespace {

static std::optional<types::RuntimeFunction>
getOptimizableRuntimeFunction(func::CallOp call) {
  if (!call || call.getNumOperands() != 0 || call.getNumResults() != 1)
    return std::nullopt;
  return getRuntimeTopologyQueryFunction(call);
}

struct RuntimeCallOptPass
    : public arts_rt::impl::RuntimeCallOptBase<RuntimeCallOptPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    int hoisted = 0;
    int deduplicated = 0;

    module.walk([&](func::FuncOp funcOp) {
      DominanceInfo domInfo(funcOp);
      SmallVector<func::CallOp> calls;
      funcOp.walk([&](func::CallOp call) {
        if (getOptimizableRuntimeFunction(call))
          calls.push_back(call);
      });

      /// Phase 1: Hoist selected loop-invariant runtime calls.
      for (func::CallOp call : calls) {
        if (!call)
          continue;
        scf::ForOp target =
            findHoistTarget(call.getOperation(), call.getOperation(), domInfo);
        if (!target || !target->isAncestor(call))
          continue;
        call->moveBefore(target);
        ++hoisted;
      }

      /// Phase 2: Deduplicate dominated duplicate calls per callee.
      DominanceInfo postHoistDomInfo(funcOp);
      llvm::DenseMap<types::RuntimeFunction, SmallVector<func::CallOp, 2>>
          dominatingByFn;
      SmallVector<func::CallOp> orderedCalls;
      funcOp.walk([&](func::CallOp call) {
        if (getOptimizableRuntimeFunction(call))
          orderedCalls.push_back(call);
      });

      for (func::CallOp call : orderedCalls) {
        if (!call)
          continue;
        std::optional<types::RuntimeFunction> fn =
            getOptimizableRuntimeFunction(call);
        if (!fn)
          continue;

        auto &dominatingCalls = dominatingByFn[*fn];
        func::CallOp replacement;
        for (func::CallOp candidate : dominatingCalls) {
          if (!candidate)
            continue;
          if (candidate.getResult(0).getType() != call.getResult(0).getType())
            continue;
          if (postHoistDomInfo.dominates(candidate.getOperation(),
                                         call.getOperation())) {
            replacement = candidate;
            break;
          }
        }

        if (replacement) {
          call.getResult(0).replaceAllUsesWith(replacement.getResult(0));
          call.erase();
          ++deduplicated;
          continue;
        }
        dominatingCalls.push_back(call);
      }
    });

    ARTS_INFO("RuntimeCallOpt: hoisted " << hoisted << " calls, deduplicated "
                                         << deduplicated << " calls");
  }
};

} // namespace

namespace mlir {
namespace carts::arts_rt {

std::unique_ptr<Pass> createRuntimeCallOptPass() {
  return std::make_unique<RuntimeCallOptPass>();
}

} // namespace carts::arts_rt
} // namespace mlir
