///==========================================================================///
/// File: GuidRangeCallOpt.cpp
///
/// Rewrites per-iteration GUID reservation calls into GUID-range calls after
/// ARTS-RT-to-LLVM lowering, when safe:
///
///   scf.for %i = 0 to N step 1 {
///     %g = call @arts_guid_reserve(type, route0)
///     ...
///   }
///
/// becomes:
///
///   %range = call @arts_guid_reserve_range(type, N, route0)
///   scf.for %i = 0 to N step 1 {
///     %g = call @arts_guid_from_index(%range, %i32)
///     ...
///   }
///
/// This pass is intentionally conservative and only applies when:
/// - lower bound is 0 and step is 1,
/// - exactly one @arts_guid_reserve call appears in the loop body,
/// - route operand is compile-time constant 0.
///
/// For dynamic trip counts, it emits a guarded rewrite:
/// - if (upperBound <= UINT32_MAX): use reserve_range/from_index
/// - else: keep per-iteration reserve semantics
///==========================================================================///

#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/CodegenSupport.h"
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Types.h"
#include "carts/dialect/arts-rt/Transforms/Passes.h"
#include "carts/dialect/arts-rt/Utils/RuntimeCallUtils.h"
#include "carts/utils/LoopUtils.h"
#include "carts/utils/ValueAnalysis.h"
namespace mlir::carts::arts_rt {
#define GEN_PASS_DEF_GUIDRANGECALLOPT
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts_rt
#include "carts/passes/Passes.h"
#include "carts/utils/Debug.h"
#include "mlir/IR/BuiltinOps.h"
#include <cstdint>
#include <limits>

ARTS_DEBUG_SETUP(guid_range_call_opt);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts_rt;
using mlir::carts::arts::debugStream;
using mlir::carts::getStaticTripCount;
using mlir::carts::arts::ValueAnalysis;

namespace {

static bool isGuidReserveCall(func::CallOp call) {
  if (call.getNumOperands() != 2 || call.getNumResults() != 1)
    return false;
  std::optional<types::RuntimeFunction> fn =
      getRuntimeFunction(call);
  return fn && *fn == types::ARTSRTL_arts_guid_reserve;
}

struct GuidRangeCallOptPass
    : public arts_rt::impl::GuidRangeCallOptBase<GuidRangeCallOptPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    ArtsCodegen codegen(module, /*debug=*/false);
    MLIRContext *ctx = module.getContext();

    Type i32 = IntegerType::get(ctx, 32);
    Type i64 = IntegerType::get(ctx, 64);
    func::FuncOp reserveFn =
        codegen.getOrCreateRuntimeFunction(types::ARTSRTL_arts_guid_reserve);
    func::FuncOp reserveRangeFn = codegen.getOrCreateRuntimeFunction(
        types::ARTSRTL_arts_guid_reserve_range);
    func::FuncOp fromIndexFn =
        codegen.getOrCreateRuntimeFunction(types::ARTSRTL_arts_guid_from_index);

    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp loop) { loops.push_back(loop); });

    int rewrittenStatic = 0;
    int rewrittenGuarded = 0;
    for (scf::ForOp loop : loops) {
      auto lowerBound =
          ValueAnalysis::tryFoldConstantIndex(loop.getLowerBound());
      auto step = ValueAnalysis::tryFoldConstantIndex(loop.getStep());
      if (!lowerBound || !step || *lowerBound != 0 || *step != 1)
        continue;

      SmallVector<func::CallOp> reserveCalls;
      for (Operation &op : loop.getBody()->without_terminator()) {
        if (auto call = dyn_cast<func::CallOp>(op))
          if (isGuidReserveCall(call))
            reserveCalls.push_back(call);
      }

      if (reserveCalls.empty())
        continue;

      for (func::CallOp reserveCall : reserveCalls) {
        auto routeConst =
            ValueAnalysis::tryFoldConstantIndex(reserveCall.getOperand(1));
        if (!routeConst || *routeConst != 0)
          continue;

        auto staticTripCount = getStaticTripCount(loop.getOperation());
        if (staticTripCount &&
            (*staticTripCount < 0 ||
             *staticTripCount >
                 static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))) {
          staticTripCount.reset();
        }
        if (staticTripCount && *staticTripCount <= 1)
          continue;

        /// Fast path: fully static trip count.
        if (staticTripCount && *staticTripCount > 1) {
          OpBuilder b(loop);
          Value tripCountI32 = arith::ConstantIntOp::create(
              b, loop.getLoc(), *staticTripCount, 32);
          auto reserveRange = func::CallOp::create(
              b, loop.getLoc(), reserveRangeFn,
              ValueRange{reserveCall.getOperand(0), tripCountI32,
                         reserveCall.getOperand(1)});
          Value rangeGuid = reserveRange.getResult(0);

          OpBuilder callBuilder(reserveCall);
          Value ivI32 = arith::IndexCastOp::create(
              callBuilder, reserveCall.getLoc(), i32, loop.getInductionVar());
          auto fromIndex =
              func::CallOp::create(callBuilder, reserveCall.getLoc(),
                                   fromIndexFn, ValueRange{rangeGuid, ivI32});
          reserveCall.getResult(0).replaceAllUsesWith(fromIndex.getResult(0));
          reserveCall.erase();
          ++rewrittenStatic;
          continue;
        }

        /// Guarded path for dynamic trip counts:
        /// preserve exact semantics when upper bound exceeds uint32 range.
        OpBuilder b(loop);
        Location loc = loop.getLoc();
        Value ubI64 =
            arith::IndexCastUIOp::create(b, loc, i64, loop.getUpperBound());
        Value oneI64 = arith::ConstantIntOp::create(b, loc, 1, 64);
        Value u32Max = arith::ConstantIntOp::create(
            b, loc, std::numeric_limits<uint32_t>::max(), 64);
        Value fitsU32 = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule,
                                              ubI64, u32Max);
        Value moreThanOne = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::ugt, ubI64, oneI64);
        Value canUseRange = arith::AndIOp::create(b, loc, fitsU32, moreThanOne);

        auto chooseRangeGuid =
            scf::IfOp::create(b, loc, TypeRange{i64}, canUseRange,
                              /*withElseRegion=*/true);
        {
          OpBuilder thenBuilder =
              OpBuilder::atBlockBegin(&chooseRangeGuid.getThenRegion().front());
          Value tripCountI32 = arith::IndexCastOp::create(thenBuilder, loc, i32,
                                                          loop.getUpperBound());
          auto reserveRange = func::CallOp::create(
              thenBuilder, loc, reserveRangeFn,
              ValueRange{reserveCall.getOperand(0), tripCountI32,
                         reserveCall.getOperand(1)});
          scf::YieldOp::create(thenBuilder, loc, reserveRange.getResult(0));
        }
        {
          OpBuilder elseBuilder =
              OpBuilder::atBlockBegin(&chooseRangeGuid.getElseRegion().front());
          Value nullGuid =
              arith::ConstantIntOp::create(elseBuilder, loc, 0, 64);
          scf::YieldOp::create(elseBuilder, loc, nullGuid);
        }
        Value rangeGuid = chooseRangeGuid.getResult(0);

        OpBuilder callBuilder(reserveCall);
        auto chooseGuid = scf::IfOp::create(callBuilder, reserveCall.getLoc(),
                                            TypeRange{i64}, canUseRange,
                                            /*withElseRegion=*/true);
        {
          OpBuilder thenBuilder =
              OpBuilder::atBlockBegin(&chooseGuid.getThenRegion().front());
          Value ivI32 = arith::IndexCastOp::create(
              thenBuilder, reserveCall.getLoc(), i32, loop.getInductionVar());
          auto fromIndex =
              func::CallOp::create(thenBuilder, reserveCall.getLoc(),
                                   fromIndexFn, ValueRange{rangeGuid, ivI32});
          scf::YieldOp::create(thenBuilder, reserveCall.getLoc(),
                               fromIndex.getResult(0));
        }
        {
          OpBuilder elseBuilder =
              OpBuilder::atBlockBegin(&chooseGuid.getElseRegion().front());
          auto reserve =
              func::CallOp::create(elseBuilder, reserveCall.getLoc(), reserveFn,
                                   reserveCall.getOperands());
          scf::YieldOp::create(elseBuilder, reserveCall.getLoc(),
                               reserve.getResult(0));
        }

        reserveCall.getResult(0).replaceAllUsesWith(chooseGuid.getResult(0));
        reserveCall.erase();
        ++rewrittenGuarded;
      }
    }

    ARTS_INFO("GuidRangeCallOpt: rewritten "
              << rewrittenStatic << " static loops and " << rewrittenGuarded
              << " guarded dynamic loops");
  }
};

} // namespace

namespace mlir {
namespace carts::arts_rt {

std::unique_ptr<Pass> createGuidRangeCallOptPass() {
  return std::make_unique<GuidRangeCallOptPass>();
}

} // namespace carts::arts_rt
} // namespace mlir
