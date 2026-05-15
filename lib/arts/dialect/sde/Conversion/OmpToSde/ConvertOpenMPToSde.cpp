///==========================================================================///
/// File: ConvertOpenMPToSde.cpp
///
/// This file implements a module pass that converts OpenMP ops into SDE ops,
/// preserving source semantics that direct target conversion would otherwise
/// lose:
/// - Reduction combiner kind + identity
/// - Nowait semantics
/// - Schedule + chunk size
/// - Task completion tokens
///
/// Example:
///   Before:
///     omp.parallel {
///       omp.wsloop reduction(+: sum) schedule(static, 4) {
///         omp.loop_nest (%i) : index = (%c0) to (%N) step (%c1) { ... }
///       }
///     }
///
///   After:
///     sde.cu_region parallel {
///       sde.su_iterate (%c0) to (%N) step (%c1)
///           schedule(<static>, %c4)
///           reduction [#sde<reduction_kind<add>>] (%sum : f64) {
///         ...
///         sde.yield
///       }
///       sde.yield
///     }
///==========================================================================///

#include "arts/dialect/sde/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_CONVERTOPENMPTOSDE
#include "arts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::arts

#include "arts/utils/Utils.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Ops.h"

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(convert_openmp_to_sde);

#include <optional>

#include "arts/utils/costs/SDECostModel.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Statistic.h"
static llvm::Statistic numParallelConverted{
    "convert_openmp_to_sde", "NumParallelConverted",
    "Number of omp.parallel regions converted to sde.cu_region"};
static llvm::Statistic numWsloopsConverted{
    "convert_openmp_to_sde", "NumWsloopsConverted",
    "Number of omp.wsloop converted to sde.su_iterate"};
static llvm::Statistic numTasksConverted{
    "convert_openmp_to_sde", "NumTasksConverted",
    "Number of omp.task converted to sde.cu_task"};
static llvm::Statistic numAtomicsConverted{
    "convert_openmp_to_sde", "NumAtomicsConverted",
    "Number of omp.atomic.update converted to sde.cu_atomic"};

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::carts;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Ensure a value has index type, inserting arith.index_cast if needed.
static Value ensureIndex(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isIndex())
    return v;
  return arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
}

/// Ensure all values in a range have index type.
static SmallVector<Value> ensureIndexRange(OpBuilder &b, Location loc,
                                           ValueRange vals) {
  SmallVector<Value> result;
  for (Value v : vals)
    result.push_back(ensureIndex(b, loc, v));
  return result;
}

/// Map OMP schedule kind to SDE schedule kind.
static std::optional<sde::SdeScheduleKind>
convertScheduleKind(omp::ClauseScheduleKind kind) {
  switch (kind) {
  case omp::ClauseScheduleKind::Static:
    return sde::SdeScheduleKind::static_;
  case omp::ClauseScheduleKind::Dynamic:
    return sde::SdeScheduleKind::dynamic;
  case omp::ClauseScheduleKind::Guided:
    return sde::SdeScheduleKind::guided;
  case omp::ClauseScheduleKind::Auto:
    return sde::SdeScheduleKind::auto_;
  case omp::ClauseScheduleKind::Runtime:
    return sde::SdeScheduleKind::runtime;
  default:
    return std::nullopt;
  }
}

static std::optional<sde::SdeAccessMode>
convertDependMode(omp::ClauseTaskDepend mode) {
  switch (mode) {
  case omp::ClauseTaskDepend::taskdependin:
    return sde::SdeAccessMode::read;
  case omp::ClauseTaskDepend::taskdependout:
    return sde::SdeAccessMode::write;
  case omp::ClauseTaskDepend::taskdependinout:
    return sde::SdeAccessMode::readwrite;
  case omp::ClauseTaskDepend::taskdependmutexinoutset:
  case omp::ClauseTaskDepend::taskdependinoutset:
    return sde::SdeAccessMode::readwrite;
  }
  return std::nullopt;
}

static void attachPendingControlTokensInBlock(Block &block) {
  SmallVector<Value> pendingTokens;

  for (auto it = block.begin(), end = block.end(); it != end;) {
    Operation &op = *it++;

    for (Region &region : op.getRegions())
      for (Block &nested : region)
        attachPendingControlTokensInBlock(nested);

    if (auto token = dyn_cast<sde::SdeControlTokenOp>(op)) {
      pendingTokens.push_back(token.getToken());
      continue;
    }

    auto barrier = dyn_cast<sde::SdeSuBarrierOp>(op);
    if (!barrier)
      continue;

    if (barrier.getTokens().empty() && !pendingTokens.empty()) {
      OpBuilder builder(barrier);
      sde::SdeSuBarrierOp::create(builder, barrier.getLoc(), pendingTokens,
                                  barrier.getBarrierEliminatedAttr(),
                                  barrier.getBarrierReasonAttr());
      barrier.erase();
    }

    pendingTokens.clear();
  }
}

static void attachTaskwaitControlTokens(ModuleOp module) {
  for (Region &region : module->getRegions())
    for (Block &block : region)
      attachPendingControlTokensInBlock(block);
}

/// Infer SDE reduction kind from OMP DeclareReductionOp combiner body.
static sde::SdeReductionKind inferReductionKind(omp::DeclareReductionOp decl) {
  Block &combinerBlock = decl.getReductionRegion().front();
  for (Operation &op : combinerBlock.without_terminator()) {
    if (isa<arith::AddFOp, arith::AddIOp>(op))
      return sde::SdeReductionKind::add;
    if (isa<arith::MulFOp, arith::MulIOp>(op))
      return sde::SdeReductionKind::mul;
    if (isa<arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp>(op))
      return sde::SdeReductionKind::min;
    if (isa<arith::MaximumFOp, arith::MaxSIOp, arith::MaxUIOp>(op))
      return sde::SdeReductionKind::max;
    if (isa<arith::AndIOp>(op))
      return sde::SdeReductionKind::land;
    if (isa<arith::OrIOp>(op))
      return sde::SdeReductionKind::lor;
    if (isa<arith::XOrIOp>(op))
      return sde::SdeReductionKind::lxor;
  }
  return sde::SdeReductionKind::custom;
}

/// Helper to create a UnitAttr when nowait is true, nullptr otherwise.
static UnitAttr nowaitAttr(MLIRContext *ctx, bool nowait) {
  return nowait ? UnitAttr::get(ctx) : nullptr;
}

struct OmpDependSlice {
  Value source;
  SmallVector<Value> offsets;
  SmallVector<Value> sizes;
};

struct TaskDependSpec {
  sde::SdeAccessMode mode = sde::SdeAccessMode::read;
  OmpDependSlice slice;
};

/// OpenMP task-depend lowering ingests normalized memref values and may reuse
/// already-materialized SDE dependency carriers in SDE-native tests.
static std::optional<OmpDependSlice>
extractDependSlice(Value depVar, OpBuilder &builder, Location loc) {
  // Path 1: SDE dependency carrier.
  if (auto muDepOp = depVar.getDefiningOp<sde::SdeMuDepOp>()) {
    OmpDependSlice slice;
    slice.source = muDepOp.getSource();
    slice.offsets.assign(muDepOp.getOffsets().begin(),
                         muDepOp.getOffsets().end());
    slice.sizes.assign(muDepOp.getSizes().begin(), muDepOp.getSizes().end());
    return slice;
  }

  // Path 2: direct memref.subview — Polygeist generates depend(in: %subview)
  if (auto subviewOp = depVar.getDefiningOp<memref::SubViewOp>()) {
    OmpDependSlice slice;
    slice.source = subviewOp.getSource();
    // Extract dynamic offsets and sizes (skip static ones — dep matching
    // in SDE uses source identity, not offset precision).
    for (Value off : subviewOp.getOffsets())
      slice.offsets.push_back(ensureIndex(builder, loc, off));
    for (Value sz : subviewOp.getSizes())
      slice.sizes.push_back(ensureIndex(builder, loc, sz));
    return slice;
  }

  // Path 3: polygeist.subindex — common for task depend(element) after C
  // lowering. Preserve the base source plus element offset so SDE can identify
  // source task-dependency patterns before boundary materialization.
  if (auto subIndexOp = depVar.getDefiningOp<polygeist::SubIndexOp>()) {
    OmpDependSlice slice;
    slice.source = subIndexOp.getSource();
    slice.offsets.push_back(ensureIndex(builder, loc, subIndexOp.getIndex()));
    slice.sizes.push_back(arts::createOneIndex(builder, loc));
    return slice;
  }

  // Path 4: plain memref — whole-array dependency
  if (isa<MemRefType>(depVar.getType())) {
    OmpDependSlice slice;
    slice.source = depVar;
    return slice;
  }

  return std::nullopt;
}

static SmallVector<Value> collectEnclosingScfLoopIvs(Operation *op) {
  SmallVector<Value> ivs;
  for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent))
      ivs.push_back(forOp.getInductionVar());
  }
  return ivs;
}

static bool dependsOnAny(Value value, ArrayRef<Value> roots) {
  for (Value root : roots)
    if (arts::ValueAnalysis::dependsOn(value, root))
      return true;
  return false;
}

static bool sameDependSource(Value lhs, Value rhs) {
  return arts::ValueAnalysis::sameValue(arts::ValueAnalysis::stripMemrefViewOps(lhs),
                                  arts::ValueAnalysis::stripMemrefViewOps(rhs));
}

static bool isElementDependSlice(const OmpDependSlice &slice) {
  if (slice.offsets.size() != 1)
    return false;
  if (slice.sizes.empty())
    return true;
  return slice.sizes.size() == 1 &&
         arts::ValueAnalysis::isOneConstant(slice.sizes.front());
}

static bool isWavefrontTaskDependPattern(Operation *taskOp,
                                         ArrayRef<TaskDependSpec> deps) {
  SmallVector<Value> enclosingIvs = collectEnclosingScfLoopIvs(taskOp);
  if (enclosingIvs.size() < 2)
    return false;

  const TaskDependSpec *write = nullptr;
  SmallVector<const TaskDependSpec *, 2> reads;
  for (const TaskDependSpec &dep : deps) {
    if (!isElementDependSlice(dep.slice))
      return false;
    if (!dependsOnAny(dep.slice.offsets.front(), enclosingIvs))
      return false;

    switch (dep.mode) {
    case sde::SdeAccessMode::write:
      if (write)
        return false;
      write = &dep;
      break;
    case sde::SdeAccessMode::read:
      reads.push_back(&dep);
      break;
    case sde::SdeAccessMode::readwrite:
      return false;
    }
  }

  if (!write || reads.size() < 2)
    return false;

  for (const TaskDependSpec *read : reads) {
    if (!sameDependSource(read->slice.source, write->slice.source))
      return false;
    if (arts::ValueAnalysis::sameValue(read->slice.offsets.front(),
                                 write->slice.offsets.front()))
      return false;
  }

  return true;
}

static void mapWsloopCapturedArgs(omp::WsloopOp op, IRMapping &mapper) {
  if (op.getRegion().empty())
    return;

  Block &wrapper = op.getRegion().front();
  unsigned argIndex = 0;

  for (auto privateVar : op.getPrivateVars()) {
    if (argIndex >= wrapper.getNumArguments())
      break;
    mapper.map(wrapper.getArgument(argIndex++), privateVar);
  }

  for (auto reductionVar : op.getReductionVars()) {
    if (argIndex >= wrapper.getNumArguments())
      break;
    mapper.map(wrapper.getArgument(argIndex++), reductionVar);
  }
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// omp.parallel -> sde.cu_region parallel
struct OMPParallelToSdePattern : public OpRewritePattern<omp::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Converting omp.parallel to sde.cu_region parallel");
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto cuRegion = sde::SdeCuRegionOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{},
        sde::SdeCuKindAttr::get(ctx, sde::SdeCuKind::parallel),
        /*concurrency_scope=*/nullptr,
        /*nowait=*/nullptr,
        /*iterArgs=*/ValueRange{});

    Block &old = op.getRegion().front();
    Block &blk = sde::ensureBlock(cuRegion.getBody());
    blk.getOperations().splice(blk.end(), old.getOperations());

    ++numParallelConverted;
    rewriter.eraseOp(op);
    return success();
  }
};

/// omp.master -> sde.cu_region single
struct MasterToSdePattern : public OpRewritePattern<omp::MasterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::MasterOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto cuRegion = sde::SdeCuRegionOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{},
        sde::SdeCuKindAttr::get(ctx, sde::SdeCuKind::single),
        sde::SdeConcurrencyScopeAttr::get(ctx, sde::SdeConcurrencyScope::local),
        /*nowait=*/nullptr,
        /*iterArgs=*/ValueRange{});
    Block &old = op.getRegion().front();
    Block &blk = sde::ensureBlock(cuRegion.getBody());
    blk.getOperations().splice(blk.end(), old.getOperations());
    // omp.master has implicit barrier (no nowait clause).
    rewriter.setInsertionPointAfter(cuRegion);
    sde::SdeSuBarrierOp::create(rewriter, loc, ValueRange{},
                                /*barrierEliminated=*/nullptr,
                                /*barrierReason=*/nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

struct SingleToSdePattern : public OpRewritePattern<omp::SingleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::SingleOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto cuRegion = sde::SdeCuRegionOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{},
        sde::SdeCuKindAttr::get(ctx, sde::SdeCuKind::single),
        sde::SdeConcurrencyScopeAttr::get(ctx, sde::SdeConcurrencyScope::local),
        nowaitAttr(ctx, op.getNowait()),
        /*iterArgs=*/ValueRange{});
    Block &old = op.getRegion().front();
    Block &blk = sde::ensureBlock(cuRegion.getBody());
    blk.getOperations().splice(blk.end(), old.getOperations());
    // Emit implicit barrier unless nowait.
    if (!op.getNowait()) {
      rewriter.setInsertionPointAfter(cuRegion);
      sde::SdeSuBarrierOp::create(rewriter, loc, ValueRange{},
                                  /*barrierEliminated=*/nullptr,
                                  /*barrierReason=*/nullptr);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// omp.wsloop + omp.loop_nest -> sde.su_iterate
struct WsloopToSdePattern : public OpRewritePattern<omp::WsloopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::WsloopOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Converting omp.wsloop to sde.su_iterate");
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto loopNest = cast<omp::LoopNestOp>(op.getWrappedLoop());
    auto lbs = ensureIndexRange(rewriter, loc, loopNest.getLoopLowerBounds());
    auto ubs = ensureIndexRange(rewriter, loc, loopNest.getLoopUpperBounds());
    auto steps = ensureIndexRange(rewriter, loc, loopNest.getLoopSteps());

    // Schedule
    sde::SdeScheduleKindAttr schedAttr;
    if (auto sched = op.getScheduleKind()) {
      if (auto kind = convertScheduleKind(*sched))
        schedAttr = sde::SdeScheduleKindAttr::get(ctx, *kind);
    }

    // Chunk size
    Value chunkSize;
    if (auto chunk = op.getScheduleChunk())
      chunkSize = ensureIndex(rewriter, loc, chunk);

    // Nowait
    bool nw = op.getNowait();

    // Reduction metadata
    SmallVector<Value> redAccs;
    SmallVector<Attribute> reductionKinds;
    if (auto reds = op.getReductionSyms()) {
      auto reductionVars = op.getReductionVars();
      ModuleOp module = op->getParentOfType<ModuleOp>();
      for (auto [attr, value] : llvm::zip(*reds, reductionVars)) {
        redAccs.push_back(value);
        if (auto symRef = dyn_cast<SymbolRefAttr>(attr)) {
          auto decl = dyn_cast_or_null<omp::DeclareReductionOp>(
              module.lookupSymbol(symRef.getLeafReference()));
          if (decl) {
            auto kind = inferReductionKind(decl);
            reductionKinds.push_back(sde::SdeReductionKindAttr::get(ctx, kind));
          }
        }
      }
    }

    auto suIter = sde::SdeSuIterateOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{}, ValueRange{lbs},
        ValueRange{ubs}, ValueRange{steps}, schedAttr, chunkSize,
        nowaitAttr(ctx, nw), ValueRange{redAccs},
        reductionKinds.empty() ? nullptr
                               : rewriter.getArrayAttr(reductionKinds),
        /*reductionStrategy=*/nullptr, /*structuredClassification=*/nullptr,
        /*pattern=*/nullptr,
        /*accessMinOffsets=*/nullptr, /*accessMaxOffsets=*/nullptr,
        /*ownerDims=*/nullptr, /*spatialDims=*/nullptr,
        /*writeFootprint=*/nullptr, /*physicalOwnerDims=*/nullptr,
        /*physicalBlockShape=*/nullptr, /*logicalWorkerSlice=*/nullptr,
        /*physicalHaloShape=*/nullptr, /*iterationTopology=*/nullptr,
        /*repetitionStructure=*/nullptr, /*asyncStrategy=*/nullptr,
        /*cps_group_id=*/nullptr, /*cps_stage_index=*/nullptr,
        /*cps_stage_count=*/nullptr,
        /*distributionKind=*/nullptr, /*inPlaceSafe=*/nullptr,
        /*inPlaceSharedState=*/nullptr, /*vectorizeWidth=*/nullptr,
        /*unrollFactor=*/nullptr, /*interleaveCount=*/nullptr);

    // Create body with one block argument per dimension.
    Region &dstRegion = suIter.getBody();
    if (dstRegion.empty())
      dstRegion.push_back(new Block());
    Block &dst = dstRegion.front();
    unsigned numDims = lbs.size();
    for (unsigned d = dst.getNumArguments(); d < numDims; ++d)
      dst.addArgument(rewriter.getIndexType(), loc);

    // Clone loop body, mapping all induction variables.
    OpBuilder::InsertionGuard IG(rewriter);
    rewriter.setInsertionPointToStart(&dst);
    IRMapping mapper;
    Block &src = loopNest.getRegion().front();
    for (unsigned d = 0; d < std::min<unsigned>(src.getNumArguments(), numDims);
         ++d) {
      Value newArg = dst.getArgument(d);
      Value oldArg = src.getArgument(d);
      // If the source IV was non-index, cast back so cloned body ops match
      if (!oldArg.getType().isIndex())
        newArg =
            arith::IndexCastOp::create(rewriter, loc, oldArg.getType(), newArg);
      mapper.map(oldArg, newArg);
    }
    mapWsloopCapturedArgs(op, mapper);

    // Wrap cloned body ops in cu_region <parallel> so downstream passes
    // see a uniform cu_region→su_iterate→cu_region nesting.
    auto innerCuRegion = sde::SdeCuRegionOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{},
        sde::SdeCuKindAttr::get(ctx, sde::SdeCuKind::parallel),
        /*concurrency_scope=*/nullptr,
        /*nowait=*/nullptr,
        /*iterArgs=*/ValueRange{});
    Block &innerBlk = sde::ensureBlock(innerCuRegion.getBody());
    OpBuilder::InsertionGuard IG2(rewriter);
    rewriter.setInsertionPointToStart(&innerBlk);
    for (Operation &srcOp : src.without_terminator())
      rewriter.clone(srcOp, mapper);
    sde::SdeYieldOp::create(rewriter, loc, ValueRange{});

    // Yield at su_iterate level (outside cu_region).
    rewriter.setInsertionPointAfter(innerCuRegion);
    sde::SdeYieldOp::create(rewriter, loc, ValueRange{});

    // Barrier if not nowait and work follows
    if (!nw && hasWorkAfterInParentBlock(op.getOperation())) {
      rewriter.setInsertionPointAfter(suIter);
      sde::SdeSuBarrierOp::create(rewriter, loc, ValueRange{},
                                  /*barrierEliminated=*/nullptr,
                                  /*barrierReason=*/nullptr);
    }

    ++numWsloopsConverted;
    rewriter.eraseOp(op);
    return success();
  }
};

/// omp.task -> sde.cu_task
struct TaskToSdePattern : public OpRewritePattern<omp::TaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::TaskOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Converting omp.task to sde.cu_task");
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    // Collect task dependencies as sde.mu_dep ops
    SmallVector<Value> deps;
    SmallVector<TaskDependSpec, 4> dependSpecs;
    rewriter.setInsertionPoint(op);
    auto dependList = op.getDependKindsAttr();
    if (dependList && !dependList.empty() &&
        dependList.size() == op.getDependVars().size()) {
      for (unsigned i = 0, e = dependList.size(); i < e; ++i) {
        auto depClause = dyn_cast<omp::ClauseTaskDependAttr>(dependList[i]);
        if (!depClause)
          continue;

        auto depSlice = extractDependSlice(op.getDependVars()[i], rewriter, loc);
        if (!depSlice)
          continue;

        auto sdeMode = convertDependMode(depClause.getValue());
        if (!sdeMode)
          return failure();

        if (auto existingMuDep =
                op.getDependVars()[i].getDefiningOp<sde::SdeMuDepOp>()) {
          if (existingMuDep.getMode() != *sdeMode)
            return op.emitOpError()
                   << "SDE dependency carrier mode disagrees with omp.task "
                      "depend clause";
          deps.push_back(op.getDependVars()[i]);
        } else {
          rewriter.setInsertionPoint(op);
          auto muDep = sde::SdeMuDepOp::create(
              rewriter, loc, sde::DepType::get(ctx),
              sde::SdeAccessModeAttr::get(ctx, *sdeMode), depSlice->source,
              depSlice->offsets, depSlice->sizes);
          deps.push_back(muDep.getDep());
        }
        dependSpecs.push_back(TaskDependSpec{*sdeMode, std::move(*depSlice)});
      }
    }

    sde::SdePatternAttr pattern;
    if (isWavefrontTaskDependPattern(op.getOperation(), dependSpecs))
      pattern =
          sde::SdePatternAttr::get(ctx, sde::SdePattern::wavefront_2d);

    auto cuTask = sde::SdeCuTaskOp::create(rewriter, loc, deps, pattern);
    Block &blk = sde::ensureBlock(cuTask.getBody());

    Block &old = op.getRegion().front();
    blk.getOperations().splice(blk.end(), old.getOperations());

    rewriter.setInsertionPointAfter(cuTask);
    sde::SdeControlTokenOp::create(rewriter, loc, sde::CompletionType::get(ctx));

    ++numTasksConverted;
    rewriter.eraseOp(op);
    return success();
  }
};

/// omp.taskloop -> sde.su_iterate
struct TaskloopToSdePattern : public OpRewritePattern<omp::TaskloopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::TaskloopOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Converting omp.taskloop to sde.su_iterate");
    auto loc = op.getLoc();
    auto loopNest = cast<omp::LoopNestOp>(op.getWrappedLoop());
    Value lb = ensureIndex(rewriter, loc, loopNest.getLoopLowerBounds()[0]);
    Value ub = ensureIndex(rewriter, loc, loopNest.getLoopUpperBounds()[0]);
    Value step = ensureIndex(rewriter, loc, loopNest.getLoopSteps()[0]);

    auto suIter = sde::SdeSuIterateOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{}, ValueRange{lb},
        ValueRange{ub}, ValueRange{step},
        /*schedule=*/nullptr, /*chunkSize=*/Value(),
        /*nowait=*/nullptr,
        /*reductionAccumulators=*/ValueRange{},
        /*reductionKinds=*/nullptr,
        /*reductionStrategy=*/nullptr, /*structuredClassification=*/nullptr,
        /*pattern=*/nullptr,
        /*accessMinOffsets=*/nullptr, /*accessMaxOffsets=*/nullptr,
        /*ownerDims=*/nullptr, /*spatialDims=*/nullptr,
        /*writeFootprint=*/nullptr, /*physicalOwnerDims=*/nullptr,
        /*physicalBlockShape=*/nullptr, /*logicalWorkerSlice=*/nullptr,
        /*physicalHaloShape=*/nullptr, /*iterationTopology=*/nullptr,
        /*repetitionStructure=*/nullptr, /*asyncStrategy=*/nullptr,
        /*cps_group_id=*/nullptr, /*cps_stage_index=*/nullptr,
        /*cps_stage_count=*/nullptr,
        /*distributionKind=*/nullptr, /*inPlaceSafe=*/nullptr,
        /*inPlaceSharedState=*/nullptr, /*vectorizeWidth=*/nullptr,
        /*unrollFactor=*/nullptr, /*interleaveCount=*/nullptr);

    Region &dstRegion = suIter.getBody();
    if (dstRegion.empty())
      dstRegion.push_back(new Block());
    Block &dst = dstRegion.front();
    if (dst.getNumArguments() == 0)
      dst.addArgument(rewriter.getIndexType(), loc);

    OpBuilder::InsertionGuard IG(rewriter);
    rewriter.setInsertionPointToStart(&dst);
    IRMapping mapper;
    Block &src = loopNest.getRegion().front();
    if (!src.getArguments().empty()) {
      Value newArg = dst.getArgument(0);
      Value oldArg = src.getArgument(0);
      if (!oldArg.getType().isIndex())
        newArg =
            arith::IndexCastOp::create(rewriter, loc, oldArg.getType(), newArg);
      mapper.map(oldArg, newArg);
    }
    for (Operation &srcOp : src.without_terminator())
      rewriter.clone(srcOp, mapper);
    sde::SdeYieldOp::create(rewriter, loc, ValueRange{});

    rewriter.eraseOp(op);
    return success();
  }
};

/// scf.parallel -> sde.cu_region parallel + sde.su_iterate
struct SCFParallelToSdePattern : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Converting scf.parallel to sde.cu_region + sde.su_iterate");
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    rewriter.setInsertionPoint(op);

    auto cuRegion = sde::SdeCuRegionOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{},
        sde::SdeCuKindAttr::get(ctx, sde::SdeCuKind::parallel),
        /*concurrency_scope=*/nullptr,
        /*nowait=*/nullptr,
        /*iterArgs=*/ValueRange{});
    Block &parBlk = sde::ensureBlock(cuRegion.getBody());

    rewriter.setInsertionPointToStart(&parBlk);
    Value lb = ensureIndex(rewriter, loc, op.getLowerBound().front());
    Value ub = ensureIndex(rewriter, loc, op.getUpperBound().front());
    Value st = ensureIndex(rewriter, loc, op.getStep().front());

    auto suIter = sde::SdeSuIterateOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{}, ValueRange{lb},
        ValueRange{ub}, ValueRange{st},
        /*schedule=*/nullptr, /*chunkSize=*/Value(),
        /*nowait=*/nullptr,
        /*reductionAccumulators=*/ValueRange{},
        /*reductionKinds=*/nullptr,
        /*reductionStrategy=*/nullptr, /*structuredClassification=*/nullptr,
        /*pattern=*/nullptr,
        /*accessMinOffsets=*/nullptr, /*accessMaxOffsets=*/nullptr,
        /*ownerDims=*/nullptr, /*spatialDims=*/nullptr,
        /*writeFootprint=*/nullptr, /*physicalOwnerDims=*/nullptr,
        /*physicalBlockShape=*/nullptr, /*logicalWorkerSlice=*/nullptr,
        /*physicalHaloShape=*/nullptr, /*iterationTopology=*/nullptr,
        /*repetitionStructure=*/nullptr, /*asyncStrategy=*/nullptr,
        /*cps_group_id=*/nullptr, /*cps_stage_index=*/nullptr,
        /*cps_stage_count=*/nullptr,
        /*distributionKind=*/nullptr, /*inPlaceSafe=*/nullptr,
        /*inPlaceSharedState=*/nullptr, /*vectorizeWidth=*/nullptr,
        /*unrollFactor=*/nullptr, /*interleaveCount=*/nullptr);

    Region &dstRegion = suIter.getBody();
    if (dstRegion.empty())
      dstRegion.push_back(new Block());
    Block &dst = dstRegion.front();
    if (dst.getNumArguments() == 0)
      dst.addArgument(rewriter.getIndexType(), loc);

    OpBuilder::InsertionGuard IG(rewriter);
    rewriter.setInsertionPointToStart(&dst);
    IRMapping mapper;
    Block &src = op.getRegion().front();
    if (!op.getInductionVars().empty())
      mapper.map(op.getInductionVars().front(), dst.getArgument(0));
    if (!src.getArguments().empty())
      mapper.map(src.getArgument(0), dst.getArgument(0));
    for (Operation &srcOp : src.without_terminator())
      rewriter.clone(srcOp, mapper);
    sde::SdeYieldOp::create(rewriter, loc, ValueRange{});

    rewriter.setInsertionPointToEnd(&parBlk);
    sde::SdeYieldOp::create(rewriter, loc, ValueRange{});

    if (hasWorkAfterInParentBlock(op.getOperation())) {
      rewriter.setInsertionPointAfter(cuRegion);
      sde::SdeSuBarrierOp::create(rewriter, loc, ValueRange{},
                                  /*barrierEliminated=*/nullptr,
                                  /*barrierReason=*/nullptr);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// omp.atomic.update -> sde.cu_atomic
struct AtomicUpdateToSdePattern : public OpRewritePattern<omp::AtomicUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::AtomicUpdateOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Converting omp.atomic.update to sde.cu_atomic");
    auto &region = op.getRegion();
    if (!region.hasOneBlock())
      return failure();
    Block &blk = region.front();
    if (blk.getNumArguments() != 1)
      return failure();

    Operation *yield = blk.getTerminator();
    if (!yield)
      return failure();
    Value yielded;
    if (auto y = dyn_cast<omp::YieldOp>(yield)) {
      if (y->getNumOperands() != 1)
        return failure();
      yielded = y->getOperand(0);
    } else {
      return failure();
    }

    Operation *def = yielded.getDefiningOp();
    if (!def)
      return failure();

    // Detect reduction kind from combiner
    sde::SdeReductionKind kind;
    if (isa<arith::AddIOp, arith::AddFOp>(def))
      kind = sde::SdeReductionKind::add;
    else if (isa<arith::MulIOp, arith::MulFOp>(def))
      kind = sde::SdeReductionKind::mul;
    else
      return failure();

    Value addr = op.getX();
    Value blockArg = blk.getArgument(0);
    Value inc;
    if (def->getOperand(0) == blockArg)
      inc = def->getOperand(1);
    else if (def->getOperand(1) == blockArg)
      inc = def->getOperand(0);
    else
      return failure();

    ++numAtomicsConverted;
    rewriter.replaceOpWithNewOp<sde::SdeCuAtomicOp>(
        op, sde::SdeReductionKindAttr::get(rewriter.getContext(), kind), addr,
        inc);
    return success();
  }
};

/// omp.terminator -> sde.yield
struct TerminatorToSdePattern : public OpRewritePattern<omp::TerminatorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::TerminatorOp op,
                                PatternRewriter &rewriter) const override {
    sde::SdeYieldOp::create(rewriter, op.getLoc(), ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

/// omp.barrier -> sde.su_barrier
struct BarrierToSdePattern : public OpRewritePattern<omp::BarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::BarrierOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Converting omp.barrier to sde.su_barrier");
    rewriter.replaceOpWithNewOp<sde::SdeSuBarrierOp>(
        op, ValueRange{}, /*barrierEliminated=*/nullptr,
        /*barrierReason=*/nullptr);
    return success();
  }
};

/// omp.taskwait -> sde.su_barrier
struct TaskwaitToSdePattern : public OpRewritePattern<omp::TaskwaitOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::TaskwaitOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Converting omp.taskwait to sde.su_barrier");
    rewriter.replaceOpWithNewOp<sde::SdeSuBarrierOp>(
        op, ValueRange{}, /*barrierEliminated=*/nullptr,
        /*barrierReason=*/nullptr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct ConvertOpenMPToSdePass
    : public arts::impl::ConvertOpenMPToSdeBase<ConvertOpenMPToSdePass> {

  explicit ConvertOpenMPToSdePass(sde::SDECostModel *costModel = nullptr)
      : costModel(costModel) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ARTS_INFO_HEADER(ConvertOpenMPToSdePass);
    MLIRContext *context = &getContext();
    // Conversion stays structural today, but the pass keeps the SDE-owned
    // model wired through so callers do not depend on downstream analysis
    // plumbing.
    (void)costModel;
    RewritePatternSet patterns(context);
    patterns.add<OMPParallelToSdePattern>(context);
    patterns.add<SCFParallelToSdePattern>(context);
    patterns.add<MasterToSdePattern>(context);
    patterns.add<SingleToSdePattern>(context);
    patterns.add<TaskloopToSdePattern>(context);
    patterns.add<WsloopToSdePattern>(context);
    patterns.add<AtomicUpdateToSdePattern>(context);
    patterns.add<TerminatorToSdePattern>(context);
    patterns.add<BarrierToSdePattern>(context);
    patterns.add<TaskwaitToSdePattern>(context);
    patterns.add<TaskToSdePattern>(context);
    GreedyRewriteConfig config;
    config.enableFolding(false);
    (void)applyPatternsGreedily(module, std::move(patterns), config);

    // Erase omp.declare_reduction symbols — they were consumed by
    // WsloopToSdePattern to infer SDE reduction kinds and are no longer needed.
    SmallVector<omp::DeclareReductionOp> declReductions;
    module.walk(
        [&](omp::DeclareReductionOp op) { declReductions.push_back(op); });
    for (auto op : declReductions)
      op.erase();

    attachTaskwaitControlTokens(module);

    ARTS_INFO_FOOTER(ConvertOpenMPToSdePass);
  }

private:
  // Keep the pass wired to the SDE-owned planning model without reintroducing
  // AnalysisManager or other downstream-layer plumbing here.
  sde::SDECostModel *costModel = nullptr;
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//

namespace mlir {
namespace carts {
namespace sde {
std::unique_ptr<Pass>
createConvertOpenMPToSdePass(SDECostModel *costModel) {
  return std::make_unique<ConvertOpenMPToSdePass>(costModel);
}
} // namespace sde
} // namespace carts
} // namespace mlir
