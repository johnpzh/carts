///==========================================================================///
/// File: LoopVectorizationHints.cpp
///
/// This pass attaches LLVM loop optimization hints to loop backedges in EDT
/// functions. It uses MLIR's native LoopAnnotationAttr which automatically
/// translates to !llvm.loop metadata during LLVM IR emission.
///
/// Vectorization hints are applied to innermost loops only. Outer loops
/// receive unroll hints but no vectorization hints to avoid warnings.
///
/// Before (no hints):
///   llvm.br ^loop_header
///
/// After (innermost loop):
///   llvm.br ^loop_header {
///     loop_annotation = #llvm.loop_annotation<
///       vectorize = #llvm.loop_vectorize<disable = false, width = 4>,
///       interleave = #llvm.loop_interleave<count = 4>,
///       unroll = #llvm.loop_unroll<disable = false, count = 8>,
///       mustProgress = true
///     >
///   }
///
/// After (outer loop):
///   llvm.br ^loop_header {
///     loop_annotation = #llvm.loop_annotation<
///       unroll = #llvm.loop_unroll<disable = false, count = 4>,
///       mustProgress = true
///     >
///   }
///==========================================================================///

#include "carts/dialect/arts-rt/Transforms/Passes.h"

namespace mlir::carts::arts_rt {
#define GEN_PASS_DEF_LOOPVECTORIZATIONHINTS
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts_rt

#include "carts/utils/Debug.h"
#include "carts/utils/LoopUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/StringRef.h"
#include <limits>
#include <optional>
ARTS_DEBUG_SETUP(loop_vectorization_hints);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

/// Check if a conditional branch forms a loop backedge by checking if
/// either target block dominates the block containing the branch.
static bool isLoopBackedge(LLVM::CondBrOp condBr, DominanceInfo &domInfo) {
  Block *currentBlock = condBr->getBlock();
  Block *trueBlock = condBr.getTrueDest();
  Block *falseBlock = condBr.getFalseDest();

  return domInfo.dominates(trueBlock, currentBlock) ||
         domInfo.dominates(falseBlock, currentBlock);
}

/// Collect natural-loop blocks for a backedge latch -> header.
static SmallPtrSet<Block *, 16> getLoopBlocks(Block *headerBlock,
                                              Block *latchBlock) {
  SmallPtrSet<Block *, 16> loopBlocks;
  SmallVector<Block *, 16> worklist;
  loopBlocks.insert(headerBlock);
  if (loopBlocks.insert(latchBlock).second)
    worklist.push_back(latchBlock);

  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();
    for (Block *pred : block->getPredecessors()) {
      if (!loopBlocks.insert(pred).second)
        continue;
      if (pred != headerBlock)
        worklist.push_back(pred);
    }
  }

  return loopBlocks;
}

/// Check if a loop is innermost (contains no nested loops).
static bool isInnermostLoop(Block *headerBlock, Block *latchBlock,
                            DominanceInfo &domInfo) {
  SmallPtrSet<Block *, 16> loopBlocks = getLoopBlocks(headerBlock, latchBlock);

  for (Block *block : loopBlocks) {
    if (block == latchBlock)
      continue;

    if (auto brOp = dyn_cast<LLVM::BrOp>(block->getTerminator())) {
      Block *dest = brOp.getDest();
      if (dest != headerBlock && loopBlocks.contains(dest) &&
          domInfo.dominates(dest, block))
        return false;
    }

    if (auto condBr = dyn_cast<LLVM::CondBrOp>(block->getTerminator())) {
      for (Block *dest : {condBr.getTrueDest(), condBr.getFalseDest()}) {
        if (dest != headerBlock && loopBlocks.contains(dest) &&
            domInfo.dominates(dest, block))
          return false;
      }
    }
  }

  return true;
}

/// Analyze function to determine optimal vector width based on data types.
/// Returns: {vectorWidth, f64Count, f32Count, totalLoads}
struct TypeAnalysisResult {
  unsigned vectorWidth;
  unsigned f64Count;
  unsigned f32Count;
  unsigned totalLoads;
};

static TypeAnalysisResult
analyzeLoadTypes(const SmallPtrSet<Block *, 16> &loopBlocks) {
  unsigned floatCount = 0, doubleCount = 0, i32Count = 0, totalLoads = 0;

  for (Block *block : loopBlocks) {
    for (Operation &opRef : *block) {
      auto loadOp = dyn_cast<LLVM::LoadOp>(&opRef);
      if (!loadOp)
        continue;

      totalLoads++;
      Type elemType = loadOp.getResult().getType();
      if (elemType.isF32())
        floatCount++;
      else if (elemType.isF64())
        doubleCount++;
      else if (elemType.isInteger(32))
        i32Count++;
    }
  }

  // The emitted CARTS modules currently target generic x86-64, whose declared
  // feature set is SSE/SSE2. Use 128-bit lane counts by default; wider SDE
  // hints are clamped again below against the concrete module features.
  unsigned width = 4;
  if (doubleCount >= floatCount && doubleCount >= i32Count)
    width = 2;

  return {width, doubleCount, floatCount, totalLoads};
}

static unsigned getTargetVectorBits(ModuleOp module) {
  unsigned bits = 128;

  auto features = module->getAttrOfType<StringAttr>("polygeist.target-features");
  if (!features)
    return bits;

  StringRef featureText = features.getValue();
  if (featureText.contains("+avx512f"))
    return 512;
  if (featureText.contains("+avx2") || featureText.contains("+avx"))
    return 256;
  return bits;
}

static unsigned
getTargetMaxVectorWidth(ModuleOp module, const TypeAnalysisResult &typeInfo) {
  unsigned elementBits = 32;
  if (typeInfo.f64Count >= typeInfo.f32Count &&
      typeInfo.f64Count >= typeInfo.totalLoads - typeInfo.f64Count -
                                  typeInfo.f32Count)
    elementBits = 64;

  return std::max(1u, getTargetVectorBits(module) / elementBits);
}

static unsigned clampVectorWidthToTarget(ModuleOp module, unsigned requested,
                                         const TypeAnalysisResult &typeInfo) {
  unsigned maxWidth = getTargetMaxVectorWidth(module, typeInfo);
  return std::clamp(requested, 1u, maxWidth);
}

static bool isVectorFriendlyFPIntrinsic(StringRef intrinsic) {
  return intrinsic.starts_with("llvm.fma") ||
         intrinsic.starts_with("llvm.fmuladd") ||
         intrinsic.starts_with("llvm.exp") ||
         intrinsic.starts_with("llvm.exp2") ||
         intrinsic.starts_with("llvm.log") ||
         intrinsic.starts_with("llvm.log2") ||
         intrinsic.starts_with("llvm.log10") ||
         intrinsic.starts_with("llvm.sin") ||
         intrinsic.starts_with("llvm.cos") ||
         intrinsic.starts_with("llvm.tan") ||
         intrinsic.starts_with("llvm.tanh") ||
         intrinsic.starts_with("llvm.pow");
}

/// Returns true when the innermost loop contains scalar FP calls that are
/// unlikely to benefit from forced vectorize/unroll metadata.
static bool
loopHasCallLikeFloatingPointWork(const SmallPtrSet<Block *, 16> &loopBlocks) {
  for (Block *block : loopBlocks) {
    for (Operation &opRef : *block) {
      if (auto call = dyn_cast<LLVM::CallOp>(&opRef)) {
        if (operationTouchesFloatingPoint(call))
          return true;
        continue;
      }

      auto intrinsic = dyn_cast<LLVM::CallIntrinsicOp>(&opRef);
      if (!intrinsic)
        continue;
      if (!operationTouchesFloatingPoint(intrinsic))
        continue;
      if (isVectorFriendlyFPIntrinsic(intrinsic.getIntrin()))
        continue;
      return true;
    }
  }
  return false;
}

/// Returns true if `value` is dataflow-dependent on `root`.
/// `followLoads` controls whether load operands are traversed.
static bool valueDependsOn(Value value, Value root, bool followLoads) {
  SmallVector<Value, 16> worklist;
  SmallPtrSet<Value, 32> visited;
  worklist.push_back(value);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (current == root)
      return true;

    Operation *def = current.getDefiningOp();
    if (!def)
      continue;

    if (auto gep = dyn_cast<LLVM::GEPOp>(def)) {
      worklist.push_back(gep.getBase());
      continue;
    }
    if (auto load = dyn_cast<LLVM::LoadOp>(def)) {
      if (followLoads)
        worklist.push_back(load.getAddr());
      continue;
    }
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
      for (Value in : cast.getInputs())
        worklist.push_back(in);
      continue;
    }
  }

  return false;
}

/// Returns true if `value` depends on a loop-varying quantity in this natural
/// loop. Values defined outside the loop are treated as invariant.
static bool
valueDependsOnLoopVariant(Value value,
                          const SmallPtrSet<Block *, 16> &loopBlocks) {
  SmallVector<Value, 16> worklist;
  SmallPtrSet<Value, 32> visited;
  worklist.push_back(value);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      if (loopBlocks.contains(blockArg.getOwner()))
        return true;
      continue;
    }

    Operation *def = current.getDefiningOp();
    if (!def)
      continue;
    if (!loopBlocks.contains(def->getBlock()))
      continue;

    if (isa<LLVM::ConstantOp>(def))
      continue;

    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }

  return false;
}

/// Returns true if the loop performs loop-varying dep-array traversal. Purely
/// invariant dep-array pointer materialization is allowed.
static bool loopHasVariantDepArrayAccess(Block *headerBlock, Block *latchBlock,
                                         Value depv) {
  if (!depv)
    return false;

  SmallPtrSet<Block *, 16> loopBlocks = getLoopBlocks(headerBlock, latchBlock);
  for (Block *block : loopBlocks) {
    for (Operation &opRef : *block) {
      Operation *op = &opRef;
      if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
        if (valueDependsOn(load.getAddr(), depv, /*followLoads=*/false) &&
            valueDependsOnLoopVariant(load.getAddr(), loopBlocks))
          return true;
        continue;
      }
      if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
        if (valueDependsOn(store.getAddr(), depv, /*followLoads=*/false) &&
            valueDependsOnLoopVariant(store.getAddr(), loopBlocks))
          return true;
        continue;
      }
    }
  }
  return false;
}

static std::optional<int64_t> getConstantIntValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto constant = value.getDefiningOp<LLVM::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constant.getValue()))
      return intAttr.getValue().getSExtValue();
  }
  return std::nullopt;
}

static Value getSuccessorOperandForBlock(Operation *terminator, Block *dest,
                                         unsigned argIndex) {
  if (auto brOp = dyn_cast<LLVM::BrOp>(terminator)) {
    if (brOp.getDest() != dest || argIndex >= brOp.getDestOperands().size())
      return Value();
    return brOp.getDestOperands()[argIndex];
  }

  if (auto condBr = dyn_cast<LLVM::CondBrOp>(terminator)) {
    if (condBr.getTrueDest() == dest &&
        argIndex < condBr.getTrueDestOperands().size())
      return condBr.getTrueDestOperands()[argIndex];
    if (condBr.getFalseDest() == dest &&
        argIndex < condBr.getFalseDestOperands().size())
      return condBr.getFalseDestOperands()[argIndex];
  }

  return Value();
}

static bool matchInductionStep(Value update, BlockArgument iv, int64_t &step) {
  if (!update)
    return false;

  if (auto addOp = update.getDefiningOp<LLVM::AddOp>()) {
    if (addOp.getLhs() == iv) {
      if (auto rhs = getConstantIntValue(addOp.getRhs())) {
        step = *rhs;
        return step != 0;
      }
    }
    if (addOp.getRhs() == iv) {
      if (auto lhs = getConstantIntValue(addOp.getLhs())) {
        step = *lhs;
        return step != 0;
      }
    }
  }

  if (auto subOp = update.getDefiningOp<LLVM::SubOp>()) {
    if (subOp.getLhs() != iv)
      return false;
    if (auto rhs = getConstantIntValue(subOp.getRhs())) {
      step = -*rhs;
      return step != 0;
    }
  }

  return false;
}

enum class LoopBoundKind {
  increasingExclusive,
  increasingInclusive,
  decreasingExclusive,
  decreasingInclusive
};

struct ConstantLoopCondition {
  BlockArgument inductionVar;
  int64_t bound;
  LoopBoundKind kind;
};

static LLVM::ICmpPredicate invertPredicate(LLVM::ICmpPredicate predicate) {
  switch (predicate) {
  case LLVM::ICmpPredicate::eq:
    return LLVM::ICmpPredicate::ne;
  case LLVM::ICmpPredicate::ne:
    return LLVM::ICmpPredicate::eq;
  case LLVM::ICmpPredicate::slt:
    return LLVM::ICmpPredicate::sge;
  case LLVM::ICmpPredicate::sle:
    return LLVM::ICmpPredicate::sgt;
  case LLVM::ICmpPredicate::sgt:
    return LLVM::ICmpPredicate::sle;
  case LLVM::ICmpPredicate::sge:
    return LLVM::ICmpPredicate::slt;
  case LLVM::ICmpPredicate::ult:
    return LLVM::ICmpPredicate::uge;
  case LLVM::ICmpPredicate::ule:
    return LLVM::ICmpPredicate::ugt;
  case LLVM::ICmpPredicate::ugt:
    return LLVM::ICmpPredicate::ule;
  case LLVM::ICmpPredicate::uge:
    return LLVM::ICmpPredicate::ult;
  }
  llvm_unreachable("unexpected icmp predicate");
}

static std::optional<ConstantLoopCondition>
matchConstantLoopCondition(LLVM::CondBrOp condBr,
                           const SmallPtrSet<Block *, 16> &loopBlocks) {
  Block *trueDest = condBr.getTrueDest();
  Block *falseDest = condBr.getFalseDest();
  bool trueInLoop =
      loopBlocks.contains(trueDest) && trueDest != condBr->getBlock();
  bool falseInLoop =
      loopBlocks.contains(falseDest) && falseDest != condBr->getBlock();
  if (trueInLoop == falseInLoop)
    return std::nullopt;

  auto icmp = condBr.getCondition().getDefiningOp<LLVM::ICmpOp>();
  if (!icmp)
    return std::nullopt;

  LLVM::ICmpPredicate predicate = icmp.getPredicate();
  if (!trueInLoop)
    predicate = invertPredicate(predicate);

  Value lhs = icmp.getLhs();
  Value rhs = icmp.getRhs();

  auto makeCondition = [&](BlockArgument iv, int64_t bound,
                           LoopBoundKind kind) -> ConstantLoopCondition {
    return ConstantLoopCondition{iv, bound, kind};
  };

  if (auto iv = dyn_cast<BlockArgument>(lhs)) {
    if (auto bound = getConstantIntValue(rhs)) {
      switch (predicate) {
      case LLVM::ICmpPredicate::slt:
      case LLVM::ICmpPredicate::ult:
        return makeCondition(iv, *bound, LoopBoundKind::increasingExclusive);
      case LLVM::ICmpPredicate::sle:
      case LLVM::ICmpPredicate::ule:
        return makeCondition(iv, *bound, LoopBoundKind::increasingInclusive);
      case LLVM::ICmpPredicate::sgt:
      case LLVM::ICmpPredicate::ugt:
        return makeCondition(iv, *bound, LoopBoundKind::decreasingExclusive);
      case LLVM::ICmpPredicate::sge:
      case LLVM::ICmpPredicate::uge:
        return makeCondition(iv, *bound, LoopBoundKind::decreasingInclusive);
      default:
        break;
      }
    }
  }

  if (auto iv = dyn_cast<BlockArgument>(rhs)) {
    if (auto bound = getConstantIntValue(lhs)) {
      switch (predicate) {
      case LLVM::ICmpPredicate::sgt:
      case LLVM::ICmpPredicate::ugt:
        return makeCondition(iv, *bound, LoopBoundKind::increasingExclusive);
      case LLVM::ICmpPredicate::sge:
      case LLVM::ICmpPredicate::uge:
        return makeCondition(iv, *bound, LoopBoundKind::increasingInclusive);
      case LLVM::ICmpPredicate::slt:
      case LLVM::ICmpPredicate::ult:
        return makeCondition(iv, *bound, LoopBoundKind::decreasingExclusive);
      case LLVM::ICmpPredicate::sle:
      case LLVM::ICmpPredicate::ule:
        return makeCondition(iv, *bound, LoopBoundKind::decreasingInclusive);
      default:
        break;
      }
    }
  }

  return std::nullopt;
}

static std::optional<unsigned> computeConstantTripCount(int64_t start,
                                                        int64_t step,
                                                        int64_t bound,
                                                        LoopBoundKind kind) {
  int64_t distance = 0;

  switch (kind) {
  case LoopBoundKind::increasingExclusive:
    if (step <= 0)
      return std::nullopt;
    distance = bound - start;
    break;
  case LoopBoundKind::increasingInclusive:
    if (step <= 0)
      return std::nullopt;
    distance = bound - start + 1;
    break;
  case LoopBoundKind::decreasingExclusive:
    if (step >= 0)
      return std::nullopt;
    distance = start - bound;
    step = -step;
    break;
  case LoopBoundKind::decreasingInclusive:
    if (step >= 0)
      return std::nullopt;
    distance = start - bound + 1;
    step = -step;
    break;
  }

  if (distance <= 0)
    return 0u;

  uint64_t tripCount =
      (static_cast<uint64_t>(distance) + static_cast<uint64_t>(step) - 1) /
      static_cast<uint64_t>(step);
  if (tripCount > std::numeric_limits<unsigned>::max())
    return std::nullopt;
  return static_cast<unsigned>(tripCount);
}

static std::optional<unsigned> inferConstantTripCount(Block *headerBlock,
                                                      Block *latchBlock) {
  auto condBr = dyn_cast<LLVM::CondBrOp>(headerBlock->getTerminator());
  if (!condBr)
    return std::nullopt;

  SmallPtrSet<Block *, 16> loopBlocks = getLoopBlocks(headerBlock, latchBlock);
  auto condition = matchConstantLoopCondition(condBr, loopBlocks);
  if (!condition)
    return std::nullopt;

  BlockArgument iv = condition->inductionVar;
  if (iv.getOwner() != headerBlock)
    return std::nullopt;

  Value startValue;
  int outsidePredCount = 0;
  for (Block *pred : headerBlock->getPredecessors()) {
    if (loopBlocks.contains(pred))
      continue;
    Value incoming = getSuccessorOperandForBlock(
        pred->getTerminator(), headerBlock, iv.getArgNumber());
    if (!incoming)
      return std::nullopt;
    startValue = incoming;
    outsidePredCount++;
  }
  if (outsidePredCount != 1)
    return std::nullopt;

  auto start = getConstantIntValue(startValue);
  if (!start)
    return std::nullopt;

  Value update = getSuccessorOperandForBlock(latchBlock->getTerminator(),
                                             headerBlock, iv.getArgNumber());
  if (!update)
    return std::nullopt;

  int64_t step = 0;
  if (!matchInductionStep(update, iv, step))
    return std::nullopt;

  return computeConstantTripCount(*start, step, condition->bound,
                                  condition->kind);
}

/// Determine optimal unroll count based on loop characteristics.
static unsigned determineUnrollCount(const TypeAnalysisResult &typeInfo) {
  unsigned cacheLineElements =
      (typeInfo.f64Count >= typeInfo.f32Count) ? 8 : 16;
  unsigned vectorWidth = typeInfo.vectorWidth;
  unsigned unrollCount = cacheLineElements / vectorWidth;
  if (unrollCount == 0)
    unrollCount = 1;
  unrollCount *= vectorWidth;

  if (typeInfo.totalLoads > 16)
    unrollCount = std::min(unrollCount, 2u);
  else if (typeInfo.totalLoads > 8)
    unrollCount = std::min(unrollCount, 4u);

  return std::clamp(unrollCount, 2u, 8u);
}

/// Create full vectorization hints for innermost loops.
static LLVM::LoopAnnotationAttr
createInnermostLoopHints(MLIRContext *ctx, unsigned width, unsigned interleave,
                         bool scalable, bool mustProgress, unsigned unrollCount,
                         ArrayRef<LLVM::AccessGroupAttr> accessGroups,
                         bool fullUnroll) {
  auto vecAttr = LLVM::LoopVectorizeAttr::get(
      ctx,
      /*disable=*/BoolAttr::get(ctx, false),
      /*predicateEnable=*/nullptr,
      /*scalableEnable=*/scalable ? BoolAttr::get(ctx, true) : nullptr,
      /*width=*/IntegerAttr::get(IntegerType::get(ctx, 32), width),
      /*followupVectorized=*/nullptr,
      /*followupEpilogue=*/nullptr,
      /*followupAll=*/nullptr);

  auto interleaveAttr = LLVM::LoopInterleaveAttr::get(
      ctx, IntegerAttr::get(IntegerType::get(ctx, 32), interleave));

  auto unrollAttr = LLVM::LoopUnrollAttr::get(
      ctx,
      /*disable=*/BoolAttr::get(ctx, false),
      /*count=*/
      fullUnroll ? IntegerAttr()
                 : IntegerAttr::get(IntegerType::get(ctx, 32), unrollCount),
      /*runtimeDisable=*/nullptr,
      /*full=*/fullUnroll ? BoolAttr::get(ctx, true) : nullptr,
      /*followupUnrolled=*/nullptr,
      /*followupRemainder=*/nullptr,
      /*followupAll=*/nullptr);

  return LLVM::LoopAnnotationAttr::get(
      ctx,
      /*disableNonforced=*/nullptr,
      /*vectorize=*/vecAttr,
      /*interleave=*/interleaveAttr,
      /*unroll=*/unrollAttr,
      /*unrollAndJam=*/nullptr,
      /*licm=*/nullptr,
      /*distribute=*/nullptr,
      /*pipeline=*/nullptr,
      /*peeled=*/nullptr,
      /*unswitch=*/nullptr,
      /*mustProgress=*/mustProgress ? BoolAttr::get(ctx, true) : nullptr,
      /*isVectorized=*/nullptr,
      /*startLoc=*/FusedLoc(),
      /*endLoc=*/FusedLoc(),
      /*parallelAccesses=*/accessGroups);
}

/// Create light hints for outer loops (no vectorization).
static LLVM::LoopAnnotationAttr
createOuterLoopHints(MLIRContext *ctx, bool mustProgress, unsigned unrollCount,
                     ArrayRef<LLVM::AccessGroupAttr> accessGroups) {
  unsigned outerUnroll = std::min(unrollCount, 4u);

  auto unrollAttr = LLVM::LoopUnrollAttr::get(
      ctx,
      /*disable=*/BoolAttr::get(ctx, false),
      /*count=*/IntegerAttr::get(IntegerType::get(ctx, 32), outerUnroll),
      /*runtimeDisable=*/nullptr,
      /*full=*/nullptr,
      /*followupUnrolled=*/nullptr,
      /*followupRemainder=*/nullptr,
      /*followupAll=*/nullptr);

  return LLVM::LoopAnnotationAttr::get(
      ctx,
      /*disableNonforced=*/nullptr,
      /*vectorize=*/nullptr,
      /*interleave=*/nullptr,
      /*unroll=*/unrollAttr,
      /*unrollAndJam=*/nullptr,
      /*licm=*/nullptr,
      /*distribute=*/nullptr,
      /*pipeline=*/nullptr,
      /*peeled=*/nullptr,
      /*unswitch=*/nullptr,
      /*mustProgress=*/mustProgress ? BoolAttr::get(ctx, true) : nullptr,
      /*isVectorized=*/nullptr,
      /*startLoc=*/FusedLoc(),
      /*endLoc=*/FusedLoc(),
      /*parallelAccesses=*/accessGroups);
}

struct LoopVectorizationHintsPass
    : public arts_rt::impl::LoopVectorizationHintsBase<
          LoopVectorizationHintsPass> {
  unsigned vectorWidth = 0;
  unsigned interleaveCount = 4;
  bool enableScalable = false;
  bool enableMustProgress = true;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    ARTS_INFO_HEADER(LoopVectorizationHintsPass);

    int totalHints = 0;
    int skippedUnsafeLoops = 0;
    int skippedCallHeavyLoops = 0;

    module.walk([&](LLVM::LLVMFuncOp funcOp) {
      if (!funcOp.getName().starts_with("__arts_edt_"))
        return;

      ARTS_DEBUG_TYPE("Processing EDT function: " << funcOp.getName());

      DominanceInfo domInfo(funcOp);
      Value depv =
          (funcOp.getNumArguments() >= 4) ? funcOp.getArgument(3) : Value{};
      SmallVector<LLVM::AccessGroupAttr> parallelAccessGroups;

      // RT reads only RT-facing hints translated from Core EDT attrs.
      unsigned rtVecWidth = 0;
      unsigned rtUnrollFactor = 0;
      unsigned rtInterleaveCount = 0;
      if (auto attr = funcOp->getAttrOfType<IntegerAttr>(
              AttrNames::Operation::Rt::VectorizeWidth))
        rtVecWidth = attr.getInt();
      if (auto attr = funcOp->getAttrOfType<IntegerAttr>(
              AttrNames::Operation::Rt::UnrollFactor))
        rtUnrollFactor = attr.getInt();
      if (auto attr = funcOp->getAttrOfType<IntegerAttr>(
              AttrNames::Operation::Rt::InterleaveCount))
        rtInterleaveCount = attr.getInt();

      int innermostCount = 0;
      int outerCount = 0;

      // Shared backedge processing for both BrOp and CondBrOp.
      auto processBackedge = [&](Operation *terminator, Block *headerBlock,
                                 Block *latchBlock) {
        if (loopHasVariantDepArrayAccess(headerBlock, latchBlock, depv)) {
          skippedUnsafeLoops++;
          ARTS_DEBUG_TYPE("Skipping loop hints in "
                          << funcOp.getName()
                          << " (loop-varying dep-array access)");
          return;
        }

        SmallPtrSet<Block *, 16> loopBlocks =
            getLoopBlocks(headerBlock, latchBlock);
        bool innermost = isInnermostLoop(headerBlock, latchBlock, domInfo);
        if (innermost && loopHasCallLikeFloatingPointWork(loopBlocks)) {
          skippedCallHeavyLoops++;
          ARTS_DEBUG_TYPE("Skipping innermost loop hints in "
                          << funcOp.getName()
                          << " (call-like floating-point work)");
          return;
        }

        // SDE attrs carry the semantic vectorization request. RT still clamps
        // the lane count to the module target so generic x86-64 IR does not
        // force 256-bit vectors on an SSE/SSE2 target.
        unsigned width, unrollCount, interleave;
        auto typeInfo = analyzeLoadTypes(loopBlocks);
        if (rtVecWidth) {
          width = rtVecWidth;
          unrollCount = rtUnrollFactor ? rtUnrollFactor : 2;
          interleave =
              rtInterleaveCount ? rtInterleaveCount : interleaveCount;
        } else {
          width = vectorWidth ? vectorWidth : typeInfo.vectorWidth;
          unrollCount = determineUnrollCount(typeInfo);
          interleave = interleaveCount;
        }
        width = clampVectorWidthToTarget(module, width, typeInfo);

        std::optional<unsigned> constantTripCount =
            inferConstantTripCount(headerBlock, latchBlock);
        bool fullUnroll = innermost && constantTripCount &&
                          *constantTripCount > 0 && *constantTripCount <= 8;
        unsigned loopUnrollCount =
            fullUnroll ? *constantTripCount : unrollCount;

        LLVM::LoopAnnotationAttr annotation;
        if (innermost) {
          annotation = createInnermostLoopHints(
              ctx, width, interleave, enableScalable, enableMustProgress,
              loopUnrollCount, parallelAccessGroups, fullUnroll);
          innermostCount++;
          ARTS_DEBUG_TYPE(
              "Innermost loop - full vectorization hints at "
              << terminator->getLoc()
              << (fullUnroll ? " with loop-local full unroll" : ""));
        } else {
          annotation = createOuterLoopHints(
              ctx, enableMustProgress, loopUnrollCount, parallelAccessGroups);
          outerCount++;
          ARTS_DEBUG_TYPE("Outer loop - light hints at "
                          << terminator->getLoc());
        }

        if (auto brOp = dyn_cast<LLVM::BrOp>(terminator))
          brOp.setLoopAnnotationAttr(annotation);
        else if (auto condBr = dyn_cast<LLVM::CondBrOp>(terminator))
          condBr.setLoopAnnotationAttr(annotation);
      };

      funcOp.walk([&](LLVM::BrOp brOp) {
        Block *currentBlock = brOp->getBlock();
        Block *destBlock = brOp.getDest();
        if (!domInfo.dominates(destBlock, currentBlock))
          return;
        if (brOp.getLoopAnnotationAttr())
          return;
        processBackedge(brOp.getOperation(), destBlock, currentBlock);
      });

      funcOp.walk([&](LLVM::CondBrOp condBr) {
        if (!isLoopBackedge(condBr, domInfo))
          return;
        if (condBr.getLoopAnnotationAttr())
          return;
        Block *currentBlock = condBr->getBlock();
        Block *trueBlock = condBr.getTrueDest();
        Block *falseBlock = condBr.getFalseDest();
        Block *headerBlock =
            domInfo.dominates(trueBlock, currentBlock) ? trueBlock : falseBlock;
        processBackedge(condBr.getOperation(), headerBlock, currentBlock);
      });

      int hintsInFunc = innermostCount + outerCount;
      if (hintsInFunc > 0) {
        ARTS_INFO("Attached hints to "
                  << hintsInFunc << " loops in " << funcOp.getName() << " ("
                  << innermostCount << " innermost with vectorization, "
                  << outerCount << " outer without)");
        totalHints += hintsInFunc;
      }
    });

    ARTS_INFO(
        "Total: attached hints to "
        << totalHints << " loop backedges; skipped " << skippedUnsafeLoops
        << " loop backedges with loop-varying dependency-array access and "
        << skippedCallHeavyLoops
        << " innermost loop backedges with call-like floating-point "
           "work");
    ARTS_INFO_FOOTER(LoopVectorizationHintsPass);
  }
};

} // namespace

///===----------------------------------------------------------------------===///
/// Pass creation and registration
///===----------------------------------------------------------------------===///
namespace mlir {
namespace carts::arts_rt {

std::unique_ptr<Pass> createLoopVectorizationHintsPass() {
  return std::make_unique<LoopVectorizationHintsPass>();
}

} // namespace carts::arts_rt
} // namespace mlir
