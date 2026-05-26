///==========================================================================///
/// File: LoopVectorizationHints.cpp
///
/// This pass attaches LLVM loop optimization hints to loop backedges in EDT
/// functions. It uses MLIR's native LoopAnnotationAttr which automatically
/// translates to !llvm.loop metadata during LLVM IR emission.
///
/// Vectorization hints are applied to innermost loops only. Outer loops
/// receive only progress metadata to avoid warnings.
///
/// Before (no hints):
///   llvm.br ^loop_header
///
/// After (innermost loop):
///   llvm.br ^loop_header {
///     loop_annotation = #llvm.loop_annotation<
///       vectorize = #llvm.loop_vectorize<disable = false, width = 4>,
///       interleave = #llvm.loop_interleave<count = 4>,
///       mustProgress = true
///     >
///   }
///
/// After (outer loop):
///   llvm.br ^loop_header {
///     loop_annotation = #llvm.loop_annotation<
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
#include "carts/utils/OperationAttributes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
ARTS_DEBUG_SETUP(loop_vectorization_hints);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

static bool hasFloatingPointType(Type type) {
  if (!type)
    return false;
  if (type.isF16() || type.isBF16() || type.isF32() || type.isF64() ||
      type.isF80() || type.isF128())
    return true;
  if (auto vectorType = dyn_cast<VectorType>(type))
    return hasFloatingPointType(vectorType.getElementType());
  return false;
}

static bool operationTouchesFloatingPoint(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (hasFloatingPointType(operand.getType()))
      return true;
  }
  for (Value result : op->getResults()) {
    if (hasFloatingPointType(result.getType()))
      return true;
  }
  return false;
}

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

  auto features =
      module->getAttrOfType<StringAttr>("polygeist.target-features");
  if (!features)
    return bits;

  StringRef featureText = features.getValue();
  if (featureText.contains("+avx512f"))
    return 512;
  if (featureText.contains("+avx2") || featureText.contains("+avx"))
    return 256;
  return bits;
}

static unsigned getTargetMaxVectorWidth(ModuleOp module,
                                        const TypeAnalysisResult &typeInfo) {
  unsigned elementBits = 32;
  if (typeInfo.f64Count >= typeInfo.f32Count &&
      typeInfo.f64Count >=
          typeInfo.totalLoads - typeInfo.f64Count - typeInfo.f32Count)
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

/// Create full vectorization hints for innermost loops.
static LLVM::LoopAnnotationAttr
createInnermostLoopHints(MLIRContext *ctx, unsigned width, unsigned interleave,
                         bool scalable, bool mustProgress,
                         ArrayRef<LLVM::AccessGroupAttr> accessGroups) {
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

  return LLVM::LoopAnnotationAttr::get(
      ctx,
      /*disableNonforced=*/nullptr,
      /*vectorize=*/vecAttr,
      /*interleave=*/interleaveAttr,
      /*unroll=*/nullptr,
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
createOuterLoopHints(MLIRContext *ctx, bool mustProgress,
                     ArrayRef<LLVM::AccessGroupAttr> accessGroups) {
  return LLVM::LoopAnnotationAttr::get(
      ctx,
      /*disableNonforced=*/nullptr,
      /*vectorize=*/nullptr,
      /*interleave=*/nullptr,
      /*unroll=*/nullptr,
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
      unsigned rtInterleaveCount = 0;
      if (auto attr = funcOp->getAttrOfType<IntegerAttr>(
              ::mlir::carts::arts::AttrNames::Operation::Rt::VectorizeWidth))
        rtVecWidth = attr.getInt();
      if (auto attr = funcOp->getAttrOfType<IntegerAttr>(
              ::mlir::carts::arts::AttrNames::Operation::Rt::InterleaveCount))
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
        unsigned width, interleave;
        auto typeInfo = analyzeLoadTypes(loopBlocks);
        if (rtVecWidth) {
          width = rtVecWidth;
          interleave = rtInterleaveCount ? rtInterleaveCount : interleaveCount;
        } else {
          width = vectorWidth ? vectorWidth : typeInfo.vectorWidth;
          interleave = interleaveCount;
        }
        width = clampVectorWidthToTarget(module, width, typeInfo);

        LLVM::LoopAnnotationAttr annotation;
        if (innermost) {
          annotation = createInnermostLoopHints(
              ctx, width, interleave, enableScalable, enableMustProgress,
              parallelAccessGroups);
          innermostCount++;
          ARTS_DEBUG_TYPE("Innermost loop - vectorization hints at "
                          << terminator->getLoc());
        } else {
          annotation = createOuterLoopHints(ctx, enableMustProgress,
                                            parallelAccessGroups);
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
