///==========================================================================///
/// File: ConvertToCodelet.cpp
///
/// Converts cu_region ops that carry tensor iter_args into the canonical SDE
/// codelet form (mu_data + mu_token + cu_codelet) consumed by boundary
/// materialization.
///
/// Scope (step 5): cu_region <single> with tensor iter_args only.
///
/// Algorithm:
///   1. Walk the module for cu_region ops that have tensor-typed iter_args.
///   2. For each tensor iter_arg:
///      a) Create sde.mu_data before the cu_region.
///      b) Inside the cu_region body, create sde.mu_token <readwrite>.
///      c) Create sde.cu_codelet wrapping the body ops.
///      d) Wire block args from codelet to replace tensor uses.
///      e) Yield from codelet; thread results via cu_region yield.
///   3. Replace cu_region tensor results with codelet results.
///   4. Non-tensor iter_args (if any) pass through unchanged.
///==========================================================================///

#include "carts/dialect/sde/Transforms/Passes.h"
namespace mlir::carts::arts {
#define GEN_PASS_DEF_CONVERTTOCODELET
#include "carts/dialect/sde/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts

#include "carts/dialect/sde/Transforms/CodeletUtils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::carts::arts;
using namespace mlir::carts;

namespace {

/// Rewrite a single cu_region <single> with tensor iter_args into codelet form.
static LogicalResult convertCuRegion(sde::SdeCuRegionOp region) {
  // Only handle <single> for now.
  if (region.getKind() != sde::SdeCuKind::single)
    return success();

  MLIRContext *ctx = region.getContext();
  Location loc = region.getLoc();
  Block &body = region.getBody().front();

  // Collect tensor iter_args indices.
  SmallVector<unsigned> tensorArgIndices;
  for (auto [idx, arg] : llvm::enumerate(region.getIterArgs())) {
    if (isa<RankedTensorType>(arg.getType()))
      tensorArgIndices.push_back(idx);
  }

  if (tensorArgIndices.empty())
    return success();

  // --- Step 1: Create mu_data ops before the cu_region for each tensor
  // iter_arg.
  OpBuilder outerBuilder(region);
  SmallVector<sde::SdeMuDataOp> muDataOps;
  muDataOps.reserve(tensorArgIndices.size());
  for (unsigned idx : tensorArgIndices) {
    Value iterArg = region.getIterArgs()[idx];
    auto tensorTy = cast<RankedTensorType>(iterArg.getType());
    auto muData =
        sde::SdeMuDataOp::create(outerBuilder, loc, tensorTy, /*init=*/nullptr,
                                 /*shared=*/UnitAttr::get(ctx));
    muDataOps.push_back(muData);
  }

  // --- Step 2: Inside the body, create mu_token + cu_codelet.
  //
  // We need to:
  //   a) Create mu_token ops at the start of the body for each tensor arg
  //   b) Create a single cu_codelet that wraps all body ops
  //   c) Move body ops into the codelet
  //   d) Wire block args

  // Classify access modes for each tensor block arg.
  SmallVector<sde::SdeAccessMode> accessModes;
  for (unsigned idx : tensorArgIndices) {
    BlockArgument blockArg = body.getArgument(idx);
    accessModes.push_back(sde::classifyTensorAccess(blockArg));
  }

  // Create mu_token ops at the start of the body.
  OpBuilder bodyBuilder(ctx);
  bodyBuilder.setInsertionPointToStart(&body);

  SmallVector<Value> tokens;
  SmallVector<Type> tokenTypes;
  SmallVector<Type> codeletBlockArgTypes;
  SmallVector<unsigned> writableTokenIndices;
  SmallVector<Type> codeletResultTypes;

  for (auto [i, idx] : llvm::enumerate(tensorArgIndices)) {
    auto tensorTy = cast<RankedTensorType>(region.getIterArgs()[idx].getType());
    auto tokenType = sde::TokenType::get(ctx, tensorTy);
    Value src = muDataOps[i].getHandle();

    auto tokOp = sde::SdeMuTokenOp::create(
        bodyBuilder, loc, tokenType,
        sde::SdeAccessModeAttr::get(ctx, accessModes[i]), src,
        /*offsets=*/ValueRange{}, /*sizes=*/ValueRange{});
    tokens.push_back(tokOp.getToken());
    tokenTypes.push_back(tokenType);
    codeletBlockArgTypes.push_back(tensorTy);

    bool writable = (accessModes[i] == sde::SdeAccessMode::write ||
                     accessModes[i] == sde::SdeAccessMode::readwrite);
    if (writable) {
      writableTokenIndices.push_back(i);
      codeletResultTypes.push_back(tensorTy);
    }
  }

  // --- Step 3: Move body ops into the codelet.
  //
  // We need to move all ops between the mu_token ops and the yield into the
  // codelet body. But cu_codelet is IsolatedFromAbove, so we need to handle
  // external references.
  //
  // For the <single> case, the body typically contains:
  //   - tensor.extract on the block arg
  //   - arithmetic ops
  //   - tensor.insert producing the updated tensor
  //   - sde.yield
  //
  // External references (constants, func args) need to be cloned into the
  // codelet body.

  // Collect the ops to move (everything after the mu_token ops, before yield).
  auto *yieldOp = body.getTerminator();
  SmallVector<Operation *> opsToMove;
  for (Operation &op : body.getOperations()) {
    // Skip mu_token ops we just created.
    if (isa<sde::SdeMuTokenOp>(op))
      continue;
    // Skip the yield — we'll handle it separately.
    if (&op == yieldOp)
      continue;
    opsToMove.push_back(&op);
  }

  DenseSet<Value> preMappedValues;
  for (unsigned idx : tensorArgIndices)
    preMappedValues.insert(body.getArgument(idx));

  DenseSet<Operation *> movedOps = sde::collectMovedOpTree(opsToMove);

  DenseSet<Value> materializableMemrefs;
  DenseMap<Type, unsigned> materializableMemrefTypeCounts;
  for (unsigned idx : tensorArgIndices) {
    Value memref = sde::traceTensorBoundaryMemref(region.getIterArgs()[idx]);
    if (!memref || sde::hasMemrefWriteInMovedOps(memref, opsToMove))
      continue;
    materializableMemrefs.insert(memref);

    auto tensorTy = cast<RankedTensorType>(region.getIterArgs()[idx].getType());
    auto memrefTy =
        MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
    ++materializableMemrefTypeCounts[memrefTy];
  }
  DenseSet<Type> materializableMemrefTypes;
  for (auto &entry : materializableMemrefTypeCounts) {
    if (entry.second == 1)
      materializableMemrefTypes.insert(entry.first);
  }

  sde::ExternalCapturePlan capturePlan;
  if (failed(sde::planExternalCaptures(
          opsToMove, preMappedValues, materializableMemrefs,
          materializableMemrefTypes, movedOps, capturePlan)))
    return failure();

  for (Value capture : capturePlan.captures)
    codeletBlockArgTypes.push_back(capture.getType());

  // Create the codelet op after scalar capture discovery so the entry block
  // has all token and explicit capture block arguments up front.
  auto codelet = sde::SdeCuCodeletOp::create(
      bodyBuilder, loc, codeletResultTypes, tokens, capturePlan.captures);
  Block *codeletBlock = new Block();
  SmallVector<Location> argLocs(codeletBlockArgTypes.size(), loc);
  codeletBlock->addArguments(codeletBlockArgTypes, argLocs);
  codelet.getBody().push_back(codeletBlock);
  OpBuilder codeletTerminatorBuilder(ctx);
  codeletTerminatorBuilder.setInsertionPointToEnd(codeletBlock);
  auto placeholderYield =
      sde::SdeYieldOp::create(codeletTerminatorBuilder, loc, ValueRange{});

  // Build an IRMapping from old block args to codelet block args.
  IRMapping mapping;
  DenseMap<Value, Value> materializableMemrefToTensorArg;
  DenseMap<Type, Value> uniqueMaterializableMemrefTypeToTensorArg;
  DenseSet<Type> ambiguousMaterializableMemrefTypes;
  for (auto [i, idx] : llvm::enumerate(tensorArgIndices)) {
    BlockArgument oldArg = body.getArgument(idx);
    BlockArgument newArg = codeletBlock->getArgument(i);
    mapping.map(oldArg, newArg);

    Value memref = sde::traceTensorBoundaryMemref(region.getIterArgs()[idx]);
    if (!memref || !materializableMemrefs.contains(memref))
      continue;
    materializableMemrefToTensorArg[memref] = newArg;
    auto tensorTy = cast<RankedTensorType>(region.getIterArgs()[idx].getType());
    auto memrefTy =
        MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
    if (uniqueMaterializableMemrefTypeToTensorArg.contains(memrefTy)) {
      uniqueMaterializableMemrefTypeToTensorArg.erase(memrefTy);
      ambiguousMaterializableMemrefTypes.insert(memrefTy);
    } else if (!ambiguousMaterializableMemrefTypes.contains(memrefTy)) {
      uniqueMaterializableMemrefTypeToTensorArg[memrefTy] = newArg;
    }
  }
  for (auto [i, capture] : llvm::enumerate(capturePlan.captures)) {
    mapping.map(capture,
                codeletBlock->getArgument(tensorArgIndices.size() + i));
  }

  // Clone ops into the codelet body, handling external references by cloning
  // pure producers, direct captures, and
  // materializing boundary memrefs from token block arguments.
  OpBuilder codeletBuilder(ctx);
  codeletBuilder.setInsertionPoint(placeholderYield);
  DenseMap<Value, Value> materializedMemrefs;

  auto materializeMemrefFromToken = [&](Value memref,
                                        Location userLoc) -> Value {
    auto tensorIt = materializableMemrefToTensorArg.find(memref);
    auto memrefTy = dyn_cast<MemRefType>(memref.getType());
    if (!memrefTy || !memrefTy.hasStaticShape())
      return Value();

    Value tensorArg;
    if (tensorIt != materializableMemrefToTensorArg.end()) {
      tensorArg = tensorIt->second;
    } else if (!ambiguousMaterializableMemrefTypes.contains(memrefTy)) {
      auto typeIt = uniqueMaterializableMemrefTypeToTensorArg.find(memrefTy);
      if (typeIt != uniqueMaterializableMemrefTypeToTensorArg.end())
        tensorArg = typeIt->second;
    }
    if (!tensorArg)
      return Value();

    auto existing = materializedMemrefs.find(memref);
    if (existing != materializedMemrefs.end())
      return existing->second;

    Value local = memref::AllocaOp::create(codeletBuilder, userLoc, memrefTy);
    sde::SdeMuTensorToMemrefOp::create(codeletBuilder, userLoc, tensorArg,
                                       local);
    materializedMemrefs[memref] = local;
    mapping.map(memref, local);
    return local;
  };

  std::function<LogicalResult(Value, Location)> clonePureValue =
      [&](Value value, Location userLoc) -> LogicalResult {
    if (mapping.contains(value) ||
        sde::isValueInternalToMovedOps(value, movedOps))
      return success();
    if (materializeMemrefFromToken(value, userLoc))
      return success();

    Operation *defOp = value.getDefiningOp();
    if (!defOp || !isMemoryEffectFree(defOp))
      return failure();

    for (Value operand : defOp->getOperands()) {
      if (failed(clonePureValue(operand, defOp->getLoc())))
        return failure();
    }

    if (defOp->getNumResults() > 0 && mapping.contains(defOp->getResult(0)))
      return success();
    Operation *cloned = codeletBuilder.clone(*defOp, mapping);
    for (auto [orig, repl] :
         llvm::zip(defOp->getResults(), cloned->getResults()))
      mapping.map(orig, repl);
    return success();
  };

  for (Value value : capturePlan.pureExternal) {
    if (failed(clonePureValue(value, loc)))
      return failure();
  }
  for (Value memref : capturePlan.materializedMemrefs) {
    if (!materializeMemrefFromToken(memref, loc))
      return failure();
  }

  for (Operation *op : opsToMove) {
    codeletBuilder.clone(*op, mapping);
  }

  // --- Step 4: Build the codelet yield.
  //
  // The codelet yields one tensor per writable token. We look at what the
  // cu_region's yield was producing and map it through.
  auto regionYield = cast<sde::SdeYieldOp>(yieldOp);
  SmallVector<Value> codeletYieldValues;
  for (unsigned wIdx : writableTokenIndices) {
    unsigned regionArgIdx = tensorArgIndices[wIdx];
    // The yield value at this position in the region yield.
    Value yieldedVal = regionYield.getValues()[regionArgIdx];
    Value mapped = mapping.lookupOrDefault(yieldedVal);
    codeletYieldValues.push_back(mapped);
  }
  codeletBuilder.setInsertionPoint(placeholderYield);
  sde::SdeYieldOp::create(codeletBuilder, loc, codeletYieldValues);
  placeholderYield.erase();

  // --- Step 5: Erase the original body ops that were cloned into the codelet.
  // Only mu_token, codelet, and yield should remain in the body.
  // Erase the yield first — it references ops we're about to erase.
  yieldOp->erase();
  for (Operation *op : llvm::reverse(opsToMove))
    op->erase();
  // Create a placeholder yield so the body remains valid until we rebuild.
  OpBuilder placeholderBuilder(ctx);
  placeholderBuilder.setInsertionPointToEnd(&body);
  sde::SdeYieldOp::create(placeholderBuilder, loc, ValueRange{});

  // --- Step 6: Rebuild the cu_region without tensor iter_args.
  //
  // We need to remove the tensor iter_args from the cu_region since they are
  // now handled by mu_data/mu_token/cu_codelet. We rebuild the cu_region
  // without iter_args (for the <single> case, all iter_args are tensor).

  // Check if ALL iter_args are tensor-typed. If so, we can remove them all.
  bool allTensor = (tensorArgIndices.size() == region.getIterArgs().size());

  if (allTensor) {
    // Collect materialize users of cu_region results — these will be erased
    // since the codelet lowering handles writeback.
    SmallVector<Operation *> materializeOps;
    for (unsigned idx : tensorArgIndices) {
      for (Operation *user : region.getResult(idx).getUsers()) {
        if (isa<bufferization::MaterializeInDestinationOp,
                sde::SdeMuTensorToMemrefOp>(user))
          materializeOps.push_back(user);
      }
    }

    // Collect to_tensor ops that feed iter_args — these will be erased
    // since mu_data replaces them.
    SmallVector<Operation *> toTensorOps;
    for (unsigned idx : tensorArgIndices) {
      Value iterArg = region.getIterArgs()[idx];
      if (auto toTensor = iterArg.getDefiningOp<bufferization::ToTensorOp>()) {
        toTensorOps.push_back(toTensor);
      } else if (auto muToTensor =
                     iterArg.getDefiningOp<sde::SdeMuMemrefToTensorOp>()) {
        toTensorOps.push_back(muToTensor);
      }
    }

    // Replace all uses of cu_region results. For materialize ops we'll erase
    // them, so first drop their operands.
    for (Operation *op : materializeOps)
      op->erase();

    // Replace cu_region results with mu_data handles (not codelet results,
    // which are inside the cu_region body and don't dominate external users).
    // The mu_data handle is a tensor value that represents the shared data
    // written by the codelet.
    for (auto [i, idx] : llvm::enumerate(tensorArgIndices)) {
      bool writable = false;
      for (unsigned wIdx : writableTokenIndices) {
        if (wIdx == i) {
          writable = true;
          break;
        }
      }
      if (!region.getResult(idx).use_empty()) {
        Value replacement =
            writable ? muDataOps[i].getHandle() : region.getIterArgs()[idx];
        region.getResult(idx).replaceAllUsesWith(replacement);
      }
    }

    // Erase to_tensor ops that fed the iter_args.
    for (Operation *op : toTensorOps) {
      if (op->use_empty())
        op->erase();
    }

    // Erase the block arguments (in reverse order to maintain indices).
    for (int i = static_cast<int>(body.getNumArguments()) - 1; i >= 0; --i)
      body.eraseArgument(i);

    // Rebuild the cu_region without iter_args.
    OpBuilder rebuilder(region);
    auto newRegion = sde::SdeCuRegionOp::create(
        rebuilder, loc, /*resultTypes=*/TypeRange{}, region.getKindAttr(),
        region.getNowaitAttr() ? UnitAttr::get(ctx) : nullptr,
        /*iterArgs=*/ValueRange{});

    // Move the body from old to new.
    Block &newBody = sde::ensureBlock(newRegion.getBody());
    newBody.getOperations().splice(newBody.end(), body.getOperations());

    // Replace the old yield (which yields tensor values) with an empty yield.
    auto *oldYield = newBody.getTerminator();
    OpBuilder yieldBuilder(oldYield);
    sde::SdeYieldOp::create(yieldBuilder, loc, ValueRange{});
    oldYield->erase();

    // Erase the old cu_region.
    region.erase();
  } else {
    return failure();
  }

  return success();
}

struct ConvertToCodeletPass
    : public arts::impl::ConvertToCodeletBase<ConvertToCodeletPass> {
  using arts::impl::ConvertToCodeletBase<
      ConvertToCodeletPass>::ConvertToCodeletBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Convert <single> cu_regions with tensor iter_args to codelet form.
    // Phases 0+1 (strip su_iterate + parallel cu_region tensor iter_args)
    // are now handled by LowerToMemref which runs before this pass.
    {
      SmallVector<sde::SdeCuRegionOp> singles;
      module.walk([&](sde::SdeCuRegionOp region) {
        if (region.getKind() == sde::SdeCuKind::single &&
            sde::hasTensorIterArgs(region))
          singles.push_back(region);
      });
      for (sde::SdeCuRegionOp region : singles) {
        if (failed(convertCuRegion(region))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace

namespace mlir::carts::sde {

std::unique_ptr<Pass> createConvertToCodeletPass() {
  return std::make_unique<ConvertToCodeletPass>();
}

} // namespace mlir::carts::sde
