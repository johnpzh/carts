///==========================================================================///
/// File: EdtLowering.cpp
/// Complete implementation of EDT lowering pass that transforms arts.edt
/// operations into runtime-compatible function calls.
///
/// This pass implements a 6-step EDT lowering process:
/// 1. Analyze EDT region for free variables and deps
/// 2. Outline EDT region to function with ARTS runtime signature
/// 3. Insert parameter packing before EDT (edt_param_pack) - It should include
///    all parameters from the EDT + unique datablock sizes and indices for all
///    deps
/// 4. Insert parameter/dependency unpacking in outlined function
/// 5. Replace EDT with edt_create call returning GUID
/// 6. Add dependency management (record_in_dep, increment_out_latch)
///
/// Dep contract (before/after):
///   BEFORE:
///     %t = arts.edt ... (%dep0, %dep1)
///
///   AFTER:
///     %t = arts.edt_create ...
///     arts.rec_dep %t(%dep0_guid, %dep1_guid) {acquire_modes = ...}
///
/// Every lowered EDT dependency must come from DbAcquire/DepDbAcquire; this
/// pass errors out otherwise to avoid bypassing runtime dependency semantics.
///==========================================================================///

#include "arts/dialect/rt/Transforms/Passes.h"
namespace mlir::arts {
#define GEN_PASS_DEF_EDTLOWERING
#include "arts/dialect/rt/Transforms/Passes.h.inc"
} // namespace mlir::arts
#include "arts/dialect/core/Analysis/AnalysisManager.h"
#include "arts/dialect/core/Analysis/db/DbAnalysis.h"
#include "arts/dialect/core/Analysis/db/OwnershipProof.h"
#include "arts/dialect/core/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "arts/dialect/core/Conversion/ArtsToRt/EdtLoweringInternal.h"
#include "arts/dialect/rt/IR/RtDialect.h"
#include "arts/passes/Passes.h"
#include "arts/utils/DbUtils.h"
#include "arts/utils/EdtUtils.h"
#include "arts/utils/IdRegistry.h"
#include "arts/utils/LoweringContractUtils.h"
#include "arts/utils/OperationAttributes.h"
#include "arts/utils/PartitionPredicates.h"
#include "arts/utils/Utils.h"
#include "arts/utils/ValueAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "polygeist/Ops.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_lowering);

static constexpr int32_t kArtsRuntimeDbModeRw = 3;

#include "llvm/ADT/Statistic.h"
static llvm::Statistic numEdtsLowered{
    "edt_lowering", "NumEdtsLowered",
    "Number of EDT operations lowered to runtime calls"};
static llvm::Statistic numParamPacksCreated{
    "edt_lowering", "NumParamPacksCreated",
    "Number of parameter packs created for EDT outlining"};
static llvm::Statistic numDepsRecorded{
    "edt_lowering", "NumDepsRecorded",
    "Number of dependencies recorded for lowered EDTs"};
static llvm::Statistic numEdtsDemotedToTask{
    "edt_lowering", "NumEdtsDemotedToTask",
    "Number of non-task EDTs demoted to task type"};

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arts;
using AttrNames::Operation::ContinuationForEpoch;
using AttrNames::Operation::ControlDep;
using namespace mlir::arts::rt;

using namespace mlir::arts::edt_lowering;

namespace {

static std::optional<std::pair<SmallVector<Value, 4>, SmallVector<Value, 4>>>
trySynthesizeElementSlice(ArtsCodegen *AC, DbAcquireOp acquire, Location loc) {
  if (!AC || !acquire)
    return std::nullopt;
  if (!acquire.getElementOffsets().empty() ||
      !acquire.getElementSizes().empty())
    return std::nullopt;

  PartitionMode mode =
      acquire.getPartitionMode().value_or(PartitionMode::coarse);
  if (!usesBlockLayout(mode))
    return std::nullopt;

  auto contract = resolveAcquireContract(acquire);
  if (!contract)
    return std::nullopt;

  const bool allowReadSlice =
      acquire.getMode() != ArtsMode::in || contract->analysis.narrowableDep ||
      shouldApplyStencilHalo(*contract, acquire) ||
      shouldUsePartitionSliceAsDepWindow(*contract, acquire);
  if (!allowReadSlice)
    return std::nullopt;

  auto partitionOffsets = acquire.getPartitionOffsets();
  auto partitionSizes = acquire.getPartitionSizes();
  unsigned explicitRank =
      std::min<unsigned>(partitionOffsets.size(), partitionSizes.size());
  if (explicitRank == 0)
    return std::nullopt;
  if (contract->spatial.ownerDims.empty())
    return std::nullopt;

  SmallVector<Value, 4> blockExtents;
  if (auto staticShape = contract->getStaticBlockShape()) {
    for (int64_t dim : *staticShape)
      blockExtents.push_back(AC->createIndexConstant(dim, loc));
  } else if (!contract->spatial.blockShape.empty()) {
    blockExtents.assign(contract->spatial.blockShape.begin(),
                        contract->spatial.blockShape.end());
  } else if (auto alloc = dyn_cast_or_null<DbAllocOp>(
                 DbUtils::getUnderlyingDbAlloc(acquire.getSourcePtr()))) {
    blockExtents.assign(alloc.getElementSizes().begin(),
                        alloc.getElementSizes().end());
  }
  if (blockExtents.empty())
    return std::nullopt;

  SmallVector<unsigned, 4> explicitDims =
      resolveContractOwnerDims(*contract, explicitRank);
  if (explicitDims.size() != explicitRank)
    return std::nullopt;

  Value zero = AC->createIndexConstant(0, loc);
  SmallVector<Value, 4> elementOffsets;
  SmallVector<Value, 4> elementSizes;
  elementOffsets.reserve(blockExtents.size());
  elementSizes.reserve(blockExtents.size());
  for (Value extent : blockExtents) {
    elementOffsets.push_back(zero);
    elementSizes.push_back(AC->castToIndex(extent, loc));
  }

  for (unsigned i = 0; i < explicitRank; ++i) {
    unsigned dim = explicitDims[i];
    if (dim >= blockExtents.size())
      return std::nullopt;
    elementOffsets[dim] = AC->castToIndex(partitionOffsets[i], loc);
    elementSizes[dim] = AC->castToIndex(partitionSizes[i], loc);
  }

  ARTS_DEBUG("Synthesized element slice for dep acquire with ownerDims rank "
             << explicitRank << " and block rank " << blockExtents.size());
  return std::pair<SmallVector<Value, 4>, SmallVector<Value, 4>>{
      std::move(elementOffsets), std::move(elementSizes)};
}

static bool hasTrustedPartitionedWriteContract(DbAcquireOp acquire) {
  if (!acquire)
    return false;

  auto mode = acquire.getPartitionMode().value_or(PartitionMode::coarse);
  if (!usesBlockLayout(mode))
    return false;

  bool hasPartitionWindow =
      (!acquire.getPartitionOffsets().empty() &&
       !acquire.getPartitionSizes().empty()) ||
      (!acquire.getOffsets().empty() && !acquire.getSizes().empty());
  if (!hasPartitionWindow)
    return false;

  if (auto contract = resolveAcquireContract(acquire))
    if (contract->hasExplicitStencilContract() &&
        contract->supportsBlockHalo() && contract->hasOwnerDims())
      return true;

  LoweringContractOp contractOp = getLoweringContractOp(acquire.getPtr());
  if (!contractOp)
    contractOp = getLoweringContractOp(acquire.getSourcePtr());
  if (!contractOp)
    return false;

  OwnershipProof proof = readOwnershipProof(contractOp.getOperation());
  if (!(proof.ownerDimReachability && proof.partitionAccessMapping &&
        proof.haloLegality))
    proof = computeOwnershipProof(contractOp);
  return proof.ownerDimReachability && proof.partitionAccessMapping &&
         proof.haloLegality;
}

static bool canUseUnorderedLocalWrite(DbAcquireOp acquire, EdtOp edtOp,
                                      mlir::arts::AnalysisManager *AM) {
  if (!acquire || !edtOp || acquire.getMode() != ArtsMode::out)
    return false;
  if (edtOp.getConcurrency() != EdtConcurrency::intranode)
    return false;
  if (AM && AM->getRuntimeConfig().hasValidNodeCount() &&
      AM->getRuntimeConfig().getNodeCount() != 1)
    return false;
  return hasTrustedPartitionedWriteContract(acquire);
}

///===----------------------------------------------------------------------===///
/// EDT Lowering Pass Implementation
///===----------------------------------------------------------------------===///
struct EdtLoweringPass : public arts::impl::EdtLoweringBase<EdtLoweringPass> {
  explicit EdtLoweringPass(mlir::arts::AnalysisManager *AM = nullptr,
                           uint64_t idStride = IdRegistry::DefaultStride)
      : idStride(idStride), AM(AM) {}
  EdtLoweringPass(const EdtLoweringPass &other)
      : arts::impl::EdtLoweringBase<EdtLoweringPass>(other),
        idStride(other.idStride), functionCounter(other.functionCounter),
        module(other.module), AM(other.AM), AC(other.AC),
        idRegistry(other.idRegistry) {}
  void runOnOperation() override;

private:
  /// Core transformation methods
  LogicalResult lowerEdt(EdtOp edtOp);
  void gatherLowerableTaskEdts(SmallVectorImpl<EdtOp> &taskEdts);
  Value computeDependencyCount(Location loc, ArrayRef<Value> edtDeps);

  /// Function outlining with ARTS signature
  func::FuncOp createOutlinedFunction(EdtOp edtOp, EdtEnvManager &envManager);

  /// Parameter handling
  FailureOr<Value> packParams(Location loc, EdtEnvManager &envManager,
                              SmallVector<Type> &packTypes,
                              SmallVectorImpl<Value> *packedValues = nullptr,
                              bool preserveUndefParams = false);

  /// Region outlining
  LogicalResult
  outlineRegionToFunction(EdtOp edtOp, func::FuncOp targetFunc,
                          EdtEnvManager &envManager,
                          SmallVector<Type> &packTypes, size_t numUserParams,
                          SmallVectorImpl<unsigned> &livePackIndices);

  void transformDepUses(ArrayRef<Value> originalDeps, Value depv,
                        ArrayRef<Value> allParams, EdtEnvManager &envManager,
                        ArrayRef<Value> depIdentifiers);

  /// Clone EDT body operations with nested region remapping
  void cloneAndRemapEdtBody(Block &sourceBlock, OpBuilder &builder,
                            IRMapping &valueMapping,
                            const DenseMap<Value, Value> &moveValueMapping);

  /// Dep satisfaction
  LogicalResult insertDepManagement(EdtOp edtOp, Location loc, Value edtGuid,
                                    const SmallVector<Value> &deps);

  mlir::arts::AnalysisManager &getAnalysisManager();

  /// Attributes
  uint64_t idStride = IdRegistry::DefaultStride;
  unsigned functionCounter = 0;
  ModuleOp module;
  mlir::arts::AnalysisManager *AM = nullptr;
  std::unique_ptr<mlir::arts::AnalysisManager> ownedAM;
  ArtsCodegen *AC = nullptr;
  IdRegistry idRegistry;
};

} // namespace

///===----------------------------------------------------------------------===///
/// Pass Implementation
///===----------------------------------------------------------------------===///

mlir::arts::AnalysisManager &EdtLoweringPass::getAnalysisManager() {
  if (!AM) {
    ownedAM = std::make_unique<mlir::arts::AnalysisManager>(module);
    AM = ownedAM.get();
  }
  return *AM;
}

void EdtLoweringPass::runOnOperation() {
  module = getOperation();
  auto ownedAC = std::make_unique<ArtsCodegen>(module, false);
  AC = ownedAC.get();
  mlir::arts::AnalysisManager &analysisManager = getAnalysisManager();

  ARTS_INFO_HEADER(EdtLoweringPass);
  ARTS_DEBUG_REGION(module.dump(););

  /// Snapshot EDT capture contracts before nested lowering mutates parent EDT
  /// bodies. DbAnalysis must also be fresh because EdtAnalysis rebuilds its
  /// graph through the DB analysis layer.
  analysisManager.getDbAnalysis().invalidate();
  analysisManager.getEdtAnalysis().invalidate();
  analysisManager.getEdtAnalysis().analyze();

  ARTS_DEBUG_HEADER(TaskEdtLowering);
  SmallVector<EdtOp, 8> taskEdts;
  gatherLowerableTaskEdts(taskEdts);
  ARTS_INFO("Found " << taskEdts.size() << " task EDTs to lower");
  for (EdtOp edtOp : taskEdts) {
    if (failed(lowerEdt(edtOp))) {
      edtOp.emitError("Failed to lower task EDT");
      return signalPassFailure();
    }
    /// Each lowerEdt call outlines the EDT body, invalidating any cached
    /// DbGraph nodes that reference block arguments or operations from the
    /// old EDT body. Invalidate so the next EDT's analysis rebuilds fresh.
    analysisManager.getDbAnalysis().invalidate();
  }
  analysisManager.getEdtAnalysis().invalidate();
  analysisManager.getDbAnalysis().invalidate();

  ARTS_INFO_FOOTER(EdtLoweringPass);
  AC = nullptr;
  ARTS_DEBUG_REGION(module.dump(););
}

void EdtLoweringPass::gatherLowerableTaskEdts(
    SmallVectorImpl<EdtOp> &taskEdts) {
  module.walk<WalkOrder::PostOrder>([&](EdtOp edtOp) {
    /// Demote any remaining non-task EDTs (e.g., single) so we can lower
    /// them uniformly.
    if (edtOp.getType() != EdtType::task) {
      ARTS_DEBUG("Demoting non-task EDT to task: " << edtOp);
      ++numEdtsDemotedToTask;
      edtOp.setType(EdtType::task);
      arts::setWorkers(edtOp.getOperation(), 0);
      arts::setWorkersPerNode(edtOp.getOperation(), 0);
      setNowait(edtOp, false);
    }
    taskEdts.push_back(edtOp);
  });
}

///===----------------------------------------------------------------------===///
/// Lower EDT operations to runtime calls
///
/// Transforms arts.edt operations into outlined functions and runtime calls.
/// Creates an outlined function with the EDT body, packs parameters and
/// dependencies, and replaces the EDT with arts.edt_create and dependency
/// management calls. Example:
///   %result = arts.edt(%dep) {
///   ^bb0(%arg: memref<f64>):
///     %val = memref.load %arg[] : memref<f64>
///     arts.return %val : f64
///   }
/// becomes:
///   %param_pack = arts.edt_param_pack ...
///   %edt_guid = arts.edt_create %param_pack, %dep_count, %route
///   arts.record_in_dep %edt_guid, %dep
///   arts.increment_out_latch %edt_guid
///===----------------------------------------------------------------------===///
LogicalResult EdtLoweringPass::lowerEdt(EdtOp edtOp) {
  OpBuilder::InsertionGuard IG(AC->getBuilder());
  AC->setInsertionPoint(edtOp);
  Location loc = edtOp.getLoc();
  /// Recompute the live capture environment from the current EDT body.
  /// EpochOpt and other structural passes can rewrite continuation captures
  /// late in the pipeline; lowering must follow the final IR contract, not a
  /// potentially stale cached summary.
  EdtEnvManager envManager(edtOp);
  for (Value captured : envManager.getCapturedValues()) {
    Operation *defOp = captured.getDefiningOp();
    if ((defOp && defOp->hasTrait<OpTrait::ConstantLike>()) ||
        isUndefLikeOp(defOp))
      continue;
    if (isa_and_nonnull<memref::AllocaOp>(defOp))
      continue;
    if (!captured.getType().isIntOrIndexOrFloat()) {
      if (isa<MemRefType>(captured.getType()))
        continue;
      return edtOp.emitError("EDT region captures value '")
             << captured
             << "' directly; pass it as an EDT dependency or param and use "
                "the corresponding block argument";
    }
    return edtOp.emitError("EDT region captures scalar value '")
           << captured
           << "' directly; pass it as an EDT param and use the param block "
              "argument";
  }
  ArrayRef<Value> edtDeps = envManager.getDependencies();

  /// Normalize dependency slices before outlining so the packed parameters,
  /// outlined dep_gep math, depCount, and rec_dep lowering all observe the
  /// same DB-space contract.
  for (Value dep : edtDeps)
    if (auto acquire = dep.getDefiningOp<DbAcquireOp>())
      if (auto contract = resolveAcquireContract(acquire))
        normalizeTaskDepSlice(AC, acquire, *contract);

  SmallVector<Type> packTypes;
  SmallVector<Value> packedValues;

  /// Create edt outlined function
  func::FuncOp outlinedFunc = createOutlinedFunction(edtOp, envManager);
  if (!outlinedFunc)
    return edtOp.emitError("Failed to create outlined function");

  /// Pack parameters
  FailureOr<Value> maybeParamPack = packParams(
      loc, envManager, packTypes, &packedValues,
      /*preserveUndefParams=*/true);
  if (failed(maybeParamPack))
    return failure();
  Value paramPack = *maybeParamPack;

  /// Outline region to function and replace EDT
  SmallVector<unsigned> livePackIndices;
  if (failed(outlineRegionToFunction(edtOp, outlinedFunc, envManager, packTypes,
                                     envManager.getParameters().size(),
                                     livePackIndices))) {
    return edtOp.emitError("Failed to outline region to function");
  }

  /// transformDepUses can forward nested continuation DB families back onto
  /// depv, leaving previously packed handle slots dead. Rebuild the caller
  /// param pack to the compact live ABI so continuations do not preserve
  /// redundant zero-filled carry state.
  if (!packedValues.empty() && livePackIndices.size() != packedValues.size()) {
    SmallVector<Value> livePackValues;
    livePackValues.reserve(livePackIndices.size());
    for (unsigned index : livePackIndices) {
      if (index < packedValues.size())
        livePackValues.push_back(packedValues[index]);
    }

    if (livePackValues.empty()) {
      auto emptyType = MemRefType::get({0}, AC->Int64);
      paramPack =
          AC->create<EdtParamPackOp>(loc, TypeRange{emptyType}, ValueRange{});
    } else {
      auto memrefType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
      paramPack = AC->create<EdtParamPackOp>(loc, TypeRange{memrefType},
                                             ValueRange(livePackValues));
    }
  }

  /// When an SDE-authored structured plan is present, emit split state/dep ops
  /// to make the state-vs-dependency separation explicit in the IR.
  /// The generic path remains the fallback for unplanned EDTs.
  if (hasStructuredPlanAttrs(edtOp.getOperation())) {
    ARTS_DEBUG("Plan-aware path: structured plan attrs present");

    /// Emit state_pack for scalar parameters (non-dep captured values).
    SmallVector<Value> stateValues;
    for (Value param : envManager.getParameters())
      stateValues.push_back(param);
    for (Value constant : envManager.getConstants())
      stateValues.push_back(constant);

    if (!stateValues.empty()) {
      auto stateMemrefType = MemRefType::get(
          {static_cast<int64_t>(stateValues.size())}, AC->Int64);
      AC->create<StatePackOp>(loc, stateMemrefType, stateValues);
      ARTS_DEBUG("  Emitted state_pack with " << stateValues.size()
                                              << " values");
    }

    /// Emit dep_bind for each dependency to make slot assignment explicit.
    for (Value dep : edtDeps) {
      auto acquire = dep.getDefiningOp<DbAcquireOp>();
      if (!acquire)
        continue;
      Value guidMemref = acquire.getGuid();
      if (!guidMemref)
        continue;
      Value guidScalar = loadRepresentativeGuidScalar(AC, loc, guidMemref);
      if (!guidScalar)
        continue;
      Value modeVal = AC->createIntConstant(
          static_cast<int64_t>(acquire.getMode()), AC->Int64, loc);
      AC->create<DepBindOp>(loc, guidScalar, modeVal);
      ARTS_DEBUG("  Emitted dep_bind for dep");
    }

    /// Propagate plan attrs to the EdtCreateOp (set below).
    /// Fall through to the generic EdtCreateOp emission.
  }

  /// Calculate dependency count from the dependency view of each DB.
  Value depCount = computeDependencyCount(loc, edtDeps);

  /// If this EDT has a control dependency (epoch continuation slot), add +1
  /// to depCount. The control slot is the last slot, satisfied by the epoch
  /// finish signal via arts_signal_edt_value.
  if (edtOp->hasAttr(ControlDep)) {
    Value one = AC->createIntConstant(1, AC->Int32, loc);
    depCount = AC->create<arith::AddIOp>(loc, depCount, one);
    ARTS_DEBUG("  Adding +1 control dep slot for epoch continuation");
  }

  AC->setInsertionPoint(edtOp);
  Value routeVal = edtOp.getRoute();
  if (!routeVal)
    routeVal = createCurrentNodeRoute(AC->getBuilder(), loc);

  ARTS_DEBUG("Creating EdtCreateOp");
  EdtCreateOp outlineOp =
      AC->create<EdtCreateOp>(loc, paramPack, depCount, routeVal);

  setOutlinedFunc(outlineOp, outlinedFunc.getName());

  /// Propagate continuation attributes so EpochLowering can find them.
  if (edtOp->hasAttr(ContinuationForEpoch))
    outlineOp->setAttr(ContinuationForEpoch, AC->getBuilder().getUnitAttr());
  if (edtOp->hasAttr(ControlDep))
    outlineOp->setAttr(ControlDep, edtOp->getAttr(ControlDep));
  if (edtOp->hasAttr(AttrNames::Operation::ReadyLocalLaunch))
    outlineOp->setAttr(AttrNames::Operation::ReadyLocalLaunch,
                       AC->getBuilder().getUnitAttr());

  /// Preserve structured launch schemas on the lowered create op for consumers
  /// that need the explicit state/dependency ABI.
  if (auto stateSchema = edtOp->getAttrOfType<DenseI64ArrayAttr>(
          AttrNames::Operation::LaunchState::StateSchema))
    outlineOp->setAttr(AttrNames::Operation::LaunchState::StateSchema,
                       stateSchema);
  if (auto depSchema = edtOp->getAttrOfType<DenseI64ArrayAttr>(
          AttrNames::Operation::LaunchState::DepSchema))
    outlineOp->setAttr(AttrNames::Operation::LaunchState::DepSchema, depSchema);
  int64_t baseId = getArtsId(edtOp);
  if (!baseId)
    baseId = idRegistry.getOrCreate(edtOp.getOperation());

  if (baseId) {
    int64_t createId = baseId * static_cast<int64_t>(idStride);
    setArtsCreateId(outlineOp, createId);
    ARTS_DEBUG("  - EDT arts.create_id=" << createId << " (base=" << baseId
                                         << " x stride=" << idStride << ")");
  }

  /// Insert dependency management after the outline op.
  Value edtGuid = outlineOp.getGuid();
  AC->setInsertionPointAfter(outlineOp);
  SmallVector<Value> depsVec(edtDeps.begin(), edtDeps.end());
  if (failed(insertDepManagement(edtOp, loc, edtGuid, depsVec)))
    return edtOp.emitError("Failed to insert dependency management");

  /// Replace all uses of EDT with the outlined function result.
  if (edtOp->getNumResults() > 0) {
    SmallVector<Value> replacementValues = {outlineOp.getResult()};
    edtOp->replaceAllUsesWith(replacementValues);
  }

  /// Remove original EDT.
  edtOp.erase();
  ++numEdtsLowered;

  return success();
}

Value EdtLoweringPass::computeDependencyCount(Location loc,
                                              ArrayRef<Value> edtDeps) {
  /// For partitioned acquires, this uses slice sizes (not full allocation
  /// sizes) so depCount matches the slots produced by rec_dep lowering.
  Value depCount = AC->createIntConstant(0, AC->Int32, loc);
  DenseSet<Operation *> seenSources;
  for (Value dep : edtDeps) {
    if (Operation *source = getCanonicalDependencySource(dep))
      if (!seenSources.insert(source).second)
        continue;
    SmallVector<Value> sizes = DbUtils::getDepSizesFromDb(dep);
    Value numElements = AC->create<DbNumElementsOp>(loc, sizes);
    numElements = AC->castToInt(AC->Int32, numElements, loc);
    depCount = AC->create<arith::AddIOp>(loc, depCount, numElements);
  }
  return depCount;
}

///===----------------------------------------------------------------------===///
/// Create outlined function for EDT body
///
/// Generates a new private function with the ARTS EDT signature to contain
/// the outlined EDT region.
///===----------------------------------------------------------------------===///
func::FuncOp
EdtLoweringPass::createOutlinedFunction(EdtOp edtOp,
                                        EdtEnvManager &envManager) {
  Location loc = edtOp.getLoc();
  OpBuilder::InsertionGuard IG(AC->getBuilder());
  AC->setInsertionPoint(module);
  std::string funcName = "__arts_edt_" + std::to_string(++functionCounter);
  auto outlinedFunc = AC->create<func::FuncOp>(loc, funcName, AC->EdtFn);
  outlinedFunc.setPrivate();

  /// Mark pointer parameters with attributes to enable LLVM optimizations.
  /// Without these attributes, LLVM cannot prove aliasing properties, which
  /// prevents hoisting of loop-invariant loads and other optimizations.
  auto unitAttr = AC->getBuilder().getUnitAttr();

  /// Arg 1 (paramv): Static parameters array - never modified during EDT
  outlinedFunc.setArgAttr(1, "llvm.noalias", unitAttr);
  outlinedFunc.setArgAttr(1, "llvm.readonly", unitAttr);
  outlinedFunc.setArgAttr(1, "llvm.nofree", unitAttr);
  outlinedFunc.setArgAttr(1, "llvm.nocapture", unitAttr);

  /// Arg 3 (depv): Deps array - struct pointers never modified, enables LICM
  outlinedFunc.setArgAttr(3, "llvm.noalias", unitAttr);
  outlinedFunc.setArgAttr(3, "llvm.readonly", unitAttr);
  outlinedFunc.setArgAttr(3, "llvm.nofree", unitAttr);
  outlinedFunc.setArgAttr(3, "llvm.nocapture", unitAttr);

  copyCoreExecutionHintAttrsToRtFunction(edtOp, outlinedFunc.getOperation());

  ARTS_INFO("Created outlined function: " << funcName);
  return outlinedFunc;
}

///===----------------------------------------------------------------------===///
/// Pack EDT parameters and dependency metadata
///
/// Creates a parameter pack containing scalar user parameters and metadata for
/// datablock dependencies (indices, offsets, sizes).
///===----------------------------------------------------------------------===///
FailureOr<Value>
EdtLoweringPass::packParams(Location loc, EdtEnvManager &envManager,
                            SmallVector<Type> &packTypes,
                            SmallVectorImpl<Value> *packedValues,
                            bool preserveUndefParams) {
  const auto &parameters = envManager.getParameters();
  const auto &deps = envManager.getDependencies();
  const auto &dbHandles = envManager.getDbHandles();

  SmallVector<Value> packValues;
  auto &valueToPackIndex = envManager.getValueToPackIndex();

  auto appendPackValue = [&](Value value) -> LogicalResult {
    if (!value.getType().isIntOrIndexOrFloat())
      return emitError(loc, "EDT parameter pack accepts only scalar values, got ")
             << value.getType();
    packTypes.push_back(value.getType());
    packValues.push_back(value);
    return success();
  };

  /// Pack user parameters first
  for (Value v : parameters) {
    /// Skip undef-like values - they can be recreated in the outlined body.
    /// CODIR explicit ABI params are positional block arguments, so even
    /// undef-like scalar slots must be preserved to keep arg mapping aligned.
    if (!preserveUndefParams && isUndefLikeOp(v.getDefiningOp()))
      continue;
    valueToPackIndex.try_emplace(v, packValues.size());
    if (failed(appendPackValue(v)))
      return failure();
  }

  if (!dbHandles.empty()) {
    InFlightDiagnostic diag =
        emitError(loc, "EDT captures pointer-bearing value that must be "
                       "passed as a dependency/datablock, not as a parameter");
    diag << ": " << dbHandles.front();
    return failure();
  }

  /// Insert indices/offsets/sizes for deps into packValues if not already
  /// present
  for (Value dep : deps) {
    auto dbAcquireOp = dep.getDefiningOp<DbAcquireOp>();
    if (!dbAcquireOp)
      continue;

    auto appendIfMissing = [&](Value val) -> LogicalResult {
      if (!val)
        return success();
      /// Skip constants; they will be recreated in outlined function
      if (val.getDefiningOp<arith::ConstantOp>())
        return success();
      if (valueToPackIndex.count(val) == 0) {
        valueToPackIndex[val] = packValues.size();
        if (failed(appendPackValue(val)))
          return failure();
      }
      return success();
    };

    for (Value idx : dbAcquireOp.getIndices())
      if (failed(appendIfMissing(idx)))
        return failure();
    for (Value off : dbAcquireOp.getOffsets())
      if (failed(appendIfMissing(off)))
        return failure();
    for (Value sz : dbAcquireOp.getSizes())
      if (failed(appendIfMissing(sz)))
        return failure();
    for (Value partIdx : dbAcquireOp.getPartitionIndices())
      if (failed(appendIfMissing(partIdx)))
        return failure();
    for (Value partOff : dbAcquireOp.getPartitionOffsets())
      if (failed(appendIfMissing(partOff)))
        return failure();
    for (Value partSize : dbAcquireOp.getPartitionSizes())
      if (failed(appendIfMissing(partSize)))
        return failure();

    /// Also pack element sizes from the underlying DbAllocOp. These are needed
    /// inside the outlined function to create DynLoadOp/DynStoreOp for
    /// multi-dynamic-dim memref accesses via DbRefOp → Pointer2MemrefOp.
    if (auto *rawAlloc =
            DbUtils::getUnderlyingDbAlloc(dbAcquireOp.getSourcePtr())) {
      if (auto alloc = dyn_cast<DbAllocOp>(rawAlloc)) {
        for (Value elemSz : alloc.getElementSizes())
          if (failed(appendIfMissing(elemSz)))
            return failure();
      }
    }
  }

  if (packValues.empty()) {
    if (packedValues)
      packedValues->clear();
    ARTS_INFO("No parameters to pack, creating empty EdtParamPackOp");
    auto emptyType = MemRefType::get({0}, AC->Int64);
    return AC->create<EdtParamPackOp>(loc, TypeRange{emptyType}, ValueRange{})
        .getResult();
  }

  if (packedValues)
    packedValues->assign(packValues.begin(), packValues.end());

  ARTS_DEBUG_REGION({
    ARTS_INFO("Creating parameter pack for " << packValues.size() << " items");
    for (size_t i = 0; i < packValues.size(); ++i)
      ARTS_DEBUG("  packValues[" << i << "]: " << packValues[i]);
  });

  auto memrefType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
  auto packOp = AC->create<EdtParamPackOp>(loc, TypeRange{memrefType},
                                           ValueRange(packValues));
  ++numParamPacksCreated;
  return packOp.getResult();
}

///===----------------------------------------------------------------------===///
/// Outline EDT region to target function
///
/// Moves the EDT body region into the outlined function, performs parameter
/// and dependency unpacking, and updates value references to work with the
/// new function context.
///===----------------------------------------------------------------------===///
LogicalResult EdtLoweringPass::outlineRegionToFunction(
    EdtOp edtOp, func::FuncOp targetFunc, EdtEnvManager &envManager,
    SmallVector<Type> &packTypes, size_t numUserParams,
    SmallVectorImpl<unsigned> &livePackIndices) {
  Location loc = edtOp.getLoc();
  auto &builder = AC->getBuilder();
  OpBuilder::InsertionGuard IG(builder);
  auto *entryBlock = targetFunc.addEntryBlock();
  AC->setInsertionPointToStart(entryBlock);

  /// Insert parameter and dependency unpacking
  auto args = entryBlock->getArguments();
  Value paramv = args[1];
  Value depv = args[3];
  if (isa<LLVM::LLVMPointerType>(paramv.getType()))
    paramv = AC->create<polygeist::Pointer2MemrefOp>(loc, AC->Int64Ptr,
                                                     paramv);
  if (isa<LLVM::LLVMPointerType>(depv.getType()))
    depv = AC->create<polygeist::Pointer2MemrefOp>(loc, AC->ArtsEdtDepPtr,
                                                   depv);

  const auto &parameters = envManager.getParameters();
  ValueRange explicitParams = edtOp.getParams();
  SmallVector<Value> unpackedParams, allParams;
  EdtParamUnpackOp paramUnpackOp = nullptr;

  if (!packTypes.empty()) {
    paramUnpackOp = AC->create<EdtParamUnpackOp>(loc, packTypes, paramv);
    auto results = paramUnpackOp.getResults();
    allParams.assign(results.begin(), results.end());
    unpackedParams.append(allParams.begin(), allParams.begin() + numUserParams);
  }

  /// Create per-dependency placeholders for later dep_gep rewrite
  Region &edtRegion = edtOp.getRegion();
  Block &edtBlock = edtRegion.front();
  ArrayRef<Value> originalDeps = envManager.getDependencies();

  SmallVector<Value> depPlaceholders(originalDeps.size());
  for (auto it : llvm::enumerate(originalDeps))
    depPlaceholders[it.index()] =
        AC->create<UndefOp>(loc, edtBlock.getArguments()[it.index()].getType());

  /// Build value mapping for cloning EDT body into outlined function
  IRMapping valueMapping;
  DenseMap<Value, Value> moveValueMapping;
  auto mapValue = [&](Value from, Value to) {
    valueMapping.map(from, to);
    moveValueMapping[from] = to;
  };

  /// Map EDT args to placeholders so cloned ops don't reference outer values.
  for (auto [edtArg, placeholder] :
       llvm::zip(edtBlock.getArguments().take_front(originalDeps.size()),
                 depPlaceholders))
    mapValue(edtArg, placeholder);

  if (explicitParams.size() > unpackedParams.size())
    return edtOp.emitOpError()
           << "explicit EDT params were not materialized in the parameter pack";
  for (auto [index, param] : llvm::enumerate(explicitParams)) {
    unsigned argIndex = originalDeps.size() + static_cast<unsigned>(index);
    if (argIndex >= edtBlock.getNumArguments())
      return edtOp.emitOpError()
             << "explicit EDT param block argument #" << index
             << " is missing";
    mapValue(edtBlock.getArgument(argIndex), unpackedParams[index]);
  }

  /// Also map original dependency values to placeholders to catch direct uses
  for (auto [originalDep, placeholder] :
       llvm::zip(originalDeps, depPlaceholders))
    mapValue(originalDep, placeholder);

  /// Clone constants and constant-like operations into the outlined function
  auto cloneConstantLike = [&](Value val) {
    if (Operation *defOp = val.getDefiningOp())
      if (defOp->hasTrait<OpTrait::ConstantLike>() || isUndefLikeOp(defOp))
        mapValue(val, builder.clone(*defOp)->getResult(0));
  };

  for (Value constant : envManager.getConstants())
    cloneConstantLike(constant);

  for (Value freeVar : envManager.getCapturedValues())
    cloneConstantLike(freeVar);

  /// Map parameters to their unpacked counterparts (skip implicit undef
  /// captures; explicit EDT params stay positional and were packed).
  size_t unpackedIndex = 0;
  for (Value param : parameters) {
    if (isUndefLikeOp(param.getDefiningOp()) &&
        !llvm::is_contained(explicitParams, param)) {
      cloneConstantLike(param);
      continue;
    }
    if (unpackedIndex < unpackedParams.size())
      mapValue(param, unpackedParams[unpackedIndex++]);
  }

  /// Clone remaining captured values that are pure and regionless (e.g.,
  /// llvm.getelementptr chains computing format strings) so the outlined
  /// function does not reference parent-scope SSA values.
  DenseSet<Operation *> visited;
  std::function<LogicalResult(Value)> cloneCaptured = [&](Value val) {
    if (valueMapping.contains(val))
      return success();

    Operation *defOp = val.getDefiningOp();
    if (!defOp)
      return success();

    if (edtOp.getOperation()->isAncestor(defOp))
      return success();

    if (defOp->getNumRegions() != 0) {
      emitError(loc, "cannot clone captured EDT value produced by "
                     "region-holding operation; route scalar values through "
                     "explicit EDT params")
          << ": " << val;
      return failure();
    }

    if (visited.contains(defOp))
      return success();
    visited.insert(defOp);

    for (Value operand : defOp->getOperands())
      if (failed(cloneCaptured(operand)))
        return failure();

    Operation *cloned = builder.clone(*defOp, valueMapping);
    for (auto [oldRes, newRes] :
         llvm::zip(defOp->getResults(), cloned->getResults()))
      mapValue(oldRes, newRes);

    /// When cloning a memref.alloca, also clone its external initialization
    /// stores so the outlined copy has the correct initial value. This handles
    /// Polygeist's pointer-wrapper pattern where `memref.store %alloc,
    /// %wrapper[]` initializes the wrapper before the EDT body accesses it.
    if (isa<memref::AllocaOp>(defOp)) {
      Value allocaResult = defOp->getResult(0);
      for (Operation *user : allocaResult.getUsers()) {
        auto storeOp = dyn_cast<memref::StoreOp>(user);
        if (!storeOp || storeOp->getBlock() != defOp->getBlock())
          continue;
        if (storeOp->isBeforeInBlock(defOp))
          continue;
        // Only clone stores OUTSIDE the EDT body (init stores)
        if (edtOp.getBody().isAncestor(storeOp->getParentRegion()))
          continue;
        // Ensure the stored value and indices are mapped
        bool canClone = succeeded(cloneCaptured(storeOp.getValue()));
        for (Value idx : storeOp.getIndices())
          canClone &= succeeded(cloneCaptured(idx));
        if (!canClone)
          continue;
        builder.clone(*storeOp, valueMapping);
      }
    }

    return success();
  };

  for (Value captured : envManager.getCapturedValues())
    if (failed(cloneCaptured(captured)))
      return failure();

  /// Move EDT body operations into the outlined function.
  builder.setInsertionPointToEnd(entryBlock);
  cloneAndRemapEdtBody(edtBlock, builder, valueMapping, moveValueMapping);

  AC->create<func::ReturnOp>(loc);

  /// Hoist constant-like ops to the front of the entry block. cloneCaptured
  /// and body cloning may produce constants after their uses; constants have
  /// no operands so moving them to the top is always safe.
  {
    Operation *firstNonConstant = nullptr;
    SmallVector<Operation *> constantsToHoist;
    for (Operation &op : *entryBlock) {
      if (op.hasTrait<OpTrait::ConstantLike>()) {
        if (firstNonConstant)
          constantsToHoist.push_back(&op);
      } else if (!firstNonConstant) {
        firstNonConstant = &op;
      }
    }
    for (Operation *op : constantsToHoist)
      op->moveBefore(firstNonConstant);
  }

  /// Transform dependency uses inside outlined region.
  transformDepUses(originalDeps, depv, allParams, envManager, depPlaceholders);

  livePackIndices.clear();
  if (paramUnpackOp) {
    SmallVector<Type> livePackTypes;
    livePackTypes.reserve(allParams.size());
    for (auto [index, param] : llvm::enumerate(allParams)) {
      if (param.use_empty())
        continue;
      livePackIndices.push_back(index);
      livePackTypes.push_back(packTypes[index]);
    }

    if (livePackIndices.size() != allParams.size()) {
      if (!livePackTypes.empty()) {
        AC->setInsertionPoint(paramUnpackOp);
        auto compactUnpack =
            AC->create<EdtParamUnpackOp>(loc, livePackTypes, paramv);
        for (auto [newIndex, oldIndex] : llvm::enumerate(livePackIndices))
          allParams[oldIndex].replaceAllUsesWith(
              compactUnpack.getResult(newIndex));
      }
      paramUnpackOp.erase();
      packTypes.assign(livePackTypes.begin(), livePackTypes.end());
    }
  }
  return success();
}

///===----------------------------------------------------------------------===///
/// Insert dependency management operations
///
/// Adds runtime dependency tracking operations (record_in_dep,
/// increment_out_latch) for EDT execution. Determines access mode based on
/// dependency source and extracts GUIDs appropriately. Example:
///   EDT with input deps %d1, %d2 and output deps %d3
///   becomes: arts.rec_dep %edt_guid, [%d1_guid, %d2_guid, %d3_guid]
///   {access_mode = direct}
///            arts.inc_dep %edt_guid, [%d3_guid] {access_mode = direct}
///===----------------------------------------------------------------------===///
LogicalResult
EdtLoweringPass::insertDepManagement(EdtOp edtOp, Location loc, Value edtGuid,
                                     const SmallVector<Value> &deps) {
  if (deps.empty())
    return success();

  /// Determine access mode based on dependency sources
  /// If any dependency comes from DepDbAcquireOp, use from_depv mode
  DepAccessMode accessMode = DepAccessMode::direct;
  for (Value dep : deps) {
    if (dep.getDefiningOp<DepDbAcquireOp>()) {
      accessMode = DepAccessMode::from_depv;
      break;
    }
  }

  /// Extract GUIDs, acquire modes, bounds validity, and ESD byte offsets.
  SmallVector<Value> depGuids, boundsValids;
  SmallVector<Value> byteOffsets, byteSizes;
  SmallVector<int32_t> acquireModes;
  SmallVector<int32_t> depFlags;
  bool hasEsdDeps = false;
  bool hasDepFlags = false;
  DenseSet<Operation *> seenSources;

  for (Value dep : deps) {
    if (Operation *source = getCanonicalDependencySource(dep))
      if (!seenSources.insert(source).second)
        continue;

    /// Handle both DbAcquireOp and DepDbAcquireOp as dependency sources, even
    /// when they are threaded through block arguments.
    DepSourceInfo depSource = resolveDepSource(dep);
    auto dbAcquireOp = depSource.dbAcquire;
    auto depDbAcquireOp = depSource.depDbAcquire;
    if (!dbAcquireOp && !depDbAcquireOp) {
      return mlir::emitError(
                 loc, "Dep must be from DbAcquireOp or DepDbAcquireOp, got: ")
             << dep;
    }

    /// Get the GUID from the acquire operation
    Value depGuid =
        dbAcquireOp ? dbAcquireOp.getGuid() : depDbAcquireOp.getGuid();
    depGuids.push_back(depGuid);

    /// Get bounds_valid if present.
    Value boundsValid =
        dbAcquireOp
            ? dbAcquireOp.getBoundsValid()
            : (depDbAcquireOp ? depDbAcquireOp.getBoundsValid() : Value());
    /// Create "true" constant for deps that don't have explicit bounds checking
    if (!boundsValid)
      boundsValid = AC->create<arith::ConstantIntOp>(loc, 1, 1);
    boundsValids.push_back(boundsValid);

    /// ESD: Check for partial acquisition (element_offsets/element_sizes)
    /// When element_offsets is non-empty, compute byte_offset and byte_size
    /// using the allocation's elementSizes for linearization.
    Value byteOffset, byteSize;
    if (dbAcquireOp) {
      SmallVector<Value, 4> elemOffsets;
      SmallVector<Value, 4> elemSizes;
      if (!dbAcquireOp.getElementOffsets().empty()) {
        elemOffsets.assign(dbAcquireOp.getElementOffsets().begin(),
                           dbAcquireOp.getElementOffsets().end());
        elemSizes.assign(dbAcquireOp.getElementSizes().begin(),
                         dbAcquireOp.getElementSizes().end());
      } else if (auto synthesized =
                     trySynthesizeElementSlice(AC, dbAcquireOp, loc)) {
        elemOffsets = synthesized->first;
        elemSizes = synthesized->second;
        ARTS_DEBUG("Using synthesized element slice for dependency");
      }

      if (elemOffsets.empty()) {
        /// No partial acquisition - use zero for standard dependency
        byteOffset = AC->createIndexConstant(0, loc);
        byteSize = AC->createIndexConstant(0, loc);
      } else {
        /// Get elementSizes from underlying allocation for stride computation
        auto alloc = dyn_cast_or_null<DbAllocOp>(
            DbUtils::getUnderlyingDbAlloc(dbAcquireOp.getSourcePtr()));

        /// ESD slice transport delivers a compacted copy of the requested
        /// byte range. When the allocation is coarse (single partition),
        /// the EDT body uses global indices to access the full array, so
        /// a compacted sub-copy would cause incorrect addressing.
        bool isCoarseAlloc =
            alloc && alloc.getPartitionMode() == PartitionMode::coarse;
        if (alloc && !alloc.getElementSizes().empty() && !isCoarseAlloc) {
          auto elementSizes = alloc.getElementSizes();
          Value useSliceTransport = AC->create<arith::ConstantIntOp>(loc, 1, 1);
          if (!dbAcquireOp.getElementOffsets().empty())
            if (auto normalized =
                    normalizeCommonElementSlice(AC, dbAcquireOp, alloc)) {
              elemOffsets.assign(normalized->offsets.begin(),
                                 normalized->offsets.end());
              elemSizes.assign(normalized->sizes.begin(),
                               normalized->sizes.end());
              /// ARTS ESD is copy-based RO transport. If the normalized slice
              /// still covers the entire local DB block, using
              /// arts_add_dependence_at would only copy the whole block instead
              /// of reusing the normal whole-DB dependence path.
              Value sliceNarrowerThanBlock = AC->create<arith::XOrIOp>(
                  loc, normalized->wholeBlock,
                  AC->create<arith::ConstantIntOp>(loc, 1, 1));
              Value sliceRepresentable = AC->create<arith::AndIOp>(
                  loc, normalized->representable, normalized->contiguous);
              useSliceTransport = AC->create<arith::AndIOp>(
                  loc, sliceNarrowerThanBlock, sliceRepresentable);
            }

          /// Get scalar type size from the element type
          Type elementType = alloc.getElementType();
          if (auto memrefType = dyn_cast<MemRefType>(elementType))
            elementType = memrefType.getElementType();
          Value scalarSize = AC->create<polygeist::TypeSizeOp>(
              loc, IndexType::get(AC->getContext()), elementType);

          /// Compute byte_offset = linearize(element_offsets, elementSizes) *
          /// scalarSize For 2D: byte_offset = (elemOffsets[0] * elementSizes[1]
          /// + elemOffsets[1]) * scalarSize
          Value linearOffset = AC->createIndexConstant(0, loc);
          for (size_t i = 0; i < elemOffsets.size(); ++i) {
            /// Compute stride for this dimension (product of trailing dims)
            Value stride = AC->createIndexConstant(1, loc);
            for (size_t j = i + 1; j < elementSizes.size(); ++j) {
              stride = AC->create<arith::MulIOp>(loc, stride, elementSizes[j]);
            }
            Value dimOffset =
                AC->create<arith::MulIOp>(loc, elemOffsets[i], stride);
            linearOffset =
                AC->create<arith::AddIOp>(loc, linearOffset, dimOffset);
          }
          byteOffset = AC->create<arith::MulIOp>(loc, linearOffset, scalarSize);

          /// Compute byte_size = product(element_sizes) * scalarSize
          Value totalElements = AC->createIndexConstant(1, loc);
          for (Value sz : elemSizes) {
            totalElements = AC->create<arith::MulIOp>(loc, totalElements, sz);
          }
          byteSize = AC->create<arith::MulIOp>(loc, totalElements, scalarSize);
          Value zeroIdx = AC->createIndexConstant(0, loc);
          byteOffset = AC->create<arith::SelectOp>(loc, useSliceTransport,
                                                   byteOffset, zeroIdx);
          byteSize = AC->create<arith::SelectOp>(loc, useSliceTransport,
                                                 byteSize, zeroIdx);
          hasEsdDeps = true;
        } else {
          /// Fallback: no allocation info available
          byteOffset = AC->createIndexConstant(0, loc);
          byteSize = AC->createIndexConstant(0, loc);
        }
      }
    } else {
      /// No partial acquisition - use zero for standard dependency
      byteOffset = AC->createIndexConstant(0, loc);
      byteSize = AC->createIndexConstant(0, loc);
    }
    byteOffsets.push_back(byteOffset);
    byteSizes.push_back(byteSize);

    DbAllocOp allocForHint =
        dyn_cast_or_null<DbAllocOp>(DbUtils::getUnderlyingDbAlloc(dep));
    if (dbAcquireOp && !allocForHint) {
      if (auto *rawAlloc =
              DbUtils::getUnderlyingDbAlloc(dbAcquireOp.getSourcePtr()))
        allocForHint = dyn_cast<DbAllocOp>(rawAlloc);
    }

    /// Extract acquire mode: Convert ArtsMode to DbMode enum
    ArtsMode artsMode = ArtsMode::inout;
    if (dbAcquireOp)
      artsMode = dbAcquireOp.getMode();

    /// Map ArtsMode to the runtime DB mode. Inout must stay RW; collapsing it
    /// to write-only EW loses the dependency's read side and can expose stale
    /// or uninitialized payload data to read-modify-write EDT bodies.
    /// Preserve explicit READ acquisitions (e.g., aggregator reading partials)
    /// even if the underlying allocation is WRITE-only. The allocation mode is
    /// only used to narrow (not widen) the access when the arts mode is not
    /// already read.
    DbMode dbMode = DbUtils::convertArtsModeToDbMode(artsMode);
    int32_t runtimeDbMode =
        artsMode == ArtsMode::inout ? kArtsRuntimeDbModeRw
                                    : static_cast<int32_t>(dbMode);
    if (allocForHint && dbMode != DbMode::read &&
        artsMode != ArtsMode::inout) {
      DbMode allocMode = allocForHint.getDbMode();
      if (allocMode == DbMode::read || allocMode == DbMode::write)
        dbMode = allocMode;
      runtimeDbMode = static_cast<int32_t>(dbMode);
    }
    if (dbMode == DbMode::write &&
        canUseUnorderedLocalWrite(dbAcquireOp, edtOp, AM)) {
      runtimeDbMode = kArtsRuntimeDbModeRw;
      ARTS_DEBUG("Using unordered local DB_MODE_RW for proven partitioned "
                 "out dependency");
    }
    acquireModes.push_back(runtimeDbMode);

    int32_t depFlagBits = 0;
    if (allocForHint &&
        hasDistributedDbAllocation(allocForHint.getOperation()) &&
        dbMode == DbMode::read) {
      bool duplicateSafe =
          allocForHint.getDbMode() == DbMode::read ||
          allocForHint->hasAttr(AttrNames::Operation::ReadOnlyAfterInit);
      if (duplicateSafe) {
        depFlagBits |= kArtsDepFlagPreferDuplicate;
      }
    }
    bool hasExplicitSlice = byteOffset && byteSize &&
                            !ValueAnalysis::isZeroConstant(
                                ValueAnalysis::stripNumericCasts(byteSize));
    if (depDbAcquireOp && hasExplicitSlice) {
      /// Depv-carried slices are the last remaining unstable path in the
      /// 64-thread stencil continuation benchmarks. Fall back to whole-block
      /// dependencies for depv acquires instead of preserving a compact slice
      /// through the runtime transport.
      Value zeroIdx = AC->createIndexConstant(0, loc);
      byteOffset = zeroIdx;
      byteSize = zeroIdx;
      hasExplicitSlice = false;
    }
    Operation *patternSource =
        dbAcquireOp
            ? dbAcquireOp.getOperation()
            : (depDbAcquireOp ? depDbAcquireOp.getOperation() : nullptr);
    if (patternSource && hasExplicitSlice && dbMode == DbMode::read) {
      if (auto depPattern = getEffectiveDepPattern(patternSource);
          depPattern && isStencilHaloDepPattern(*depPattern)) {
        /// Halo-exchange stencil consumers keep full block coordinates in the
        /// lowered worker body. Mark explicit byte slices as preserve-shape so
        /// the final lowering can keep them on the whole-block dependency path
        /// instead of compacting them into a shifted payload.
        depFlagBits |= kArtsDepFlagPreserveShape;
      }
    }
    /// Explicit element slices already encode the producer/consumer contract.
    /// When upstream rewrites localize the consumer to a compact halo view,
    /// forcing "preserve shape" here would discard the byte slice later in
    /// ConvertArtsToLLVM and hand the task a whole DB block instead.
    ///
    /// Keep preserve-shape as a late-lowering inference only, for cases where
    /// ConvertArtsToLLVM derives a face slice after consumer indexing has
    /// already been fixed to full-block coordinates.
    if (depFlagBits != 0)
      hasDepFlags = true;
    depFlags.push_back(depFlagBits);
  }

  /// Create dependency management ops with appropriate access mode
  auto modeAttr =
      DepAccessModeAttr::get(AC->getBuilder().getContext(), accessMode);

  DenseI32ArrayAttr acquireAttr;
  if (!acquireModes.empty())
    acquireAttr =
        DenseI32ArrayAttr::get(AC->getBuilder().getContext(), acquireModes);

  DenseI32ArrayAttr depFlagsAttr;
  if (hasDepFlags)
    depFlagsAttr =
        DenseI32ArrayAttr::get(AC->getBuilder().getContext(), depFlags);

  /// Only include byte offsets if we have ESD dependencies
  SmallVector<Value> finalByteOffsets, finalByteSizes;
  if (hasEsdDeps) {
    finalByteOffsets = byteOffsets;
    finalByteSizes = byteSizes;
  }

  AC->create<RecordDepOp>(loc, edtGuid, depGuids, boundsValids,
                          finalByteOffsets, finalByteSizes, modeAttr,
                          acquireAttr, depFlagsAttr);
  numDepsRecorded += depGuids.size();

  return success();
}

///===----------------------------------------------------------------------===///
/// Move EDT body operations with nested region remapping
///
/// Moves all operations from the EDT body into the outlined function while
/// recursively remapping operands in nested regions to use values from the
/// provided IRMapping. The original EDT is erased after lowering, so moving
/// avoids recursively cloning large nested control-flow regions.
///===----------------------------------------------------------------------===///
void EdtLoweringPass::cloneAndRemapEdtBody(Block &sourceBlock,
                                           OpBuilder &builder,
                                           IRMapping &valueMapping,
                                           const DenseMap<Value, Value>
                                               &moveValueMapping) {
  Block *destBlock = builder.getInsertionBlock();
  (void)valueMapping;
  auto remapOperands = [&](Operation *op) {
    for (const auto &it : moveValueMapping) {
      Value oldValue = it.first;
      Value newValue = it.second;
      if (!oldValue || !newValue || oldValue == newValue ||
          oldValue.getType() != newValue.getType())
        continue;
      op->replaceUsesOfWith(oldValue, newValue);
      for (Region &region : op->getRegions())
        replaceAllUsesInRegionWith(oldValue, newValue, region);
    }
  };

  for (Operation &op :
       llvm::make_early_inc_range(sourceBlock.without_terminator())) {
    remapOperands(&op);
    op.moveBefore(destBlock, destBlock->end());
  }
}

///===----------------------------------------------------------------------===///
/// Transform dependency uses in outlined function
///
/// Rewrites dependency access operations in the outlined EDT function to use
/// the packed dependency data structure. Computes proper base offsets and
/// strides for each dependency, adjusting indices to account for datablock
/// offsets.
///===----------------------------------------------------------------------===///
void EdtLoweringPass::transformDepUses(ArrayRef<Value> originalDeps, Value depv,
                                       ArrayRef<Value> allParams,
                                       EdtEnvManager &envManager,
                                       ArrayRef<Value> depIdentifiers) {
  /// Get the parameter map
  const auto &paramMap = envManager.getValueToPackIndex();

  /// Collect placeholders to erase at the end
  SmallVector<Operation *> placeholdersToErase;

  /// Resolve parameters within the outlined Edt
  auto resolveParam = [&](ArrayRef<Value> params, Location loc) {
    SmallVector<Value> resolved;
    for (Value param : params) {
      auto it = paramMap.find(param);
      if (it != paramMap.end() && it->second < allParams.size())
        resolved.push_back(allParams[it->second]);
      else if (auto c = param.getDefiningOp<arith::ConstantIndexOp>())
        resolved.push_back(AC->createIndexConstant(c.value(), loc));
      else
        resolved.push_back(param);
    }
    if (resolved.empty())
      resolved.push_back(AC->createIndexConstant(1, loc));
    return resolved;
  };

  auto resolveExplicitParams = [&](ValueRange params, Location loc) {
    SmallVector<Value> resolved;
    resolved.reserve(params.size());
    for (Value param : params) {
      auto it = paramMap.find(param);
      if (it != paramMap.end() && it->second < allParams.size())
        resolved.push_back(allParams[it->second]);
      else if (auto c = param.getDefiningOp<arith::ConstantIndexOp>())
        resolved.push_back(AC->createIndexConstant(c.value(), loc));
      else
        resolved.push_back(param);
    }
    return resolved;
  };

  /// Compute the base offset of the dependency within the outlined Edt
  /// This corresponds to the sum of number of elements in the previous
  /// dependencies
  SmallVector<Operation *> canonicalDepSources;
  canonicalDepSources.reserve(originalDeps.size());
  for (Value dep : originalDeps)
    canonicalDepSources.push_back(getCanonicalDependencySource(dep));

  auto computeBaseOffset = [&](size_t depIndex, Location loc) {
    Operation *currentSource = depIndex < canonicalDepSources.size()
                                   ? canonicalDepSources[depIndex]
                                   : nullptr;
    Value base = AC->createIndexConstant(0, loc);
    DenseSet<Operation *> seenSources;
    for (size_t i = 0; i < depIndex; ++i) {
      Operation *prevSource =
          i < canonicalDepSources.size() ? canonicalDepSources[i] : nullptr;
      if (prevSource) {
        if (prevSource == currentSource)
          continue;
        if (!seenSources.insert(prevSource).second)
          continue;
      }
      SmallVector<Value> prevSizes =
          DbUtils::getDepSizesFromDb(originalDeps[i]);
      SmallVector<Value> prevResolved = resolveParam(prevSizes, loc);
      Value prevElems = AC->computeTotalElements(prevResolved, loc);
      base = AC->create<arith::AddIOp>(loc, base, prevElems);
    }
    return base;
  };

  /// For each dependency placeholder, rewrite its direct uses.
  for (size_t depIndex = 0; depIndex < depIdentifiers.size(); ++depIndex) {
    Value placeholder = depIdentifiers[depIndex];
    Location loc = placeholder.getLoc();

    /// Get the base offset, sizes, strides, and offsets of the dependency
    ARTS_DEBUG("Processing Dep[" << depIndex
                                 << "]: " << originalDeps[depIndex]);
    AC->setInsertionPoint(placeholder.getDefiningOp());
    Value baseOffset = computeBaseOffset(depIndex, loc);
    SmallVector<Value> depSizes =
        resolveParam(DbUtils::getDepSizesFromDb(originalDeps[depIndex]), loc);
    SmallVector<Value> depOffsets =
        resolveParam(DbUtils::getDepOffsetsFromDb(originalDeps[depIndex]), loc);
    SmallVector<Value> depStrides = AC->computeStridesFromSizes(depSizes, loc);

    auto originalAcquire = originalDeps[depIndex].getDefiningOp<DbAcquireOp>();
    const bool indicesAlreadySliceRelative =
        originalAcquire && originalAcquire.getPartitionMode() &&
        usesBlockLayout(*originalAcquire.getPartitionMode());

    /// Get element sizes from the underlying DbAllocOp. These are the
    /// dimensions within each partition (e.g. [N, 100] for a 2D array).
    /// Needed for DynLoadOp/DynStoreOp when DbRefOp produces multi-dynamic-dim
    /// memrefs.
    SmallVector<Value> allocElementSizes;
    if (auto *rawAlloc =
            DbUtils::getUnderlyingDbAlloc(originalDeps[depIndex])) {
      if (auto alloc = dyn_cast<DbAllocOp>(rawAlloc))
        allocElementSizes.assign(alloc.getElementSizes().begin(),
                                 alloc.getElementSizes().end());
    }
    /// Fallback for broken DB traces: scan the module for a DbAllocOp with
    /// matching inner memref element type.
    if (allocElementSizes.empty()) {
      auto depType = dyn_cast<MemRefType>(placeholder.getType());
      if (depType) {
        Type innerElemType;
        if (auto innerMrt = dyn_cast<MemRefType>(depType.getElementType()))
          innerElemType = innerMrt.getElementType();
        if (innerElemType) {
          if (auto moduleOp =
                  placeholder.getDefiningOp()->getParentOfType<ModuleOp>()) {
            moduleOp.walk([&](DbAllocOp alloc) {
              if (!allocElementSizes.empty())
                return WalkResult::interrupt();
              if (alloc.getElementType() == innerElemType &&
                  !alloc.getElementSizes().empty())
                allocElementSizes.assign(alloc.getElementSizes().begin(),
                                         alloc.getElementSizes().end());
              return WalkResult::advance();
            });
          }
        }
      }
    }
    ARTS_DEBUG(" - allocElementSizes.size() = " << allocElementSizes.size()
                                                << " for dep " << depIndex);

    /// Replace remaining uses of the dependency placeholder with the dependency
    bool isSingleElement = DbAnalysis::hasSingleSize(
        originalDeps[depIndex].getDefiningOp<DbAcquireOp>());
    bool isNestedMemref = false;
    if (auto mt = dyn_cast<MemRefType>(placeholder.getType()))
      isNestedMemref = isa<MemRefType>(mt.getElementType());

    SmallVector<Value> accessSizes(depSizes.begin(), depSizes.end());
    SmallVector<Value> accessOffsets(depOffsets.begin(), depOffsets.end());
    SmallVector<Value> accessStrides(depStrides.begin(), depStrides.end());
    SmallVector<Value> depSlotSizes(depSizes.begin(), depSizes.end());
    SmallVector<Value> depSlotOffsets(depOffsets.begin(), depOffsets.end());
    if (originalAcquire) {
      if (!originalAcquire.getOffsets().empty())
        depSlotOffsets = resolveExplicitParams(originalAcquire.getOffsets(),
                                               originalAcquire.getLoc());
      if (!originalAcquire.getSizes().empty())
        depSlotSizes = resolveExplicitParams(originalAcquire.getSizes(),
                                             originalAcquire.getLoc());
    }
    const bool usePayloadIndexing =
        isSingleElement && !isNestedMemref && !allocElementSizes.empty();
    if (usePayloadIndexing) {
      accessSizes = resolveParam(allocElementSizes, loc);
      accessOffsets.clear();
      accessOffsets.reserve(accessSizes.size());
      for (size_t i = 0; i < accessSizes.size(); ++i)
        accessOffsets.push_back(AC->createIndexConstant(0, loc));
      accessStrides = AC->computeStridesFromSizes(accessSizes, loc);
    }
    ArrayRef<Value> depIndexOffsets = usePayloadIndexing
                                          ? ArrayRef<Value>(accessOffsets)
                                          : ArrayRef<Value>(depSlotOffsets);
    ArrayRef<Value> depIndexSizes = usePayloadIndexing
                                        ? ArrayRef<Value>(accessSizes)
                                        : ArrayRef<Value>(depSlotSizes);
    /// SDE MU/token lowering rewrites block-layout accesses to
    /// dependency-window-local DB slots before EDT outlining. Do not subtract
    /// the acquire's DB-space offset a second time here; only keep the window
    /// size clamp below.
    const bool depIndicesAlreadySliceRelative =
        usePayloadIndexing || indicesAlreadySliceRelative;
    SmallVector<unsigned, 4> depHandlePackIndices;
    if (auto sourceAcquire =
            originalDeps[depIndex].getDefiningOp<DbAcquireOp>()) {
      for (Value candidate :
           {sourceAcquire.getGuid(), sourceAcquire.getPtr()}) {
        auto it = paramMap.find(candidate);
        if (it != paramMap.end())
          depHandlePackIndices.push_back(it->second);
      }
    } else if (auto sourceAcquire =
                   originalDeps[depIndex].getDefiningOp<DepDbAcquireOp>()) {
      for (Value candidate :
           {sourceAcquire.getGuid(), sourceAcquire.getPtr()}) {
        auto it = paramMap.find(candidate);
        if (it != paramMap.end())
          depHandlePackIndices.push_back(it->second);
      }
    }

    auto tracePackedDepHandleIndex =
        [&](Value value) -> std::optional<unsigned> {
      for (unsigned depth = 0; depth < 8 && value; ++depth) {
        for (unsigned packIdx : depHandlePackIndices) {
          if (packIdx < allParams.size() && value == allParams[packIdx])
            return packIdx;
        }
        if (isa<BlockArgument>(value))
          return std::nullopt;
        Operation *defOp = value.getDefiningOp();
        if (!defOp || defOp->getNumOperands() != 1)
          return std::nullopt;
        value = defOp->getOperand(0);
      }
      return std::nullopt;
    };

    auto dbAcquireUsesCurrentDepWindow = [&](DbAcquireOp dbAcquire) {
      if (dbAcquire.getSourceGuid() == placeholder)
        return true;
      return tracePackedDepHandleIndex(dbAcquire.getSourceGuid()).has_value();
    };

    /// Get the users of the dependency placeholder.
    SmallVector<Operation *, 16> users, dbAcquireUsers;
    SmallVector<OpOperand *, 8> nestedEdtDepUses;
    SmallVector<OpOperand *, 8> recordDepUses;
    SmallVector<OpOperand *, 8> preservedDbAcquirePtrUses;
    for (auto &use : placeholder.getUses()) {
      if (auto dbAcquire = dyn_cast<arts::DbAcquireOp>(use.getOwner())) {
        if (dbAcquireUsesCurrentDepWindow(dbAcquire)) {
          dbAcquireUsers.push_back(use.getOwner());
        } else if (dbAcquire.getSourcePtr() == placeholder) {
          /// Nested acquire source-pointer uses may intentionally derive child
          /// windows from local tables and a parent payload pointer. Forwarding
          /// that acquire through this dependency's depv would collapse the
          /// child window back to this EDT's dependency slots.
          preservedDbAcquirePtrUses.push_back(&use);
        } else {
          users.push_back(use.getOwner());
        }
      } else if (isa<RecordDepOp>(use.getOwner()))
        recordDepUses.push_back(&use);
      else if (auto edt = dyn_cast<arts::EdtOp>(use.getOwner())) {
        /// Nested EDT dependency operands must stay on the depv-backed path.
        /// Leaving them as captured memrefs violates dependency/datablock
        /// routing and breaks RecordDep lowering.
        if (edt.getRoute() != placeholder)
          nestedEdtDepUses.push_back(&use);
        else
          users.push_back(use.getOwner());
      } else
        users.push_back(use.getOwner());
    }

    const bool hasDbGepUsers = llvm::any_of(
        users, [](Operation *op) { return isa<arts::DbGepOp>(op); });
    const bool useInvariantSingleDepView =
        indicesAlreadySliceRelative &&
        (depSizes.empty() ||
         llvm::all_of(depSizes, ValueAnalysis::isOneLikeValue)) &&
        !hasDbGepUsers;

    auto normalizeSliceIndices = [&](ArrayRef<Value> indices,
                                     ArrayRef<Value> offsets,
                                     ArrayRef<Value> sizes,
                                     bool alreadySliceRelative) {
      SmallVector<Value> adjusted;
      adjusted.reserve(indices.size());
      Value one = AC->createIndexConstant(1, loc);
      Value zero = AC->createIndexConstant(0, loc);
      for (size_t i = 0; i < indices.size(); ++i) {
        Value idx = indices[i];
        if (i < offsets.size()) {
          Value off = offsets[i];
          if (off.getType() != idx.getType())
            off = AC->castToInt(idx.getType(), off, loc);
          if (auto sub = idx.getDefiningOp<arith::SubIOp>();
              sub && sub.getRhs() == off) {
            Value lhs = sub.getLhs();
            if (lhs.getType() != idx.getType())
              lhs = AC->castToInt(idx.getType(), lhs, loc);
            Value belowOffset = AC->create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::ult, lhs, off);
            Value shifted = AC->create<arith::SubIOp>(loc, lhs, off);
            idx = AC->create<arith::SelectOp>(loc, belowOffset, zero, shifted);
          } else if (!alreadySliceRelative) {
            Value belowOffset = AC->create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::ult, idx, off);
            Value shifted = AC->create<arith::SubIOp>(loc, idx, off);
            idx = AC->create<arith::SelectOp>(loc, belowOffset, zero, shifted);
          }
        }
        if (i < sizes.size()) {
          Value size = sizes[i];
          if (size.getType() != idx.getType())
            size = AC->castToInt(idx.getType(), size, loc);
          Value clampedSize = AC->create<arith::MaxUIOp>(loc, size, one);
          Value maxIndex = AC->create<arith::SubIOp>(loc, clampedSize, one);
          idx = AC->create<arith::MinUIOp>(loc, idx, maxIndex);
          Value isOne = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  size, one);
          idx = AC->create<arith::SelectOp>(loc, isOne, zero, idx);
        }
        adjusted.push_back(idx);
      }
      return adjusted;
    };

    /// Rewrite the dbAcquire users
    ARTS_DEBUG(" - Rewriting " << dbAcquireUsers.size() << " dbAcquire users");
    for (Operation *op : dbAcquireUsers) {
      /// Replace with DepDbAcquireOp to access datablock from depv
      auto dbAcquire = dyn_cast<arts::DbAcquireOp>(op);
      assert(dbAcquire && "dbAcquire must be a DbAcquireOp");
      AC->setInsertionPoint(op);
      SmallVector<Value> indices(dbAcquire.getIndices().begin(),
                                 dbAcquire.getIndices().end());
      SmallVector<Value> offsets(dbAcquire.getOffsets().begin(),
                                 dbAcquire.getOffsets().end());
      SmallVector<Value> sizes(dbAcquire.getSizes().begin(),
                               dbAcquire.getSizes().end());
      Value boundsValid = dbAcquire.getBoundsValid();
      indices = resolveExplicitParams(indices, dbAcquire.getLoc());
      indices = normalizeSliceIndices(indices, depIndexOffsets, depIndexSizes,
                                      depIndicesAlreadySliceRelative);
      offsets = resolveExplicitParams(offsets, dbAcquire.getLoc());
      sizes = resolveExplicitParams(sizes, dbAcquire.getLoc());
      auto depDbAcquire = AC->create<DepDbAcquireOp>(
          dbAcquire.getLoc(), dbAcquire.getResult(0).getType(),
          dbAcquire.getResult(1).getType(), depv, baseOffset, indices, offsets,
          sizes, boundsValid);
      /// Replace the uses of the dbAcquire with the depDbAcquire
      dbAcquire.getResult(0).replaceAllUsesWith(depDbAcquire.getResult(0));
      dbAcquire.getResult(1).replaceAllUsesWith(depDbAcquire.getResult(1));
      dbAcquire.erase();
    }

    auto createForwardedDepAcquire = [&](Location userLoc) {
      Type depGuidType = placeholder.getType();
      Type depPtrType = placeholder.getType();
      if (auto sourceAcquire =
              originalDeps[depIndex].getDefiningOp<DbAcquireOp>()) {
        depGuidType = sourceAcquire.getResult(0).getType();
        depPtrType = sourceAcquire.getResult(1).getType();
      } else if (auto sourceAcquire =
                     originalDeps[depIndex].getDefiningOp<DepDbAcquireOp>()) {
        depGuidType = sourceAcquire.getResult(0).getType();
        depPtrType = sourceAcquire.getResult(1).getType();
      }

      SmallVector<Value> forwardedIndices;
      if (auto sourceAcquire =
              originalDeps[depIndex].getDefiningOp<DbAcquireOp>()) {
        forwardedIndices.assign(sourceAcquire.getIndices().begin(),
                                sourceAcquire.getIndices().end());
      } else if (auto sourceAcquire =
                     originalDeps[depIndex].getDefiningOp<DepDbAcquireOp>()) {
        forwardedIndices.assign(sourceAcquire.getIndices().begin(),
                                sourceAcquire.getIndices().end());
      }
      forwardedIndices = resolveExplicitParams(forwardedIndices, loc);
      forwardedIndices =
          normalizeSliceIndices(forwardedIndices, depIndexOffsets,
                                depIndexSizes, depIndicesAlreadySliceRelative);

      return AC->create<DepDbAcquireOp>(userLoc, depGuidType, depPtrType, depv,
                                        baseOffset, forwardedIndices,
                                        depOffsets, depSizes, Value());
    };

    if (!recordDepUses.empty()) {
      for (OpOperand *use : recordDepUses) {
        AC->setInsertionPoint(use->getOwner());
        auto forwardedDep =
            createForwardedDepAcquire(use->getOwner()->getLoc());
        use->set(forwardedDep.getResult(0));
      }
    }

    if (!depHandlePackIndices.empty())
      if (Operation *scope = placeholder.getParentBlock()->getParentOp())
        scope->walk([&](RecordDepOp recordDep) {
          for (OpOperand &operand : recordDep->getOpOperands()) {
            Value operandValue = operand.get();
            if (!isa<BaseMemRefType>(operandValue.getType()))
              continue;
            if (!tracePackedDepHandleIndex(operandValue))
              continue;
            AC->setInsertionPoint(recordDep);
            auto forwardedDep = createForwardedDepAcquire(recordDep.getLoc());
            operand.set(forwardedDep.getResult(0));
          }
        });

    if (!nestedEdtDepUses.empty()) {
      for (OpOperand *use : nestedEdtDepUses) {
        auto childEdt = cast<arts::EdtOp>(use->getOwner());
        AC->setInsertionPoint(childEdt);
        auto forwardedDep = createForwardedDepAcquire(childEdt.getLoc());
        use->set(forwardedDep.getResult(1));
      }
    }

    /// If single element, we can optionally replace the placeholder directly.
    /// Skip the shortcut when the dependency is indexed (db_ref/db_gep/loads)
    /// or when the element type is itself a memref (block-of-blocks), because
    /// we must index depv, not treat a single dep entry as an array.
    bool hasIndexedUsers = llvm::any_of(users, [](Operation *op) {
      return isa<arts::DbRefOp, DbGepOp, memref::LoadOp, memref::StoreOp,
                 polygeist::Memref2PointerOp, polygeist::Pointer2MemrefOp>(op);
    });
    if (isSingleElement && !isNestedMemref && !hasIndexedUsers &&
        preservedDbAcquirePtrUses.empty()) {
      ARTS_DEBUG(" - Rewriting single element placeholder");
      Operation *placeholderOp = placeholder.getDefiningOp();
      AC->setInsertionPoint(placeholderOp);
      auto depGep =
          AC->create<DepGepOp>(loc, AC->llvmPtr, AC->llvmPtr, depv, baseOffset,
                               ValueRange(), ValueRange());
      placeholder.replaceAllUsesWith(depGep.getPtr());
      /// Collect placeholder for later erasure
      placeholdersToErase.push_back(placeholderOp);
      continue;
    }

    Value payloadView;
    auto getPayloadView = [&]() -> Value {
      if (payloadView)
        return payloadView;

      OpBuilder::InsertionGuard payloadIG(AC->getBuilder());
      AC->setInsertionPoint(placeholder.getDefiningOp());

      Type depGuidType = placeholder.getType();
      Type depPtrType = placeholder.getType();
      if (auto sourceAcquire =
              originalDeps[depIndex].getDefiningOp<DbAcquireOp>()) {
        depGuidType = sourceAcquire.getResult(0).getType();
      } else if (auto sourceAcquire =
                     originalDeps[depIndex].getDefiningOp<DepDbAcquireOp>()) {
        depGuidType = sourceAcquire.getResult(0).getType();
      }

      auto depDbAcquire = AC->create<DepDbAcquireOp>(
          loc, depGuidType, depPtrType, depv, baseOffset,
          /*indices=*/SmallVector<Value>{},
          /*offsets=*/SmallVector<Value>{},
          /*sizes=*/SmallVector<Value>{}, Value());
      payloadView = depDbAcquire.getResult(1);
      return payloadView;
    };

    if (!preservedDbAcquirePtrUses.empty())
      for (OpOperand *use : preservedDbAcquirePtrUses)
        use->set(getPayloadView());

    Value invariantDepBasePtr;
    DenseMap<Type, Value> invariantDepMemrefViews;
    auto getInvariantDepBasePtr = [&]() -> Value {
      if (invariantDepBasePtr)
        return invariantDepBasePtr;

      OpBuilder::InsertionGuard payloadIG(AC->getBuilder());
      AC->setInsertionPoint(placeholder.getDefiningOp());
      SmallVector<Value> emptyArgs;
      Value ptrField = AC->create<DepGepOp>(loc, AC->llvmPtr, AC->llvmPtr, depv,
                                            baseOffset, emptyArgs, emptyArgs)
                           .getPtr();
      invariantDepBasePtr =
          AC->create<LLVM::LoadOp>(loc, AC->llvmPtr, ptrField);
      return invariantDepBasePtr;
    };

    auto getInvariantDepMemrefView = [&](Type memrefType) -> Value {
      auto it = invariantDepMemrefViews.find(memrefType);
      if (it != invariantDepMemrefViews.end())
        return it->second;

      OpBuilder::InsertionGuard payloadIG(AC->getBuilder());
      AC->setInsertionPoint(placeholder.getDefiningOp());
      Value view = AC->create<polygeist::Pointer2MemrefOp>(
          loc, memrefType, getInvariantDepBasePtr());
      invariantDepMemrefViews[memrefType] = view;
      return view;
    };

    /// If not, for each user of the dependency placeholder, rewrite the
    /// operation
    ARTS_DEBUG(" - Rewriting " << users.size() << " users");
    for (Operation *op : users) {
      if (auto ptr2Memref = dyn_cast<polygeist::Pointer2MemrefOp>(op)) {
        AC->setInsertionPoint(op);
        if (useInvariantSingleDepView) {
          ptr2Memref.getResult().replaceAllUsesWith(
              getInvariantDepMemrefView(ptr2Memref.getType()));
          ptr2Memref.erase();
          continue;
        }
        Value basePtr;
        if (usePayloadIndexing) {
          SmallVector<Value> emptyArgs;
          basePtr = AC->create<DbGepOp>(loc, AC->llvmPtr, getPayloadView(),
                                        emptyArgs, emptyArgs)
                        .getPtr();
        } else {
          SmallVector<Value> emptyArgs;
          Value ptrField =
              AC->create<DepGepOp>(loc, AC->llvmPtr, AC->llvmPtr, depv,
                                   baseOffset, emptyArgs, emptyArgs)
                  .getPtr();
          basePtr = AC->create<LLVM::LoadOp>(loc, AC->llvmPtr, ptrField);
        }
        auto newMemref = AC->create<polygeist::Pointer2MemrefOp>(
            loc, ptr2Memref.getType(), basePtr);
        ptr2Memref.getResult().replaceAllUsesWith(newMemref.getResult());
        ptr2Memref.erase();
      } else if (auto mp = dyn_cast<polygeist::Memref2PointerOp>(op)) {
        AC->setInsertionPoint(op);
        if (useInvariantSingleDepView) {
          op->getResult(0).replaceAllUsesWith(getInvariantDepBasePtr());
          op->erase();
          continue;
        }
        if (usePayloadIndexing) {
          SmallVector<Value> emptyArgs;
          auto payloadGep = AC->create<DbGepOp>(
              loc, AC->llvmPtr, getPayloadView(), emptyArgs, emptyArgs);
          op->getResult(0).replaceAllUsesWith(payloadGep.getPtr());
        } else {
          SmallVector<Value> emptyArgs;
          auto depGep =
              AC->create<DepGepOp>(loc, AC->llvmPtr, AC->llvmPtr, depv,
                                   baseOffset, emptyArgs, emptyArgs);
          op->getResult(0).replaceAllUsesWith(depGep.getPtr());
        }
        op->erase();
      } else if (auto dbGep = dyn_cast<DbGepOp>(op)) {
        AC->setInsertionPoint(op);
        SmallVector<Value> dbGepIndices(dbGep.getIndices().begin(),
                                        dbGep.getIndices().end());
        dbGepIndices = resolveParam(dbGepIndices, dbGep.getLoc());
        dbGepIndices =
            normalizeSliceIndices(dbGepIndices, depIndexOffsets, depIndexSizes,
                                  depIndicesAlreadySliceRelative);
        if (usePayloadIndexing) {
          auto payloadGep = AC->create<DbGepOp>(
              loc, AC->llvmPtr, getPayloadView(), dbGepIndices, accessStrides);
          op->getResult(0).replaceAllUsesWith(payloadGep.getPtr());
        } else {
          auto depGep =
              AC->create<DepGepOp>(loc, AC->llvmPtr, AC->llvmPtr, depv,
                                   baseOffset, dbGepIndices, accessStrides);
          op->getResult(0).replaceAllUsesWith(depGep.getPtr());
        }
        op->erase();
      } else if (auto dbRef = dyn_cast<arts::DbRefOp>(op)) {
        AC->setInsertionPoint(op);
        if (useInvariantSingleDepView) {
          Value invariantView = getInvariantDepMemrefView(dbRef.getType());

          auto resultType = dyn_cast<MemRefType>(dbRef.getType());
          bool needsDynOps = false;
          if (resultType && !allocElementSizes.empty()) {
            unsigned numDynDims =
                llvm::count_if(resultType.getShape(), [](int64_t d) {
                  return d == ShapedType::kDynamic;
                });
            needsDynOps = numDynDims > 1;
          }

          if (needsDynOps) {
            SmallVector<Value> resolvedElemSizes =
                resolveParam(allocElementSizes, dbRef.getLoc());
            for (auto &use :
                 llvm::make_early_inc_range(dbRef.getResult().getUses())) {
              Operation *userOp = use.getOwner();
              AC->setInsertionPoint(userOp);
              if (auto loadOp = dyn_cast<memref::LoadOp>(userOp)) {
                SmallVector<Value> indices(loadOp.getIndices().begin(),
                                           loadOp.getIndices().end());
                auto dynLoad = AC->create<polygeist::DynLoadOp>(
                    loadOp.getLoc(), loadOp.getResult().getType(),
                    invariantView, indices, resolvedElemSizes);
                loadOp.getResult().replaceAllUsesWith(dynLoad.getResult());
                loadOp.erase();
              } else if (auto storeOp = dyn_cast<memref::StoreOp>(userOp)) {
                SmallVector<Value> indices(storeOp.getIndices().begin(),
                                           storeOp.getIndices().end());
                AC->create<polygeist::DynStoreOp>(
                    storeOp.getLoc(), storeOp.getValueToStore(), invariantView,
                    indices, resolvedElemSizes);
                storeOp.erase();
              } else {
                use.set(invariantView);
              }
            }
          } else {
            dbRef.getResult().replaceAllUsesWith(invariantView);
          }
          op->erase();
          continue;
        }
        SmallVector<Value> refIndices(dbRef.getIndices().begin(),
                                      dbRef.getIndices().end());
        refIndices = resolveParam(refIndices, dbRef.getLoc());
        refIndices =
            normalizeSliceIndices(refIndices, depIndexOffsets, depIndexSizes,
                                  depIndicesAlreadySliceRelative);
        Value basePtr;
        if (usePayloadIndexing) {
          basePtr = AC->create<DbGepOp>(loc, AC->llvmPtr, getPayloadView(),
                                        refIndices, accessStrides)
                        .getPtr();
        } else {
          Value ptrField =
              AC->create<DepGepOp>(loc, AC->llvmPtr, AC->llvmPtr, depv,
                                   baseOffset, refIndices, accessStrides)
                  .getPtr();
          basePtr = AC->create<LLVM::LoadOp>(loc, AC->llvmPtr, ptrField);
        }
        auto newMemref = AC->create<polygeist::Pointer2MemrefOp>(
            loc, dbRef.getType(), basePtr);

        /// Rewrite memref.load/store users of DbRefOp to DynLoadOp/DynStoreOp
        /// when the result type has multiple dynamic dims. Without this,
        /// ConvertPolygeistToLLVM crashes because CLoadOpLowering cannot handle
        /// multi-dynamic-dim memrefs via Pointer2MemrefOp.
        auto resultType = dyn_cast<MemRefType>(dbRef.getType());
        bool needsDynOps = false;
        if (resultType && !allocElementSizes.empty()) {
          unsigned numDynDims =
              llvm::count_if(resultType.getShape(), [](int64_t d) {
                return d == ShapedType::kDynamic;
              });
          needsDynOps = numDynDims > 1;
        }

        if (needsDynOps) {
          SmallVector<Value> resolvedElemSizes =
              resolveParam(allocElementSizes, dbRef.getLoc());
          for (auto &use :
               llvm::make_early_inc_range(dbRef.getResult().getUses())) {
            Operation *userOp = use.getOwner();
            AC->setInsertionPoint(userOp);
            if (auto loadOp = dyn_cast<memref::LoadOp>(userOp)) {
              SmallVector<Value> indices(loadOp.getIndices().begin(),
                                         loadOp.getIndices().end());
              auto dynLoad = AC->create<polygeist::DynLoadOp>(
                  loadOp.getLoc(), loadOp.getResult().getType(), newMemref,
                  indices, resolvedElemSizes);
              loadOp.getResult().replaceAllUsesWith(dynLoad.getResult());
              loadOp.erase();
            } else if (auto storeOp = dyn_cast<memref::StoreOp>(userOp)) {
              SmallVector<Value> indices(storeOp.getIndices().begin(),
                                         storeOp.getIndices().end());
              AC->create<polygeist::DynStoreOp>(
                  storeOp.getLoc(), storeOp.getValueToStore(), newMemref,
                  indices, resolvedElemSizes);
              storeOp.erase();
            } else {
              use.set(newMemref.getResult());
            }
          }
        } else {
          dbRef.getResult().replaceAllUsesWith(newMemref.getResult());
        }
        op->erase();
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        AC->setInsertionPoint(op);
        SmallVector<Value> storeIndices(store.getIndices().begin(),
                                        store.getIndices().end());
        storeIndices = resolveParam(storeIndices, store.getLoc());
        storeIndices =
            normalizeSliceIndices(storeIndices, depIndexOffsets, depIndexSizes,
                                  depIndicesAlreadySliceRelative);
        if (usePayloadIndexing) {
          auto payloadGep = AC->create<DbGepOp>(
              loc, AC->llvmPtr, getPayloadView(), storeIndices, accessStrides);
          AC->create<LLVM::StoreOp>(loc, store.getValue(), payloadGep.getPtr());
        } else {
          auto depGep =
              AC->create<DepGepOp>(loc, AC->llvmPtr, AC->llvmPtr, depv,
                                   baseOffset, storeIndices, accessStrides);
          AC->create<LLVM::StoreOp>(loc, store.getValue(), depGep.getPtr());
        }
        op->erase();
      } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
        AC->setInsertionPoint(op);
        SmallVector<Value> loadIndices(load.getIndices().begin(),
                                       load.getIndices().end());
        loadIndices = resolveParam(loadIndices, load.getLoc());
        loadIndices =
            normalizeSliceIndices(loadIndices, depIndexOffsets, depIndexSizes,
                                  depIndicesAlreadySliceRelative);
        Value loaded;
        if (usePayloadIndexing) {
          auto payloadGep = AC->create<DbGepOp>(
              loc, AC->llvmPtr, getPayloadView(), loadIndices, accessStrides);
          loaded = AC->create<LLVM::LoadOp>(loc, load.getType(),
                                            payloadGep.getPtr());
        } else {
          auto depGep =
              AC->create<DepGepOp>(loc, AC->llvmPtr, AC->llvmPtr, depv,
                                   baseOffset, loadIndices, accessStrides);
          loaded =
              AC->create<LLVM::LoadOp>(loc, load.getType(), depGep.getPtr());
        }
        op->getResult(0).replaceAllUsesWith(loaded);
        op->erase();
      } else if (auto release = dyn_cast<DbReleaseOp>(op)) {
        /// DbReleaseOp using placeholder can be safely erased
        /// Dep management is now handled through depv
        op->erase();
      }
    }

    /// Collect placeholder for later erasure
    if (auto defOp = placeholder.getDefiningOp())
      placeholdersToErase.push_back(defOp);
  }

  /// Erase all placeholders at the end
  for (Operation *op : placeholdersToErase) {
    /// Replace any remaining placeholder uses with undef before erasing
    if (op->getNumResults() > 0 && !op->getResult(0).use_empty()) {
      OpBuilder::InsertionGuard IG(AC->getBuilder());
      AC->setInsertionPoint(op);
      auto newUndef =
          AC->create<UndefOp>(op->getLoc(), op->getResult(0).getType());
      op->getResult(0).replaceAllUsesWith(newUndef);
    }
    op->erase();
  }
}

///===----------------------------------------------------------------------===///
/// Pass Registration
///===----------------------------------------------------------------------===///

namespace mlir {
namespace arts {

std::unique_ptr<Pass> createEdtLoweringPass(uint64_t idStride) {
  return std::make_unique<EdtLoweringPass>(nullptr, idStride);
}

std::unique_ptr<Pass> createEdtLoweringPass(AnalysisManager *AM,
                                            uint64_t idStride) {
  return std::make_unique<EdtLoweringPass>(AM, idStride);
}

} // namespace arts
} // namespace mlir
