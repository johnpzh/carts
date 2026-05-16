///==========================================================================///
/// File: CodirToArts.cpp
///
/// Materializes CODIR codelets as abstract ARTS DB/EDT objects.
///==========================================================================///
#include "../ConversionUtils.h"
namespace mlir::carts::codir {
#define GEN_PASS_DEF_CONVERTCODIRTOARTS
#include "carts/dialect/codir/Conversion/Passes.h.inc"
} // namespace mlir::carts::codir
namespace {
struct ArtsLaunchPolicy {
  arts::EdtConcurrency concurrency = arts::EdtConcurrency::intranode;
  Value route;
};

struct ConvertCodirToArtsPass
    : public codir::impl::ConvertCodirToArtsBase<
          ConvertCodirToArtsPass> {
  llvm::SmallDenseSet<Operation *, 16> loopCompletionBarriers;

  bool hasArtsInterNodeRuntime(ModuleOp module) const {
    std::optional<int64_t> totalNodes = arts::getRuntimeTotalNodes(module);
    return totalNodes && *totalNodes > 1;
  }

  bool hasGenericWorkerPlan(codir::CodeletOp codelet) const {
    if (!codelet)
      return false;
    return codelet.getDistributionKindAttr() ||
           codelet.getIterationTopologyAttr() ||
           codelet.getLogicalWorkerSliceAttr() ||
           codelet.getPhysicalBlockShapeAttr();
  }

  bool hasDistributedLaunchStoragePlan(codir::CodeletOp codelet) const {
    if (!hasGenericWorkerPlan(codelet))
      return false;
    if (codelet.getDeps().empty())
      return true;
    return hasCodirPhysicalOwnerSlicePlan(codelet);
  }

  scf::ForOp findGenericWorkerDispatchLoop(codir::CodeletOp codelet) const {
    for (Operation *parent = codelet ? codelet->getParentOp() : nullptr; parent;
         parent = parent->getParentOp()) {
      auto loop = dyn_cast<scf::ForOp>(parent);
      if (loop && containsValue(codelet.getParams(), loop.getInductionVar()))
        return loop;
    }
    return {};
  }

  ArtsLaunchPolicy resolveArtsLaunchPolicy(codir::CodeletOp codelet,
                                           OpBuilder &builder,
                                           Location loc) const {
    ArtsLaunchPolicy policy;
    ModuleOp module = codelet ? codelet->getParentOfType<ModuleOp>() : nullptr;
    if (!module || !hasArtsInterNodeRuntime(module) ||
        !hasDistributedLaunchStoragePlan(codelet))
      return policy;

    scf::ForOp loop = findGenericWorkerDispatchLoop(codelet);
    if (!loop)
      return policy;

    policy.concurrency = arts::EdtConcurrency::internode;
    Value relative =
        arith::SubIOp::create(builder, loc, loop.getInductionVar(),
                              loop.getLowerBound());
    Value ordinal =
        arith::DivUIOp::create(builder, loc, relative, loop.getStep());
    Value ordinalI32 =
        arith::IndexCastOp::create(builder, loc, builder.getI32Type(), ordinal);
    auto totalNodes =
        arts::RuntimeQueryOp::create(builder, loc,
                                     arts::RuntimeQueryKind::totalNodes);
    policy.route = arith::RemUIOp::create(builder, loc, ordinalI32,
                                          totalNodes.getResult());
    return policy;
  }

  Operation *getCompletionBarrierAnchor(codir::CodeletOp codelet,
                                        arts::EdtOp task) {
    Operation *nearestLoop = nullptr;
    Operation *dispatchAnchor = nullptr;
    bool matchedDispatchLoop = false;

    for (Operation *parent = task->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto loop = dyn_cast<scf::ForOp>(parent);
      if (!loop) {
        if (matchedDispatchLoop)
          break;
        continue;
      }

      if (!nearestLoop)
        nearestLoop = parent;

      if (containsValue(codelet.getParams(), loop.getInductionVar())) {
        dispatchAnchor = parent;
        matchedDispatchLoop = true;
        continue;
      }

      if (matchedDispatchLoop)
        break;
    }

    if (dispatchAnchor)
      return dispatchAnchor;
    if (nearestLoop)
      return nearestLoop;
    return task.getOperation();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<sde::SdeMuDataOp> muDatas;
    module.walk([&](sde::SdeMuDataOp op) { muDatas.push_back(op); });
    for (sde::SdeMuDataOp op : muDatas) {
      if (failed(lowerMuData(op))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<sde::SdeMuAllocOp> muAllocs;
    module.walk([&](sde::SdeMuAllocOp op) { muAllocs.push_back(op); });
    for (sde::SdeMuAllocOp op : muAllocs) {
      if (failed(lowerMuAlloc(op))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<sde::SdeResourceQueryOp> resourceQueries;
    module.walk([&](sde::SdeResourceQueryOp op) {
      resourceQueries.push_back(op);
    });
    for (sde::SdeResourceQueryOp op : resourceQueries) {
      if (failed(lowerSdeResourceQuery(op))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<codir::CodeletOp> codelets;
    module.walk([&](codir::CodeletOp op) { codelets.push_back(op); });
    for (codir::CodeletOp codelet : codelets) {
      for (Value dep : codelet.getDeps()) {
        if (findBackingDbAlloc(dep))
          continue;
        if (failed(materializeRawCodirDependency(dep, codelet))) {
          codelet.emitOpError()
              << "dependency is not backed by SDE/CODIR DB materialization "
                 "and cannot be materialized from a local memref allocation";
          signalPassFailure();
          return;
        }
      }
    }

    for (codir::CodeletOp codelet : codelets) {
      if (failed(lowerCodelet(codelet))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<sde::SdeSuBarrierOp> controlBarriers;
    module.walk([&](sde::SdeSuBarrierOp op) { controlBarriers.push_back(op); });
    for (sde::SdeSuBarrierOp op : controlBarriers) {
      if (failed(lowerSdeControlBarrier(op))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<sde::SdeControlTokenOp> controlTokens;
    module.walk([&](sde::SdeControlTokenOp op) { controlTokens.push_back(op); });
    for (sde::SdeControlTokenOp op : controlTokens) {
      if (failed(eraseConsumedSdeControlToken(op))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<sde::SdeMuTokenOp> tokens;
    module.walk([&](sde::SdeMuTokenOp op) { tokens.push_back(op); });
    for (sde::SdeMuTokenOp token : tokens) {
      if (!token.getToken().use_empty()) {
        token.emitOpError()
            << "survived CODIR-to-ARTS materialization; run "
               "`convert-sde-to-codir` before `convert-codir-to-arts` so SDE "
               "codelets become CODIR codelets before ARTS lowering";
        signalPassFailure();
        return;
      }
      token.erase();
    }
  }

  LogicalResult lowerCodelet(codir::CodeletOp codelet) {
    Location loc = codelet.getLoc();
    OpBuilder builder(codelet);

    ArrayAttr depModes = codelet.getDepModesAttr();
    if (codelet.getDeps().size() != (depModes ? depModes.size() : 0))
      return codelet.emitOpError()
             << "requires one dep_modes entry per dependency before "
                "CODIR-to-ARTS materialization";

    SmallVector<Value> taskDeps;
    SmallVector<Type> blockArgTypes;
    SmallVector<CodirDepSlice, 4> depSlices;
    SmallVector<Operation *, 4> depViewCleanup;
    taskDeps.reserve(codelet.getDeps().size());
    blockArgTypes.reserve(codelet.getDeps().size());
    depSlices.reserve(codelet.getDeps().size());

    for (auto [idx, dep] : llvm::enumerate(codelet.getDeps())) {
      if (isCodirViewDep(dep))
        depViewCleanup.push_back(dep.getDefiningOp());

      auto modeAttr = cast<codir::CodirAccessModeAttr>(depModes[idx]);
      arts::DbAllocOp alloc = findBackingDbAlloc(dep);
      if (!alloc)
        return codelet.emitOpError()
               << "dependency #" << idx
               << " is not backed by SDE/CODIR DB materialization";

      CodirDepSlice slice = getCodirDepSlice(dep, builder, loc);
      std::optional<arts::PartitionMode> partitionMode;
      SmallVector<Value> partitionOffsets;
      SmallVector<Value> partitionSizes;
      if (slice.sliced) {
        partitionMode = arts::PartitionMode::block;
        partitionOffsets.assign(slice.offsets.begin(), slice.offsets.end());
        partitionSizes.assign(slice.sizes.begin(), slice.sizes.end());
      }

      SmallVector<Value> dbOffsets{createZeroIndex(builder, loc)};
      SmallVector<Value> dbSizes{createOneIndex(builder, loc)};
      if (canUseCodirOwnerSliceForAlloc(codelet, alloc) &&
          !codelet.getParams().empty()) {
        if (std::optional<int64_t> blockSize =
                getFirstPositiveI64(codelet.getPhysicalBlockShapeAttr())) {
          Value blockSizeValue = createConstantIndex(builder, loc, *blockSize);
          Value base = codelet.getParams().back();
          Value blockIndex =
              arith::DivUIOp::create(builder, loc, base, blockSizeValue);
          dbOffsets.assign({blockIndex});
          dbSizes.assign({createOneIndex(builder, loc)});
        }
      }
      auto acquire = arts::DbAcquireOp::create(
          builder, loc, convertAccessMode(modeAttr.getValue()),
          alloc.getGuid(), alloc.getPtr(), partitionMode,
          /*indices=*/SmallVector<Value>{}, std::move(dbOffsets),
          std::move(dbSizes),
          /*partitionIndices=*/SmallVector<Value>{},
          std::move(partitionOffsets), std::move(partitionSizes),
          /*boundsValid=*/Value{},
          /*elementOffsets=*/SmallVector<Value>{},
          /*elementSizes=*/SmallVector<Value>{});
      taskDeps.push_back(acquire.getPtr());
      blockArgTypes.push_back(acquire.getPtr().getType());
      depSlices.push_back(std::move(slice));
    }

    SmallVector<Value> taskParams(codelet.getParams().begin(),
                                  codelet.getParams().end());
    SmallVector<Value> codeletDeps(codelet.getDeps().begin(),
                                   codelet.getDeps().end());
    appendDynamicCodirDepSliceParams(codeletDeps, taskParams);
    for (Value dep : codelet.getDeps()) {
      arts::DbAllocOp alloc = findBackingDbAlloc(dep);
      if (!alloc)
        continue;
      for (Value elementSize : alloc.getElementSizes()) {
        if (!isCodirScalarParamType(elementSize.getType()) ||
            getConstantIndexValue(elementSize) ||
            containsValue(taskParams, elementSize))
          continue;
        taskParams.push_back(elementSize);
      }
    }
    // CODIR carries only generic worker-plan facts. The ARTS boundary is the
    // first place where runtime topology can turn that plan into inter-node
    // EDT placement and routing.
    ArtsLaunchPolicy launch =
        resolveArtsLaunchPolicy(codelet, builder, loc);
    auto task = launch.route
                    ? arts::EdtOp::create(builder, loc, arts::EdtType::task,
                                          launch.concurrency, launch.route,
                                          taskDeps, taskParams)
                    : arts::EdtOp::create(builder, loc, arts::EdtType::task,
                                          launch.concurrency, taskDeps,
                                          taskParams);
    propagateCodirPlanToArts(codelet, task);
    bool isTaskDepend = static_cast<bool>(codelet.getTaskDependAttr());
    bool requiresOrderedDependBarrier =
        static_cast<bool>(codelet.getOrderedTaskDependAttr());
    bool requiresCompletionBarrier =
        static_cast<bool>(codelet.getCompletionBarrierAttr());
    Block &taskBlock = task.getBody().front();
    for (Type type : blockArgTypes)
      taskBlock.addArgument(type, loc);
    DenseMap<Value, Value> paramBlockArgs;
    DenseMap<Value, Value> sourceByBlockArgument;
    for (auto [idx, param] : llvm::enumerate(taskParams)) {
      Value arg = taskBlock.addArgument(param.getType(), loc);
      paramBlockArgs.try_emplace(param, arg);
      sourceByBlockArgument.try_emplace(arg, param);
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&taskBlock);

    IRMapping mapper;
    Block &codeletBlock = codelet.getBody().front();
    unsigned numDeps = codelet.getDeps().size();
    for (unsigned idx = 0; idx < numDeps; ++idx) {
      Value payload = materializeInnerPayload(builder, loc,
                                              taskBlock.getArgument(idx));
      const CodirDepSlice &slice = depSlices[idx];
      if (slice.sliced) {
        auto depType = cast<MemRefType>(codelet.getDeps()[idx].getType());
        if (slice.subindex) {
          Value subindex = slice.subindexIndex;
          auto it = paramBlockArgs.find(subindex);
          if (it != paramBlockArgs.end())
            subindex = it->second;
          else if (std::optional<int64_t> constant =
                       getConstantIndexValue(subindex))
            subindex = createConstantIndex(builder, loc, *constant);
          payload =
              polygeist::SubIndexOp::create(builder, loc, depType, payload,
                                            subindex);
        } else {
          SmallVector<OpFoldResult> offsets =
              remapIndexFoldResults(builder, loc, slice.mixedOffsets,
                                    paramBlockArgs);
          SmallVector<OpFoldResult> sizes =
              remapIndexFoldResults(builder, loc, slice.mixedSizes,
                                    paramBlockArgs);
          SmallVector<OpFoldResult> strides =
              remapIndexFoldResults(builder, loc, slice.mixedStrides,
                                    paramBlockArgs);
          auto resultType = memref::SubViewOp::inferResultType(
              cast<MemRefType>(payload.getType()), offsets, sizes, strides);
          payload = memref::SubViewOp::create(
              builder, loc, resultType, payload, offsets, sizes, strides);
        }
      }
      mapper.map(codeletBlock.getArgument(idx), payload);
    }

    for (auto [idx, param] : llvm::enumerate(codelet.getParams()))
      mapper.map(codeletBlock.getArgument(numDeps + idx),
                 taskBlock.getArgument(numDeps + idx));
    for (Operation &nested : codeletBlock.without_terminator())
      builder.insert(nested.clone(mapper));

    if (isReductionCodelet(codelet))
      lowerIntegerAddReductionsToAtomics(task.getBody(), sourceByBlockArgument);

    arts::YieldOp::create(builder, loc);

    if (requiresOrderedDependBarrier) {
      OpBuilder barrierBuilder(task);
      barrierBuilder.setInsertionPointAfter(task);
      auto reason = arts::ArtsBarrierReasonAttr::get(
          codelet.getContext(), arts::ArtsBarrierReason::required_memory);
      arts::BarrierOp::create(barrierBuilder, loc, reason);
    } else if (isTaskDepend || requiresCompletionBarrier) {
      Operation *barrierAnchor = getCompletionBarrierAnchor(codelet, task);

      if (barrierAnchor == task.getOperation() ||
          loopCompletionBarriers.insert(barrierAnchor).second) {
        OpBuilder barrierBuilder(barrierAnchor);
        barrierBuilder.setInsertionPointAfter(barrierAnchor);
        auto reason = arts::ArtsBarrierReasonAttr::get(
            codelet.getContext(), arts::ArtsBarrierReason::required_memory);
        arts::BarrierOp::create(barrierBuilder, loc, reason);
      }
    }

    codelet.erase();
    for (Operation *view : depViewCleanup)
      if (view && view->use_empty())
        view->erase();
    return success();
  }
};
} // namespace
std::unique_ptr<Pass> mlir::carts::codir::createConvertCodirToArtsPass() {
  return std::make_unique<ConvertCodirToArtsPass>();
}
