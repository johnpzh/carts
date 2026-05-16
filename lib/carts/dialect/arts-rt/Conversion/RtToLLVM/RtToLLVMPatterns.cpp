///==========================================================================///
/// File: RtToLLVMPatterns.cpp
///
/// ARTS runtime dialect (arts_rt) to LLVM conversion patterns.
/// These patterns convert arts_rt ops into LLVM runtime calls via the
/// ArtsCodegen infrastructure shared with residual ARTS/ARTS-RT-to-LLVM
/// conversion.
///==========================================================================///

#include "carts/dialect/arts-rt/Conversion/ArtsToLLVM/ConvertArtsToLLVMInternal.h"

#include "carts/dialect/arts-rt/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/LoweringContractUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/dialect/arts/Utils/PartitionPredicates.h"
#include "carts/utils/StencilAttributes.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

#include "carts/utils/Debug.h"
#include "llvm/ADT/Statistic.h"
ARTS_DEBUG_SETUP(convert_arts_to_llvm);

static llvm::Statistic numEdtOpsConverted{
    "rt_to_llvm", "NumEdtOpsConverted",
    "Number of EDT operations converted to LLVM runtime calls"};
static llvm::Statistic numEpochOpsConverted{
    "rt_to_llvm", "NumEpochOpsConverted",
    "Number of epoch operations converted to LLVM runtime calls"};
static llvm::Statistic numDepOpsConverted{
    "rt_to_llvm", "NumDepOpsConverted",
    "Number of dependency operations converted to LLVM runtime calls"};
static llvm::Statistic numDbOpsConverted{
    "rt_to_llvm", "NumDbOpsConverted",
    "Number of DataBlock operations converted to LLVM runtime calls"};
static llvm::Statistic numMiscOpsConverted{
    "rt_to_llvm", "NumMiscOpsConverted",
    "Number of miscellaneous ARTS operations converted"};

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
using namespace mlir::carts::arts::convert_arts_to_llvm;
using namespace mlir::carts::arts_rt;

///===----------------------------------------------------------------------===///
/// Epoch Patterns
///===----------------------------------------------------------------------===///

struct CreateEpochPattern : public ArtsToLLVMPattern<CreateEpochOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(CreateEpochOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering CreateEpoch Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto loc = op.getLoc();

    Value guid, edtSlot;
    if (op.hasFinishTarget()) {
      guid = op.getFinishEdtGuid();
      edtSlot = op.getFinishSlot();
    } else {
      guid = AC->createIntConstant(0, AC->Int64, loc);
      edtSlot = AC->createIntConstant(DEFAULT_EDT_SLOT, AC->Int32, loc);
    }

    /// Create epoch guid
    Value epochGuid;
    if (op->hasAttr(AttrNames::Operation::NoStartEpoch))
      epochGuid = AC->createEpochNoStart(guid, edtSlot, loc);
    else
      epochGuid = AC->createEpoch(guid, edtSlot, loc);
    rewriter.replaceOp(op, epochGuid);
    ++numEpochOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.wait_on_epoch operations
struct WaitOnEpochPattern : public ArtsToLLVMPattern<WaitOnEpochOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(WaitOnEpochOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering WaitOnEpoch Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    AC->waitOnHandle(op.getEpochGuid(), op.getLoc());
    rewriter.eraseOp(op);
    ++numEpochOpsConverted;
    return success();
  }
};

///===----------------------------------------------------------------------===///
/// Dependency Patterns
///===----------------------------------------------------------------------===///

/// Pattern to convert arts.record_dep operations
struct RecordDepPattern : public ArtsToLLVMPattern<RecordDepOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(RecordDepOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering RecordInDep Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto edtGuid = op.getEdtGuid();
    auto loc = op.getLoc();
    auto readyLocalSite = resolveReadyLocalLaunchSite(edtGuid);
    const bool useReadyLocalLaunch =
        readyLocalSite && canUseReadyLocalLaunchSite(*readyLocalSite);

    /// Get access mode from attribute
    auto accessMode = op.getAccessMode();
    auto acquireModesAttr = op.getAcquireModes();
    ArrayRef<int32_t> acquireModeValues =
        acquireModesAttr ? *acquireModesAttr : ArrayRef<int32_t>{};
    auto depFlagsAttr = op.getDepFlags();
    ArrayRef<int32_t> depFlagValues =
        depFlagsAttr ? *depFlagsAttr : ArrayRef<int32_t>{};

    /// Get bounds validity flags for stencil boundary guarding
    auto boundsValids = op.getBoundsValids();

    /// Get ESD byte_offsets and byte_sizes for halo dependencies
    auto byteOffsets = op.getByteOffsets();
    auto byteSizes = op.getByteSizes();

    /// Create shared slot counter for all dependencies
    auto slotTy = MemRefType::get({}, AC->Int32);
    Value sharedSlotAlloc = AC->create<memref::AllocaOp>(loc, slotTy);
    Value zeroI32 = AC->createIntConstant(0, AC->Int32, loc);
    AC->create<memref::StoreOp>(loc, zeroI32, sharedSlotAlloc);
    Value readyLocalDepBuffer;
    if (useReadyLocalLaunch) {
      Value depCount = readyLocalSite->representativeCreate().getDepCount();
      if (depCount.getType() != AC->Int32)
        depCount = AC->castToInt(AC->Int32, depCount, loc);
      readyLocalDepBuffer = allocateReadyLocalDepBuffer(depCount, loc);
    }

    /// Add dependencies for each datablock using shared slot counter
    unsigned dbIdx = 0;
    for (Value dbGuid : op.getDatablocks()) {
      std::optional<int32_t> acquireMode = std::nullopt;
      if (dbIdx < acquireModeValues.size())
        acquireMode = acquireModeValues[dbIdx];
      std::optional<int32_t> depFlags = std::nullopt;
      if (dbIdx < depFlagValues.size())
        depFlags = depFlagValues[dbIdx];
      /// Get bounds_valid for this dependency if available
      Value boundsValid =
          (dbIdx < boundsValids.size()) ? boundsValids[dbIdx] : Value();
      /// Get ESD byte offset/size for this dependency if available
      Value byteOffset =
          (dbIdx < byteOffsets.size()) ? byteOffsets[dbIdx] : Value();
      Value byteSize = (dbIdx < byteSizes.size()) ? byteSizes[dbIdx] : Value();
      recordDepsForDb(dbGuid, edtGuid, sharedSlotAlloc, accessMode, acquireMode,
                      depFlags, boundsValid, byteOffset, byteSize,
                      readyLocalDepBuffer, loc);
      ++dbIdx;
    }

    if (useReadyLocalLaunch) {
      if (readyLocalSite->isMergedIfResult()) {
        Value launchGuid = emitMergedIfReadyLocalLaunch(
            *readyLocalSite, readyLocalDepBuffer, loc);
        readyLocalSite->ifOp.getResult(readyLocalSite->ifResultIndex)
            .replaceAllUsesWith(launchGuid);
        if (readyLocalSite->ifOp->use_empty())
          rewriter.eraseOp(readyLocalSite->ifOp);
      } else {
        auto readyLocalCreate = readyLocalSite->representativeCreate();
        func::CallOp launchCall =
            emitReadyLocalLaunch(readyLocalCreate, readyLocalDepBuffer, loc);
        readyLocalCreate.replaceAllUsesWith(launchCall.getResult(0));
        if (readyLocalCreate->use_empty())
          rewriter.eraseOp(readyLocalCreate);
      }
      rewriter.eraseOp(op);
      ++numDepOpsConverted;
      return success();
    }

    rewriter.eraseOp(op);
    ++numDepOpsConverted;
    return success();
  }

private:
  struct ReadyLocalLaunchSite {
    SmallVector<EdtCreateOp, 2> creates;
    scf::IfOp ifOp;
    unsigned ifResultIndex = 0;

    bool isValid() const { return !creates.empty(); }
    bool isMergedIfResult() const { return static_cast<bool>(ifOp); }
    EdtCreateOp representativeCreate() const { return creates.front(); }
  };

  std::optional<ReadyLocalLaunchSite>
  resolveReadyLocalLaunchSite(Value edtGuid) const {
    if (auto create = edtGuid.getDefiningOp<EdtCreateOp>()) {
      ReadyLocalLaunchSite site;
      site.creates.push_back(create);
      return site;
    }

    auto ifOp = edtGuid.getDefiningOp<scf::IfOp>();
    if (!ifOp)
      return std::nullopt;

    auto result = dyn_cast<OpResult>(edtGuid);
    if (!result)
      return std::nullopt;

    unsigned resultNumber = result.getResultNumber();
    auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    if (!thenYield || !elseYield)
      return std::nullopt;
    if (resultNumber >= thenYield.getNumOperands() ||
        resultNumber >= elseYield.getNumOperands())
      return std::nullopt;

    auto thenCreate =
        thenYield.getOperand(resultNumber).getDefiningOp<EdtCreateOp>();
    auto elseCreate =
        elseYield.getOperand(resultNumber).getDefiningOp<EdtCreateOp>();
    if (!thenCreate || !elseCreate)
      return std::nullopt;

    ReadyLocalLaunchSite site;
    site.creates.push_back(thenCreate);
    site.creates.push_back(elseCreate);
    site.ifOp = ifOp;
    site.ifResultIndex = resultNumber;
    return site;
  }

  bool canUseReadyLocalLaunchSite(ReadyLocalLaunchSite &site) const {
    if (!site.isValid())
      return false;
    if (site.isMergedIfResult() && site.ifOp->getNumResults() != 1)
      return false;

    auto reference = site.representativeCreate();
    if (!reference->hasAttr(AttrNames::Operation::ReadyLocalLaunch) ||
        !reference.getEpochGuid())
      return false;

    for (EdtCreateOp create : site.creates) {
      if (!create->hasAttr(AttrNames::Operation::ReadyLocalLaunch) ||
          !create.getEpochGuid())
        return false;
      if (create.getParamMemref() != reference.getParamMemref() ||
          create.getDepCount() != reference.getDepCount() ||
          create.getEpochGuid() != reference.getEpochGuid() ||
          create.getRoute() != reference.getRoute())
        return false;
    }
    return true;
  }

  Value emitMergedIfReadyLocalLaunch(ReadyLocalLaunchSite &site,
                                     Value depBuffer, Location loc) const {
    auto launchIf = AC->create<scf::IfOp>(loc, site.ifOp.getResultTypes(),
                                          site.ifOp.getCondition(),
                                          /*withElseRegion=*/true);

    AC->setInsertionPointToStart(&launchIf.getThenRegion().front());
    func::CallOp thenLaunch = emitReadyLocalLaunch(
        site.creates.front(), depBuffer, site.creates.front().getLoc());
    AC->create<scf::YieldOp>(site.creates.front().getLoc(),
                             thenLaunch.getResult(0));

    AC->setInsertionPointToStart(&launchIf.getElseRegion().front());
    func::CallOp elseLaunch = emitReadyLocalLaunch(
        site.creates.back(), depBuffer, site.creates.back().getLoc());
    AC->create<scf::YieldOp>(site.creates.back().getLoc(),
                             elseLaunch.getResult(0));

    AC->setInsertionPointAfter(launchIf);
    return launchIf.getResult(0);
  }

  func::CallOp emitReadyLocalLaunch(EdtCreateOp op, Value depBuffer,
                                    Location loc) const {
    auto funcNameAttr =
        op->getAttrOfType<StringAttr>(AttrNames::Operation::OutlinedFunc);
    if (!funcNameAttr)
      op.emitError("Missing outlined_func attribute for ready-local launch");

    auto outlined =
        AC->getModule().lookupSymbol<func::FuncOp>(funcNameAttr.getValue());
    if (!outlined)
      op.emitError("ready-local outlined function not found");

    Value funcPtr = AC->createFnPtr(outlined, loc);
    Value paramv = op.getParamMemref();
    Value depc = op.getDepCount();
    if (depc.getType() != AC->Int32)
      depc = AC->castToInt(AC->Int32, depc, loc);

    if (auto memrefType = dyn_cast<MemRefType>(paramv.getType())) {
      auto runtimeParamType =
          MemRefType::get({ShapedType::kDynamic}, AC->Int64);
      if (memrefType != runtimeParamType)
        paramv = AC->create<memref::CastOp>(loc, runtimeParamType, paramv)
                     .getResult();
    }

    Value paramc;
    if (auto memrefType = dyn_cast<MemRefType>(paramv.getType())) {
      if (memrefType.hasStaticShape() && memrefType.getNumElements() == 0) {
        paramc = AC->createIntConstant(0, AC->Int32, loc);
      } else {
        auto zeroIndex = AC->createIndexConstant(0, loc);
        auto memrefSize = AC->create<memref::DimOp>(loc, paramv, zeroIndex);
        paramc = AC->create<arith::IndexCastOp>(loc, AC->Int32, memrefSize);
      }
    } else {
      paramc = AC->createIntConstant(0, AC->Int32, loc);
    }

    Value route = op.getRoute();
    if (!route)
      route = createCurrentNodeRoute(AC->getBuilder(), loc);

    auto createIdAttr =
        op->getAttrOfType<IntegerAttr>(AttrNames::Operation::ArtsCreateId);
    Value artsIdVal;
    if (createIdAttr)
      artsIdVal = AC->create<arith::ConstantOp>(loc, AC->Int64, createIdAttr);
    else
      artsIdVal = AC->createIntConstant(0, AC->Int64, loc);
    Value hintMemref = buildArtsHintMemref(AC, route, artsIdVal, loc);

    ArtsCodegen::RuntimeCallBuilder RCB(*AC, loc);
    func::CallOp launch =
        RCB.callOp(types::ARTSRTL_arts_edt_create_ready_local_with_epoch,
                   {funcPtr, paramc, paramv, depc, depBuffer,
                    op.getEpochGuid(), hintMemref});
    freeReadyLocalDepBuffer(depBuffer, loc);
    return launch;
  }

  Value allocateReadyLocalDepBuffer(Value depCount, Location loc) const {
    Value depCountSize = AC->castToInt(AC->PtrSize, depCount, loc);
    Value depElementSize = AC->create<polygeist::TypeSizeOp>(
        loc, IndexType::get(AC->getContext()), AC->ArtsEdtDep);
    depElementSize = AC->castToInt(AC->PtrSize, depElementSize, loc);
    ArtsCodegen::RuntimeCallBuilder RCB(*AC, loc);
    return RCB.call(types::ARTSRTL_arts_calloc, {depCountSize, depElementSize});
  }

  void freeReadyLocalDepBuffer(Value depBuffer, Location loc) const {
    ArtsCodegen::RuntimeCallBuilder RCB(*AC, loc);
    RCB.callVoid(types::ARTSRTL_arts_free, {depBuffer});
  }

  void storeReadyLocalDepEntry(Value depBuffer, Value slotValue,
                               Value guidValue, Value modeValue,
                               std::optional<int32_t> depFlags,
                               Value byteOffsetI64, Value byteSizeI64,
                               Location loc) const {
    Value slotI64 = AC->ensureI64(slotValue, loc);
    Value depEntryPtr = AC->create<LLVM::GEPOp>(
        loc, AC->llvmPtr, AC->ArtsEdtDep, depBuffer, ValueRange{slotI64});
    Value c0 = AC->createIntConstant(0, AC->Int64, loc);
    Value c1 = AC->createIntConstant(1, AC->Int64, loc);
    Value c2 = AC->createIntConstant(2, AC->Int64, loc);
    Value c3 = AC->createIntConstant(3, AC->Int64, loc);
    Value c4 = AC->createIntConstant(4, AC->Int64, loc);
    Value c5 = AC->createIntConstant(5, AC->Int64, loc);
    Value flagsValue =
        AC->createIntConstant(depFlags.value_or(0), AC->Int32, loc);
    Value nullPtr = AC->create<LLVM::ZeroOp>(loc, AC->llvmPtr);
    if (!byteOffsetI64)
      byteOffsetI64 = AC->createIntConstant(0, AC->Int64, loc);
    if (!byteSizeI64)
      byteSizeI64 = AC->createIntConstant(0, AC->Int64, loc);

    Value guidPtr = AC->create<LLVM::GEPOp>(loc, AC->llvmPtr, AC->ArtsEdtDep,
                                            depEntryPtr, ValueRange{c0, c0});
    Value ptrPtr = AC->create<LLVM::GEPOp>(loc, AC->llvmPtr, AC->ArtsEdtDep,
                                           depEntryPtr, ValueRange{c0, c1});
    Value modePtr = AC->create<LLVM::GEPOp>(loc, AC->llvmPtr, AC->ArtsEdtDep,
                                            depEntryPtr, ValueRange{c0, c2});
    Value flagsPtr = AC->create<LLVM::GEPOp>(loc, AC->llvmPtr, AC->ArtsEdtDep,
                                             depEntryPtr, ValueRange{c0, c3});
    Value offsetPtr = AC->create<LLVM::GEPOp>(loc, AC->llvmPtr, AC->ArtsEdtDep,
                                              depEntryPtr, ValueRange{c0, c4});
    Value sizePtr = AC->create<LLVM::GEPOp>(loc, AC->llvmPtr, AC->ArtsEdtDep,
                                            depEntryPtr, ValueRange{c0, c5});
    AC->create<LLVM::StoreOp>(loc, AC->ensureI64(guidValue, loc), guidPtr);
    AC->create<LLVM::StoreOp>(loc, nullPtr, ptrPtr);
    AC->create<LLVM::StoreOp>(loc, modeValue, modePtr);
    AC->create<LLVM::StoreOp>(loc, flagsValue, flagsPtr);
    AC->create<LLVM::StoreOp>(loc, byteOffsetI64, offsetPtr);
    AC->create<LLVM::StoreOp>(loc, byteSizeI64, sizePtr);
  }

  SmallVector<Value, 4>
  inferStencilCenterCoordsFromContract(DbAcquireOp dbAcquireOp,
                                       const DbLoweringInfo &dbInfo,
                                       Location loc) const {
    SmallVector<Value, 4> globalCoords;
    if (!dbAcquireOp || dbInfo.sizes.empty())
      return globalCoords;

    auto contract = getAcquireStencilContract(dbAcquireOp, loc);
    if (!contract ||
        !(contract->isStencilFamily() || contract->usesStencilDistribution()))
      return globalCoords;
    if (contract->spatial.minOffsets.empty()) {
      if (contract->spatial.centerOffset)
        return inferSymmetricStencilCenterCoords(
            *contract->spatial.centerOffset, dbInfo, loc);
      return globalCoords;
    }

    unsigned rank = std::min<unsigned>(dbInfo.sizes.size(),
                                       contract->spatial.minOffsets.size());
    if (!contract->spatial.writeFootprint.empty())
      rank = std::min<unsigned>(rank, contract->spatial.writeFootprint.size());
    if (rank == 0)
      return globalCoords;

    Value zero = AC->createIndexConstant(0, loc);
    Value one = AC->createIndexConstant(1, loc);

    globalCoords.reserve(rank);
    for (unsigned i = 0; i < rank; ++i) {
      Value writeCoord = contract->spatial.writeFootprint.empty()
                             ? zero
                             : contract->spatial.writeFootprint[i];
      Value minOffset = contract->spatial.minOffsets[i];
      Value dimSize = AC->castToIndex(dbInfo.sizes[i], loc);

      Value rawCoord =
          AC->create<arith::SubIOp>(loc, AC->castToIndex(writeCoord, loc),
                                    AC->castToIndex(minOffset, loc));
      Value nonNegative = AC->create<arith::MaxSIOp>(loc, rawCoord, zero);
      Value sizeMinusOne = AC->create<arith::SubIOp>(loc, dimSize, one);
      sizeMinusOne = AC->create<arith::MaxSIOp>(loc, sizeMinusOne, zero);
      Value localCoord =
          AC->create<arith::MinSIOp>(loc, nonNegative, sizeMinusOne);
      Value globalCoord = AC->create<arith::AddIOp>(
          loc, AC->castToIndex(dbInfo.offsets[i], loc), localCoord);
      globalCoords.push_back(globalCoord);
    }

    return globalCoords;
  }

  SmallVector<Value, 4> inferSymmetricStencilCenterCoords(
      int64_t centerOffset, const DbLoweringInfo &dbInfo, Location loc) const {
    SmallVector<Value, 4> globalCoords;
    if (centerOffset < 0 || dbInfo.sizes.empty())
      return globalCoords;

    Value zero = AC->createIndexConstant(0, loc);
    Value one = AC->createIndexConstant(1, loc);
    Value center = AC->createIndexConstant(centerOffset, loc);

    globalCoords.reserve(dbInfo.sizes.size());
    for (unsigned i = 0; i < dbInfo.sizes.size(); ++i) {
      Value dimSize = AC->castToIndex(dbInfo.sizes[i], loc);
      Value sizeMinusOne = AC->create<arith::SubIOp>(loc, dimSize, one);
      sizeMinusOne = AC->create<arith::MaxSIOp>(loc, sizeMinusOne, zero);
      Value localCoord = AC->create<arith::MinSIOp>(loc, center, sizeMinusOne);
      Value globalCoord = AC->create<arith::AddIOp>(
          loc, AC->castToIndex(dbInfo.offsets[i], loc), localCoord);
      globalCoords.push_back(globalCoord);
    }

    return globalCoords;
  }

  Value buildCoordsEqual(ArrayRef<Value> lhs, ArrayRef<Value> rhs,
                         Location loc) const {
    assert(lhs.size() == rhs.size() && "coordinate rank mismatch");
    Value allEqual = AC->create<arith::ConstantIntOp>(loc, 1, 1);
    for (auto [lhsCoord, rhsCoord] : llvm::zip(lhs, rhs)) {
      Value lhsIdx = AC->castToIndex(lhsCoord, loc);
      Value rhsIdx = AC->castToIndex(rhsCoord, loc);
      Value dimEqual = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 lhsIdx, rhsIdx);
      allEqual = AC->create<arith::AndIOp>(loc, allEqual, dimEqual);
    }
    return allEqual;
  }

  SmallVector<Value, 4>
  getDirectLookupStrides(Value storage, ArrayRef<Value> directIndices,
                         ArrayRef<Value> directLayoutSizes,
                         Location loc) const {
    SmallVector<Value, 4> strideSizes;
    strideSizes.reserve(directIndices.size());

    if (directLayoutSizes.size() == directIndices.size()) {
      strideSizes.append(directLayoutSizes.begin(), directLayoutSizes.end());
    } else if (auto storageTy = dyn_cast<MemRefType>(storage.getType())) {
      if (storageTy.getRank() == static_cast<int64_t>(directIndices.size())) {
        for (unsigned i = 0; i < directIndices.size(); ++i) {
          if (storageTy.isDynamicDim(i)) {
            Value dimIdx = AC->createIndexConstant(i, loc);
            strideSizes.push_back(
                AC->create<memref::DimOp>(loc, storage, dimIdx));
          } else {
            strideSizes.push_back(
                AC->createIndexConstant(storageTy.getDimSize(i), loc));
          }
        }
      }
    }

    if (strideSizes.size() != directIndices.size())
      return {};
    return SmallVector<Value, 4>(AC->computeStridesFromSizes(strideSizes, loc));
  }

  /// Load the GUID value for a dependency, handling both direct and depv paths.
  Value loadDbGuidValue(Value dbGuid, Value guidStorage, Value linearIndex,
                        ArrayRef<Value> directIndices,
                        ArrayRef<Value> directLayoutSizes, bool useDepv,
                        Value depStruct, Value baseOffset, Location loc) const {
    if (useDepv) {
      Value finalOffset =
          AC->create<arith::AddIOp>(loc, baseOffset, linearIndex);
      auto depGep =
          AC->create<DepGepOp>(loc, AC->llvmPtr, AC->llvmPtr, depStruct,
                               finalOffset, ValueRange(), ValueRange());
      return AC->create<LLVM::LoadOp>(loc, AC->Int64, depGep.getGuid());
    }
    Value storage = guidStorage ? guidStorage : dbGuid;
    auto storageTy = dyn_cast<MemRefType>(storage.getType());
    auto llvmGuidType = LLVM::LLVMPointerType::get(
        AC->getContext(), storageTy ? storageTy.getMemorySpaceAsInt() : 0);

    SmallVector<Value, 4> gepIndices;
    SmallVector<Value, 4> gepStrides;
    if (!directIndices.empty()) {
      gepIndices.append(directIndices.begin(), directIndices.end());
      gepStrides = getDirectLookupStrides(storage, directIndices,
                                          directLayoutSizes, loc);
    } else {
      gepIndices.push_back(AC->castToIndex(linearIndex, loc));
    }

    auto guidAddr =
        AC->create<DbGepOp>(loc, llvmGuidType, storage, gepIndices, gepStrides);
    return AC->create<LLVM::LoadOp>(loc, AC->Int64, guidAddr);
  }

  /// Get totalDBs from source allocation for bounds checking (stencil cases).
  /// Returns nullptr if boundsValid is not present or allocation not found.
  Value getTotalDBsForBoundsCheck(Value dbGuid, Value boundsValid) const {
    if (!boundsValid)
      return nullptr;

    DbAllocOp allocOp = DbUtils::getAllocOpFromGuid(dbGuid);

    if (allocOp && !allocOp.getSizes().empty())
      return AC->computeTotalElements(allocOp.getSizes(), allocOp.getLoc());
    SmallVector<Value, 4> sizes =
        resolveOuterSizesForGuid(dbGuid, AC, dbGuid.getLoc());
    if (!sizes.empty())
      return AC->computeTotalElements(sizes, dbGuid.getLoc());
    return nullptr;
  }

  /// Holds extracted dependency info for a single datablock source.
  struct DepDbInfo {
    DbLoweringInfo dbInfo;
    SmallVector<Value, 4> allocSizes;
    Value guidStorage = nullptr;
    Value depStruct = nullptr;
    Value baseOffset = nullptr;
    Value stencilCenterLinear;
    SmallVector<Value, 4> stencilCenterCoords;
    std::optional<LoweringContractInfo> stencilContract;
    SmallVector<unsigned, 4> blockOwnerDims;
    SmallVector<Value, 4> blockElementSizes;
    Value scalarSize = nullptr;
  };

  static Value buildTrailingExtentProduct(ArtsCodegen *AC,
                                          ArrayRef<Value> extents,
                                          unsigned firstDim, Location loc) {
    return AC->computeTotalElements(extents.drop_front(firstDim), loc);
  }

  static std::pair<Value, Value>
  buildFaceSliceByteRange(ArtsCodegen *AC, ArrayRef<Value> blockExtents,
                          unsigned dim, int64_t haloExtent, bool upperFace,
                          Value scalarSize, Location loc) {
    Value zero = AC->createIndexConstant(0, loc);
    Value halo = AC->createIndexConstant(haloExtent, loc);
    Value dimExtent = AC->castToIndex(blockExtents[dim], loc);
    Value clampedHalo = AC->create<arith::MinUIOp>(loc, halo, dimExtent);
    Value trailingExtent =
        buildTrailingExtentProduct(AC, blockExtents, dim + 1, loc);
    Value elementOffset = zero;
    if (upperFace)
      elementOffset = AC->create<arith::SubIOp>(loc, dimExtent, clampedHalo);
    Value linearOffset =
        AC->create<arith::MulIOp>(loc, elementOffset, trailingExtent);
    Value elementCount =
        AC->create<arith::MulIOp>(loc, clampedHalo, trailingExtent);
    Value byteOffset = AC->create<arith::MulIOp>(loc, linearOffset, scalarSize);
    Value byteSize = AC->create<arith::MulIOp>(loc, elementCount, scalarSize);
    return {byteOffset, byteSize};
  }

  std::pair<Value, Value>
  inferStencilFaceSliceForSlot(const DepDbInfo &depInfo, Value linearIndex,
                               ArrayRef<Value> directIndices,
                               Location loc) const {
    if (!depInfo.stencilContract ||
        !depInfo.stencilContract->isStencilFamily() ||
        !depInfo.stencilContract->supportsBlockHalo() || !depInfo.scalarSize)
      return {Value(), Value()};

    auto staticMin = depInfo.stencilContract->getStaticMinOffsets();
    auto staticMax = depInfo.stencilContract->getStaticMaxOffsets();
    if (!staticMin || !staticMax || depInfo.blockElementSizes.empty())
      return {Value(), Value()};

    Value zero = AC->createIndexConstant(0, loc);
    Value zeroSize = AC->createIndexConstant(0, loc);
    Value trueI1 = AC->create<arith::ConstantIntOp>(loc, 1, 1);
    Value selectedOffset = zero;
    Value selectedSize = zeroSize;

    auto applyCandidate = [&](Value match, std::pair<Value, Value> slice) {
      selectedOffset =
          AC->create<arith::SelectOp>(loc, match, slice.first, selectedOffset);
      selectedSize =
          AC->create<arith::SelectOp>(loc, match, slice.second, selectedSize);
    };

    if (!directIndices.empty() && !depInfo.stencilCenterCoords.empty()) {
      unsigned ownerRank = std::min<unsigned>(
          directIndices.size(),
          std::min<unsigned>(
              depInfo.stencilCenterCoords.size(),
              std::min<unsigned>(
                  depInfo.blockOwnerDims.empty()
                      ? depInfo.blockElementSizes.size()
                      : depInfo.blockOwnerDims.size(),
                  std::min<unsigned>(staticMin->size(), staticMax->size()))));
      for (unsigned dim = 0; dim < ownerRank; ++dim) {
        unsigned physicalDim =
            depInfo.blockOwnerDims.empty() ? dim : depInfo.blockOwnerDims[dim];
        if (physicalDim >= depInfo.blockElementSizes.size())
          continue;

        // A byte slice can only represent a contiguous face. If any leading
        // extent above the narrowed dimension is wider than 1, the face would
        // become strided and must stay on the whole-DB dependency path.
        bool contiguousFace = true;
        for (unsigned lead = 0; lead < physicalDim; ++lead)
          contiguousFace &=
              ValueAnalysis::isOneConstant(ValueAnalysis::stripNumericCasts(
                  depInfo.blockElementSizes[lead]));
        if (!contiguousFace)
          continue;

        Value sameOtherDims = trueI1;
        for (unsigned other = 0; other < ownerRank; ++other) {
          if (other == dim)
            continue;
          Value equalCoord = AC->create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, directIndices[other],
              depInfo.stencilCenterCoords[other]);
          sameOtherDims =
              AC->create<arith::AndIOp>(loc, sameOtherDims, equalCoord);
        }

        int64_t haloBefore = std::max<int64_t>(0, -(*staticMin)[dim]);
        int64_t haloAfter = std::max<int64_t>(0, (*staticMax)[dim]);
        if (haloBefore > 0) {
          Value isLowerNeighbor = AC->create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, directIndices[dim],
              depInfo.stencilCenterCoords[dim]);
          Value match =
              AC->create<arith::AndIOp>(loc, sameOtherDims, isLowerNeighbor);
          applyCandidate(match, buildFaceSliceByteRange(
                                    AC, depInfo.blockElementSizes, physicalDim,
                                    haloBefore, /*upperFace=*/true,
                                    depInfo.scalarSize, loc));
        }
        if (haloAfter > 0) {
          Value isUpperNeighbor = AC->create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ugt, directIndices[dim],
              depInfo.stencilCenterCoords[dim]);
          Value match =
              AC->create<arith::AndIOp>(loc, sameOtherDims, isUpperNeighbor);
          applyCandidate(match, buildFaceSliceByteRange(
                                    AC, depInfo.blockElementSizes, physicalDim,
                                    haloAfter, /*upperFace=*/false,
                                    depInfo.scalarSize, loc));
        }
      }
      return {selectedOffset, selectedSize};
    }

    if (depInfo.stencilCenterLinear && !depInfo.blockElementSizes.empty()) {
      unsigned ownerRank =
          depInfo.blockOwnerDims.empty()
              ? static_cast<unsigned>(staticMin->size())
              : static_cast<unsigned>(depInfo.blockOwnerDims.size());
      if (ownerRank != 1)
        return {Value(), Value()};

      unsigned physicalDim =
          depInfo.blockOwnerDims.empty() ? 0 : depInfo.blockOwnerDims.front();
      if (physicalDim >= depInfo.blockElementSizes.size())
        return {Value(), Value()};

      for (unsigned lead = 0; lead < physicalDim; ++lead)
        if (!ValueAnalysis::isOneConstant(ValueAnalysis::stripNumericCasts(
                depInfo.blockElementSizes[lead])))
          return {Value(), Value()};

      int64_t haloBefore = std::max<int64_t>(0, -(*staticMin)[0]);
      int64_t haloAfter = std::max<int64_t>(0, (*staticMax)[0]);
      if (haloBefore > 0) {
        Value isLowerNeighbor = AC->create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, AC->castToIndex(linearIndex, loc),
            AC->castToIndex(depInfo.stencilCenterLinear, loc));
        applyCandidate(isLowerNeighbor,
                       buildFaceSliceByteRange(AC, depInfo.blockElementSizes,
                                               physicalDim, haloBefore,
                                               /*upperFace=*/true,
                                               depInfo.scalarSize, loc));
      }
      if (haloAfter > 0) {
        Value isUpperNeighbor = AC->create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ugt, AC->castToIndex(linearIndex, loc),
            AC->castToIndex(depInfo.stencilCenterLinear, loc));
        applyCandidate(isUpperNeighbor,
                       buildFaceSliceByteRange(AC, depInfo.blockElementSizes,
                                               physicalDim, haloAfter,
                                               /*upperFace=*/false,
                                               depInfo.scalarSize, loc));
      }
      return {selectedOffset, selectedSize};
    }

    return {Value(), Value()};
  }

  Value localLinearToGlobalLinear(Value localLinear,
                                  const DbLoweringInfo &dbInfo,
                                  ArrayRef<Value> allocSizes,
                                  Location loc) const {
    if (!localLinear || dbInfo.sizes.empty() || dbInfo.offsets.empty() ||
        allocSizes.size() < dbInfo.sizes.size() ||
        dbInfo.offsets.size() < dbInfo.sizes.size()) {
      return localLinear;
    }

    unsigned rank = dbInfo.sizes.size();
    SmallVector<Value> localCoords = AC->computeIndicesFromLinearIndex(
        ArrayRef<Value>(dbInfo.sizes).take_front(rank), localLinear, loc);
    if (localCoords.size() != rank)
      return AC->castToIndex(localLinear, loc);

    SmallVector<OpFoldResult, 4> globalCoords;
    globalCoords.reserve(rank);
    for (auto [localCoord, offset] : llvm::zip(localCoords, dbInfo.offsets)) {
      Value globalCoord = AC->create<arith::AddIOp>(
          loc, localCoord, AC->castToIndex(offset, loc));
      globalCoords.push_back(globalCoord);
    }

    SmallVector<Value> globalCoordValues;
    globalCoordValues.reserve(globalCoords.size());
    for (OpFoldResult coord : globalCoords)
      globalCoordValues.push_back(
          getValueOrCreateConstantIndexOp(AC->getBuilder(), loc, coord));

    return AC->computeLinearIndex(allocSizes.take_front(rank),
                                  globalCoordValues, loc);
  }

  std::optional<LoweringContractInfo>
  getAcquireStencilContract(DbAcquireOp dbAcquireOp, Location loc) const {
    if (!dbAcquireOp)
      return std::nullopt;

    if (auto info = getLoweringContract(dbAcquireOp.getPtr()))
      return info;
    return getLoweringContract(dbAcquireOp.getOperation(), AC->getBuilder(),
                               loc);
  }

  Value inferStencilCenterLinearFromContract(DbAcquireOp dbAcquireOp,
                                             const DbLoweringInfo &dbInfo,
                                             ArrayRef<Value> allocSizes,
                                             Location loc) const {
    if (!dbAcquireOp || dbInfo.sizes.empty())
      return nullptr;

    auto contract = getAcquireStencilContract(dbAcquireOp, loc);
    if (!contract ||
        !(contract->isStencilFamily() || contract->usesStencilDistribution()))
      return nullptr;
    if (contract->spatial.minOffsets.empty()) {
      if (contract->spatial.centerOffset)
        return inferSymmetricStencilCenterLinear(
            *contract->spatial.centerOffset, dbInfo, allocSizes, loc);
      return nullptr;
    }

    unsigned rank = std::min<unsigned>(dbInfo.sizes.size(),
                                       contract->spatial.minOffsets.size());
    if (!contract->spatial.writeFootprint.empty())
      rank = std::min<unsigned>(rank, contract->spatial.writeFootprint.size());
    if (rank == 0)
      return nullptr;

    Value zero = AC->createIndexConstant(0, loc);
    Value one = AC->createIndexConstant(1, loc);

    SmallVector<Value, 4> localCoords;
    localCoords.reserve(rank);
    for (unsigned i = 0; i < rank; ++i) {
      Value writeCoord = contract->spatial.writeFootprint.empty()
                             ? zero
                             : contract->spatial.writeFootprint[i];
      Value minOffset = contract->spatial.minOffsets[i];
      Value dimSize = AC->castToIndex(dbInfo.sizes[i], loc);

      Value rawCoord =
          AC->create<arith::SubIOp>(loc, AC->castToIndex(writeCoord, loc),
                                    AC->castToIndex(minOffset, loc));

      Value nonNegative = AC->create<arith::MaxSIOp>(loc, rawCoord, zero);
      Value sizeMinusOne = AC->create<arith::SubIOp>(loc, dimSize, one);
      sizeMinusOne = AC->create<arith::MaxSIOp>(loc, sizeMinusOne, zero);
      localCoords.push_back(
          AC->create<arith::MinSIOp>(loc, nonNegative, sizeMinusOne));
    }

    Value localLinearIndex = AC->computeLinearIndex(
        ArrayRef<Value>(dbInfo.sizes).take_front(rank), localCoords, loc);

    return localLinearToGlobalLinear(localLinearIndex, dbInfo, allocSizes, loc);
  }

  Value inferSymmetricStencilCenterLinear(int64_t centerOffset,
                                          const DbLoweringInfo &dbInfo,
                                          ArrayRef<Value> allocSizes,
                                          Location loc) const {
    if (centerOffset < 0 || dbInfo.sizes.empty())
      return nullptr;

    Value zero = AC->createIndexConstant(0, loc);
    Value one = AC->createIndexConstant(1, loc);
    Value center = AC->createIndexConstant(centerOffset, loc);

    SmallVector<Value, 4> localCoords;
    localCoords.reserve(dbInfo.sizes.size());
    for (unsigned i = 0; i < dbInfo.sizes.size(); ++i) {
      Value dimSize = AC->castToIndex(dbInfo.sizes[i], loc);
      Value sizeMinusOne = AC->create<arith::SubIOp>(loc, dimSize, one);
      sizeMinusOne = AC->create<arith::MaxSIOp>(loc, sizeMinusOne, zero);
      localCoords.push_back(
          AC->create<arith::MinSIOp>(loc, center, sizeMinusOne));
    }

    Value localLinearIndex =
        AC->computeLinearIndex(dbInfo.sizes, localCoords, loc);

    return localLinearToGlobalLinear(localLinearIndex, dbInfo, allocSizes, loc);
  }

  /// DepDbAcquireOp addresses dependency storage through depv plus an explicit
  /// baseOffset. RecordDep therefore iterates the local [0, size) window and
  /// lets baseOffset select the correct slice in depv.
  void rebaseDepIterationWindow(DepDbInfo &info, Location loc) const {
    if (info.dbInfo.sizes.empty())
      return;

    info.dbInfo.offsets.clear();
    info.dbInfo.offsets.reserve(info.dbInfo.sizes.size());
    for (size_t i = 0; i < info.dbInfo.sizes.size(); ++i)
      info.dbInfo.offsets.push_back(AC->createIndexConstant(0, loc));
  }

  /// Extract DB lowering info and stencil metadata from a dbGuid's defining op.
  DepDbInfo extractDbInfoForDeps(Value dbGuid,
                                 std::optional<int32_t> acquireMode,
                                 Location loc) const {
    DepDbInfo result;

    /// Resolve the underlying DB operation, tracing through pointer casts
    /// (polygeist.pointer2memref, memref.cast, etc.) that may wrap the GUID.
    auto dbAcquireOp = dbGuid.getDefiningOp<DbAcquireOp>();
    auto depDbAcquireOp = dbGuid.getDefiningOp<DepDbAcquireOp>();
    if (!dbAcquireOp && !depDbAcquireOp) {
      Operation *underlying = DbUtils::getUnderlyingDb(dbGuid);
      if (underlying) {
        dbAcquireOp = dyn_cast<DbAcquireOp>(underlying);
        depDbAcquireOp = dyn_cast<DepDbAcquireOp>(underlying);
      }
    }

    if (dbAcquireOp) {
      result.dbInfo = DbUtils::extractDbLoweringInfo(dbAcquireOp);
      result.guidStorage =
          dbAcquireOp.getSourceGuid() ? dbAcquireOp.getSourceGuid() : dbGuid;
      result.allocSizes = resolveOuterSizesForGuid(dbGuid, AC, loc);
      /// Stencil writer acquires frequently cover [halo..., center, halo...]
      /// DB entries. Recording every entry as WRITE over-serializes adjacent
      /// blocks. Use the acquire's stencil contract to identify the owned
      /// center block and downgrade only the non-center entries to read-only.
      ///
      /// Prefer the full lowering contract so boundary-clamped windows keep
      /// the correct owned-center block. stencil_center_offset is only a
      /// symmetric-radius fallback when richer contract data is unavailable.
      int32_t writeMode = static_cast<int32_t>(DbMode::write);
      bool writerMode = !acquireMode || *acquireMode == writeMode;
      auto partitionMode = dbAcquireOp.getPartitionMode();
      if (writerMode && partitionMode && usesBlockLayout(*partitionMode)) {
        result.stencilContract = getAcquireStencilContract(dbAcquireOp, loc);
        result.stencilCenterLinear = inferStencilCenterLinearFromContract(
            dbAcquireOp, result.dbInfo, result.allocSizes, loc);
        result.stencilCenterCoords = inferStencilCenterCoordsFromContract(
            dbAcquireOp, result.dbInfo, loc);
      }

      if (auto alloc = dyn_cast_or_null<DbAllocOp>(
              DbUtils::getUnderlyingDbAlloc(dbAcquireOp.getSourcePtr()))) {
        Type elementType = alloc.getElementType();
        if (auto memrefType = dyn_cast<MemRefType>(elementType))
          elementType = memrefType.getElementType();
        result.scalarSize = AC->create<polygeist::TypeSizeOp>(
            loc, IndexType::get(AC->getContext()), elementType);

        auto allocExtents = alloc.getElementSizes();
        result.blockElementSizes.assign(allocExtents.begin(),
                                        allocExtents.end());

        unsigned ownerRank = result.dbInfo.sizes.size();
        if (result.stencilContract &&
            !result.stencilContract->spatial.ownerDims.empty())
          ownerRank = static_cast<unsigned>(
              result.stencilContract->spatial.ownerDims.size());
        if (ownerRank != 0) {
          if (result.stencilContract)
            result.blockOwnerDims =
                resolveContractOwnerDims(*result.stencilContract, ownerRank);
          else
            for (unsigned dim = 0; dim < ownerRank; ++dim)
              result.blockOwnerDims.push_back(dim);
        }
      }
    } else if (depDbAcquireOp) {
      result.dbInfo = DbUtils::extractDbLoweringInfo(depDbAcquireOp);
      rebaseDepIterationWindow(result, loc);
      result.guidStorage = depDbAcquireOp.getGuid();
      result.depStruct = depDbAcquireOp.getDepStruct();
      result.baseOffset = depDbAcquireOp.getOffset();
    } else {
      /// Fallback: the GUID value went through casts/outlining that made
      /// the original DbAcquireOp unreachable.  Treat as a single-element
      /// DB with the value itself as guidStorage.
      result.dbInfo.isSingleElement = true;
      result.guidStorage = dbGuid;
    }

    return result;
  }

  /// Holds bounds-checking and allocation size info for dependency emission.
  struct DepBoundsInfo {
    bool useDepv = false;
    Value totalDBs;
    SmallVector<Value> allocSizes;
  };

  /// Compute useDepv flag, bounds-check values, and allocation sizes.
  DepBoundsInfo computeDepBounds(Value dbGuid, const DepDbInfo &depInfo,
                                 DepAccessMode accessMode,
                                 Value boundsValid) const {
    DepBoundsInfo result;
    result.useDepv =
        depInfo.depStruct && depInfo.baseOffset &&
        (accessMode == DepAccessMode::from_depv ||
         isa_and_nonnull<DepDbAcquireOp>(DbUtils::getUnderlyingDb(dbGuid)));

    result.totalDBs = result.useDepv
                          ? Value()
                          : getTotalDBsForBoundsCheck(dbGuid, boundsValid);
    if (!result.useDepv)
      result.allocSizes = resolveOuterSizesForGuid(dbGuid, AC, dbGuid.getLoc());

    return result;
  }

  /// Iterate over DB elements and emit record-dep calls for each index.
  void emitRecordDepCalls(Value dbGuid, Value edtGuid, Value sharedSlotAlloc,
                          DepAccessMode accessMode,
                          std::optional<int32_t> acquireMode,
                          std::optional<int32_t> depFlags, Value boundsValid,
                          Value byteOffset, Value byteSize,
                          const DepDbInfo &depInfo, const DepBoundsInfo &bounds,
                          Value readyLocalDepBuffer, Location loc) const {
    auto guidStorageType = dyn_cast<MemRefType>(
        (depInfo.guidStorage ? depInfo.guidStorage : dbGuid).getType());
    bool useDirectCoords = !bounds.useDepv && guidStorageType &&
                           guidStorageType.getRank() > 1 &&
                           depInfo.dbInfo.sizes.size() > 1;
    if (useDirectCoords) {
      Value one = AC->createIndexConstant(1, loc);
      SmallVector<Value, 4> globalCoords;

      std::function<void(unsigned)> emitForCoords = [&](unsigned dim) {
        if (dim == depInfo.dbInfo.sizes.size()) {
          SmallVector<Value, 4> localCoords;
          localCoords.reserve(globalCoords.size());
          for (auto [coord, offset] :
               llvm::zip(globalCoords, depInfo.dbInfo.offsets)) {
            localCoords.push_back(AC->create<arith::SubIOp>(
                loc, coord, AC->castToIndex(offset, loc)));
          }

          Value linearIndex = bounds.allocSizes.empty()
                                  ? AC->computeLinearIndex(depInfo.dbInfo.sizes,
                                                           localCoords, loc)
                                  : AC->computeLinearIndex(bounds.allocSizes,
                                                           globalCoords, loc);

          recordSingleDb(
              dbGuid, depInfo.guidStorage, edtGuid, sharedSlotAlloc,
              linearIndex, ArrayRef<Value>(globalCoords), bounds.allocSizes,
              accessMode, acquireMode, depFlags, boundsValid, depInfo.depStruct,
              depInfo.baseOffset, bounds.totalDBs, byteOffset, byteSize,
              depInfo.stencilCenterLinear, depInfo.stencilCenterCoords,
              &depInfo, readyLocalDepBuffer, loc);
          return;
        }

        Value lowerBound = depInfo.dbInfo.offsets[dim];
        Value upperBound = AC->create<arith::AddIOp>(loc, lowerBound,
                                                     depInfo.dbInfo.sizes[dim]);
        auto loopOp = AC->create<scf::ForOp>(loc, lowerBound, upperBound, one);
        Block &loopBlock = loopOp.getRegion().front();
        AC->setInsertionPointToStart(&loopBlock);
        globalCoords.push_back(loopOp.getInductionVar());
        emitForCoords(dim + 1);
        globalCoords.pop_back();
        AC->setInsertionPointAfter(loopOp);
      };

      emitForCoords(0);
      return;
    }

    if (!bounds.useDepv && depInfo.dbInfo.isSingleElement &&
        !depInfo.dbInfo.indices.empty()) {
      Value zero = AC->createIndexConstant(0, loc);
      recordSingleDb(dbGuid, depInfo.guidStorage, edtGuid, sharedSlotAlloc,
                     zero, depInfo.dbInfo.indices, bounds.allocSizes,
                     accessMode, acquireMode, depFlags, boundsValid,
                     depInfo.depStruct, depInfo.baseOffset, bounds.totalDBs,
                     byteOffset, byteSize, depInfo.stencilCenterLinear,
                     depInfo.stencilCenterCoords, &depInfo,
                     readyLocalDepBuffer, loc);
      return;
    }

    AC->iterateDbElements(
        dbGuid, edtGuid, depInfo.dbInfo.sizes, depInfo.dbInfo.offsets,
        depInfo.dbInfo.isSingleElement, loc,
        [&](Value linearIndex) {
          recordSingleDb(dbGuid, depInfo.guidStorage, edtGuid, sharedSlotAlloc,
                         linearIndex, ArrayRef<Value>(), bounds.allocSizes,
                         accessMode, acquireMode, depFlags, boundsValid,
                         depInfo.depStruct, depInfo.baseOffset, bounds.totalDBs,
                         byteOffset, byteSize, depInfo.stencilCenterLinear,
                         depInfo.stencilCenterCoords, &depInfo,
                         readyLocalDepBuffer, loc);
        },
        bounds.allocSizes);
  }

  /// Record all dependencies for a single datablock
  void recordDepsForDb(Value dbGuid, Value edtGuid, Value sharedSlotAlloc,
                       DepAccessMode accessMode,
                       std::optional<int32_t> acquireMode,
                       std::optional<int32_t> depFlags, Value boundsValid,
                       Value byteOffset, Value byteSize,
                       Value readyLocalDepBuffer, Location loc) const {
    DepDbInfo depInfo = extractDbInfoForDeps(dbGuid, acquireMode, loc);
    DepBoundsInfo bounds =
        computeDepBounds(dbGuid, depInfo, accessMode, boundsValid);
    emitRecordDepCalls(dbGuid, edtGuid, sharedSlotAlloc, accessMode,
                       acquireMode, depFlags, boundsValid, byteOffset, byteSize,
                       depInfo, bounds, readyLocalDepBuffer, loc);
  }

  /// Emit the appropriate runtime call for recording a dependency.
  /// Standard path: arts_add_dependence(dbGuid, edtGuid, slot, mode)
  /// ESD path: arts_add_dependence_at(dbGuid, edtGuid, slot, mode, offset,
  /// size)
  void emitRecordDepCall(Value dbGuidValue, Value edtGuidValue,
                         Value currentSlotI32, Value modeValue,
                         Value byteOffsetI64, Value byteSizeI64,
                         std::optional<int32_t> depFlags, Location loc) const {
    auto emitWholeDbDep = [&]() {
      ArtsCodegen::RuntimeCallBuilder RCB(*AC, loc);
      /// Standard path: full datablock dependency
      if (depFlags && *depFlags != 0) {
        Value flagsValue = AC->createIntConstant(*depFlags, AC->Int32, loc);
        RCB.callVoid(
            types::ARTSRTL_arts_add_dependence_ex,
            {dbGuidValue, edtGuidValue, currentSlotI32, modeValue, flagsValue});
      } else {
        RCB.callVoid(types::ARTSRTL_arts_add_dependence,
                     {dbGuidValue, edtGuidValue, currentSlotI32, modeValue});
      }
    };

    auto emitSliceDep = [&]() {
      ArtsCodegen::RuntimeCallBuilder RCB(*AC, loc);
      /// ESD path: partial slice dependency with byte offset/size
      if (depFlags && *depFlags != 0) {
        Value flagsValue = AC->createIntConstant(*depFlags, AC->Int32, loc);
        RCB.callVoid(types::ARTSRTL_arts_add_dependence_at_ex,
                     {dbGuidValue, edtGuidValue, currentSlotI32, modeValue,
                      byteOffsetI64, byteSizeI64, flagsValue});
      } else {
        RCB.callVoid(types::ARTSRTL_arts_add_dependence_at,
                     {dbGuidValue, edtGuidValue, currentSlotI32, modeValue,
                      byteOffsetI64, byteSizeI64});
      }
    };

    if (!byteOffsetI64 || !byteSizeI64) {
      emitWholeDbDep();
      return;
    }

    Value normalizedByteSize = ValueAnalysis::stripNumericCasts(byteSizeI64);
    /// byte_size == 0 is the cross-pass sentinel for "no partial slice".
    /// Respect it both when the zero is constant and when it only becomes
    /// known after runtime guards (whole-block or center-block fallback).
    if (ValueAnalysis::isZeroConstant(normalizedByteSize)) {
      emitWholeDbDep();
      return;
    }
    if (ValueAnalysis::isProvablyNonZero(normalizedByteSize)) {
      emitSliceDep();
      return;
    }

    Value zeroI64 = AC->createIntConstant(0, AC->Int64, loc);
    Value useWholeDbDep = AC->create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, byteSizeI64, zeroI64);
    auto ifOp =
        AC->create<scf::IfOp>(loc, useWholeDbDep, /*withElseRegion=*/true);
    AC->setInsertionPointToStart(&ifOp.getThenRegion().front());
    emitWholeDbDep();
    AC->setInsertionPointToStart(&ifOp.getElseRegion().front());
    emitSliceDep();
    AC->setInsertionPointAfter(ifOp);
  }

  /// Record a single datablock dependency for an EDT slot.
  ///
  /// Two main paths:
  /// 1. Guarded (boundsValid set): if-else to handle boundary workers
  ///    - Then: arts_add_dependence[_at] for valid indices
  ///    - Else: arts_signal_edt_null for out-of-bounds
  /// 2. Unconditional: direct arts_add_dependence[_at] call
  ///
  /// Within each path, ESD vs standard is handled by emitRecordDepCall.
  void recordSingleDb(
      Value dbGuid, Value guidStorage, Value edtGuid, Value slotAlloc,
      Value linearIndex, ArrayRef<Value> directIndices,
      ArrayRef<Value> directLayoutSizes, DepAccessMode accessMode,
      std::optional<int32_t> acquireMode, std::optional<int32_t> depFlags,
      Value boundsValid, Value depStruct, Value baseOffset, Value totalDBs,
      Value byteOffset, Value byteSize, Value stencilCenterLinear,
      ArrayRef<Value> stencilCenterCoords, const DepDbInfo *depInfo,
      Value readyLocalDepBuffer, Location loc) const {
    const bool useDepv =
        depStruct && baseOffset &&
        (accessMode == DepAccessMode::from_depv ||
         isa_and_nonnull<DepDbAcquireOp>(DbUtils::getUnderlyingDb(dbGuid)));

    /// Extend boundsValid with DB index check for stencil cases.
    /// When the allocation is coarse (totalDBs == 1), all workers bind to the
    /// single partition — clamp linearIndex to 0 so the bounds check passes
    /// and all workers receive the dependency.
    Value effectiveBoundsValid = boundsValid;
    if (totalDBs) {
      if (ValueAnalysis::isOneConstant(totalDBs))
        linearIndex = AC->createIndexConstant(0, loc);
      Value indexInBounds = AC->create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, linearIndex, totalDBs);
      effectiveBoundsValid =
          AC->create<arith::AndIOp>(loc, boundsValid, indexInBounds);
    }

    /// Prepare common values
    Value edtGuidValue = edtGuid;
    if (auto mt = dyn_cast<MemRefType>(edtGuid.getType())) {
      auto zeroIndex = AC->createIndexConstant(0, loc);
      edtGuidValue =
          AC->create<memref::LoadOp>(loc, edtGuid, ValueRange{zeroIndex});
    }
    edtGuidValue = AC->ensureI64(edtGuidValue, loc);

    auto currentSlotI32 = AC->create<memref::LoadOp>(loc, slotAlloc);
    int32_t readMode = static_cast<int32_t>(DbMode::read);
    int32_t writeMode = static_cast<int32_t>(DbMode::write);
    int32_t modeInt = acquireMode.value_or(writeMode);
    Value modeValue = AC->createIntConstant(modeInt, AC->Int32, loc);

    Value isCenterBlock;
    if (!stencilCenterCoords.empty() && !directIndices.empty() &&
        stencilCenterCoords.size() == directIndices.size() &&
        modeInt == writeMode) {
      isCenterBlock = buildCoordsEqual(directIndices, stencilCenterCoords, loc);
      Value readValue = AC->createIntConstant(readMode, AC->Int32, loc);
      Value writeValue = AC->createIntConstant(writeMode, AC->Int32, loc);
      modeValue = AC->create<arith::SelectOp>(loc, isCenterBlock, writeValue,
                                              readValue);
    } else if (stencilCenterLinear && modeInt == writeMode) {
      Value linearIndexIdx = AC->castToIndex(linearIndex, loc);
      Value centerIdx = AC->castToIndex(stencilCenterLinear, loc);
      isCenterBlock = AC->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                linearIndexIdx, centerIdx);
      Value readValue = AC->createIntConstant(readMode, AC->Int32, loc);
      Value writeValue = AC->createIntConstant(writeMode, AC->Int32, loc);
      modeValue = AC->create<arith::SelectOp>(loc, isCenterBlock, writeValue,
                                              readValue);
    }

    // When an acquire expands into multiple DB slots, derive read-only face
    // slices from the stencil contract per emitted slot instead of widening
    // every neighbor block back to a whole-DB dependence.
    Value effectiveByteOffset = byteOffset;
    Value effectiveByteSize = byteSize;
    std::optional<int32_t> effectiveDepFlags = depFlags;
    bool preserveShape =
        depFlags && ((*depFlags & kArtsDepFlagPreserveShape) != 0);
    bool inferredPreserveShape = false;
    bool hasExplicitSlice = byteOffset && byteSize &&
                            !ValueAnalysis::isZeroConstant(
                                ValueAnalysis::stripNumericCasts(byteSize));
    if (!preserveShape && !hasExplicitSlice && depInfo) {
      auto inferredSlice = inferStencilFaceSliceForSlot(*depInfo, linearIndex,
                                                        directIndices, loc);
      if (inferredSlice.first && inferredSlice.second) {
        effectiveByteOffset = inferredSlice.first;
        effectiveByteSize = inferredSlice.second;
        /// Late stencil-face inference keeps the consumer IR on the original
        /// block coordinate system (for example, Seidel's left halo still
        /// indexes row 252 of the predecessor block). Compact slice payloads
        /// would break that contract, so inferred face slices must request the
        /// shape-preserving runtime path.
        inferredPreserveShape = true;
        int32_t depFlagBits = effectiveDepFlags.value_or(0);
        depFlagBits |= kArtsDepFlagPreserveShape;
        effectiveDepFlags = depFlagBits;
      }
    }
    if (preserveShape && !inferredPreserveShape) {
      /// Explicit preserve-shape markings currently act as an analysis-time
      /// "do not compact this acquire" contract. Keep those on the whole-DB
      /// path until the upstream acquire rewrite carries a compact index space.
      effectiveByteOffset = nullptr;
      effectiveByteSize = nullptr;
      if (effectiveDepFlags) {
        int32_t depFlagBits = *effectiveDepFlags & ~kArtsDepFlagPreserveShape;
        if (depFlagBits == 0)
          effectiveDepFlags.reset();
        else
          effectiveDepFlags = depFlagBits;
      }
    }

    /// Prepare ESD values if needed (cast Index to Int64).
    /// byte_size == 0 denotes "no partial slice" and should use full-db deps.
    bool hasPartialSlice =
        effectiveByteOffset && effectiveByteSize &&
        !ValueAnalysis::isZeroConstant(
            ValueAnalysis::stripNumericCasts(effectiveByteSize));
    if (hasPartialSlice && modeInt != readMode && !isCenterBlock) {
      /// ARTS runtime only supports arts_add_dependence_at (sliced deps) for
      /// DB_MODE_RO. For write-mode (EW) deps without stencil center-block
      /// semantics, fall back to whole-DB dependencies.
      effectiveByteOffset = nullptr;
      effectiveByteSize = nullptr;
      hasPartialSlice = false;
    }
    Value byteOffsetI64 =
        hasPartialSlice ? AC->ensureI64(effectiveByteOffset, loc) : nullptr;
    Value byteSizeI64 =
        hasPartialSlice ? AC->ensureI64(effectiveByteSize, loc) : nullptr;
    if (hasPartialSlice && isCenterBlock) {
      /// ARTS only supports sliced DB transport for RO slots. Keep halo
      /// dependencies as byte slices, but force the owned center block back to
      /// a whole-DB dependence by zeroing the slice at runtime.
      Value zeroI64 = AC->createIntConstant(0, AC->Int64, loc);
      byteOffsetI64 = AC->create<arith::SelectOp>(loc, isCenterBlock, zeroI64,
                                                  byteOffsetI64);
      byteSizeI64 =
          AC->create<arith::SelectOp>(loc, isCenterBlock, zeroI64, byteSizeI64);
    }

    if (effectiveBoundsValid) {
      /// GUARDED PATH: boundary workers may have invalid indices
      auto ifOp = AC->create<scf::IfOp>(loc, effectiveBoundsValid,
                                        /*withElseRegion=*/true);

      /// Then: valid index - record the dependency
      AC->setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value dbGuidValue = loadDbGuidValue(dbGuid, guidStorage, linearIndex,
                                          directIndices, directLayoutSizes,
                                          useDepv, depStruct, baseOffset, loc);
      if (readyLocalDepBuffer) {
        storeReadyLocalDepEntry(readyLocalDepBuffer, currentSlotI32,
                                dbGuidValue, modeValue, effectiveDepFlags,
                                byteOffsetI64, byteSizeI64, loc);
      } else {
        emitRecordDepCall(dbGuidValue, edtGuidValue, currentSlotI32, modeValue,
                          byteOffsetI64, byteSizeI64, effectiveDepFlags, loc);
      }

      /// Else: invalid index - signal null dependency
      AC->setInsertionPointToStart(&ifOp.getElseRegion().front());
      if (readyLocalDepBuffer) {
        Value nullGuid = AC->createIntConstant(0, AC->Int64, loc);
        Value nullMode = AC->createIntConstant(DB_MODE_NULL, AC->Int32, loc);
        storeReadyLocalDepEntry(readyLocalDepBuffer, currentSlotI32, nullGuid,
                                nullMode, std::nullopt, Value(), Value(), loc);
      } else {
        ArtsCodegen::RuntimeCallBuilder RCB(*AC, loc);
        RCB.callVoid(types::ARTSRTL_arts_signal_edt_null,
                     {edtGuidValue, currentSlotI32});
      }

      AC->setInsertionPointAfter(ifOp);
    } else {
      /// UNCONDITIONAL PATH: all indices are valid
      Value dbGuidValue = loadDbGuidValue(dbGuid, guidStorage, linearIndex,
                                          directIndices, directLayoutSizes,
                                          useDepv, depStruct, baseOffset, loc);
      if (readyLocalDepBuffer) {
        storeReadyLocalDepEntry(readyLocalDepBuffer, currentSlotI32,
                                dbGuidValue, modeValue, effectiveDepFlags,
                                byteOffsetI64, byteSizeI64, loc);
      } else {
        emitRecordDepCall(dbGuidValue, edtGuidValue, currentSlotI32, modeValue,
                          byteOffsetI64, byteSizeI64, effectiveDepFlags, loc);
      }
    }

    /// Increment slot counter
    auto oneI32 = AC->createIntConstant(1, AC->Int32, loc);
    auto incrementedSlot =
        AC->create<arith::AddIOp>(loc, currentSlotI32, oneI32);
    AC->create<memref::StoreOp>(loc, incrementedSlot, slotAlloc);
  }
};

/// Pattern to convert arts.increment_dep operations.
/// In arts v2, the latch mechanism has been removed. IncrementDepOp is now a
/// no-op that simply gets erased during lowering.
struct IncrementDepPattern : public ArtsToLLVMPattern<IncrementDepOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(IncrementDepOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Erasing IncrementDep Op (latch removed in v2) " << op);
    rewriter.eraseOp(op);
    ++numDepOpsConverted;
    return success();
  }
};

///===----------------------------------------------------------------------===///
/// EDT Patterns
///===----------------------------------------------------------------------===///

/// Pattern to convert arts.edt_param_pack operations
struct EdtParamPackPattern : public ArtsToLLVMPattern<EdtParamPackOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(EdtParamPackOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering EdtParamPack Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);

    auto loc = op.getLoc();
    auto params = op.getParams();
    auto resultType = dyn_cast<MemRefType>(op.getMemref().getType());
    if (!resultType)
      return op.emitError("Expected MemRef type for result");

    memref::AllocaOp allocOp;
    if (params.empty() ||
        (resultType.hasStaticShape() && resultType.getNumElements() == 0)) {
      /// Empty parameter pack: allocate dynamic memref<?xi64> with size 0
      auto dynamicType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
      auto zeroIndex = AC->createIndexConstant(0, loc);
      allocOp =
          AC->create<memref::AllocaOp>(loc, dynamicType, ValueRange{zeroIndex});
    } else if (resultType.getNumDynamicDims() > 0) {
      /// Dynamic memref: allocate with runtime size and store parameters.
      auto numParams = AC->createIndexConstant(params.size(), loc);
      allocOp =
          AC->create<memref::AllocaOp>(loc, resultType, ValueRange{numParams});
    } else {
      /// Static non-empty pack: preserve its compile-time extent.
      allocOp = AC->create<memref::AllocaOp>(loc, resultType, ValueRange{});
    }

    for (unsigned i = 0; i < params.size(); ++i) {
      auto index = AC->createIndexConstant(i, loc);
      auto castParam = AC->castParameter(
          AC->Int64, params[i], loc, ArtsCodegen::ParameterCastMode::Bitwise);
      AC->create<memref::StoreOp>(loc, castParam, allocOp, ValueRange{index});
    }

    rewriter.replaceOp(op, allocOp);
    ++numEdtOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.edt_param_unpack operations
struct EdtParamUnpackPattern : public ArtsToLLVMPattern<EdtParamUnpackOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(EdtParamUnpackOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering EdtParamUnpack Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto paramMemref = op.getMemref();
    auto results = op.getUnpacked();

    SmallVector<Value> newResults;
    for (unsigned i = 0; i < results.size(); ++i) {
      auto idx = AC->createIndexConstant(i, op.getLoc());
      auto loadedParam =
          AC->create<memref::LoadOp>(op.getLoc(), paramMemref, ValueRange{idx});
      auto castedParam =
          AC->castParameter(results[i].getType(), loadedParam, op.getLoc(),
                            ArtsCodegen::ParameterCastMode::Bitwise);
      newResults.push_back(castedParam);
    }

    rewriter.replaceOp(op, newResults);
    ++numEdtOpsConverted;
    return success();
  }
};

///===----------------------------------------------------------------------===///
/// Dependency GEP and Acquire Patterns
///===----------------------------------------------------------------------===///

/// Pattern to convert arts.dep_gep operations
struct DepGepOpPattern : public ArtsToLLVMPattern<DepGepOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DepGepOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DepGep Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto depStruct = op.getDepStruct();
    auto offset = op.getOffset();
    auto indices = op.getIndices();
    auto strides = op.getStrides();
    auto loc = op.getLoc();

    /// Cast dep_struct to typed pointer to LLVM pointer
    Value typedDepPtr = AC->castToLLVMPtr(depStruct, loc);

    /// Compute linearized index: offset + sum_i indices[i] * strides[i]
    Value linearIndex = offset;
    if (linearIndex.getType() != AC->Int64)
      linearIndex = AC->ensureI64(linearIndex, loc);

    for (size_t i = 0; i < indices.size(); ++i) {
      Value idx = indices[i];
      if (idx.getType() != AC->Int64)
        idx = AC->ensureI64(idx, loc);
      Value strideVal = (i < strides.size())
                            ? strides[i]
                            : AC->createIntConstant(1, AC->Int64, loc);
      if (strideVal.getType() != AC->Int64)
        strideVal = AC->ensureI64(strideVal, loc);

      Value contrib = AC->create<arith::MulIOp>(loc, idx, strideVal);
      linearIndex = AC->create<arith::AddIOp>(loc, linearIndex, contrib);
    }

    Value depEntryPtr = AC->create<LLVM::GEPOp>(
        loc, AC->llvmPtr, AC->ArtsEdtDep, typedDepPtr, ValueRange{linearIndex});

    /// Extract both guid (field #0) and ptr (field #1) from depv
    /// v2 arts_edt_dep_t layout: { guid, ptr, mode }
    auto c0 = AC->createIntConstant(0, AC->Int64, loc);
    auto cPtrIdx = AC->createIntConstant(1, AC->Int64, loc);

    /// Get the guid pointer (field #0)
    auto guidPtr = AC->create<LLVM::GEPOp>(loc, AC->llvmPtr, AC->ArtsEdtDep,
                                           depEntryPtr, ValueRange{c0, c0});

    /// Get the data pointer (field #1)
    auto dataPtr = AC->create<LLVM::GEPOp>(
        loc, AC->llvmPtr, AC->ArtsEdtDep, depEntryPtr, ValueRange{c0, cPtrIdx});

    /// Return both: guid pointer and data pointer
    rewriter.replaceOp(op, ValueRange{guidPtr, dataPtr});
    ++numDepOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.dep_db_acquire operations
struct DepDbAcquireOpPattern : public ArtsToLLVMPattern<DepDbAcquireOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DepDbAcquireOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DepDbAcquire Op " << op);
    for (OpOperand &use : op.getGuid().getUses())
      if (isa<RecordDepOp>(use.getOwner()))
        return failure();
    for (Operation *user : op.getPtr().getUsers())
      if (auto acquire = dyn_cast<DbAcquireOp>(user))
        if (!acquire.getSourceGuid())
          return failure();

    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto loc = op.getLoc();
    auto depStruct = op.getDepStruct();

    Value typedDepPtr = AC->castToLLVMPtr(depStruct, loc);
    Value linearIndex = op.getOffset();
    if (linearIndex.getType() != AC->Int64)
      linearIndex = AC->ensureI64(linearIndex, loc);

    SmallVector<Value> strides;
    if (!op.getSizes().empty()) {
      SmallVector<Value> sizes(op.getSizes().begin(), op.getSizes().end());
      strides = AC->computeStridesFromSizes(sizes, loc);
    }

    auto indices = op.getIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      Value idx = indices[i];
      if (idx.getType() != AC->Int64)
        idx = AC->ensureI64(idx, loc);
      Value strideVal = (i < strides.size())
                            ? strides[i]
                            : AC->createIntConstant(1, AC->Int64, loc);
      if (strideVal.getType() != AC->Int64)
        strideVal = AC->ensureI64(strideVal, loc);

      Value contrib = AC->create<arith::MulIOp>(loc, idx, strideVal);
      linearIndex = AC->create<arith::AddIOp>(loc, linearIndex, contrib);
    }

    Value depEntryPtr = AC->create<LLVM::GEPOp>(
        loc, AC->llvmPtr, AC->ArtsEdtDep, typedDepPtr, ValueRange{linearIndex});

    auto c0 = AC->createIntConstant(0, AC->Int64, loc);
    auto cPtrIdx = AC->createIntConstant(1, AC->Int64, loc);

    Value guidPtr = AC->create<LLVM::GEPOp>(loc, AC->llvmPtr, AC->ArtsEdtDep,
                                            depEntryPtr, ValueRange{c0, c0});
    Value dataPtrAddr = AC->create<LLVM::GEPOp>(
        loc, AC->llvmPtr, AC->ArtsEdtDep, depEntryPtr, ValueRange{c0, cPtrIdx});
    Value payloadPtr = AC->create<LLVM::LoadOp>(loc, AC->llvmPtr, dataPtrAddr);

    /// DepDbAcquireOp feeds two distinct downstream shapes:
    /// - flat payload memrefs (e.g. memref<?x?xf32>) expect the dep entry's
    ///   ptr field to be resolved to the payload base pointer directly.
    /// - handle-table memrefs (e.g. memref<?x!llvm.ptr> or
    ///   memref<?xmemref<?xf64>>) still need one level of pointer indirection
    ///   preserved so later db_gep/load pairs can recover the concrete payload
    ///   pointer. Loading the dep entry here for those cases turns data bytes
    ///   into fake pointers at runtime.
    ///
    /// Some outlined paths may preserve memref<?xmemref<...>> types (the
    /// original pre-DbAlloc types) instead of memref<?x!llvm.ptr>. Both
    /// represent block-partitioned pointer tables and need dataPtrAddr.
    Value ptrBase = payloadPtr;
    if (auto ptrType = dyn_cast<MemRefType>(op.getPtr().getType())) {
      auto elemTy = ptrType.getElementType();
      if (isa<LLVM::LLVMPointerType>(elemTy) || isa<MemRefType>(elemTy))
        ptrBase = dataPtrAddr;
    }

    auto guidView = AC->create<polygeist::Pointer2MemrefOp>(
        loc, op.getGuid().getType(), guidPtr);
    auto ptrView = AC->create<polygeist::Pointer2MemrefOp>(
        loc, op.getPtr().getType(), ptrBase);
    rewriter.replaceOp(op, ValueRange{guidView, ptrView});
    ++numDepOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.db_gep operations to LLVM GEP using element strides
struct DbGepOpPattern : public ArtsToLLVMPattern<DbGepOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbGepOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbGep Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto base = op.getBasePtr();
    auto indices = op.getIndices();
    auto strides = op.getStrides();
    auto loc = op.getLoc();

    /// Cast base to LLVM pointer type
    Value basePtr = AC->castToLLVMPtr(base, loc);

    /// If base is a memref of pointers, we must load the pointer element;
    /// otherwise we compute the address within the element buffer.
    auto baseMT = dyn_cast<MemRefType>(base.getType());

    /// Pad strides with 1s to match indices length
    SmallVector<Value> paddedStrides(strides.begin(), strides.end());
    while (paddedStrides.size() < indices.size())
      paddedStrides.push_back(AC->createIndexConstant(1, loc));

    /// Compute linear element index using provided strides
    Value linearIdx =
        AC->computeLinearIndexFromStrides(paddedStrides, indices, loc);
    Value idx64 = AC->ensureI64(linearIdx, loc);

    /// Use typed GEP based on the element type; default to pointer elements.
    /// memref element types model ARTS pointer tables at this stage, not in-line
    /// memref descriptors, so address those slots as pointer-sized entries.
    Type elemTy = baseMT ? baseMT.getElementType() : AC->llvmPtr;
    if (isa<MemRefType>(elemTy))
      elemTy = AC->llvmPtr;
    Value elemAddr = AC->create<LLVM::GEPOp>(loc, AC->llvmPtr, elemTy, basePtr,
                                             ValueRange{idx64});

    rewriter.replaceOp(op, elemAddr);
    ++numDbOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.edt_create operations (key EDT creation)
struct EdtCreatePattern : public ArtsToLLVMPattern<EdtCreateOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(EdtCreateOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(AttrNames::Operation::ReadyLocalLaunch) &&
        op.getEpochGuid()) {
      for (Operation *user : op.getGuid().getUsers())
        if (isa<RecordDepOp>(user))
          return failure();
    }

    ARTS_INFO("Lowering EdtCreate Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    /// Get outlined function name
    auto funcNameAttr =
        op->getAttrOfType<StringAttr>(AttrNames::Operation::OutlinedFunc);
    if (!funcNameAttr)
      return op.emitError("Missing outlined_func attribute");

    auto outlined =
        AC->getModule().lookupSymbol<func::FuncOp>(funcNameAttr.getValue());
    if (!outlined)
      return op.emitError("EDT Outlined function not found");

    /// Build arts_edt_create arguments
    auto funcPtr = AC->createFnPtr(outlined, op.getLoc());
    auto loc = op.getLoc();
    Value paramv = op.getParamMemref();
    Value depc = op.getDepCount();
    if (depc && depc.getType() != AC->Int32)
      depc = AC->castToInt(AC->Int32, depc, loc);

    /// Normalize paramv to the runtime ABI type expected by the helper decls.
    if (auto memrefType = dyn_cast<MemRefType>(paramv.getType())) {
      auto runtimeParamType =
          MemRefType::get({ShapedType::kDynamic}, AC->Int64);
      if (memrefType != runtimeParamType)
        paramv = AC->create<memref::CastOp>(loc, runtimeParamType, paramv)
                     .getResult();
    }

    /// Calculate parameter count from memref size
    Value paramc;
    if (auto memrefType = dyn_cast<MemRefType>(paramv.getType())) {
      if (memrefType.hasStaticShape() && memrefType.getNumElements() == 0) {
        paramc = AC->createIntConstant(0, AC->Int32, loc);
      } else {
        /// Dynamic memref case - get size from memref
        auto zeroIndex = AC->createIndexConstant(0, loc);
        auto memrefSize = AC->create<memref::DimOp>(loc, paramv, zeroIndex);
        paramc = AC->create<arith::IndexCastOp>(loc, AC->Int32, memrefSize);
      }
    } else {
      paramc = AC->createIntConstant(0, AC->Int32, loc);
    }

    /// Build arts_hint_t struct: { uint32_t route, uint64_t id }
    Value route = op.getRoute();
    if (!route)
      route = createCurrentNodeRoute(AC->getBuilder(), loc);

    auto createIdAttr =
        op->getAttrOfType<IntegerAttr>(AttrNames::Operation::ArtsCreateId);
    Value artsIdVal;
    if (createIdAttr)
      artsIdVal = AC->create<arith::ConstantOp>(loc, AC->Int64, createIdAttr);
    else
      artsIdVal = AC->createIntConstant(0, AC->Int64, loc);

    Value hintMemref = buildArtsHintMemref(AC, route, artsIdVal, loc);

    /// Create arts_edt_create call with hint as last argument
    ArtsCodegen::RuntimeCallBuilder RCB(*AC, loc);
    func::CallOp callOp;
    if (op.getEpochGuid()) {
      callOp = RCB.callOp(
          types::ARTSRTL_arts_edt_create_with_epoch,
          {funcPtr, paramc, paramv, depc, op.getEpochGuid(), hintMemref});
    } else {
      callOp = RCB.callOp(types::ARTSRTL_arts_edt_create,
                          {funcPtr, paramc, paramv, depc, hintMemref});
    }
    if (createIdAttr)
      callOp->setAttr(AttrNames::Operation::ArtsCreateId, createIdAttr);
    rewriter.replaceOp(op, callOp.getResult(0));
    ++numEdtOpsConverted;
    return success();
  }
};

///===----------------------------------------------------------------------===///
/// State and Bind Patterns
///===----------------------------------------------------------------------===///

struct StatePackPattern : public ArtsToLLVMPattern<StatePackOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(StatePackOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering StatePack Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);

    auto loc = op.getLoc();
    auto values = op.getValues();
    auto resultType = dyn_cast<MemRefType>(op.getState().getType());
    if (!resultType)
      return op.emitError("Expected MemRef type for state result");

    memref::AllocaOp allocOp;
    if (values.empty()) {
      auto dynamicType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
      auto zeroIndex = AC->createIndexConstant(0, loc);
      allocOp =
          AC->create<memref::AllocaOp>(loc, dynamicType, ValueRange{zeroIndex});
    } else if (resultType.getNumDynamicDims() > 0) {
      auto numValues = AC->createIndexConstant(values.size(), loc);
      allocOp =
          AC->create<memref::AllocaOp>(loc, resultType, ValueRange{numValues});
    } else {
      allocOp = AC->create<memref::AllocaOp>(loc, resultType, ValueRange{});
    }

    for (unsigned i = 0; i < values.size(); ++i) {
      auto index = AC->createIndexConstant(i, loc);
      auto castVal = AC->castParameter(AC->Int64, values[i], loc,
                                       ArtsCodegen::ParameterCastMode::Bitwise);
      AC->create<memref::StoreOp>(loc, castVal, allocOp, ValueRange{index});
    }

    rewriter.replaceOp(op, allocOp);
    ++numMiscOpsConverted;
    return success();
  }
};

/// Pattern to lower arts.state_unpack to loads + casts.
/// Each i64 element is loaded from the state memref and cast back to the
/// target type.
struct StateUnpackPattern : public ArtsToLLVMPattern<StateUnpackOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(StateUnpackOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering StateUnpack Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto stateMemref = op.getState();
    auto results = op.getValues();

    SmallVector<Value> newResults;
    for (unsigned i = 0; i < results.size(); ++i) {
      auto idx = AC->createIndexConstant(i, op.getLoc());
      auto loadedVal =
          AC->create<memref::LoadOp>(op.getLoc(), stateMemref, ValueRange{idx});
      auto castedVal =
          AC->castParameter(results[i].getType(), loadedVal, op.getLoc(),
                            ArtsCodegen::ParameterCastMode::Bitwise);
      newResults.push_back(castedVal);
    }

    rewriter.replaceOp(op, newResults);
    ++numMiscOpsConverted;
    return success();
  }
};

/// Pattern to lower arts.dep_bind to a pass-through.
/// At LLVM level, the GUID itself is the slot identifier.
struct DepBindPattern : public ArtsToLLVMPattern<DepBindOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DepBindOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DepBind Op " << op);
    rewriter.replaceOp(op, op.getGuid());
    ++numMiscOpsConverted;
    return success();
  }
};

/// Pattern to lower arts.dep_forward to identity (pass-through).
/// At runtime level the forwarded dependency slot value is unchanged.
struct DepForwardPattern : public ArtsToLLVMPattern<DepForwardOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DepForwardOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DepForward Op " << op);
    rewriter.replaceOp(op, op.getSlot());
    ++numMiscOpsConverted;
    return success();
  }
};

///===----------------------------------------------------------------------===///
/// Pattern Population
///===----------------------------------------------------------------------===///

namespace mlir::carts::arts::convert_arts_to_llvm {

void populateRtToLLVMPatterns(RewritePatternSet &patterns, ArtsCodegen *AC) {
  MLIRContext *context = patterns.getContext();

  /// Epoch patterns
  patterns.add<CreateEpochPattern, WaitOnEpochPattern>(context, AC);

  /// EDT patterns
  patterns.add<EdtParamPackPattern, EdtParamUnpackPattern>(context, AC);
  patterns.add<EdtCreatePattern>(context, AC);

  /// Dep patterns
  patterns.add<DepGepOpPattern>(context, AC);
  patterns.add<RecordDepPattern, IncrementDepPattern>(context, AC);
  patterns.add<DepDbAcquireOpPattern>(context, AC);
  patterns.add<DbGepOpPattern>(context, AC);

  /// State and bind patterns
  patterns.add<StatePackPattern, StateUnpackPattern>(context, AC);
  patterns.add<DepBindPattern, DepForwardPattern>(context, AC);
}

} // namespace mlir::carts::arts::convert_arts_to_llvm
