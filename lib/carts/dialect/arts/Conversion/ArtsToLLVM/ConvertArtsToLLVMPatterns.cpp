///==========================================================================///
/// File: ConvertArtsToLLVMPatterns.cpp
///
/// ARTS-to-LLVM conversion patterns extracted from ConvertArtsToLLVM.cpp.
/// Each pattern struct converts one ARTS dialect operation into LLVM
/// runtime calls via the ArtsCodegen infrastructure.
///==========================================================================///

#include "carts/dialect/arts/Conversion/ArtsToLLVM/ConvertArtsToLLVMInternal.h"

#include "carts/dialect/arts/Conversion/ArtsToLLVM/CodegenSupport.h"
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

#include <limits>

#include "carts/utils/Debug.h"
#include "llvm/ADT/Statistic.h"
ARTS_DEBUG_SETUP(convert_arts_to_llvm);

static llvm::Statistic numEdtOpsConverted{
    "convert_arts_to_llvm", "NumEdtOpsConverted",
    "Number of EDT operations converted to LLVM runtime calls"};
static llvm::Statistic numDbOpsConverted{
    "convert_arts_to_llvm", "NumDbOpsConverted",
    "Number of DataBlock operations converted to LLVM runtime calls"};
static llvm::Statistic numEpochOpsConverted{
    "convert_arts_to_llvm", "NumEpochOpsConverted",
    "Number of epoch operations converted to LLVM runtime calls"};
static llvm::Statistic numDepOpsConverted{
    "convert_arts_to_llvm", "NumDepOpsConverted",
    "Number of dependency operations converted to LLVM runtime calls"};
static llvm::Statistic numMiscOpsConverted{
    "convert_arts_to_llvm", "NumMiscOpsConverted",
    "Number of miscellaneous ARTS operations converted"};

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::arts::convert_arts_to_llvm;

///===----------------------------------------------------------------------===///
/// Helper Functions
///===----------------------------------------------------------------------===///

namespace mlir::arts::convert_arts_to_llvm {

SmallVector<Value, 4>
materializeStaticDbOuterShape(Value handle, ArtsCodegen *AC, Location loc) {
  SmallVector<Value, 4> sizes;
  auto shape = getDbStaticOuterShape(handle);
  if (!shape)
    return sizes;

  sizes.reserve(shape->size());
  for (int64_t dim : *shape)
    sizes.push_back(AC->createIndexConstant(dim, loc));
  return sizes;
}

SmallVector<Value, 4> resolveSourceOuterSizes(Value sourceGuid, Value sourcePtr,
                                              ArtsCodegen *AC, Location loc) {
  SmallVector<Value, 4> sizes;

  auto resolveFromDefiningDb = [&](Value handle) -> bool {
    handle = ValueAnalysis::stripMemrefViewOps(handle);
    if (!handle)
      return false;

    Operation *def = handle.getDefiningOp();
    if (auto allocOp = dyn_cast_or_null<DbAllocOp>(def)) {
      sizes.assign(allocOp.getSizes().begin(), allocOp.getSizes().end());
      return true;
    }
    if (auto acqOp = dyn_cast_or_null<DbAcquireOp>(def)) {
      sizes.assign(acqOp.getSizes().begin(), acqOp.getSizes().end());
      return true;
    }
    if (auto depAcqOp = dyn_cast_or_null<DepDbAcquireOp>(def)) {
      sizes.assign(depAcqOp.getSizes().begin(), depAcqOp.getSizes().end());
      return true;
    }
    return false;
  };

  if (sourceGuid)
    resolveFromDefiningDb(sourceGuid);
  if (sizes.empty() && sourcePtr)
    resolveFromDefiningDb(sourcePtr);

  if (sizes.empty())
    sizes = materializeStaticDbOuterShape(sourceGuid, AC, loc);
  if (sizes.empty())
    sizes = materializeStaticDbOuterShape(sourcePtr, AC, loc);
  return sizes;
}

Value resolveGuidStorageForAcquireSource(Value sourceGuid, Value sourcePtr) {
  if (sourceGuid)
    return sourceGuid;

  sourcePtr = ValueAnalysis::stripMemrefViewOps(sourcePtr);
  if (!sourcePtr)
    return {};

  Operation *def = sourcePtr.getDefiningOp();
  if (auto allocOp = dyn_cast_or_null<DbAllocOp>(def))
    return allocOp.getGuid();
  if (auto acquireOp = dyn_cast_or_null<DbAcquireOp>(def))
    return acquireOp.getGuid();
  if (auto depAcquireOp = dyn_cast_or_null<DepDbAcquireOp>(def))
    return depAcquireOp.getGuid();
  if (auto dbRefOp = dyn_cast_or_null<DbRefOp>(def))
    return resolveGuidStorageForAcquireSource({}, dbRefOp.getSource());

  Operation *underlying = DbUtils::getUnderlyingDb(sourcePtr);
  if (auto allocOp = dyn_cast_or_null<DbAllocOp>(underlying))
    return allocOp.getGuid();
  if (auto acquireOp = dyn_cast_or_null<DbAcquireOp>(underlying))
    return acquireOp.getGuid();
  if (auto depAcquireOp = dyn_cast_or_null<DepDbAcquireOp>(underlying))
    return depAcquireOp.getGuid();

  return {};
}

SmallVector<Value, 4> resolveOuterSizesForGuid(Value dbGuid, ArtsCodegen *AC,
                                               Location loc) {
  SmallVector<Value, 4> sizes;
  if (!dbGuid)
    return sizes;

  if (auto allocOp = DbUtils::getAllocOpFromGuid(dbGuid)) {
    sizes.assign(allocOp.getSizes().begin(), allocOp.getSizes().end());
    return sizes;
  }

  if (auto dbAcquireOp = dbGuid.getDefiningOp<DbAcquireOp>())
    return resolveSourceOuterSizes(dbAcquireOp.getSourceGuid(),
                                   dbAcquireOp.getSourcePtr(), AC, loc);

  if (auto depDbAcquireOp = dbGuid.getDefiningOp<DepDbAcquireOp>())
    return resolveSourceOuterSizes(depDbAcquireOp.getGuid(),
                                   depDbAcquireOp.getPtr(), AC, loc);

  return sizes;
}

static bool isPointerTableElement(Type type) {
  return isa<LLVM::LLVMPointerType, MemRefType>(type);
}

static unsigned countDynamicDims(MemRefType type) {
  unsigned count = 0;
  for (int64_t dim : type.getShape())
    if (dim == ShapedType::kDynamic)
      ++count;
  return count;
}

static SmallVector<Value> materializeDbRefElementSizes(DbRefOp op,
                                                       ArtsCodegen *AC) {
  auto resultType = dyn_cast<MemRefType>(op.getResult().getType());
  if (!resultType)
    return {};

  DbAllocOp selectedAlloc = nullptr;
  if (auto *rawAlloc = DbUtils::getUnderlyingDbAlloc(op.getSource()))
    selectedAlloc = dyn_cast<DbAllocOp>(rawAlloc);
  if (selectedAlloc &&
      selectedAlloc.getElementSizes().size() !=
          static_cast<size_t>(resultType.getRank()))
    selectedAlloc = nullptr;

  if (!selectedAlloc) {
    Type elementType = resultType.getElementType();
    if (auto moduleOp = op->getParentOfType<ModuleOp>()) {
      moduleOp.walk([&](DbAllocOp alloc) {
        if (selectedAlloc)
          return WalkResult::interrupt();
        if (alloc.getElementType() == elementType &&
            alloc.getElementSizes().size() ==
                static_cast<size_t>(resultType.getRank()))
          selectedAlloc = alloc;
        return WalkResult::advance();
      });
    }
  }

  if (!selectedAlloc || selectedAlloc.getElementSizes().empty())
    return {};

  SmallVector<Value> sizes;
  sizes.reserve(selectedAlloc.getElementSizes().size());
  func::FuncOp ownerFunc = op->getParentOfType<func::FuncOp>();
  for (Value size : selectedAlloc.getElementSizes()) {
    int64_t constantSize;
    if (ValueAnalysis::getConstantIndex(size, constantSize)) {
      sizes.push_back(AC->createIndexConstant(constantSize, op.getLoc()));
      continue;
    }

    Operation *def = size.getDefiningOp();
    if (def && ownerFunc && def->getParentOfType<func::FuncOp>() == ownerFunc) {
      sizes.push_back(size);
      continue;
    }

    return {};
  }
  return sizes;
}

Value buildArtsHintMemref(ArtsCodegen *AC, Value route, Value artsId,
                          Location loc) {
  Value hintAlloc =
      AC->create<LLVM::AllocaOp>(loc, AC->llvmPtr, AC->ArtsHintType,
                                 AC->createIntConstant(1, AC->Int32, loc));

  ArtsHintBuilder hint =
      ArtsHintBuilder::undef(AC->getBuilder(), loc, AC->ArtsHintType);
  hint.setRoute(AC->getBuilder(), loc, route);
  hint.setArtsId(AC->getBuilder(), loc, artsId);
  AC->create<LLVM::StoreOp>(loc, Value(hint), hintAlloc);

  return AC->create<polygeist::Pointer2MemrefOp>(loc, AC->ArtsHintTypePtr,
                                                 hintAlloc);
}

} // namespace mlir::arts::convert_arts_to_llvm

///===----------------------------------------------------------------------===///
/// Conversion Patterns
///===----------------------------------------------------------------------===///

/// Pattern to convert arts.runtime_query operations to LLVM runtime calls.
/// Dispatches on the RuntimeQueryKind attribute to call the appropriate
/// ARTS runtime function (getTotalWorkers, getTotalNodes, etc.).
struct RuntimeQueryPattern : public ArtsToLLVMPattern<RuntimeQueryOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(RuntimeQueryOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering RuntimeQuery Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    Value result;
    switch (op.getKind()) {
    case RuntimeQueryKind::totalWorkers:
      result = AC->getTotalWorkers(op.getLoc());
      break;
    case RuntimeQueryKind::totalNodes:
      result = AC->getTotalNodes(op.getLoc());
      break;
    case RuntimeQueryKind::currentWorker:
      result = AC->getCurrentWorker(op.getLoc());
      break;
    case RuntimeQueryKind::currentNode:
      result = AC->getCurrentNode(op.getLoc());
      break;
    }
    rewriter.replaceOp(op, result);
    ++numMiscOpsConverted;
    return success();
  }
};

///===----------------------------------------------------------------------===///
/// Event and Barrier Patterns
///===----------------------------------------------------------------------===///

/// Pattern to convert arts.barrier operations
struct BarrierPattern : public ArtsToLLVMPattern<BarrierOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(BarrierOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering Barrier Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    ArtsCodegen::RuntimeCallBuilder RCB(*AC, op.getLoc());
    RCB.callVoid(types::ARTSRTL_arts_yield, {});
    rewriter.eraseOp(op);
    ++numMiscOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.get_edt_epoch_guid operations
struct GetEdtEpochGuidPattern : public ArtsToLLVMPattern<GetEdtEpochGuidOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(GetEdtEpochGuidOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering GetEdtEpochGuid Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    Value result = AC->getEdtEpochGuid(op.getLoc());
    rewriter.replaceOp(op, result);
    ++numEpochOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.create_epoch operations
/// Pattern to convert arts.atomic_add operations
struct AtomicAddPattern : public ArtsToLLVMPattern<AtomicAddOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(AtomicAddOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering AtomicAdd Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);

    auto loc = op.getLoc();
    auto addr = op.getAddr();
    auto value = op.getValue();

    /// Convert memref to LLVM pointer
    Value llvmPtr = polygeist::Memref2PointerOp::create(
        rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()), addr);

    /// Create LLVM atomic add operation
    LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::add, llvmPtr,
                              value, LLVM::AtomicOrdering::seq_cst);

    rewriter.eraseOp(op);
    ++numMiscOpsConverted;
    return success();
  }
};

/// Pattern to convert __builtin_object_size calls to llvm.objectsize intrinsic.
///
/// Polygeist's cgeist bypasses Clang's CodeGen, so __builtin_object_size is
/// emitted as a regular function call instead of the llvm.objectsize intrinsic.
///
/// __builtin_object_size(ptr, type) -> llvm.objectsize(ptr, min, null_unknown,
/// dynamic)
///
/// Parameter mapping (from Clang's CGBuiltin.cpp):
///   - ptr: First argument
///   - min: (type & 2) != 0  (bit 1 of type parameter)
///   - null_is_unknown: Always true to match GCC semantics
///   - dynamic: false (true only for __builtin_dynamic_object_size)
struct BuiltinObjectSizePattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto callee = callOp.getCallee();
    if (callee != "__builtin_object_size")
      return failure();

    Location loc = callOp.getLoc();

    /// Get ptr argument (memref) and convert to llvm.ptr
    Value memrefArg = callOp.getOperand(0);
    Value ptr = polygeist::Memref2PointerOp::create(
        rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
        memrefArg);

    Value typeArg = callOp.getOperand(1);

    /// Extract type value to compute min flag: min = (type & 2) != 0
    bool minFlag = false;
    if (auto constOp = typeArg.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        minFlag = (intAttr.getInt() & 2) != 0;
      }
    }

    /// Create boolean constants for intrinsic parameters
    Type i1Type = rewriter.getI1Type();
    Value min = LLVM::ConstantOp::create(rewriter, loc, i1Type, minFlag);
    Value nullIsUnknown = LLVM::ConstantOp::create(rewriter, loc, i1Type, true);
    Value dynamic = LLVM::ConstantOp::create(rewriter, loc, i1Type, false);

    /// Create llvm.call_intrinsic for llvm.objectsize
    Type resultType = callOp.getResult(0).getType();
    auto intrinsicOp = LLVM::CallIntrinsicOp::create(
        rewriter, loc, resultType,
        StringAttr::get(rewriter.getContext(), "llvm.objectsize"),
        ValueRange{ptr, min, nullIsUnknown, dynamic});

    rewriter.replaceOp(callOp, intrinsicOp.getResults());
    ++numMiscOpsConverted;
    return success();
  }
};

///===----------------------------------------------------------------------===///
/// DB Patterns
///===----------------------------------------------------------------------===///
/// Pattern to convert arts.db_alloc operations
struct DbAllocPattern : public ArtsToLLVMPattern<DbAllocOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbAllocOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbAlloc Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto loc = op.getLoc();
    Value guidMemref, dbMemref;

    Value route = op.getRoute();
    if (!route)
      route = createCurrentNodeRoute(AC->getBuilder(), loc);
    bool distributedOwnership = hasDistributedDbAllocation(op.getOperation());
    Value elementSize = AC->computeElementTypeSize(op.getElementType(), loc);

    /// Compute payload size from elementSizes (product of all payload
    /// dimensions)
    Value payloadSize = AC->createIndexConstant(1, loc);
    for (Value payloadDim : op.getElementSizes())
      payloadSize = AC->create<arith::MulIOp>(loc, payloadSize, payloadDim);

    /// Total datablock size = elementSize * payloadSize
    Value totalDbSize =
        AC->create<arith::MulIOp>(loc, elementSize, payloadSize);
    std::optional<int64_t> nextId;
    if (auto createIdAttr =
            op->getAttrOfType<IntegerAttr>(AttrNames::Operation::ArtsCreateId))
      nextId = createIdAttr.getInt();
    else
      nextId = getArtsId(op);

    DbLoweringInfo dbLowering = DbUtils::extractDbLoweringInfo(op);
    auto &dbSizes = dbLowering.sizes;
    bool isSingleElement = dbLowering.isSingleElement;
    DbMemoryPlacement dbMemoryPlacement = getDbMemoryPlacement(
        op, dbSizes, isSingleElement, distributedOwnership);

    if (distributedOwnership) {
      if (failed(lowerDistributedRuntimeDbAlloc(
              op, nextId, dbSizes, isSingleElement, guidMemref, dbMemref)))
        return failure();
      if (failed(linearizeRankedHandleUses(op.getGuid(), guidMemref, dbSizes,
                                           rewriter)) ||
          failed(linearizeRankedHandleUses(op.getPtr(), dbMemref, dbSizes,
                                           rewriter)))
        return failure();
      rewriter.replaceOp(op, {guidMemref, dbMemref});
      ++numDbOpsConverted;
      return success();
    }

    if (isSingleElement) {
      ARTS_DEBUG("Creating single DB");
      /// Allocate 1D linear array of GUIDs
      Value totalElems = AC->createIndexConstant(1, loc);
      auto guidType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
      guidMemref =
          AC->create<memref::AllocOp>(loc, guidType, ValueRange{totalElems});
      /// Allocate 1D linear array of DB Pointers
      auto payloadPtrType =
          MemRefType::get({ShapedType::kDynamic}, AC->llvmPtr);
      dbMemref = AC->create<memref::AllocOp>(loc, payloadPtrType,
                                             ValueRange{totalElems});
      createSingleDb(dbMemref, guidMemref, route, totalDbSize,
                     nextId ? &nextId : nullptr, loc, distributedOwnership,
                     /*createDb=*/true, DbMemoryPlacement::Default);
    } else {
      ARTS_DEBUG("Creating multi-dim DB");
      /// Compute total number of elements
      Value totalElems = AC->computeTotalElements(dbSizes, loc);
      /// Allocate 1D linear array of GUIDs
      auto guidType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
      guidMemref =
          AC->create<memref::AllocOp>(loc, guidType, ValueRange{totalElems});
      /// Allocate 1D linear array of DB Pointers
      auto payloadPtrType =
          MemRefType::get({ShapedType::kDynamic}, AC->llvmPtr);
      dbMemref = AC->create<memref::AllocOp>(loc, payloadPtrType,
                                             ValueRange{totalElems});
      createMultiDbs(dbMemref, guidMemref, dbSizes, route, totalDbSize,
                     nextId ? &nextId : nullptr, loc, distributedOwnership,
                     /*createDb=*/true, dbMemoryPlacement);
    }

    if (failed(linearizeRankedHandleUses(op.getGuid(), guidMemref, dbSizes,
                                         rewriter)) ||
        failed(linearizeRankedHandleUses(op.getPtr(), dbMemref, dbSizes,
                                         rewriter)))
      return failure();

    rewriter.replaceOp(op, {guidMemref, dbMemref});
    ++numDbOpsConverted;
    return success();
  }

private:
  enum class DbMemoryPlacement { Default, Interleaved };

  LogicalResult linearizeRankedHandleUses(Value original, Value flatMemref,
                                          ArrayRef<Value> dbSizes,
                                          PatternRewriter &rewriter) const {
    auto originalType = dyn_cast<MemRefType>(original.getType());
    if (!originalType || originalType.getRank() <= 1)
      return success();
    if (dbSizes.size() != static_cast<size_t>(originalType.getRank()))
      return failure();

    for (OpOperand &use : llvm::make_early_inc_range(original.getUses())) {
      Operation *user = use.getOwner();

      if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
        if (loadOp.getMemref() != original)
          continue;
        if (loadOp.getIndices().size() <= 1)
          continue;
        if (loadOp.getIndices().size() != dbSizes.size())
          return loadOp.emitOpError(
              "cannot linearize DB handle load with rank/shape mismatch");
        AC->setInsertionPoint(loadOp);
        SmallVector<Value> indices(loadOp.getIndices().begin(),
                                   loadOp.getIndices().end());
        Value linearIndex =
            AC->computeLinearIndex(dbSizes, indices, loadOp.getLoc());
        Value linearLoad = AC->create<memref::LoadOp>(
            loadOp.getLoc(), flatMemref, ValueRange{linearIndex});
        rewriter.replaceOp(loadOp, linearLoad);
        continue;
      }

      if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
        if (storeOp.getMemref() != original)
          continue;
        if (storeOp.getIndices().size() <= 1)
          continue;
        if (storeOp.getIndices().size() != dbSizes.size())
          return storeOp.emitOpError(
              "cannot linearize DB handle store with rank/shape mismatch");
        AC->setInsertionPoint(storeOp);
        SmallVector<Value> indices(storeOp.getIndices().begin(),
                                   storeOp.getIndices().end());
        Value linearIndex =
            AC->computeLinearIndex(dbSizes, indices, storeOp.getLoc());
        AC->create<memref::StoreOp>(storeOp.getLoc(),
                                    storeOp.getValueToStore(), flatMemref,
                                    ValueRange{linearIndex});
        rewriter.eraseOp(storeOp);
      }
    }

    return success();
  }

  memref::GlobalOp getOrCreateMemrefGlobal(StringRef symbolName,
                                           MemRefType type,
                                           Location loc) const {
    ModuleOp module = AC->getModule();
    if (auto existing = module.lookupSymbol<memref::GlobalOp>(symbolName))
      return existing;

    OpBuilder::InsertionGuard IG(AC->getBuilder());
    AC->setInsertionPoint(module);
    auto global = AC->create<memref::GlobalOp>(
        loc, symbolName, StringAttr(), type, Attribute(),
        /*constant=*/false, IntegerAttr());
    global.setSymVisibility("private");
    return global;
  }

  LogicalResult
  lowerDistributedRuntimeDbAlloc(DbAllocOp op, std::optional<int64_t> nextId,
                                 ArrayRef<Value> dbSizes, bool isSingleElement,
                                 Value &guidMemref, Value &dbMemref) const {
    uint64_t baseId = getArtsId(op);
    if (baseId == 0)
      baseId =
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(op.getOperation()));
    std::string baseName = "__carts_dist_alloc_" + std::to_string(baseId);
    std::string guidHolderSymbol = baseName + "_guid_holder";
    std::string ptrHolderSymbol = baseName + "_ptr_holder";
    bool parallelInit = AC->useDistributedInitInWorkers();
    std::string nodeInitSymbol =
        baseName + (parallelInit ? "_reserve_init" : "_init");
    std::string workerInitSymbol = baseName + "_worker_init";

    auto guidDynamicType = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
    auto ptrDynamicType = MemRefType::get({ShapedType::kDynamic}, AC->llvmPtr);
    auto guidHolderType = MemRefType::get({1}, guidDynamicType);
    auto ptrHolderType = MemRefType::get({1}, ptrDynamicType);
    getOrCreateMemrefGlobal(guidHolderSymbol, guidHolderType, op.getLoc());
    getOrCreateMemrefGlobal(ptrHolderSymbol, ptrHolderType, op.getLoc());

    ModuleOp module = AC->getModule();
    func::FuncOp initFn = module.lookupSymbol<func::FuncOp>(nodeInitSymbol);
    func::FuncOp workerInitFn;
    if (!initFn) {
      OpBuilder::InsertionGuard IG(AC->getBuilder());
      AC->setInsertionPoint(module);
      initFn = AC->create<func::FuncOp>(op.getLoc(), nodeInitSymbol,
                                        AC->InitPerNodeFn);
      initFn.setPrivate();

      Block *entry = initFn.addEntryBlock();
      AC->setInsertionPointToStart(entry);

      IRMapping mapper;
      auto parentFn = op->getParentOfType<func::FuncOp>();
      if (parentFn && parentFn.getNumArguments() > 0) {
        if (parentFn.getNumArguments() != 2 || initFn.getNumArguments() != 3)
          return failure();
        mapper.map(parentFn.getArgument(0), initFn.getArgument(1));
        mapper.map(parentFn.getArgument(1), initFn.getArgument(2));
      }

      llvm::SetVector<Value> valuesToClone;
      llvm::DenseSet<Value> visited;
      std::function<LogicalResult(Value)> collectDependencies =
          [&](Value value) -> LogicalResult {
        if (!value)
          return failure();
        if (!visited.insert(value).second || mapper.contains(value))
          return success();

        if (auto blockArg = dyn_cast<BlockArgument>(value))
          return mapper.contains(blockArg) ? success() : failure();

        Operation *defOp = value.getDefiningOp();
        if (!defOp)
          return failure();
        for (Value operand : defOp->getOperands()) {
          if (failed(collectDependencies(operand)))
            return failure();
        }
        valuesToClone.insert(value);
        return success();
      };

      for (Value size : dbSizes) {
        if (failed(collectDependencies(size))) {
          ARTS_WARN("Failed to collect distributed allocation size expression "
                    "dependencies for "
                    << op);
          return failure();
        }
      }
      for (Value size : op.getElementSizes()) {
        if (failed(collectDependencies(size))) {
          ARTS_WARN("Failed to collect distributed allocation element size "
                    "expression dependencies for "
                    << op);
          return failure();
        }
      }

      auto isRuntimeTopologyCall = [](Operation *cloneOp) {
        if (auto queryOp = dyn_cast<RuntimeQueryOp>(cloneOp)) {
          auto kind = queryOp.getKind();
          return kind == RuntimeQueryKind::totalNodes ||
                 kind == RuntimeQueryKind::totalWorkers;
        }
        auto callOp = dyn_cast<func::CallOp>(cloneOp);
        if (!callOp)
          return false;
        auto callee = callOp.getCallee();
        return callee == "artsGetTotalNodes" || callee == "artsGetTotalWorkers";
      };

      if (!ValueAnalysis::cloneValuesIntoRegion(
              valuesToClone, &initFn.getBody(), mapper, AC->getBuilder(),
              /*allowMemoryEffectFree=*/true, isRuntimeTopologyCall)) {
        ARTS_WARN(
            "Failed to clone distributed allocation size expressions into "
            "init callback for "
            << op);
        return failure();
      }

      auto mapValue = [&](Value v) { return mapper.lookupOrNull(v); };

      SmallVector<Value, 4> callbackDbSizes;
      callbackDbSizes.reserve(dbSizes.size());
      for (Value size : dbSizes) {
        Value mapped = mapValue(size);
        if (!mapped) {
          ARTS_WARN("Missing mapped DbAlloc size value in distributed init for "
                    << op);
          return failure();
        }
        callbackDbSizes.push_back(mapped);
      }

      SmallVector<Value, 4> callbackElementSizes;
      callbackElementSizes.reserve(op.getElementSizes().size());
      for (Value size : op.getElementSizes()) {
        Value mapped = mapValue(size);
        if (!mapped) {
          ARTS_WARN("Missing mapped DbAlloc element size value in distributed "
                    "init for "
                    << op);
          return failure();
        }
        callbackElementSizes.push_back(mapped);
      }

      Value callbackTotalElems =
          AC->computeTotalElements(callbackDbSizes, op.getLoc());
      Value guidBuffer = AC->create<memref::AllocOp>(
          op.getLoc(), guidDynamicType, ValueRange{callbackTotalElems});
      Value ptrBuffer = AC->create<memref::AllocOp>(
          op.getLoc(), ptrDynamicType, ValueRange{callbackTotalElems});

      Value guidHolder = AC->create<memref::GetGlobalOp>(
          op.getLoc(), guidHolderType, guidHolderSymbol);
      Value ptrHolder = AC->create<memref::GetGlobalOp>(
          op.getLoc(), ptrHolderType, ptrHolderSymbol);
      Value zero = AC->createIndexConstant(0, op.getLoc());
      AC->create<memref::StoreOp>(op.getLoc(), guidBuffer, guidHolder,
                                  ValueRange{zero});
      AC->create<memref::StoreOp>(op.getLoc(), ptrBuffer, ptrHolder,
                                  ValueRange{zero});

      Value callbackElementSize =
          AC->computeElementTypeSize(op.getElementType(), op.getLoc());
      Value callbackPayloadSize = AC->createIndexConstant(1, op.getLoc());
      for (Value dim : callbackElementSizes) {
        callbackPayloadSize =
            AC->create<arith::MulIOp>(op.getLoc(), callbackPayloadSize, dim);
      }
      Value callbackTotalDbSize = AC->create<arith::MulIOp>(
          op.getLoc(), callbackElementSize, callbackPayloadSize);
      Value callbackRoute = AC->createIntConstant(0, AC->Int32, op.getLoc());
      std::optional<int64_t> callbackNextId = nextId;

      if (!parallelInit) {
        if (isSingleElement) {
          createSingleDb(
              ptrBuffer, guidBuffer, callbackRoute, callbackTotalDbSize,
              callbackNextId ? &callbackNextId : nullptr, op.getLoc(),
              /*distributedOwnership=*/true);
        } else {
          createMultiDbs(ptrBuffer, guidBuffer, callbackDbSizes, callbackRoute,
                         callbackTotalDbSize,
                         callbackNextId ? &callbackNextId : nullptr,
                         op.getLoc(),
                         /*distributedOwnership=*/true);
        }
        AC->create<func::ReturnOp>(op.getLoc());
      } else {
        /// Reserve deterministic GUIDs once per node; DB creation runs in
        /// initPerWorker.
        if (isSingleElement) {
          createSingleDb(
              ptrBuffer, guidBuffer, callbackRoute, callbackTotalDbSize,
              callbackNextId ? &callbackNextId : nullptr, op.getLoc(),
              /*distributedOwnership=*/true,
              /*createDb=*/false);
        } else {
          createMultiDbs(ptrBuffer, guidBuffer, callbackDbSizes, callbackRoute,
                         callbackTotalDbSize,
                         callbackNextId ? &callbackNextId : nullptr,
                         op.getLoc(),
                         /*distributedOwnership=*/true,
                         /*createDb=*/false);
        }
        AC->create<func::ReturnOp>(op.getLoc());

        workerInitFn = module.lookupSymbol<func::FuncOp>(workerInitSymbol);
        if (!workerInitFn) {
          AC->setInsertionPoint(module);
          workerInitFn = AC->create<func::FuncOp>(op.getLoc(), workerInitSymbol,
                                                  AC->InitPerWorkerFn);
          workerInitFn.setPrivate();

          Block *workerEntry = workerInitFn.addEntryBlock();
          AC->setInsertionPointToStart(workerEntry);

          IRMapping workerMapper;
          if (parentFn && parentFn.getNumArguments() > 0) {
            if (parentFn.getNumArguments() != 2 ||
                workerInitFn.getNumArguments() != 4)
              return failure();
            workerMapper.map(parentFn.getArgument(0),
                             workerInitFn.getArgument(2));
            workerMapper.map(parentFn.getArgument(1),
                             workerInitFn.getArgument(3));
          }

          llvm::SetVector<Value> workerValuesToClone;
          llvm::DenseSet<Value> workerVisited;
          std::function<LogicalResult(Value)> collectWorkerDependencies =
              [&](Value value) -> LogicalResult {
            if (!value)
              return failure();
            if (!workerVisited.insert(value).second ||
                workerMapper.contains(value))
              return success();

            if (auto blockArg = dyn_cast<BlockArgument>(value))
              return workerMapper.contains(blockArg) ? success() : failure();

            Operation *defOp = value.getDefiningOp();
            if (!defOp)
              return failure();
            for (Value operand : defOp->getOperands()) {
              if (failed(collectWorkerDependencies(operand)))
                return failure();
            }
            workerValuesToClone.insert(value);
            return success();
          };

          for (Value size : dbSizes) {
            if (failed(collectWorkerDependencies(size))) {
              ARTS_WARN("Failed to collect distributed worker init size "
                        "expression dependencies for "
                        << op);
              return failure();
            }
          }

          for (Value size : op.getElementSizes()) {
            if (failed(collectWorkerDependencies(size))) {
              ARTS_WARN("Failed to collect distributed worker init element "
                        "size expression dependencies for "
                        << op);
              return failure();
            }
          }

          if (!ValueAnalysis::cloneValuesIntoRegion(
                  workerValuesToClone, &workerInitFn.getBody(), workerMapper,
                  AC->getBuilder(), /*allowMemoryEffectFree=*/true,
                  isRuntimeTopologyCall)) {
            ARTS_WARN("Failed to clone distributed allocation size "
                      "expressions into worker init callback for "
                      << op);
            return failure();
          }

          auto mapWorkerValue = [&](Value v) {
            return workerMapper.lookupOrNull(v);
          };

          SmallVector<Value, 4> workerDbSizes;
          workerDbSizes.reserve(dbSizes.size());
          for (Value size : dbSizes) {
            Value mapped = mapWorkerValue(size);
            if (!mapped) {
              ARTS_WARN("Missing mapped DbAlloc size value in distributed "
                        "worker init for "
                        << op);
              return failure();
            }
            workerDbSizes.push_back(mapped);
          }

          SmallVector<Value, 4> workerElementSizes;
          workerElementSizes.reserve(op.getElementSizes().size());
          for (Value size : op.getElementSizes()) {
            Value mapped = mapWorkerValue(size);
            if (!mapped) {
              ARTS_WARN("Missing mapped DbAlloc element size value in "
                        "distributed worker init for "
                        << op);
              return failure();
            }
            workerElementSizes.push_back(mapped);
          }

          Value workerGuidHolder = AC->create<memref::GetGlobalOp>(
              op.getLoc(), guidHolderType, guidHolderSymbol);
          Value workerPtrHolder = AC->create<memref::GetGlobalOp>(
              op.getLoc(), ptrHolderType, ptrHolderSymbol);
          Value zeroIdx = AC->createIndexConstant(0, op.getLoc());
          Value workerGuidBuffer = AC->create<memref::LoadOp>(
              op.getLoc(), workerGuidHolder, ValueRange{zeroIdx});
          Value workerPtrBuffer = AC->create<memref::LoadOp>(
              op.getLoc(), workerPtrHolder, ValueRange{zeroIdx});
          Value workerTotalElems =
              AC->computeTotalElements(workerDbSizes, op.getLoc());

          Value workerElementSize =
              AC->computeElementTypeSize(op.getElementType(), op.getLoc());
          Value workerPayloadSize = AC->createIndexConstant(1, op.getLoc());
          for (Value dim : workerElementSizes) {
            workerPayloadSize =
                AC->create<arith::MulIOp>(op.getLoc(), workerPayloadSize, dim);
          }
          Value workerTotalDbSize = AC->create<arith::MulIOp>(
              op.getLoc(), workerElementSize, workerPayloadSize);

          Value workerNodeId = workerInitFn.getArgument(0);
          Value workerLocalId = workerInitFn.getArgument(1);
          Value zeroLoopIdx = AC->createIndexConstant(0, op.getLoc());
          Value oneIdx = AC->createIndexConstant(1, op.getLoc());
          Value zeroI32 = AC->createIntConstant(0, AC->Int32, op.getLoc());
          Value isPrimaryWorker = AC->create<arith::CmpIOp>(
              op.getLoc(), arith::CmpIPredicate::eq, workerLocalId, zeroI32);
          auto primaryWorkerIf =
              AC->create<scf::IfOp>(op.getLoc(), isPrimaryWorker, false);
          AC->setInsertionPointToStart(
              &primaryWorkerIf.getThenRegion().front());
          auto workerLoop = AC->create<scf::ForOp>(op.getLoc(), zeroLoopIdx,
                                                   workerTotalElems, oneIdx);
          AC->setInsertionPointToStart(&workerLoop.getRegion().front());
          Value linearIndex = workerLoop.getInductionVar();
          Value reservedGuid = AC->create<memref::LoadOp>(
              op.getLoc(), workerGuidBuffer, ValueRange{linearIndex});
          ArtsCodegen::RuntimeCallBuilder RCB(*AC, op.getLoc());
          Value guidRank =
              RCB.call(types::ARTSRTL_arts_guid_get_rank, {reservedGuid});
          Value isLocalGuid = AC->create<arith::CmpIOp>(
              op.getLoc(), arith::CmpIPredicate::eq, guidRank, workerNodeId);
          auto localGuidIf =
              AC->create<scf::IfOp>(op.getLoc(), isLocalGuid, false);
          AC->setInsertionPointToStart(&localGuidIf.getThenRegion().front());
          createDbFromGuidAtIndex(workerPtrBuffer, reservedGuid, linearIndex,
                                  workerTotalDbSize, callbackNextId,
                                  op.getLoc(), DbMemoryPlacement::Default);
          AC->setInsertionPointAfter(localGuidIf);
          AC->setInsertionPointAfter(workerLoop);
          AC->setInsertionPointAfter(primaryWorkerIf);
          AC->create<func::ReturnOp>(op.getLoc());
        }
      }
    }

    AC->registerDistributedInitNodeCallback(initFn);
    if (parallelInit) {
      if (!workerInitFn)
        workerInitFn = module.lookupSymbol<func::FuncOp>(workerInitSymbol);
      if (workerInitFn)
        AC->registerDistributedInitWorkerCallback(workerInitFn);
    }

    Value zero = AC->createIndexConstant(0, op.getLoc());
    Value guidHolder = AC->create<memref::GetGlobalOp>(
        op.getLoc(), guidHolderType, guidHolderSymbol);
    Value ptrHolder = AC->create<memref::GetGlobalOp>(
        op.getLoc(), ptrHolderType, ptrHolderSymbol);
    guidMemref =
        AC->create<memref::LoadOp>(op.getLoc(), guidHolder, ValueRange{zero});
    dbMemref =
        AC->create<memref::LoadOp>(op.getLoc(), ptrHolder, ValueRange{zero});
    return success();
  }

  void createDbFromGuidAtIndex(Value dbMemref, Value guid, Value linearIndex,
                               Value elementSize, std::optional<int64_t> nextId,
                               Location loc,
                               DbMemoryPlacement memoryPlacement) const {
    Value elemSize64 = AC->ensureI64(elementSize, loc);

    /// Build arts_hint_t with arts_id if available.
    Value artsIdValue;
    if (nextId.has_value()) {
      Value baseArtsId = AC->create<arith::ConstantOp>(
          loc, AC->Int64, AC->getBuilder().getI64IntegerAttr(*nextId));
      Value linearIndex64 = AC->ensureI64(linearIndex, loc);
      artsIdValue = AC->create<arith::AddIOp>(loc, baseArtsId, linearIndex64);
    } else {
      artsIdValue = AC->createIntConstant(0, AC->Int64, loc);
    }
    Value zeroRoute = AC->createIntConstant(0, AC->Int32, loc);
    Value hintMemref = buildArtsHintMemref(AC, zeroRoute, artsIdValue, loc);
    Value dbType = AC->createIntConstant(0, AC->Int32, loc);
    Value nullPtr = AC->create<LLVM::ZeroOp>(loc, AC->llvmPtr);
    Value nullData =
        AC->create<polygeist::Pointer2MemrefOp>(loc, AC->VoidPtr, nullPtr);

    ArtsCodegen::RuntimeCallBuilder RCB(*AC, loc);
    auto runtimeFn = types::ARTSRTL_arts_db_create_with_guid;
    if (memoryPlacement == DbMemoryPlacement::Interleaved)
      runtimeFn = types::ARTSRTL_arts_db_create_with_guid_interleaved;
    auto dbCall =
        RCB.callOp(runtimeFn, {guid, elemSize64, dbType, nullData, hintMemref});

    AC->create<memref::StoreOp>(loc, dbCall.getResult(0), dbMemref,
                                ValueRange{linearIndex});
  }

  void createSingleDb(
      Value dbMemref, Value guidMemref, Value route, Value elementSize,
      std::optional<int64_t> *nextId, Location loc,
      bool distributedOwnership = false, bool createDb = true,
      DbMemoryPlacement memoryPlacement = DbMemoryPlacement::Default,
      ArrayRef<Value> sizes = {},
      ArrayRef<Value> indices = {},
      std::optional<Value> linearIndexOverride = std::nullopt) const {
    Value linearIndex;
    if (linearIndexOverride.has_value()) {
      linearIndex = *linearIndexOverride;
    } else {
      linearIndex = indices.empty()
                        ? AC->createIndexConstant(0, loc)
                        : AC->computeLinearIndex(sizes, indices, loc);
    }

    Value reserveRoute = route;
    if (distributedOwnership) {
      Value linearIndexI32 = AC->castToInt(AC->Int32, linearIndex, loc);
      Value totalNodes = AC->getTotalNodes(loc);
      reserveRoute =
          AC->create<arith::RemUIOp>(loc, linearIndexI32, totalNodes);
    }

    /// Reserve GUID for the DB (v2: use ARTS_DB type for all datablocks)
    ArtsCodegen::RuntimeCallBuilder RCB(*AC, loc);
    Value dbTypeConst = AC->createIntConstant(ARTS_DB, AC->Int32, loc);
    auto guid =
        RCB.call(types::ARTSRTL_arts_guid_reserve, {dbTypeConst, reserveRoute});

    /// Store reserved GUID in the linearized guid memref.
    AC->create<memref::StoreOp>(loc, guid, guidMemref, ValueRange{linearIndex});

    /// Optionally create DB and store pointer in the linearized db memref.
    if (createDb) {
      std::optional<int64_t> baseId = std::nullopt;
      if (nextId && nextId->has_value()) {
        if (indices.empty() && !linearIndexOverride.has_value()) {
          baseId = **nextId;
          **nextId = **nextId + 1;
        } else {
          baseId = **nextId;
        }
      }
      createDbFromGuidAtIndex(dbMemref, guid, linearIndex, elementSize, baseId,
                              loc, memoryPlacement);
    }
  }

  void createMultiDbs(Value dbMemref, Value guidMemref, ArrayRef<Value> sizes,
                      Value route, Value elementSize,
                      std::optional<int64_t> *nextId, Location loc,
                      bool distributedOwnership = false,
                      bool createDb = true,
                      DbMemoryPlacement memoryPlacement =
                          DbMemoryPlacement::Default) const {
    Value totalElems = AC->computeTotalElements(sizes, loc);
    /// Keep DB creation always linearized here. The dedicated GuidRangCallOpt
    /// pass handles reserve->reserve_range promotion centrally after
    /// ARTS-to-LLVM conversion.
    auto lowerBound = AC->createIndexConstant(0, loc);
    auto step = AC->createIndexConstant(1, loc);
    auto linearLoop = AC->create<scf::ForOp>(loc, lowerBound, totalElems, step);
    auto &loopBlock = linearLoop.getRegion().front();
    AC->setInsertionPointToStart(&loopBlock);
    Value linearIndex = linearLoop.getInductionVar();
    createSingleDb(dbMemref, guidMemref, route, elementSize, nextId, loc,
                   distributedOwnership, createDb, memoryPlacement, /*sizes=*/{},
                   /*indices=*/{}, /*linearIndexOverride=*/linearIndex);
    AC->setInsertionPointAfter(linearLoop);
  }

  DbMemoryPlacement getDbMemoryPlacement(DbAllocOp op, ArrayRef<Value> dbSizes,
                                         bool isSingleElement,
                                         bool distributedOwnership) const {
    if (distributedOwnership || isSingleElement)
      return DbMemoryPlacement::Default;

    auto depPattern = getDepPattern(op.getOperation());
    bool hasBlockPlan = static_cast<bool>(
        getPlanPhysicalBlockShapeAttr(op.getOperation()));
    if (auto contract = getLoweringContract(op.getPtr())) {
      if (!depPattern && contract->pattern.depPattern)
        depPattern = contract->pattern.depPattern;
      hasBlockPlan = hasBlockPlan || !contract->spatial.blockShape.empty() ||
                     !contract->spatial.staticBlockShape.empty();
    }

    if (depPattern && isUniformFamilyDepPattern(*depPattern) &&
        hasMultipleRuntimeDbs(dbSizes)) {
      if (isLargeUniformRuntimeDbAllocation(op, dbSizes))
        return DbMemoryPlacement::Interleaved;
    }

    if (!depPattern || !isStencilHaloDepPattern(*depPattern))
      return DbMemoryPlacement::Default;

    if (!hasBlockPlan)
      return DbMemoryPlacement::Default;

    return hasMultipleRuntimeDbs(dbSizes) ? DbMemoryPlacement::Interleaved
                                          : DbMemoryPlacement::Default;
  }

  static constexpr int64_t kLargeUniformInterleaveBytes =
      16LL * 1024LL * 1024LL;

  static bool isLargeUniformRuntimeDbAllocation(DbAllocOp op,
                                                ArrayRef<Value> dbSizes) {
    std::optional<int64_t> payloadBytes = getStaticPayloadBytes(op);
    if (!payloadBytes)
      return false;
    if (*payloadBytes >= kLargeUniformInterleaveBytes)
      return true;

    std::optional<int64_t> runtimeDbCount = getStaticRuntimeDbCount(dbSizes);
    if (!runtimeDbCount || *runtimeDbCount <= 1)
      return false;
    if (*payloadBytes >
        std::numeric_limits<int64_t>::max() / *runtimeDbCount)
      return true;
    return *payloadBytes * *runtimeDbCount >= kLargeUniformInterleaveBytes;
  }

  static std::optional<int64_t> getStaticElementTypeBytes(Type type) {
    if (auto intTy = dyn_cast<IntegerType>(type))
      return std::max<int64_t>(1, (intTy.getWidth() + 7) / 8);
    if (auto floatTy = dyn_cast<FloatType>(type))
      return std::max<int64_t>(1, (floatTy.getWidth() + 7) / 8);
    if (isa<IndexType>(type))
      return int64_t{8};
    if (isa<LLVM::LLVMPointerType>(type))
      return int64_t{8};
    return std::nullopt;
  }

  static std::optional<int64_t> getStaticPayloadBytes(DbAllocOp op) {
    std::optional<int64_t> elemBytes =
        getStaticElementTypeBytes(op.getElementType());
    if (!elemBytes)
      return std::nullopt;

    int64_t payloadBytes = *elemBytes;
    for (Value dimValue : op.getElementSizes()) {
      std::optional<int64_t> dim = getConstantIntValue(dimValue);
      if (!dim || *dim <= 0)
        return std::nullopt;
      if (payloadBytes >
          std::numeric_limits<int64_t>::max() / std::max<int64_t>(1, *dim))
        return std::nullopt;
      payloadBytes *= *dim;
    }
    return payloadBytes;
  }

  static bool hasMultipleRuntimeDbs(ArrayRef<Value> dbSizes) {
    std::optional<int64_t> staticBlockCount = getStaticRuntimeDbCount(dbSizes);
    if (!staticBlockCount)
      return true;
    return *staticBlockCount > 1;
  }

  static std::optional<int64_t> getStaticRuntimeDbCount(
      ArrayRef<Value> dbSizes) {
    int64_t staticBlockCount = 1;
    for (Value size : dbSizes) {
      std::optional<int64_t> constant = getConstantIntValue(size);
      if (!constant)
        return std::nullopt;
      int64_t dim = *constant > 1 ? *constant : 1;
      if (staticBlockCount >
          std::numeric_limits<int64_t>::max() / std::max<int64_t>(1, dim))
        return std::nullopt;
      staticBlockCount *= dim;
    }
    return staticBlockCount;
  }
};

/// Pattern to convert residual arts.db_ref operations.
///
/// Most db_ref operations inside EDT bodies are consumed while lowering the EDT
/// region. Single-worker and already-outlined functions can leave db_ref
/// operations behind, where they must be lowered before their DbAcquireOp
/// source is rewritten to raw dependency storage.
struct DbRefPattern : public ArtsToLLVMPattern<DbRefOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbRefOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbRef Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    Location loc = op.getLoc();
    Value source = op.getSource();

    auto sourceType = dyn_cast<MemRefType>(source.getType());
    if (!sourceType)
      return failure();

    SmallVector<Value> indices(op.getIndices().begin(), op.getIndices().end());
    SmallVector<Value> outerSizes;
    if (Operation *underlyingDb = DbUtils::getUnderlyingDb(source))
      outerSizes = DbUtils::getSizesFromDb(underlyingDb);
    if (outerSizes.empty())
      outerSizes = materializeStaticDbOuterShape(source, AC, loc);

    SmallVector<Value> strides;
    if (!outerSizes.empty() && outerSizes.size() == indices.size())
      strides = AC->computeStridesFromSizes(outerSizes, loc);
    while (strides.size() < indices.size())
      strides.push_back(AC->createIndexConstant(1, loc));

    Value slotPtr =
        AC->create<DbGepOp>(loc, AC->llvmPtr, source, indices, strides).getPtr();

    Value payloadPtr = slotPtr;
    if (isPointerTableElement(sourceType.getElementType()))
      payloadPtr = AC->create<LLVM::LoadOp>(loc, AC->llvmPtr, slotPtr);

    Type resultType = op.getResult().getType();
    Value replacement;
    if (auto resultMemRefType = dyn_cast<MemRefType>(resultType)) {
      replacement =
          AC->create<polygeist::Pointer2MemrefOp>(loc, resultType, payloadPtr)
              .getResult();

      if (countDynamicDims(resultMemRefType) > 1) {
        SmallVector<Value> elementSizes = materializeDbRefElementSizes(op, AC);
        if (!elementSizes.empty()) {
          for (auto &use :
               llvm::make_early_inc_range(op.getResult().getUses())) {
            Operation *userOp = use.getOwner();
            AC->setInsertionPoint(userOp);
            if (auto loadOp = dyn_cast<memref::LoadOp>(userOp)) {
              SmallVector<Value> loadIndices(loadOp.getIndices().begin(),
                                             loadOp.getIndices().end());
              auto dynLoad = AC->create<polygeist::DynLoadOp>(
                  loadOp.getLoc(), loadOp.getResult().getType(), replacement,
                  loadIndices, elementSizes);
              rewriter.replaceOp(loadOp, dynLoad.getResult());
              continue;
            }
            if (auto storeOp = dyn_cast<memref::StoreOp>(userOp)) {
              SmallVector<Value> storeIndices(storeOp.getIndices().begin(),
                                              storeOp.getIndices().end());
              AC->create<polygeist::DynStoreOp>(
                  storeOp.getLoc(), storeOp.getValueToStore(), replacement,
                  storeIndices, elementSizes);
              rewriter.eraseOp(storeOp);
              continue;
            }
            use.set(replacement);
          }
          rewriter.eraseOp(op);
          ++numDbOpsConverted;
          return success();
        }
      }
    } else {
      replacement = AC->create<LLVM::LoadOp>(loc, resultType, payloadPtr);
    }

    rewriter.replaceOp(op, replacement);
    ++numDbOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.db_acquire operations
struct DbAcquirePattern : public ArtsToLLVMPattern<DbAcquireOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbAcquireOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbAcquire Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto loc = op.getLoc();

    Value sourceGuid = op.getSourceGuid();
    Value sourcePtr = op.getSourcePtr();
    Value guidStorage =
        resolveGuidStorageForAcquireSource(sourceGuid, sourcePtr);

    auto hasDbRefUser = [](Value value) {
      return value && llvm::any_of(value.getUsers(), [](Operation *user) {
               return isa<DbRefOp>(user);
             });
    };
    if (hasDbRefUser(op.getGuid()) || hasDbRefUser(op.getPtr()))
      return failure();

    /// Get source sizes from the defining op
    Value strippedGuidStorage = ValueAnalysis::stripMemrefViewOps(guidStorage);
    auto sourceOp =
        strippedGuidStorage ? strippedGuidStorage.getDefiningOp() : nullptr;
    SmallVector<Value> sourceSizes;
    if (auto allocOp = dyn_cast_or_null<DbAllocOp>(sourceOp))
      sourceSizes.assign(allocOp.getSizes().begin(), allocOp.getSizes().end());
    else if (auto acqOp = dyn_cast_or_null<DbAcquireOp>(sourceOp))
      sourceSizes.assign(acqOp.getSizes().begin(), acqOp.getSizes().end());
    else if (auto depAcqOp = dyn_cast_or_null<DepDbAcquireOp>(sourceOp))
      sourceSizes.assign(depAcqOp.getSizes().begin(),
                         depAcqOp.getSizes().end());
    if (sourceSizes.empty())
      sourceSizes = resolveSourceOuterSizes(guidStorage, sourcePtr, AC, loc);

    if (!guidStorage)
      return op.emitOpError("cannot lower without a source_guid or a "
                            "source_ptr that preserves a paired GUID handle");

    if (!sourceSizes.empty()) {
      SmallVector<Value> indices(op.getIndices().begin(),
                                 op.getIndices().end());
      SmallVector<Value> strides =
          AC->computeStridesFromSizes(sourceSizes, loc);
      /// Guid llvm type - underlying storage is always linear (rank-1)
      /// regardless of DbAcquireOp's multi-dimensional sizes
      auto origGuidMT = dyn_cast<MemRefType>(op.getGuid().getType());
      auto llvmGuidType = LLVM::LLVMPointerType::get(
          AC->getContext(), origGuidMT.getMemorySpaceAsInt());
      auto loadedGuid =
          AC->create<DbGepOp>(loc, llvmGuidType, guidStorage, indices, strides);
      /// Ptr llvm type - same: underlying storage is linear
      auto origPtrMT = dyn_cast<MemRefType>(op.getPtr().getType());
      auto llvmPtrType = LLVM::LLVMPointerType::get(
          AC->getContext(), origPtrMT.getMemorySpaceAsInt());
      auto loadedPtr =
          AC->create<DbGepOp>(loc, llvmPtrType, sourcePtr, indices, strides);
      /// Convert to memref type - use rank-1 linear memrefs since underlying
      /// storage from DbAllocOp is always linearized. The multi-dimensional
      /// sizes in DbAcquireOp are for logical indexing, not physical layout.
      auto guidMT = MemRefType::get({ShapedType::kDynamic}, AC->Int64);
      auto ptrMT = MemRefType::get({ShapedType::kDynamic}, AC->llvmPtr);
      auto guidMemref =
          AC->create<polygeist::Pointer2MemrefOp>(loc, guidMT, loadedGuid);
      auto ptrMemref =
          AC->create<polygeist::Pointer2MemrefOp>(loc, ptrMT, loadedPtr);
      rewriter.replaceOp(op, ValueRange{guidMemref, ptrMemref});
    } else {
      if (!op.getIndices().empty())
        return op.emitOpError(
            "cannot lower indexed acquire without source DB shape");
      /// Fallback: just forward the source values
      rewriter.replaceOp(op, ValueRange{guidStorage, sourcePtr});
    }

    ++numDbOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.db_release operations
struct DbReleasePattern : public ArtsToLLVMPattern<DbReleaseOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbReleaseOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbRelease Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    Value source = op.getSource();
    Operation *underlyingDb = DbUtils::getUnderlyingDb(source);
    Value guidStorage;
    if (auto alloc = dyn_cast_or_null<DbAllocOp>(underlyingDb))
      guidStorage = alloc.getGuid();
    else if (auto acquire = dyn_cast_or_null<DbAcquireOp>(underlyingDb))
      guidStorage = acquire.getGuid();
    else if (auto depAcquire = dyn_cast_or_null<DepDbAcquireOp>(underlyingDb))
      guidStorage = depAcquire.getGuid();

    if (guidStorage) {
      auto emitRelease = [&](Value guidValue) {
        guidValue = AC->ensureI64(guidValue, op.getLoc());
        ArtsCodegen::RuntimeCallBuilder RCB(*AC, op.getLoc());
        RCB.callVoid(types::ARTSRTL_arts_db_release, {guidValue});
      };

      if (auto storageTy = dyn_cast<MemRefType>(guidStorage.getType())) {
        SmallVector<Value> releaseSizes = DbUtils::getSizesFromDb(underlyingDb);
        if (releaseSizes.empty()) {
          Value zero = AC->createIndexConstant(0, op.getLoc());
          emitRelease(AC->create<memref::LoadOp>(op.getLoc(), guidStorage,
                                                 ValueRange{zero}));
        } else {
          SmallVector<Value> releaseOffsets;
          releaseOffsets.reserve(releaseSizes.size());
          for (size_t i = 0; i < releaseSizes.size(); ++i)
            releaseOffsets.push_back(AC->createIndexConstant(0, op.getLoc()));
          bool isSingle = releaseSizes.size() == 1;
          AC->iterateDbElements(
              guidStorage, Value(), releaseSizes, releaseOffsets, isSingle,
              op.getLoc(), [&](Value linearIndex) {
                Value idx = AC->castToIndex(linearIndex, op.getLoc());
                SmallVector<Value> indices;
                if (storageTy.getRank() == 1) {
                  indices.push_back(idx);
                } else {
                  indices = AC->computeIndicesFromLinearIndex(releaseSizes, idx,
                                                              op.getLoc());
                }
                emitRelease(AC->create<memref::LoadOp>(op.getLoc(), guidStorage,
                                                       indices));
              });
        }
      } else {
        emitRelease(guidStorage);
      }
    }
    rewriter.eraseOp(op);
    ++numDbOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.db_free operations
struct DbFreePattern : public ArtsToLLVMPattern<DbFreeOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbFreeOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbFree Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    Value source = op.getSource();
    Operation *underlyingDb = DbUtils::getUnderlyingDb(source);
    Value guidStorage;
    bool sourceIsGuidStorage = false;

    if (auto alloc = dyn_cast_or_null<DbAllocOp>(underlyingDb)) {
      guidStorage = alloc.getGuid();
      sourceIsGuidStorage = (source == alloc.getGuid());
    } else if (auto acquire = dyn_cast_or_null<DbAcquireOp>(underlyingDb)) {
      guidStorage = acquire.getGuid();
      sourceIsGuidStorage = (source == acquire.getGuid());
    } else if (auto depAcquire =
                   dyn_cast_or_null<DepDbAcquireOp>(underlyingDb)) {
      guidStorage = depAcquire.getGuid();
      sourceIsGuidStorage = (source == depAcquire.getGuid());
    }

    if (sourceIsGuidStorage && guidStorage) {
      auto emitDestroy = [&](Value guidValue) {
        guidValue = AC->ensureI64(guidValue, op.getLoc());
        ArtsCodegen::RuntimeCallBuilder RCB(*AC, op.getLoc());
        RCB.callVoid(types::ARTSRTL_arts_db_destroy, {guidValue});
      };

      if (auto storageTy = dyn_cast<MemRefType>(guidStorage.getType())) {
        SmallVector<Value> destroySizes = DbUtils::getSizesFromDb(underlyingDb);
        if (destroySizes.empty()) {
          Value zero = AC->createIndexConstant(0, op.getLoc());
          emitDestroy(AC->create<memref::LoadOp>(op.getLoc(), guidStorage,
                                                 ValueRange{zero}));
        } else {
          SmallVector<Value> destroyOffsets;
          destroyOffsets.reserve(destroySizes.size());
          for (size_t i = 0; i < destroySizes.size(); ++i)
            destroyOffsets.push_back(AC->createIndexConstant(0, op.getLoc()));
          bool isSingle = destroySizes.size() == 1;
          AC->iterateDbElements(
              guidStorage, Value(), destroySizes, destroyOffsets, isSingle,
              op.getLoc(), [&](Value linearIndex) {
                Value idx = AC->castToIndex(linearIndex, op.getLoc());
                SmallVector<Value> indices;
                if (storageTy.getRank() == 1) {
                  indices.push_back(idx);
                } else {
                  indices = AC->computeIndicesFromLinearIndex(destroySizes, idx,
                                                              op.getLoc());
                }
                emitDestroy(AC->create<memref::LoadOp>(op.getLoc(), guidStorage,
                                                       indices));
              });
        }
      } else {
        emitDestroy(guidStorage);
      }
    }

    Operation *rootOp = ValueAnalysis::getUnderlyingOperation(source);
    if (!isa_and_nonnull<memref::GetGlobalOp>(rootOp))
      AC->create<memref::DeallocOp>(op.getLoc(), source);
    rewriter.eraseOp(op);
    ++numDbOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.db_num_elements operations
struct DbNumElementsPattern : public ArtsToLLVMPattern<DbNumElementsOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(DbNumElementsOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering DbNumElements Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto loc = op.getLoc();

    /// Get the sizes from the operation arguments
    SmallVector<Value> sizes = op.getSizes();

    /// If no sizes are provided, return constant 1 (index)
    if (sizes.empty()) {
      rewriter.replaceOp(op, AC->createIndexConstant(1, loc));
      ++numDbOpsConverted;
      return success();
    }

    /// If all sizes are constants, fold to a single index constant.
    bool allConst = true;
    int64_t folded = 1;
    for (Value sz : sizes) {
      if (auto cst = dyn_cast_or_null<arith::ConstantOp>(sz.getDefiningOp())) {
        if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
          folded *= intAttr.getInt();
          continue;
        }
      }
      allConst = false;
      break;
    }
    if (allConst) {
      rewriter.replaceOp(op, AC->createIndexConstant(folded, loc));
      ++numDbOpsConverted;
      return success();
    }

    /// Otherwise, compute product at runtime as index.
    Value productVal = AC->createIndexConstant(1, loc);
    for (Value sz : sizes) {
      Value szIndex = AC->castToIndex(sz, loc);
      productVal = AC->create<arith::MulIOp>(loc, productVal, szIndex);
    }

    rewriter.replaceOp(op, productVal);
    ++numDbOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.alloc operations (ARTS memory allocation)
struct AllocPattern : public ArtsToLLVMPattern<AllocOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(AllocOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering Alloc Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    auto resultType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!resultType)
      return op.emitError("Expected MemRef type for result");
    ++numMiscOpsConverted;
    return success();
  }
};

///===----------------------------------------------------------------------===///
/// Terminator Patterns
///===----------------------------------------------------------------------===///

/// Pattern to convert arts.yield operations (terminator)
struct YieldPattern : public ArtsToLLVMPattern<YieldOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(YieldOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering Yield Op " << op);
    ArtsCodegen::RewriterGuard RG(*AC, rewriter);
    /// arts.yield is a terminator that should be erased during conversion
    rewriter.eraseOp(op);
    ++numMiscOpsConverted;
    return success();
  }
};

/// Pattern to convert arts.undef to polygeist.undef
struct UndefPattern : public ArtsToLLVMPattern<UndefOp> {
  using ArtsToLLVMPattern::ArtsToLLVMPattern;

  LogicalResult matchAndRewrite(UndefOp op,
                                PatternRewriter &rewriter) const override {
    ARTS_INFO("Lowering Undef Op " << op);
    Type resultType = op.getResult().getType();
    rewriter.replaceOpWithNewOp<polygeist::UndefOp>(op, resultType);
    ++numMiscOpsConverted;
    return success();
  }
};

///===----------------------------------------------------------------------===///
/// Split Launch State Lowering Patterns (Structured Kernel Plan, Phase 2)
///===----------------------------------------------------------------------===///

///===----------------------------------------------------------------------===///
/// Pattern Population
///===----------------------------------------------------------------------===///

namespace mlir::arts::convert_arts_to_llvm {

void populateRuntimePatterns(RewritePatternSet &patterns, ArtsCodegen *AC) {
  MLIRContext *context = patterns.getContext();

  /// Runtime helper patterns
  patterns.add<RuntimeQueryPattern>(context, AC);

  /// Synchronization patterns
  patterns.add<BarrierPattern, AtomicAddPattern>(context, AC);

  /// Builtin patterns (Polygeist emits these as calls, not intrinsics)
  patterns.add<BuiltinObjectSizePattern>(context);

  /// Epoch patterns (GetEdtEpochGuid stays in core; CreateEpoch/WaitOnEpoch
  /// moved to RtToLLVMPatterns)
  patterns.add<GetEdtEpochGuidPattern>(context, AC);
}

void populateDbPatterns(RewritePatternSet &patterns, ArtsCodegen *AC) {
  MLIRContext *context = patterns.getContext();
  /// DB patterns (DbGepOp/DepDbAcquireOp moved to RtToLLVMPatterns)
  patterns.add<DbRefPattern, DbAcquirePattern, DbReleasePattern>(context, AC);
}

void populateOtherPatterns(RewritePatternSet &patterns, ArtsCodegen *AC) {
  MLIRContext *context = patterns.getContext();
  patterns.add<DbAllocPattern, DbFreePattern>(context, AC);
  patterns.add<DbNumElementsPattern>(context, AC);
  patterns.add<YieldPattern, UndefPattern>(context, AC);
}

} // namespace mlir::arts::convert_arts_to_llvm
