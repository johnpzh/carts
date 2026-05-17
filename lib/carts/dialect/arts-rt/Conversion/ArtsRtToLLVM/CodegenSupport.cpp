///==========================================================================///
/// File: Codegen.cpp
///==========================================================================///

#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/CodegenSupport.h"

/// Other dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "polygeist/Dialect.h"
#include "polygeist/Ops.h"
#include "llvm/IR/Attributes.h"
/// Other
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
/// Arts
#include "carts/Dialect.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/Utils.h"
#include "carts/dialect/arts/Utils/RuntimeConfig.h"
/// Debug

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <functional>

/// DEBUG
#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(codegen);

using namespace mlir;
using namespace mlir::func;
using namespace mlir::LLVM;
using namespace mlir::carts;
using namespace mlir::carts::arts;
using namespace mlir::carts::arts_rt;
using namespace mlir::arith;
using namespace mlir::polygeist;
using namespace mlir::cf;
using namespace mlir::scf;

/// ARTS Codegen
ArtsCodegen::ArtsCodegen(ModuleOp module, bool debug)
    : module(module), builder(OpBuilder(module->getContext())), debug(debug) {
  ARTS_DEBUG_HEADER(ArtsCodegen);
  if (failed(extractDataLayouts())) {
    mlir::emitError(module.getLoc()) << "Failed to extract data layouts";
    ARTS_UNREACHABLE("Failed to extract data layouts");
    return;
  }
  initializeTypes();
  collectGlobalLLVMStrings();
}

ArtsCodegen::ArtsCodegen(ModuleOp &module, llvm::DataLayout &llvmDL,
                         mlir::DataLayout &mlirDL, bool debug)
    : module(module), builder(OpBuilder(module->getContext())), debug(debug) {
  this->llvmDL = std::make_unique<llvm::DataLayout>(llvmDL);
  this->mlirDL = std::make_unique<mlir::DataLayout>(mlirDL);

  initializeTypes();
  collectGlobalLLVMStrings();
}

ArtsCodegen::~ArtsCodegen() { ARTS_DEBUG_FOOTER(ArtsCodegen); }

func::FuncOp ArtsCodegen::getOrCreateRuntimeFunction(RuntimeFunction FnID) {
  OpBuilder::InsertionGuard IG(getBuilder());
  getBuilder().setInsertionPointToStart(module.getBody());

  func::FuncOp funcOp;
  auto getRuntimeAbiType = [&](Type type) -> Type {
    if (isa<MemRefType>(type))
      return llvmPtr;
    return type;
  };
#define ARTS_RTL_FUNCTIONS
#define ARTS_RTL(Enum, Str, ReturnType, ...)                                   \
  case Enum: {                                                                 \
    SmallVector<Type, 4> argumentTypes{__VA_ARGS__};                           \
    for (Type &argumentType : argumentTypes)                                   \
      argumentType = getRuntimeAbiType(argumentType);                          \
    Type resultType = getRuntimeAbiType(ReturnType);                           \
    auto fnType = getBuilder().getFunctionType(                                \
        argumentTypes, isa<mlir::NoneType>(ReturnType)                         \
                           ? ArrayRef<Type>{}                                  \
                           : ArrayRef<Type>{resultType});                      \
    auto funcOrError = func::lookupFnDecl(module, Str, fnType);                \
    assert(succeeded(funcOrError) &&                                           \
           "Failed to look up ARTS runtime function declaration");             \
    funcOp = *funcOrError;                                                     \
    if (!funcOp) {                                                             \
      funcOp = func::createFnDecl(getBuilder(), module, Str, fnType,           \
                                  /*setPrivate=*/true);                        \
      funcOp.setPrivate();                                                     \
    }                                                                          \
    funcOp->setAttr(                                                           \
        "llvm.linkage",                                                        \
        LLVM::LinkageAttr::get(getContext(), LLVM::Linkage::External));        \
    /* Apply function attributes for optimization */                           \
    applyRuntimeFunctionAttributes(funcOp, Enum);                              \
    break;                                                                     \
  }
  switch (FnID) {
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Kinds.def"
  }
#undef ARTS_RTL_FUNCTIONS
#undef ARTS_RTL

  assert(funcOp && "Failed to create ARTS runtime function");
  return funcOp;
}

void ArtsCodegen::applyRuntimeFunctionAttributes(func::FuncOp funcOp,
                                                 RuntimeFunction fnID) {
  /// Apply function attributes based on ARTS runtime function properties
  auto unitAttr = getBuilder().getUnitAttr();

  /// Helper to apply MLIR attributes from a list of attribute names
  auto applyMLIRAttrs = [&](ArrayRef<StringRef> attrNames) {
    for (StringRef attrName : attrNames)
      funcOp->setAttr(attrName, unitAttr);
  };

  /// Helper to apply argument attributes from a list of attribute names
  auto applyArgMLIRAttrs = [&](unsigned argIndex,
                               ArrayRef<StringRef> attrNames) {
    for (StringRef attrName : attrNames)
      funcOp.setArgAttr(argIndex, attrName, unitAttr);
  };

  /// Helper to apply result attributes from a list of attribute names.
  auto applyResultMLIRAttrs = [&](unsigned resultIndex,
                                  ArrayRef<StringRef> attrNames) {
    for (StringRef attrName : attrNames)
      funcOp.setResultAttr(resultIndex, attrName, unitAttr);
  };

  /// Ensure memory-effect information survives Func->LLVM lowering and
  /// LLVM IR translation. Generic unit attrs like `llvm.readnone` are not
  /// always carried through by downstream translators, while the canonical
  /// LLVM memory attr is.
  auto applyReadNoneMemoryAttr = [&]() {
    auto noModRef = LLVM::ModRefInfo::NoModRef;
    auto memoryAttr =
        LLVM::MemoryEffectsAttr::get(getContext(), noModRef, noModRef, noModRef,
                                     noModRef, noModRef, noModRef);
    funcOp->setAttr("memory", memoryAttr);
  };
  auto applyReadOnlyMemoryAttr = [&]() {
    auto ref = LLVM::ModRefInfo::Ref;
    auto memoryAttr = LLVM::MemoryEffectsAttr::get(getContext(), ref, ref, ref,
                                                   ref, ref, ref);
    funcOp->setAttr("memory", memoryAttr);
  };
  auto attrSetContains = [](ArrayRef<StringRef> attrs, StringRef attrName) {
    return llvm::any_of(attrs,
                        [&](StringRef current) { return current == attrName; });
  };

  /// Generic LLVM function attrs such as nounwind/nofree/nosync/willreturn
  /// must be forwarded via the LLVM dialect `passthrough` attribute to survive
  /// translation to LLVM IR. Keep memory semantics on the canonical `memory`
  /// attribute above.
  auto appendFunctionPassthroughAttrs = [&](ArrayRef<StringRef> attrNames) {
    SmallVector<Attribute, 8> passthroughAttrs;
    if (auto existing = funcOp->getAttrOfType<ArrayAttr>("passthrough"))
      passthroughAttrs.append(existing.begin(), existing.end());

    auto hasStringAttr = [&](StringRef value) {
      return llvm::any_of(passthroughAttrs, [&](Attribute attr) {
        auto strAttr = dyn_cast<StringAttr>(attr);
        return strAttr && strAttr.getValue() == value;
      });
    };

    for (StringRef attrName : attrNames) {
      if (attrName == LLVM::LLVMDialect::getReadnoneAttrName() ||
          attrName == LLVM::LLVMDialect::getReadonlyAttrName())
        continue;

      StringRef passthroughName = attrName;
      if (!passthroughName.consume_front("llvm."))
        continue;
      if (hasStringAttr(passthroughName))
        continue;
      passthroughAttrs.push_back(getBuilder().getStringAttr(passthroughName));
    }

    if (!passthroughAttrs.empty())
      funcOp->setAttr("passthrough",
                      getBuilder().getArrayAttr(passthroughAttrs));
  };

  /// Define attribute sets
#define ARTS_ATTRS_SET(VarName, AttrList)                                      \
  SmallVector<StringRef> VarName = AttrList;
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Kinds.def"

  /// Add attributes to the function declaration
  switch (fnID) {
#define ARTS_RTL_ATTRS(Enum, FnAttrSet, RetAttrSet, ArgAttrSets)               \
  case Enum:                                                                   \
    applyMLIRAttrs(FnAttrSet);                                                 \
    appendFunctionPassthroughAttrs(FnAttrSet);                                 \
    /* Apply argument attributes */                                            \
    for (size_t ArgNo = 0; ArgNo < ArgAttrSets.size(); ++ArgNo)                \
      applyArgMLIRAttrs(ArgNo, ArgAttrSets[ArgNo]);                            \
    if (funcOp.getNumResults() > 0)                                            \
      applyResultMLIRAttrs(0, RetAttrSet);                                     \
    if (attrSetContains(FnAttrSet,                                             \
                        LLVM::LLVMDialect::getReadnoneAttrName())) {           \
      applyReadNoneMemoryAttr();                                               \
    } else if (attrSetContains(FnAttrSet,                                      \
                               LLVM::LLVMDialect::getReadonlyAttrName())) {    \
      applyReadOnlyMemoryAttr();                                               \
    }                                                                          \
    break;
#define ARTS_RTL_FUNCTIONS
#define ARTS_RTL(Enum, Name, ReturnType, ...) /// No-op, we only want attributes
#define ParamAttrs(...) SmallVector<SmallVector<StringRef>>({__VA_ARGS__})
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Kinds.def"
#undef ParamAttrs
#undef ARTS_RTL
#undef ARTS_RTL_FUNCTIONS
    /// All cases handled by .def file
  }
#undef ARTS_RTL_ATTRS
}

void ArtsCodegen::initializeTypes() {
  MLIRContext *context = module.getContext();
  llvmPtr = LLVM::LLVMPointerType::get(context);

#define ARTS_TYPE(VarName, InitValue) VarName = InitValue;
#define ARTS_FUNCTION_TYPE(VarName, ReturnType, ...)                           \
  VarName = FunctionType::get(                                                 \
      context, {__VA_ARGS__},                                                  \
      ((ReturnType == Void) ? ArrayRef<Type>{} : ArrayRef<Type>{ReturnType})); \
  VarName##Ptr = MemRefType::get(                                              \
      {ShapedType::kDynamic},                                                  \
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),            \
                                  VarName.getInputs(), false));
#define ARTS_STRUCT_TYPE(VarName, StructName, Packed, ...)                     \
  VarName = LLVM::LLVMStructType::getLiteral(context, {__VA_ARGS__}, Packed);  \
  VarName##Ptr = MemRefType::get({ShapedType::kDynamic}, VarName);
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Kinds.def"

  auto edtAbiInputs = SmallVector<Type>{Int32, llvmPtr, Int32, llvmPtr};
  EdtFn = FunctionType::get(context, edtAbiInputs, {});
  EdtFnPtr = MemRefType::get(
      {ShapedType::kDynamic},
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                  edtAbiInputs, false));
  ArtsMainFn = FunctionType::get(context, edtAbiInputs, {});
  ArtsMainFnPtr = MemRefType::get(
      {ShapedType::kDynamic},
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                  edtAbiInputs, false));

  auto mainAbiInputs = SmallVector<Type>{Int32, llvmPtr};
  MainFn = FunctionType::get(context, mainAbiInputs, ArrayRef<Type>{Int32});
  MainFnPtr = MemRefType::get(
      {ShapedType::kDynamic},
      LLVM::LLVMFunctionType::get(Int32, mainAbiInputs, false));
}

LogicalResult ArtsCodegen::extractDataLayouts() {
  /// Create LLVM DataLayout from the string attribute
  auto llvmDLAttr = module->getAttrOfType<StringAttr>("llvm.data_layout");
  if (!llvmDLAttr)
    return module.emitError("Module missing required LLVM data layout");
  llvmDL = std::make_unique<llvm::DataLayout>(llvmDLAttr.getValue().str());

  /// Create MLIR DataLayout from the module
  mlirDL = std::make_unique<mlir::DataLayout>(module);
  return success();
}

func::CallOp ArtsCodegen::createRuntimeCall(RuntimeFunction FnID,
                                            ArrayRef<Value> args,
                                            Location loc) {
  func::FuncOp func = getOrCreateRuntimeFunction(FnID);
  assert(func && "Runtime function should exist");
  SmallVector<Value, 8> callArgs;
  callArgs.reserve(args.size());
  FunctionType fnType = func.getFunctionType();
  for (auto [idx, arg] : llvm::enumerate(args)) {
    Type targetType = fnType.getInput(idx);
    if (arg.getType() == targetType) {
      callArgs.push_back(arg);
      continue;
    }

    if (targetType == llvmPtr && isa<MemRefType>(arg.getType())) {
      if (auto ptrToMemref =
              arg.getDefiningOp<polygeist::Pointer2MemrefOp>()) {
        Value source = ptrToMemref.getSource();
        if (source.getType() == llvmPtr) {
          callArgs.push_back(source);
          continue;
        }
      }
      callArgs.push_back(create<polygeist::Memref2PointerOp>(loc, llvmPtr, arg));
      continue;
    }

    callArgs.push_back(arg);
  }
  return create<func::CallOp>(loc, func, callArgs);
}

/// EDT management

/// Epoch
Value ArtsCodegen::createEpoch(Value finishEdtGuid, Value finishEdtSlot,
                               Location loc) {
  auto finishEdtSlotInt = castToInt(Int32, finishEdtSlot, loc);
  return createRuntimeCall(ARTSRTL_arts_initialize_and_start_epoch,
                           {finishEdtGuid, finishEdtSlotInt}, loc)
      .getResult(0);
}

Value ArtsCodegen::createEpochNoStart(Value finishEdtGuid, Value finishEdtSlot,
                                      Location loc) {
  auto finishEdtSlotInt = castToInt(Int32, finishEdtSlot, loc);
  auto rank = getCurrentNode(loc);
  auto epochGuid =
      createRuntimeCall(ARTSRTL_arts_initialize_epoch,
                        {rank, finishEdtGuid, finishEdtSlotInt}, loc)
          .getResult(0);
  createRuntimeCall(ARTSRTL_arts_set_current_epoch_guid, {epochGuid}, loc);
  return epochGuid;
}

/// Utils
Value ArtsCodegen::getCurrentEpochGuid(Location loc) {
  return createRuntimeCall(ARTSRTL_arts_get_current_epoch_guid, {}, loc)
      .getResult(0);
}

Value ArtsCodegen::getEdtEpochGuid(Location loc) {
  return createRuntimeCall(ARTSRTL_arts_get_edt_epoch_guid, {}, loc)
      .getResult(0);
}

Value ArtsCodegen::getCurrentEdtGuid(Location loc) {
  func::FuncOp func = getOrCreateRuntimeFunction(ARTSRTL_arts_get_current_guid);

  return create<func::CallOp>(loc, func, ArrayRef<Value>{}).getResult(0);
}

Value ArtsCodegen::getTotalWorkers(Location loc) {
  bool staticWorkers = getRuntimeStaticWorkers(module);
  if (staticWorkers && runtimeConfig && runtimeConfig->hasValidThreads()) {
    int runtimeWorkers = runtimeConfig->getRuntimeTotalWorkers();
    if (runtimeWorkers > 0)
      return createIntConstant(runtimeWorkers, Int32, loc);
  }

  func::FuncOp func =
      getOrCreateRuntimeFunction(ARTSRTL_arts_get_total_workers);
  Value workersPerNode =
      create<func::CallOp>(loc, func, ArrayRef<Value>{}).getResult(0);
  if (runtimeConfig && runtimeConfig->hasValidNodeCount() &&
      runtimeConfig->getNodeCount() <= 1)
    return workersPerNode;

  Value totalNodes = getTotalNodes(loc);
  return create<arith::MulIOp>(loc, workersPerNode,
                               castToInt(Int32, totalNodes, loc));
}

Value ArtsCodegen::getTotalNodes(Location loc) {
  if (runtimeConfig && runtimeConfig->hasValidNodeCount()) {
    int nodeCount = runtimeConfig->getNodeCount();
    if (nodeCount > 0)
      return createIntConstant(nodeCount, Int32, loc);
  }
  func::FuncOp func = getOrCreateRuntimeFunction(ARTSRTL_arts_get_total_nodes);
  return create<func::CallOp>(loc, func, ArrayRef<Value>{}).getResult(0);
}

Value ArtsCodegen::getCurrentWorker(Location loc) {
  func::FuncOp func =
      getOrCreateRuntimeFunction(ARTSRTL_arts_get_current_worker);
  return create<func::CallOp>(loc, func, ArrayRef<Value>{}).getResult(0);
}

Value ArtsCodegen::getCurrentNode(Location loc) {
  func::FuncOp func = getOrCreateRuntimeFunction(ARTSRTL_arts_get_current_node);
  return create<func::CallOp>(loc, func, ArrayRef<Value>{}).getResult(0);
}

void ArtsCodegen::waitOnHandle(Value epochGuid, Location loc) {
  createRuntimeCall(ARTSRTL_arts_wait_on_handle, {epochGuid}, loc);
}

func::FuncOp ArtsCodegen::insertInitPerWorker(Location loc,
                                              func::FuncOp callback) {
  OpBuilder::InsertionGuard IG(getBuilder());
  setInsertionPoint(module);

  func::FuncOp initPerWorkerFn =
      module.lookupSymbol<func::FuncOp>("init_per_worker");
  if (!initPerWorkerFn) {
    initPerWorkerFn =
        create<func::FuncOp>(loc, "init_per_worker", InitPerWorkerFn);
    initPerWorkerFn.setPublic();
    initPerWorkerFn.addEntryBlock();
    getBuilder().setInsertionPointToStart(&initPerWorkerFn.getBody().front());
    create<func::ReturnOp>(loc);
  }

  if (!callback)
    return initPerWorkerFn;

  Block &entryBlock = initPerWorkerFn.getBody().front();
  Operation *retTerminator = entryBlock.getTerminator();
  if (!retTerminator || !isa<func::ReturnOp>(retTerminator))
    return initPerWorkerFn;

  getBuilder().setInsertionPoint(retTerminator);
  ValueRange args = initPerWorkerFn.getArguments();
  if (callback.getNumArguments() == 4) {
    create<func::CallOp>(loc, callback, args);
  } else if (callback.getNumArguments() == 2) {
    create<func::CallOp>(loc, callback,
                         ValueRange{initPerWorkerFn.getArgument(2),
                                    initPerWorkerFn.getArgument(3)});
  } else if (callback.getNumArguments() == 0) {
    create<func::CallOp>(loc, callback, ValueRange{});
  }

  return initPerWorkerFn;
}

func::FuncOp ArtsCodegen::insertInitPerNode(Location loc,
                                            func::FuncOp callback) {
  OpBuilder::InsertionGuard IG(getBuilder());
  setInsertionPoint(module);

  func::FuncOp initPerNodeFn =
      module.lookupSymbol<func::FuncOp>("init_per_node");
  if (!initPerNodeFn) {
    /// Create the function using the InitPerNodeFn type.
    initPerNodeFn = create<func::FuncOp>(loc, "init_per_node", InitPerNodeFn);
    initPerNodeFn.setPublic();
    initPerNodeFn.addEntryBlock();
    getBuilder().setInsertionPointToStart(&initPerNodeFn.getBody().front());
    create<func::ReturnOp>(loc);
  }

  if (!callback)
    return initPerNodeFn;

  Block &entryBlock = initPerNodeFn.getBody().front();
  Operation *retTerminator = entryBlock.getTerminator();
  if (!retTerminator || !isa<func::ReturnOp>(retTerminator))
    return initPerNodeFn;

  getBuilder().setInsertionPoint(retTerminator);
  ValueRange args = initPerNodeFn.getArguments();
  if (callback.getNumArguments() == 3) {
    create<func::CallOp>(loc, callback, args);
  } else if (callback.getNumArguments() == 2) {
    create<func::CallOp>(
        loc, callback,
        ValueRange{initPerNodeFn.getArgument(1), initPerNodeFn.getArgument(2)});
  } else if (callback.getNumArguments() == 0) {
    create<func::CallOp>(loc, callback, ValueRange{});
  }

  return initPerNodeFn;
}

func::FuncOp ArtsCodegen::insertDistributedDbInitNodeFn(Location loc) {
  if (distributedInitNodeCallbacks.empty())
    return func::FuncOp();

  OpBuilder::InsertionGuard IG(getBuilder());
  setInsertionPoint(module);

  if (auto existing = module.lookupSymbol<func::FuncOp>("distributed_db_init"))
    return existing;

  auto fn = create<func::FuncOp>(loc, "distributed_db_init", InitPerNodeFn);
  fn.setPrivate();

  Block *entryBlock = fn.addEntryBlock();
  getBuilder().setInsertionPointToStart(entryBlock);

  for (func::FuncOp callback : distributedInitNodeCallbacks) {
    if (!callback)
      continue;
    if (callback.getNumArguments() == 3) {
      create<func::CallOp>(loc, callback, fn.getArguments());
      continue;
    }
    if (callback.getNumArguments() == 2) {
      create<func::CallOp>(loc, callback,
                           ValueRange{fn.getArgument(1), fn.getArgument(2)});
      continue;
    }
    if (callback.getNumArguments() == 0)
      create<func::CallOp>(loc, callback, ValueRange{});
  }

  create<func::ReturnOp>(loc);
  return fn;
}

func::FuncOp ArtsCodegen::insertDistributedDbInitWorkerFn(Location loc) {
  if (distributedInitWorkerCallbacks.empty())
    return func::FuncOp();

  OpBuilder::InsertionGuard IG(getBuilder());
  setInsertionPoint(module);

  if (auto existing =
          module.lookupSymbol<func::FuncOp>("distributed_db_init_worker"))
    return existing;

  auto fn =
      create<func::FuncOp>(loc, "distributed_db_init_worker", InitPerWorkerFn);
  fn.setPrivate();

  Block *entryBlock = fn.addEntryBlock();
  getBuilder().setInsertionPointToStart(entryBlock);

  for (func::FuncOp callback : distributedInitWorkerCallbacks) {
    if (!callback)
      continue;
    if (callback.getNumArguments() == 4) {
      create<func::CallOp>(loc, callback, fn.getArguments());
      continue;
    }
    if (callback.getNumArguments() == 2) {
      create<func::CallOp>(loc, callback,
                           ValueRange{fn.getArgument(2), fn.getArgument(3)});
      continue;
    }
    if (callback.getNumArguments() == 0)
      create<func::CallOp>(loc, callback, ValueRange{});
  }

  create<func::ReturnOp>(loc);
  return fn;
}

static void adaptMainArgvToRuntimePointer(func::FuncOp mainFn,
                                          LLVM::LLVMPointerType llvmPtr,
                                          OpBuilder &builder) {
  if (mainFn.getNumArguments() < 2 || mainFn.getBody().empty())
    return;

  Block &entry = mainFn.getBody().front();
  BlockArgument argvArg = entry.getArgument(1);
  auto argvTy = dyn_cast<MemRefType>(argvArg.getType());
  if (!argvTy || !isa<MemRefType>(argvTy.getElementType()))
    return;

  FunctionType oldType = mainFn.getFunctionType();
  SmallVector<Type, 4> inputs(oldType.getInputs().begin(),
                              oldType.getInputs().end());
  inputs[1] = llvmPtr;
  mainFn.setType(
      FunctionType::get(mainFn.getContext(), inputs, oldType.getResults()));
  argvArg.setType(llvmPtr);

  if (argvArg.use_empty())
    return;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);
  Value argvMemref = polygeist::Pointer2MemrefOp::create(builder,
                                                         mainFn.getLoc(),
                                                         argvTy, argvArg);
  argvArg.replaceAllUsesExcept(argvMemref, argvMemref.getDefiningOp());
}

func::FuncOp ArtsCodegen::insertArtsMainFn(Location loc,
                                           func::FuncOp callback) {
  OpBuilder::InsertionGuard IG(getBuilder());
  setInsertionPoint(module);

  /// ARTS v2 entry point: main_edt(paramc, paramv, depc, depv).
  /// The runtime schedules this as an EDT on rank 0 after init.
  auto newFn = create<func::FuncOp>(loc, "main_edt", ArtsMainFn);
  newFn.setPublic();

  /// Create the entry block.
  auto *entryBlock = newFn.addEntryBlock();
  getBuilder().setInsertionPointToStart(entryBlock);

  /// Call the user's mainBody.  If it expects (argc, argv), extract them
  /// from paramv[0] and paramv[1] (the runtime packs them there as uint64_t).
  auto callArgs = ValueRange{};
  if (callback.getNumArguments() > 0) {
    SmallVector<Value, 2> args;
    Value paramv = newFn.getArgument(1); // uint64_t *paramv
    if (isa<LLVM::LLVMPointerType>(paramv.getType()))
      paramv = create<polygeist::Pointer2MemrefOp>(loc, Int64Ptr, paramv);
    Value idx0 = create<arith::ConstantIndexOp>(loc, 0);
    Value idx1 = create<arith::ConstantIndexOp>(loc, 1);
    Value argcRaw = create<memref::LoadOp>(loc, paramv, ValueRange{idx0});
    Value argvRaw = create<memref::LoadOp>(loc, paramv, ValueRange{idx1});

    /// argc: truncate uint64_t to i32
    Type argcTy = callback.getArgumentTypes()[0];
    Value argc = create<arith::TruncIOp>(loc, argcTy, argcRaw);
    args.push_back(argc);

    /// argv: uint64_t to ptr to memref<?xi8*>
    if (callback.getNumArguments() > 1) {
      Type argvTy = callback.getArgumentTypes()[1];
      Value argvPtr = create<LLVM::IntToPtrOp>(loc, llvmPtr, argvRaw);
      Value argv = isa<LLVM::LLVMPointerType>(argvTy)
                       ? argvPtr
                       : create<polygeist::Pointer2MemrefOp>(loc, argvTy,
                                                             argvPtr)
                             .getResult();
      args.push_back(argv);
    }
    callArgs = args;
  }
  create<func::CallOp>(loc, callback, callArgs);

  /// Shutdown the runtime.
  createRuntimeCall(ARTSRTL_arts_shutdown, {}, loc);
  create<func::ReturnOp>(loc);
  return newFn;
}

func::FuncOp ArtsCodegen::insertMainFn(Location loc) {
  OpBuilder::InsertionGuard IG(getBuilder());
  setInsertionPoint(module);

  /// Create the function using the "MainFn" type.
  auto newFn = create<func::FuncOp>(loc, "main", MainFn);
  newFn.setPublic();

  /// Create the entry block.
  auto *entryBlock = newFn.addEntryBlock();
  getBuilder().setInsertionPointToStart(entryBlock);

  /// If config data is embedded, inject it before artsRT(). Self-contained
  /// binary.
  if (auto configData = getRuntimeConfigData(module)) {
    Value dataGlobal = getOrCreateGlobalLLVMString(loc, *configData);
    Value dataPtr = create<LLVM::BitcastOp>(loc, llvmPtr, dataGlobal);
    Value dataMemref =
        create<polygeist::Pointer2MemrefOp>(loc, Int8Ptr, dataPtr);
    createRuntimeCall(ARTSRTL_artsSetConfigData, {dataMemref}, loc);
  }
  /// Fallback to path-based runtime configuration when embedded config data
  /// is not present.
  else if (auto configPath = getRuntimeConfigPath(module)) {
    Value configGlobal = getOrCreateGlobalLLVMString(loc, *configPath);
    Value configPtr = create<LLVM::BitcastOp>(loc, llvmPtr, configGlobal);
    Value configMemref =
        create<polygeist::Pointer2MemrefOp>(loc, Int8Ptr, configPtr);
    createRuntimeCall(ARTSRTL_artsSetConfigPath, {configMemref}, loc);
  }

  /// Insert call to 'artsRT' function
  createRuntimeCall(ARTSRTL_arts_rt,
                    {newFn.getArgument(0), newFn.getArgument(1)}, loc);

  /// Return 0
  create<func::ReturnOp>(loc, createIntConstant(0, Int32, loc));
  return newFn;
}

void ArtsCodegen::initRT(Location loc) {
  auto mainFn = module.lookupSymbol<func::FuncOp>("main");
  if (!mainFn)
    return;

  /// If the module already contains a user-defined main_edt,
  /// skip the automatic RT scaffolding to avoid symbol conflicts.
  if (module.lookupSymbol<func::FuncOp>("main_edt"))
    return;

  /// Rename main function to "mainBody"
  mainFn.setName("mainBody");
  adaptMainArgvToRuntimePointer(mainFn, llvmPtr, getBuilder());

  /// Insert distributed init callbacks when distributed allocations were
  /// lowered into callback helpers.
  if (!distributedInitNodeCallbacks.empty()) {
    func::FuncOp distributedInitFn = insertDistributedDbInitNodeFn(loc);
    insertInitPerNode(loc, distributedInitFn);
  }
  if (!distributedInitWorkerCallbacks.empty()) {
    func::FuncOp distributedWorkerInitFn = insertDistributedDbInitWorkerFn(loc);
    insertInitPerWorker(loc, distributedWorkerInitFn);
  }

  /// Insert runtime entry functions (main_edt and main)
  insertArtsMainFn(loc, mainFn);
  insertMainFn(loc);
}

void ArtsCodegen::registerDistributedInitNodeCallback(func::FuncOp callback) {
  if (!callback)
    return;
  for (func::FuncOp existing : distributedInitNodeCallbacks) {
    if (existing == callback)
      return;
  }
  distributedInitNodeCallbacks.push_back(callback);
}

void ArtsCodegen::registerDistributedInitWorkerCallback(func::FuncOp callback) {
  if (!callback)
    return;
  for (func::FuncOp existing : distributedInitWorkerCallbacks) {
    if (existing == callback)
      return;
  }
  distributedInitWorkerCallbacks.push_back(callback);
}

/// Helpers

/// Create a zero/identity value matching the provided element type.
Value ArtsCodegen::createZeroValue(Type elemType, Location loc) {
  if (!elemType)
    return Value();

  if (isa<IndexType>(elemType))
    return createIndexConstant(0, loc);

  if (auto intTy = dyn_cast<IntegerType>(elemType))
    return create<arith::ConstantIntOp>(loc, 0, intTy.getWidth());

  if (auto floatTy = dyn_cast<FloatType>(elemType)) {
    llvm::APFloat zero = llvm::APFloat::getZero(floatTy.getFloatSemantics());
    return create<arith::ConstantFloatOp>(loc, floatTy, zero);
  }

  return Value();
}

Value ArtsCodegen::createFnPtr(func::FuncOp funcOp, Location loc) {
  auto FT = funcOp.getFunctionType();
  auto LFT = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(funcOp.getContext()), FT.getInputs(), false);
  auto getFuncOp = create<polygeist::GetFuncOp>(
      loc, LLVM::LLVMPointerType::get(funcOp.getContext()), funcOp.getName());
  return create<polygeist::Pointer2MemrefOp>(
             loc, MemRefType::get({ShapedType::kDynamic}, LFT), getFuncOp)
      .getResult();
}

Value ArtsCodegen::createIndexConstant(int64_t value, Location loc) {
  return create<arith::ConstantIndexOp>(loc, value);
}

Value ArtsCodegen::createIntConstant(int64_t value, Type type, Location loc) {
  assert(isa<IntegerType>(type) && "Expected integer type");
  return create<arith::ConstantOp>(loc, type,
                                   getBuilder().getIntegerAttr(type, value));
}

static bool isUndefValue(Value value) {
  Operation *def = value.getDefiningOp();
  return def && (isa<LLVM::UndefOp>(def) || isa<polygeist::UndefOp>(def) ||
                 isa<mlir::carts::arts::UndefOp>(def));
}

static Value createTypedUndef(ArtsCodegen &AC, Type type, Location loc) {
  if (LLVM::isCompatibleType(type))
    return AC.create<LLVM::UndefOp>(loc, type);
  return AC.create<polygeist::UndefOp>(loc, type);
}

/// Casting
Value ArtsCodegen::castParameter(mlir::Type targetType, Value source,
                                 Location loc, ParameterCastMode mode) {
  assert(targetType.isIntOrIndexOrFloat() && "Target type should be a number");
  if (source.getType() == targetType)
    return source;

  if (isUndefValue(source))
    return createTypedUndef(*this, targetType, loc);

  if (mode == ParameterCastMode::Bitwise) {
    auto srcType = source.getType();
    if (auto dstIntType = dyn_cast<IntegerType>(targetType)) {
      if (auto srcFloatType = dyn_cast<FloatType>(srcType)) {
        unsigned srcBits = srcFloatType.getWidth();
        unsigned dstBits = dstIntType.getWidth();
        if (srcBits == dstBits)
          return create<arith::BitcastOp>(loc, dstIntType, source);
        auto srcIntType = IntegerType::get(getContext(), srcBits);
        Value intBits = create<arith::BitcastOp>(loc, srcIntType, source);
        if (srcBits < dstBits)
          return create<arith::ExtUIOp>(loc, dstIntType, intBits);
        return create<arith::TruncIOp>(loc, dstIntType, intBits);
      }
    } else if (auto dstFloatType = dyn_cast<FloatType>(targetType)) {
      if (auto srcIntType = dyn_cast<IntegerType>(srcType)) {
        unsigned srcBits = srcIntType.getWidth();
        unsigned dstBits = dstFloatType.getWidth();
        if (srcBits == dstBits)
          return create<arith::BitcastOp>(loc, dstFloatType, source);
        auto dstIntType = IntegerType::get(getContext(), dstBits);
        Value intBits = source;
        if (srcBits < dstBits)
          intBits = create<arith::ExtUIOp>(loc, dstIntType, source);
        else
          intBits = create<arith::TruncIOp>(loc, dstIntType, source);
        return create<arith::BitcastOp>(loc, dstFloatType, intBits);
      }
    }
  }
  /// If target type is an integer
  if (isa<IntegerType>(targetType))
    return castToInt(targetType, source, loc);
  /// If target type is an index
  if (isa<IndexType>(targetType))
    return castToIndex(source, loc);
  /// If target type is a float
  if (isa<FloatType>(targetType))
    return castToFloat(targetType, source, loc);
  return source;
}

Value ArtsCodegen::castToIndex(Value source, Location loc) {
  if (source.getType().isIndex())
    return source;
  /// If it is not an index, cast it to an index.
  auto indexType = IndexType::get(source.getContext());
  return create<arith::IndexCastOp>(loc, indexType, source).getResult();
}

Value ArtsCodegen::castToFloat(mlir::Type targetType, Value source,
                               Location loc) {
  assert(isa<FloatType>(targetType) && "Target type should be a float");
  if (isa<FloatType>(source.getType()))
    return source;
  /// If it is not a float, cast it to a float.
  return create<arith::SIToFPOp>(loc, targetType, source).getResult();
}

Value ArtsCodegen::castToInt(Type targetType, Value source, Location loc) {
  assert(isa<IntegerType>(targetType) &&
         "Target type should be an integer (e.g. i64, i32, etc.)");

  /// If types match exactly, no cast is needed.
  if (source.getType() == targetType)
    return source;

  /// If source is index => use IndexCastOp to target integer type
  if (source.getType().isIndex())
    return create<arith::IndexCastOp>(loc, targetType, source);

  /// If source is float => use FPToSIOp
  if (isa<FloatType>(source.getType()))
    return create<arith::FPToSIOp>(loc, targetType, source);

  /// If source is integer => handle extension or truncation
  if (auto srcIntType = dyn_cast<IntegerType>(source.getType())) {
    auto dstIntType = dyn_cast<IntegerType>(targetType);
    if (!dstIntType)
      return source;
    unsigned srcWidth = srcIntType.getWidth();
    unsigned dstWidth = dstIntType.getWidth();

    if (srcWidth == dstWidth)
      return source;
    else if (srcWidth < dstWidth)
      return create<arith::ExtSIOp>(loc, targetType, source);
    return create<arith::TruncIOp>(loc, targetType, source);
  }

  /// If none of the above matched => unsupported type
  ARTS_ERROR("Unsupported type for casting to integer: " << source);
  assert(false && "Unsupported type in castToInt");
  return nullptr;
}

Value ArtsCodegen::ensureI64(Value source, Location loc) {
  return castToInt(Int64, source, loc);
}

Value ArtsCodegen::castToLLVMPtr(Value source, Location loc) {
  auto ptrType = getLLVMPointerType(source);
  if (!ptrType)
    return source;
  return create<polygeist::Memref2PointerOp>(loc, ptrType, source);
}

void ArtsCodegen::collectGlobalLLVMStrings() {
  for (auto &op : module.getOps()) {
    if (auto global = dyn_cast<LLVM::GlobalOp>(op))
      llvmStringGlobals[global.getName().str()] = global;
  }
}

Value ArtsCodegen::getOrCreateGlobalLLVMString(Location loc, StringRef value) {
  /// Check if we already have this string cached. If so, return the cached
  /// global
  if (llvmStringGlobals.find(value.str()) == llvmStringGlobals.end()) {
    OpBuilder::InsertionGuard IG(getBuilder());
    getBuilder().setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(Int8, value.size() + 1);

    /// Create an unique name for the global
    std::string globalName = "str_" + std::to_string(llvmStringGlobals.size());
    llvmStringGlobals[value.str()] = create<LLVM::GlobalOp>(
        loc, type, true, LLVM::Linkage::Internal, globalName,
        getBuilder().getStringAttr(value.str() + '\0'));
  }

  LLVM::GlobalOp global = llvmStringGlobals[value.str()];
  return create<LLVM::AddressOfOp>(loc, global);
}

void ArtsCodegen::createPrintfCall(Location loc, llvm::StringRef format,
                                   ValueRange args) {
  if (!debug)
    return;

  /// Create the printf function declaration if it doesn't exist
  auto printfFn = module.lookupSymbol<LLVM::LLVMFuncOp>("printf");
  if (!printfFn) {
    auto printfType = LLVM::LLVMFunctionType::get(Int32, {llvmPtr}, true);
    printfFn = create<LLVM::LLVMFuncOp>(loc, "printf", printfType);
    printfFn.setLinkage(LLVM::Linkage::External);
  }
  assert(printfFn && "printf function not found");

  /// Get or create the format string global and cast it to a generic pointer
  auto formatStrPtr = getOrCreateGlobalLLVMString(loc, format);
  auto castedFormatPtr = create<LLVM::BitcastOp>(loc, llvmPtr, formatStrPtr);

  /// Create the printf call with all arguments
  SmallVector<Value> callArgs;
  callArgs.push_back(castedFormatPtr);
  callArgs.append(args.begin(), args.end());

  /// Create the call operation with a symbol reference
  create<LLVM::CallOp>(loc, TypeRange{getBuilder().getI32Type()},
                       SymbolRefAttr::get(getContext(), printfFn.getName()),
                       callArgs);
}

/// Insertion point management
void ArtsCodegen::setInsertionPoint(Operation *op) {
  getBuilder().setInsertionPoint(op);
}

void ArtsCodegen::setInsertionPointToStart(Block *block) {
  getBuilder().setInsertionPointToStart(block);
}

void ArtsCodegen::setInsertionPointToEnd(Block *block) {
  getBuilder().setInsertionPointToEnd(block);
}

void ArtsCodegen::setInsertionPointAfter(Operation *op) {
  getBuilder().setInsertionPointAfter(op);
}

void ArtsCodegen::setInsertionPoint(ModuleOp &module) {
  getBuilder().setInsertionPointToStart(module.getBody());
}

Operation *ArtsCodegen::clone(Operation &op) { return getBuilder().clone(op); }

Operation *ArtsCodegen::clone(Operation &op, IRMapping &mapper) {
  return getBuilder().clone(op, mapper);
}
///===----------------------------------------------------------------------===///
/// Memref helpers
///===----------------------------------------------------------------------===///

LLVM::LLVMPointerType ArtsCodegen::getLLVMPointerType(Value source) {
  if (isa<LLVM::LLVMPointerType>(source.getType()))
    return nullptr;
  auto MT = dyn_cast<MemRefType>(source.getType());
  if (!MT)
    return nullptr;

  return LLVM::LLVMPointerType::get(getContext(), MT.getMemorySpaceAsInt());
}

Value ArtsCodegen::computeElementTypeSize(Type elementType, Location loc) {
  return create<polygeist::TypeSizeOp>(loc, IndexType::get(getContext()),
                                       elementType);
}

Value ArtsCodegen::computeTotalElements(ValueRange sizes, Location loc) {
  if (sizes.empty())
    return createIndexConstant(1, loc);
  if (sizes.size() == 1)
    return castToIndex(sizes.front(), loc);

  Value product = castToIndex(sizes.front(), loc);
  for (Value size : sizes.drop_front())
    product = create<arith::MulIOp>(loc, product, castToIndex(size, loc));
  return product;
}

Value ArtsCodegen::computeLinearIndex(ArrayRef<Value> sizes,
                                      ArrayRef<Value> indices, Location loc) {
  if (indices.empty())
    return createIndexConstant(0, loc);
  if (indices.size() == 1)
    return castToIndex(indices.front(), loc);

  return computeLinearIndexFromStrides(computeStridesFromSizes(sizes, loc),
                                       indices, loc);
}

SmallVector<Value>
ArtsCodegen::computeIndicesFromLinearIndex(ArrayRef<Value> sizes,
                                           Value linearIndex, Location loc) {
  SmallVector<Value> coords;
  if (sizes.empty())
    return coords;

  Value remaining = castToIndex(linearIndex, loc);
  SmallVector<Value> strides = computeStridesFromSizes(sizes, loc);
  coords.reserve(sizes.size());

  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i + 1 == sizes.size()) {
      coords.push_back(remaining);
      break;
    }

    Value stride = castToIndex(strides[i], loc);
    Value coord = create<arith::DivUIOp>(loc, remaining, stride);
    coords.push_back(coord);
    remaining = create<arith::RemUIOp>(loc, remaining, stride);
  }

  return coords;
}

Value ArtsCodegen::computeLinearIndexFromStrides(ValueRange strides,
                                                 ValueRange indices,
                                                 Location loc) {
  if (indices.empty())
    return createIndexConstant(0, loc);

  Value linearIndex = createIndexConstant(0, loc);

  /// Use strides directly: linear_index = sum(indices[i] * strides[i])
  for (size_t i = 0; i < indices.size() && i < strides.size(); ++i) {
    Value index = castToIndex(indices[i], loc);
    Value stride = castToIndex(strides[i], loc);
    auto contribution = create<arith::MulIOp>(loc, index, stride);
    linearIndex = create<arith::AddIOp>(loc, linearIndex, contribution);
  }

  return linearIndex;
}

SmallVector<Value> ArtsCodegen::computeStridesFromSizes(ArrayRef<Value> sizes,
                                                        Location loc) {
  if (sizes.empty())
    return {};

  SmallVector<OpFoldResult, 4> mixedSizes;
  mixedSizes.reserve(sizes.size());
  for (Value size : sizes)
    mixedSizes.push_back(castToIndex(size, loc));
  return getValueOrCreateConstantIndexOp(
      getBuilder(), loc,
      memref::computeStridesIRBlock(loc, getBuilder(), mixedSizes));
}

/// Helper to iterate over all Db elements
void ArtsCodegen::iterateDbElements(Value dbGuid, Value edtGuid,
                                    ArrayRef<Value> dbSizes,
                                    ArrayRef<Value> dbOffsets, bool isSingle,
                                    Location loc,
                                    std::function<void(Value)> elementCallback,
                                    ArrayRef<Value> linearSizes) {

  /// If the db sizes are empty, we are dealing with a single element
  if (dbSizes.empty()) {
    auto zeroIndex = createIndexConstant(0, loc);
    elementCallback(zeroIndex);
    return;
  }

  /// If the db is single, we can use a single loop
  if (isSingle) {
    /// The loop goes from the (offset) to (offset + size)
    auto lowerBound = dbOffsets[0];
    auto upperBound = create<arith::AddIOp>(loc, lowerBound, dbSizes[0]);
    auto stepConstant = createIndexConstant(1, loc);
    auto loopOp = create<scf::ForOp>(loc, lowerBound, upperBound, stepConstant);
    auto &loopBlock = loopOp.getRegion().front();
    setInsertionPointToStart(&loopBlock);
    elementCallback(loopOp.getInductionVar());
    setInsertionPointAfter(loopOp);
    return;
  }

  /// Otherwise, we need to iterate over all dimensions of the db
  iterateMultiDb(dbGuid, edtGuid, dbSizes, dbOffsets, loc, elementCallback,
                 linearSizes);
}

/// Helper to iterate over multi-dimensional db elements
void ArtsCodegen::iterateMultiDb(Value dbGuid, Value edtGuid,
                                 ArrayRef<Value> dbSizes,
                                 ArrayRef<Value> dbOffsets, Location loc,
                                 std::function<void(Value)> elementCallback,
                                 ArrayRef<Value> linearSizes) {
  const unsigned dbRank = dbSizes.size();
  auto stepConstant = createIndexConstant(1, loc);
  ArrayRef<Value> sizesForLinear = linearSizes.empty() ? dbSizes : linearSizes;

  std::function<void(unsigned, SmallVector<Value, 4> &)>
      addDependenciesRecursive = [&](unsigned dim,
                                     SmallVector<Value, 4> &indices) {
        if (dim == dbRank) {
          auto linearIndex = computeLinearIndex(sizesForLinear, indices, loc);
          elementCallback(linearIndex);
          return;
        }
        /// The loop goes from (offset) for this dimension
        auto lowerBound = dbOffsets[dim];
        auto upperBound = create<arith::AddIOp>(loc, lowerBound, dbSizes[dim]);
        auto loopOp =
            create<scf::ForOp>(loc, lowerBound, upperBound, stepConstant);

        auto &loopBlock = loopOp.getRegion().front();
        setInsertionPointToStart(&loopBlock);
        indices.push_back(loopOp.getInductionVar());
        addDependenciesRecursive(dim + 1, indices);
        indices.pop_back();
        setInsertionPointAfter(loopOp);
      };

  SmallVector<Value, 4> initIndices;
  addDependenciesRecursive(0, initIndices);
}

///===----------------------------------------------------------------------===///
/// Debug printing helpers
///===----------------------------------------------------------------------===///

void ArtsCodegen::printDebugInfo(Location loc, const Twine &message,
                                 ValueRange args) {
  if (!debug)
    return;

  SmallString<128> formattedMessage;
  message.toVector(formattedMessage);
  formattedMessage.append("\n");
  createPrintfCall(loc, formattedMessage, args);
}

void ArtsCodegen::printDebugValue(Location loc, const Twine &label,
                                  Value value) {
  if (!debug)
    return;

  SmallString<128> formattedLabel;
  label.toVector(formattedLabel);
  formattedLabel.append(": %lu\n");

  auto castedValue = castToInt(Int64, value, loc);
  createPrintfCall(loc, formattedLabel, {castedValue});
}
