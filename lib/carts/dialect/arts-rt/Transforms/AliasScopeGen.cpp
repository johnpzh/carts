///==========================================================================///
/// File: AliasScopeGen.cpp
///
/// This pass generates LLVM alias scope metadata for ARTS data arrays to enable
/// LLVM optimizations like vectorization. Without this metadata, LLVM cannot
/// prove that distinct data arrays don't overlap.
///
/// Before (LLVM cannot prove arrays don't alias - no vectorization):
///   %ptr_A = llvm.load %deps[0].ptr : !llvm.ptr
///   %ptr_B = llvm.load %deps[1].ptr : !llvm.ptr
///   scf.for %i = ... {
///     %a = llvm.load %ptr_A[%i] : f64
///     %b = llvm.load %ptr_B[%i] : f64
///     /// May alias with %ptr_B - blocks opt!
///     llvm.store %result, %ptr_A[%i]
///   }
///
/// After (distinct alias scopes enable vectorization):
///   llvm.experimental.noalias.scope.decl %scope_A
///   llvm.experimental.noalias.scope.decl %scope_B
///   %ptr_A = llvm.load %deps[0].ptr : !llvm.ptr
///   %ptr_B = llvm.load %deps[1].ptr : !llvm.ptr
///   scf.for %i = ... {
///     %a = llvm.load %ptr_A[%i] {alias_scopes=[@scope_A]}
///     %b = llvm.load %ptr_B[%i] {alias_scopes=[@scope_B]}
///     llvm.store %result, %ptr_A[%i] {alias_scopes=[@scope_A]}
///   }
///
/// The pass identifies data pointer loads from the deps struct and attaches
/// alias scope metadata to all memory accesses using those pointers. Each
/// data array gets a unique scope, and accesses are marked as not aliasing
/// with other arrays.
///
/// This pass must emit llvm.experimental.noalias.scope.decl intrinsics
/// at function entry for LLVM to recognize the alias scopes.
///==========================================================================///

#include "carts/dialect/arts-rt/Transforms/Passes.h"

namespace mlir::carts::arts {
#define GEN_PASS_DEF_ALIASSCOPEGEN
#include "carts/dialect/arts-rt/Transforms/Passes.h.inc"
} // namespace mlir::carts::arts

#include "carts/utils/Debug.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
ARTS_DEBUG_SETUP(alias_scope_gen);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

/// Maximum number of arrays for alias scope generation
constexpr unsigned MaxArraysForAliasScopes = 10;

/// Information about a data pointer loaded from the deps struct
struct DataPointerInfo {
  SmallVector<LLVM::LoadOp, 2>
      loadOps; /// Load ops that materialize this dep-pointer slot
  SmallVector<Value, 2>
      ptrValues;    /// Loaded pointer SSA values (equivalent dep-pointer slot)
  int64_t depIndex; /// Slot id (constant dep-index or dynamic dep-index)
  bool isDynamicDepIndex; /// True when this slot comes from a dynamic dep-index
  LLVM::AliasScopeAttr scope; /// Assigned alias scope
};

struct DataPointerCollectionResult {
  SmallVector<DataPointerInfo> dataPointers;
  int skippedUnknownDepLoads = 0;
};

/// Returns true if this load reads the dependency data-pointer field from an
/// arts_edt_dep_t entry GEP.
static bool isDepStructPointerLoad(LLVM::LoadOp loadOp) {
  if (!isa<LLVM::LLVMPointerType>(loadOp.getResult().getType()))
    return false;

  auto gep = loadOp.getAddr().getDefiningOp<LLVM::GEPOp>();
  if (!gep)
    return false;

  auto structType = dyn_cast<LLVM::LLVMStructType>(gep.getElemType());
  if (!structType)
    return false;

  auto rawIndices = gep.getRawConstantIndices();
  if (rawIndices.size() < 2)
    return false;

  int32_t penultimate = rawIndices[rawIndices.size() - 2];
  int32_t last = rawIndices.back();

  /// Current v2 lowering stores the dependency pointer field in slot 1. Slot 2
  /// can still appear in archived IR snapshots consumed by debug tools.
  return penultimate == 0 && (last == 1 || last == 2);
}

struct DepIndexInfo {
  bool valid = false;
  std::optional<int64_t> constantIndex;
  Value dynamicIndex;
};

/// Try to recover dep index info for a load of depv[idx].ptr.
static DepIndexInfo extractDepIndexInfo(LLVM::LoadOp loadOp) {
  DepIndexInfo info;

  /// Prefer LLVM GEP pattern after Arts->LLVM lowering:
  ///   %depEntry = llvm.getelementptr %depv[%idx]
  ///   %ptrField = llvm.getelementptr %depEntry[0, 2]
  ///   %ptr = llvm.load %ptrField : !llvm.ptr
  if (auto ptrFieldGep = loadOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
    if (isDepStructPointerLoad(loadOp)) {
      if (auto depEntryGep =
              ptrFieldGep.getBase().getDefiningOp<LLVM::GEPOp>()) {
        auto depIndices = depEntryGep.getRawConstantIndices();
        if (depIndices.empty())
          return info;
        SmallVector<Value, 4> dynIndices(
            depEntryGep.getDynamicIndices().begin(),
            depEntryGep.getDynamicIndices().end());
        int32_t depIdxRaw = depIndices.back();
        if (depIdxRaw == LLVM::GEPOp::kDynamicIndex) {
          unsigned dynPos = 0;
          for (size_t i = 0; i < depIndices.size(); ++i) {
            int32_t raw = depIndices[i];
            if (raw != LLVM::GEPOp::kDynamicIndex)
              continue;
            if (dynPos >= dynIndices.size())
              return info;
            if (i == depIndices.size() - 1) {
              info.valid = true;
              info.dynamicIndex = dynIndices[dynPos];
              return info;
            }
            dynPos++;
          }
          return info;
        }
        info.valid = true;
        info.constantIndex = static_cast<int64_t>(depIdxRaw);
        return info;
      }
    }
  }

  /// Fallback for pre-conversion form.
  if (auto depGep = loadOp.getAddr().getDefiningOp<DepGepOp>()) {
    info.valid = true;
    if (auto cst = ValueAnalysis::getConstantValue(depGep.getOffset()))
      info.constantIndex = *cst;
    else
      info.dynamicIndex = depGep.getOffset();
    return info;
  }

  return info;
}

static void
addAlignmentToDataPointerLoads(SmallVectorImpl<DataPointerInfo> &dataPointers) {
  for (auto &info : dataPointers) {
    for (auto loadOp : info.loadOps) {
      if (!loadOp.getAlignment()) {
        loadOp.setAlignment(8);
        ARTS_DEBUG_TYPE("Set alignment=8 on data pointer load for array_"
                        << info.depIndex);
      }
    }
  }
}

/// Collect data pointer loads from EDT function
static DataPointerCollectionResult
collectDataPointerLoads(LLVM::LLVMFuncOp funcOp) {
  DataPointerCollectionResult result;
  llvm::DenseMap<int64_t, unsigned> depIndexToSlot;
  llvm::DenseMap<Value, unsigned> dynamicDepIndexToSlot;
  unsigned nextDynamicOrdinal = 0;

  funcOp.walk([&](LLVM::LoadOp loadOp) {
    if (!isa<LLVM::LLVMPointerType>(loadOp.getResult().getType()))
      return;

    DepIndexInfo depIndexInfo = extractDepIndexInfo(loadOp);
    if (!depIndexInfo.valid) {
      if (isDepStructPointerLoad(loadOp))
        result.skippedUnknownDepLoads++;
      return;
    }

    unsigned slot = 0;
    if (depIndexInfo.constantIndex) {
      auto it = depIndexToSlot.find(*depIndexInfo.constantIndex);
      if (it == depIndexToSlot.end()) {
        slot = result.dataPointers.size();
        depIndexToSlot[*depIndexInfo.constantIndex] = slot;
        DataPointerInfo info;
        info.depIndex = *depIndexInfo.constantIndex;
        info.isDynamicDepIndex = false;
        info.loadOps.push_back(loadOp);
        info.ptrValues.push_back(loadOp.getResult());
        result.dataPointers.push_back(std::move(info));
        return;
      }
      slot = it->second;
    } else {
      if (!depIndexInfo.dynamicIndex)
        return;
      auto it = dynamicDepIndexToSlot.find(depIndexInfo.dynamicIndex);
      if (it == dynamicDepIndexToSlot.end()) {
        slot = result.dataPointers.size();
        dynamicDepIndexToSlot[depIndexInfo.dynamicIndex] = slot;
        DataPointerInfo info;
        info.depIndex = static_cast<int64_t>(nextDynamicOrdinal++);
        info.isDynamicDepIndex = true;
        info.loadOps.push_back(loadOp);
        info.ptrValues.push_back(loadOp.getResult());
        result.dataPointers.push_back(std::move(info));
        return;
      }
      slot = it->second;
    }

    DataPointerInfo &info = result.dataPointers[slot];
    info.loadOps.push_back(loadOp);
    if (!llvm::is_contained(info.ptrValues, loadOp.getResult()))
      info.ptrValues.push_back(loadOp.getResult());
  });

  return result;
}

/// Create alias scope domain and scopes for data pointers
static void createAliasScopes(MLIRContext *ctx,
                              SmallVectorImpl<DataPointerInfo> &dataPointers) {
  if (dataPointers.empty())
    return;

  auto domain = LLVM::AliasScopeDomainAttr::get(
      ctx, DistinctAttr::create(UnitAttr::get(ctx)),
      StringAttr::get(ctx, "ARTS_data_arrays"));

  for (auto &info : dataPointers) {
    std::string scopeDesc = (info.isDynamicDepIndex ? "array_dyn_" : "array_") +
                            std::to_string(info.depIndex);
    info.scope =
        LLVM::AliasScopeAttr::get(ctx, DistinctAttr::create(UnitAttr::get(ctx)),
                                  domain, StringAttr::get(ctx, scopeDesc));
  }
}

/// Emit llvm.experimental.noalias.scope.decl intrinsics at function entry.
static void
emitScopeDeclarations(LLVM::LLVMFuncOp funcOp,
                      SmallVectorImpl<DataPointerInfo> &dataPointers) {
  if (dataPointers.empty() || funcOp.getBody().empty())
    return;

  OpBuilder builder(funcOp.getBody().front().front().getContext());
  builder.setInsertionPointToStart(&funcOp.getBody().front());

  for (auto &info : dataPointers) {
    LLVM::NoAliasScopeDeclOp::create(builder, funcOp.getLoc(), info.scope);
    ARTS_DEBUG_TYPE("Emitted noalias scope decl for array_" << info.depIndex);
  }
}

/// Find which data pointer (if any) a memory access address is derived from
static SmallVector<DataPointerInfo *, 2>
findCandidateDataPointers(Value addr,
                          SmallVectorImpl<DataPointerInfo> &dataPointers) {
  SmallVector<DataPointerInfo *, 2> result;
  for (auto &info : dataPointers) {
    bool derived = false;
    for (Value ptr : info.ptrValues) {
      if (ValueAnalysis::isDerivedFromPtr(addr, ptr)) {
        derived = true;
        break;
      }
    }
    if (derived)
      result.push_back(&info);
  }
  return result;
}

static void attachAliasScopes(Operation *op, DataPointerInfo &source,
                              SmallVectorImpl<DataPointerInfo> &allPointers,
                              bool emitNoAliasScopes) {
  MLIRContext *ctx = op->getContext();

  SmallVector<Attribute> aliasScopes = {source.scope};
  auto aliasScopesAttr = ArrayAttr::get(ctx, aliasScopes);

  ArrayAttr noaliasScopesAttr = nullptr;
  if (emitNoAliasScopes) {
    SmallVector<Attribute> noaliasScopes;
    for (auto &info : allPointers) {
      if (&info != &source)
        noaliasScopes.push_back(info.scope);
    }
    noaliasScopesAttr =
        noaliasScopes.empty() ? nullptr : ArrayAttr::get(ctx, noaliasScopes);
  }

  if (auto loadOp = dyn_cast<LLVM::LoadOp>(op)) {
    loadOp.setAliasScopesAttr(aliasScopesAttr);
    if (noaliasScopesAttr)
      loadOp.setNoaliasScopesAttr(noaliasScopesAttr);
  } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
    storeOp.setAliasScopesAttr(aliasScopesAttr);
    if (noaliasScopesAttr)
      storeOp.setNoaliasScopesAttr(noaliasScopesAttr);
  } else {
    op->setAttr(LLVM::LLVMDialect::getAliasScopesAttrName(), aliasScopesAttr);
    if (noaliasScopesAttr)
      op->setAttr(LLVM::LLVMDialect::getNoAliasScopesAttrName(),
                  noaliasScopesAttr);
  }
}

/// Process memory accesses in the function.
static int processMemoryAccesses(LLVM::LLVMFuncOp funcOp,
                                 SmallVectorImpl<DataPointerInfo> &dataPointers,
                                 bool emitNoAliasScopes, int &ambiguousCount) {
  int count = 0;

  /// Build set of data pointer values to skip when processing.
  DenseSet<Value> dataPointerValues;
  for (auto &info : dataPointers) {
    for (Value ptr : info.ptrValues)
      dataPointerValues.insert(ptr);
  }

  funcOp.walk([&](Operation *op) {
    Value addr;

    if (auto loadOp = dyn_cast<LLVM::LoadOp>(op)) {
      if (dataPointerValues.contains(loadOp.getResult()))
        return;
      addr = loadOp.getAddr();
    } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
      addr = storeOp.getAddr();
    } else if (auto dynLoad = dyn_cast<polygeist::DynLoadOp>(op)) {
      addr = dynLoad.getMemref();
    } else if (auto dynStore = dyn_cast<polygeist::DynStoreOp>(op)) {
      addr = dynStore.getMemref();
    } else {
      return;
    }

    if (!addr)
      return;

    auto matches = findCandidateDataPointers(addr, dataPointers);
    if (matches.empty())
      return;
    if (matches.size() > 1) {
      ambiguousCount++;
      ARTS_DEBUG_TYPE("Skipping ambiguous alias-scope assignment in "
                      << funcOp.getName() << " for op: " << *op);
      return;
    }

    if (auto *source = matches.front()) {
      attachAliasScopes(op, *source, dataPointers, emitNoAliasScopes);
      count++;
    }
  });

  return count;
}

struct AliasScopeGenPass
    : public arts::impl::AliasScopeGenBase<AliasScopeGenPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    ARTS_INFO_HEADER(AliasScopeGenPass);

    int totalCount = 0;
    int totalDecls = 0;
    int totalSkippedDynamic = 0;
    int totalAmbiguous = 0;

    module.walk([&](LLVM::LLVMFuncOp funcOp) {
      if (!funcOp.getName().starts_with("__arts_edt_"))
        return;

      ARTS_DEBUG_TYPE("Processing EDT function: " << funcOp.getName());

      auto collected = collectDataPointerLoads(funcOp);
      auto &dataPointers = collected.dataPointers;
      if (dataPointers.empty()) {
        ARTS_INFO("No dependency pointer loads found in " << funcOp.getName());
        return;
      }

      if (dataPointers.size() > MaxArraysForAliasScopes) {
        ARTS_INFO("Skipping alias scopes for "
                  << funcOp.getName() << ": " << dataPointers.size()
                  << " arrays exceeds limit (" << MaxArraysForAliasScopes
                  << ")");
        return;
      }

      int dynamicSlots = 0;
      for (auto &info : dataPointers)
        if (info.isDynamicDepIndex)
          dynamicSlots++;
      totalSkippedDynamic += collected.skippedUnknownDepLoads;
      ARTS_INFO("Found " << dataPointers.size()
                         << " unique dependency pointer slots in "
                         << funcOp.getName() << " (" << dynamicSlots
                         << " dynamic), skipped "
                         << collected.skippedUnknownDepLoads
                         << " unrecognized dep-pointer loads");

      addAlignmentToDataPointerLoads(dataPointers);

      createAliasScopes(ctx, dataPointers);
      emitScopeDeclarations(funcOp, dataPointers);
      totalDecls += dataPointers.size();

      bool emitNoAliasScopes = (dataPointers.size() > 1);
      int ambiguousCount = 0;
      int count = processMemoryAccesses(funcOp, dataPointers, emitNoAliasScopes,
                                        ambiguousCount);
      totalCount += count;
      totalAmbiguous += ambiguousCount;

      ARTS_INFO("Attached alias scopes to "
                << count << " memory accesses (noalias "
                << (emitNoAliasScopes ? "enabled" : "disabled")
                << ", ambiguous skipped: " << ambiguousCount << ")");
    });

    ARTS_INFO("Total: emitted "
              << totalDecls << " scope declarations, "
              << "attached alias scopes to " << totalCount
              << " memory accesses, skipped " << totalSkippedDynamic
              << " unrecognized dep-pointer loads, skipped " << totalAmbiguous
              << " ambiguous memory accesses");
    ARTS_INFO_FOOTER(AliasScopeGenPass);
  }
};

} // end anonymous namespace

///===----------------------------------------------------------------------===///
/// Pass creation and registration
///===----------------------------------------------------------------------===///
namespace mlir {
namespace carts::arts {

std::unique_ptr<Pass> createAliasScopeGenPass() {
  return std::make_unique<AliasScopeGenPass>();
}

} // namespace carts::arts
} // namespace mlir
