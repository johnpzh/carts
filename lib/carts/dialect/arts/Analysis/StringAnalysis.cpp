///==========================================================================///
/// File: StringAnalysis.cpp
///==========================================================================///

#include "carts/dialect/arts/Analysis/StringAnalysis.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/utils/Utils.h"
/// Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "polygeist/Ops.h"
/// Others
#include "mlir/Transforms/RegionUtils.h"

/// Debug
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(string_analysis);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

StringAnalysis::StringAnalysis(AnalysisManager &manager)
    : ArtsAnalysis(manager), module(manager.getModule()) {}

void StringAnalysis::run() {
  invalidate();
  ARTS_DEBUG_HEADER(StringAnalysis);
  trackGlobalUsers();
  trackMemrefInitializations();
  trackCallOps();
  ARTS_DEBUG_REGION({
    if (stringMemRefs.empty()) {
      ARTS_WARN("No string memrefs found");
    } else {
      ARTS_INFO("Found " << stringMemRefs.size() << " string memrefs");
    }
  });
  built = true;
  ARTS_DEBUG_FOOTER(StringAnalysis);
}

void StringAnalysis::invalidate() {
  stringMemRefs.clear();
  globalSources.clear();
  built = false;
}

bool StringAnalysis::isStringMemRef(Value value) const {
  ensureAnalyzed();
  return stringMemRefs.contains(value);
}

void StringAnalysis::ensureAnalyzed() const {
  if (!built)
    const_cast<StringAnalysis *>(this)->run();
}

bool StringAnalysis::isStringGlobal(LLVM::GlobalOp globalOp) {
  if (auto attr = globalOp.getValueOrNull()) {
    if (auto strAttr = dyn_cast<StringAttr>(attr)) {
      /// We assume a string global is null-terminated.
      return strAttr.getValue().ends_with("\00");
    }
  }
  return false;
}

void StringAnalysis::trackGlobalUsers() {
  module.walk([&](LLVM::AddressOfOp addressOf) {
    auto globalOp =
        module.lookupSymbol<LLVM::GlobalOp>(addressOf.getGlobalName());
    /// Check if the addressOf references a global and if it's a string global.
    if (globalOp && isStringGlobal(globalOp)) {
      SmallVector<Value, 8> worklist{addressOf.getResult()};
      while (!worklist.empty()) {
        Value curr = worklist.pop_back_val();
        for (auto *curUser : curr.getUsers()) {
          /// Process pointer arithmetic (GEP) and load operations.
          if (isa<LLVM::GEPOp, LLVM::LoadOp>(curUser)) {
            for (Value result : curUser->getResults()) {
              globalSources.try_emplace(result, globalOp);
              worklist.push_back(result);
            }
          }
        }
      }
    }
  });
}

void StringAnalysis::trackMemrefInitializations() {
  module.walk([&](Operation *op) {
    /// Check for affine store operations.
    if (auto affineStore = dyn_cast<affine::AffineStoreOp>(op)) {
      Value value = affineStore.getValue();
      if (globalSources.count(value))
        stringMemRefs.insert(affineStore.getMemRef());
    }
    /// Also check for non-affine memref store operations.
    else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      Value value = storeOp.getValueToStore();
      if (globalSources.count(value))
        stringMemRefs.insert(storeOp.getMemRef());
    }
  });
}

void StringAnalysis::trackCallOps() {
  module.walk([&](func::CallOp callOp) {
    if (isStringFunction(callOp.getCallee())) {
      for (Value operand : callOp.getOperands()) {
        /// If operand is a direct memref, cache it.
        if (isa<MemRefType>(operand.getType())) {
          stringMemRefs.insert(operand);
        }
        /// Otherwise, if operand is an LLVM pointer, try to recover the
        /// underlying memref.
        else if (isa<LLVM::LLVMPointerType>(operand.getType())) {
          if (auto defOp = operand.getDefiningOp()) {
            /// Check for polygeist conversion.
            if (isa<polygeist::Pointer2MemrefOp>(defOp)) {
              Value memrefOperand = defOp->getOperand(0);
              if (isa<MemRefType>(memrefOperand.getType()))
                stringMemRefs.insert(memrefOperand);
            }
          }
        }
      }
    }
  });
}

bool StringAnalysis::isStringFunction(StringRef funcName) {
  /// Set of common string functions.
  const DenseSet<StringRef> stringFunctions = {
      "strlen", "strcpy", "strcat", "strcmp", "printf", "sprintf",
      "puts",   "fputs",  "scanf",  "fscanf", "gets"};
  return stringFunctions.contains(funcName);
}
