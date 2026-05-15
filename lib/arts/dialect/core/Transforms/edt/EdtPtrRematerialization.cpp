///==========================================================================///=
/// EdtPtrRematerialization
///
/// Before:
///   %sub = polygeist.subindex %A[%i] : memref<?x?xf64> to memref<?xf64>
///   arts.edt <task>(%sub) { ... }
///
/// After:
///   arts.edt <task>(%A, %i) {
///     %sub = polygeist.subindex %A[%i] : memref<?x?xf64> to memref<?xf64>
///     ...
///   }
///==========================================================================///=

/// Dialects
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Arts
#include "arts/Dialect.h"
#include "arts/dialect/core/Transforms/edt/EdtPtrRematerialization.h"
#include "arts/utils/Utils.h"

/// Others
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVector.h"

/// LLVM support
#include "arts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_ptr_rematerialization);

namespace mlir {
namespace arts {

/// Check if an operation produces pointer or memref types that should be
/// rematerialized. Returns true if ANY result is a pointer-like type.
static bool producesPointerLikeType(Operation *op) {
  for (Value result : op->getResults()) {
    Type resultType = result.getType();
    /// LLVM pointers (e.g., from llvm.getelementptr)
    if (isa<LLVM::LLVMPointerType>(resultType))
      return true;
    /// MemRef types (e.g., from memref.cast, memref.subview)
    if (isa<MemRefType>(resultType))
      return true;
    /// Unranked memref types
    if (isa<UnrankedMemRefType>(resultType))
      return true;
  }
  return false;
}

/// Check if an operation is a load from a memref.
/// These loads should be rematerialized to ensure proper dependency tracking
/// in the CreateDbs pass.
static bool loadsFromMemref(Operation *op) {
  return isa<memref::LoadOp>(op);
}

/// Check if an operation uses any value that was rematerialized.
/// This allows us to transitively rematerialize arithmetic operations
/// that depend on rematerialized loads.
static bool usesRematerializedValue(Operation *op,
                                    const DenseMap<Value, Value> &valueMap) {
  for (Value operand : op->getOperands()) {
    if (valueMap.count(operand))
      return true;
  }
  return false;
}

/// Determine if an operation should be rematerialized into the EDT.
/// Returns true if:
/// 1. It produces a pointer-like type (existing behavior), OR
/// 2. It loads from a memref (new behavior), OR
/// 3. It uses values that were already rematerialized (transitive)
///
/// In all cases, the operation must not have WRITE/ALLOCATE/FREE side effects.
static bool shouldRematerialize(Operation *defOp,
                                const DenseMap<Value, Value> &valueMap) {
  if (!defOp || defOp->getNumRegions() != 0)
    return false;

  /// Check for write/allocate/free side effects (disqualifying)
  bool hasSideEffects = false;
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(defOp)) {
    hasSideEffects = memInterface.hasEffect<MemoryEffects::Write>() ||
                     memInterface.hasEffect<MemoryEffects::Allocate>() ||
                     memInterface.hasEffect<MemoryEffects::Free>();
  } else {
    hasSideEffects = !isMemoryEffectFree(defOp);
  }
  if (hasSideEffects)
    return false;

  /// Condition 1: Produces pointer-like type
  if (producesPointerLikeType(defOp))
    return true;

  /// Condition 2: Loads from memref
  if (loadsFromMemref(defOp))
    return true;

  /// Condition 3: Uses rematerialized values (transitive)
  if (usesRematerializedValue(defOp, valueMap))
    return true;

  return false;
}

void rematerializePointersInEdt(EdtOp edt) {
  ARTS_DEBUG("Processing EDT: " << edt);

  SetVector<Value> usedValues;
  auto &edtRegion = edt.getRegion();
  getUsedValuesDefinedAbove(edtRegion, usedValues);

  ARTS_DEBUG("Found " << usedValues.size() << " values used from outside EDT");

  /// Helper function to recursively clone operations and their pointer
  /// dependencies
  OpBuilder builder(edtRegion.getContext());
  builder.setInsertionPointToStart(&edtRegion.front());

  std::function<Value(Value, DenseMap<Value, Value> &)> cloneWithDeps =
      [&](Value val, DenseMap<Value, Value> &valueMap) -> Value {
    ARTS_DEBUG("cloneWithDeps: processing value " << val);

    /// If already cloned, return the mapped value
    if (valueMap.count(val)) {
      ARTS_DEBUG("  Already cloned, returning mapped value");
      return valueMap[val];
    }

    /// Skip if there's no defining operation
    Operation *defOp = val.getDefiningOp();
    if (!defOp) {
      ARTS_DEBUG("  No defining operation, returning original value");
      return val;
    }

    ARTS_DEBUG("  Defining op: " << *defOp);

    /// Check if operation should be rematerialized
    if (!shouldRematerialize(defOp, valueMap)) {
      ARTS_DEBUG("  Skipping: does not meet rematerialization criteria");
      return val;
    }

    /// First clone dependencies
    ARTS_DEBUG("  Cloning dependencies for " << defOp->getNumOperands()
                                             << " operands");
    SmallVector<Value, 4> newOperands;
    for (Value operand : defOp->getOperands()) {
      /// Check if operand needs to be cloned
      Operation *opDefOp = operand.getDefiningOp();
      if (opDefOp && shouldRematerialize(opDefOp, valueMap)) {
        ARTS_DEBUG("    Cloning operand: " << operand);
        newOperands.push_back(cloneWithDeps(operand, valueMap));
      } else {
        ARTS_DEBUG("    Keeping operand as-is: " << operand);
        newOperands.push_back(operand);
      }
    }
    /// Update operands with the cloned versions
    ARTS_DEBUG("  Creating clone of operation");
    auto clonedOp = val.getDefiningOp()->clone();
    for (auto [idx, newOp] : llvm::enumerate(newOperands))
      clonedOp->setOperand(idx, newOp);
    clonedOp = builder.insert(clonedOp);
    builder.setInsertionPointAfter(clonedOp);

    ARTS_DEBUG("  Cloned operation: " << *clonedOp);
    ARTS_DEBUG("  Mapping " << val << " -> " << clonedOp->getResult(0));

    /// Store the mapping
    valueMap[val] = clonedOp->getResult(0);
    return clonedOp->getResult(0);
  };

  DenseMap<Value, Value> valueMap;

  /// Pass 1: Rematerialize pointer-like operations and memref loads
  for (Value operand : usedValues) {
    ARTS_DEBUG("\n--- Processing used value: " << operand);

    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      ARTS_DEBUG("  Skipping: no defining operation");
      continue;
    }

    ARTS_DEBUG("  Defining operation: " << *defOp);

    /// Check if operation should be rematerialized
    if (!shouldRematerialize(defOp, valueMap)) {
      ARTS_DEBUG("  Skipping: does not meet rematerialization criteria");
      continue;
    }

    /// Clone operation and its dependencies recursively
    ARTS_DEBUG("  Rematerializing operation");
    Value newVal = cloneWithDeps(operand, valueMap);
    if (newVal != operand) {
      ARTS_DEBUG("  Replacing uses of " << operand << " with " << newVal);
      replaceInRegion(edtRegion, operand, newVal);
    } else {
      ARTS_DEBUG("  No replacement needed (newVal == operand)");
    }
  }

  /// Pass 2: Handle transitive dependencies
  /// E.g., %a = load ...; %b = addi %a, 5 -> addi must also be rematerialized
  bool changed = true;
  while (changed) {
    changed = false;
    for (Value operand : usedValues) {
      /// Skip if already rematerialized
      if (valueMap.count(operand))
        continue;

      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;

      /// Check if this operation uses any rematerialized values
      if (shouldRematerialize(defOp, valueMap)) {
        ARTS_DEBUG("  Rematerializing transitive dep: " << *defOp);
        Value newVal = cloneWithDeps(operand, valueMap);
        if (newVal != operand) {
          replaceInRegion(edtRegion, operand, newVal);
          changed = true;
        }
      }
    }
  }
}
} // namespace arts
} // namespace mlir
