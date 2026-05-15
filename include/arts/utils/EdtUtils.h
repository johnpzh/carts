///==========================================================================///
/// File: EdtUtils.h
///
/// Static utility surface and EdtEnvManager for working with ARTS EDTs
/// (EdtOp). This keeps EDT-local IR helpers out of the dialect op classes
/// while avoiding unrelated free functions in the top-level arts namespace.
///==========================================================================///

#ifndef CARTS_UTILS_EDTUTILS_H
#define CARTS_UTILS_EDTUTILS_H

#include "arts/Dialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
class BlockArgument;
}

namespace mlir {
namespace arts {

/// Forward declarations
class EdtOp;
class EpochOp;
class DbAcquireOp;

///===----------------------------------------------------------------------===//
/// EdtUtils
///===----------------------------------------------------------------------===//

/// Static utilities for EDT-local structure, capture contracts, and dependency
/// argument inspection. Keep helpers here when they interpret EdtOp regions but
/// are not fundamental MLIR op accessors.
class EdtUtils {
public:
  /// Move all non-terminator operations from src block into dst block,
  /// inserting them before dst's terminator.
  static void spliceBodyBeforeTerminator(Block &src, Block &dst) {
    Operation *terminator = dst.getTerminator();
    for (Operation &op : llvm::make_early_inc_range(src.without_terminator()))
      op.moveBefore(terminator);
  }

  /// Fuse consecutive pairs of operations of type OpT in a block.
  /// canFuse checks if two consecutive ops can be fused.
  /// doFuse performs the fusion (must erase or modify the second op).
  /// Returns true if any fusion was performed.
  template <typename OpT>
  static bool fuseConsecutivePairs(
      Block &block, llvm::function_ref<bool(OpT, OpT)> canFuse,
      llvm::function_ref<void(OpT, OpT)> doFuse) {
    bool changed = false;
    for (auto it = block.begin(); it != block.end();) {
      auto first = dyn_cast<OpT>(&*it);
      if (!first) {
        ++it;
        continue;
      }
      auto nextIt = std::next(it);
      if (nextIt == block.end()) {
        ++it;
        continue;
      }
      auto second = dyn_cast<OpT>(&*nextIt);
      if (!second) {
        ++it;
        continue;
      }
      if (!canFuse(first, second)) {
        ++it;
        continue;
      }
      doFuse(first, second);
      changed = true;
      /// Re-run on same first op to chain-fuse.
    }
    return changed;
  }

  /// Return true when an EDT is nested inside an EpochOp.
  static bool isInsideEpoch(EdtOp op) {
    return op->getParentOfType<EpochOp>() != nullptr;
  }

  /// Return true when an operation is nested inside an EpochOp.
  static bool isInsideEpoch(Operation *op) {
    return op->getParentOfType<EpochOp>() != nullptr;
  }

  /// Wrap all operations (except terminator) in a block inside an EpochOp.
  /// Returns the created EpochOp, or nullptr if no operations to wrap.
  static EpochOp wrapBodyInEpoch(Block &body, Location loc);

  /// Finds the EdtOp that uses a DbAcquireOp and returns the corresponding
  /// block argument. Returns {EdtOp, BlockArgument} pair, or {nullptr,
  /// BlockArgument()} if not found.
  static std::pair<EdtOp, BlockArgument>
  getBlockArgumentForAcquire(DbAcquireOp acquireOp);

  /// Map a memref value back to its EDT block argument index by stripping
  /// view-like wrapper ops and DbRefOp.
  static std::optional<unsigned> mapMemrefToArg(EdtOp edt, Value memrefValue);

  /// Classify each EDT dependency as read, written, or both by walking all
  /// load/store operations in the EDT body and tracing accessed memrefs
  /// back to their block arguments.
  static void classifyArgAccesses(EdtOp edt, SmallVectorImpl<bool> &reads,
                                  SmallVectorImpl<bool> &writes);

  /// Return true when an alloca initialization store can be cloned into an EDT
  /// body without needing its surrounding control flow. This accepts constant
  /// or pure regionless operand chains whose inputs can be captured by the EDT.
  static bool canCloneAllocaInitStore(memref::StoreOp store, Value memref);

  /// Trace an externally captured DB/heap view back to its pointer-bearing
  /// handle. This intentionally follows only view-like operations; it returns
  /// null when the value is not rooted in a DB or heap handle.
  static Value traceCapturedDbHandle(Value value);

  /// Classify an explicit ordered list of EDT user values using the same
  /// scalar parameter, constant, and pointer-bearing handle contract as EDT
  /// lowering.
  static void classifyUserValues(ArrayRef<Value> userValues,
                                 llvm::SetVector<Value> &parameters,
                                 llvm::SetVector<Value> &constants,
                                 llvm::SetVector<Value> &dbHandles);

  /// Analyze the values an EDT captures from above its region using the same
  /// classification contract as EDT lowering.
  static void analyzeCapturedValues(EdtOp edt,
                                    llvm::SetVector<Value> &capturedValues,
                                    llvm::SetVector<Value> &parameters,
                                    llvm::SetVector<Value> &constants,
                                    llvm::SetVector<Value> &dbHandles);

  /// Collect scalar values in the same logical order EdtLowering packs them
  /// into `arts.edt_param_pack`: user params, then dep-derived scalars
  /// (indices/offsets/sizes/partition slices/element sizes).
  static SmallVector<Value> collectPackedValues(EdtOp edt);
};

///===----------------------------------------------------------------------===//
/// EdtEnvManager
///===----------------------------------------------------------------------===//

/// Manages EDT environment classification: captured values are partitioned
/// into explicit scalar parameters, constants, pointer-bearing DB handles, and
/// dependencies. Direct scalar captures are not recovered here; they must
/// already be listed on `arts.edt params(...)`.
class EdtEnvManager {
public:
  EdtEnvManager(EdtOp edtOp) : edtOp(edtOp) { analyze(); }

  void analyze() {
    llvm::SetVector<Value> uniqueParameters;
    for (Value param : edtOp.getParams()) {
      parameters.push_back(param);
      uniqueParameters.insert(param);
    }
    EdtUtils::analyzeCapturedValues(edtOp, capturedValues, uniqueParameters,
                                    constants, dbHandles);
    for (Value operand : edtOp.getDependencies())
      deps.insert(operand);
  }

  ArrayRef<Value> getParameters() const { return parameters; }
  ArrayRef<Value> getConstants() const { return constants.getArrayRef(); }
  ArrayRef<Value> getCapturedValues() const {
    return capturedValues.getArrayRef();
  }
  ArrayRef<Value> getDependencies() const { return deps.getArrayRef(); }
  ArrayRef<Value> getDbHandles() const { return dbHandles.getArrayRef(); }
  const DenseMap<Value, unsigned> &getValueToPackIndex() const {
    return valueToPackIndex;
  }
  DenseMap<Value, unsigned> &getValueToPackIndex() { return valueToPackIndex; }

private:
  EdtOp edtOp;
  SmallVector<Value> parameters;
  SetVector<Value> capturedValues, constants, deps, dbHandles;
  DenseMap<Value, unsigned> valueToPackIndex;
};

} // namespace arts
} // namespace mlir

#endif // CARTS_UTILS_EDTUTILS_H
