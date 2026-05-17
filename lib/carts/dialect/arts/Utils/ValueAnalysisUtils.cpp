///==========================================================================///
/// File: ValueAnalysisUtils.cpp
///
/// ARTS-owned value analysis extensions.
///==========================================================================///

#include "carts/dialect/arts/Utils/ValueAnalysisUtils.h"

#include "carts/utils/OperationAttributes.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {

static constexpr unsigned kMaxTraceDepth = 64;

static std::optional<int64_t> foldRuntimeQuery(Value value, unsigned) {
  auto query = value.getDefiningOp<RuntimeQueryOp>();
  if (!query)
    return std::nullopt;

  ModuleOp module = query->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;

  switch (query.getKind()) {
  case RuntimeQueryKind::totalWorkers:
    return getRuntimeTotalWorkers(module);
  case RuntimeQueryKind::totalNodes:
    return getRuntimeTotalNodes(module);
  default:
    return std::nullopt;
  }
}

static bool isDerivedFromPtrImpl(Value value, Value source,
                                 llvm::SmallPtrSetImpl<Value> &visited,
                                 unsigned depth) {
  if (depth > kMaxTraceDepth)
    return false;
  if (value == source)
    return true;
  if (!visited.insert(value).second)
    return false;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *parentBlock = blockArg.getParentBlock();
    if (parentBlock && parentBlock->getParentOp()) {
      if (auto edt = dyn_cast<EdtOp>(parentBlock->getParentOp())) {
        unsigned argIndex = blockArg.getArgNumber();
        ValueRange deps = edt.getDependencies();
        if (argIndex < deps.size())
          return isDerivedFromPtrImpl(deps[argIndex], source, visited,
                                      depth + 1);
      }
    }
    return false;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;

  auto trace = [&](Value next) -> bool {
    return isDerivedFromPtrImpl(next, source, visited, depth + 1);
  };

  if (auto dbAcquire = dyn_cast<DbAcquireOp>(defOp))
    return trace(dbAcquire.getSourcePtr());
  if (auto dbRef = dyn_cast<DbRefOp>(defOp))
    return trace(dbRef.getSource());
  if (auto gepOp = dyn_cast<LLVM::GEPOp>(defOp))
    return trace(gepOp.getBase());
  if (auto ptr2memref = dyn_cast<polygeist::Pointer2MemrefOp>(defOp))
    return trace(ptr2memref.getSource());
  if (auto memref2ptr = dyn_cast<polygeist::Memref2PointerOp>(defOp))
    return trace(memref2ptr.getSource());
  if (auto subview = dyn_cast<memref::SubViewOp>(defOp))
    return trace(subview.getSource());
  if (auto cast = dyn_cast<memref::CastOp>(defOp))
    return trace(cast.getSource());
  if (auto view = dyn_cast<memref::ViewOp>(defOp))
    return trace(view.getSource());
  if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(defOp))
    return trace(reinterpretCast.getSource());

  if (defOp->getNumOperands() > 0)
    return trace(defOp->getOperand(0));

  return false;
}

static Value getUnderlyingValueImpl(Value value,
                                    llvm::SmallPtrSetImpl<Value> &visited,
                                    unsigned depth) {
  if (!value || depth > kMaxTraceDepth || !visited.insert(value).second)
    return nullptr;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *owner = blockArg.getOwner();
    if (owner && owner->getParentOp()) {
      if (auto edt = dyn_cast<EdtOp>(owner->getParentOp())) {
        unsigned argIndex = blockArg.getArgNumber();
        ValueRange deps = edt.getDependencies();
        if (argIndex < deps.size())
          return getUnderlyingValueImpl(deps[argIndex], visited, depth + 1);
      }
    }
    return value;
  }

  Operation *op = value.getDefiningOp();
  if (!op)
    return nullptr;
  if (!op->getBlock())
    return nullptr;

  if (isa<DbAllocOp, memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(
          op))
    return value;

  auto trace = [&](Value source) -> Value {
    return getUnderlyingValueImpl(source, visited, depth + 1);
  };

  if (auto dbAcquire = dyn_cast<DbAcquireOp>(op))
    return trace(dbAcquire.getSourcePtr());
  if (auto dbRef = dyn_cast<DbRefOp>(op))
    return trace(dbRef.getSource());
  if (auto subview = dyn_cast<memref::SubViewOp>(op))
    return trace(subview.getSource());
  if (auto castOp = dyn_cast<memref::CastOp>(op))
    return trace(castOp.getSource());
  if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(op)) {
    ValueRange inputs = unrealized.getInputs();
    return inputs.empty() ? nullptr : trace(inputs.front());
  }
  if (auto view = dyn_cast<memref::ViewOp>(op))
    return trace(view.getSource());
  if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(op))
    return trace(reinterpret.getSource());
  if (auto p2m = dyn_cast<polygeist::Pointer2MemrefOp>(op))
    return trace(p2m.getSource());
  if (auto m2p = dyn_cast<polygeist::Memref2PointerOp>(op))
    return trace(m2p.getSource());
  if (auto gep = dyn_cast<LLVM::GEPOp>(op))
    return trace(gep.getBase());
  if (auto subindex = dyn_cast<polygeist::SubIndexOp>(op))
    return trace(subindex.getSource());

  return nullptr;
}

} // namespace

std::optional<int64_t> mlir::carts::arts::tryFoldConstantIndex(Value value,
                                                               unsigned depth) {
  return ValueAnalysis::tryFoldConstantIndexWith(value, foldRuntimeQuery,
                                                 depth);
}

std::optional<int64_t> mlir::carts::arts::getConstantIndexStripped(
    Value value) {
  if (!value)
    return std::nullopt;
  return tryFoldConstantIndex(ValueAnalysis::stripNumericCasts(value));
}

Value mlir::carts::arts::getUnderlyingValue(Value value) {
  llvm::SmallPtrSet<Value, 16> visited;
  return getUnderlyingValueImpl(value, visited, 0);
}

Operation *mlir::carts::arts::getUnderlyingOperation(Value value) {
  Value underlying = getUnderlyingValue(value);
  return underlying ? underlying.getDefiningOp() : nullptr;
}

bool mlir::carts::arts::isDerivedFromPtr(Value value, Value source) {
  llvm::SmallPtrSet<Value, 16> visited;
  return isDerivedFromPtrImpl(value, source, visited, 0);
}
