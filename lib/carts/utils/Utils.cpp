///==========================================================================///
/// File: Utils.cpp
/// Defines utility functions for the ARTS dialect.
///==========================================================================///
#include "carts/utils/Utils.h"
#include "carts/Dialect.h"
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/Types.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/RegionUtils.h"
#include <algorithm>
#include <cassert>

/// Debug
#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(utils);
[[maybe_unused]] static const auto *kArtsUtilsDebugType = ARTS_DEBUG_TYPE_STR;

namespace mlir {
namespace carts::arts {

///===----------------------------------------------------------------------===///
/// Dominance Utilities
///===----------------------------------------------------------------------===///

bool dominatesOrInAncestor(Value v, Operation *op, DominanceInfo &domInfo) {
  if (!v || !op)
    return false;
  if (Operation *def = v.getDefiningOp()) {
    Region *defRegion = def->getParentRegion();
    Region *opRegion = op->getParentRegion();
    if (defRegion && opRegion && defRegion != opRegion &&
        defRegion->isAncestor(opRegion))
      return true;
  } else {
    /// Block arguments from ancestor regions are considered in-scope.
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (Region *argRegion = arg.getOwner()->getParent())
        if (Region *opRegion = op->getParentRegion())
          if (argRegion->isAncestor(opRegion))
            return true;
    }
  }
  return domInfo.dominates(v, op);
}

///===----------------------------------------------------------------------===///
/// Constant Index Creation Utilities
///===----------------------------------------------------------------------===///

Value createConstantIndex(OpBuilder &builder, Location loc, int64_t val) {
  return arith::ConstantIndexOp::create(builder, loc, val);
}

Value createZeroIndex(OpBuilder &builder, Location loc) {
  return createConstantIndex(builder, loc, 0);
}

Value createOneIndex(OpBuilder &builder, Location loc) {
  return createConstantIndex(builder, loc, 1);
}

Value createCeilDivUI(OpBuilder &builder, Location loc, Value numerator,
                      Value denominator) {
  Value one = createOneIndex(builder, loc);
  Value adjusted = arith::AddIOp::create(
      builder, loc, numerator,
      arith::SubIOp::create(builder, loc, denominator, one));
  return arith::DivUIOp::create(builder, loc, adjusted, denominator);
}

Value createCurrentNodeRoute(OpBuilder &builder, Location loc) {
  return arith::ConstantIntOp::create(builder, loc, kCurrentNodeRoute, 32);
}

///===----------------------------------------------------------------------===///
/// OMP Region Utilities
///===----------------------------------------------------------------------===///

bool isInsideOmpRegion(Operation *op) {
  for (Operation *ancestor = op ? op->getParentOp() : nullptr; ancestor;
       ancestor = ancestor->getParentOp()) {
    if (ancestor->getDialect() &&
        ancestor->getDialect()->getNamespace() == "omp")
      return true;
  }
  return false;
}

bool containsOmpOp(Operation *op) {
  bool found = false;
  op->walk([&](Operation *nested) {
    if (nested->getDialect() && nested->getDialect()->getNamespace() == "omp") {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

///===----------------------------------------------------------------------===///
/// Operation Ordering Utilities
///===----------------------------------------------------------------------===///

Operation *getTopLevelFuncChild(Operation *op, func::FuncOp parentFunc) {
  if (!op || !parentFunc)
    return nullptr;

  Operation *cursor = op;
  while (cursor && cursor->getParentOp() != parentFunc)
    cursor = cursor->getParentOp();
  return cursor;
}

bool isOrderedAfter(Operation *anchor, Operation *candidate,
                    func::FuncOp parentFunc) {
  Operation *anchorTop = getTopLevelFuncChild(anchor, parentFunc);
  Operation *candidateTop = getTopLevelFuncChild(candidate, parentFunc);
  if (!anchorTop || !candidateTop)
    return false;
  if (anchorTop->getBlock() != candidateTop->getBlock())
    return false;
  return anchorTop->isBeforeInBlock(candidateTop);
}

///===----------------------------------------------------------------------===///
/// ARTS Runtime Query Utilities
///===----------------------------------------------------------------------===///

bool isArtsRuntimeQuery(Value val) {
  if (!val)
    return false;

  val = ValueAnalysis::stripNumericCasts(val);
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return false;

  /// Check for ARTS dialect ops (before lowering to func::CallOp)
  if (isa<RuntimeQueryOp>(defOp))
    return true;

  /// Check for func::CallOp (after lowering)
  if (auto callOp = dyn_cast<func::CallOp>(defOp)) {
    std::optional<types::RuntimeFunction> fn =
        types::runtimeFunctionFromName(callOp.getCallee());
    if (!fn)
      return false;
    switch (*fn) {
    case types::ARTSRTL_arts_get_current_node:
    case types::ARTSRTL_arts_get_total_nodes:
    case types::ARTSRTL_arts_get_current_worker:
    case types::ARTSRTL_arts_get_total_workers:
      return true;
    default:
      return false;
    }
  }

  return false;
}

///===----------------------------------------------------------------------===///
/// IR Simplification Utilities
///===----------------------------------------------------------------------===///

/// Simplifies the IR by running common subexpression elimination (CSE)
/// across the module using the provided dominance information.
bool simplifyIR(ModuleOp module, DominanceInfo &domInfo) {
  IRRewriter rewriter(module.getContext());
  bool changed = false;
  eliminateCommonSubExpressions(rewriter, domInfo, module, &changed);

  return changed;
}

///===----------------------------------------------------------------------===///
/// Type and Size Utilities
///===----------------------------------------------------------------------===///

/// Computes the byte size of the given element type, supporting integer and
/// floating-point types. When the type is a memref, all static dimensions must
/// be known; otherwise, the size is treated as unknown (returns 0).
uint64_t getElementTypeByteSize(Type elemTy) {
  /// Safety check: return 0 for null or invalid types
  if (!elemTy) {
    return 0;
  }

  if (auto memTy = dyn_cast<MemRefType>(elemTy)) {
    uint64_t elementBytes = getElementTypeByteSize(memTy.getElementType());
    if (elementBytes == 0)
      return 0;

    uint64_t total = elementBytes;
    for (int64_t dim : memTy.getShape()) {
      if (dim == ShapedType::kDynamic)
        return 0;
      total *= static_cast<uint64_t>(std::max<int64_t>(dim, 0));
    }
    return total;
  }
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return intTy.getWidth() / 8u;
  if (auto fTy = dyn_cast<FloatType>(elemTy))
    return fTy.getWidth() / 8u;
  /// Unknown type
  return 0;
}

/// Gets the element memref type for a given element type and sizes.
MemRefType getElementMemRefType(Type elementType,
                                ArrayRef<Value> elementSizes) {
  /// Enforce scalar payloads use a single trailing dimension of 1 instead of
  /// an empty shape to keep rank handling uniform across the pipeline.
  const size_t rank = elementSizes.empty() ? 1 : elementSizes.size();
  SmallVector<int64_t> elementShape(rank, ShapedType::kDynamic);
  return MemRefType::get(elementShape, elementType);
}

///===----------------------------------------------------------------------===///
/// String Utilities
///===----------------------------------------------------------------------===///

/// Sanitizes the given string by replacing dots and dashes with underscores,
/// making it suitable for use as an identifier.
std::string sanitizeString(StringRef s) {
  std::string id = s.str();
  std::replace(id.begin(), id.end(), '.', '_');
  std::replace(id.begin(), id.end(), '-', '_');
  return id;
}

///===----------------------------------------------------------------------===///
/// Access Mode Utilities
///===----------------------------------------------------------------------===///

/// Combine two access modes and return the more permissive mode
ArtsMode combineAccessModes(ArtsMode mode1, ArtsMode mode2) {
  /// If either mode is uninitialized, return the other mode
  if (mode1 == ArtsMode::uninitialized)
    return mode2;
  if (mode2 == ArtsMode::uninitialized)
    return mode1;

  /// If either mode is inout, the result is inout (most permissive)
  if (mode1 == ArtsMode::inout || mode2 == ArtsMode::inout)
    return ArtsMode::inout;

  /// If one is 'in' and the other is 'out', the result is inout
  if ((mode1 == ArtsMode::in && mode2 == ArtsMode::out) ||
      (mode1 == ArtsMode::out && mode2 == ArtsMode::in))
    return ArtsMode::inout;

  /// If both are the same, return that mode
  if (mode1 == mode2)
    return mode1;

  /// Default to inout for any other combination (shouldn't happen)
  return ArtsMode::inout;
}

///===----------------------------------------------------------------------===///
/// Operation Replacement Utilities
///===----------------------------------------------------------------------===///

/// Replaces uses of one value with another under dominance constraints,
/// skipping the dominating operation.
void replaceUses(Value from, Value to, DominanceInfo &domInfo,
                 Operation *dominatingOp) {
  /// Ensure that 'from' has a defining operation.
  auto *definingOp = from.getDefiningOp();
  if (!definingOp)
    return;

  /// Replace all uses of 'from' with 'to' if the user is dominated by the
  /// defining operation of 'from'.
  from.replaceUsesWithIf(to, [&](OpOperand &operand) {
    if (operand.getOwner() == dominatingOp)
      return false;
    return domInfo.dominates(dominatingOp, operand.getOwner());
  });
}

/// Replaces uses according to a mapping of values.
void replaceUses(DenseMap<Value, Value> &rewireMap) {
  for (auto &rewire : rewireMap)
    rewire.first.replaceAllUsesWith(rewire.second);
  rewireMap.clear();
}

/// Replaces uses of a value within a specific region.
void replaceInRegion(Region &region, Value from, Value to) {
  from.replaceUsesWithIf(to, [&](OpOperand &operand) {
    return region.isAncestor(operand.getOwner()->getParentRegion());
  });
}

/// Replaces uses according to a mapping within a specific region.
void replaceInRegion(Region &region, DenseMap<Value, Value> &rewireMap,
                     bool clear) {
  for (auto &rewire : rewireMap)
    replaceInRegion(region, rewire.first, rewire.second);
  if (clear)
    rewireMap.clear();
}

///===----------------------------------------------------------------------===///
/// Attribute Transfer Utilities
///===----------------------------------------------------------------------===///

void transferAttributes(Operation *src, Operation *dst,
                        ArrayRef<StringRef> excludes) {
  /// Build set of attributes to exclude
  llvm::SmallDenseSet<StringRef> excludeSet;
  /// Always exclude auto-generated MLIR attributes
  excludeSet.insert("operandSegmentSizes");
  excludeSet.insert("resultSegmentSizes");
  /// Exclude builder-set attributes for ARTS ops
  excludeSet.insert("mode");
  /// Add any additional excludes
  for (StringRef name : excludes)
    excludeSet.insert(name);

  /// Transfer all non-excluded attributes
  for (NamedAttribute attr : src->getAttrs()) {
    if (!excludeSet.contains(attr.getName().getValue()))
      dst->setAttr(attr.getName(), attr.getValue());
  }
}

///===----------------------------------------------------------------------===///
/// OMP Mode Conversion Utilities
///===----------------------------------------------------------------------===///

bool isSideEffectFreeArithmeticLikeOp(Operation *op) {
  if (!op || op->hasTrait<OpTrait::IsTerminator>())
    return true;

  return isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
             arith::DivUIOp, arith::DivSIOp, arith::RemUIOp, arith::RemSIOp,
             arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::NegFOp, arith::CmpIOp, arith::CmpFOp, arith::SelectOp,
             arith::MaxUIOp, arith::MinUIOp, arith::MaximumFOp,
             arith::MinimumFOp, arith::IndexCastOp, arith::IndexCastUIOp,
             arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncIOp,
             arith::TruncFOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
             arith::FPToUIOp>(op);
}

ArtsMode convertOmpMode(omp::ClauseTaskDepend mode) {
  switch (mode) {
  case omp::ClauseTaskDepend::taskdependin:
    return ArtsMode::in;
  case omp::ClauseTaskDepend::taskdependout:
    return ArtsMode::out;
  case omp::ClauseTaskDepend::taskdependinout:
  case omp::ClauseTaskDepend::taskdependmutexinoutset:
  case omp::ClauseTaskDepend::taskdependinoutset:
    return ArtsMode::inout;
  }
  llvm_unreachable("Unknown OMP depend mode");
}

///===----------------------------------------------------------------------===///
/// Dependency Index Clamping Utilities
///===----------------------------------------------------------------------===///

SmallVector<Value> clampDepIndices(Value source, ArrayRef<Value> indices,
                                   OpBuilder &builder, Location loc,
                                   ArrayRef<Value> dimSizes) {
  SmallVector<Value> clamped;
  clamped.reserve(indices.size());

  auto sourceType = dyn_cast<MemRefType>(source.getType());
  if (!sourceType) {
    clamped.append(indices.begin(), indices.end());
    return clamped;
  }

  auto castToType = [&](Value v, Type targetTy) -> Value {
    if (v.getType() == targetTy)
      return v;
    if ((v.getType().isIndex() && isa<IntegerType>(targetTy)) ||
        (isa<IntegerType>(v.getType()) && targetTy.isIndex()))
      return arith::IndexCastOp::create(builder, loc, targetTy, v);
    return v;
  };

  auto oneForType = [&](Type ty) -> Value {
    if (ty.isIndex())
      return createOneIndex(builder, loc);
    if (auto intTy = dyn_cast<IntegerType>(ty))
      return arith::ConstantIntOp::create(builder, loc, intTy, 1);
    return Value();
  };

  auto zeroForType = [&](Type ty) -> Value {
    if (ty.isIndex())
      return createZeroIndex(builder, loc);
    if (auto intTy = dyn_cast<IntegerType>(ty))
      return arith::ConstantIntOp::create(builder, loc, intTy, 0);
    return Value();
  };

  for (auto [dim, idx] : llvm::enumerate(indices)) {
    Type idxTy = idx.getType();
    if (!idxTy.isIndex() && !isa<IntegerType>(idxTy)) {
      clamped.push_back(idx);
      continue;
    }

    if (dim >= static_cast<size_t>(sourceType.getRank())) {
      clamped.push_back(idx);
      continue;
    }

    Value zero = zeroForType(idxTy);
    Value one = oneForType(idxTy);
    if (!zero || !one) {
      clamped.push_back(idx);
      continue;
    }

    Value dimSize;
    if (dim < dimSizes.size()) {
      dimSize = dimSizes[dim];
    } else if (sourceType.isDynamicDim(dim)) {
      dimSize = memref::DimOp::create(builder, loc, source, dim);
    } else {
      dimSize = createConstantIndex(builder, loc, sourceType.getDimSize(dim));
    }

    Value dimSizeTyped = castToType(dimSize, idxTy);
    Value upper = arith::SubIOp::create(builder, loc, dimSizeTyped, one);

    Value belowZero = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::slt, idx, zero);
    Value afterLowerClamp =
        arith::SelectOp::create(builder, loc, belowZero, zero, idx);

    Value aboveUpper = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sgt, afterLowerClamp, upper);
    Value finalIdx = arith::SelectOp::create(builder, loc, aboveUpper, upper,
                                             afterLowerClamp);
    clamped.push_back(finalIdx);
  }

  return clamped;
}

///===----------------------------------------------------------------------===///
/// Block Work Detection Utilities
///===----------------------------------------------------------------------===///

bool hasWorkAfterInParentBlock(Operation *op) {
  if (!op || !op->getBlock())
    return false;

  for (Operation *next = op->getNextNode(); next; next = next->getNextNode()) {
    if (next->hasTrait<OpTrait::IsTerminator>())
      continue;
    return true;
  }
  return false;
}

///===----------------------------------------------------------------------===///
/// Pure Operation Detection Utilities
///===----------------------------------------------------------------------===///

bool isPureOp(Operation *op) {
  if (!op)
    return false;
  if (isa<arts::YieldOp, memref::LoadOp, memref::StoreOp,
          affine::AffineLoadOp, affine::AffineStoreOp, affine::AffineApplyOp,
          scf::ForOp, scf::YieldOp, affine::AffineForOp, affine::AffineYieldOp>(
          op))
    return true;
  if (isa<CallOpInterface>(op))
    return false;
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op))
    return iface.hasNoEffect();
  return op->getNumRegions() == 0;
}

///===----------------------------------------------------------------------===///
/// Store Ordering Utilities
///===----------------------------------------------------------------------===///

void sortStoresInProgramOrder(MutableArrayRef<memref::StoreOp> stores) {
  std::stable_sort(stores.begin(), stores.end(),
                   [](memref::StoreOp lhs, memref::StoreOp rhs) {
                     Operation *lhsOp = lhs.getOperation();
                     Operation *rhsOp = rhs.getOperation();
                     if (lhsOp == rhsOp)
                       return false;
                     if (lhsOp->getBlock() == rhsOp->getBlock())
                       return lhsOp->isBeforeInBlock(rhsOp);
                     return lhsOp < rhsOp;
                   });
}

///===----------------------------------------------------------------------===///
/// Undef-Like Operation Detection Utilities
///===----------------------------------------------------------------------===///

bool isUndefLikeOp(Operation *op) {
  if (!op)
    return false;
  StringRef name = op->getName().getStringRef();
  return name == "llvm.mlir.undef" || name == "polygeist.undef" ||
         name == "arts.undef";
}

} // namespace carts::arts
} // namespace mlir
