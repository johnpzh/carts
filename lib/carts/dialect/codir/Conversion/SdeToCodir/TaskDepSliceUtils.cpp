#include "TaskDepSliceUtils.h"

#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::carts;

namespace mlir::carts::codir {
namespace {

static std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();

  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  }

  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)))
    return std::nullopt;
  return constant.getSExtValue();
}

static bool indexValuesMatch(Value actual, Value expected) {
  std::optional<int64_t> actualConstant = getConstantIndexValue(actual);
  std::optional<int64_t> expectedConstant = getConstantIndexValue(expected);
  if (actualConstant && expectedConstant)
    return *actualConstant == *expectedConstant;
  return arts::ValueAnalysis::sameValue(actual, expected);
}

static bool opFoldResultMatchesValue(OpFoldResult actual, Value expected) {
  if (auto actualValue = dyn_cast<Value>(actual))
    return indexValuesMatch(actualValue, expected);

  auto actualAttr = dyn_cast<Attribute>(actual);
  auto actualInteger = dyn_cast_if_present<IntegerAttr>(actualAttr);
  std::optional<int64_t> expectedConstant = getConstantIndexValue(expected);
  return actualInteger && expectedConstant &&
         actualInteger.getInt() == *expectedConstant;
}

static bool isOneIndexFoldResult(OpFoldResult value) {
  if (auto valueAttr = dyn_cast<Attribute>(value)) {
    auto integer = dyn_cast<IntegerAttr>(valueAttr);
    return integer && integer.getInt() == 1;
  }

  auto valueOperand = dyn_cast<Value>(value);
  std::optional<int64_t> constant = getConstantIndexValue(valueOperand);
  return constant && *constant == 1;
}

static bool isConstantIndexValue(Value value, int64_t expected) {
  std::optional<int64_t> constant = getConstantIndexValue(value);
  return constant && *constant == expected;
}

} // namespace

bool hasCompleteMuDepSlice(sde::SdeMuDepOp muDep) {
  auto sourceType = dyn_cast<MemRefType>(muDep.getSource().getType());
  return sourceType && sourceType.getRank() > 0 &&
         muDep.getOffsets().size() == static_cast<size_t>(sourceType.getRank()) &&
         muDep.getSizes().size() == static_cast<size_t>(sourceType.getRank());
}

bool hasOnlyStaticMuDepSliceBounds(sde::SdeMuDepOp muDep) {
  return llvm::all_of(muDep.getOffsets(), [](Value value) {
           return getConstantIndexValue(value).has_value();
         }) &&
         llvm::all_of(muDep.getSizes(), [](Value value) {
           return getConstantIndexValue(value).has_value();
         });
}

bool hasCodirTaskDepSliceBoundsSupport(sde::SdeMuDepOp muDep) {
  return hasCompleteMuDepSlice(muDep) && hasOnlyStaticMuDepSliceBounds(muDep);
}

bool subviewMatchesMuDepSlice(memref::SubViewOp subview,
                              sde::SdeMuDepOp muDep) {
  if (!subview || !muDep || subview.getSource() != muDep.getSource() ||
      !hasCompleteMuDepSlice(muDep))
    return false;

  if (subview.getMixedOffsets().size() != muDep.getOffsets().size() ||
      subview.getMixedSizes().size() != muDep.getSizes().size() ||
      subview.getMixedStrides().size() != muDep.getOffsets().size())
    return false;

  for (auto [actual, expected] :
       llvm::zip(subview.getMixedOffsets(), muDep.getOffsets()))
    if (!opFoldResultMatchesValue(actual, expected))
      return false;

  for (auto [actual, expected] :
       llvm::zip(subview.getMixedSizes(), muDep.getSizes()))
    if (!opFoldResultMatchesValue(actual, expected))
      return false;

  return llvm::all_of(subview.getMixedStrides(), isOneIndexFoldResult);
}

bool subindexMatchesMuDepSlice(polygeist::SubIndexOp subindex,
                               sde::SdeMuDepOp muDep) {
  if (!subindex || !muDep || subindex.getSource() != muDep.getSource() ||
      !hasCompleteMuDepSlice(muDep))
    return false;

  auto sourceType = dyn_cast<MemRefType>(muDep.getSource().getType());
  auto resultType = dyn_cast<MemRefType>(subindex.getResult().getType());
  if (!sourceType || !resultType || sourceType.getRank() == 0 ||
      resultType.getRank() + 1 != sourceType.getRank())
    return false;

  if (!indexValuesMatch(subindex.getIndex(), muDep.getOffsets().front()) ||
      !isConstantIndexValue(muDep.getSizes().front(), 1))
    return false;

  ValueRange subindexSizes = subindex.getSizes();
  unsigned dynamicSizeIdx = 0;
  for (int64_t sourceDim = 1, rank = sourceType.getRank(); sourceDim < rank;
       ++sourceDim) {
    if (!isConstantIndexValue(muDep.getOffsets()[sourceDim], 0))
      return false;

    unsigned resultDim = static_cast<unsigned>(sourceDim - 1);
    if (resultType.isDynamicDim(resultDim)) {
      if (dynamicSizeIdx >= subindexSizes.size() ||
          !indexValuesMatch(subindexSizes[dynamicSizeIdx],
                            muDep.getSizes()[sourceDim]))
        return false;
      ++dynamicSizeIdx;
      continue;
    }

    if (!isConstantIndexValue(muDep.getSizes()[sourceDim],
                              resultType.getDimSize(resultDim)))
      return false;
  }

  return dynamicSizeIdx == subindexSizes.size();
}

bool haveSameMuDepSlice(sde::SdeMuDepOp lhs, sde::SdeMuDepOp rhs) {
  if (!lhs || !rhs || lhs.getSource() != rhs.getSource() ||
      !hasCompleteMuDepSlice(lhs) || !hasCompleteMuDepSlice(rhs) ||
      lhs.getOffsets().size() != rhs.getOffsets().size() ||
      lhs.getSizes().size() != rhs.getSizes().size())
    return false;

  for (auto [lhsOffset, rhsOffset] :
       llvm::zip(lhs.getOffsets(), rhs.getOffsets()))
    if (!indexValuesMatch(lhsOffset, rhsOffset))
      return false;

  for (auto [lhsSize, rhsSize] : llvm::zip(lhs.getSizes(), rhs.getSizes()))
    if (!indexValuesMatch(lhsSize, rhsSize))
      return false;

  return true;
}

bool accessIndicesMatchMuDepOffsets(ValueRange indices,
                                    sde::SdeMuDepOp muDep) {
  if (!hasCompleteMuDepSlice(muDep) ||
      indices.size() != muDep.getOffsets().size())
    return false;

  for (auto [index, offset] : llvm::zip(indices, muDep.getOffsets()))
    if (!indexValuesMatch(index, offset))
      return false;

  return true;
}

bool rootAccessMatchesMuDepOffsets(Operation *user, sde::SdeMuDepOp muDep) {
  if (auto load = dyn_cast<memref::LoadOp>(user))
    return load.getMemref() == muDep.getSource() &&
           accessIndicesMatchMuDepOffsets(load.getIndices(), muDep);

  if (auto store = dyn_cast<memref::StoreOp>(user))
    return store.getMemref() == muDep.getSource() &&
           accessIndicesMatchMuDepOffsets(store.getIndices(), muDep);

  return false;
}

bool hasOnlyDirectLoadStoreUsersInTask(Value memref, Region &taskRegion) {
  for (Operation *user : memref.getUsers()) {
    if (!taskRegion.isAncestor(user->getParentRegion()))
      continue;
    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      if (load.getMemref() != memref)
        return false;
      continue;
    }
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (store.getMemref() != memref)
        return false;
      continue;
    }
    return false;
  }
  return true;
}

bool hasExactSubviewAccessProofInTask(sde::SdeMuDepOp muDep,
                                      Region &taskRegion) {
  bool sawMatchingSubview = false;
  for (Operation *user : muDep.getSource().getUsers()) {
    if (!taskRegion.isAncestor(user->getParentRegion()))
      continue;

    auto subview = dyn_cast<memref::SubViewOp>(user);
    if (!subview || !subviewMatchesMuDepSlice(subview, muDep) ||
        !hasOnlyDirectLoadStoreUsersInTask(subview.getResult(), taskRegion))
      return false;

    sawMatchingSubview = true;
  }

  return sawMatchingSubview;
}

bool hasExactRootAccessProofInTask(sde::SdeMuDepOp muDep,
                                   Region &taskRegion) {
  bool sawMatchingAccess = false;
  for (Operation *user : muDep.getSource().getUsers()) {
    if (!taskRegion.isAncestor(user->getParentRegion()))
      continue;

    if (rootAccessMatchesMuDepOffsets(user, muDep)) {
      sawMatchingAccess = true;
      continue;
    }

    return false;
  }

  return sawMatchingAccess;
}

bool hasPartitionedExactAccessProofInTask(
    Value source, ArrayRef<sde::SdeMuDepOp> sourceDeps, Region &taskRegion) {
  bool sawMatchingAccess = false;
  for (Operation *user : source.getUsers()) {
    if (!taskRegion.isAncestor(user->getParentRegion()))
      continue;

    bool matchesSourceDep = false;
    if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
      if (!hasOnlyDirectLoadStoreUsersInTask(subview.getResult(), taskRegion))
        return false;
      matchesSourceDep = llvm::any_of(sourceDeps, [&](sde::SdeMuDepOp dep) {
        return subviewMatchesMuDepSlice(subview, dep);
      });
    } else if (auto subindex = dyn_cast<polygeist::SubIndexOp>(user)) {
      if (!hasOnlyDirectLoadStoreUsersInTask(subindex.getResult(), taskRegion))
        return false;
      matchesSourceDep = llvm::any_of(sourceDeps, [&](sde::SdeMuDepOp dep) {
        return subindexMatchesMuDepSlice(subindex, dep);
      });
    } else {
      matchesSourceDep = llvm::any_of(sourceDeps, [&](sde::SdeMuDepOp dep) {
        return rootAccessMatchesMuDepOffsets(user, dep);
      });
    }

    if (!matchesSourceDep)
      return false;

    sawMatchingAccess = true;
  }

  return sawMatchingAccess;
}

void collectExactSubviewAccessProofsInTask(
    sde::SdeMuDepOp muDep, Region &taskRegion,
    SmallVectorImpl<memref::SubViewOp> &subviews) {
  for (Operation *user : muDep.getSource().getUsers()) {
    if (!taskRegion.isAncestor(user->getParentRegion()))
      continue;

    auto subview = dyn_cast<memref::SubViewOp>(user);
    if (subview && subviewMatchesMuDepSlice(subview, muDep) &&
        hasOnlyDirectLoadStoreUsersInTask(subview.getResult(), taskRegion))
      subviews.push_back(subview);
  }
}

void collectExactSubindexAccessProofsInTask(
    sde::SdeMuDepOp muDep, Region &taskRegion,
    SmallVectorImpl<polygeist::SubIndexOp> &subindices) {
  for (Operation *user : muDep.getSource().getUsers()) {
    if (!taskRegion.isAncestor(user->getParentRegion()))
      continue;

    auto subindex = dyn_cast<polygeist::SubIndexOp>(user);
    if (subindex && subindexMatchesMuDepSlice(subindex, muDep) &&
        hasOnlyDirectLoadStoreUsersInTask(subindex.getResult(), taskRegion))
      subindices.push_back(subindex);
  }
}

void collectExactRootAccessProofsInTask(
    sde::SdeMuDepOp muDep, Region &taskRegion,
    SmallVectorImpl<Operation *> &accesses) {
  for (Operation *user : muDep.getSource().getUsers()) {
    if (!taskRegion.isAncestor(user->getParentRegion()))
      continue;

    if (rootAccessMatchesMuDepOffsets(user, muDep))
      accesses.push_back(user);
  }
}

} // namespace mlir::carts::codir
