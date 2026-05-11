///==========================================================================///
/// File: DbLayoutStrategy.cpp
///
/// Layout-aware element pointer computation for datablock lowering.
///
/// Before:
///   %ref = arts.db_ref %db[%idx]
///   %elt = arts.db_element_ptr %ref ...
///
/// After:
///   // layout info says "outer blocks + inner element tile"
///   %elt = arts.db_element_ptr %ref[%block][%inner_offset] ...
///==========================================================================///

#include "arts/dialect/core/Transforms/db/DbLayoutStrategy.h"
#include "arts/dialect/core/Analysis/db/DbAnalysis.h"
#include "arts/dialect/core/Conversion/ArtsToLLVM/CodegenSupport.h"
#include "arts/dialect/rt/IR/RtDialect.h"
#include "arts/utils/DbUtils.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::arts;
using namespace mlir::arts::rt;

LayoutInfo mlir::arts::buildLayoutInfo(Value source) {
  LayoutInfo info;

  if (!source)
    return info;

  if (Operation *rootAllocOp = DbUtils::getUnderlyingDbAlloc(source)) {
    if (auto alloc = dyn_cast<DbAllocOp>(rootAllocOp)) {
      info.alloc = alloc;
      info.mode = DbAnalysis::getPartitionModeFromStructure(alloc);
      info.sizes.assign(alloc.getSizes().begin(), alloc.getSizes().end());
      info.elementSizes.assign(alloc.getElementSizes().begin(),
                               alloc.getElementSizes().end());
    }
  }

  if (info.sizes.empty()) {
    SmallVector<Value> sizes = DbUtils::getSizesFromDb(source);
    info.sizes.assign(sizes.begin(), sizes.end());
  }

  info.outerRank = static_cast<unsigned>(info.sizes.size());
  info.innerRank = static_cast<unsigned>(info.elementSizes.size());
  return info;
}

Value mlir::arts::computeDbElementPointer(ArtsCodegen &AC, Location loc,
                                          Value base, ArrayRef<Value> indices,
                                          const LayoutInfo &layout) {
  if (indices.empty())
    return AC.castToLLVMPtr(base, loc);

  SmallVector<Value> fallbackSizes;
  ArrayRef<Value> sizes = layout.sizes;
  if (sizes.empty()) {
    fallbackSizes = DbUtils::getSizesFromDb(base);
    sizes = fallbackSizes;
  }

  SmallVector<Value> strides = AC.computeStridesFromSizes(sizes, loc);
  SmallVector<Value> linearIndices(indices.begin(), indices.end());
  return AC.create<DbGepOp>(loc, AC.getLLVMPointerType(base), base,
                            linearIndices, strides);
}
