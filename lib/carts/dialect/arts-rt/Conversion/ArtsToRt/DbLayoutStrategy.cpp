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

#include "carts/dialect/arts-rt/Conversion/ArtsToRt/DbLayoutStrategy.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/CodegenSupport.h"
#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
using namespace mlir::carts::arts_rt;

LayoutInfo mlir::carts::arts::buildLayoutInfo(Value source) {
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

Value mlir::carts::arts::computeDbElementPointer(ArtsCodegen &AC, Location loc,
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
