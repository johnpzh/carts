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
#include "carts/dialect/arts-rt/Conversion/ArtsRtToLLVM/CodegenInternal.h"
#include "carts/dialect/arts-rt/IR/RtDialect.h"
#include "carts/dialect/arts-rt/Utils/RtDbUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;
using namespace mlir::carts::arts_rt;

namespace {

PartitionMode getLoweredDbPartitionMode(DbAllocOp alloc) {
  if (auto mode = getPartitionMode(alloc.getOperation()))
    return *mode;

  bool singleOuterSlot = llvm::all_of(alloc.getSizes(), [](Value size) {
    int64_t value;
    return ValueAnalysis::getConstantIndex(size, value) && value == 1;
  });
  return singleOuterSlot ? PartitionMode::coarse : PartitionMode::fine_grained;
}

} // namespace

LayoutInfo mlir::carts::arts_rt::buildLayoutInfo(Value source) {
  LayoutInfo info;

  if (!source)
    return info;

  if (Operation *rootAllocOp = RtDbUtils::getUnderlyingDbAlloc(source)) {
    if (auto alloc = dyn_cast<DbAllocOp>(rootAllocOp)) {
      info.alloc = alloc;
      info.mode = getLoweredDbPartitionMode(alloc);
      info.sizes.assign(alloc.getSizes().begin(), alloc.getSizes().end());
      info.elementSizes.assign(alloc.getElementSizes().begin(),
                               alloc.getElementSizes().end());
    }
  }

  if (info.sizes.empty()) {
    SmallVector<Value> sizes = RtDbUtils::getSizesFromDb(source);
    info.sizes.assign(sizes.begin(), sizes.end());
  }

  info.outerRank = static_cast<unsigned>(info.sizes.size());
  info.innerRank = static_cast<unsigned>(info.elementSizes.size());
  return info;
}

Value mlir::carts::arts_rt::computeDbElementPointer(ArtsCodegen &AC,
                                                    Location loc, Value base,
                                                    ArrayRef<Value> indices,
                                                    const LayoutInfo &layout) {
  if (indices.empty())
    return AC.castToLLVMPtr(base, loc);

  SmallVector<Value> fallbackSizes;
  ArrayRef<Value> sizes = layout.sizes;
  if (sizes.empty()) {
    fallbackSizes = RtDbUtils::getSizesFromDb(base);
    sizes = fallbackSizes;
  }

  SmallVector<Value> strides = AC.computeStridesFromSizes(sizes, loc);
  SmallVector<Value> linearIndices(indices.begin(), indices.end());
  return AC.create<DbGepOp>(loc, AC.getLLVMPointerType(base), base,
                            linearIndices, strides);
}
