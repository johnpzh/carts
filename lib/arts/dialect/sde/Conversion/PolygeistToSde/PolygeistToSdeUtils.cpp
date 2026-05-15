///==========================================================================///
/// File: PolygeistToSdeUtils.cpp
///
/// Shared helpers for the Polygeist-to-SDE conversion passes.
///==========================================================================///

#include "PolygeistToSdeUtils.h"

namespace mlir::arts {

Value materializeDependView(OpBuilder &builder, Location loc, Value source,
                            ArrayRef<Value> indices, ArrayRef<Value> sizes) {
  if (indices.empty() && sizes.empty())
    return source;

  auto sourceType = dyn_cast<MemRefType>(source.getType());
  if (!sourceType)
    return source;

  unsigned rank = sourceType.getRank();
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> viewSizes;
  SmallVector<OpFoldResult> strides;
  offsets.reserve(rank);
  viewSizes.reserve(rank);
  strides.reserve(rank);

  for (unsigned dim = 0; dim < rank; ++dim) {
    offsets.push_back(dim < indices.size() ? OpFoldResult(indices[dim])
                                           : builder.getIndexAttr(0));
    if (dim < sizes.size()) {
      viewSizes.push_back(sizes[dim]);
    } else if (!sourceType.isDynamicDim(dim)) {
      viewSizes.push_back(builder.getIndexAttr(sourceType.getDimSize(dim)));
    } else {
      viewSizes.push_back(memref::DimOp::create(builder, loc, source, dim)
                              .getResult());
    }
    strides.push_back(builder.getIndexAttr(1));
  }

  return memref::SubViewOp::create(builder, loc, source, offsets, viewSizes,
                                   strides)
      .getResult();
}

} // namespace mlir::arts
