///==========================================================================///
/// File: Db/DbTransforms.cpp
/// Unified transformations for datablock operations.
///
/// Before:
///   %db = arts.db_alloc ... {partitioning = #arts.partitioning<elementwise>}
///
/// After:
///   // dispatched to the elementwise/block/stencil rewriter selected by mode
///   %db = arts.db_alloc ... {partitioning = #arts.partitioning<elementwise>}
///   %a  = arts.db_acquire %db ...
///==========================================================================///

#include "carts/dialect/arts/Transforms/db/DbTransforms.h"
