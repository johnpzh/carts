///==========================================================================///
/// File: EpochOptInternal.h
///
/// Local implementation contract for EpochOpt. This header is intentionally
/// private to the epoch-opt implementation split and should not be used as
/// shared compiler infrastructure.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_TRANSFORMS_EPOCHOPTINTERNAL_H
#define ARTS_DIALECT_CORE_TRANSFORMS_EPOCHOPTINTERNAL_H

#include "carts/Dialect.h"
#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/edt/EpochAnalysis.h"
#include "carts/dialect/arts/Analysis/heuristics/EpochHeuristics.h"
#include "carts/dialect/arts/Utils/DbUtils.h"
#include "carts/dialect/arts/Utils/EdtUtils.h"
#include "carts/utils/LoopUtils.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <climits>
#include <optional>
#include <string>

namespace mlir::carts::arts::epoch_opt {

struct EpochNarrowingCounts {
  unsigned epochsNarrowed = 0;
  unsigned newEpochsCreated = 0;
};

EpochNarrowingCounts narrowEpochScopes(ModuleOp module);
unsigned processRegionForEpochFusion(Region &region,
                                     const EpochAnalysis &epochAnalysis);
bool tryAmortizeRepeatedEpochLoop(EpochOp epochOp);

} // namespace mlir::carts::arts::epoch_opt

#endif // ARTS_DIALECT_CORE_TRANSFORMS_EPOCHOPTINTERNAL_H
