///==========================================================================///
/// File: DbAliasAnalysis.cpp
/// Defines alias analysis for DB operations and memory tracking.
///==========================================================================///

#include "carts/dialect/arts/Analysis/db/DbAliasAnalysis.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbGraph.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/utils/Utils.h"
#include "carts/utils/ValueAnalysis.h"

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(db_alias_analysis);

#include <cassert>
#include <optional>

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {
struct ConstantSlice {
  SmallVector<int64_t, 4> offsets;
  SmallVector<int64_t, 4> sizes;
};

const DbAllocNode *getRootAlloc(const NodeBase &node) {
  if (auto *acquire = dyn_cast<DbAcquireNode>(&node))
    return acquire->getRootAlloc();
  if (auto *alloc = dyn_cast<DbAllocNode>(&node))
    return alloc;
  return nullptr;
}

std::optional<ConstantSlice> extractConstantSlice(DbAcquireOp op) {
  ConstantSlice slice;
  auto extractRange = [&](ValueRange range,
                          SmallVectorImpl<int64_t> &storage) -> bool {
    storage.clear();
    storage.reserve(range.size());
    for (Value v : range) {
      int64_t cst = 0;
      if (!ValueAnalysis::getConstantIndex(v, cst))
        return false;
      storage.push_back(cst);
    }
    return true;
  };

  if (!extractRange(op.getOffsets(), slice.offsets))
    return std::nullopt;
  if (!extractRange(op.getSizes(), slice.sizes))
    return std::nullopt;
  return slice;
}

std::optional<DbAliasAnalysis::AliasResult>
refineAliasWithSlices(const DbAcquireNode *a, const DbAcquireNode *b) {
  if (!a || !b)
    return std::nullopt;
  auto sliceA = extractConstantSlice(a->getDbAcquireOp());
  auto sliceB = extractConstantSlice(b->getDbAcquireOp());
  if (!sliceA || !sliceB)
    return std::nullopt;

  /// Normalize dimensionality
  if (sliceA->offsets.size() != sliceB->offsets.size() ||
      sliceA->sizes.size() != sliceB->sizes.size())
    return std::nullopt;

  bool identical =
      sliceA->offsets == sliceB->offsets && sliceA->sizes == sliceB->sizes;
  if (identical)
    return DbAliasAnalysis::AliasResult::MustAlias;

  bool disjoint = false;
  for (size_t dim = 0; dim < sliceA->sizes.size(); ++dim) {
    int64_t beginA = sliceA->offsets[dim];
    int64_t endA = beginA + sliceA->sizes[dim];
    int64_t beginB = sliceB->offsets[dim];
    int64_t endB = beginB + sliceB->sizes[dim];
    if (endA <= beginB || endB <= beginA) {
      disjoint = true;
      break;
    }
  }

  if (disjoint)
    return DbAliasAnalysis::AliasResult::NoAlias;
  return DbAliasAnalysis::AliasResult::MayAlias;
}

/// Refine overlap using partition offsets/sizes from DbAcquireOp.
/// These are element-space partition hints (from depend clauses) and are more
/// reliable than beginIndex/endIndex ordering counters for proving
/// disjointness.
std::optional<DbAliasAnalysis::OverlapKind>
refineOverlapWithPartitionRanges(const DbAcquireNode *a,
                                 const DbAcquireNode *b) {
  DbAcquireOp opA = a->getDbAcquireOp();
  DbAcquireOp opB = b->getDbAcquireOp();
  if (!opA || !opB)
    return std::nullopt;

  ValueRange partOffsA = opA.getPartitionOffsets();
  ValueRange partSzA = opA.getPartitionSizes();
  ValueRange partOffsB = opB.getPartitionOffsets();
  ValueRange partSzB = opB.getPartitionSizes();

  if (partOffsA.empty() || partOffsB.empty())
    return std::nullopt;
  if (partOffsA.size() != partOffsB.size())
    return std::nullopt;

  /// If any dimension is provably disjoint, the regions are disjoint.
  for (size_t dim = 0; dim < partOffsA.size(); ++dim) {
    int64_t offA = 0, szA = 0, offB = 0, szB = 0;
    bool aConst = ValueAnalysis::getConstantIndex(partOffsA[dim], offA) &&
                  dim < partSzA.size() &&
                  ValueAnalysis::getConstantIndex(partSzA[dim], szA);
    bool bConst = ValueAnalysis::getConstantIndex(partOffsB[dim], offB) &&
                  dim < partSzB.size() &&
                  ValueAnalysis::getConstantIndex(partSzB[dim], szB);
    if (!aConst || !bConst)
      continue;

    /// Disjoint if [offA, offA+szA) and [offB, offB+szB) do not overlap.
    if (offA + szA <= offB || offB + szB <= offA)
      return DbAliasAnalysis::OverlapKind::Disjoint;
  }
  return std::nullopt;
}

} // namespace

static const char *aliasResultToString(DbAliasAnalysis::AliasResult res) {
  switch (res) {
  case DbAliasAnalysis::AliasResult::NoAlias:
    return "NO ALIAS";
  case DbAliasAnalysis::AliasResult::MustAlias:
    return "MUST ALIAS";
  case DbAliasAnalysis::AliasResult::MayAlias:
    return "MAY ALIAS";
  }
  ARTS_UNREACHABLE("Unhandled alias result");
}

///===----------------------------------------------------------------------===///
/// Alias Classification - Main Entry Point
///
/// This method implements memory-region aliasing for DB graph nodes:
/// 1. Identical nodes must alias.
/// 2. Different root allocations do not alias.
/// 3. Acquires on the same root are refined by slice/partition overlap.
/// 4. Everything else is conservatively may-alias.
///
/// Returns: NoAlias (definitely disjoint), MustAlias (identical), or
///          MayAlias (possibly overlapping, conservative)
///===----------------------------------------------------------------------===///
DbAliasAnalysis::AliasResult
DbAliasAnalysis::classifyAlias(const NodeBase &a, const NodeBase &b,
                               const std::string &indent) {
  ARTS_INFO("Analyzing alias between: " << a.getHierId() << " -- "
                                        << b.getHierId());

  if (&a == &b || a.getOp() == b.getOp()) {
    ARTS_INFO("  Result: MUST ALIAS (same node)" << indent);
    return AliasResult::MustAlias;
  }

  const DbAllocNode *allocA = getRootAlloc(a);
  const DbAllocNode *allocB = getRootAlloc(b);
  if (allocA && allocB && allocA != allocB) {
    ARTS_INFO("  Result: NO ALIAS (different DB roots)" << indent);
    return AliasResult::NoAlias;
  }

  const auto *acqA = dyn_cast<DbAcquireNode>(&a);
  const auto *acqB = dyn_cast<DbAcquireNode>(&b);
  if (acqA && acqB) {
    OverlapKind overlap = estimateOverlap(acqA, acqB);
    if (overlap == OverlapKind::Disjoint) {
      ARTS_INFO("  Result: NO ALIAS (disjoint DB slices)" << indent);
      return AliasResult::NoAlias;
    }
    if (overlap == OverlapKind::Full) {
      ARTS_INFO("  Result: MUST ALIAS (same DB slice)" << indent);
      return AliasResult::MustAlias;
    }
  }

  ARTS_INFO("  Result: " << aliasResultToString(AliasResult::MayAlias)
                         << indent);
  return AliasResult::MayAlias;
}

DbAliasAnalysis::OverlapKind
DbAliasAnalysis::estimateOverlap(const DbAcquireNode *a,
                                 const DbAcquireNode *b) {
  if (!a || !b)
    return OverlapKind::Unknown;
  if (a == b)
    return OverlapKind::Full;

  DbAllocNode *allocA = a->getRootAlloc();
  DbAllocNode *allocB = b->getRootAlloc();
  if (allocA && allocB && allocA != allocB)
    return OverlapKind::Disjoint;

  DbAcquireOp opA = a->getDbAcquireOp();
  DbAcquireOp opB = b->getDbAcquireOp();
  if (opA && opB) {
    if (opA.getPtr() == opB.getPtr())
      return OverlapKind::Full;
    if (auto sliceResult = refineAliasWithSlices(a, b)) {
      if (*sliceResult == AliasResult::NoAlias)
        return OverlapKind::Disjoint;
      if (*sliceResult == AliasResult::MustAlias)
        return OverlapKind::Full;
    }
  }

  /// Check partition offsets/sizes (element-space hints from depend clauses).
  /// These are distinct from regular offsets/sizes and may prove disjointness
  /// even when the regular slice check above was inconclusive.
  if (auto partResult = refineOverlapWithPartitionRanges(a, b)) {
    if (*partResult == OverlapKind::Disjoint)
      return OverlapKind::Disjoint;
  }

  return OverlapKind::Partial;
}
