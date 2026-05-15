///==========================================================================///
/// File: DbDistributedEligibility.cpp
///
/// Eligibility analysis for distributed DB ownership marking.
///==========================================================================///

#include "carts/dialect/arts/Analysis/db/DbDistributedEligibility.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbGraph.h"
#include "carts/dialect/arts/Analysis/graphs/db/DbNode.h"
#include "carts/utils/DbUtils.h"
#include "carts/utils/OperationAttributes.h"
#include "carts/utils/ValueAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::arts;

namespace {

static bool hasMultipleAllocationBlocks(DbAllocOp alloc) {
  auto sizes = alloc.getSizes();
  if (sizes.empty())
    return false;
  if (sizes.size() > 1)
    return true;

  int64_t blockCount = 0;
  if (ValueAnalysis::getConstantIndex(sizes.front(), blockCount))
    return blockCount > 1;
  return true;
}

static bool hasSupportedAllocationShape(DbAllocOp alloc) {
  /// Distributed ownership currently targets matrix/tensor-style allocations.
  /// Keep vectors/scalars local to avoid over-marking small or temporary DBs.
  if (alloc.getElementSizes().size() < 2)
    return false;

  auto isPositiveOrDynamic = [](Value value) -> bool {
    int64_t constant = 0;
    if (!ValueAnalysis::getConstantIndex(
            ValueAnalysis::stripNumericCasts(value), constant))
      return true;
    return constant > 0;
  };

  for (Value size : alloc.getSizes())
    if (!isPositiveOrDynamic(size))
      return false;
  for (Value size : alloc.getElementSizes())
    if (!isPositiveOrDynamic(size))
      return false;
  return true;
}

static bool hasOnlyAllowedHandleUsers(Value rootHandle) {
  if (!rootHandle)
    return false;

  SmallVector<Value, 8> worklist{rootHandle};
  llvm::DenseSet<Value> visitedValues;

  auto enqueue = [&](Value value) {
    if (value)
      worklist.push_back(value);
  };

  auto isMemoryUserInsideEdt = [](Operation *user) {
    return user && static_cast<bool>(user->getParentOfType<EdtOp>());
  };

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visitedValues.insert(value).second)
      continue;

    for (Operation *user : value.getUsers()) {
      if (isa<DbAcquireOp, DbFreeOp, DbReleaseOp, EdtOp, LoweringContractOp>(
              user))
        continue;

      if (auto castOp = dyn_cast<memref::CastOp>(user)) {
        if (castOp.getSource() == value)
          enqueue(castOp.getResult());
        continue;
      }

      if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
        if (subview.getSource() == value)
          enqueue(subview.getResult());
        continue;
      }

      if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(user)) {
        if (reinterpret.getSource() == value)
          enqueue(reinterpret.getResult());
        continue;
      }

      if (auto view = dyn_cast<memref::ViewOp>(user)) {
        if (view.getSource() == value)
          enqueue(view.getResult());
        continue;
      }

      if (auto dbRefOp = dyn_cast<DbRefOp>(user)) {
        if (dbRefOp.getSource() == value)
          enqueue(dbRefOp.getResult());
        continue;
      }

      if (auto dbGepOp = dyn_cast<DbGepOp>(user)) {
        if (dbGepOp.getBasePtr() == value)
          enqueue(dbGepOp.getResult());
        continue;
      }

      if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(user)) {
        if (llvm::is_contained(unrealized.getInputs(), value))
          for (Value result : unrealized->getResults())
            enqueue(result);
        continue;
      }

      if (auto access = DbUtils::getMemoryAccessInfo(user)) {
        if (access->memref == value && isMemoryUserInsideEdt(user))
          continue;
        return false;
      }

      if (isa<memref::DimOp, memref::DeallocOp>(user))
        continue;

      if (isMemoryEffectFree(user)) {
        for (Value result : user->getResults())
          enqueue(result);
        continue;
      }

      return false;
    }
  }

  return true;
}

/// Returns true when an acquire node uses stencil semantics, checking the
/// contract summary, the analysis access pattern, and the node access pattern.
static bool isStencilAcquire(DbAcquireNode *node, DbAnalysis &dbAnalysis) {
  DbAcquireOp acquireOp = node->getDbAcquireOp();
  if (!acquireOp)
    return false;

  if (auto contract = dbAnalysis.getAcquireContractSummary(acquireOp);
      contract && contract->contract.hasExplicitStencilContract())
    return true;

  if (auto accessPattern = dbAnalysis.getAcquireAccessPattern(acquireOp))
    return *accessPattern == AccessPattern::Stencil;

  return node->getAccessPattern() == AccessPattern::Stencil;
}

/// Summary of facts collected in a single pass over all acquire nodes
/// belonging to a given DB allocation.
struct EligibilityFacts {
  bool hasInternodeEdtUse = false;
  bool hasStencilReadInternodeUse = false;
  bool allAcquiresReadOnly = true;
  bool isStencilFamily = false;
  bool allHaveEdtAcquireUsers = true;
};

/// Performs a single traversal of the acquire nodes for |alloc|, collecting
/// every fact needed by evaluateDistributedDbEligibility.  Returns std::nullopt
/// when the parent function or alloc node cannot be resolved.
static std::optional<EligibilityFacts>
collectEligibilityFacts(DbAllocOp alloc, DbAnalysis &dbAnalysis) {
  func::FuncOp parentFunc = alloc->getParentOfType<func::FuncOp>();
  if (!parentFunc)
    return std::nullopt;

  DbGraph &graph = dbAnalysis.getOrCreateGraph(parentFunc);
  DbAllocNode *allocNode = graph.getDbAllocNode(alloc);
  if (!allocNode)
    return std::nullopt;

  EligibilityFacts facts;

  graph.forEachAcquireNode([&](DbAcquireNode *acquireNode) {
    if (!acquireNode)
      return;
    if (acquireNode->getRootAlloc() != allocNode)
      return;

    /// allHaveEdtAcquireUsers (universal)
    EdtOp edt = acquireNode->getEdtUser();
    if (!edt)
      facts.allHaveEdtAcquireUsers = false;

    /// allAcquiresReadOnly (universal)
    bool readOnly = acquireNode->isReadOnlyAccess();
    if (!readOnly)
      facts.allAcquiresReadOnly = false;

    /// isStencilFamily (existential)
    bool stencil = isStencilAcquire(acquireNode, dbAnalysis);
    if (stencil)
      facts.isStencilFamily = true;

    /// hasInternodeEdtUse (existential)
    bool internode = edt && edt.getConcurrency() == EdtConcurrency::internode;
    if (internode)
      facts.hasInternodeEdtUse = true;

    /// hasStencilReadInternodeUse (existential)
    if (internode && readOnly && stencil)
      facts.hasStencilReadInternodeUse = true;
  });

  return facts;
}

static bool hasReadOnlyAfterInitAttr(DbAllocOp alloc) {
  return alloc->hasAttr(AttrNames::Operation::ReadOnlyAfterInit);
}

} // namespace

const char *mlir::arts::toString(DistributedDbEligibilityRejectReason reason) {
  switch (reason) {
  case DistributedDbEligibilityRejectReason::None:
    return "eligible";
  case DistributedDbEligibilityRejectReason::NestedInEdt:
    return "nested_in_edt";
  case DistributedDbEligibilityRejectReason::GlobalAllocType:
    return "global_alloc";
  case DistributedDbEligibilityRejectReason::SingleBlock:
    return "single_block";
  case DistributedDbEligibilityRejectReason::UnsupportedShape:
    return "unsupported_shape";
  case DistributedDbEligibilityRejectReason::StencilReadInternodeUse:
    return "stencil_read_internode_use";
  case DistributedDbEligibilityRejectReason::UnsupportedPtrUsers:
    return "unsupported_ptr_users";
  case DistributedDbEligibilityRejectReason::UnsupportedGuidUsers:
    return "unsupported_guid_users";
  case DistributedDbEligibilityRejectReason::NonEdtAcquireUse:
    return "non_edt_acquire_use";
  case DistributedDbEligibilityRejectReason::NoInternodeEdtUse:
    return "no_internode_edt_use";
  }
  return "unknown";
}

DistributedDbEligibilityResult
mlir::arts::evaluateDistributedDbEligibility(DbAllocOp alloc,
                                             DbAnalysis &dbAnalysis) {
  if (!alloc)
    return {false, DistributedDbEligibilityRejectReason::UnsupportedShape};
  if (alloc->getParentOfType<EdtOp>())
    return {false, DistributedDbEligibilityRejectReason::NestedInEdt};
  if (alloc.getAllocType() == DbAllocType::global)
    return {false, DistributedDbEligibilityRejectReason::GlobalAllocType};
  if (!hasMultipleAllocationBlocks(alloc))
    return {false, DistributedDbEligibilityRejectReason::SingleBlock};
  if (!hasSupportedAllocationShape(alloc))
    return {false, DistributedDbEligibilityRejectReason::UnsupportedShape};
  if (!hasOnlyAllowedHandleUsers(alloc.getPtr()))
    return {false, DistributedDbEligibilityRejectReason::UnsupportedPtrUsers};
  if (!hasOnlyAllowedHandleUsers(alloc.getGuid()))
    return {false, DistributedDbEligibilityRejectReason::UnsupportedGuidUsers};

  /// Single-pass collection of all acquire-node facts.
  auto maybeFacts = collectEligibilityFacts(alloc, dbAnalysis);
  if (!maybeFacts)
    return {false, DistributedDbEligibilityRejectReason::NonEdtAcquireUse};

  const EligibilityFacts &facts = *maybeFacts;

  if (!facts.allHaveEdtAcquireUsers)
    return {false, DistributedDbEligibilityRejectReason::NonEdtAcquireUse};
  if (!facts.hasInternodeEdtUse)
    return {false, DistributedDbEligibilityRejectReason::NoInternodeEdtUse};
  if (facts.hasStencilReadInternodeUse) {
    /// EXT-DIST-1: Allow read-only stencil DBs as replicated distributed DBs.
    /// A stencil-pattern DB is safe for distribution when no writes occur —
    /// each node can hold a complete copy of the data including halo regions.
    /// TODO(DT-5): Lowering must emit arts_add_db_duplicate() to actually
    /// replicate the data. Until then, this falls through to block
    /// distribution.
    bool readOnly =
        facts.allAcquiresReadOnly || hasReadOnlyAfterInitAttr(alloc);
    if (readOnly && facts.isStencilFamily)
      return {true, DistributedDbEligibilityRejectReason::None,
              EdtDistributionKind::replicated};
    return {false,
            DistributedDbEligibilityRejectReason::StencilReadInternodeUse};
  }
  return {true, DistributedDbEligibilityRejectReason::None};
}
