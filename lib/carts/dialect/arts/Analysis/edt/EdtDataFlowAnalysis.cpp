///==========================================================================///
/// File: EdtDataFlowAnalysis.cpp
/// Builds EDT-to-EDT dependencies by tracking datablock acquires.
///==========================================================================///

#include "carts/dialect/arts/Analysis/edt/EdtDataFlowAnalysis.h"

#include "carts/dialect/arts/Analysis/AnalysisManager.h"
#include "carts/dialect/arts/Analysis/db/DbAliasAnalysis.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(edt_data_flow_analysis);

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

namespace {
struct IndentScope {
  IndentScope() { ++depth; }
  ~IndentScope() { --depth; }
  static int depth;
};
int IndentScope::depth = 0;

std::string indent() { return std::string(IndentScope::depth * 2, ' '); }
} // namespace

EdtDataFlowAnalysis::EdtDataFlowAnalysis(DbGraph *dbGraph, AnalysisManager *AM)
    : dbGraph(dbGraph) {
  assert(dbGraph && "DbGraph is required");
  assert(AM && "AnalysisManager is required");
  aliasAA = AM->getDbAnalysis().getAliasAnalysis();
  assert(aliasAA && "Alias analysis is required");
}

SmallVector<EdtDep, 16> EdtDataFlowAnalysis::run(func::FuncOp func) {
  dependencyMap.clear();
  Environment env;
  processRegion(func.getBody(), env);

  using DepKey = std::pair<Operation *, Operation *>;
  using DepEntry = std::pair<DepKey, SmallVector<DbEdge, 2>>;
  SmallVector<DepEntry, 16> entries;
  entries.reserve(dependencyMap.size());
  for (auto &entry : dependencyMap) {
    SmallVector<DbEdge, 2> edges(entry.second.begin(), entry.second.end());
    entries.emplace_back(entry.first, std::move(edges));
  }

  llvm::DenseMap<Operation *, unsigned> opOrder;
  unsigned order = 0;
  func.walk([&](Operation *op) { opOrder[op] = order++; });

  llvm::sort(entries, [&](const auto &lhs, const auto &rhs) {
    auto lhsFrom = opOrder.lookup(lhs.first.first);
    auto rhsFrom = opOrder.lookup(rhs.first.first);
    if (lhsFrom != rhsFrom)
      return lhsFrom < rhsFrom;
    return opOrder.lookup(lhs.first.second) < opOrder.lookup(rhs.first.second);
  });

  SmallVector<EdtDep, 16> result;
  for (auto &entry : entries) {
    Operation *fromOp = entry.first.first;
    Operation *toOp = entry.first.second;
    auto from = dyn_cast_or_null<EdtOp>(fromOp);
    auto to = dyn_cast_or_null<EdtOp>(toOp);
    if (!from || !to || from == to)
      continue;
    result.push_back(EdtDep{from, to, entry.second});
  }
  dependencyMap.clear();
  return result;
}

std::pair<EdtDataFlowAnalysis::Environment, bool>
EdtDataFlowAnalysis::processRegion(Region &region, Environment &env) {
  Environment newEnv = env;
  bool changed = false;

  for (Block &block : region) {
    for (Operation &op : block) {
      IndentScope scope;
      std::pair<Environment, bool> result = {env, false};

      if (auto edt = dyn_cast<EdtOp>(&op))
        result = processEdt(edt, newEnv);
      else if (auto epoch = dyn_cast<EpochOp>(&op))
        result = processEpoch(epoch, newEnv);
      else if (auto ifOp = dyn_cast<scf::IfOp>(&op))
        result = processIf(ifOp, newEnv);
      else if (auto forOp = dyn_cast<scf::ForOp>(&op))
        result = processFor(forOp, newEnv);
      else
        continue;

      newEnv = std::move(result.first);
      changed |= result.second;
    }
  }

  return {std::move(newEnv), changed};
}

std::pair<EdtDataFlowAnalysis::Environment, bool>
EdtDataFlowAnalysis::processEdt(EdtOp edtOp, Environment &env) {
  ARTS_DEBUG(indent() << "Processing EDT");

  auto deps = edtOp.getDependenciesAsVector();
  EdtType edtType = edtOp.getType();
  bool isSyncEdt = (edtType == EdtType::sync);
  bool changed = false;
  Environment newEnv = env;

  SmallVector<DbAcquireNode *, 4> inputAcquires;
  SmallVector<DbAcquireNode *, 4> outputAcquires;

  for (auto dep : deps) {
    auto acquireOp = dyn_cast_or_null<DbAcquireOp>(dep.getDefiningOp());
    if (!acquireOp)
      continue;
    auto *node = dbGraph->getDbAcquireNode(acquireOp);
    if (!node)
      continue;
    auto mode = acquireOp.getMode();
    if (mode == ArtsMode::in || mode == ArtsMode::inout)
      inputAcquires.push_back(node);
    if (mode == ArtsMode::out || mode == ArtsMode::inout)
      outputAcquires.push_back(node);
  }

  ARTS_DEBUG(indent() << "  RAW candidates: " << inputAcquires.size());
  for (auto *consumer : inputAcquires) {
    auto prodDefs = findDefinitions(consumer, newEnv);
    if (prodDefs.empty())
      continue;

    for (auto *producer : prodDefs) {
      auto depType = classifyDep(producer, consumer);
      if (!depType || *depType != DbDepType::RAW)
        continue;
      recordDep(producer, consumer, *depType);
    }
  }

  ARTS_DEBUG(indent() << "  WAW candidates: " << outputAcquires.size());
  for (auto *writer : outputAcquires) {
    auto prodDefs = findDefinitions(writer, newEnv);
    for (auto *producer : prodDefs) {
      auto depType = classifyDep(producer, writer);
      if (!depType || *depType != DbDepType::WAW)
        continue;
      recordDep(producer, writer, *depType);
    }

    DbAllocNode *parentAlloc = writer->getParent();
    auto &defs = newEnv.writers[parentAlloc];
    bool needsUpdate = defs.size() != 1 || !defs.contains(writer);
    if (needsUpdate) {
      defs.clear();
      defs.insert(writer);
      changed = true;
    }
  }

  ARTS_DEBUG(indent() << "  WAR candidates: " << outputAcquires.size());
  for (auto *writer : outputAcquires) {
    DbAllocNode *parentAlloc = writer->getParent();
    auto &liveReaders = newEnv.readers[parentAlloc];
    if (liveReaders.empty())
      continue;
    for (auto *reader : liveReaders) {
      auto depType = classifyDep(reader, writer);
      if (!depType || *depType != DbDepType::WAR)
        continue;
      recordDep(reader, writer, *depType);
    }
    liveReaders.clear();
  }

  for (auto *reader : inputAcquires) {
    DbAllocNode *parentAlloc = reader->getParent();
    newEnv.readers[parentAlloc].insert(reader);
  }

  if (isSyncEdt) {
    changed |= !newEnv.writers.empty() || !newEnv.readers.empty();
    newEnv.writers.clear();
    newEnv.readers.clear();
  }

  return {newEnv, changed};
}

std::pair<EdtDataFlowAnalysis::Environment, bool>
EdtDataFlowAnalysis::processEpoch(EpochOp epochOp, Environment &env) {
  ARTS_DEBUG(indent() << "Processing Epoch");
  auto bodyResult = processRegion(epochOp.getBody(), env);
  Environment merged = bodyResult.first;
  merged.writers.clear();
  merged.readers.clear();
  return {std::move(merged), bodyResult.second};
}

std::pair<EdtDataFlowAnalysis::Environment, bool>
EdtDataFlowAnalysis::processIf(scf::IfOp ifOp, Environment &env) {
  ARTS_DEBUG(indent() << "Processing scf.if");
  auto thenResult = processRegion(ifOp.getThenRegion(), env);

  Environment elseEnv = env;
  bool elseChanged = false;
  if (!ifOp.getElseRegion().empty()) {
    auto elseResult = processRegion(ifOp.getElseRegion(), env);
    elseEnv = std::move(elseResult.first);
    elseChanged = elseResult.second;
  }

  Environment merged = thenResult.first;
  bool mergeChanged = unionInto(merged, elseEnv);
  bool changed = thenResult.second || elseChanged || mergeChanged;
  return {std::move(merged), changed};
}

std::pair<EdtDataFlowAnalysis::Environment, bool>
EdtDataFlowAnalysis::processFor(scf::ForOp forOp, Environment &env) {
  ARTS_DEBUG(indent() << "Processing scf.for");

  Environment loopEnv = env;
  bool overallChanged = false;
  constexpr unsigned maxIterations = 5;

  for (unsigned iter = 0; iter < maxIterations; ++iter) {
    auto bodyResult = processRegion(forOp.getRegion(), loopEnv);
    overallChanged |= bodyResult.second;
    bool envChanged = unionInto(loopEnv, bodyResult.first);
    overallChanged |= envChanged;
    if (!bodyResult.second && !envChanged)
      break;
  }

  return {loopEnv, overallChanged};
}

SmallVector<DbAcquireNode *>
EdtDataFlowAnalysis::findDefinitions(DbAcquireNode *acquire, Environment &env) {
  SmallVector<DbAcquireNode *> defs;
  DbAllocNode *parentAlloc = acquire->getParent();

  auto it = env.writers.find(parentAlloc);
  if (it == env.writers.end())
    return defs;

  for (auto *def : it->second) {
    auto aliasRes = aliasAA->classifyAlias(*def, *acquire);
    if (aliasRes == DbAliasAnalysis::AliasResult::NoAlias)
      continue;
    if (aliasRes == DbAliasAnalysis::AliasResult::MayAlias) {
      auto k = aliasAA->estimateOverlap(def, acquire);
      if (k == DbAliasAnalysis::OverlapKind::Disjoint)
        continue;
    }
    defs.push_back(def);
  }

  return defs;
}

bool EdtDataFlowAnalysis::unionInto(Environment &target,
                                    const Environment &source) {
  bool changed = false;

  for (const auto &pair : source.writers) {
    auto &defs = target.writers[pair.first];
    for (auto *acq : pair.second)
      if (!defs.contains(acq)) {
        defs.insert(acq);
        changed = true;
      }
  }

  for (const auto &pair : source.readers) {
    auto &defs = target.readers[pair.first];
    for (auto *acq : pair.second)
      if (!defs.contains(acq)) {
        defs.insert(acq);
        changed = true;
      }
  }

  return changed;
}

std::optional<DbDepType>
EdtDataFlowAnalysis::classifyDep(DbAcquireNode *producer,
                                 DbAcquireNode *consumer) {
  if (!producer || !consumer)
    return std::nullopt;
  if (producer->getParent() != consumer->getParent())
    return std::nullopt;

  auto prodMode = producer->getDbAcquireOp().getMode();
  auto consMode = consumer->getDbAcquireOp().getMode();

  if ((prodMode == ArtsMode::out || prodMode == ArtsMode::inout) &&
      (consMode == ArtsMode::in || consMode == ArtsMode::inout))
    return DbDepType::RAW;

  if ((prodMode == ArtsMode::out || prodMode == ArtsMode::inout) &&
      (consMode == ArtsMode::out || consMode == ArtsMode::inout))
    return DbDepType::WAW;

  if ((prodMode == ArtsMode::in || prodMode == ArtsMode::inout) &&
      (consMode == ArtsMode::out || consMode == ArtsMode::inout))
    return DbDepType::WAR;

  if ((prodMode == ArtsMode::in || prodMode == ArtsMode::inout) &&
      (consMode == ArtsMode::in || consMode == ArtsMode::inout))
    return DbDepType::RAR;

  return std::nullopt;
}

void EdtDataFlowAnalysis::recordDep(DbAcquireNode *producer,
                                    DbAcquireNode *consumer,
                                    DbDepType depType) {
  if (!producer || !consumer)
    return;
  EdtOp from = producer->getEdtUser();
  EdtOp to = consumer->getEdtUser();
  if (!from || !to || from == to)
    return;

  DbEdge edge{producer, consumer, depType};
  dependencyMap[{from.getOperation(), to.getOperation()}].insert(edge);
}

void EdtDataFlowAnalysis::recordDep(DbAcquireNode *reader,
                                    DbAcquireNode *writer) {
  auto depType = classifyDep(reader, writer);
  if (depType)
    recordDep(reader, writer, *depType);
}
