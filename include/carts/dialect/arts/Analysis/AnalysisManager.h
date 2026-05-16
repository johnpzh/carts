///==========================================================================///
/// File: AnalysisManager.h
///
/// This file defines the AnalysisManager for centralized management
/// of all ARTS analysis objects.
///==========================================================================///

#ifndef ARTS_DIALECT_CORE_ANALYSIS_ANALYSISMANAGER_H
#define ARTS_DIALECT_CORE_ANALYSIS_ANALYSISMANAGER_H

#include "carts/dialect/arts/Analysis/StringAnalysis.h"
#include "carts/dialect/arts/Analysis/db/DbAnalysis.h"
#include "carts/dialect/arts/Analysis/edt/EdtAnalysis.h"
#include "carts/dialect/arts/Analysis/edt/EpochAnalysis.h"
#include "carts/dialect/arts/Analysis/heuristics/DbHeuristics.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "carts/dialect/sde/Utils/SDECostModel.h"
#include "carts/utils/RuntimeConfig.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>
#include <optional>

namespace mlir {

class Pass;

namespace carts::arts {

/// Centralized manager for all ARTS analysis objects.
class AnalysisManager {
public:
  AnalysisManager(ModuleOp module, const std::string &configFile = "");
  ~AnalysisManager();

  /// Invalidate all analysis objects and graphs
  void invalidate();

  /// Get analysis objects
  DbAnalysis &getDbAnalysis();
  EdtAnalysis &getEdtAnalysis();
  EpochAnalysis &getEpochAnalysis();
  LoopAnalysis &getLoopAnalysis();
  StringAnalysis &getStringAnalysis();
  DbHeuristics &getDbHeuristics();

  /// Invalidate and rebuild DB graphs for all functions in the module.
  void invalidateAndRebuildGraphs(ModuleOp module);

  /// Other functions
  bool invalidateFunction(func::FuncOp func);

  ModuleOp &getModule() { return module; }
  const std::string &getConfigFile() const { return configFile; }

  /// Get the ARTS runtime configuration
  RuntimeConfig &getRuntimeConfig() { return runtimeConfig; }
  const RuntimeConfig &getRuntimeConfig() const { return runtimeConfig; }

  /// Get the runtime-agnostic cost model (SDE passes use this interface).
  carts::sde::SDECostModel &getCostModel();

  const StringAnalysis &getStringAnalysis() const;

  /// Capture diagnostic data
  void captureDiagnostics();

  /// Export analysis objects and graphs to JSON
  void exportToJson(llvm::raw_ostream &os, bool includeAnalysis = false);
  bool hasCapturedDiagnostics() const {
    return cachedDiagnosticJson.has_value();
  }

private:
  ModuleOp module;
  std::string configFile;
  RuntimeConfig runtimeConfig;
  std::unique_ptr<carts::sde::SDECostModel> costModel;
  std::unique_ptr<DbAnalysis> dbAnalysis;
  std::unique_ptr<EdtAnalysis> edtAnalysis;
  std::unique_ptr<EpochAnalysis> epochAnalysis;
  std::unique_ptr<DbHeuristics> dbHeuristics;
  std::unique_ptr<LoopAnalysis> loopAnalysis;
  std::unique_ptr<StringAnalysis> stringAnalysis;
  /// Cached diagnostic data
  std::optional<std::string> cachedDiagnosticJson;

};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_ANALYSISMANAGER_H
