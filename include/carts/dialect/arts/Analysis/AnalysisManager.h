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
#include "carts/dialect/arts/Analysis/heuristics/EdtHeuristics.h"
#include "carts/dialect/arts/Analysis/loop/LoopAnalysis.h"
#include "carts/dialect/sde/Utils/SDECostModel.h"
#include "carts/dialect/arts/Utils/RuntimeConfig.h"
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
  AnalysisManager(ModuleOp module, const std::string &configFile = "",
                  const std::string &metadataFile = "");
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
  EdtHeuristics &getEdtHeuristics();

  /// Invalidate and rebuild DB graphs for all functions in the module.
  void invalidateAndRebuildGraphs(ModuleOp module);

  /// Other functions
  bool invalidateFunction(func::FuncOp func);

  ModuleOp &getModule() { return module; }
  const std::string &getConfigFile() const { return configFile; }
  const std::string &getMetadataFile() const { return metadataFile; }

  /// Get the ARTS runtime configuration
  RuntimeConfig &getRuntimeConfig() { return runtimeConfig; }
  const RuntimeConfig &getRuntimeConfig() const { return runtimeConfig; }

  /// Get the runtime-agnostic cost model (SDE passes use this interface).
  carts::sde::SDECostModel &getCostModel();

  const StringAnalysis &getStringAnalysis() const;

  /// Print summary of analysis objects and their graphs
  void print(llvm::raw_ostream &os);

  /// Capture diagnostic data
  void captureDiagnostics();

  /// Export analysis objects and graphs to JSON
  void exportToJson(llvm::raw_ostream &os, bool includeAnalysis = false);
  bool hasCapturedDiagnostics() const {
    return cachedDiagnosticJson.has_value();
  }

  /// Metadata coverage data
  struct MetadataCoverage {
    int64_t loopsAnalyzed = 0;
    int64_t loopsTotal = 0;
    int64_t memrefsAnalyzed = 0;
    int64_t memrefsTotal = 0;
    bool hasData = false;
  };

  /// Set metadata coverage data
  void setMetadataCoverage(int64_t loopsAnalyzed, int64_t loopsTotal,
                           int64_t memrefsAnalyzed, int64_t memrefsTotal) {
    metadataCoverage.loopsAnalyzed = loopsAnalyzed;
    metadataCoverage.loopsTotal = loopsTotal;
    metadataCoverage.memrefsAnalyzed = memrefsAnalyzed;
    metadataCoverage.memrefsTotal = memrefsTotal;
    metadataCoverage.hasData = true;
  }

  /// Get metadata coverage data
  const MetadataCoverage &getMetadataCoverage() const {
    return metadataCoverage;
  }

private:
  ModuleOp module;
  std::string configFile;
  std::string metadataFile;
  RuntimeConfig runtimeConfig;
  std::unique_ptr<carts::sde::SDECostModel> costModel;
  std::unique_ptr<DbAnalysis> dbAnalysis;
  std::unique_ptr<EdtAnalysis> edtAnalysis;
  std::unique_ptr<EpochAnalysis> epochAnalysis;
  std::unique_ptr<DbHeuristics> dbHeuristics;
  std::unique_ptr<EdtHeuristics> edtHeuristics;
  std::unique_ptr<LoopAnalysis> loopAnalysis;
  std::unique_ptr<StringAnalysis> stringAnalysis;
  /// Cached diagnostic data
  std::optional<std::string> cachedDiagnosticJson;

  /// Metadata coverage from VerifyMetadataPass
  MetadataCoverage metadataCoverage;
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_DIALECT_CORE_ANALYSIS_ANALYSISMANAGER_H
