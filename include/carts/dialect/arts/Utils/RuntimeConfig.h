///==========================================================================///
/// File: RuntimeConfig.h
///
/// This file defines the ARTS runtime configuration representation that reads
/// configuration files and provides comprehensive information about
/// machine capabilities, threading, networking, GPU support, and
/// execution parameters.
///==========================================================================///

#pragma once

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace arts {

/// Execution mode classification for chunking decisions.
enum class ExecutionMode { SingleThreaded, IntraNode, InterNode };

class RuntimeConfig {
public:
  RuntimeConfig(const std::string &configFile = "");

  /// Core Machine Configuration
  int getThreads() const { return threads; }
  int getNodeCount() const { return nodeCount; }
  const std::vector<std::string> &getNodes() const { return nodes; }
  const std::string &getMasterNode() const { return masterNode; }
  const std::string &getLauncher() const { return launcher; }

  /// GPU Configuration
  int getScheduler() const { return scheduler; }
  int getGpuCount() const { return gpu; }
  int getGpuRouteTableSize() const { return gpuRouteTableSize; }
  int getGpuLocality() const { return gpuLocality; }
  int getGpuFit() const { return gpuFit; }
  int getGpuLCSync() const { return gpuLCSync; }
  bool isGpuBufferEnabled() const { return gpuBufferOn; }
  int getGpuMaxMemory() const { return gpuMaxMemory; }
  int getGpuMaxEdts() const { return gpuMaxEdts; }

  /// Network Configuration
  int getOutgoingThreads() const { return outgoing; }
  int getIncomingThreads() const { return incoming; }
  int getPorts() const { return ports; }
  const std::string &getProtocol() const { return protocol; }
  int getPort() const { return port; }
  const std::string &getNetInterface() const { return netInterface; }

  /// Hardware Configuration
  int getPinStride() const { return pinStride; }
  int getWorkerInitDequeSize() const { return workerInitDequeSize; }
  int getRouteTableSize() const { return routeTableSize; }
  bool isCoreDumpEnabled() const { return coreDump; }

  /// Performance Monitoring
  const std::string &getCounterFolder() const { return counterFolder; }
  int getCounterStartPoint() const { return counterStartPoint; }
  bool isKillModeEnabled() const { return killMode; }

  /// Machine Capability Queries
  bool hasGpuSupport() const { return scheduler == 3 && gpu > 0; }
  bool isSingleNode() const { return nodeCount == 1; }
  bool isDistributed() const { return nodeCount > 1; }
  bool isLocalExecution() const {
    return nodeCount == 1 && nodes.size() == 1 && nodes[0] == "localhost";
  }
  int getTotalWorkerThreads() const { return getRuntimeTotalWorkers(); }

  /// Runtime worker count per node used by ARTS scheduling.
  /// `worker_threads` is already the worker count on multi-node runs.
  /// On single-node runs ARTS reclaims sender/receiver threads as workers.
  int getRuntimeWorkersPerNode() const {
    int workers = threads;
    if (nodeCount <= 1) {
      workers += outgoing + incoming;
    }
    return workers > 0 ? workers : 1;
  }
  /// Runtime total workers across cluster (nodes * workers-per-node).
  int getRuntimeTotalWorkers() const {
    int nodesCount = nodeCount > 0 ? nodeCount : 1;
    return nodesCount * getRuntimeWorkersPerNode();
  }
  int getTotalGpuThreads() const { return hasGpuSupport() ? gpu : 0; }

  /// Execution mode derived from runtime-visible worker concurrency.
  ExecutionMode getExecutionMode() const {
    if (nodeCount > 1)
      return ExecutionMode::InterNode;
    if (getRuntimeWorkersPerNode() > 1)
      return ExecutionMode::IntraNode;
    return ExecutionMode::SingleThreaded;
  }

  /// Total parallelism (for threshold calculations)
  int getTotalParallelism() const { return getRuntimeTotalWorkers(); }

  /// Should chunking be prioritized? (true for distributed execution)
  bool shouldPrioritizeChunking() const { return nodeCount > 1; }

  /// Get machine description as a formatted string
  std::string getMachineDescription() const;

  /// Get configuration summary
  std::string getConfigurationSummary() const;

  /// Configuration file status
  bool hasConfigFile() const { return configFileExists; }
  const std::string &getConfigPath() const { return configPath; }
  bool isValid() const { return isValidFlag; }

  /// Validation methods
  bool hasValidThreads() const { return threads > 0; }
  bool hasValidNodeCount() const { return nodeCount > 0; }
  bool validateConfiguration();

  /// Parse arts.cfg at the given path. Returns false on failure.
  bool parseFromFile(const std::string &path);

private:
  static std::string trim(const std::string &s);
  static bool startsWith(const std::string &s, const char *prefix);
  static std::vector<std::string> splitCSV(const std::string &s);
  static int parseInt(const std::string &value, int defaultValue = -1);
  static bool parseBool(const std::string &value, bool defaultValue = false);

  /// Core Configuration
  int threads = 1;
  int nodeCount = 1;

  std::vector<std::string> nodes = {"localhost"};
  std::string masterNode = "localhost";
  std::string launcher = "ssh";

  /// GPU Configuration
  int scheduler = 0;
  int gpu = 0;
  int gpuRouteTableSize = 12;
  bool freeDbAfterGpuRun = false;
  bool deleteZerosGpuGc = true;
  bool runGpuGcIdle = true;
  bool runGpuGcPreEdt = false;
  int gpuLocality = 0;
  int gpuFit = 0;
  int gpuLCSync = 0;
  bool gpuBufferOn = true;
  int gpuMaxMemory = -1;
  int gpuMaxEdts = -1;
  bool gpuP2P = false;

  /// Network Configuration
  int outgoing = 0;
  int incoming = 0;
  int ports = 0;
  std::string protocol = "tcp";
  int port = 34739;
  std::string netInterface = "";

  /// Hardware Configuration
  int pinStride = 1;
  int workerInitDequeSize = 2048;
  int routeTableSize = 16;
  bool coreDump = false;

  /// Performance Monitoring
  std::string counterFolder = "./counters";
  int counterStartPoint = 1;
  bool killMode = false;

  /// Configuration file status
  std::string configPath = "";
  bool configFileExists = false;
  bool isValidFlag = false;
};

} // namespace arts
} // namespace mlir
