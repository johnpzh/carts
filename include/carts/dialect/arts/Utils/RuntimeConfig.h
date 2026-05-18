///==========================================================================///
/// File: RuntimeConfig.h
///
/// ARTS runtime configuration representation used by compiler scheduling,
/// placement, and runtime-lowering decisions.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_UTILS_RUNTIMECONFIG_H
#define CARTS_DIALECT_ARTS_UTILS_RUNTIMECONFIG_H

#include <string>
#include <vector>

namespace mlir {
namespace carts::arts {

/// Execution mode classification for chunking decisions.
enum class ExecutionMode { SingleThreaded, IntraNode, InterNode };

class RuntimeConfig {
public:
  RuntimeConfig(const std::string &configFile = "");

  /// Core Machine Configuration
  int getThreads() const { return threads; }
  int getNodeCount() const { return nodeCount; }

  /// Runtime total workers across cluster (nodes * workers-per-node).
  int getRuntimeTotalWorkers() const {
    int nodesCount = nodeCount > 0 ? nodeCount : 1;
    return nodesCount * getRuntimeWorkersPerNode();
  }

  /// Optional compiler scheduling floor for SDE task grain.
  int getMinIterationsPerWorker() const { return minIterationsPerWorker; }

  /// Execution mode derived from runtime-visible worker concurrency.
  ExecutionMode getExecutionMode() const {
    if (nodeCount > 1)
      return ExecutionMode::InterNode;
    if (getRuntimeWorkersPerNode() > 1)
      return ExecutionMode::IntraNode;
    return ExecutionMode::SingleThreaded;
  }

  /// Configuration file status
  bool hasConfigFile() const { return configFileExists; }
  const std::string &getConfigPath() const { return configPath; }
  bool isValid() const { return isValidFlag; }

  /// Validation methods
  bool hasValidThreads() const { return threads > 0; }
  bool hasValidNodeCount() const { return nodeCount > 0; }

private:
  bool hasGpuSupport() const { return scheduler == 3 && gpu > 0; }

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

  bool validateConfiguration();
  bool parseFromFile(const std::string &path);

  static std::string trim(const std::string &s);
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
  int gpuLocality = 0;
  int gpuFit = 0;
  int gpuLCSync = 0;
  bool gpuBufferOn = true;
  int gpuMaxMemory = -1;
  int gpuMaxEdts = -1;

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
  int minIterationsPerWorker = 0;
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

} // namespace carts::arts
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_UTILS_RUNTIMECONFIG_H
