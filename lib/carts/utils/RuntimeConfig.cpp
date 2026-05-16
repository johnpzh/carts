///==========================================================================///
/// File: RuntimeConfig.cpp
///
/// Implementation of ARTS runtime configuration with comprehensive
/// configuration parsing and machine information capabilities.
///==========================================================================///

/// Debug
#include "carts/utils/Debug.h"
ARTS_DEBUG_SETUP(runtime_config);

#include "carts/utils/RuntimeConfig.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace mlir {
namespace carts::arts {

///===----------------------------------------------------------------------===///
/// RuntimeConfig Implementation
///===----------------------------------------------------------------------===///

RuntimeConfig::RuntimeConfig(const std::string &configFile) {
  ARTS_DEBUG_HEADER(RuntimeConfig);

  /// Helper function to get default config path
  auto getDefaultConfigPath = [&]() -> std::string {
    std::error_code ec;
    auto cwd = std::filesystem::current_path(ec);
    if (ec) {
      ARTS_ERROR("Failed to get current working directory: " << ec.message());
      configFileExists = false;
      return "";
    }
    return (cwd / "arts.cfg").string();
  };

  /// Check if config file is provided - if not, use default config path
  std::string path;
  if (!configFile.empty()) {
    std::error_code ec;
    auto absolutePath = std::filesystem::absolute(configFile, ec);
    path = ec ? configFile : absolutePath.string();
  } else {
    path = getDefaultConfigPath();
  }
  configPath = path;

  ARTS_DEBUG("Looking for configuration file at: " << path);
  if (!parseFromFile(path)) {
    ARTS_ERROR("No arts.cfg file found at " << path);
    configFileExists = false;
    return;
  }

  /// Validate the configuration
  if (!validateConfiguration()) {
    ARTS_ERROR("Configuration validation failed for " << configPath);
  } else {
    ARTS_INFO("Configuration loaded successfully");
  }
  ARTS_DEBUG_FOOTER(RuntimeConfig);
}

bool RuntimeConfig::parseFromFile(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    ARTS_DEBUG("Could not open configuration file: " << path);
    return false;
  }

  ARTS_DEBUG("Successfully opened configuration file: " << path);

  configFileExists = true;
  std::string line;
  bool inArtsSection = false;
  int lineNumber = 0;

  while (std::getline(in, line)) {
    lineNumber++;
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      if (!line.empty())
        ARTS_DEBUG("Line " << lineNumber << " (comment): " << line);
      continue;
    }
    if (line.front() == '[' && line.back() == ']') {
      inArtsSection = (line == "[ARTS]");
      ARTS_DEBUG("Line " << lineNumber << " (section): " << line
                         << " - inArtsSection="
                         << (inArtsSection ? "true" : "false"));
      continue;
    }
    if (!inArtsSection) {
      ARTS_DEBUG("Line " << lineNumber << " (skipped): " << line);
      continue;
    }

    auto pos = line.find('=');
    if (pos == std::string::npos) {
      ARTS_DEBUG("Line " << lineNumber << " (no equals): " << line);
      continue;
    }
    std::string key = trim(line.substr(0, pos));
    std::string val = trim(line.substr(pos + 1));
    ARTS_DEBUG("Line " << lineNumber << " (config): " << key << "=" << val);

    /// [ARTS] Section Parsing
    if (inArtsSection) {
      if (key == "worker_threads") {
        threads = parseInt(val);
        ARTS_DEBUG("Set threads=" << threads);
      } else if (key == "node_count") {
        nodeCount = parseInt(val);
        ARTS_DEBUG("Set node_count=" << nodeCount);
      } else if (key == "nodes") {
        nodes = splitCSV(val);
        ARTS_DEBUG("Set nodes=" << val << " (parsed " << nodes.size()
                                << " nodes)");
      } else if (key == "master_node") {
        masterNode = val;
        ARTS_DEBUG("Set master_node=" << masterNode);
      } else if (key == "launcher") {
        launcher = val;
        ARTS_DEBUG("Set launcher=" << launcher);
      }

      /// GPU Configuration
      else if (key == "scheduler")
        scheduler = parseInt(val, 0);
      else if (key == "gpu")
        gpu = parseInt(val, 0);
      else if (key == "gpu_route_table_size")
        gpuRouteTableSize = parseInt(val, 12);
      else if (key == "free_db_after_gpu_run")
        freeDbAfterGpuRun = parseBool(val, false);
      else if (key == "delete_zeros_gpu_gc")
        deleteZerosGpuGc = parseBool(val, true);
      else if (key == "run_gpu_gc_idle")
        runGpuGcIdle = parseBool(val, true);
      else if (key == "run_gpu_gc_pre_edt")
        runGpuGcPreEdt = parseBool(val, false);
      else if (key == "gpu_locality")
        gpuLocality = parseInt(val, 0);
      else if (key == "gpu_fit")
        gpuFit = parseInt(val, 0);
      else if (key == "gpu_lc_sync")
        gpuLCSync = parseInt(val, 0);
      else if (key == "gpu_buff_on")
        gpuBufferOn = parseBool(val, true);
      else if (key == "gpu_max_memory")
        gpuMaxMemory = parseInt(val, -1);
      else if (key == "gpu_max_edts")
        gpuMaxEdts = parseInt(val, -1);
      else if (key == "gpu_p2p")
        gpuP2P = parseBool(val, false);

      /// Network Configuration
      else if (key == "sender_threads")
        outgoing = parseInt(val, 0);
      else if (key == "receiver_threads")
        incoming = parseInt(val, 0);
      else if (key == "port_count")
        ports = parseInt(val, 0);
      else if (key == "protocol")
        protocol = val;
      else if (key == "default_ports")
        port = parseInt(val, 34739);
      else if (key == "net_interface")
        netInterface = val;

      /// Hardware Configuration
      else if (key == "pin")
        pinStride = parseInt(val, 1);
      else if (key == "worker_init_deque_size")
        workerInitDequeSize = parseInt(val, 2048);
      else if (key == "route_table_size")
        routeTableSize = parseInt(val, 16);
      else if (key == "core_dump")
        coreDump = parseBool(val, false);

      /// Performance Monitoring
      else if (key == "counter_folder")
        counterFolder = val;
      else if (key == "counter_capture_interval")
        counterStartPoint = parseInt(val, 1);
      else if (key == "kill_mode")
        killMode = parseBool(val, false);
    }
  }

  ARTS_DEBUG("Finished parsing configuration file");
  ARTS_INFO("Final configuration - worker_threads="
            << threads << ", node_count=" << nodeCount
            << ", launcher=" << launcher);
  return true;
}

std::string RuntimeConfig::getMachineDescription() const {
  std::ostringstream oss;
  oss << "ARTS Abstract Machine Configuration:\n";
  oss << "  Execution Mode: " << (isLocalExecution() ? "Local" : "Distributed")
      << "\n";
  oss << "  Configured Worker Threads: " << threads << " per node\n";
  oss << "  Runtime Workers: " << getRuntimeWorkersPerNode() << " per node\n";
  oss << "  Total Nodes: " << nodeCount << "\n";
  oss << "  Total Worker Threads: " << getTotalWorkerThreads() << "\n";

  if (hasGpuSupport()) {
    oss << "  GPU Support: Enabled (" << gpu << " GPUs)\n";
    oss << "  GPU Scheduler: " << scheduler << "\n";
    oss << "  Total GPU Threads: " << getTotalGpuThreads() << "\n";
  } else {
    oss << "  GPU Support: Disabled\n";
  }
  oss << "  Launcher: " << launcher << "\n";
  oss << "  Master Node: " << masterNode << "\n";

  if (!nodes.empty()) {
    oss << "  Nodes: ";
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (i > 0)
        oss << ", ";
      oss << nodes[i];
    }
    oss << "\n";
  }

  return oss.str();
}

std::string RuntimeConfig::getConfigurationSummary() const {
  std::ostringstream oss;
  oss << "Configuration Summary:\n";
  oss << "  Threading: " << threads << " worker threads per node\n";
  oss << "  Networking: sender_threads=" << outgoing
      << ", receiver_threads=" << incoming << "\n";
  oss << "  Protocol: " << protocol << " on default_ports=" << port << "\n";
  oss << "  Hardware: pin=" << pinStride
      << ", dequeSize=" << workerInitDequeSize << "\n";
  oss << "  Monitoring: counters in " << counterFolder << "\n";

  if (hasGpuSupport()) {
    oss << "  GPU: " << gpu << " devices, locality=" << gpuLocality
        << ", fit=" << gpuFit << "\n";
  }

  return oss.str();
}

std::string RuntimeConfig::trim(const std::string &s) {
  size_t i = 0, j = s.size();
  while (i < j && std::isspace(static_cast<unsigned char>(s[i])))
    ++i;
  while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1])))
    --j;
  return s.substr(i, j - i);
}

bool RuntimeConfig::startsWith(const std::string &s, const char *prefix) {
  size_t n = std::strlen(prefix);
  return s.size() >= n && std::equal(prefix, prefix + n, s.begin());
}

std::vector<std::string> RuntimeConfig::splitCSV(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(trim(item));
  }
  if (out.empty() && !s.empty())
    out.push_back(trim(s));
  return out;
}

int RuntimeConfig::parseInt(const std::string &value, int defaultValue) {
  if (value.empty())
    return defaultValue;
  return std::atoi(value.c_str());
}

bool RuntimeConfig::parseBool(const std::string &value, bool defaultValue) {
  if (value.empty())
    return defaultValue;
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

bool RuntimeConfig::validateConfiguration() {
  ARTS_DEBUG("Validating configuration...");

  bool isValid = true;
  const bool isSlurmLauncher = (launcher == "slurm");

  /// Check core requirements
  if (threads <= 0) {
    ARTS_ERROR("Invalid worker_threads value: " << threads << " (must be > 0)");
    isValid = false;
  }

  if (nodeCount <= 0) {
    ARTS_ERROR("Invalid node_count value: " << nodeCount << " (must be > 0)");
    isValid = false;
  }

  /// Check nodes consistency for non-SLURM launchers.
  /// In SLURM mode, node list/master are supplied by scheduler at runtime.
  if (!isSlurmLauncher) {
    if (nodes.empty()) {
      ARTS_ERROR("No nodes specified");
      isValid = false;
    } else if (static_cast<int>(nodes.size()) != nodeCount) {
      ARTS_ERROR("Nodes count (" << nodes.size()
                                 << ") doesn't match node_count (" << nodeCount
                                 << ")");
      isValid = false;
    }

    if (!nodes.empty() &&
        std::find(nodes.begin(), nodes.end(), masterNode) == nodes.end()) {
      ARTS_ERROR("Master node '" << masterNode << "' not found in nodes list");
      isValid = false;
    }
  }

  /// Check GPU configuration consistency
  if (hasGpuSupport()) {
    if (gpu <= 0) {
      ARTS_ERROR("GPU support enabled but gpu count is " << gpu);
      isValid = false;
    }
    if (scheduler != 3) {
      ARTS_ERROR("GPU support requires scheduler=3, got " << scheduler);
      isValid = false;
    }
  }

  /// Set the validation flag
  isValidFlag = isValid;

  ARTS_DEBUG("Configuration validation " << (isValid ? "PASSED" : "FAILED"));
  return isValid;
}

} // namespace carts::arts
} // namespace mlir
