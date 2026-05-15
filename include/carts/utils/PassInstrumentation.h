///==========================================================================///
/// File: PassInstrumentation.h
///
/// CARTS-specific pass instrumentation for per-pass timing and diagnostics.
///==========================================================================///

#ifndef ARTS_UTILS_PASSINSTRUMENTATION_H
#define ARTS_UTILS_PASSINSTRUMENTATION_H

#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/ADT/StringMap.h"
#include <chrono>
#include <memory>

namespace mlir {
namespace arts {

/// Shared timing data accumulated across multiple PassManager instances.
struct PassTimingData {
  struct PassTiming {
    double totalMs = 0.0;
    unsigned runCount = 0;
  };

  llvm::StringMap<PassTiming> timings;

  /// Print timing report to the given stream.
  void printTimingReport(llvm::raw_ostream &os) const;

  /// Export timing data as JSON.
  void exportTimingJson(llvm::raw_ostream &os) const;

  /// Get total compilation time across all passes.
  double getTotalTimeMs() const;
};

/// Per-pass timing instrumentation for CARTS. Each instance holds a raw
/// pointer to shared PassTimingData so that multiple PassManager instances
/// (one per pipeline stage) accumulate into the same report.
class CartsPassInstrumentation : public PassInstrumentation {
public:
  explicit CartsPassInstrumentation(PassTimingData *data) : data(data) {}

  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override;

private:
  void recordElapsed(Pass *pass);

  using TimePoint = std::chrono::steady_clock::time_point;
  TimePoint currentPassStart;
  PassTimingData *data;
};

} // namespace arts
} // namespace mlir

#endif // ARTS_UTILS_PASSINSTRUMENTATION_H
