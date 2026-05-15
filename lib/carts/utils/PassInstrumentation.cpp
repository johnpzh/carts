///==========================================================================///
/// File: PassInstrumentation.cpp
///
/// Implementation of CARTS-specific pass instrumentation.
///==========================================================================///

#include "carts/utils/PassInstrumentation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <vector>

using namespace mlir;
using namespace mlir::arts;

void CartsPassInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  currentPassStart = std::chrono::steady_clock::now();
}

void CartsPassInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  recordElapsed(pass);
}

void CartsPassInstrumentation::runAfterPassFailed(Pass *pass, Operation *op) {
  recordElapsed(pass);
}

void CartsPassInstrumentation::recordElapsed(Pass *pass) {
  auto elapsed = std::chrono::steady_clock::now() - currentPassStart;
  double ms =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() /
      1000.0;
  StringRef name = pass->getName();
  auto &entry = data->timings[name];
  entry.totalMs += ms;
  entry.runCount++;
}

void PassTimingData::printTimingReport(llvm::raw_ostream &os) const {
  if (timings.empty())
    return;

  using Entry = std::pair<StringRef, PassTiming>;
  std::vector<Entry> sorted;
  sorted.reserve(timings.size());
  for (const auto &entry : timings)
    sorted.emplace_back(entry.getKey(), entry.getValue());
  std::sort(sorted.begin(), sorted.end(), [](const Entry &a, const Entry &b) {
    return a.second.totalMs > b.second.totalMs;
  });

  double total = getTotalTimeMs();

  os << "\n===--- CARTS Pass Timing Report ---===\n";
  os << llvm::format("  %-50s %10s %6s %8s\n", "Pass", "Time (ms)", "Runs",
                     "% Total");
  os << llvm::format("  %-50s %10s %6s %8s\n",
                     "--------------------------------------------------",
                     "----------", "------", "--------");
  for (const auto &pair : sorted) {
    double pct = total > 0.0 ? (pair.second.totalMs / total) * 100.0 : 0.0;
    os << llvm::format("  %-50s %10.2f %6u %7.1f%%\n", pair.first.str().c_str(),
                       pair.second.totalMs, pair.second.runCount, pct);
  }
  os << llvm::format("  %-50s %10.2f\n",
                     "--------------------------------------------------",
                     total);
  os << llvm::format("  %-50s %10.2f ms\n", "Total", total);
  os << "===-----------------------------------===\n\n";
}

void PassTimingData::exportTimingJson(llvm::raw_ostream &os) const {
  os << "[\n";
  bool first = true;
  for (const auto &entry : timings) {
    if (!first)
      os << ",\n";
    first = false;
    os << "  {\"name\": \"" << entry.getKey()
       << "\", \"totalMs\": " << llvm::format("%.2f", entry.getValue().totalMs)
       << ", \"runCount\": " << entry.getValue().runCount << "}";
  }
  os << "\n]\n";
}

double PassTimingData::getTotalTimeMs() const {
  double total = 0.0;
  for (const auto &entry : timings)
    total += entry.getValue().totalMs;
  return total;
}
