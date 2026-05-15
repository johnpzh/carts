///==========================================================================///
/// File: Debug.h
///
/// Centralized debug utilities for ARTS passes and components
///==========================================================================///

#ifndef ARTS_UTILS_ARTSDEBUG_H
#define ARTS_UTILS_ARTSDEBUG_H

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace arts {
#define ARTS_DEBUG_COLORS true

/// Returns the ARTS debug output stream. Intended as the single entrypoint
/// for emitting debug logs so callers don't use llvm::dbgs() directly.
static inline llvm::raw_ostream &debugStream() {
  static llvm::raw_ostream &stream = llvm::errs();
  /// Force color support if we're connected to a terminal
  if (stream.is_displayed())
    stream.enable_colors(ARTS_DEBUG_COLORS);
  return stream;
}

/// Common line separator used in debug headers/footers
#ifndef ARTS_LINE
#define ARTS_LINE "-----------------------------------------\n"
#endif

#ifndef ARTS_SEPARATOR
#define ARTS_SEPARATOR "===================================\n"
#endif

/// Macro to set up debug infrastructure for a pass/component
/// Usage: ARTS_DEBUG_SETUP(edt) at the top of your file
#define ARTS_DEBUG_SETUP(pass_name)                                            \
  static constexpr const char *ARTS_DEBUG_TYPE_STR = #pass_name;

/// ARTS debug stream accessor. Use this instead of llvm::dbgs()
#define ARTS_DBGS() debugStream()

#define ARTS_DEBUG_TYPE(x)                                                     \
  DEBUG_WITH_TYPE(ARTS_DEBUG_TYPE_STR, {                                       \
    ARTS_DBGS() << "[" << ARTS_DEBUG_TYPE_STR << "] " << x << "\n";            \
  })

#define ARTS_DEBUG_MSG(msg)                                                    \
  DEBUG_WITH_TYPE(ARTS_DEBUG_TYPE_STR, { ARTS_DBGS() << msg << "\n"; })

/// Multi-statement debug region. Wraps an entire block under LLVM_DEBUG.
/// Example:
///   ARTS_DEBUG_REGION(
///     ARTS_DBGS() << "String memrefs:\n";
///     for (auto value : stringMemRefs)
///       ARTS_DBGS() << "  " << value << "\n";
///   );
#define ARTS_DEBUG_REGION(...)                                                 \
  DEBUG_WITH_TYPE(ARTS_DEBUG_TYPE_STR, {__VA_ARGS__})

/// Debug section with automatic header/footer around a region.
/// Example:
///   ARTS_DEBUG_SECTION("StringAnalysis",
///     ARTS_DBGS() << "String memrefs:\n";
///     for (auto value : stringMemRefs)
///       ARTS_DBGS() << "  " << value << "\n";
///   );
#define ARTS_DEBUG_SECTION(title, ...)                                         \
  DEBUG_WITH_TYPE(ARTS_DEBUG_TYPE_STR, {                                       \
    ARTS_DBGS() << "\n" << ARTS_LINE << title << " STARTED\n" << ARTS_LINE;    \
    __VA_ARGS__;                                                               \
    ARTS_DBGS() << "\n" << ARTS_LINE << title << " FINISHED\n" << ARTS_LINE;   \
  })

/// Standard header/footer patterns for passes
#define ARTS_DEBUG_HEADER(x)                                                   \
  DEBUG_WITH_TYPE(ARTS_DEBUG_TYPE_STR, {                                       \
    auto &__os = ARTS_DBGS();                                                  \
    __os.changeColor(llvm::raw_ostream::MAGENTA, /*bold=*/true);               \
    __os << "\n" << #x " STARTED\n" << ARTS_SEPARATOR;                         \
    __os.resetColor();                                                         \
  })

#define ARTS_DEBUG_FOOTER(x)                                                   \
  DEBUG_WITH_TYPE(ARTS_DEBUG_TYPE_STR, {                                       \
    auto &__os = ARTS_DBGS();                                                  \
    __os.changeColor(llvm::raw_ostream::MAGENTA, /*bold=*/true);               \
    __os << "\n" << #x " FINISHED\n" << ARTS_SEPARATOR << "\n";                \
    __os.resetColor();                                                         \
  })

/// Colored levels
#define ARTS_INFO(x)                                                           \
  DEBUG_WITH_TYPE(ARTS_DEBUG_TYPE_STR, {                                       \
    auto &__os = ARTS_DBGS();                                                  \
    __os.changeColor(llvm::raw_ostream::BLUE, /*bold=*/true);                  \
    __os << "[INFO] [" << ARTS_DEBUG_TYPE_STR << "]";                          \
    __os.resetColor();                                                         \
    __os << " " << x << "\n";                                                  \
  })

#define ARTS_INFO_HEADER(x)                                                    \
  DEBUG_WITH_TYPE(ARTS_DEBUG_TYPE_STR, {                                       \
    auto &__os = ARTS_DBGS();                                                  \
    __os.changeColor(llvm::raw_ostream::BLUE, /*bold=*/true);                  \
    __os << "\n" << ARTS_LINE << #x " STARTED\n" << ARTS_LINE;                 \
    __os.resetColor();                                                         \
  })

#define ARTS_INFO_FOOTER(x)                                                    \
  DEBUG_WITH_TYPE(ARTS_DEBUG_TYPE_STR, {                                       \
    auto &__os = ARTS_DBGS();                                                  \
    __os.changeColor(llvm::raw_ostream::BLUE, /*bold=*/true);                  \
    __os << "\n" << ARTS_LINE << #x " FINISHED\n" << ARTS_LINE << "\n";        \
    __os.resetColor();                                                         \
  })

#define ARTS_DEBUG(x)                                                          \
  DEBUG_WITH_TYPE(ARTS_DEBUG_TYPE_STR, {                                       \
    auto &__os = ARTS_DBGS();                                                  \
    __os.changeColor(llvm::raw_ostream::MAGENTA, /*bold=*/true);               \
    __os << "[DEBUG] [" << ARTS_DEBUG_TYPE_STR << "]";                         \
    __os.resetColor();                                                         \
    __os << " " << x << "\n";                                                  \
  })

#define ARTS_WARN(x)                                                           \
  DEBUG_WITH_TYPE(ARTS_DEBUG_TYPE_STR, {                                       \
    auto &__os = ARTS_DBGS();                                                  \
    __os.changeColor(llvm::raw_ostream::YELLOW, /*bold=*/true);                \
    __os << "[WARN] [" << ARTS_DEBUG_TYPE_STR << "]";                          \
    __os.resetColor();                                                         \
    __os << " " << x << "\n";                                                  \
  })

#define ARTS_ERROR(x)                                                          \
  {                                                                            \
    auto &__os = llvm::errs();                                                 \
    __os.changeColor(llvm::raw_ostream::RED, /*bold=*/true);                   \
    __os << "[ERROR] [" << ARTS_DEBUG_TYPE_STR << "]";                         \
    __os.resetColor();                                                         \
    __os << " " << x << "\n";                                                  \
  }

/// Unreachable code marker with ARTS formatting
#define ARTS_UNREACHABLE(msg)                                                  \
  {                                                                            \
    auto &__os = llvm::errs();                                                 \
    __os.changeColor(llvm::raw_ostream::RED, /*bold=*/true);                   \
    __os << "[UNREACHABLE] [" << ARTS_DEBUG_TYPE_STR << "]";                   \
    __os.resetColor();                                                         \
    __os << " " << msg << "\n";                                                \
    llvm_unreachable(msg);                                                     \
  }
} // namespace arts
} // namespace mlir
#endif /// ARTS_UTILS_ARTSDEBUG_H
