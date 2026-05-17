///==========================================================================///
/// File: LocationMetadata.h
///
/// ARTS source-location metadata for graph diagnostics and runtime IDs.
///==========================================================================///

#ifndef CARTS_DIALECT_ARTS_UTILS_LOCATIONMETADATA_H
#define CARTS_DIALECT_ARTS_UTILS_LOCATIONMETADATA_H

#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace carts::arts {

class LocationMetadata {
public:
  std::string file;
  unsigned line = 0, column = 0;
  std::string key;

  LocationMetadata() = default;

  LocationMetadata(llvm::StringRef file, unsigned line, unsigned column)
      : file(file.str()), line(line), column(column) {
    updateKey();
  }

  /// Create source-location metadata from FileLineColLoc, FusedLoc,
  /// CallSiteLoc, or NameLoc.
  static LocationMetadata fromLocation(Location loc);

  /// Extract basename from a file path.
  static std::string getBasename(llvm::StringRef path);

  bool isValid() const { return line > 0; }

private:
  void updateKey();
};

} // namespace carts::arts
} // namespace mlir

#endif // CARTS_DIALECT_ARTS_UTILS_LOCATIONMETADATA_H
