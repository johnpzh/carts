///==========================================================================///
/// File: LocationMetadata.h
///
/// This file defines LocationMetadata for tracking and matching source code
/// locations. This is a lightweight value type for representing source
/// locations.
///==========================================================================///

#ifndef ARTS_UTILS_LOCATIONMETADATA_H
#define ARTS_UTILS_LOCATIONMETADATA_H

#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace carts::arts {

///===----------------------------------------------------------------------===///
/// LocationMetadata
///===----------------------------------------------------------------------===///
class LocationMetadata {
public:
  /// Attributes
  std::string file;
  unsigned line = 0, column = 0;
  /// Computed: "basename:line:col"
  std::string key;

  /// Constructor
  LocationMetadata() = default;

  /// Construct from file, line, column
  LocationMetadata(llvm::StringRef file, unsigned line, unsigned column)
      : file(file.str()), line(line), column(column) {
    updateKey();
  }

  ///===-------------------------------------------------------------===///
  /// Factory Methods
  ///===-------------------------------------------------------------===///

  /// SINGLE factory method for creating LocationMetadata from MLIR Location
  /// Handles FileLineColLoc, FusedLoc, CallSiteLoc, NameLoc
  static LocationMetadata fromLocation(Location loc);

  /// Extract basename from a file path
  static std::string getBasename(llvm::StringRef path);

  /// Check if location is valid
  bool isValid() const { return line > 0; }

private:
  /// Update the location key based on file, line, column
  void updateKey();
};

} // namespace carts::arts
} // namespace mlir

#endif // ARTS_UTILS_LOCATIONMETADATA_H
