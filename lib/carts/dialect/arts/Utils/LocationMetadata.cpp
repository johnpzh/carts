///===----------------------------------------------------------------------===///
/// LocationMetadata.cpp - Source Location Metadata Implementation
///===----------------------------------------------------------------------===///

#include "carts/dialect/arts/Utils/LocationMetadata.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::carts;
using namespace mlir::carts::arts;

///===----------------------------------------------------------------------===///
/// Interface Implementation
///===----------------------------------------------------------------------===///

void LocationMetadata::updateKey() {
  if (line == 0) {
    key = "";
    return;
  }

  std::string keyStr;
  llvm::raw_string_ostream os(keyStr);
  os << getBasename(file) << ":" << line << ":" << column;
  key = os.str();
}

std::string LocationMetadata::getBasename(llvm::StringRef path) {
  size_t lastSlash = path.rfind('/');
  if (lastSlash != llvm::StringRef::npos)
    return path.substr(lastSlash + 1).str();
  return path.str();
}

///===----------------------------------------------------------------------===///
/// Factory Methods
///===----------------------------------------------------------------------===///

/// Helper to check if filename is a C/C++ source file
static bool isCSourceFile(llvm::StringRef filename) {
  return filename.ends_with(".c") || filename.ends_with(".cpp") ||
         filename.ends_with(".h") || filename.ends_with(".hpp");
}

LocationMetadata LocationMetadata::fromLocation(Location loc) {
  LocationMetadata metadata;

  /// Try to extract FileLineColLoc directly
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    metadata.file = fileLoc.getFilename().str();
    metadata.line = fileLoc.getLine();
    metadata.column = fileLoc.getColumn();
  }
  /// Handle FusedLoc - may contain C source location
  else if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    /// Try to find a C/C++ source FileLineColLoc
    for (Location subLoc : fusedLoc.getLocations()) {
      if (auto fileLoc = dyn_cast<FileLineColLoc>(subLoc)) {
        llvm::StringRef filename = fileLoc.getFilename();
        if (isCSourceFile(filename)) {
          metadata.file = filename.str();
          metadata.line = fileLoc.getLine();
          metadata.column = fileLoc.getColumn();
          metadata.updateKey();
          return metadata;
        }
      }
    }
    /// If no C source file found, recurse on first location
    if (!fusedLoc.getLocations().empty())
      return fromLocation(fusedLoc.getLocations()[0]);
  }
  /// Handle CallSiteLoc - prefer caller location for per-callsite metadata
  else if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) {
    LocationMetadata callerMeta = fromLocation(callLoc.getCaller());
    if (callerMeta.isValid())
      return callerMeta;
    return fromLocation(callLoc.getCallee());
  }
  /// Handle NameLoc - check child location
  else if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    Location childLoc = nameLoc.getChildLoc();
    if (!isa<UnknownLoc>(childLoc)) {
      LocationMetadata childMeta = fromLocation(childLoc);
      if (childMeta.isValid())
        return childMeta;
    }
  }

  metadata.updateKey();
  return metadata;
}
