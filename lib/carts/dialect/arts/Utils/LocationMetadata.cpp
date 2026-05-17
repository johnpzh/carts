///===----------------------------------------------------------------------===///
/// LocationMetadata.cpp - ARTS source-location metadata
///===----------------------------------------------------------------------===///

#include "carts/dialect/arts/Utils/LocationMetadata.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::carts::arts;

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

static bool isCSourceFile(llvm::StringRef filename) {
  return filename.ends_with(".c") || filename.ends_with(".cpp") ||
         filename.ends_with(".h") || filename.ends_with(".hpp");
}

LocationMetadata LocationMetadata::fromLocation(Location loc) {
  LocationMetadata metadata;

  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    metadata.file = fileLoc.getFilename().str();
    metadata.line = fileLoc.getLine();
    metadata.column = fileLoc.getColumn();
  } else if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
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
    if (!fusedLoc.getLocations().empty())
      return fromLocation(fusedLoc.getLocations()[0]);
  } else if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) {
    LocationMetadata callerMeta = fromLocation(callLoc.getCaller());
    if (callerMeta.isValid())
      return callerMeta;
    return fromLocation(callLoc.getCallee());
  } else if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
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
